"""Value detection: de-vigging, expected value, fractional Kelly, REGRESSION_TRAP.

This module turns a :class:`~src.common.schemas.MatchPrediction` (model side)
and an :class:`~src.common.schemas.OddsQuote` (bookmaker side) into
:class:`ValueOpportunity` rows, one per quoted selection, carrying the
expected value, the de-vigged market edge, the Kelly stake and a tuple of
diagnostic flags (``VALUE``, ``REGRESSION_TRAP``, ``TRAP:<team>``,
``LONGSHOT``, ``SUSPICIOUS_EDGE``).

It deliberately does **not** import ``src.engine.model``: it only depends on the
shared contracts in :mod:`src.common.schemas`, so it can be unit-tested against
the synthetic league whose true parameters are known.

Formulas
--------
* implied probability      ``q_i = 1 / o_i``
* over-round               ``sum(q_i) - 1``
* proportional de-vig      ``p_i = q_i / sum(q_j)``
* power de-vig             ``p_i = q_i ** k`` with ``k`` such that ``sum(p_i) == 1``
* Shin (1993) de-vig       ``p_i = (sqrt(z^2 + 4(1-z) q_i^2 / Q) - z) / (2(1-z))``
* expected value           ``EV = p_model * odds - 1``
* Kelly                    ``f* = (p * b - (1 - p)) / b`` with ``b = odds - 1``
* fractional stake         ``f* * fraction * bankroll`` (capped, rounded, floored)
"""
from __future__ import annotations

import math
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass, field
from datetime import date, datetime
from decimal import ROUND_FLOOR, ROUND_HALF_UP, Decimal
from pathlib import Path
from typing import Any, Final, Literal

import numpy as np
import pandas as pd
from scipy.optimize import brentq

from src.common.logging_config import get_logger
from src.common.schemas import SELECTIONS, Girone, Market, MatchPrediction, OddsQuote, validate_matches_df

__all__ = [
    "DEFAULT_KELLY",
    "DEVIG_METHODS",
    "FLAG_LONGSHOT",
    "FLAG_REGRESSION_TRAP",
    "FLAG_SUSPICIOUS_EDGE",
    "FLAG_VALUE",
    "KELLY_EIGHTH",
    "KELLY_QUARTER",
    "TRAPS_FRAME_COLUMNS",
    "TRAP_FLAG_PREFIX",
    "VALUE_FRAME_COLUMNS",
    "DevigMethod",
    "KellyConfig",
    "RegressionTrap",
    "ValueEngine",
    "ValueOpportunity",
    "detect_regression_traps",
    "devig",
    "expected_value",
    "implied_probabilities",
    "kelly_fraction",
    "overround",
    "stake_units",
]

log = get_logger(__name__)

DevigMethod = Literal["proportional", "power", "shin"]
DEVIG_METHODS: Final[tuple[str, ...]] = ("proportional", "power", "shin")

#: Fractional-Kelly presets used by the CLI (``--kelly 0.125 | 0.25``).
KELLY_EIGHTH: Final[float] = 0.125
KELLY_QUARTER: Final[float] = 0.25

FLAG_VALUE: Final[str] = "VALUE"
FLAG_REGRESSION_TRAP: Final[str] = "REGRESSION_TRAP"
FLAG_LONGSHOT: Final[str] = "LONGSHOT"
FLAG_SUSPICIOUS_EDGE: Final[str] = "SUSPICIOUS_EDGE"
TRAP_FLAG_PREFIX: Final[str] = "TRAP:"

#: Column order of :meth:`ValueEngine.to_frame` (identical to the spec).
VALUE_FRAME_COLUMNS: Final[tuple[str, ...]] = (
    "match_id",
    "date",
    "girone",
    "home_team",
    "away_team",
    "market",
    "selection",
    "bookmaker",
    "book_odds",
    "fair_odds",
    "p_model",
    "p_implied_devig",
    "edge",
    "ev",
    "kelly_full",
    "stake_units",
    "flags",
    "value",
)

_VALUE_FRAME_DTYPES: Final[dict[str, str]] = {
    "match_id": "string",
    "date": "datetime64[ns]",
    "girone": "string",
    "home_team": "string",
    "away_team": "string",
    "market": "string",
    "selection": "string",
    "bookmaker": "string",
    "book_odds": "float64",
    "fair_odds": "float64",
    "p_model": "float64",
    "p_implied_devig": "float64",
    "edge": "float64",
    "ev": "float64",
    "kelly_full": "float64",
    "stake_units": "float64",
    "flags": "string",
    "value": "bool",
}

#: Column order of :meth:`ValueEngine.traps_frame`.
TRAPS_FRAME_COLUMNS: Final[tuple[str, ...]] = (
    "team",
    "girone",
    "goals_against",
    "xg_against",
    "severity",
    "last_date",
    "match_ids",
)

_TRAPS_FRAME_DTYPES: Final[dict[str, str]] = {
    "team": "string",
    "girone": "string",
    "goals_against": "int64",
    "xg_against": "float64",
    "severity": "float64",
    "last_date": "datetime64[ns]",
    "match_ids": "string",
}

#: Books whose over-round is below this are treated as exactly fair.
_FAIR_BOOK_TOL: Final[float] = 1e-12
#: Decimal places at which binary floating-point noise is absorbed before rounding stakes.
_STAKE_NOISE_PLACES: Final[Decimal] = Decimal("1e-9")


# ----------------------------------------------------------------------------- validation helpers
def _validate_odds(odds: Mapping[str, float]) -> dict[str, float]:
    """Return ``odds`` as a plain ``dict[str, float]`` after validating it.

    Raises ``ValueError`` when the mapping is empty, has a single selection
    (a one-way market cannot be de-vigged) or contains any decimal odds that
    are not finite and strictly greater than 1.
    """
    if len(odds) < 2:
        raise ValueError(f"at least two selections are required to form a book, got {len(odds)}")
    out: dict[str, float] = {}
    for selection, raw in odds.items():
        try:
            value = float(raw)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"decimal odds for {selection!r} are not numeric: {raw!r}") from exc
        if not math.isfinite(value) or value <= 1.0:
            raise ValueError(f"decimal odds must be finite and > 1.0 ({selection}={raw!r})")
        out[str(selection)] = value
    return out


def _validate_probability(p: float, name: str = "p_model") -> float:
    value = float(p)
    if not math.isfinite(value) or value < 0.0 or value > 1.0:
        raise ValueError(f"{name} must be a probability in [0, 1], got {p!r}")
    return value


def _validate_decimal_odds(odds: float) -> float:
    value = float(odds)
    if not math.isfinite(value) or value <= 1.0:
        raise ValueError(f"decimal odds must be finite and > 1.0, got {odds!r}")
    return value


def _validate_method(method: str) -> str:
    text = str(method).strip().lower()
    if text not in DEVIG_METHODS:
        raise ValueError(f"unknown de-vig method {method!r}; choose one of {DEVIG_METHODS}")
    return text


def _coerce_date(value: date | datetime | pd.Timestamp | str) -> date:
    """Normalise the various date-like inputs used across the pipeline to ``datetime.date``."""
    if isinstance(value, pd.Timestamp):
        return value.to_pydatetime().date()
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    if isinstance(value, str):
        return date.fromisoformat(value[:10])
    raise TypeError(f"cannot interpret {value!r} as a date")


# ----------------------------------------------------------------------------- implied probabilities
def implied_probabilities(odds: Mapping[str, float]) -> dict[str, float]:
    """Raw implied probabilities ``1 / odds`` per selection.

    The values sum to ``1 + overround`` (they are *not* normalised). Insertion
    order of ``odds`` is preserved. Raises ``ValueError`` for an empty or
    one-element book or for odds ``<= 1``.
    """
    clean = _validate_odds(odds)
    return {selection: 1.0 / o for selection, o in clean.items()}


def overround(odds: Mapping[str, float]) -> float:
    """Bookmaker margin ``sum(1 / odds) - 1`` (0.06 means a 6% over-round)."""
    return math.fsum(implied_probabilities(odds).values()) - 1.0


def _devig_proportional(q: np.ndarray) -> np.ndarray:
    return q / q.sum()


def _devig_power(q: np.ndarray) -> np.ndarray:
    """Solve ``sum(q_i ** k) == 1`` for ``k`` with Brent's method.

    For an over-round book ``k > 1``; for an under-round (arbitrage) book the
    same equation yields ``k < 1``; a fair book has ``k == 1``.
    """
    total = q.sum()
    if abs(total - 1.0) <= _FAIR_BOOK_TOL:
        return _devig_proportional(q)

    def f(k: float) -> float:
        return float(np.sum(q**k) - 1.0)

    if total > 1.0:
        lo, hi = 1.0, 2.0
        while f(hi) > 0.0:
            hi *= 2.0
            if hi > 1e9:  # pragma: no cover - unreachable for odds > 1
                raise ValueError("power de-vig failed to bracket the exponent")
    else:
        lo, hi = 1e-9, 1.0
    k = brentq(f, lo, hi, xtol=1e-14, rtol=1e-14, maxiter=500)
    return _devig_proportional(q**k)


def _devig_shin(q: np.ndarray) -> np.ndarray:
    """Shin (1993): share ``z`` of insider money solved so that ``sum(p_i) == 1``.

    ``p_i = (sqrt(z^2 + 4(1-z) q_i^2 / Q) - z) / (2(1-z))`` with ``Q = sum(q_i)``.
    Longshots lose probability relative to proportional normalisation, which is
    the favourite-longshot bias the method was designed to remove. For a book
    without over-round (``Q <= 1``) the insider share is ``z = 0`` and the
    result coincides with proportional normalisation.
    """
    total = q.sum()
    if total - 1.0 <= _FAIR_BOOK_TOL:
        return _devig_proportional(q)

    def probs(z: float) -> np.ndarray:
        return (np.sqrt(z * z + 4.0 * (1.0 - z) * q**2 / total) - z) / (2.0 * (1.0 - z))

    def f(z: float) -> float:
        return float(probs(z).sum() - 1.0)

    z = brentq(f, 0.0, 1.0 - 1e-12, xtol=1e-14, rtol=1e-14, maxiter=500)
    return _devig_proportional(probs(z))


def devig(odds: Mapping[str, float], method: DevigMethod = "proportional") -> dict[str, float]:
    """Remove the bookmaker margin and return probabilities summing to 1.

    Parameters
    ----------
    odds:
        ``selection -> decimal odds`` for one market (at least two selections, all > 1).
    method:
        ``'proportional'`` (default, the spec's *normalizzazione proporzionale*),
        ``'power'`` (``q_i ** k``) or ``'shin'`` (Shin 1993 insider model).

    Returns a ``dict`` in the input order whose values sum to 1 within ``1e-9``.
    Raises ``ValueError`` on invalid odds or an unknown method.
    """
    clean = _validate_odds(odds)
    chosen = _validate_method(method)
    q = np.fromiter((1.0 / o for o in clean.values()), dtype=float, count=len(clean))
    if chosen == "proportional":
        p = _devig_proportional(q)
    elif chosen == "power":
        p = _devig_power(q)
    else:
        p = _devig_shin(q)
    return {selection: float(pi) for selection, pi in zip(clean, p)}


# ----------------------------------------------------------------------------- EV & Kelly
def expected_value(p_model: float, odds: float) -> float:
    """Expected value per unit staked: ``p_model * odds - 1`` (0.10 == +10%)."""
    p = _validate_probability(p_model)
    o = _validate_decimal_odds(odds)
    return p * o - 1.0


def kelly_fraction(p_model: float, odds: float) -> float:
    """Full-Kelly fraction of bankroll ``f* = (p * b - (1 - p)) / b`` with ``b = odds - 1``.

    Negative-edge bets return ``0.0`` (never stake against the model).
    """
    p = _validate_probability(p_model)
    b = _validate_decimal_odds(odds) - 1.0
    f_star = (p * b - (1.0 - p)) / b
    return max(f_star, 0.0)


@dataclass(frozen=True)
class KellyConfig:
    """Fractional-Kelly staking parameters (all amounts in bankroll *units*).

    Attributes
    ----------
    fraction:
        Share of full Kelly to stake (``0 < fraction <= 1``); ``KELLY_EIGHTH`` by default.
    bankroll_units:
        Bankroll size expressed in units (100 => one unit is 1% of the bankroll).
    max_stake_units:
        Hard cap per selection.
    min_stake_units:
        Stakes that round below this are reported as ``0.0`` (not worth placing).
    round_to:
        Stake granularity (0.1 => stakes like 1.3, 2.5 ...).
    """

    fraction: float = KELLY_EIGHTH
    bankroll_units: float = 100.0
    max_stake_units: float = 3.0
    min_stake_units: float = 0.1
    round_to: float = 0.1

    def __post_init__(self) -> None:
        for name in ("fraction", "bankroll_units", "max_stake_units", "min_stake_units", "round_to"):
            value = getattr(self, name)
            if not isinstance(value, (int, float)) or isinstance(value, bool) or not math.isfinite(float(value)):
                raise ValueError(f"KellyConfig.{name} must be a finite number, got {value!r}")
            object.__setattr__(self, name, float(value))
        if not (0.0 < self.fraction <= 1.0):
            raise ValueError(f"KellyConfig.fraction must satisfy 0 < fraction <= 1, got {self.fraction}")
        if self.bankroll_units <= 0.0:
            raise ValueError("KellyConfig.bankroll_units must be > 0")
        if self.max_stake_units <= 0.0:
            raise ValueError("KellyConfig.max_stake_units must be > 0")
        if self.min_stake_units < 0.0 or self.min_stake_units > self.max_stake_units:
            raise ValueError("KellyConfig.min_stake_units must satisfy 0 <= min <= max_stake_units")
        if self.round_to <= 0.0:
            raise ValueError("KellyConfig.round_to must be > 0")


#: Default staking configuration (1/8 Kelly on a 100-unit bankroll, 3-unit cap).
DEFAULT_KELLY: Final[KellyConfig] = KellyConfig()


def _round_half_up_to_step(value: float, step: float) -> float:
    """Round ``value`` to the nearest multiple of ``step``, halves rounding up.

    The value is first quantised to 9 decimal places so that binary noise such
    as ``1.2500000000000004`` or ``1.2499999999999998`` is read as ``1.25``.
    """
    noise_free = Decimal(repr(float(value))).quantize(_STAKE_NOISE_PLACES, rounding=ROUND_HALF_UP)
    quantum = Decimal(repr(float(step)))
    multiples = (noise_free / quantum).quantize(Decimal(1), rounding=ROUND_HALF_UP)
    return float(multiples * quantum)


def _floor_to_step(value: float, step: float) -> float:
    noise_free = Decimal(repr(float(value))).quantize(_STAKE_NOISE_PLACES, rounding=ROUND_HALF_UP)
    quantum = Decimal(repr(float(step)))
    multiples = (noise_free / quantum).quantize(Decimal(1), rounding=ROUND_FLOOR)
    return float(multiples * quantum)


def stake_units(p_model: float, odds: float, config: KellyConfig = DEFAULT_KELLY) -> float:
    """Fractional-Kelly stake in bankroll units.

    Rule (in this order):

    1. ``raw = kelly_fraction(p_model, odds) * config.fraction * config.bankroll_units``
    2. cap: ``raw = min(raw, config.max_stake_units)``
    3. round **half up** to the nearest multiple of ``config.round_to`` using
       :class:`decimal.Decimal` (``1.25 -> 1.3`` with ``round_to=0.1``; the raw
       value is first quantised to 9 decimals so floating-point noise never
       flips a half). If rounding pushed the stake above the cap (cap not a
       multiple of ``round_to``) the stake is floored to the step instead.
    4. ``0.0`` when the rounded stake is below ``config.min_stake_units``.
    """
    full = kelly_fraction(p_model, odds)
    if full <= 0.0:
        return 0.0
    raw = full * config.fraction * config.bankroll_units
    capped = min(raw, config.max_stake_units)
    rounded = _round_half_up_to_step(capped, config.round_to)
    if rounded > config.max_stake_units:
        rounded = _floor_to_step(config.max_stake_units, config.round_to)
    if rounded < config.min_stake_units:
        return 0.0
    return rounded


# ----------------------------------------------------------------------------- regression trap
@dataclass(frozen=True)
class RegressionTrap:
    """A team whose recent clean sheets are not backed by the xG it conceded.

    ``goals_against <= max`` while ``xg_against >= min`` over the last
    ``window`` matches: the defence over-performed and is expected to regress,
    so goals-based models (and the market) under-price goals against it.

    ``severity = xg_against - goals_against`` and ``xga_per_match = xg_against / window``
    are stored explicitly so the object is self-describing once serialised; the
    constructor checks they are consistent. Use :meth:`from_window` to build one
    from the raw totals.
    """

    team: str
    girone: Girone
    window: int
    goals_against: int
    xg_against: float
    match_ids: tuple[str, ...]
    last_date: date
    xga_per_match: float
    severity: float

    def __post_init__(self) -> None:
        object.__setattr__(self, "team", str(self.team))
        object.__setattr__(self, "girone", Girone.parse(self.girone))
        object.__setattr__(self, "window", int(self.window))
        object.__setattr__(self, "goals_against", int(self.goals_against))
        object.__setattr__(self, "xg_against", float(self.xg_against))
        object.__setattr__(self, "match_ids", tuple(str(m) for m in self.match_ids))
        object.__setattr__(self, "last_date", _coerce_date(self.last_date))
        object.__setattr__(self, "xga_per_match", float(self.xga_per_match))
        object.__setattr__(self, "severity", float(self.severity))
        if self.window < 1:
            raise ValueError("RegressionTrap.window must be >= 1")
        if self.goals_against < 0:
            raise ValueError("RegressionTrap.goals_against cannot be negative")
        if not math.isfinite(self.xg_against) or self.xg_against < 0.0:
            raise ValueError("RegressionTrap.xg_against must be a finite non-negative number")
        if abs(self.severity - (self.xg_against - self.goals_against)) > 1e-6:
            raise ValueError("RegressionTrap.severity must equal xg_against - goals_against")
        if abs(self.xga_per_match - self.xg_against / self.window) > 1e-6:
            raise ValueError("RegressionTrap.xga_per_match must equal xg_against / window")

    @classmethod
    def from_window(
        cls,
        team: str,
        girone: Girone | str,
        window: int,
        goals_against: int,
        xg_against: float,
        match_ids: Sequence[str],
        last_date: date | datetime | pd.Timestamp | str,
    ) -> RegressionTrap:
        """Build a trap from the window totals, deriving ``xga_per_match`` and ``severity``."""
        window_i = int(window)
        if window_i < 1:
            raise ValueError("window must be >= 1")
        xga = float(xg_against)
        ga = int(goals_against)
        return cls(
            team=team,
            girone=girone,
            window=window_i,
            goals_against=ga,
            xg_against=xga,
            match_ids=tuple(match_ids),
            last_date=last_date,
            xga_per_match=xga / window_i,
            severity=xga - ga,
        )

    def description(self) -> str:
        """One-line Italian summary for console output and alerts."""
        return (
            f"{self.team} (Girone {self.girone.value}): {self.goals_against} gol subiti nelle ultime "
            f"{self.window} gare contro {self.xg_against:.2f} xGA ({self.xga_per_match:.2f} a partita) "
            f"- difesa in sovra-rendimento, regressione attesa, severita {self.severity:+.2f}, "
            f"ultima gara {self.last_date.isoformat()}"
        )

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["girone"] = self.girone.value
        d["match_ids"] = "|".join(self.match_ids)
        return d


def _team_match_long_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Melt the canonical frame into one row per (team, match) with the goals/xG *conceded*."""
    home = pd.DataFrame(
        {
            "match_id": df["match_id"],
            "date": df["date"],
            "girone": df["girone"],
            "team": df["home_team"],
            "opponent": df["away_team"],
            "goals_against": df["away_goals"],
            "xg_against": df["away_xg"],
        }
    )
    away = pd.DataFrame(
        {
            "match_id": df["match_id"],
            "date": df["date"],
            "girone": df["girone"],
            "team": df["away_team"],
            "opponent": df["home_team"],
            "goals_against": df["home_goals"],
            "xg_against": df["home_xg"],
        }
    )
    return pd.concat([home, away], ignore_index=True)


def detect_regression_traps(
    matches: pd.DataFrame,
    *,
    window: int = 3,
    max_goals_against: int = 1,
    min_xg_against: float = 3.5,
    as_of: date | None = None,
) -> list[RegressionTrap]:
    """Flag teams whose last ``window`` matches show ``GA <= max`` but ``xGA >= min``.

    ``matches`` is the canonical frame (:func:`validate_matches_df` is applied).
    For every team the last ``window`` matches by date (ties broken by
    ``match_id``) with ``date <= as_of`` are taken; a team qualifies only when it
    has exactly ``window`` such matches and the opponent's xG is known in all of
    them. The result is sorted by ``severity`` descending (ties by team name).
    Pure function: the input frame is never modified.
    """
    window = int(window)
    if window < 1:
        raise ValueError("window must be >= 1")
    if max_goals_against < 0:
        raise ValueError("max_goals_against cannot be negative")
    if not math.isfinite(float(min_xg_against)) or min_xg_against < 0.0:
        raise ValueError("min_xg_against must be a finite non-negative number")

    df = validate_matches_df(matches)
    if as_of is not None:
        cutoff = pd.Timestamp(_coerce_date(as_of))
        df = df.loc[df["date"] <= cutoff]
    if df.empty:
        log.info("regression trap scan on empty frame", extra={"window": window, "as_of": str(as_of)})
        return []

    long = _team_match_long_frame(df)
    long = long.sort_values(["team", "date", "match_id"], kind="stable")
    recent = long.groupby("team", sort=False, observed=True).tail(window)
    agg = recent.groupby("team", sort=True, observed=True).agg(
        n=("match_id", "size"),
        n_xg=("xg_against", "count"),
        goals_against=("goals_against", "sum"),
        xg_against=("xg_against", "sum"),
        last_date=("date", "max"),
        girone=("girone", "last"),
        match_ids=("match_id", lambda s: tuple(str(m) for m in s)),
    )
    eligible = (agg["n"] == window) & (agg["n_xg"] == window)
    flagged = eligible & (agg["goals_against"] <= max_goals_against) & (agg["xg_against"] >= float(min_xg_against))
    hits = agg.loc[flagged]

    traps = [
        RegressionTrap.from_window(
            team=str(team),
            girone=Girone.parse(str(row.girone)),
            window=window,
            goals_against=int(row.goals_against),
            xg_against=float(row.xg_against),
            match_ids=tuple(row.match_ids),
            last_date=row.last_date,
        )
        for team, row in hits.iterrows()
    ]
    traps.sort(key=lambda t: (-t.severity, t.team))
    log.info(
        "regression trap scan",
        extra={
            "teams": len(agg),
            "eligible": int(eligible.sum()),
            "traps": len(traps),
            "window": window,
            "max_goals_against": int(max_goals_against),
            "min_xg_against": float(min_xg_against),
            "as_of": None if as_of is None else _coerce_date(as_of).isoformat(),
        },
    )
    return traps


# ----------------------------------------------------------------------------- opportunities
def _is_goals_selection(selection: str) -> bool:
    """``Over x.x`` and ``GG`` are the selections a regression trap concerns."""
    text = str(selection).strip()
    return text.startswith("Over") or text == "GG"


@dataclass(frozen=True)
class ValueOpportunity:
    """One quoted selection evaluated against the model (value or not)."""

    match_id: str
    date: date
    girone: Girone
    home_team: str
    away_team: str
    market: Market
    selection: str
    bookmaker: str
    book_odds: float
    fair_odds: float
    p_model: float
    p_implied_raw: float
    p_implied_devig: float
    ev: float
    edge: float
    kelly_full: float
    kelly_fraction: float
    stake_units: float
    value: bool
    flags: tuple[str, ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        object.__setattr__(self, "date", _coerce_date(self.date))
        object.__setattr__(self, "girone", Girone.parse(self.girone))
        object.__setattr__(self, "market", Market(self.market))
        object.__setattr__(self, "flags", tuple(str(f) for f in self.flags))
        object.__setattr__(self, "value", bool(self.value))

    @property
    def label(self) -> str:
        return f"{self.home_team} - {self.away_team} | {self.market.value} {self.selection} @ {self.book_odds:.2f}"

    @property
    def is_trap(self) -> bool:
        return FLAG_REGRESSION_TRAP in self.flags

    def to_dict(self) -> dict[str, Any]:
        """Plain-type mapping: ``market``/``girone`` as their ``.value``, ``flags`` joined by ``'|'``."""
        d = asdict(self)
        d["girone"] = self.girone.value
        d["market"] = self.market.value
        d["flags"] = "|".join(self.flags)
        return d


class ValueEngine:
    """Joins model predictions with bookmaker quotes and sizes the value bets.

    Parameters
    ----------
    min_ev:
        Minimum expected value (fraction) for the ``VALUE`` flag and a non-zero stake.
    kelly:
        Fractional-Kelly staking configuration.
    devig_method:
        How the bookmaker margin is removed for ``p_implied_devig`` / ``edge``.
    trap_window, trap_max_goals_against, trap_min_xg_against:
        Parameters forwarded to :func:`detect_regression_traps` when ``evaluate``
        receives ``matches`` but no pre-computed ``traps``.
    longshot_odds:
        Selections priced above this get the ``LONGSHOT`` flag.
    suspicious_ev:
        An EV above this is flagged ``SUSPICIOUS_EDGE`` (likely stale odds or bad data).
    """

    def __init__(
        self,
        *,
        min_ev: float = 0.035,
        kelly: KellyConfig = DEFAULT_KELLY,
        devig_method: str = "proportional",
        trap_window: int = 3,
        trap_max_goals_against: int = 1,
        trap_min_xg_against: float = 3.5,
        longshot_odds: float = 6.0,
        suspicious_ev: float = 0.25,
    ) -> None:
        if not math.isfinite(float(min_ev)):
            raise ValueError("min_ev must be finite")
        if not isinstance(kelly, KellyConfig):
            raise TypeError("kelly must be a KellyConfig")
        if int(trap_window) < 1:
            raise ValueError("trap_window must be >= 1")
        if int(trap_max_goals_against) < 0:
            raise ValueError("trap_max_goals_against cannot be negative")
        if float(trap_min_xg_against) < 0.0:
            raise ValueError("trap_min_xg_against cannot be negative")
        if float(longshot_odds) <= 1.0:
            raise ValueError("longshot_odds must be > 1")
        if not math.isfinite(float(suspicious_ev)):
            raise ValueError("suspicious_ev must be finite")
        self.min_ev: float = float(min_ev)
        self.kelly: KellyConfig = kelly
        self.devig_method: str = _validate_method(devig_method)
        self.trap_window: int = int(trap_window)
        self.trap_max_goals_against: int = int(trap_max_goals_against)
        self.trap_min_xg_against: float = float(trap_min_xg_against)
        self.longshot_odds: float = float(longshot_odds)
        self.suspicious_ev: float = float(suspicious_ev)

    # ------------------------------------------------------------------ core
    def _trapped_teams(
        self,
        traps: Iterable[RegressionTrap] | None,
        matches: pd.DataFrame | None,
    ) -> dict[str, RegressionTrap]:
        if traps is None:
            if matches is None:
                return {}
            traps = detect_regression_traps(
                matches,
                window=self.trap_window,
                max_goals_against=self.trap_max_goals_against,
                min_xg_against=self.trap_min_xg_against,
            )
        by_team: dict[str, RegressionTrap] = {}
        for trap in traps:
            current = by_team.get(trap.team)
            if current is None or trap.severity > current.severity:
                by_team[trap.team] = trap
        return by_team

    def _flags_for(
        self,
        *,
        ev: float,
        book_odds: float,
        selection: str,
        home_team: str,
        away_team: str,
        trapped: Mapping[str, RegressionTrap],
    ) -> tuple[str, ...]:
        flags: list[str] = []
        if ev >= self.min_ev:
            flags.append(FLAG_VALUE)
        if _is_goals_selection(selection):
            trapped_here = [t for t in (home_team, away_team) if t in trapped]
            if trapped_here:
                flags.append(FLAG_REGRESSION_TRAP)
                flags.extend(f"{TRAP_FLAG_PREFIX}{t}" for t in trapped_here)
        if book_odds > self.longshot_odds:
            flags.append(FLAG_LONGSHOT)
        if ev > self.suspicious_ev:
            flags.append(FLAG_SUSPICIOUS_EDGE)
        return tuple(flags)

    def evaluate(
        self,
        predictions: Iterable[MatchPrediction],
        quotes: Iterable[OddsQuote],
        *,
        matches: pd.DataFrame | None = None,
        traps: Iterable[RegressionTrap] | None = None,
    ) -> list[ValueOpportunity]:
        """Evaluate every selection of every quote that has a matching prediction.

        Quotes whose ``match_id`` has no prediction (or whose market the
        prediction does not cover) are skipped with a warning; predictions
        without quotes produce nothing. Negative-EV selections are returned
        too (``value=False``, ``stake_units=0.0``) so callers can audit the
        whole book. Result sorted by ``ev`` descending.
        """
        preds: dict[str, MatchPrediction] = {}
        for pred in predictions:
            if pred.match_id in preds:
                log.debug("duplicate prediction, keeping the last one", extra={"match_id": pred.match_id})
            preds[pred.match_id] = pred
        quote_list = list(quotes)
        trapped = self._trapped_teams(traps, matches)

        opportunities: list[ValueOpportunity] = []
        skipped = 0
        for quote in quote_list:
            pred = preds.get(quote.match_id)
            if pred is None:
                skipped += 1
                log.warning(
                    "quote without prediction skipped",
                    extra={"match_id": quote.match_id, "bookmaker": quote.bookmaker, "market": quote.market.value},
                )
                continue
            model_probs = pred.probabilities.get(quote.market)
            if model_probs is None:
                skipped += 1
                log.warning(
                    "prediction does not cover quoted market",
                    extra={"match_id": quote.match_id, "bookmaker": quote.bookmaker, "market": quote.market.value},
                )
                continue
            fair_table = pred.fair_odds.get(quote.market, {})
            raw = implied_probabilities(quote.odds)
            fair_probs = devig(quote.odds, self.devig_method)
            fx = pred.fixture
            for selection in SELECTIONS[quote.market]:
                book_odds = float(quote.odds[selection])
                p_model = _validate_probability(model_probs[selection])
                fair_odds = float(fair_table.get(selection, math.inf if p_model <= 0.0 else 1.0 / p_model))
                ev = expected_value(p_model, book_odds)
                p_devig = fair_probs[selection]
                full = kelly_fraction(p_model, book_odds)
                flags = self._flags_for(
                    ev=ev,
                    book_odds=book_odds,
                    selection=selection,
                    home_team=fx.home_team,
                    away_team=fx.away_team,
                    trapped=trapped,
                )
                is_value = FLAG_VALUE in flags
                stake = stake_units(p_model, book_odds, self.kelly) if is_value else 0.0
                opportunities.append(
                    ValueOpportunity(
                        match_id=quote.match_id,
                        date=fx.date,
                        girone=fx.girone,
                        home_team=fx.home_team,
                        away_team=fx.away_team,
                        market=quote.market,
                        selection=selection,
                        bookmaker=quote.bookmaker,
                        book_odds=book_odds,
                        fair_odds=fair_odds,
                        p_model=p_model,
                        p_implied_raw=raw[selection],
                        p_implied_devig=p_devig,
                        ev=ev,
                        edge=p_model - p_devig,
                        kelly_full=full,
                        kelly_fraction=self.kelly.fraction,
                        stake_units=stake,
                        value=is_value,
                        flags=flags,
                    )
                )
        opportunities.sort(key=lambda o: (-o.ev, o.match_id, o.market.value, o.selection, o.bookmaker))
        log.info(
            "value evaluation",
            extra={
                "predictions": len(preds),
                "quotes": len(quote_list),
                "skipped_quotes": skipped,
                "selections": len(opportunities),
                "value": sum(1 for o in opportunities if o.value),
                "trapped_teams": sorted(trapped),
                "min_ev": self.min_ev,
                "devig_method": self.devig_method,
            },
        )
        return opportunities

    # ------------------------------------------------------------------ views
    def value_bets(self, opps: Iterable[ValueOpportunity]) -> list[ValueOpportunity]:
        """Only the ``VALUE`` selections, EV descending."""
        return sorted(
            (o for o in opps if o.value),
            key=lambda o: (-o.ev, o.match_id, o.market.value, o.selection, o.bookmaker),
        )

    def to_frame(self, opps: Iterable[ValueOpportunity]) -> pd.DataFrame:
        """Tabular view with columns exactly :data:`VALUE_FRAME_COLUMNS` (``ev``/``edge`` as fractions)."""
        rows = [o.to_dict() for o in opps]
        if not rows:
            empty = pd.DataFrame({c: pd.Series(dtype=t) for c, t in _VALUE_FRAME_DTYPES.items()})
            return empty[list(VALUE_FRAME_COLUMNS)]
        frame = pd.DataFrame(rows)[list(VALUE_FRAME_COLUMNS)]
        frame["date"] = pd.to_datetime(frame["date"]).astype("datetime64[ns]")
        for column, dtype in _VALUE_FRAME_DTYPES.items():
            if dtype == "string":
                frame[column] = frame[column].astype("string")
            elif dtype == "float64":
                frame[column] = frame[column].astype("float64")
        frame["value"] = frame["value"].astype("bool")
        return frame.reset_index(drop=True)

    def summary(self, opps: Iterable[ValueOpportunity]) -> dict[str, Any]:
        """Aggregate counters: selections, value bets, mean EV of value bets, stakes, traps, per-market counts."""
        items = list(opps)
        value = [o for o in items if o.value]
        by_market = {m.value: 0 for m in Market}
        for o in value:
            by_market[o.market.value] += 1
        return {
            "n_selections": len(items),
            "n_value": len(value),
            "mean_ev_value": float(np.mean([o.ev for o in value])) if value else 0.0,
            "total_stake_units": float(math.fsum(o.stake_units for o in items)),
            "n_trap_flags": sum(1 for o in items if FLAG_REGRESSION_TRAP in o.flags),
            "by_market": by_market,
        }

    def export_csv(self, opps: Iterable[ValueOpportunity], path: Path | str) -> Path:
        """Write :meth:`to_frame` to ``path`` (parents created) and return the path."""
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        frame = self.to_frame(opps)
        frame.to_csv(target, index=False, date_format="%Y-%m-%d")
        log.info("value report exported", extra={"path": str(target), "rows": len(frame)})
        return target

    @staticmethod
    def traps_frame(traps: Iterable[RegressionTrap]) -> pd.DataFrame:
        """Tabular view of traps with columns :data:`TRAPS_FRAME_COLUMNS` (``match_ids`` joined by ``'|'``)."""
        rows = [t.to_dict() for t in traps]
        if not rows:
            empty = pd.DataFrame({c: pd.Series(dtype=t) for c, t in _TRAPS_FRAME_DTYPES.items()})
            return empty[list(TRAPS_FRAME_COLUMNS)]
        frame = pd.DataFrame(rows)[list(TRAPS_FRAME_COLUMNS)]
        frame["last_date"] = pd.to_datetime(frame["last_date"]).astype("datetime64[ns]")
        for column, dtype in _TRAPS_FRAME_DTYPES.items():
            if dtype == "string":
                frame[column] = frame[column].astype("string")
        frame["goals_against"] = frame["goals_against"].astype("int64")
        return frame.reset_index(drop=True)
