"""Matchday slips (schedine): three accumulators proposed by the model for one matchday.

This module is pure (no ``streamlit``, no network): it depends only on the
shared contracts of :mod:`src.common.schemas`, on the kick-off rule of
:mod:`src.engine.calendar`, on the hints of :mod:`src.engine.picks` and on the
staking helpers of :mod:`src.engine.value`, so the dashboard, the console and
the walk-forward backtest (:func:`src.engine.backtest.slips_track_record`) share
one implementation.

The three slips
---------------
For the matches of the selected matchday that are **still to play** (a fixture
that has kicked off at ``now`` or is postponed is never offered), in the
sections chosen by the user:

* **Schedina X** (:data:`DEFAULT_N_X` = 5 legs): the matches with the highest
  model probability of the draw (``X`` of the 1X2 market);
* **Schedina GG** (:data:`DEFAULT_N_GG` = 10 legs): the matches with the highest
  model probability of both teams scoring (``GG``);
* **Schedina mista** (:data:`DEFAULT_N_MIXED` = 6 legs, configurable between
  :data:`MIXED_MIN_LEGS` and :data:`MIXED_MAX_LEGS`): for every match its best
  sign among :data:`DEFAULT_MIXED_MARKETS` (``1``, ``2``, ``GG``, ``Over 2.5``)
  by model probability, then the matches whose best sign is the most probable.

A match appears at most once per slip. Legs are ranked by model probability;
ties go to the higher edge against the best book price (``EV = p * odds - 1``,
quoted legs before unquoted ones), then to the earlier kick-off, then to the
``match_id`` (a total order: the result is deterministic). Optional filters: a
minimum leg probability (``min_probability``) and "only legs with book odds"
(``only_quoted``; in the mixed slip the best sign is then chosen among the
quoted ones).

Probabilities, prices and stake
-------------------------------
The leg probability is the model probability, optionally passed through
``adjust`` (the validated recalibration of :mod:`src.engine.calibration`:
:meth:`RecalibrationModel.apply` has the expected signature, see
:class:`ProbabilityAdjuster`); the leg fair odds are ``1 / p``. The best book
price of a selection is the **highest** across bookmakers (value opportunities,
raw quotes and, for frames, ``odds_<suffix>`` columns).

* slip probability = product of the leg probabilities (**independence
  assumption**, :data:`INDEPENDENCE_NOTE`), fair odds = ``1 / probability``;
* book odds = product of the best book prices, only when **every** leg is
  quoted (else ``None``), ``EV = probability * book odds - 1``;
* expected winning legs = sum of the leg probabilities;
* the bookmaker margin compounds over the legs (:attr:`Slip.margin_note`):
  ``1 - prod(1 / (1 + m_i))`` with ``m_i`` the margin of leg ``i``;
* :func:`suggested_stake`: fractional Kelly on the whole slip when its EV is
  positive, capped (default 1 unit); ``0`` (*non consigliata*) otherwise.

Backtest replay
---------------
:func:`slips_from_frame` builds the same slips from a probability frame (one
row per match, ``p_1``, ``p_X``, ``p_2``, ``p_GG``, ``p_O25`` ... columns, the
pre-match frame of :func:`src.engine.backtest.slips_track_record`), with the
identical ranking logic: called with the frame alone it is the
``SlipBuilder`` the backtest expects (``(frame) -> list[Slip]``);
:func:`make_slip_builder` binds non-default options. :func:`predictions_frame`
turns predictions into such a frame (``slips_from_frame(predictions_frame(p))``
equals ``build_slips(p)``).
"""
from __future__ import annotations

import math
import operator
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, date, datetime
from typing import Any, Final, Literal, NamedTuple, Protocol

import pandas as pd

from src.common.logging_config import get_logger
from src.common.schemas import SELECTIONS, Fixture, FixtureStatus, Girone, Market, MatchPrediction, OddsQuote
from src.engine.calendar import format_kickoff, kicked_off, rome_midnight
from src.engine.picks import (
    _BTTS_BOTH_ATTACKS_MIN,
    _BTTS_HIGH_XG_TOTAL,
    _DRAW_CLOSE_RATES_GAP,
    _DRAW_FEW_GOALS_TOTAL,
    BTTS_NOTE_BOTH_ATTACKS,
    BTTS_NOTE_HIGH_XG,
    DRAW_NOTE_CLOSE_RATES,
    DRAW_NOTE_FEW_GOALS,
    Matchday,
)
from src.engine.value import KellyConfig, ValueOpportunity, stake_units

__all__ = [
    "DEFAULT_MIXED_MARKETS",
    "DEFAULT_N_GG",
    "DEFAULT_N_MIXED",
    "DEFAULT_N_X",
    "DEFAULT_STAKE_CAP_UNITS",
    "DEFAULT_STAKE_FRACTION",
    "DEFAULT_STAKE_ROUND_TO",
    "FRAME_MARKET_SUFFIX",
    "FRAME_SELECTION_SUFFIX",
    "INDEPENDENCE_NOTE",
    "MIXED_MAX_LEGS",
    "MIXED_MIN_LEGS",
    "PREDICTIONS_FRAME_COLUMNS",
    "SLIPS_COLUMNS",
    "SLIPS_SUMMARY_COLUMNS",
    "SLIP_KINDS",
    "STAKE_NOT_ADVISED",
    "TYPICAL_LEG_MARGIN",
    "ProbabilityAdjuster",
    "Slip",
    "SlipKind",
    "SlipLeg",
    "build_slips",
    "format_stake",
    "make_slip_builder",
    "predictions_frame",
    "slips_frame",
    "slips_from_frame",
    "slips_summary",
    "suggested_stake",
]

log = get_logger(__name__)

# =============================================================================
# Constants
# =============================================================================
SlipKind = Literal["X", "GG", "MISTA"]
#: The three slips, in the order :func:`build_slips` returns them.
SLIP_KINDS: Final[tuple[str, ...]] = ("X", "GG", "MISTA")

#: Default number of legs of the draw slip, of the GG slip and of the mixed slip.
DEFAULT_N_X: Final[int] = 5
DEFAULT_N_GG: Final[int] = 10
DEFAULT_N_MIXED: Final[int] = 6
#: Allowed length of the mixed slip (the dashboard slider range).
MIXED_MIN_LEGS: Final[int] = 3
MIXED_MAX_LEGS: Final[int] = 10
#: Signs the mixed slip chooses from, as ``(market, selection)``: home win, away win, GG, Over 2.5.
DEFAULT_MIXED_MARKETS: Final[tuple[tuple[str, str], ...]] = (
    ("1X2", "1"),
    ("1X2", "2"),
    ("GG/NG", "GG"),
    ("OU2.5", "Over 2.5"),
)

#: Default fractional Kelly of :func:`suggested_stake` (1/8), its cap and its granularity (bankroll units).
DEFAULT_STAKE_FRACTION: Final[float] = 0.125
DEFAULT_STAKE_CAP_UNITS: Final[float] = 1.0
DEFAULT_STAKE_ROUND_TO: Final[float] = 0.01
#: Margin per leg assumed by :attr:`Slip.margin_note` when the real one is unknown (5%).
TYPICAL_LEG_MARGIN: Final[float] = 0.05

#: Italian statement of the independence assumption behind the slip probability.
INDEPENDENCE_NOTE: Final[str] = (
    "Probabilità della schedina = prodotto delle probabilità delle singole partite "
    "(ipotesi di indipendenza tra le partite)."
)
#: Italian label of a zero stake (:func:`format_stake`).
STAKE_NOT_ADVISED: Final[str] = "non consigliata"

#: Column suffix of every selection in a probability frame (``p_O25`` = model probability of ``Over 2.5``;
#: the same suffixes as :data:`src.engine.backtest.SELECTION_SUFFIX`).
FRAME_SELECTION_SUFFIX: Final[dict[str, str]] = {
    "1": "1",
    "X": "X",
    "2": "2",
    "GG": "GG",
    "NG": "NG",
    "Over 1.5": "O15",
    "Under 1.5": "U15",
    "Over 2.5": "O25",
    "Under 2.5": "U25",
    "Over 3.5": "O35",
    "Under 3.5": "U35",
}
#: Column suffix of every market for the optional ``overround_<suffix>`` columns of a frame
#: (as :func:`src.engine.backtest.odds_table`).
FRAME_MARKET_SUFFIX: Final[dict[Market, str]] = {
    Market.MATCH_ODDS: "1X2",
    Market.BTTS: "GGNG",
    Market.OVER_UNDER_15: "OU15",
    Market.OVER_UNDER_25: "OU25",
    Market.OVER_UNDER_35: "OU35",
}

#: Columns of :func:`predictions_frame`, in order.
PREDICTIONS_FRAME_COLUMNS: Final[tuple[str, ...]] = (
    "match_id",
    "date",
    "kickoff",
    "status",
    "round",
    "girone",
    "family",
    "section",
    "home_team",
    "away_team",
    "lam",
    "mu",
    "rho",
    *(f"p_{FRAME_SELECTION_SUFFIX[s]}" for market in SELECTIONS for s in SELECTIONS[market]),
)

#: Italian columns of :func:`slips_frame`, in order.
SLIPS_COLUMNS: Final[tuple[str, ...]] = (
    "Schedina",
    "#",
    "Competizione",
    "Data e ora",
    "Partita",
    "Esito",
    "P. Modello (%)",
    "Quota Fair",
    "Quota Book",
    "Bookmaker",
    "Nota",
)
_SLIPS_DTYPES: Final[dict[str, str]] = {
    "Schedina": "string",
    "#": "int64",
    "Competizione": "string",
    "Data e ora": "string",
    "Partita": "string",
    "Esito": "string",
    "P. Modello (%)": "float64",
    "Quota Fair": "float64",
    "Quota Book": "float64",
    "Bookmaker": "string",
    "Nota": "string",
}
#: Italian columns of :func:`slips_summary`, in order.
SLIPS_SUMMARY_COLUMNS: Final[tuple[str, ...]] = (
    "Schedina",
    "Partite",
    "Probabilità (%)",
    "Quota fair",
    "Quota book",
    "EV (%)",
    "Partite vincenti attese",
    "Stake suggerito",
)
_SUMMARY_DTYPES: Final[dict[str, str]] = {
    "Schedina": "string",
    "Partite": "int64",
    "Probabilità (%)": "float64",
    "Quota fair": "float64",
    "Quota book": "float64",
    "EV (%)": "float64",
    "Partite vincenti attese": "float64",
    "Stake suggerito": "string",
}

_TITLE_X: Final[str] = "Schedina X · pareggi"
_TITLE_GG: Final[str] = "Schedina GG · goal"
_TITLE_MIXED: Final[str] = "Schedina mista"
#: Frame columns every row must carry.
_FRAME_REQUIRED: Final[tuple[str, ...]] = ("match_id", "date", "girone", "home_team", "away_team")
#: Probabilities slightly outside ``[0, 1]`` (floating-point noise of an adjuster) are clipped within this tolerance.
_PROBABILITY_TOLERANCE: Final[float] = 1e-9


class ProbabilityAdjuster(Protocol):
    """Correction applied to the model probabilities of one match before the slips are built.

    It receives the probabilities of every market of the match
    (``{Market: {selection: p}}``) and the match's family (``'Serie C'``) and
    returns the corrected mapping (keys as :class:`Market` members or their
    values). :meth:`src.engine.calibration.RecalibrationModel.apply` matches
    this signature, so the dashboard passes ``recalibration.apply`` as is.
    """

    def __call__(self, probabilities: Mapping[Market, Mapping[str, float]], *, family: str | None = None) -> Mapping[Any, Mapping[str, float]]:
        ...  # pragma: no cover - protocol


# =============================================================================
# Data classes
# =============================================================================
def _product(values: Iterable[float]) -> float:
    """Product of floats (``1.0`` for an empty iterable)."""
    out = 1.0
    for value in values:
        out *= float(value)
    return out


@dataclass(frozen=True)
class SlipLeg:
    """One leg of a slip: a selection of one match with its model probability and the best book price.

    ``p_model`` is the (possibly recalibrated) model probability and
    ``fair_odds`` its inverse; ``book_odds``/``bookmaker`` are the best price
    across bookmakers (``None`` when nobody quotes it), ``ev`` is
    ``p_model * book_odds - 1`` and ``book_margin`` the bookmaker margin on the
    selection (``1 / payout - 1``: the over-round of the book the price comes
    from), both ``None`` without a price. ``kickoff`` is the aware UTC kick-off
    (``None`` when the source gave only the day), ``competition`` the
    :class:`Girone` member, ``section`` the dashboard section. ``note`` is
    ``''`` or a short Italian hint derived from the model (never from the price).
    """

    match_id: str
    date: date
    kickoff: datetime | None
    competition: Girone
    section: str
    home_team: str
    away_team: str
    market: Market
    selection: str
    p_model: float
    fair_odds: float
    book_odds: float | None = None
    bookmaker: str | None = None
    ev: float | None = None
    book_margin: float | None = None
    note: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(self, "match_id", str(self.match_id))
        object.__setattr__(self, "competition", Girone.parse(self.competition))
        object.__setattr__(self, "market", Market(self.market))
        if self.selection not in SELECTIONS[self.market]:
            raise ValueError(f"{self.selection!r} is not a selection of {self.market.value}")
        p = float(self.p_model)
        if not math.isfinite(p) or not 0.0 <= p <= 1.0:
            raise ValueError(f"p_model must be in [0, 1], got {self.p_model!r}")
        object.__setattr__(self, "p_model", p)
        fair = float(self.fair_odds)
        if math.isnan(fair) or fair <= 0.0:
            raise ValueError(f"fair_odds must be > 0, got {self.fair_odds!r}")
        object.__setattr__(self, "fair_odds", fair)
        if self.book_odds is not None:
            odds = float(self.book_odds)
            if not math.isfinite(odds) or odds <= 1.0:
                raise ValueError(f"book_odds must be finite and > 1.0, got {self.book_odds!r}")
            object.__setattr__(self, "book_odds", odds)

    @property
    def girone(self) -> Girone:
        """Alias of :attr:`competition` (the name used by :class:`~src.engine.picks.Pick`)."""
        return self.competition

    @property
    def label(self) -> str:
        """``'Home - Away'``."""
        return f"{self.home_team} - {self.away_team}"

    @property
    def is_quoted(self) -> bool:
        """Whether some bookmaker quotes the selection."""
        return self.book_odds is not None

    @property
    def when(self) -> str:
        """The kick-off as shown in tables, ``'dom 08/11 14:30'`` (Europe/Rome; the day alone without a time)."""
        return format_kickoff(self.kickoff, self.date)


@dataclass(frozen=True)
class Slip:
    """A slip (schedina): one selection per match, at most one leg per match.

    ``kind`` is ``'X'``, ``'GG'`` or ``'MISTA'``, ``title`` its Italian name,
    ``n_requested`` the number of legs asked for (the slip is shorter when fewer
    matches qualify, see :attr:`is_complete`). The figures are derived from the
    legs: :attr:`probability` (product, independence assumed), :attr:`fair_odds`,
    :attr:`book_odds` (only when every leg is quoted), :attr:`ev`,
    :attr:`expected_winning_legs` and :attr:`margin_note`. An empty slip has
    ``NaN`` probability and fair odds and no book odds.
    """

    kind: SlipKind
    title: str
    legs: tuple[SlipLeg, ...]
    n_requested: int = 0

    def __post_init__(self) -> None:
        if self.kind not in SLIP_KINDS:
            raise ValueError(f"slip kind must be one of {SLIP_KINDS}, got {self.kind!r}")
        legs = tuple(self.legs)
        ids = [leg.match_id for leg in legs]
        if len(set(ids)) != len(ids):
            duplicated = sorted({i for i in ids if ids.count(i) > 1})
            raise ValueError(f"a match can appear only once per slip, duplicated: {duplicated}")
        object.__setattr__(self, "legs", legs)
        object.__setattr__(self, "n_requested", int(self.n_requested))

    # ------------------------------------------------------------------ figures
    @property
    def n_legs(self) -> int:
        return len(self.legs)

    @property
    def is_empty(self) -> bool:
        return not self.legs

    @property
    def is_complete(self) -> bool:
        """Whether the slip has the number of legs asked for."""
        return self.n_legs >= self.n_requested

    @property
    def probability(self) -> float:
        """Probability that every leg wins, the product of the leg probabilities (``NaN`` for an empty slip)."""
        if not self.legs:
            return float("nan")
        return _product(leg.p_model for leg in self.legs)

    @property
    def fair_odds(self) -> float:
        """``1 / probability`` (``inf`` for a zero probability, ``NaN`` for an empty slip)."""
        p = self.probability
        if math.isnan(p):
            return float("nan")
        return 1.0 / p if p > 0.0 else float("inf")

    @property
    def is_quoted(self) -> bool:
        """Whether every leg has a book price (and the slip has legs)."""
        return bool(self.legs) and all(leg.book_odds is not None for leg in self.legs)

    @property
    def book_odds(self) -> float | None:
        """Product of the best book prices, ``None`` unless every leg is quoted."""
        if not self.is_quoted:
            return None
        return _product(float(leg.book_odds) for leg in self.legs if leg.book_odds is not None)

    @property
    def ev(self) -> float | None:
        """Expected value per unit staked, ``probability * book_odds - 1`` (``None`` without book odds)."""
        odds = self.book_odds
        return None if odds is None else self.probability * odds - 1.0

    @property
    def expected_winning_legs(self) -> float:
        """Expected number of winning legs, the sum of the leg probabilities."""
        return float(sum(leg.p_model for leg in self.legs))

    @property
    def compound_margin(self) -> float | None:
        """Bookmaker margin of the whole slip, ``1 - prod(1 / (1 + m_i))``; ``None`` unless every leg's margin is known."""
        margins = [leg.book_margin for leg in self.legs]
        if not margins or any(m is None for m in margins):
            return None
        return 1.0 - _product(1.0 / (1.0 + float(m)) for m in margins if m is not None)

    @property
    def margin_note(self) -> str:
        """Italian warning that an accumulator multiplies the bookmaker margin, with the figure for this slip."""
        n = self.n_legs
        if n == 0:
            return ""
        compound = self.compound_margin
        if compound is not None:
            mean_leg = sum(float(leg.book_margin) for leg in self.legs if leg.book_margin is not None) / n
            return (
                f"Le multiple moltiplicano il margine del bookmaker: su queste {n} partite "
                f"il margine composto è {compound * 100:.1f}% (in media {mean_leg * 100:.1f}% per partita)."
            )
        typical = 1.0 - (1.0 / (1.0 + TYPICAL_LEG_MARGIN)) ** n
        return (
            f"Le multiple moltiplicano il margine del bookmaker: con un margine tipico del "
            f"{TYPICAL_LEG_MARGIN * 100:.0f}% per partita, {n} partite cedono circa il {typical * 100:.1f}% della vincita equa."
        )

    @property
    def coverage_note(self) -> str:
        """``''`` for a complete slip, else ``'Solo 3 partite idonee su 5 richieste.'``."""
        if self.is_complete:
            return ""
        if self.is_empty:
            return "Nessuna partita idonea."
        return f"Solo {self.n_legs} partite idonee su {self.n_requested} richieste."


# =============================================================================
# Internal representation shared by the two builders
# =============================================================================
class _Quote(NamedTuple):
    odds: float
    bookmaker: str | None
    margin: float | None


_QuoteKey = tuple[str, Market, str]


@dataclass(frozen=True)
class _Candidate:
    """One match the slips may use: its fixture, section, family, (adjusted) probabilities and rates."""

    fixture: Fixture
    section: str
    family: str
    probabilities: dict[Market, dict[str, float]]
    home_rate: float | None
    away_rate: float | None

    @property
    def kick(self) -> datetime:
        """The instant used to order ties: the kick-off, else midnight of the day in Europe/Rome."""
        if self.fixture.kickoff is not None:
            return self.fixture.kickoff
        return rome_midnight(self.fixture.date).astimezone(UTC)

    def probability(self, market: Market, selection: str) -> float | None:
        return self.probabilities.get(market, {}).get(selection)


@dataclass(frozen=True)
class _Option:
    """One selection of one candidate with its probability and best quote."""

    candidate: _Candidate
    market: Market
    selection: str
    p: float
    quote: _Quote | None
    ev: float | None = field(init=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "ev", None if self.quote is None else self.p * self.quote.odds - 1.0)

    @property
    def match_id(self) -> str:
        return self.candidate.fixture.match_id


@dataclass(frozen=True)
class _Options:
    """Validated builder options (shared by :func:`build_slips`, :func:`slips_from_frame` and :func:`make_slip_builder`)."""

    n_x: int
    n_gg: int
    n_mixed: int
    mixed: tuple[tuple[Market, str], ...]
    min_probability: float
    only_quoted: bool
    sections: frozenset[str] | None


def _as_market(value: Market | str) -> Market:
    """The :class:`Market` of a member, its value (``'GG/NG'``) or its name (``'BTTS'``)."""
    if isinstance(value, Market):
        return value
    text = str(value).strip()
    try:
        return Market(text)
    except ValueError:
        pass
    try:
        return Market[text.upper()]
    except KeyError:
        raise ValueError(f"unknown market {value!r} (expected one of {[m.value for m in Market]})") from None


def _count(value: int, name: str) -> int:
    if isinstance(value, bool):
        raise TypeError(f"{name} must be an integer, got {value!r}")
    try:
        number = operator.index(value)
    except TypeError:
        raise TypeError(f"{name} must be an integer, got {value!r}") from None
    if number < 0:
        raise ValueError(f"{name} must be >= 0, got {number}")
    return number


def _normalise_section(value: object) -> str:
    return " ".join(str(value).split()).casefold()


def _options(
    *,
    n_x: int,
    n_gg: int,
    n_mixed: int,
    mixed_markets: Iterable[tuple[Market | str, str]],
    min_probability: float,
    only_quoted: bool,
    sections: Iterable[str] | None,
) -> _Options:
    n_mixed_value = _count(n_mixed, "n_mixed")
    if not MIXED_MIN_LEGS <= n_mixed_value <= MIXED_MAX_LEGS:
        raise ValueError(f"n_mixed must be between {MIXED_MIN_LEGS} and {MIXED_MAX_LEGS}, got {n_mixed_value}")
    mixed: list[tuple[Market, str]] = []
    for item in mixed_markets:
        try:
            raw_market, selection = item
        except (TypeError, ValueError):
            raise ValueError(f"mixed_markets items must be (market, selection) pairs, got {item!r}") from None
        market = _as_market(raw_market)
        selection = str(selection)
        if selection not in SELECTIONS[market]:
            raise ValueError(f"{selection!r} is not a selection of {market.value} (expected one of {SELECTIONS[market]})")
        if (market, selection) not in mixed:
            mixed.append((market, selection))
    if not mixed:
        raise ValueError("mixed_markets must name at least one (market, selection) pair")
    threshold = float(min_probability)
    if not math.isfinite(threshold) or not 0.0 <= threshold <= 1.0:
        raise ValueError(f"min_probability must be in [0, 1], got {min_probability!r}")
    if sections is not None and isinstance(sections, str):
        sections = (sections,)
    wanted = None if sections is None else frozenset(_normalise_section(s) for s in sections)
    return _Options(
        n_x=_count(n_x, "n_x"),
        n_gg=_count(n_gg, "n_gg"),
        n_mixed=n_mixed_value,
        mixed=tuple(mixed),
        min_probability=threshold,
        only_quoted=bool(only_quoted),
        sections=wanted,
    )


def _section_of(girone: Girone) -> str:
    """Dashboard section of a competition (its ``section``, else its family)."""
    try:
        value = getattr(girone, "section", "")
    except LookupError:  # pragma: no cover - only a hand-built enum lacks the competition records
        value = ""
    return str(value or girone.family)


def _probability_value(value: Any, *, where: str) -> float:
    p = float(value)
    if not math.isfinite(p) or p < -_PROBABILITY_TOLERANCE or p > 1.0 + _PROBABILITY_TOLERANCE:
        raise ValueError(f"{where}: probability must be in [0, 1], got {value!r}")
    return min(max(p, 0.0), 1.0)


def _mapping_probabilities(raw: Mapping[Any, Mapping[str, Any]], *, where: str) -> dict[Market, dict[str, float]]:
    """``{Market: {selection: p}}`` from a mapping keyed by markets or their values; unknown markets/selections are dropped."""
    out: dict[Market, dict[str, float]] = {}
    for key, selections in raw.items():
        try:
            market = _as_market(key)
        except ValueError:
            continue
        values = {str(s): _probability_value(p, where=f"{where} {market.value} {s}") for s, p in selections.items() if str(s) in SELECTIONS[market]}
        if values:
            out[market] = values
    return out


def _adjusted(probabilities: dict[Market, dict[str, float]], family: str, adjust: ProbabilityAdjuster | None, match_id: str) -> dict[Market, dict[str, float]]:
    if adjust is None:
        return probabilities
    corrected = adjust({m: dict(s) for m, s in probabilities.items()}, family=family)
    return _mapping_probabilities(corrected, where=f"adjusted probabilities of {match_id}")


def _candidates_from_predictions(predictions: Iterable[MatchPrediction], adjust: ProbabilityAdjuster | None, matchday: Matchday | None) -> list[_Candidate]:
    out: list[_Candidate] = []
    for prediction in predictions:
        fixture = prediction.fixture
        if matchday is not None and not matchday.contains(fixture):
            continue
        girone = Girone.parse(fixture.girone)
        family = girone.family
        probabilities = _mapping_probabilities(prediction.probabilities, where=f"prediction {fixture.match_id}")
        out.append(
            _Candidate(
                fixture=fixture,
                section=_section_of(girone),
                family=family,
                probabilities=_adjusted(probabilities, family, adjust, fixture.match_id),
                home_rate=float(prediction.home_rate),
                away_rate=float(prediction.away_rate),
            )
        )
    return out


def _is_missing(value: Any) -> bool:
    if value is None:
        return True
    try:
        return bool(pd.isna(value))
    except (TypeError, ValueError):
        return False


def _optional_float(value: Any) -> float | None:
    if _is_missing(value):
        return None
    number = float(value)
    return number if math.isfinite(number) else None


def _optional_text(value: Any) -> str | None:
    if _is_missing(value):
        return None
    text = str(value).strip()
    return text or None


def _frame_status(value: Any) -> FixtureStatus:
    """``'postponed'`` for a postponed row, ``'scheduled'`` for any other (or missing) status."""
    text = _optional_text(value)
    return "postponed" if text is not None and text.casefold() == "postponed" else "scheduled"


def _required_columns(options: _Options) -> list[str]:
    needed = list(_FRAME_REQUIRED)
    if options.n_x > 0:
        needed.append(f"p_{FRAME_SELECTION_SUFFIX['X']}")
    if options.n_gg > 0:
        needed.append(f"p_{FRAME_SELECTION_SUFFIX['GG']}")
    needed.extend(f"p_{FRAME_SELECTION_SUFFIX[s]}" for _m, s in options.mixed)
    return list(dict.fromkeys(needed))


def _candidates_from_frame(frame: pd.DataFrame, adjust: ProbabilityAdjuster | None) -> tuple[list[_Candidate], dict[_QuoteKey, _Quote]]:
    """Candidates and the price columns (``odds_<suffix>``, ``bookmaker_<suffix>``, ``overround_<market>``) of a frame."""
    columns = set(frame.columns)
    candidates: list[_Candidate] = []
    prices: dict[_QuoteKey, _Quote] = {}
    for row in frame.to_dict(orient="records"):
        match_id = str(row["match_id"])
        try:
            girone = Girone.parse(str(row["girone"]))
        except ValueError:
            log.warning("slip frame row skipped: unknown competition", extra={"match_id": match_id, "girone": str(row["girone"])})
            continue
        round_value = _optional_float(row.get("round"))
        fixture = Fixture(
            match_id=match_id,
            date=row["date"],
            round=int(round_value) if round_value is not None else 0,
            girone=girone,
            home_team=str(row["home_team"]),
            away_team=str(row["away_team"]),
            kickoff=None if _is_missing(row.get("kickoff")) else row.get("kickoff"),
            status=_frame_status(row.get("status")),
        )
        probabilities: dict[Market, dict[str, float]] = {}
        for market, selections in SELECTIONS.items():
            values: dict[str, float] = {}
            for selection in selections:
                column = f"p_{FRAME_SELECTION_SUFFIX[selection]}"
                if column not in columns:
                    continue
                value = _optional_float(row[column])
                if value is not None:
                    values[selection] = _probability_value(value, where=f"frame row {match_id} {column}")
            if values:
                probabilities[market] = values
            margin = _optional_float(row.get(f"overround_{FRAME_MARKET_SUFFIX[market]}"))
            for selection in selections:
                odds = _optional_float(row.get(f"odds_{FRAME_SELECTION_SUFFIX[selection]}"))
                if odds is None or odds <= 1.0:
                    continue
                bookmaker = _optional_text(row.get(f"bookmaker_{FRAME_SELECTION_SUFFIX[selection]}"))
                _offer(prices, (match_id, market, selection), _Quote(odds, bookmaker, margin))
        family = _optional_text(row.get("family")) or girone.family
        section = _optional_text(row.get("section")) or _section_of(girone)
        candidates.append(
            _Candidate(
                fixture=fixture,
                section=section,
                family=family,
                probabilities=_adjusted(probabilities, family, adjust, match_id),
                home_rate=_optional_float(row.get("lam")),
                away_rate=_optional_float(row.get("mu")),
            )
        )
    return candidates, prices


def _offer(book: dict[_QuoteKey, _Quote], key: _QuoteKey, quote: _Quote) -> None:
    """Keep the highest price per selection; equal prices go to the bookmaker first in alphabetical order."""
    current = book.get(key)
    if current is None:
        book[key] = quote
        return
    same_price_first_name = quote.odds == current.odds and (quote.bookmaker is None, quote.bookmaker or "") < (
        current.bookmaker is None,
        current.bookmaker or "",
    )
    if quote.odds > current.odds or same_price_first_name:
        book[key] = quote


def _best_quotes(opportunities: Iterable[ValueOpportunity], quotes: Iterable[OddsQuote], base: Mapping[_QuoteKey, _Quote] | None = None) -> dict[_QuoteKey, _Quote]:
    """Best (highest) price per ``(match_id, market, selection)`` across opportunities, raw quotes and ``base``."""
    book: dict[_QuoteKey, _Quote] = dict(base or {})
    for opportunity in opportunities:
        odds = float(opportunity.book_odds)
        if not math.isfinite(odds) or odds <= 1.0:
            continue
        raw, devigged = float(opportunity.p_implied_raw), float(opportunity.p_implied_devig)
        margin = raw / devigged - 1.0 if raw > 0.0 and devigged > 0.0 and math.isfinite(raw / devigged) else None
        _offer(book, (str(opportunity.match_id), Market(opportunity.market), str(opportunity.selection)), _Quote(odds, opportunity.bookmaker, margin))
    for quote in quotes:
        prices = {s: float(o) for s, o in quote.odds.items()}
        if not prices or any(not math.isfinite(o) or o <= 1.0 for o in prices.values()):
            continue
        margin = sum(1.0 / o for o in prices.values()) - 1.0
        for selection, odds in prices.items():
            _offer(book, (str(quote.match_id), Market(quote.market), str(selection)), _Quote(odds, quote.bookmaker, margin))
    return book


# =============================================================================
# Ranking
# =============================================================================
def _rank_key(option: _Option) -> tuple[float, int, float, datetime, str]:
    """Probability descending, then EV descending (quoted first), then earlier kick-off, then ``match_id``."""
    ev = option.ev
    return (-option.p, 0 if ev is not None else 1, -(ev if ev is not None else 0.0), option.candidate.kick, option.match_id)


def _option(candidate: _Candidate, market: Market, selection: str, prices: Mapping[_QuoteKey, _Quote], options: _Options) -> _Option | None:
    """The option of one selection, ``None`` when it is missing, below the threshold or unquoted under ``only_quoted``."""
    p = candidate.probability(market, selection)
    if p is None or p < options.min_probability:
        return None
    quote = prices.get((candidate.fixture.match_id, market, selection))
    if options.only_quoted and quote is None:
        return None
    return _Option(candidate, market, selection, p, quote)


def _hints(candidate: _Candidate, market: Market, selection: str) -> list[str]:
    """The Italian hints of the picks (:mod:`src.engine.picks`) for a draw or a GG leg."""
    lam, mu = candidate.home_rate, candidate.away_rate
    if lam is None or mu is None:
        return []
    hints: list[str] = []
    if market is Market.MATCH_ODDS and selection == "X":
        if abs(lam - mu) <= _DRAW_CLOSE_RATES_GAP:
            hints.append(DRAW_NOTE_CLOSE_RATES)
        if lam + mu <= _DRAW_FEW_GOALS_TOTAL:
            hints.append(DRAW_NOTE_FEW_GOALS)
    elif market is Market.BTTS and selection == "GG":
        if lam + mu >= _BTTS_HIGH_XG_TOTAL:
            hints.append(BTTS_NOTE_HIGH_XG)
        if min(lam, mu) >= _BTTS_BOTH_ATTACKS_MIN:
            hints.append(BTTS_NOTE_BOTH_ATTACKS)
    return hints


def _leg(option: _Option, extra_note: str = "") -> SlipLeg:
    candidate = option.candidate
    fixture = candidate.fixture
    quote = option.quote
    note = ", ".join(_hints(candidate, option.market, option.selection))
    if extra_note:
        note = f"{note}; {extra_note}" if note else extra_note
    return SlipLeg(
        match_id=fixture.match_id,
        date=fixture.date,
        kickoff=fixture.kickoff,
        competition=Girone.parse(fixture.girone),
        section=candidate.section,
        home_team=fixture.home_team,
        away_team=fixture.away_team,
        market=option.market,
        selection=option.selection,
        p_model=option.p,
        fair_odds=1.0 / option.p if option.p > 0.0 else float("inf"),
        book_odds=None if quote is None else quote.odds,
        bookmaker=None if quote is None else quote.bookmaker,
        ev=option.ev,
        book_margin=None if quote is None else quote.margin,
        note=note,
    )


def _single_sign_slip(kind: SlipKind, title: str, market: Market, selection: str, n: int, eligible: Sequence[_Candidate], prices: Mapping[_QuoteKey, _Quote], options: _Options) -> Slip:
    ranked = sorted((o for c in eligible if (o := _option(c, market, selection, prices, options)) is not None), key=_rank_key)
    return Slip(kind=kind, title=title, legs=tuple(_leg(o) for o in ranked[:n]), n_requested=n)


def _mixed_title(mixed: Sequence[tuple[Market, str]]) -> str:
    return f"{_TITLE_MIXED} · {' / '.join(s for _m, s in mixed)}"


def _runner_up_note(chosen: _Option, alternative: _Option) -> str:
    """The note of a mixed leg naming its second-best sign.

    The alternative is shown with the one-decimal precision of the leg's
    probability column, plus its distance in percentage points from the chosen
    sign: a rounder format (``GG 63.9`` next to ``alternativa: Over 2.5 64%``)
    made the alternative look more likely than the sign the model picked.
    """
    gap = max(chosen.p - alternative.p, 0.0) * 100.0
    if gap == 0.0:
        distance = "stessa probabilità"
    elif gap < 0.05:
        distance = "meno di 0.1 punti in meno"
    else:
        distance = f"{gap:.1f} punti in meno"
    return f"alternativa: {alternative.selection} {alternative.p * 100:.1f}% ({distance})"


def _mixed_slip(eligible: Sequence[_Candidate], prices: Mapping[_QuoteKey, _Quote], options: _Options) -> Slip:
    best: list[tuple[_Option, str]] = []
    for candidate in eligible:
        signs = [
            (index, o)
            for index, (market, selection) in enumerate(options.mixed)
            if (o := _option(candidate, market, selection, prices, options)) is not None
        ]
        if not signs:
            continue
        # best sign by model probability; equal probabilities go to the sign listed first in ``mixed_markets``
        signs.sort(key=lambda item: (-item[1].p, item[0]))
        runner_up = _runner_up_note(signs[0][1], signs[1][1]) if len(signs) > 1 else ""
        best.append((signs[0][1], runner_up))
    best.sort(key=lambda item: _rank_key(item[0]))
    legs = tuple(_leg(o, note) for o, note in best[: options.n_mixed])
    return Slip(kind="MISTA", title=_mixed_title(options.mixed), legs=legs, n_requested=options.n_mixed)


def _build(candidates: Sequence[_Candidate], prices: Mapping[_QuoteKey, _Quote], *, now: datetime | None, options: _Options, source: str) -> list[Slip]:
    """The three slips over ``candidates`` (the shared core of :func:`build_slips` and :func:`slips_from_frame`)."""
    seen: set[str] = set()
    eligible: list[_Candidate] = []
    n_duplicates = n_postponed = n_kicked_off = n_other_sections = 0
    for candidate in candidates:
        fixture = candidate.fixture
        if fixture.match_id in seen:
            n_duplicates += 1
            continue
        seen.add(fixture.match_id)
        if fixture.is_postponed:
            n_postponed += 1
            continue
        if kicked_off(fixture, now):
            n_kicked_off += 1
            continue
        if options.sections is not None and _normalise_section(candidate.section) not in options.sections:
            n_other_sections += 1
            continue
        eligible.append(candidate)
    if n_duplicates:
        log.warning("slips: duplicated matches ignored (first occurrence kept)", extra={"source": source, "n_duplicates": n_duplicates})
    slips = [
        _single_sign_slip("X", _TITLE_X, Market.MATCH_ODDS, "X", options.n_x, eligible, prices, options),
        _single_sign_slip("GG", _TITLE_GG, Market.BTTS, "GG", options.n_gg, eligible, prices, options),
        _mixed_slip(eligible, prices, options),
    ]
    log.info(
        "slips built",
        extra={
            "source": source,
            "n_candidates": len(candidates),
            "n_eligible": len(eligible),
            "n_duplicates": n_duplicates,
            "n_postponed": n_postponed,
            "n_kicked_off": n_kicked_off,
            "n_other_sections": n_other_sections,
            "sections": None if options.sections is None else sorted(options.sections),
            "min_probability": options.min_probability,
            "only_quoted": options.only_quoted,
            "legs": {s.kind: s.n_legs for s in slips},
            "quoted": {s.kind: s.is_quoted for s in slips},
        },
    )
    return slips


# =============================================================================
# Public builders
# =============================================================================
def build_slips(
    predictions: Iterable[MatchPrediction],
    *,
    now: datetime | None,
    opportunities: Iterable[ValueOpportunity] = (),
    quotes: Iterable[OddsQuote] = (),
    n_x: int = DEFAULT_N_X,
    n_gg: int = DEFAULT_N_GG,
    n_mixed: int = DEFAULT_N_MIXED,
    mixed_markets: Iterable[tuple[Market | str, str]] = DEFAULT_MIXED_MARKETS,
    min_probability: float = 0.0,
    only_quoted: bool = False,
    sections: Iterable[str] | None = None,
    adjust: ProbabilityAdjuster | None = None,
    matchday: Matchday | None = None,
) -> list[Slip]:
    """The three slips of a matchday: ``[X, GG, MISTA]`` (see the module docstring for the rules).

    ``predictions`` are the matchday's predictions (``matchday`` restricts them
    to its matches still to play when given). ``now`` is the reference instant
    (aware; naive = UTC): a fixture that has kicked off at ``now`` -- or, without
    a kick-off time, whose day in Europe/Rome is over -- is excluded, and so is
    a postponed fixture; ``None`` skips the time check (postponed fixtures are
    still excluded). A ``match_id`` seen twice keeps its first prediction.
    ``opportunities`` (value engine rows) and ``quotes`` (raw bookmaker quotes)
    give the book prices, the highest across bookmakers per selection.
    ``n_x``/``n_gg`` (>= 0) and ``n_mixed`` (:data:`MIXED_MIN_LEGS` to
    :data:`MIXED_MAX_LEGS`) are the slip lengths, ``mixed_markets`` the
    ``(market, selection)`` signs of the mixed slip, ``min_probability`` the
    minimum leg probability, ``only_quoted`` keeps only legs with a book price,
    ``sections`` the dashboard sections to draw from (case-insensitive labels,
    ``None`` = every section, empty = none) and ``adjust`` the probability
    correction (:class:`ProbabilityAdjuster`, e.g. the recalibration).
    A slip is shorter than asked when fewer matches qualify (empty when none).
    """
    options = _options(
        n_x=n_x,
        n_gg=n_gg,
        n_mixed=n_mixed,
        mixed_markets=mixed_markets,
        min_probability=min_probability,
        only_quoted=only_quoted,
        sections=sections,
    )
    candidates = _candidates_from_predictions(predictions, adjust, matchday)
    prices = _best_quotes(opportunities, quotes)
    return _build(candidates, prices, now=now, options=options, source="predictions")


def slips_from_frame(
    frame: pd.DataFrame,
    *,
    now: datetime | None = None,
    opportunities: Iterable[ValueOpportunity] = (),
    quotes: Iterable[OddsQuote] = (),
    n_x: int = DEFAULT_N_X,
    n_gg: int = DEFAULT_N_GG,
    n_mixed: int = DEFAULT_N_MIXED,
    mixed_markets: Iterable[tuple[Market | str, str]] = DEFAULT_MIXED_MARKETS,
    min_probability: float = 0.0,
    only_quoted: bool = False,
    sections: Iterable[str] | None = None,
    adjust: ProbabilityAdjuster | None = None,
) -> list[Slip]:
    """The three slips of :func:`build_slips` from a probability frame (one row per match of one matchday).

    Required columns: ``match_id``, ``date``, ``girone``, ``home_team``,
    ``away_team`` and the ``p_<suffix>`` columns of the requested selections
    (:data:`FRAME_SELECTION_SUFFIX`: ``p_X``, ``p_GG`` and ``p_1``, ``p_2``,
    ``p_O25`` for the default mixed slip); a missing one raises
    :class:`ValueError`. Optional: ``kickoff`` and ``status`` (kick-off and
    postponement checks against ``now``), ``round``, ``family`` (passed to
    ``adjust``), ``section`` (else the competition's), ``lam``/``mu`` (hints),
    the other ``p_*`` columns (passed to ``adjust``) and the prices
    ``odds_<suffix>`` with ``bookmaker_<suffix>`` and ``overround_<market>``
    (:data:`FRAME_MARKET_SUFFIX`), merged with ``opportunities``/``quotes``.
    Rows of an unknown competition are skipped with a warning.

    The ranking is the one of :func:`build_slips`, so the backtest replays past
    matchdays with the slips the dashboard shows: with the frame alone (``now``
    = ``None``: every row is pre-match) this is the ``SlipBuilder`` callable of
    :func:`src.engine.backtest.slips_track_record`.
    """
    options = _options(
        n_x=n_x,
        n_gg=n_gg,
        n_mixed=n_mixed,
        mixed_markets=mixed_markets,
        min_probability=min_probability,
        only_quoted=only_quoted,
        sections=sections,
    )
    if frame.empty:
        log.info("slips built", extra={"source": "frame", "n_candidates": 0})
        return _build([], {}, now=now, options=options, source="frame")
    missing = [c for c in _required_columns(options) if c not in frame.columns]
    if missing:
        raise ValueError(f"slip frame is missing columns {missing}")
    candidates, frame_prices = _candidates_from_frame(frame, adjust)
    prices = _best_quotes(opportunities, quotes, frame_prices)
    return _build(candidates, prices, now=now, options=options, source="frame")


def make_slip_builder(
    *,
    now: datetime | None = None,
    opportunities: Iterable[ValueOpportunity] = (),
    quotes: Iterable[OddsQuote] = (),
    n_x: int = DEFAULT_N_X,
    n_gg: int = DEFAULT_N_GG,
    n_mixed: int = DEFAULT_N_MIXED,
    mixed_markets: Iterable[tuple[Market | str, str]] = DEFAULT_MIXED_MARKETS,
    min_probability: float = 0.0,
    only_quoted: bool = False,
    sections: Iterable[str] | None = None,
    adjust: ProbabilityAdjuster | None = None,
) -> Callable[[pd.DataFrame], list[Slip]]:
    """A ``(frame) -> list[Slip]`` callable (the backtest's ``SlipBuilder``) bound to these options.

    The options are validated now (a bad length raises here, not at the first
    matchday); the iterables are materialised once, so a generator can be
    passed. Each call is :func:`slips_from_frame` with these options.
    """
    bound_opportunities = tuple(opportunities)
    bound_quotes = tuple(quotes)
    bound_mixed = tuple(mixed_markets)
    bound_sections = None if sections is None else ((sections,) if isinstance(sections, str) else tuple(sections))
    _options(
        n_x=n_x,
        n_gg=n_gg,
        n_mixed=n_mixed,
        mixed_markets=bound_mixed,
        min_probability=min_probability,
        only_quoted=only_quoted,
        sections=bound_sections,
    )

    def builder(frame: pd.DataFrame) -> list[Slip]:
        """Slips of one matchday's probability frame (see :func:`slips_from_frame`)."""
        return slips_from_frame(
            frame,
            now=now,
            opportunities=bound_opportunities,
            quotes=bound_quotes,
            n_x=n_x,
            n_gg=n_gg,
            n_mixed=n_mixed,
            mixed_markets=bound_mixed,
            min_probability=min_probability,
            only_quoted=only_quoted,
            sections=bound_sections,
            adjust=adjust,
        )

    return builder


def predictions_frame(predictions: Iterable[MatchPrediction]) -> pd.DataFrame:
    """Probability frame of predictions, columns :data:`PREDICTIONS_FRAME_COLUMNS` (the input of :func:`slips_from_frame`).

    One row per prediction, in input order: fixture fields (``date`` as
    ``datetime64[ns]``, ``kickoff`` as UTC timestamps, ``NaT`` without a time),
    ``family``/``section`` of the competition, the rates and every model
    probability as ``p_<suffix>`` (``NaN`` for a market the prediction lacks).
    """
    rows: list[dict[str, Any]] = []
    for prediction in predictions:
        fixture = prediction.fixture
        girone = Girone.parse(fixture.girone)
        row: dict[str, Any] = {
            "match_id": fixture.match_id,
            "date": pd.Timestamp(fixture.date),
            "kickoff": pd.Timestamp(fixture.kickoff) if fixture.kickoff is not None else pd.NaT,
            "status": fixture.status,
            "round": int(fixture.round),
            "girone": girone.value,
            "family": girone.family,
            "section": _section_of(girone),
            "home_team": fixture.home_team,
            "away_team": fixture.away_team,
            "lam": float(prediction.home_rate),
            "mu": float(prediction.away_rate),
            "rho": float(prediction.rho),
        }
        probabilities = _mapping_probabilities(prediction.probabilities, where=f"prediction {fixture.match_id}")
        for market, selections in SELECTIONS.items():
            for selection in selections:
                value = probabilities.get(market, {}).get(selection)
                row[f"p_{FRAME_SELECTION_SUFFIX[selection]}"] = float("nan") if value is None else value
        rows.append(row)
    frame = pd.DataFrame(rows, columns=list(PREDICTIONS_FRAME_COLUMNS))
    frame["date"] = pd.to_datetime(frame["date"]).astype("datetime64[ns]")
    frame["kickoff"] = pd.to_datetime(frame["kickoff"], utc=True).astype("datetime64[ns, UTC]")
    for column in ("match_id", "status", "girone", "family", "section", "home_team", "away_team"):
        frame[column] = frame[column].astype("string")
    frame["round"] = frame["round"].astype("int64")
    for column in PREDICTIONS_FRAME_COLUMNS[PREDICTIONS_FRAME_COLUMNS.index("lam"):]:
        frame[column] = frame[column].astype("float64")
    return frame.reset_index(drop=True)


# =============================================================================
# Stake and display
# =============================================================================
def suggested_stake(
    slip: Slip,
    *,
    bankroll_units: float,
    fraction: float = DEFAULT_STAKE_FRACTION,
    cap_units: float = DEFAULT_STAKE_CAP_UNITS,
    floor_units: float = 0.0,
    round_to: float = DEFAULT_STAKE_ROUND_TO,
) -> float:
    """Suggested stake of a slip in bankroll units: fractional Kelly on the whole slip, capped.

    ``f* = (p * b - (1 - p)) / b`` with ``p`` the slip probability and ``b =
    book_odds - 1``; the stake is ``f* * fraction * bankroll_units``, capped at
    ``cap_units``, rounded half up to ``round_to`` (see
    :func:`src.engine.value.stake_units`) and reported as ``0`` when below
    ``floor_units``. A slip without book odds or with ``EV <= 0`` gets ``0``
    (*non consigliata*, :func:`format_stake`). Invalid parameters raise
    :class:`ValueError` (``0 < fraction <= 1``, ``bankroll_units > 0``,
    ``cap_units > 0``, ``0 <= floor_units <= cap_units``, ``round_to > 0``).
    """
    config = KellyConfig(
        fraction=fraction,
        bankroll_units=bankroll_units,
        max_stake_units=cap_units,
        min_stake_units=floor_units,
        round_to=round_to,
    )
    odds, ev = slip.book_odds, slip.ev
    if odds is None or ev is None or ev <= 0.0:
        return 0.0
    return stake_units(slip.probability, odds, config)


def format_stake(units: float) -> str:
    """``'0.25 unità'``, or :data:`STAKE_NOT_ADVISED` for a zero (or invalid) stake."""
    value = float(units)
    if not math.isfinite(value) or value <= 0.0:
        return STAKE_NOT_ADVISED
    return f"{value:.2f} unità"


def slips_frame(slips: Iterable[Slip]) -> pd.DataFrame:
    """Italian table of the legs, one row per leg, columns exactly :data:`SLIPS_COLUMNS` (an empty frame keeps them).

    ``Schedina`` is the slip title, ``#`` the leg position, ``Data e ora`` the
    kick-off in Europe/Rome (``'dom 08/11 14:30'``), probabilities in percent
    (1 decimal), odds with 2 decimals; ``Quota Book`` is ``NaN`` and
    ``Bookmaker`` ``<NA>`` for an unquoted leg. An empty slip has no rows.
    """
    rows = [
        {
            "Schedina": slip.title,
            "#": position,
            "Competizione": leg.competition.short,
            "Data e ora": leg.when,
            "Partita": leg.label,
            "Esito": leg.selection,
            "P. Modello (%)": round(leg.p_model * 100.0, 1),
            "Quota Fair": round(leg.fair_odds, 2),
            "Quota Book": float("nan") if leg.book_odds is None else round(leg.book_odds, 2),
            "Bookmaker": pd.NA if leg.bookmaker is None else leg.bookmaker,
            "Nota": leg.note,
        }
        for slip in slips
        for position, leg in enumerate(slip.legs, start=1)
    ]
    return _typed_frame(rows, SLIPS_COLUMNS, _SLIPS_DTYPES)


def slips_summary(
    slips: Iterable[Slip],
    *,
    bankroll_units: float = 100.0,
    fraction: float = DEFAULT_STAKE_FRACTION,
    cap_units: float = DEFAULT_STAKE_CAP_UNITS,
    floor_units: float = 0.0,
) -> pd.DataFrame:
    """One row per slip, columns exactly :data:`SLIPS_SUMMARY_COLUMNS`.

    ``Probabilità (%)`` and ``EV (%)`` with 2 decimals, odds with 2 decimals
    (``Quota book``/``EV (%)`` are ``NaN`` unless every leg is quoted),
    ``Partite vincenti attese`` with 2 decimals, ``Stake suggerito`` the text of
    :func:`format_stake` for :func:`suggested_stake` with these parameters.
    """
    rows: list[dict[str, Any]] = []
    for slip in slips:
        odds, ev = slip.book_odds, slip.ev
        stake = suggested_stake(slip, bankroll_units=bankroll_units, fraction=fraction, cap_units=cap_units, floor_units=floor_units)
        rows.append(
            {
                "Schedina": slip.title,
                "Partite": slip.n_legs,
                "Probabilità (%)": round(slip.probability * 100.0, 2),
                "Quota fair": round(slip.fair_odds, 2),
                "Quota book": float("nan") if odds is None else round(odds, 2),
                "EV (%)": float("nan") if ev is None else round(ev * 100.0, 2),
                "Partite vincenti attese": round(slip.expected_winning_legs, 2),
                "Stake suggerito": format_stake(stake),
            }
        )
    return _typed_frame(rows, SLIPS_SUMMARY_COLUMNS, _SUMMARY_DTYPES)


def _typed_frame(rows: list[dict[str, Any]], columns: Sequence[str], dtypes: Mapping[str, str]) -> pd.DataFrame:
    if not rows:
        empty = pd.DataFrame({c: pd.Series(dtype=dtypes[c]) for c in columns})
        return empty[list(columns)]
    frame = pd.DataFrame(rows, columns=list(columns))
    for column in columns:
        frame[column] = frame[column].astype(dtypes[column])
    return frame.reset_index(drop=True)
