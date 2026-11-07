"""Matchday and results tables: the whole round (or the whole season so far) readable at a glance.

The dashboard used to show one fixture at a time, picked from a dropdown; this
module builds the two ``pandas`` frames with Italian labels that replace that
workflow and that ``main.py`` prints and exports as well:

* :func:`matchday_table` -- **one row per fixture** of a matchday: the strength
  parameters of the two teams, their recent form (points, goals, xG over the
  last *n* played matches), the model's expected goals, the most likely score,
  the market probabilities and fair odds, the best bookmaker price per
  selection, the best EV among the quoted selections and the union of the value
  engine's flags (``REGRESSION_TRAP`` / ``TRAP:<team>`` included);
* :func:`results_table` -- **one row per played match**: score, outcome, shots,
  shots on target, corners and xG of both teams and, when an engine is given,
  the fitted rates, the probability the model assigned to the outcome that
  happened (``P esito``) and a short tag (``sorpresa`` / ``atteso``).

Both are pure (no ``streamlit``), depend only on the shared contracts and run
on the scale of the real perimeter (three seasons, ~3,500 matches, a
season-long fixture list of ~150 fixtures) in well under a second: the form of
every team is computed **once** on a per-team long frame (:func:`team_matches`,
one row per team and played match) with a single ``groupby``, never by filtering
the match frame fixture by fixture.

Form
----
:func:`team_form` / :func:`form_table` summarise a team's last ``n`` played
matches (home and away, ``date < as_of``, optionally one season): points (3 for
a win, 1 for a draw), goals for and against, xG for and against (``NaN`` as soon
as one xG value of the window is missing), clean sheets and the results string
``'V-P-N-V-V'`` -- most recent first, ``V`` vittoria, ``N`` pareggio (nullo),
``P`` sconfitta.
"""
from __future__ import annotations

import math
import time
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import date, datetime
from typing import Any, Final

import numpy as np
import pandas as pd
from scipy.special import gammaln, xlogy

from src.common.logging_config import get_logger
from src.common.schemas import Girone, Market, MatchPrediction, season_of, validate_matches_df
from src.engine.model import DixonColesEngine, FittedParams, tau
from src.engine.picks import Matchday, in_gironi
from src.engine.value import (
    FLAG_LONGSHOT,
    FLAG_REGRESSION_TRAP,
    FLAG_SUSPICIOUS_EDGE,
    FLAG_VALUE,
    TRAP_FLAG_PREFIX,
    RegressionTrap,
    ValueOpportunity,
)

__all__ = [
    "DEFAULT_FORM_N",
    "DEFAULT_XG_N",
    "EXPECTED_ABOVE",
    "EXPECTED_TAG",
    "FLAG_SEPARATOR",
    "MATCHDAY_COLUMNS",
    "MISSING_TEXT",
    "P_OUTCOME_DECIMALS",
    "RESULTS_COLUMNS",
    "RESULTS_MODEL_COLUMNS",
    "RESULT_DRAW",
    "RESULT_LOSS",
    "RESULT_WIN",
    "SURPRISE_BELOW",
    "SURPRISE_TAG",
    "TEAM_MATCHES_COLUMNS",
    "TeamForm",
    "form_table",
    "matchday_columns",
    "matchday_table",
    "outcome_probabilities",
    "p_outcome_percent",
    "results_table",
    "surprise_tag",
    "team_form",
    "team_matches",
]

log = get_logger(__name__)

#: Default form window (points, goals, results string) and xG window of :func:`matchday_table`.
DEFAULT_FORM_N: Final[int] = 5
DEFAULT_XG_N: Final[int] = 3
#: Letters of the results string: V vittoria, N pareggio (nullo), P sconfitta (the wording of the legend and the UI).
RESULT_WIN: Final[str] = "V"
RESULT_DRAW: Final[str] = "N"
RESULT_LOSS: Final[str] = "P"
#: ``Sorpresa`` tags of :func:`results_table`: ``P esito`` below :data:`SURPRISE_BELOW` is a surprise,
#: above :data:`EXPECTED_ABOVE` the expected outcome, anything else is left blank. The thresholds are
#: applied to the **displayed** percentage (``P esito (%)``, 1 decimal, see :func:`surprise_tag`), so the tag
#: never contradicts the number next to it.
SURPRISE_BELOW: Final[float] = 0.20
EXPECTED_ABOVE: Final[float] = 0.50
#: Decimals of the ``P esito (%)`` column (and of the percentage :func:`surprise_tag` reads).
P_OUTCOME_DECIMALS: Final[int] = 1
SURPRISE_TAG: Final[str] = "sorpresa"
EXPECTED_TAG: Final[str] = "atteso"
#: Placeholder of a missing statistic in the text columns (``'12/-'``, ``'-'``).
MISSING_TEXT: Final[str] = "-"
#: Separator of the flags in the ``Flag`` column (same as the value panel).
FLAG_SEPARATOR: Final[str] = " | "
#: Points per result.
_POINTS: Final[dict[str, int]] = {RESULT_WIN: 3, RESULT_DRAW: 1, RESULT_LOSS: 0}
#: Canonical order of the known flags in the ``Flag`` column; unknown flags follow, sorted.
_FLAG_ORDER: Final[tuple[str, ...]] = (FLAG_VALUE, FLAG_REGRESSION_TRAP, FLAG_LONGSHOT, FLAG_SUSPICIOUS_EDGE)

#: Columns of :func:`team_matches` (one row per team and played match), in order.
TEAM_MATCHES_COLUMNS: Final[tuple[str, ...]] = (
    "team",
    "match_id",
    "date",
    "girone",
    "opponent",
    "home",
    "goals_for",
    "goals_against",
    "xg_for",
    "xg_against",
    "result",
    "points",
    "clean_sheet",
)

#: Columns of :func:`results_table` without an engine, in order.
RESULTS_COLUMNS: Final[tuple[str, ...]] = (
    "Competizione",
    "Data",
    "Giornata",
    "Casa",
    "Ospiti",
    "Risultato",
    "Esito",
    "Tiri C/O",
    "In porta C/O",
    "Angoli C/O",
    "xG Casa",
    "xG Ospiti",
    "match_id",
)
#: Columns of :func:`results_table` with an engine (the model columns sit before ``match_id``).
RESULTS_MODEL_COLUMNS: Final[tuple[str, ...]] = (*RESULTS_COLUMNS[:-1], "λ", "μ", "P esito (%)", "Sorpresa", "match_id")


def matchday_columns(form_n: int = DEFAULT_FORM_N, xg_n: int = DEFAULT_XG_N) -> tuple[str, ...]:
    """Column labels of :func:`matchday_table` for a form window ``form_n`` and an xG window ``xg_n``.

    The windows are spelled out in the labels (``'Pt Casa (5)'``, ``'xG F/S
    Casa (3)'``) so a table is self-describing wherever it ends up (screen,
    console, CSV). :data:`MATCHDAY_COLUMNS` are the labels for the defaults.
    """
    return (
        "Competizione",
        "Data",
        "Giornata",
        "Casa",
        "Ospiti",
        "α Casa",
        "β Casa",
        "α Ospiti",
        "β Ospiti",
        "Forma Casa",
        "Forma Ospiti",
        f"Pt Casa ({form_n})",
        f"Pt Ospiti ({form_n})",
        f"GF/GS Casa ({form_n})",
        f"GF/GS Ospiti ({form_n})",
        f"xG F/S Casa ({xg_n})",
        f"xG F/S Ospiti ({xg_n})",
        "λ",
        "μ",
        "Tot. att.",
        "Ris. prob.",
        "P1 (%)",
        "PX (%)",
        "P2 (%)",
        "O1.5 (%)",
        "O2.5 (%)",
        "O3.5 (%)",
        "GG (%)",
        "Fair 1",
        "Fair X",
        "Fair 2",
        "Fair O2.5",
        "Fair GG",
        "Book 1",
        "Book X",
        "Book 2",
        "Book O2.5",
        "Book GG",
        "EV max (%)",
        "Miglior EV",
        "Flag",
        "match_id",
    )


#: Column labels of :func:`matchday_table` with the default windows (5 matches of form, 3 of xG).
MATCHDAY_COLUMNS: Final[tuple[str, ...]] = matchday_columns()

#: Selections shown with their own probability / fair / book column, ``(market, selection, suffix)``.
_PROB_COLUMNS: Final[tuple[tuple[Market, str, str], ...]] = (
    (Market.MATCH_ODDS, "1", "P1 (%)"),
    (Market.MATCH_ODDS, "X", "PX (%)"),
    (Market.MATCH_ODDS, "2", "P2 (%)"),
    (Market.OVER_UNDER_15, "Over 1.5", "O1.5 (%)"),
    (Market.OVER_UNDER_25, "Over 2.5", "O2.5 (%)"),
    (Market.OVER_UNDER_35, "Over 3.5", "O3.5 (%)"),
    (Market.BTTS, "GG", "GG (%)"),
)
_ODDS_COLUMNS: Final[tuple[tuple[Market, str, str], ...]] = (
    (Market.MATCH_ODDS, "1", "1"),
    (Market.MATCH_ODDS, "X", "X"),
    (Market.MATCH_ODDS, "2", "2"),
    (Market.OVER_UNDER_25, "Over 2.5", "O2.5"),
    (Market.BTTS, "GG", "GG"),
)

_MATCHDAY_STRING_COLUMNS: Final[tuple[str, ...]] = (
    "Competizione",
    "Casa",
    "Ospiti",
    "Forma Casa",
    "Forma Ospiti",
    "Ris. prob.",
    "Miglior EV",
    "Flag",
    "match_id",
)
_RESULTS_STRING_COLUMNS: Final[tuple[str, ...]] = (
    "Competizione",
    "Casa",
    "Ospiti",
    "Risultato",
    "Esito",
    "Tiri C/O",
    "In porta C/O",
    "Angoli C/O",
    "Sorpresa",
    "match_id",
)


# =============================================================================
# Helpers
# =============================================================================
def _coerce_date(value: date | datetime | pd.Timestamp | str) -> date:
    if isinstance(value, pd.Timestamp):
        return value.to_pydatetime().date()
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    if isinstance(value, str):
        return date.fromisoformat(value.strip()[:10])
    raise TypeError(f"cannot interpret {value!r} as a date")


def _season_label(label: str) -> str:
    return str(label).strip().replace("_", "/")


def _restrict_matches(matches: pd.DataFrame, *, as_of: date | None, season: str | None) -> pd.DataFrame:
    """Canonical copy of ``matches`` with only the rows dated before ``as_of`` and, when given, of one season."""
    df = validate_matches_df(matches)
    if as_of is not None:
        df = df.loc[df["date"] < pd.Timestamp(_coerce_date(as_of))]
    if season is not None and len(df):
        wanted = _season_label(season)
        seasons = df["date"].map(season_of)
        df = df.loc[(seasons == wanted).to_numpy(dtype=bool)]
    return df.reset_index(drop=True)


def _pair_text(left: float, right: float, decimals: int) -> str:
    """``'12/7'`` (``decimals=0``) or ``'4.1/2.6'`` (``decimals=1``); a missing side reads ``'-'``, both missing ``'-'``."""
    left_missing = left is None or (isinstance(left, float) and math.isnan(left))
    right_missing = right is None or (isinstance(right, float) and math.isnan(right))
    if left_missing and right_missing:
        return MISSING_TEXT
    fmt = f"{{:.{decimals}f}}"
    a = MISSING_TEXT if left_missing else fmt.format(float(left))
    b = MISSING_TEXT if right_missing else fmt.format(float(right))
    return f"{a}/{b}"


# =============================================================================
# Per-team long frame and form
# =============================================================================
def team_matches(matches: pd.DataFrame, *, as_of: date | None = None, season: str | None = None) -> pd.DataFrame:
    """One row per **team and played match** (two rows per match), the base of every form computation.

    Columns :data:`TEAM_MATCHES_COLUMNS`: ``goals_for`` / ``goals_against`` /
    ``xg_for`` / ``xg_against`` from the team's point of view (``xg_*`` ``NaN``
    when the feed has no xG), ``result`` in ``V`` / ``N`` / ``P``, ``points`` (3 /
    1 / 0) and ``clean_sheet``. Only the rows of ``matches`` dated **before**
    ``as_of`` (every row when ``None``) and, with ``season``, of that season label
    (:func:`~src.common.schemas.season_of`) are used. Sorted by team, date and
    ``match_id`` so the last ``n`` rows of a team are its most recent matches.
    Build it once and pass it to :func:`form_table` / :func:`matchday_table`
    when several tables are derived from the same matches.
    """
    df = _restrict_matches(matches, as_of=as_of, season=season)
    if df.empty:
        return _empty_team_matches()
    hg = df["home_goals"].to_numpy(dtype="int64")
    ag = df["away_goals"].to_numpy(dtype="int64")
    base = {
        "match_id": df["match_id"].astype("string").to_numpy(),
        "date": df["date"].to_numpy(),
        "girone": df["girone"].astype("string").to_numpy(),
    }
    home = pd.DataFrame(
        {
            "team": df["home_team"].astype("string").to_numpy(),
            **base,
            "opponent": df["away_team"].astype("string").to_numpy(),
            "home": np.ones(len(df), dtype=bool),
            "goals_for": hg,
            "goals_against": ag,
            "xg_for": df["home_xg"].to_numpy(dtype=float),
            "xg_against": df["away_xg"].to_numpy(dtype=float),
        }
    )
    away = pd.DataFrame(
        {
            "team": df["away_team"].astype("string").to_numpy(),
            **base,
            "opponent": df["home_team"].astype("string").to_numpy(),
            "home": np.zeros(len(df), dtype=bool),
            "goals_for": ag,
            "goals_against": hg,
            "xg_for": df["away_xg"].to_numpy(dtype=float),
            "xg_against": df["home_xg"].to_numpy(dtype=float),
        }
    )
    long = pd.concat([home, away], ignore_index=True)
    gf = long["goals_for"].to_numpy()
    ga = long["goals_against"].to_numpy()
    result = np.where(gf > ga, RESULT_WIN, np.where(gf == ga, RESULT_DRAW, RESULT_LOSS))
    long["result"] = pd.array(result, dtype="string")
    long["points"] = np.where(gf > ga, 3, np.where(gf == ga, 1, 0)).astype("int64")
    long["clean_sheet"] = ga == 0
    for column in ("team", "match_id", "girone", "opponent"):
        long[column] = long[column].astype("string")
    long = long.sort_values(["team", "date", "match_id"], kind="stable").reset_index(drop=True)
    return long[list(TEAM_MATCHES_COLUMNS)]


def _empty_team_matches() -> pd.DataFrame:
    dtypes = {
        "team": "string",
        "match_id": "string",
        "date": "datetime64[ns]",
        "girone": "string",
        "opponent": "string",
        "home": "bool",
        "goals_for": "int64",
        "goals_against": "int64",
        "xg_for": "float64",
        "xg_against": "float64",
        "result": "string",
        "points": "int64",
        "clean_sheet": "bool",
    }
    return pd.DataFrame({c: pd.Series(dtype=t) for c, t in dtypes.items()})[list(TEAM_MATCHES_COLUMNS)]


def _results_string(results: pd.Series) -> str:
    """``'V-P-N'`` from the chronologically ordered results of a window, most recent first."""
    return "-".join(reversed([str(r) for r in results.tolist()]))


def form_table(
    matches: pd.DataFrame | None = None,
    *,
    n: int = DEFAULT_FORM_N,
    as_of: date | None = None,
    season: str | None = None,
    long: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Form of **every** team over its last ``n`` played matches, one row per team (index = team name).

    Columns: ``played`` (all the team's matches in scope, not only the window),
    ``n_window`` (matches actually in the window, ``<= n``), ``points``,
    ``goals_for``, ``goals_against``, ``xg_for`` / ``xg_against`` (both ``NaN`` as
    soon as one xG value, for or against, of the window is missing), ``clean_sheets``, ``results`` (``'V-P-N'``,
    most recent first) and ``last_date``. ``long`` is a :func:`team_matches`
    frame already built (``matches``, ``as_of`` and ``season`` are then ignored);
    otherwise it is built from ``matches``. One ``groupby`` for all the teams,
    whatever their number. ``n < 1`` raises :class:`ValueError`.
    """
    n = int(n)
    if n < 1:
        raise ValueError(f"n must be >= 1, got {n}")
    if long is None:
        if matches is None:
            raise ValueError("form_table needs either matches or a team_matches frame")
        long = team_matches(matches, as_of=as_of, season=season)
    if long.empty:
        return _empty_form_table()
    played = long.groupby("team", sort=True, observed=True).size().rename("played")
    recent = long.groupby("team", sort=False, observed=True).tail(n)
    agg = recent.groupby("team", sort=True, observed=True).agg(
        n_window=("match_id", "size"),
        points=("points", "sum"),
        goals_for=("goals_for", "sum"),
        goals_against=("goals_against", "sum"),
        xg_for=("xg_for", "sum"),
        n_xg_for=("xg_for", "count"),
        xg_against=("xg_against", "sum"),
        n_xg_against=("xg_against", "count"),
        clean_sheets=("clean_sheet", "sum"),
        last_date=("date", "max"),
        results=("result", _results_string),
    )
    # both xG sums are NaN as soon as one xG value (for or against) of the window is missing: a partial window
    # would read as a complete one ('4.1/2.6' with a hidden gap), so the column says '-' instead
    complete = (agg["n_xg_for"] == agg["n_window"]) & (agg["n_xg_against"] == agg["n_window"])
    agg["xg_for"] = agg["xg_for"].where(complete, np.nan)
    agg["xg_against"] = agg["xg_against"].where(complete, np.nan)
    out = agg.join(played, how="left")
    out = out[["played", "n_window", "points", "goals_for", "goals_against", "xg_for", "xg_against", "clean_sheets", "results", "last_date"]]
    for column in ("played", "n_window", "points", "goals_for", "goals_against", "clean_sheets"):
        out[column] = out[column].astype("int64")
    out["results"] = out["results"].astype("string")
    out.index = out.index.astype("string")
    out.index.name = "team"
    return out


def _empty_form_table() -> pd.DataFrame:
    dtypes = {
        "played": "int64",
        "n_window": "int64",
        "points": "int64",
        "goals_for": "int64",
        "goals_against": "int64",
        "xg_for": "float64",
        "xg_against": "float64",
        "clean_sheets": "int64",
        "results": "string",
        "last_date": "datetime64[ns]",
    }
    frame = pd.DataFrame({c: pd.Series(dtype=t) for c, t in dtypes.items()})
    frame.index = pd.Index([], dtype="string", name="team")
    return frame


@dataclass(frozen=True)
class TeamForm:
    """Recent form of one team over its last ``n`` played matches (see :func:`team_form`).

    ``played`` counts every match of the team in scope (before ``as_of``, in the
    season), the ``*_last_n`` fields only the window. ``xg_for_last_n`` /
    ``xg_against_last_n`` are ``NaN`` as soon as one match of the window has no
    xG. ``results_last_n`` reads most recent first (``'V-P-N-V-V'``: ``V``
    vittoria, ``N`` pareggio (nullo), ``P`` sconfitta; ``''`` without matches); ``last_date``
    is the date of the most recent match (``None`` without matches).
    """

    team: str
    played: int
    points_last_n: int
    goals_for_last_n: int
    goals_against_last_n: int
    xg_for_last_n: float
    xg_against_last_n: float
    results_last_n: str
    clean_sheets_last_n: int
    last_date: date | None
    n: int = DEFAULT_FORM_N

    @property
    def n_window(self) -> int:
        """Matches actually in the window (``0`` without matches, ``<= n``)."""
        return 0 if not self.results_last_n else self.results_last_n.count("-") + 1

    @property
    def goals_text(self) -> str:
        """``'9/4'``: goals for / goals against over the window."""
        return f"{self.goals_for_last_n}/{self.goals_against_last_n}"

    @property
    def xg_text(self) -> str:
        """``'4.1/2.6'``: xG for / against over the window, ``'-'`` when any xG is missing."""
        if math.isnan(self.xg_for_last_n) or math.isnan(self.xg_against_last_n):
            return MISSING_TEXT
        return f"{self.xg_for_last_n:.1f}/{self.xg_against_last_n:.1f}"

    @classmethod
    def empty(cls, team: str, n: int = DEFAULT_FORM_N) -> "TeamForm":
        """The form of a team without played matches in scope."""
        return cls(str(team), 0, 0, 0, 0, float("nan"), float("nan"), "", 0, None, int(n))

    @classmethod
    def from_row(cls, team: str, row: Mapping[str, Any], n: int) -> "TeamForm":
        """Build from one row of :func:`form_table`."""
        last = row["last_date"]
        return cls(
            team=str(team),
            played=int(row["played"]),
            points_last_n=int(row["points"]),
            goals_for_last_n=int(row["goals_for"]),
            goals_against_last_n=int(row["goals_against"]),
            xg_for_last_n=float(row["xg_for"]),
            xg_against_last_n=float(row["xg_against"]),
            results_last_n=str(row["results"]),
            clean_sheets_last_n=int(row["clean_sheets"]),
            last_date=None if pd.isna(last) else _coerce_date(pd.Timestamp(last)),
            n=int(n),
        )


def team_form(
    matches: pd.DataFrame,
    team: str,
    *,
    as_of: date | None = None,
    n: int = DEFAULT_FORM_N,
    season: str | None = None,
) -> TeamForm:
    """Form of ``team`` over its last ``n`` played matches (home and away) dated before ``as_of``.

    ``as_of=None`` uses every row of ``matches`` (the canonical frame holds
    played matches only); ``season`` restricts the matches to one season label
    (``'2026/27'`` or ``'2026_27'``). A team without matches in scope yields
    :meth:`TeamForm.empty`. For every team at once use :func:`form_table`.
    """
    n = int(n)
    if n < 1:
        raise ValueError(f"n must be >= 1, got {n}")
    df = validate_matches_df(matches)
    mine = df.loc[(df["home_team"] == team) | (df["away_team"] == team)]
    table = form_table(mine, n=n, as_of=as_of, season=season)
    if str(team) not in table.index:
        return TeamForm.empty(team, n)
    return TeamForm.from_row(team, table.loc[str(team)], n)


def _form_lookup(table: pd.DataFrame, n: int) -> dict[str, TeamForm]:
    """``team -> TeamForm`` for every row of a :func:`form_table` frame."""
    return {str(team): TeamForm.from_row(str(team), row, n) for team, row in table.iterrows()}


# =============================================================================
# Matchday table
# =============================================================================
def _order_flags(flags: Iterable[str], home_team: str, away_team: str) -> str:
    """Join a set of flags in a stable order: the known flags first (``TRAP:`` home then away), then the rest sorted."""
    pool = {str(f) for f in flags}
    ordered: list[str] = []
    for flag in _FLAG_ORDER:
        if flag in pool:
            ordered.append(flag)
            pool.discard(flag)
            if flag == FLAG_REGRESSION_TRAP:
                for team in (home_team, away_team):
                    trap_flag = f"{TRAP_FLAG_PREFIX}{team}"
                    if trap_flag in pool:
                        ordered.append(trap_flag)
                        pool.discard(trap_flag)
    ordered.extend(sorted(pool))
    return FLAG_SEPARATOR.join(ordered)


@dataclass
class _FixtureQuotes:
    """What the value engine says about one fixture: best price per selection, best EV, every flag.

    The two containers are per-instance defaults (``default_factory``), so
    every fixture starts with its own empty price map and flag set.
    """

    best_price: dict[tuple[Market, str], float] = field(default_factory=dict)
    best: ValueOpportunity | None = None
    flags: set[str] = field(default_factory=set)

    def add(self, o: ValueOpportunity) -> None:
        key = (o.market, o.selection)
        if key not in self.best_price or o.book_odds > self.best_price[key]:
            self.best_price[key] = float(o.book_odds)
        if self.best is None or o.ev > self.best.ev or (o.ev == self.best.ev and (o.book_odds, o.bookmaker) > (self.best.book_odds, self.best.bookmaker)):
            self.best = o
        self.flags.update(o.flags)


def _quotes_by_fixture(opportunities: Iterable[ValueOpportunity]) -> dict[str, _FixtureQuotes]:
    by_fixture: dict[str, _FixtureQuotes] = {}
    for o in opportunities:
        entry = by_fixture.get(o.match_id)
        if entry is None:
            entry = by_fixture[o.match_id] = _FixtureQuotes()
        entry.add(o)
    return by_fixture


def _param(mapping: Mapping[str, float], team: str) -> float:
    value = mapping.get(team)
    return float("nan") if value is None else round(float(value), 2)


def matchday_table(
    predictions: Iterable[MatchPrediction],
    *,
    params: FittedParams,
    matches: pd.DataFrame,
    opportunities: Iterable[ValueOpportunity] = (),
    traps: Iterable[RegressionTrap] = (),
    as_of: date | None = None,
    form_n: int = DEFAULT_FORM_N,
    xg_n: int = DEFAULT_XG_N,
    matchday: Matchday | None = None,
    girone: Girone | None = None,
    long: pd.DataFrame | None = None,
    gironi: Iterable[Girone | str] | None = None,
) -> pd.DataFrame:
    """One row per fixture of the matchday: strengths, form, model, markets, best prices, EV and flags.

    Columns exactly :func:`matchday_columns` ``(form_n, xg_n)`` (:data:`MATCHDAY_COLUMNS`
    with the defaults), sorted by date, competition order and home team:

    * ``α`` / ``β`` of both teams from ``params`` (2 decimals, ``NaN`` for a team
      the fit does not know);
    * form over the last ``form_n`` played matches of ``matches`` dated before
      ``as_of`` (every match when ``None``): ``Forma`` (``'V-P-N-V-V'``, most
      recent first), ``Pt`` (points), ``GF/GS`` (``'9/4'``); xG for / against over
      the last ``xg_n`` matches (``'4.1/2.6'``, ``'-'`` when an xG is missing);
    * ``λ``, ``μ``, ``Tot. att.`` (2 decimals), ``Ris. prob.`` (``'1-0 (14%)'``),
      the market probabilities in percent (1 decimal) and the fair odds (2
      decimals) from the prediction;
    * ``Book ...``: the best (highest) bookmaker price per selection among
      ``opportunities`` (``NaN`` when no bookmaker quotes it), ``EV max (%)`` the
      best EV among the fixture's quoted selections (``NaN`` without quotes),
      ``Miglior EV`` that selection as ``'X @3.40'`` (``''`` without quotes),
      ``Flag`` the union of the flags of every quoted selection plus
      ``REGRESSION_TRAP`` / ``TRAP:<team>`` for a team in ``traps`` even when
      nothing is quoted;
    * ``match_id`` last, so a UI can map a selected row back to its fixture.

    ``matchday`` keeps only its fixtures, ``girone`` only one competition and
    ``gironi`` only a set of competitions (the dashboard section's, see
    :func:`~src.engine.picks.in_gironi`; ``None`` = every competition).
    ``long`` is a :func:`team_matches` frame built in advance (``matches`` and
    ``as_of`` are then ignored for the form): callers that build several tables
    pass it so the per-team frame is computed once. The form is computed with
    one ``groupby`` for every team, never by filtering per fixture, so the
    function stays fast on a season-long fixture list.
    """
    started = time.perf_counter()
    form_n, xg_n = int(form_n), int(xg_n)
    if form_n < 1 or xg_n < 1:
        raise ValueError(f"form_n and xg_n must be >= 1, got {form_n} and {xg_n}")
    g = None if girone is None else Girone.parse(girone)
    allowed = None if gironi is None else tuple(Girone.parse(x) for x in gironi)
    preds = [
        p
        for p in predictions
        if (matchday is None or matchday.contains(p))
        and (g is None or Girone.parse(p.fixture.girone) == g)
        and in_gironi(p.fixture.girone, allowed)
    ]
    columns = matchday_columns(form_n, xg_n)
    if not preds:
        return _empty_matchday_table(columns)
    team_frame = team_matches(matches, as_of=as_of) if long is None else long
    form = _form_lookup(form_table(long=team_frame, n=form_n), form_n)
    xg_form = _form_lookup(form_table(long=team_frame, n=xg_n), xg_n)
    quotes = _quotes_by_fixture(opportunities)
    trapped: dict[str, RegressionTrap] = {}
    for trap in traps:
        current = trapped.get(trap.team)
        if current is None or trap.severity > current.severity:
            trapped[trap.team] = trap

    rows: list[dict[str, Any]] = []
    for p in preds:
        fx = p.fixture
        home, away = str(fx.home_team), str(fx.away_team)
        f_home = form.get(home) or TeamForm.empty(home, form_n)
        f_away = form.get(away) or TeamForm.empty(away, form_n)
        x_home = xg_form.get(home) or TeamForm.empty(home, xg_n)
        x_away = xg_form.get(away) or TeamForm.empty(away, xg_n)
        hs, as_, p_top = p.most_likely_score()
        quote = quotes.get(fx.match_id)
        flags: set[str] = set(quote.flags) if quote is not None else set()
        for team in (home, away):
            if team in trapped:
                flags.add(FLAG_REGRESSION_TRAP)
                flags.add(f"{TRAP_FLAG_PREFIX}{team}")
        row: dict[str, Any] = {
            "Competizione": Girone.parse(fx.girone).short,
            "Data": pd.Timestamp(fx.date),
            "Giornata": int(fx.round),
            "Casa": home,
            "Ospiti": away,
            "α Casa": _param(params.alpha, home),
            "β Casa": _param(params.beta, home),
            "α Ospiti": _param(params.alpha, away),
            "β Ospiti": _param(params.beta, away),
            "Forma Casa": f_home.results_last_n,
            "Forma Ospiti": f_away.results_last_n,
            f"Pt Casa ({form_n})": f_home.points_last_n,
            f"Pt Ospiti ({form_n})": f_away.points_last_n,
            f"GF/GS Casa ({form_n})": f_home.goals_text,
            f"GF/GS Ospiti ({form_n})": f_away.goals_text,
            f"xG F/S Casa ({xg_n})": x_home.xg_text,
            f"xG F/S Ospiti ({xg_n})": x_away.xg_text,
            "λ": round(float(p.home_rate), 2),
            "μ": round(float(p.away_rate), 2),
            "Tot. att.": round(float(p.home_rate + p.away_rate), 2),
            "Ris. prob.": f"{hs}-{as_} ({p_top:.0%})",
        }
        for market, selection, label in _PROB_COLUMNS:
            row[label] = round(float(p.probabilities[market][selection]) * 100.0, 1)
        for market, selection, suffix in _ODDS_COLUMNS:
            row[f"Fair {suffix}"] = round(float(p.fair_odds[market][selection]), 2)
        for market, selection, suffix in _ODDS_COLUMNS:
            price = quote.best_price.get((market, selection)) if quote is not None else None
            row[f"Book {suffix}"] = float("nan") if price is None else round(price, 2)
        best = quote.best if quote is not None else None
        row["EV max (%)"] = float("nan") if best is None else round(best.ev * 100.0, 2)
        row["Miglior EV"] = "" if best is None else f"{best.selection} @{best.book_odds:.2f}"
        row["Flag"] = _order_flags(flags, home, away)
        row["match_id"] = str(fx.match_id)
        rows.append(row)

    frame = pd.DataFrame(rows, columns=list(columns))
    frame = frame.assign(Data=pd.to_datetime(frame["Data"]).astype("datetime64[ns]"), Giornata=frame["Giornata"].astype("int64"))
    for column in (f"Pt Casa ({form_n})", f"Pt Ospiti ({form_n})"):
        frame[column] = frame[column].astype("int64")
    for column in _MATCHDAY_STRING_COLUMNS + (f"GF/GS Casa ({form_n})", f"GF/GS Ospiti ({form_n})", f"xG F/S Casa ({xg_n})", f"xG F/S Ospiti ({xg_n})"):
        frame[column] = frame[column].astype("string")
    numeric = [c for c in columns if c not in _MATCHDAY_STRING_COLUMNS and c not in ("Data", "Giornata") and not c.startswith(("Pt ", "GF/GS", "xG F/S"))]
    for column in numeric:
        frame[column] = frame[column].astype("float64")
    order = frame["Competizione"].map({g.short: i for i, g in enumerate(Girone)}).astype("int64")
    frame = frame.assign(_order=order).sort_values(["Data", "_order", "Casa"], kind="stable").drop(columns="_order").reset_index(drop=True)
    log.debug(
        "matchday table built",
        extra={
            "n_fixtures": len(frame),
            "n_quoted": int(frame["EV max (%)"].notna().sum()),
            "n_trapped": int(frame["Flag"].str.contains(FLAG_REGRESSION_TRAP, regex=False).fillna(False).sum()),
            "matchday": None if matchday is None else matchday.key,
            "girone": None if g is None else g.value,
            "gironi": None if allowed is None else [x.value for x in allowed],
            "seconds": round(time.perf_counter() - started, 4),
        },
    )
    return frame


def _empty_matchday_table(columns: Sequence[str]) -> pd.DataFrame:
    frame = pd.DataFrame({c: pd.Series(dtype="object") for c in columns})
    frame["Data"] = frame["Data"].astype("datetime64[ns]")
    frame["Giornata"] = frame["Giornata"].astype("int64")
    for column in columns:
        if column in _MATCHDAY_STRING_COLUMNS or column.startswith(("GF/GS", "xG F/S")):
            frame[column] = frame[column].astype("string")
        elif column.startswith("Pt "):
            frame[column] = frame[column].astype("int64")
        elif column not in ("Data", "Giornata"):
            frame[column] = frame[column].astype("float64")
    return frame[list(columns)]


# =============================================================================
# Results table
# =============================================================================
def outcome_probabilities(
    lam: np.ndarray | Sequence[float],
    mu: np.ndarray | Sequence[float],
    rho: float,
    max_goals: int = 10,
) -> np.ndarray:
    """``P(1), P(X), P(2)`` of many matches at once, shape ``(n, 3)``, from their Dixon-Coles rates.

    The same construction as :func:`src.engine.model.dixon_coles_pmf` (Poisson
    grid times :func:`~src.engine.model.tau`, clipped at zero, renormalised on
    the ``(max_goals + 1)^2`` grid) summed below / on / above the diagonal, so
    the values agree with :func:`~src.engine.model.markets_from_matrix` of the
    engine's score matrix to floating-point precision. Rows whose rate is not
    finite (a team unknown to the fit) are ``NaN``.
    """
    lam_arr = np.asarray(lam, dtype=float).reshape(-1)
    mu_arr = np.asarray(mu, dtype=float).reshape(-1)
    if lam_arr.shape != mu_arr.shape:
        raise ValueError(f"lam and mu must have the same length, got {lam_arr.shape} and {mu_arr.shape}")
    if int(max_goals) < 1:
        raise ValueError(f"max_goals must be >= 1, got {max_goals}")
    out = np.full((len(lam_arr), 3), np.nan)
    valid = np.isfinite(lam_arr) & np.isfinite(mu_arr) & (lam_arr >= 0.0) & (mu_arr >= 0.0)
    if not valid.any():
        return out
    ks = np.arange(int(max_goals) + 1)
    log_fact = gammaln(ks + 1.0)[None, :]
    l_col = lam_arr[valid][:, None]
    m_col = mu_arr[valid][:, None]
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        p_home = np.exp(xlogy(ks[None, :], l_col) - l_col - log_fact)
        p_away = np.exp(xlogy(ks[None, :], m_col) - m_col - log_fact)
    grid = p_home[:, :, None] * p_away[:, None, :]
    xs, ys = np.meshgrid(ks, ks, indexing="ij")
    correction = tau(xs[None, :, :], ys[None, :, :], l_col[:, :, None], m_col[:, :, None], float(rho))
    grid = np.clip(grid * np.asarray(correction, dtype=float), 0.0, None)
    total = grid.sum(axis=(1, 2), keepdims=True)
    grid = grid / total
    p1 = grid[:, xs > ys].sum(axis=1)
    px = grid[:, xs == ys].sum(axis=1)
    p2 = grid[:, xs < ys].sum(axis=1)
    out[valid, 0] = p1
    out[valid, 1] = px
    out[valid, 2] = p2
    return out


def p_outcome_percent(p: float | np.ndarray | Sequence[float]) -> np.ndarray:
    """``P esito (%)`` as displayed: the probability in percent rounded to :data:`P_OUTCOME_DECIMALS` decimals."""
    return np.round(np.asarray(p, dtype=float) * 100.0, P_OUTCOME_DECIMALS)


def surprise_tag(p: float) -> str:
    """``Sorpresa`` tag of one played match from the probability ``p`` the model gave the outcome that happened.

    The thresholds (:data:`SURPRISE_BELOW` = 20%, :data:`EXPECTED_ABOVE` = 50%)
    are compared with the **displayed** percentage, ``P esito (%)`` rounded to
    :data:`P_OUTCOME_DECIMALS` decimal (:func:`p_outcome_percent`), never with
    the raw probability: ``p = 0.1996`` shows ``20.0`` and is **not** a surprise,
    while ``p = 0.1994`` shows ``19.9`` and is. ``''`` for a ``NaN`` (a team the
    fit does not know) and for anything between the two thresholds.
    """
    if p is None or math.isnan(p):
        return ""
    shown = float(p_outcome_percent(p))
    if shown < SURPRISE_BELOW * 100.0:
        return SURPRISE_TAG
    if shown > EXPECTED_ABOVE * 100.0:
        return EXPECTED_TAG
    return ""


def results_table(
    matches: pd.DataFrame,
    *,
    engine: DixonColesEngine | None = None,
    season: str | None = None,
    girone: Girone | None = None,
    team: str | None = None,
    gironi: Iterable[Girone | str] | None = None,
) -> pd.DataFrame:
    """One row per **played** match with its statistics and, with ``engine``, the model's view of it.

    Columns :data:`RESULTS_COLUMNS` (``Risultato`` ``'2-1'``, ``Esito`` ``'1'`` /
    ``'X'`` / ``'2'`` from the goals, ``Tiri C/O`` / ``In porta C/O`` / ``Angoli
    C/O`` as ``'12/7'`` with ``'-'`` for a missing side, ``xG Casa`` / ``xG
    Ospiti`` with 2 decimals) or :data:`RESULTS_MODEL_COLUMNS` when ``engine`` is
    given: ``λ`` / ``μ`` (the fitted rates of the match, 2 decimals), ``P esito
    (%)`` (the probability the model assigns to the outcome that happened, 1
    decimal, from the rates of :meth:`DixonColesEngine.residuals` and
    :func:`outcome_probabilities`) and ``Sorpresa`` (``'sorpresa'`` below 20%,
    ``'atteso'`` above 50%, else ``''``, judged on the displayed 1-decimal
    percentage, see :func:`surprise_tag`). The engine was fitted on these very
    matches, so ``P esito`` is an in-sample reading, not a forecast.

    ``season`` (``'2026/27'`` or ``'2026_27'``) keeps one season, ``girone`` one
    competition, ``gironi`` a set of competitions (the dashboard section's, see
    :func:`~src.engine.picks.in_gironi`), ``team`` the matches of one team, home
    or away -- historical teams included, since they are rows of ``matches``
    like any other. Sorted by date descending, then competition order and home
    team; ``match_id`` last.
    """
    df = validate_matches_df(matches)
    if season is not None and len(df):
        wanted = _season_label(season)
        df = df.loc[(df["date"].map(season_of) == wanted).to_numpy(dtype=bool)]
    if girone is not None and len(df):
        df = df.loc[df["girone"] == Girone.parse(girone).value]
    if gironi is not None and len(df):
        codes = {Girone.parse(x).value for x in gironi}
        df = df.loc[df["girone"].isin(codes).to_numpy(dtype=bool)]
    if team is not None and len(df):
        df = df.loc[(df["home_team"] == team) | (df["away_team"] == team)]
    df = df.reset_index(drop=True)
    columns = RESULTS_MODEL_COLUMNS if engine is not None else RESULTS_COLUMNS
    if df.empty:
        return _empty_results_table(columns)

    hg = df["home_goals"].to_numpy(dtype="int64")
    ag = df["away_goals"].to_numpy(dtype="int64")
    outcome = np.where(hg > ag, "1", np.where(hg == ag, "X", "2"))
    frame = pd.DataFrame(
        {
            "Competizione": df["girone"].map(lambda code: Girone.parse(str(code)).short).to_numpy(),
            "Data": df["date"].to_numpy(),
            "Giornata": df["round"].to_numpy(dtype="int64"),
            "Casa": df["home_team"].astype("string").to_numpy(),
            "Ospiti": df["away_team"].astype("string").to_numpy(),
            "Risultato": [f"{h}-{a}" for h, a in zip(hg, ag)],
            "Esito": outcome,
            "Tiri C/O": [_pair_text(h, a, 0) for h, a in zip(df["home_shots"].to_numpy(dtype=float), df["away_shots"].to_numpy(dtype=float))],
            "In porta C/O": [
                _pair_text(h, a, 0) for h, a in zip(df["home_shots_target"].to_numpy(dtype=float), df["away_shots_target"].to_numpy(dtype=float))
            ],
            "Angoli C/O": [_pair_text(h, a, 0) for h, a in zip(df["home_corners"].to_numpy(dtype=float), df["away_corners"].to_numpy(dtype=float))],
            "xG Casa": df["home_xg"].to_numpy(dtype=float).round(2),
            "xG Ospiti": df["away_xg"].to_numpy(dtype=float).round(2),
        }
    )
    if engine is not None:
        params = engine.params_
        residuals = engine.residuals(df)
        lam = residuals["lam"].to_numpy(dtype=float)
        mu = residuals["mu"].to_numpy(dtype=float)
        probs = outcome_probabilities(lam, mu, params.rho, params.max_goals)
        index = np.where(hg > ag, 0, np.where(hg == ag, 1, 2))
        p_outcome = probs[np.arange(len(df)), index]
        frame["λ"] = np.round(lam, 2)
        frame["μ"] = np.round(mu, 2)
        frame["P esito (%)"] = p_outcome_percent(p_outcome)
        frame["Sorpresa"] = [surprise_tag(p) for p in p_outcome]
    frame["match_id"] = df["match_id"].astype("string").to_numpy()
    frame = frame[list(columns)]
    frame = frame.assign(Data=pd.to_datetime(frame["Data"]).astype("datetime64[ns]"), Giornata=frame["Giornata"].astype("int64"))
    for column in _RESULTS_STRING_COLUMNS:
        if column in frame.columns:
            frame[column] = frame[column].astype("string")
    for column in ("xG Casa", "xG Ospiti", "λ", "μ", "P esito (%)"):
        if column in frame.columns:
            frame[column] = frame[column].astype("float64")
    order = frame["Competizione"].map({g.short: i for i, g in enumerate(Girone)}).astype("int64")
    frame = (
        frame.assign(_order=order)
        .sort_values(["Data", "_order", "Casa"], ascending=[False, True, True], kind="stable")
        .drop(columns="_order")
        .reset_index(drop=True)
    )
    log.debug(
        "results table built",
        extra={
            "n_matches": len(frame),
            "season": season,
            "girone": None if girone is None else Girone.parse(girone).value,
            "gironi": None if gironi is None else sorted(Girone.parse(x).value for x in gironi),
            "team": team,
            "with_model": engine is not None,
        },
    )
    return frame


def _empty_results_table(columns: Sequence[str]) -> pd.DataFrame:
    frame = pd.DataFrame({c: pd.Series(dtype="object") for c in columns})
    frame["Data"] = frame["Data"].astype("datetime64[ns]")
    frame["Giornata"] = frame["Giornata"].astype("int64")
    for column in columns:
        if column in _RESULTS_STRING_COLUMNS:
            frame[column] = frame[column].astype("string")
        elif column not in ("Data", "Giornata"):
            frame[column] = frame[column].astype("float64")
    return frame[list(columns)]
