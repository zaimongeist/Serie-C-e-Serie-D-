"""Pure data layer of the Streamlit dashboard (no ``streamlit`` import).

Everything the UI shows is computed here from the shared contracts, so the
dashboard can be unit-tested without a browser:

* :func:`build_state` runs the whole pipeline (load or simulate -> fit ->
  predict -> value) and returns an immutable-by-convention
  :class:`PipelineState` snapshot;
* the ``*_table`` helpers turn that snapshot into ``pandas`` frames with
  Italian column labels, ready for ``st.dataframe``;
* :func:`heatmap_figure` builds the score-matrix Plotly figure (whole matrix
  by default, or a compact view whose last row/column aggregate the tail);
* :func:`export_csv_bytes` serialises any frame for ``st.download_button``.

Bookmaker odds are read from ``odds_dir/*.csv`` in the same format
``main.load_odds_dir`` understands, see :func:`load_odds_quotes`.

Perimeter and history
---------------------
The dashboard ranks, predicts and prices only the teams of the registry of the
selected ``season`` (``config/teams_2026_27.json`` by default: 80 teams, the
three Serie C gironi and Serie B, i.e. every competition of
``config/competitions.toml`` the registry holds; the perimeter banner is
``registry.perimeter_label``, never a hard-coded string). The
match frame may hold up to three seasons: the rows of the registry season are
the *current* matches (coverage, traps and the auto-prior rule are judged on
them), the older rows are *history* that only sharpens the strength estimates
(the engine fits every team, flags the ones outside the registry as historical
and keeps them out of :func:`ranking_table`; fixtures involving them, and
fixtures dated in another season than the registry's -- a stale ``fixtures.csv``
of last season -- are skipped and listed in the notes). Without processed data
a league is simulated on the registry season's real calendar (2026/27 by
default; played rounds derived from ``as_of``, today by default, see
:func:`simulate_current_season`) together with ``history_seasons`` synthetic
seasons *preceding* it (:func:`src.simulation.generator.history_seasons_for`).
"""
from __future__ import annotations

import math
import time
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Final, Literal

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from src.common.logging_config import get_logger
from src.common.paths import ODDS_DIR, PROCESSED_DIR, ROOT
from src.common.schemas import (
    SELECTIONS,
    Fixture,
    Girone,
    Market,
    MatchPrediction,
    OddsQuote,
    empty_matches_frame,
    season_of,
    validate_matches_df,
)
from src.common.teams import DEFAULT_SEASON, TeamRegistry, load_registry
from src.engine.model import DixonColesConfig, DixonColesEngine, FittedParams
from src.engine.picks import (
    DEFAULT_N_BTTS,
    DEFAULT_N_DRAWS,
    PICKS_COLUMNS,
    Matchday,
    Pick,
    btts_picks,
    draw_picks,
    matchdays,
    picks_frame,
)
from src.engine.value import (
    KellyConfig,
    RegressionTrap,
    ValueEngine,
    ValueOpportunity,
    detect_regression_traps,
)
from src.ingestion.harvester import load_fixtures, load_processed
from src.simulation.generator import (
    HISTORY_SEASONS,
    SimulatedLeague,
    history_seasons_for,
    inject_regression_trap,
    round_robin,
    simulate_history,
    simulate_league,
)

__all__ = [
    "ALL_MATCHDAYS",
    "DEFAULT_HISTORY_SEASONS",
    "MARKETS_COLUMNS",
    "MARKET_LABELS",
    "MAX_HISTORY_SEASONS",
    "MIN_SIM_ROUNDS",
    "ODDS_COLUMNS",
    "PICKS_EXPORT_COLUMNS",
    "PRIOR_MIN_MATCHES_PER_TEAM",
    "PRIOR_THRESHOLD_MATCHES",
    "RANKING_COLUMNS",
    "TRAPS_COLUMNS",
    "VALUE_COLUMNS",
    "CoverageReport",
    "MatchdaySelector",
    "PipelineState",
    "Source",
    "btts_picks_table",
    "build_state",
    "competition_label",
    "coverage_report",
    "draw_picks_table",
    "export_csv_bytes",
    "find_matchday",
    "fixtures_for",
    "heatmap_figure",
    "history_label",
    "load_odds_quotes",
    "markets_table",
    "matchday_predictions",
    "matches_per_team",
    "max_sim_rounds",
    "no_odds_message",
    "opportunities_table",
    "other_season_reason",
    "perimeter_matches",
    "picks_csv_bytes",
    "picks_export_frame",
    "predictions_table",
    "prior_recommended",
    "quotes_for",
    "ranking_table",
    "resolve_matchday",
    "simulate_current_season",
    "split_by_season",
    "strength_figure",
    "traps_table",
    "value_opportunities",
    "value_table",
]

log = get_logger(__name__)

Source = Literal["processed", "simulated"]

#: Selector value meaning "every matchday" for :func:`fixtures_for`, :func:`value_table` and
#: :func:`matchday_predictions`; ``None`` means "the next matchday" and a :class:`Matchday` that one.
ALL_MATCHDAYS: Final = "all"
MatchdaySelector = Matchday | Literal["all"] | None

#: Columns of :func:`picks_export_frame`: ``Lista`` (``'X'`` / ``'GG'``) and ``Giornata`` in front of :data:`PICKS_COLUMNS`.
PICKS_EXPORT_COLUMNS: Final[tuple[str, ...]] = ("Lista", "Giornata", *PICKS_COLUMNS)

#: ``use_prior=None`` switches the Gaussian (MAP) prior on while the least-covered
#: girone has fewer than this many played matches per team (same rule as ``main.py``).
PRIOR_MIN_MATCHES_PER_TEAM: Final[float] = 15.0
#: Kept for backwards compatibility: the old rule was a total-match threshold.
PRIOR_THRESHOLD_MATCHES: Final[int] = 100

#: Smallest number of played rounds a simulated league can have (the fit needs matches).
MIN_SIM_ROUNDS: Final[int] = 1
#: Synthetic past seasons prepended to the simulated league by default (the two before the registry season:
#: 2024/25 and 2025/26 for 2026/27).
DEFAULT_HISTORY_SEASONS: Final[int] = 2
#: Largest number of synthetic past seasons available (``len(HISTORY_SEASONS)``).
MAX_HISTORY_SEASONS: Final[int] = len(HISTORY_SEASONS)

#: Required columns of a bookmaker odds CSV (one row per quoted selection).
ODDS_COLUMNS: Final[tuple[str, ...]] = ("match_id", "bookmaker", "market", "selection", "odds")

#: Italian labels of :func:`ranking_table`, in order. ``Competizione`` holds the competition's
#: short label (``'Girone A'``, ``'Serie B'``, see :func:`competition_label`), not the code.
RANKING_COLUMNS: Final[tuple[str, ...]] = (
    "Squadra",
    "Competizione",
    "Forza Offensiva (α)",
    "Solidità Difensiva (β)",
    "Vantaggio Casa (γ)",
    "Indice Attacco",
    "Indice Difesa",
)
_RANKING_RENAME: Final[dict[str, str]] = {
    "team": "Squadra",
    "girone": "Competizione",
    "alpha": "Forza Offensiva (α)",
    "beta": "Solidità Difensiva (β)",
    "gamma": "Vantaggio Casa (γ)",
    "attack_index": "Indice Attacco",
    "defence_index": "Indice Difesa",
}

#: Italian labels of :func:`markets_table`, in order.
MARKETS_COLUMNS: Final[tuple[str, ...]] = (
    "Mercato",
    "Esito",
    "Probabilità (%)",
    "Quota Fair",
    "Quota Book",
    "EV (%)",
)

#: Italian labels of :func:`value_table`, in order.
VALUE_COLUMNS: Final[tuple[str, ...]] = (
    "Data",
    "Competizione",
    "Partita",
    "Mercato",
    "Esito",
    "Bookmaker",
    "Quota Book",
    "Quota Fair",
    "P. Modello (%)",
    "EV (%)",
    "Stake (unità)",
    "Flag",
)

#: Italian labels of :func:`traps_table`, in order.
TRAPS_COLUMNS: Final[tuple[str, ...]] = ("Squadra", "Competizione", "Gol Subiti", "xGA", "Severità", "Ultima Gara")

#: Human-readable market names for the UI.
MARKET_LABELS: Final[dict[Market, str]] = {
    Market.MATCH_ODDS: "Esito finale 1X2",
    Market.OVER_UNDER_15: "Over/Under 1.5",
    Market.OVER_UNDER_25: "Over/Under 2.5",
    Market.OVER_UNDER_35: "Over/Under 3.5",
    Market.BTTS: "Gol/No Gol",
}

#: Outline colour of the most likely score on the heatmap (readable on light and dark surfaces).
_HIGHLIGHT_COLOR: Final[str] = "#eb6834"
#: Fixed categorical hues of the strength chart: slot 1 (blue) = attack, slot 2 (orange) = defence.
_SERIES_ATTACK: Final[str] = "#2a78d6"
_SERIES_DEFENCE: Final[str] = "#eb6834"


# =============================================================================
# State
# =============================================================================
@dataclass
class PipelineState:
    """Snapshot of one pipeline run, consumed read-only by the dashboard.

    ``source`` tells whether ``matches`` came from the harvested CSVs
    (``'processed'``) or from the synthetic generator (``'simulated'``);
    ``notes`` is the human-readable log of what happened (data source, injected
    trap, fit diagnostics, skipped quotes...) shown in the *Dettagli pipeline*
    expander.

    ``matches`` is everything the fit saw (history included); ``current_matches``
    are its rows of the registry season (``season_label``) and ``history_matches``
    the older ones. ``registry`` is the perimeter: ``fixtures`` are the fixtures
    loaded (filtered or not), while ``predictions``, ``opportunities`` and
    ``traps`` only ever concern current teams and the current season;
    ``skipped_fixtures`` names the fixtures left out (dated in another season,
    historical or unknown teams). ``coverage`` is the per-competition
    current-season coverage the ``use_prior=None`` rule looked at
    (:class:`CoverageReport`, with the registry competitions it ignored because
    they have no match in the data). ``matchdays`` groups the *predicted*
    fixtures into "giornate" (:func:`src.engine.picks.matchdays`): with a real
    provider the fixture list is the whole remaining season, so the match
    selector, the value panel and the picks default to ``matchdays[0]``, the
    next one (:attr:`next_matchday`).
    """

    matches: pd.DataFrame
    fixtures: list[Fixture]
    quotes: list[OddsQuote]
    engine: DixonColesEngine
    params: FittedParams
    predictions: list[MatchPrediction]
    opportunities: list[ValueOpportunity]
    traps: list[RegressionTrap]
    source: Source
    built_at: datetime
    notes: list[str] = field(default_factory=list)
    registry: TeamRegistry | None = None
    season_label: str = ""
    current_matches: pd.DataFrame = field(default_factory=empty_matches_frame)
    history_matches: pd.DataFrame = field(default_factory=empty_matches_frame)
    skipped_fixtures: list[str] = field(default_factory=list)
    coverage: "CoverageReport | None" = None
    matchdays: list[Matchday] = field(default_factory=list)

    @property
    def next_matchday(self) -> Matchday | None:
        """The first upcoming matchday (``matchdays[0]``), ``None`` without predicted fixtures."""
        return self.matchdays[0] if self.matchdays else None

    @property
    def n_matches(self) -> int:
        """Number of played matches used by the fit (current season and history)."""
        return len(self.matches)

    @property
    def n_current_matches(self) -> int:
        """Played matches of the registry season (the perimeter's own evidence)."""
        return len(self.current_matches)

    @property
    def n_history_matches(self) -> int:
        """Played matches of previous seasons used only to sharpen the strengths."""
        return len(self.history_matches)

    @property
    def n_fixtures(self) -> int:
        """Number of upcoming fixtures loaded (predicted or not)."""
        return len(self.fixtures)

    @property
    def n_current_teams(self) -> int:
        """Fitted teams of the perimeter (every fitted team when no registry was used)."""
        return len(self.params.current_teams)

    @property
    def n_historical_teams(self) -> int:
        """Fitted teams outside the perimeter (opponents from past seasons)."""
        return len(self.params.historical_teams)

    @property
    def history_label(self) -> str:
        """Season range of the history, ``'2024/25-2025/26'`` (``''`` without history)."""
        return history_label(self.history_matches)

    @property
    def perimeter_label(self) -> str:
        """The perimeter as the configuration describes it, ``'Serie C 2026/27 Gironi A/B/C + Serie B (80 squadre)'``.

        Derived from the registry (:attr:`TeamRegistry.perimeter_label`); without
        a registry, ``'Stagione <season> (<n> squadre)'`` with the fitted current teams.
        """
        if self.registry is not None:
            return self.registry.perimeter_label
        return f"Stagione {self.season_label} ({self.n_current_teams} squadre)"

    @property
    def perimeter_text(self) -> str:
        """Italian one-liner for badges and subtitles:
        ``'Perimetro: Serie C 2026/27 Gironi A/B/C + Serie B (80 squadre) · 440 partite stagione corrente · 3040 partite storiche (2024/25-2025/26)'``."""
        text = (
            f"Perimetro: {self.perimeter_label} · {self.n_current_matches} partite stagione corrente · "
            f"{self.n_history_matches} partite storiche"
        )
        if self.n_history_matches:
            text += f" ({self.history_label})"
        return text

    @property
    def gironi(self) -> tuple[Girone, ...]:
        """Competitions present in the fitted parameters, in configuration order (``config/competitions.toml``)."""
        return tuple(g for g in Girone if g in self.params.teams_by_girone)

    @property
    def prior_enabled(self) -> bool:
        """Whether the fit used the Gaussian (MAP) prior."""
        cfg = self.engine.config
        return bool(cfg.l2_penalty > 0.0 or cfg.l2_penalty_home > 0.0 or cfg.l2_penalty_rho > 0.0)


# =============================================================================
# Seasons and perimeter helpers
# =============================================================================
def split_by_season(matches: pd.DataFrame, season_label: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """``(current, history)``: rows of ``matches`` whose :func:`season_of` is ``season_label`` and the rest.

    Both frames keep the canonical columns and a fresh index; an empty input
    yields two empty frames. Rows of seasons *after* ``season_label`` count as
    history too (they cannot exist for a current season, so this only matters
    for mistyped dates, which the engine's weight diagnostics report anyway).
    """
    if matches.empty:
        return empty_matches_frame(), empty_matches_frame()
    label = str(season_label).replace("_", "/")
    seasons = pd.to_datetime(matches["date"]).map(season_of)
    is_current = (seasons == label).to_numpy(dtype=bool)
    return matches.loc[is_current].reset_index(drop=True), matches.loc[~is_current].reset_index(drop=True)


def history_label(matches: pd.DataFrame) -> str:
    """``'2024/25-2025/26'`` for the seasons spanned by ``matches`` (one label when a single season, ``''`` when empty)."""
    if matches.empty:
        return ""
    seasons = sorted(set(pd.to_datetime(matches["date"]).map(season_of)))
    return seasons[0] if len(seasons) == 1 else f"{seasons[0]}-{seasons[-1]}"


def perimeter_matches(
    matches: pd.DataFrame,
    *,
    registry: TeamRegistry | None = None,
    season: str | None = None,
) -> pd.DataFrame:
    """Rows of ``matches`` that count as *coverage of the perimeter*.

    With ``season`` only the rows of that season label are kept (see
    :func:`split_by_season`); with ``registry`` only the rows where **both** teams
    are current teams of the registry. With neither the frame is returned
    unchanged, which is the pre-2026/27 behaviour of the callers.
    """
    out = matches
    if season is not None:
        out, _ = split_by_season(out, season)
    if registry is not None and not out.empty:
        current = set(registry.canonical_names)
        keep = out["home_team"].isin(current) & out["away_team"].isin(current)
        out = out.loc[keep.to_numpy(dtype=bool)].reset_index(drop=True)
    return out


def other_season_reason(fixture_season: str, season_label: str) -> str:
    """Italian reason for skipping a fixture dated outside the registry season.

    ``'partita di una stagione precedente (2025/26), fuori dal perimetro 2026/27'``
    for a stale ``fixtures.csv`` of last season; ``successiva`` when the fixture
    is dated after the registry season. Shared by the dashboard and ``main.py``.
    """
    kind = "precedente" if str(fixture_season) < str(season_label) else "successiva"
    return f"partita di una stagione {kind} ({fixture_season}), fuori dal perimetro {season_label}"


# =============================================================================
# Loading helpers
# =============================================================================
def load_odds_quotes(odds_dir: Path) -> list[OddsQuote]:
    """Read bookmaker odds from every ``*.csv`` in ``odds_dir``.

    Format (identical to the one ``main.load_odds_dir`` reads)::

        match_id,bookmaker,market,selection,odds
        SIM-A-16-0451,SimBook,1X2,1,2.10
        SIM-A-16-0451,SimBook,1X2,X,3.30
        SIM-A-16-0451,SimBook,1X2,2,3.60
        SIM-A-16-0451,SimBook,OU2.5,Over 2.5,1.95
        SIM-A-16-0451,SimBook,OU2.5,Under 2.5,1.85

    One row per quoted selection; ``market`` uses the :class:`Market` values
    (``1X2``, ``OU1.5``, ``OU2.5``, ``OU3.5``, ``GG/NG``) and ``selection`` the
    :data:`SELECTIONS` keys of that market. Rows sharing ``(match_id,
    bookmaker, market)`` are grouped into one :class:`OddsQuote`. Files with
    missing columns, unknown markets, incomplete selection sets or odds ``<= 1``
    are skipped with a warning: this function never raises on bad odds. A
    missing directory yields an empty list.
    """
    directory = Path(odds_dir)
    if not directory.is_dir():
        log.info("odds directory not found", extra={"odds_dir": str(directory)})
        return []
    quotes: list[OddsQuote] = []
    skipped = 0
    for csv_path in sorted(directory.glob("*.csv")):
        try:
            frame = pd.read_csv(csv_path, dtype={"match_id": "string", "bookmaker": "string", "market": "string", "selection": "string"})
        except (pd.errors.EmptyDataError, pd.errors.ParserError, OSError, UnicodeDecodeError) as exc:
            log.warning("odds file unreadable, skipped", extra={"file": str(csv_path), "error": str(exc)})
            continue
        missing = [c for c in ODDS_COLUMNS if c not in frame.columns]
        if missing:
            log.warning("odds file skipped: missing columns", extra={"file": str(csv_path), "missing": missing})
            continue
        frame = frame.dropna(subset=["match_id", "bookmaker", "market", "selection"])
        frame = frame.assign(odds=pd.to_numeric(frame["odds"], errors="coerce"))
        for (match_id, bookmaker, market), group in frame.groupby(["match_id", "bookmaker", "market"], sort=False):
            try:
                odds = {str(sel): float(o) for sel, o in zip(group["selection"], group["odds"])}
                if any(math.isnan(o) for o in odds.values()):
                    raise ValueError("non-numeric odds")
                quotes.append(OddsQuote(match_id=str(match_id), bookmaker=str(bookmaker), market=Market(str(market)), odds=odds))
            except ValueError as exc:
                skipped += 1
                log.warning(
                    "odds group skipped",
                    extra={"file": csv_path.name, "match_id": str(match_id), "market": str(market), "error": str(exc)},
                )
    log.info("odds loaded", extra={"odds_dir": str(directory), "quotes": len(quotes), "skipped_groups": skipped})
    return quotes


def _pick_trap_team(matches: pd.DataFrame, fixtures: Sequence[Fixture], girone: Girone, *, window: int = 3) -> str | None:
    """First ``girone`` team that plays in the next round and has >= ``window`` played matches."""
    played = pd.concat([matches["home_team"], matches["away_team"]]).value_counts()
    for fx in sorted(fixtures, key=lambda f: (f.date, f.match_id)):
        if fx.girone != girone:
            continue
        for team in (fx.home_team, fx.away_team):
            if int(played.get(team, 0)) >= window:
                return str(team)
    return None


@dataclass(frozen=True)
class CoverageReport:
    """Current-season coverage of the perimeter, competition by competition.

    ``per_competition`` maps every competition that has **at least one match in
    the frame, in any season**, to the average played current-season matches
    per team of its current teams (``2 * matches / teams``). ``ignored`` lists
    the registry competitions that have no match at all in the frame: they were
    never harvested (Serie B for a SofaScore-only installation, whose provider
    cannot serve it) so they cannot be *covered* and must not drag the minimum
    to zero for ever; the engine likewise estimates no home advantage for them.
    A competition whose matches exist but whose current season has not started
    stays in ``per_competition`` with ``0.0``: that is genuinely thin data.
    Without a registry the competitions are those of the frame and nothing is
    ignored.
    """

    per_competition: Mapping[Girone, float]
    ignored: tuple[Girone, ...] = ()
    season: str | None = None

    @property
    def thinnest(self) -> float:
        """Matches per team of the least-covered competition (``0.0`` when nothing is covered)."""
        return float(min(self.per_competition.values())) if self.per_competition else 0.0

    @property
    def thinnest_competition(self) -> Girone | None:
        """The least-covered competition (first in configuration order on ties), ``None`` when none is covered."""
        best: Girone | None = None
        for g, value in self.per_competition.items():
            if best is None or value < self.per_competition[best]:
                best = g
        return best

    def describe_ignored(self) -> str:
        """Italian fragment naming the ignored competitions, ``''`` when none.

        ``'Serie B senza partite nei dati (ignorata nel calcolo della copertura)'``;
        with several competitions the names are joined with ``' e '`` and the
        participle becomes plural.
        """
        if not self.ignored:
            return ""
        names = " e ".join(g.short for g in self.ignored)
        participle = "ignorata" if len(self.ignored) == 1 else "ignorate"
        return f"{names} senza partite nei dati ({participle} nel calcolo della copertura)"


def coverage_report(
    matches: pd.DataFrame,
    *,
    registry: TeamRegistry | None = None,
    season: str | None = None,
) -> CoverageReport:
    """Per-competition coverage behind :func:`matches_per_team` and :func:`prior_recommended`.

    ``registry``/``season`` restrict the count to the perimeter (see
    :func:`perimeter_matches`): with three seasons in the frame the history is
    large, but the question is how much of the *current* season the current
    teams have played. With a registry every competition the registry holds is
    considered (the Serie C gironi and Serie B alike), except those without a
    single match in ``matches`` in any season, which are reported as
    :attr:`CoverageReport.ignored` instead of counting as zero coverage.
    """
    present = set(matches["girone"].astype(str)) if len(matches) else set()
    frame = perimeter_matches(matches, registry=registry, season=season)
    per_competition: dict[Girone, float] = {}
    ignored: list[Girone] = []
    if registry is not None:
        for g, teams in registry.teams_by_girone.items():
            if not teams:
                continue
            if g.value not in present:
                ignored.append(g)
                continue
            members = set(teams)
            n = int((frame["home_team"].isin(members) | frame["away_team"].isin(members)).sum()) if len(frame) else 0
            per_competition[g] = 2.0 * n / len(members)
    elif len(frame):
        for code, group in frame.groupby("girone", observed=True):
            teams = set(group["home_team"]) | set(group["away_team"])
            per_competition[Girone(str(code))] = 2.0 * len(group) / len(teams)
    if ignored:
        log.debug(
            "registry competitions without matches ignored by the coverage rule",
            extra={"ignored": [g.value for g in ignored], "season": season},
        )
    return CoverageReport(per_competition=per_competition, ignored=tuple(ignored), season=season)


def matches_per_team(
    matches: pd.DataFrame,
    *,
    registry: TeamRegistry | None = None,
    season: str | None = None,
) -> float:
    """Average played matches per team in the least-covered competition (``2 * n / teams``), 0.0 when empty.

    Shorthand for ``coverage_report(...).thinnest``: with a registry, coverage is
    counted for every competition the registry holds that has matches in the
    frame (a competition never harvested is ignored, see
    :class:`CoverageReport`), and a competition with history but no
    current-season match counts as zero coverage.
    """
    return coverage_report(matches, registry=registry, season=season).thinnest


def prior_recommended(
    matches: pd.DataFrame,
    *,
    min_matches_per_team: float = PRIOR_MIN_MATCHES_PER_TEAM,
    registry: TeamRegistry | None = None,
    season: str | None = None,
) -> bool:
    """Auto rule: MAP estimate while some competition has fewer than ``min_matches_per_team`` matches per team.

    A total-match threshold is misleading for an 80-team perimeter (100 matches
    are barely 2 per team), so coverage is measured per team in the thinnest
    competition (each Serie C girone and Serie B on its own). Coverage is counted
    on the **current-season matches of current teams** when ``registry``/``season``
    are given: history never counts, however long it is, and a registry
    competition with no match at all in the frame is left out of the rule
    (:attr:`CoverageReport.ignored`) instead of forcing the prior for ever.
    """
    return matches_per_team(matches, registry=registry, season=season) < min_matches_per_team


def _model_config(
    matches: pd.DataFrame,
    *,
    xi: float,
    use_prior: bool | None,
    registry: TeamRegistry | None = None,
    season: str | None = None,
) -> tuple[DixonColesConfig, bool]:
    """Resolve the ``use_prior`` tri-state into a config; returns ``(config, prior_on)``."""
    prior_on = prior_recommended(matches, registry=registry, season=season) if use_prior is None else bool(use_prior)
    if prior_on:
        return DixonColesConfig.with_gaussian_prior(xi=xi), True
    return DixonColesConfig(xi=xi), False


def _load_processed_inputs(
    processed_dir: Path, odds_dir: Path, notes: list[str]
) -> tuple[pd.DataFrame, list[Fixture], list[OddsQuote]] | None:
    """Load matches/fixtures/odds from disk; ``None`` when there is nothing usable."""
    matches_path = Path(processed_dir) / "matches.csv"
    fixtures_path = Path(processed_dir) / "fixtures.csv"
    try:
        matches = load_processed(matches_path)
    except (ValueError, OSError, pd.errors.ParserError) as exc:
        log.warning("processed matches unreadable", extra={"path": str(matches_path), "error": str(exc)})
        notes.append(f"File {matches_path} non leggibile ({exc}); uso i dati simulati.")
        return None
    if matches.empty:
        notes.append(f"File {matches_path} vuoto; uso i dati simulati.")
        return None
    fixtures: list[Fixture] = []
    if fixtures_path.exists():
        try:
            fixtures = load_fixtures(fixtures_path)
        except (ValueError, OSError, pd.errors.ParserError) as exc:
            log.warning("fixtures unreadable", extra={"path": str(fixtures_path), "error": str(exc)})
            notes.append(f"File {fixtures_path} non leggibile ({exc}): nessuna partita in programma.")
    else:
        notes.append(f"Nessun file {fixtures_path.name}: nessuna partita in programma da prevedere.")
    quotes = load_odds_quotes(Path(odds_dir))
    notes.append(f"Dati reali da {matches_path}: {len(matches)} partite, {len(fixtures)} partite in programma, {len(quotes)} quote da {odds_dir}.")
    if not quotes:
        notes.append("Nessuna quota bookmaker trovata: il pannello EV resta vuoto.")
    return matches, fixtures, quotes


def max_sim_rounds(registry: TeamRegistry | None = None) -> int:
    """Largest number of *played* rounds that still leaves one round of fixtures.

    :func:`simulate_league` plays at most the whole double round-robin schedule
    (38 rounds for 20-team gironi) and uses the round after the last played one
    as the fixture list, so asking for the full season (or more) yields no
    fixtures, no predictions and no odds. The dashboard therefore clamps
    ``n_rounds`` to ``[MIN_SIM_ROUNDS, max_sim_rounds()]`` (``37`` with the
    default registry). ``registry`` defaults to the real 2026/27 composition.
    """
    reg = load_registry() if registry is None else registry
    total_rounds = max(len(round_robin(teams)) for teams in reg.teams_by_girone.values())
    return max(MIN_SIM_ROUNDS, total_rounds - 1)


def simulate_current_season(
    seed: int,
    registry: TeamRegistry,
    *,
    n_rounds: int | None = None,
    as_of: date | None = None,
    notes: list[str] | None = None,
) -> tuple[SimulatedLeague, int]:
    """Simulated current season of ``registry`` on its real calendar; returns ``(league, played_rounds)``.

    Shared by the dashboard fallback and ``main.py --mode full`` so both read
    the calendar the same way. With ``n_rounds`` the played rounds are fixed,
    clamped to ``[MIN_SIM_ROUNDS, max_sim_rounds(registry)]`` so one round of
    fixtures always remains (a clamp is recorded in ``notes``). Otherwise the
    rounds dated before ``as_of`` (today by default) are played and the first
    round on/after it is the fixture list: before the kick-off one round is
    played anyway, after the last round the last one is kept as fixtures, both
    with a note. ``notes`` (Italian, for the UI and the console) is extended in
    place when given.
    """
    notes = [] if notes is None else notes
    upper = max_sim_rounds(registry)
    if n_rounds is not None:
        played = min(max(int(n_rounds), MIN_SIM_ROUNDS), upper)
        if played != n_rounds:
            notes.append(
                f"Giornate simulate richieste: {n_rounds}; valore limitato a {played} (intervallo valido "
                f"{MIN_SIM_ROUNDS}-{upper}) così resta una giornata in programma da prevedere."
            )
            log.warning("simulated rounds clamped", extra={"requested": n_rounds, "used": played, "max": upper})
        return simulate_league(seed=seed, registry=registry, n_rounds=played), played
    today = date.today() if as_of is None else as_of
    league = simulate_league(seed=seed, registry=registry, as_of=today)
    played = int(league.matches["round"].max()) if not league.matches.empty else 0
    if played < MIN_SIM_ROUNDS:
        notes.append(
            f"Calendario simulato al {today.isoformat()}: la stagione {registry.season_label} non è ancora iniziata; "
            f"simulo comunque {MIN_SIM_ROUNDS} giornata così il modello ha dati."
        )
        played = MIN_SIM_ROUNDS
        league = simulate_league(seed=seed, registry=registry, n_rounds=played)
    elif not league.fixtures:
        notes.append(
            f"Calendario simulato al {today.isoformat()}: la stagione {registry.season_label} è conclusa; "
            f"tengo l'ultima giornata come partite in programma ({upper} giornate giocate)."
        )
        played = upper
        league = simulate_league(seed=seed, registry=registry, n_rounds=played)
    else:
        notes.append(
            f"Calendario simulato al {today.isoformat()}: giornate con data precedente giocate ({played}), "
            f"la prima giornata in data uguale o successiva ({league.fixtures[0].date.isoformat()}) è il turno da prevedere."
        )
    log.info(
        "simulated current season positioned on the calendar",
        extra={"season": registry.season_label, "as_of": today.isoformat(), "played_rounds": played, "n_fixtures": len(league.fixtures)},
    )
    return league, played


def _simulated_inputs(
    seed: int,
    n_rounds: int | None,
    notes: list[str],
    *,
    registry: TeamRegistry | None = None,
    history_seasons: int = DEFAULT_HISTORY_SEASONS,
    as_of: date | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, list[Fixture], list[OddsQuote]]:
    """Synthetic league on the real calendar plus synthetic history; returns ``(current, history, fixtures, quotes)``.

    The current season is :func:`simulate_current_season` on ``registry``
    (2026/27 by default): with ``n_rounds`` the played rounds are fixed and
    clamped, otherwise they follow the calendar as of ``as_of`` (today by
    default). ``history_seasons`` (0-2) synthetic seasons *before* the registry
    season (:func:`history_seasons_for`) from :func:`simulate_history` share the
    same true parameters. A REGRESSION_TRAP is injected on a Girone A team of
    the next round (current season only).
    """
    reg = load_registry() if registry is None else registry
    league, played = simulate_current_season(seed, reg, n_rounds=n_rounds, as_of=as_of, notes=notes)
    matches = league.matches
    notes.append(
        f"Dataset simulato (seed={seed}, {played} giornate giocate): {len(matches)} partite della stagione {reg.season_label}, "
        f"{len(league.fixtures)} partite in programma, {len(league.odds)} quote del bookmaker sintetico."
    )
    history = empty_matches_frame()
    if history_seasons > 0:
        past = simulate_history(seed=seed, registry=reg, seasons=history_seasons_for(reg, int(history_seasons)))
        history = past.matches
        swapped = sorted(set(past.teams) - set(reg.canonical_names))
        notes.append(
            f"Storico simulato: {len(history)} partite delle stagioni {', '.join(past.seasons)} con {len(swapped)} squadre "
            f"non più nel perimetro {reg.season_label} ({', '.join(swapped[:6])}{', ...' if len(swapped) > 6 else ''}): "
            "usate solo per stimare meglio le forze, escluse da classifica, previsioni e valore."
        )
    # the showcase trap goes to the first competition of the registry (Girone A with the default one)
    trap_girone = reg.gironi[0] if reg.gironi else next(iter(Girone))
    trap_team = _pick_trap_team(matches, league.fixtures, trap_girone)
    if trap_team is not None:
        matches = inject_regression_trap(matches, trap_team)
        notes.append(
            f"Trappola di regressione iniettata su {trap_team} ({trap_girone.short}): ultime 3 gare riscritte con "
            f"1 gol subito contro 4.20 xGA, così il filtro REGRESSION_TRAP compare nel pannello valore."
        )
        log.info("regression trap injected", extra={"team": trap_team, "girone": trap_girone.value})
    else:
        notes.append(f"Nessuna squadra di {trap_girone.short} con 3 gare giocate e una partita in programma: trappola non iniettata.")
    return matches, history, league.fixtures, league.odds


# =============================================================================
# Pipeline
# =============================================================================
def build_state(
    *,
    processed_dir: Path = PROCESSED_DIR,
    odds_dir: Path = ODDS_DIR,
    min_ev: float = 0.035,
    kelly_fraction: float = 0.125,
    bankroll_units: float = 100.0,
    xi: float = 0.0018,
    seed: int = 7,
    n_rounds: int | None = None,
    simulate: bool | None = None,
    use_prior: bool | None = None,
    season: str = DEFAULT_SEASON,
    history_seasons: int = DEFAULT_HISTORY_SEASONS,
    as_of: date | None = None,
    registry: TeamRegistry | None = None,
) -> PipelineState:
    """Run load/simulate -> fit -> predict -> value and return a :class:`PipelineState`.

    Perimeter
        ``season`` selects the registry (``config/teams_<season>.json``, 2026/27
        by default; ``registry`` overrides it, e.g. for a simulated composition).
        Its teams -- in every competition it holds, the three Serie C gironi and
        Serie B with the shipped files -- are the only ones ranked, predicted and
        priced: the fit sees every match of the frame, but teams outside the
        registry are fitted as *historical* opponents only. The rows of the
        registry season are the current matches; older rows are history.
    Data source
        ``processed_dir/matches.csv`` is used when it exists and ``simulate`` is
        not ``True`` (``fixtures.csv`` alongside it when present, odds from
        ``odds_dir/*.csv`` in the :func:`load_odds_quotes` format); it may hold
        several seasons. Otherwise a league is simulated with
        :func:`simulate_current_season` on the registry season's real calendar
        (2026/27 by default) -- with ``n_rounds`` the played rounds are fixed,
        with ``n_rounds=None`` they are the rounds dated before ``as_of`` (today
        by default) and the next round is the fixture list -- plus
        ``history_seasons`` (0-2) synthetic seasons preceding it from
        :func:`simulate_history`; a REGRESSION_TRAP is injected on one Girone A
        team of the next round.
    Model
        ``use_prior=None`` enables the Gaussian prior while the least-covered
        girone has fewer than :data:`PRIOR_MIN_MATCHES_PER_TEAM` played
        *current-season* matches per current team (see
        :func:`prior_recommended`; history never counts as coverage); ``True``
        forces :meth:`DixonColesConfig.with_gaussian_prior`, ``False`` the plain
        MLE. ``xi`` is the time-decay rate per day.
    Value
        ``ValueEngine(min_ev, KellyConfig(kelly_fraction, bankroll_units))`` is
        run on the predictions and the quotes; without quotes the
        opportunities simply stay empty. Fixtures dated in another season than
        the registry's (a stale ``fixtures.csv`` of last season), with a
        historical team or with a team unknown to the fit are skipped with a
        note instead of raising, so Match & Heatmap and the value panel only
        ever show current-season matches between current teams; traps are
        detected on the current-season matches of current teams only.
    Validation
        ``seed`` must be ``>= 0`` (``numpy`` rejects negative seeds),
        ``n_rounds`` (when given) ``>= 1`` and ``history_seasons`` in
        ``[0, MAX_HISTORY_SEASONS]``; all raise :class:`ValueError` up front.
        Values of ``n_rounds`` above :func:`max_sim_rounds` are clamped with a
        note, not rejected, so a "full season" request still produces fixtures.
    """
    if seed < 0:
        raise ValueError(f"seed must be >= 0 (got {seed})")
    if n_rounds is not None and n_rounds < MIN_SIM_ROUNDS:
        raise ValueError(f"n_rounds must be >= {MIN_SIM_ROUNDS} (got {n_rounds})")
    if not 0 <= int(history_seasons) <= MAX_HISTORY_SEASONS:
        raise ValueError(f"history_seasons must be in [0, {MAX_HISTORY_SEASONS}] (got {history_seasons})")
    reg = load_registry(season) if registry is None else registry
    season_label = reg.season_label
    notes: list[str] = []
    started = time.perf_counter()
    loaded: tuple[pd.DataFrame, list[Fixture], list[OddsQuote]] | None = None
    if simulate is not True:
        matches_path = Path(processed_dir) / "matches.csv"
        if matches_path.exists():
            loaded = _load_processed_inputs(Path(processed_dir), Path(odds_dir), notes)
        elif simulate is False:
            notes.append(f"Richiesti dati reali ma {matches_path} non esiste: uso i dati simulati.")
            log.warning("processed matches requested but missing", extra={"path": str(matches_path)})
    if loaded is not None:
        source: Source = "processed"
        all_matches, fixtures, quotes = loaded
        current, history = split_by_season(all_matches, season_label)
    else:
        source = "simulated"
        current, history, fixtures, quotes = _simulated_inputs(
            seed, n_rounds, notes, registry=reg, history_seasons=int(history_seasons), as_of=as_of
        )
    matches = validate_matches_df(pd.concat([history, current], ignore_index=True)) if not history.empty else current
    if matches.empty:
        raise ValueError(f"no played matches available for the {season_label} perimeter")

    coverage = coverage_report(current, registry=reg, season=season_label)
    config, prior_on = _model_config(current, xi=xi, use_prior=use_prior, registry=reg, season=season_label)
    engine = DixonColesEngine(config)
    params = engine.fit(matches, registry=reg)
    hist_label = history_label(history)
    notes.append(
        f"Perimetro: {reg.perimeter_label}, {len(params.current_teams)} squadre con partite nei dati; "
        f"{len(current)} partite della stagione corrente"
        + (f" e {len(history)} partite storiche ({hist_label})" if len(history) else " e nessuna partita storica")
        + f"; {len(params.historical_teams)} squadre storiche stimate solo come avversarie"
        + (f" ({', '.join(params.historical_teams[:6])}{', ...' if len(params.historical_teams) > 6 else ''})" if params.historical_teams else "")
        + "."
    )
    without_matches = sorted(set(reg.canonical_names) - set(params.current_teams))
    if without_matches:
        notes.append(
            f"{len(without_matches)} squadre del dizionario {season_label} senza partite nei dati (non stimate): "
            f"{', '.join(without_matches[:8])}{', ...' if len(without_matches) > 8 else ''}."
        )
    notes.append(
        f"Stima Dixon-Coles: metodo {params.method}, convergenza {'sì' if params.converged else 'no'} "
        f"({params.message}), log-likelihood {params.log_likelihood:.2f}, {params.n_iter} iterazioni in "
        f"{params.fit_seconds:.2f} s, xi={params.xi}, prior gaussiana {'attiva' if prior_on else 'disattivata'}"
        + (f" (penalità {params.penalty:.3f})" if prior_on else "")
        + f" (copertura {coverage.thinnest:.1f} partite per squadra nel girone meno coperto"
        + (f"; {coverage.describe_ignored()}" if coverage.ignored else "")
        + f"), rho={params.rho:+.4f}."
    )
    log.info(
        "dashboard fit",
        extra={
            "source": source,
            "season": season_label,
            "n_matches": params.n_matches,
            "n_current_matches": len(current),
            "n_history_matches": len(history),
            "n_current_teams": len(params.current_teams),
            "n_historical_teams": len(params.historical_teams),
            "method": params.method,
            "converged": params.converged,
            "log_likelihood": params.log_likelihood,
            "fit_seconds": params.fit_seconds,
            "prior": prior_on,
        },
    )

    predictions: list[MatchPrediction] = []
    skipped_unknown: list[str] = []
    skipped_historical: list[str] = []
    skipped_season: list[str] = []
    other_seasons: set[str] = set()
    for fx in fixtures:
        fixture_season = season_of(fx.date)
        if fixture_season != season_label:
            skipped_season.append(fx.label)
            other_seasons.add(fixture_season)
            log.warning(
                "fixture of another season skipped",
                extra={"match_id": fx.match_id, "date": fx.date.isoformat(), "fixture_season": fixture_season, "season": season_label},
            )
            continue
        outside = [t for t in (fx.home_team, fx.away_team) if t in params.alpha and not params.is_current(t)]
        if outside:
            skipped_historical.append(fx.label)
            log.warning(
                "fixture outside the perimeter skipped",
                extra={"match_id": fx.match_id, "season": season_label, "historical_teams": outside},
            )
            continue
        try:
            predictions.append(engine.predict(fx))
        except (KeyError, ValueError) as exc:
            skipped_unknown.append(fx.label)
            log.warning("fixture not predictable", extra={"match_id": fx.match_id, "error": str(exc)})
    if skipped_season:
        reasons = "; ".join(other_season_reason(s, season_label) for s in sorted(other_seasons))
        notes.append(
            f"{len(skipped_season)} partite in programma saltate ({reasons}): "
            f"{', '.join(skipped_season[:5])}" + (" ..." if len(skipped_season) > 5 else "") + "."
        )
    if skipped_historical:
        notes.append(
            f"{len(skipped_historical)} partite in programma saltate (squadre fuori dal perimetro {season_label}): "
            f"{', '.join(skipped_historical[:5])}" + (" ..." if len(skipped_historical) > 5 else "") + "."
        )
    if skipped_unknown:
        notes.append(f"{len(skipped_unknown)} partite in programma saltate (squadre non presenti nello storico): {', '.join(skipped_unknown[:5])}" + (" ..." if len(skipped_unknown) > 5 else "") + ".")

    value_engine = ValueEngine(min_ev=min_ev, kelly=KellyConfig(fraction=kelly_fraction, bankroll_units=bankroll_units))
    traps = [
        t
        for t in detect_regression_traps(
            current,
            window=value_engine.trap_window,
            max_goals_against=value_engine.trap_max_goals_against,
            min_xg_against=value_engine.trap_min_xg_against,
        )
        if params.is_current(t.team)
    ] if not current.empty else []
    opportunities = value_engine.evaluate(predictions, quotes, matches=current, traps=traps)
    n_value = sum(1 for o in opportunities if o.value)
    n_trap_flags = sum(1 for o in opportunities if o.is_trap)
    notes.append(
        f"Motore valore: {len(opportunities)} selezioni valutate, {n_value} sopra soglia EV {min_ev:.1%}, "
        f"Kelly x{kelly_fraction:g} su {bankroll_units:g} unità, {len(traps)} squadre in trappola di regressione "
        f"(stagione {season_label}), {n_trap_flags} selezioni Over/GG segnalate."
    )
    giornate = matchdays(p.fixture for p in predictions)
    if giornate:
        first = giornate[0]
        notes.append(
            f"Partite in programma raggruppate in {len(giornate)} {'giornata' if len(giornate) == 1 else 'giornate'}; "
            f"la prossima è {first.label} ({first.n_fixtures} partite): selettore partite, pannello valore e selezioni "
            "di giornata la usano come default."
        )
    log.info(
        "matchdays grouped",
        extra={"n_matchdays": len(giornate), "n_predicted": len(predictions), "next": giornate[0].key if giornate else None},
    )
    notes.append(f"Pipeline completata in {time.perf_counter() - started:.2f} s.")
    return PipelineState(
        matches=matches,
        fixtures=list(fixtures),
        quotes=list(quotes),
        engine=engine,
        params=params,
        predictions=predictions,
        opportunities=list(opportunities),
        traps=list(traps),
        source=source,
        built_at=datetime.now(UTC),
        notes=notes,
        registry=reg,
        season_label=season_label,
        current_matches=current,
        history_matches=history,
        skipped_fixtures=skipped_season + skipped_historical + skipped_unknown,
        coverage=coverage,
        matchdays=giornate,
    )


# =============================================================================
# Views
# =============================================================================
def _as_girone(girone: Girone | str | None) -> Girone | None:
    return None if girone is None else Girone.parse(girone)


def competition_label(girone: Girone | str) -> str:
    """Short label shown in the tables for a competition code: ``'A'`` -> ``'Girone A'``, ``'SB'`` -> ``'Serie B'``."""
    return Girone.parse(girone).short


def ranking_table(state: PipelineState, girone: Girone | None) -> pd.DataFrame:
    """Strength ranking (alpha desc) with Italian labels, rounded to 3 decimals.

    Only the current teams of the perimeter are listed and ``Competizione`` is
    the short label of their registry competition (historical opponents never
    appear). ``girone=None`` returns every current team; a competition with no
    fitted teams yields an empty frame with the same columns instead of raising.
    """
    g = _as_girone(girone)
    if g is not None and g not in state.params.teams_by_girone:
        return pd.DataFrame({c: pd.Series(dtype="string" if c in ("Squadra", "Competizione") else "float64") for c in RANKING_COLUMNS})
    frame = state.engine.ranking(g)[list(_RANKING_RENAME)].rename(columns=_RANKING_RENAME)
    numeric = [c for c in RANKING_COLUMNS if c not in ("Squadra", "Competizione")]
    frame = frame.assign(**{c: frame[c].astype("float64").round(3) for c in numeric})
    frame = frame.assign(
        Squadra=frame["Squadra"].astype("string"),
        Competizione=frame["Competizione"].astype(str).map(competition_label).astype("string"),
    )
    return frame[list(RANKING_COLUMNS)].reset_index(drop=True)


def find_matchday(state: PipelineState, key: str | None) -> Matchday | None:
    """The matchday of ``state`` with that :attr:`Matchday.key`, ``None`` when unknown (or ``key`` is ``None``)."""
    if key is None:
        return None
    return next((m for m in state.matchdays if m.key == key), None)


def resolve_matchday(state: PipelineState, matchday: MatchdaySelector | str = None) -> Matchday | None:
    """Turn a selector into a :class:`Matchday` (or ``None`` meaning "no restriction").

    ``None`` -> the next matchday (``None`` when the state has none, which then
    means every prediction, i.e. none); :data:`ALL_MATCHDAYS` -> ``None`` (no
    restriction); a :class:`Matchday` -> itself; a key string -> the matchday
    with that key, falling back to the next one when the key is stale.
    """
    if matchday is ALL_MATCHDAYS or (isinstance(matchday, str) and matchday == ALL_MATCHDAYS):
        return None
    if matchday is None:
        return state.next_matchday
    if isinstance(matchday, Matchday):
        return matchday
    found = find_matchday(state, str(matchday))
    if found is None:
        log.debug("unknown matchday key, using the next matchday", extra={"key": str(matchday)})
        return state.next_matchday
    return found


def _restrict(state: PipelineState, matchday: MatchdaySelector) -> list[MatchPrediction]:
    """Predictions of the selected matchday (every prediction for :data:`ALL_MATCHDAYS`).

    With ``None`` and no matchday in the state (no predicted fixture) there is
    nothing to show, so the result is empty, consistent with the selectors.
    """
    if matchday is ALL_MATCHDAYS or (isinstance(matchday, str) and matchday == ALL_MATCHDAYS):
        return list(state.predictions)
    chosen = resolve_matchday(state, matchday)
    if chosen is None:
        return []
    return [p for p in state.predictions if chosen.contains(p)]


def matchday_predictions(state: PipelineState, matchday: MatchdaySelector = None) -> list[MatchPrediction]:
    """Predictions of one matchday (the next by default, all with :data:`ALL_MATCHDAYS`), sorted by date, competition, label."""
    preds = _restrict(state, matchday)
    order = list(Girone)
    return sorted(preds, key=lambda p: (p.fixture.date, order.index(p.fixture.girone), p.fixture.label, p.fixture.match_id))


def fixtures_for(state: PipelineState, girone: Girone | None, matchday: MatchdaySelector = None) -> list[MatchPrediction]:
    """Predictions of the upcoming fixtures (optionally one girone), sorted by date then label.

    ``matchday`` defaults to the **next** matchday, so with a season-long fixture
    list the match selector shows one "giornata" instead of hundreds of
    fixtures; pass a :class:`Matchday` to pick another one or
    :data:`ALL_MATCHDAYS` for every predicted fixture (the exports do).
    """
    g = _as_girone(girone)
    preds = [p for p in _restrict(state, matchday) if g is None or p.fixture.girone == g]
    return sorted(preds, key=lambda p: (p.fixture.date, p.fixture.label))


def quotes_for(state: PipelineState, match_id: str) -> list[OddsQuote]:
    """Every bookmaker quote of one fixture."""
    return [q for q in state.quotes if q.match_id == match_id]


def no_odds_message(state: PipelineState) -> str:
    """Italian hint shown by the value panel when ``state.quotes`` is empty.

    The text depends on the data source: with harvested data the user can add
    odds CSVs to the odds directory; with the synthetic generator the odds come
    from the simulated bookmaker, which quotes only the upcoming fixtures, so
    the only way to have no quotes is to have no fixtures left.
    """
    if state.source == "processed":
        try:
            odds_dir = ODDS_DIR.relative_to(ROOT).as_posix()
        except ValueError:
            odds_dir = str(ODDS_DIR)
        return (
            f"Nessuna quota bookmaker disponibile: aggiungi file CSV in {odds_dir} "
            "(colonne match_id, bookmaker, market, selection, odds) per calcolare l'EV."
        )
    if not state.fixtures:
        return (
            "Nessuna quota: il bookmaker sintetico quota solo le partite in programma e il dataset simulato "
            "non ne contiene (tutte le giornate risultano già giocate). Riduci le giornate simulate."
        )
    return "Nessuna quota del bookmaker sintetico per le partite in programma: il pannello EV resta vuoto."


def _fold_matrix(matrix: np.ndarray, n: int) -> np.ndarray:
    """``n x n`` copy of a score matrix whose last row/column collect the mass beyond ``n - 1`` goals.

    ``n`` must be in ``[1, matrix.shape[0]]``. Cell ``(i, j)`` with ``i, j < n - 1``
    is ``matrix[i, j]``; row ``n - 1`` holds ``P(home >= n - 1, away = j)``,
    column ``n - 1`` holds ``P(home = i, away >= n - 1)`` and the corner holds
    ``P(home >= n - 1, away >= n - 1)``, so the result sums to the same total
    as ``matrix``. With ``n == matrix.shape[0]`` it is a plain copy.
    """
    m = np.asarray(matrix, dtype=float)
    if not 1 <= n <= m.shape[0]:
        raise ValueError(f"n must be in [1, {m.shape[0]}], got {n}")
    folded = m[:n, :n].copy()
    last = n - 1
    folded[last, :last] = m[last:, :last].sum(axis=0)
    folded[:last, last] = m[:last, last:].sum(axis=1)
    folded[last, last] = m[last:, last:].sum()
    return folded


def heatmap_figure(prediction: MatchPrediction, max_goals_shown: int | None = None) -> go.Figure:
    """Score-matrix heatmap: home goals on the y axis, away goals on the x axis.

    Exactly one :class:`plotly.graph_objects.Heatmap` trace holding the score
    matrix in percentages, annotated with one decimal, single-hue ``Blues``
    colourscale, the most likely score outlined with a rectangle. By default
    (``max_goals_shown=None``) the **whole** ``prediction.matrix`` is drawn
    (``11 x 11`` for the engine default ``max_goals=10``), so the cells add up
    to 100%. A smaller ``max_goals_shown`` gives a compact
    ``(max_goals_shown + 1)^2`` view whose last row and column, labelled
    ``N+``, collect the probability of ``N`` or more goals: the displayed cells
    still sum to 100% instead of silently dropping the tail (the outline is
    skipped when the most likely score falls into an aggregated cell). Values
    at or above the matrix size draw the full matrix; negative values raise
    :class:`ValueError`. The ``plotly`` template is used so Streamlit can
    re-theme it for light and dark modes; fixed height 520.
    """
    full = int(prediction.matrix.shape[0])
    if max_goals_shown is None:
        n = full
    elif max_goals_shown < 0:
        raise ValueError("max_goals_shown must be >= 0")
    else:
        n = min(int(max_goals_shown) + 1, full)
    folded = n < full
    fx = prediction.fixture
    z = _fold_matrix(prediction.matrix, n) * 100.0
    goals = list(range(n))
    labels = [str(g) for g in goals]
    if folded:
        labels[-1] = f"{n - 1}+"
    text = [[f"{z[i, j]:.1f}%" for j in goals] for i in goals]
    hover = [
        [f"{fx.home_team} {labels[i]} - {labels[j]} {fx.away_team}<br>P = {z[i, j]:.2f}%" for j in goals] for i in goals
    ]
    heat = go.Heatmap(
        z=z,
        x=goals,
        y=goals,
        text=text,
        texttemplate="%{text}",
        textfont={"size": 12},
        colorscale="Blues",
        zmin=0.0,
        zmax=float(z.max()) if z.size else 1.0,
        colorbar={"title": {"text": "%"}, "ticksuffix": "%", "thickness": 14},
        hovertext=hover,
        hovertemplate="%{hovertext}<extra></extra>",
        xgap=2,
        ygap=2,
    )
    fig = go.Figure(data=[heat])
    home_goals, away_goals, p_top = prediction.most_likely_score()
    exact = n - 1 if folded else n  # indices below this are single scores, not aggregated tails
    if home_goals < exact and away_goals < exact:
        fig.add_shape(
            type="rect",
            x0=away_goals - 0.5,
            x1=away_goals + 0.5,
            y0=home_goals - 0.5,
            y1=home_goals + 0.5,
            line={"color": _HIGHLIGHT_COLOR, "width": 3},
            fillcolor="rgba(0,0,0,0)",
        )
    subtitle = (
        f"λ casa = {prediction.home_rate:.2f} · μ ospiti = {prediction.away_rate:.2f} · "
        f"risultato più probabile {home_goals}-{away_goals} ({p_top:.1%})"
    )
    if folded:
        subtitle += f" · {n - 1}+ = {n - 1} o più gol"
    fig.update_layout(
        template="plotly",
        title={"text": f"{fx.label} · {fx.girone.short} · {fx.date.isoformat()}<br><sup>{subtitle}</sup>"},
        xaxis={
            "title": {"text": f"Gol {fx.away_team}"},
            "tickmode": "array",
            "tickvals": goals,
            "ticktext": labels,
            "side": "top",
            "showgrid": False,
        },
        yaxis={
            "title": {"text": f"Gol {fx.home_team}"},
            "tickmode": "array",
            "tickvals": goals,
            "ticktext": labels,
            "autorange": "reversed",
            "showgrid": False,
        },
        height=520,
        margin={"l": 70, "r": 30, "t": 110, "b": 40},
    )
    return fig


def strength_figure(ranking: pd.DataFrame, title: str = "") -> go.Figure:
    """Horizontal grouped bars of alpha (attack) and beta (defence) per team.

    ``ranking`` is a :func:`ranking_table` frame. Teams are ordered by alpha
    (strongest attack at the top); two fixed hues (blue = attack, orange =
    defence) with a legend, thin bars and a hairline grid. The height grows
    with the number of teams so no team label is dropped.
    """
    frame = ranking.sort_values("Forza Offensiva (α)", ascending=True, kind="stable")
    teams = frame["Squadra"].astype(str).tolist()
    fig = go.Figure(
        data=[
            go.Bar(
                name="Forza offensiva (α)",
                y=teams,
                x=frame["Forza Offensiva (α)"].to_numpy(dtype=float),
                orientation="h",
                marker={"color": _SERIES_ATTACK, "line": {"width": 0}},
                hovertemplate="%{y}<br>α = %{x:+.3f}<extra>attacco</extra>",
            ),
            go.Bar(
                name="Solidità difensiva (β)",
                y=teams,
                x=frame["Solidità Difensiva (β)"].to_numpy(dtype=float),
                orientation="h",
                marker={"color": _SERIES_DEFENCE, "line": {"width": 0}},
                hovertemplate="%{y}<br>β = %{x:+.3f}<extra>difesa</extra>",
            ),
        ]
    )
    n_teams = max(len(teams), 1)
    fig.update_layout(
        template="plotly",
        title={"text": title} if title else None,
        barmode="group",
        bargap=0.35,
        bargroupgap=0.08,
        height=int(max(360, 30 * n_teams + 140)),
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.0, "xanchor": "left", "x": 0.0, "title": None},
        xaxis={"title": {"text": "Parametro (scala log, 0 = media del girone)"}, "zeroline": True, "zerolinewidth": 1, "gridwidth": 1},
        yaxis={"title": None, "tickfont": {"size": 11}, "showgrid": False},
        margin={"l": 10, "r": 20, "t": 70 if title else 50, "b": 50},
    )
    return fig


def _book_lookup(quotes: Iterable[OddsQuote]) -> dict[tuple[Market, str], float]:
    """Best (highest) bookmaker price per ``(market, selection)``."""
    best: dict[tuple[Market, str], float] = {}
    for q in quotes:
        for sel, odds in q.odds.items():
            key = (q.market, sel)
            if key not in best or odds > best[key]:
                best[key] = float(odds)
    return best


def markets_table(prediction: MatchPrediction, quotes: list[OddsQuote]) -> pd.DataFrame:
    """One row per selection of every market: model probability, fair odds, book odds and EV.

    ``Quota Book``/``EV (%)`` are ``NaN`` when no bookmaker quotes the selection;
    with several bookmakers the best price is used.
    """
    best = _book_lookup(q for q in quotes if q.match_id == prediction.match_id)
    rows: list[dict[str, object]] = []
    for market in Market:
        probs = prediction.probabilities.get(market)
        fair = prediction.fair_odds.get(market, {})
        if probs is None:
            continue
        for sel in SELECTIONS[market]:
            p = float(probs[sel])
            book = best.get((market, sel), float("nan"))
            ev = (p * book - 1.0) * 100.0 if not math.isnan(book) else float("nan")
            rows.append(
                {
                    "Mercato": MARKET_LABELS[market],
                    "Esito": sel,
                    "Probabilità (%)": round(p * 100.0, 1),
                    "Quota Fair": round(float(fair.get(sel, 1.0 / p if p > 0 else float("inf"))), 2),
                    "Quota Book": round(book, 2) if not math.isnan(book) else float("nan"),
                    "EV (%)": round(ev, 2) if not math.isnan(ev) else float("nan"),
                }
            )
    frame = pd.DataFrame(rows, columns=list(MARKETS_COLUMNS))
    frame = frame.assign(Mercato=frame["Mercato"].astype("string"), Esito=frame["Esito"].astype("string"))
    for c in ("Probabilità (%)", "Quota Fair", "Quota Book", "EV (%)"):
        frame[c] = frame[c].astype("float64")
    return frame


def value_opportunities(
    state: PipelineState,
    min_ev: float,
    girone: Girone | None = None,
    matchday: MatchdaySelector = None,
) -> list[ValueOpportunity]:
    """The rows of the value panel as objects: ``ev >= min_ev``, one competition when given, one matchday.

    Exactly the selection and the order of :func:`value_table` (EV descending,
    then match label, market and selection), so a caller that needs the panel's
    first row -- the worked example of the *Legenda* tab -- reads it from here
    instead of re-filtering ``state.opportunities``. ``matchday`` defaults to
    the next matchday like :func:`fixtures_for`; :data:`ALL_MATCHDAYS` lifts the
    restriction.
    """
    g = _as_girone(girone)
    allowed = {p.fixture.match_id for p in _restrict(state, matchday)}
    rows = [o for o in state.opportunities if o.ev >= min_ev and (g is None or o.girone == g) and o.match_id in allowed]
    # the table sorts on the EV rounded to 2 decimals in percent; do the same so the two orders agree
    rows.sort(key=lambda o: (-round(o.ev * 100.0, 2), f"{o.home_team} - {o.away_team}", MARKET_LABELS[o.market], o.selection))
    return rows


def value_table(
    state: PipelineState,
    min_ev: float,
    girone: Girone | None = None,
    matchday: MatchdaySelector = None,
) -> pd.DataFrame:
    """Selections with ``ev >= min_ev`` (optionally one competition), EV descending, Italian labels.

    Like :func:`fixtures_for`, ``matchday`` defaults to the next matchday; pass
    :data:`ALL_MATCHDAYS` to list the value selections of the whole fixture list.
    The rows are those of :func:`value_opportunities`, in the same order.
    """
    rows = [
        {
            "Data": pd.Timestamp(o.date),
            "Competizione": o.girone.short,
            "Partita": f"{o.home_team} - {o.away_team}",
            "Mercato": MARKET_LABELS[o.market],
            "Esito": o.selection,
            "Bookmaker": o.bookmaker,
            "Quota Book": round(o.book_odds, 2),
            "Quota Fair": round(o.fair_odds, 2),
            "P. Modello (%)": round(o.p_model * 100.0, 1),
            "EV (%)": round(o.ev * 100.0, 2),
            "Stake (unità)": round(o.stake_units, 1),
            "Flag": " | ".join(o.flags),
        }
        for o in value_opportunities(state, min_ev, girone, matchday)
    ]
    frame = pd.DataFrame(rows, columns=list(VALUE_COLUMNS))
    frame = frame.assign(Data=pd.to_datetime(frame["Data"]).astype("datetime64[ns]"))
    for c in ("Competizione", "Partita", "Mercato", "Esito", "Bookmaker", "Flag"):
        frame[c] = frame[c].astype("string")
    for c in ("Quota Book", "Quota Fair", "P. Modello (%)", "EV (%)", "Stake (unità)"):
        frame[c] = frame[c].astype("float64")
    frame = frame.sort_values(["EV (%)", "Partita", "Mercato", "Esito"], ascending=[False, True, True, True], kind="stable")
    return frame.reset_index(drop=True)


def traps_table(state: PipelineState) -> pd.DataFrame:
    """REGRESSION_TRAP teams (severity descending) with Italian labels."""
    rows = [
        {
            "Squadra": t.team,
            "Competizione": t.girone.short,
            "Gol Subiti": int(t.goals_against),
            "xGA": round(t.xg_against, 2),
            "Severità": round(t.severity, 2),
            "Ultima Gara": pd.Timestamp(t.last_date),
        }
        for t in sorted(state.traps, key=lambda t: (-t.severity, t.team))
    ]
    frame = pd.DataFrame(rows, columns=list(TRAPS_COLUMNS))
    frame = frame.assign(
        Squadra=frame["Squadra"].astype("string"),
        Competizione=frame["Competizione"].astype("string"),
        **{"Gol Subiti": frame["Gol Subiti"].astype("int64"), "Ultima Gara": pd.to_datetime(frame["Ultima Gara"]).astype("datetime64[ns]")},
    )
    for c in ("xGA", "Severità"):
        frame[c] = frame[c].astype("float64")
    return frame.reset_index(drop=True)


def _slug(selection: str) -> str:
    return selection.strip().lower().replace(" ", "_")


def predictions_table(state: PipelineState) -> pd.DataFrame:
    """Export frame: one row per predicted fixture with rates, every market probability and fair odds.

    Columns: ``match_id, date, round, girone, home_team, away_team, lambda, mu, rho,
    most_likely_score, p_most_likely`` followed by ``p_<selection>`` and
    ``fair_<selection>`` for every selection in :data:`SELECTIONS`
    (``p_1, p_x, p_2, p_over_1.5, ...``).
    """
    rows: list[dict[str, object]] = []
    for p in fixtures_for(state, None, matchday=ALL_MATCHDAYS):
        fx = p.fixture
        hs, as_, p_top = p.most_likely_score()
        row: dict[str, object] = {
            "match_id": fx.match_id,
            "date": pd.Timestamp(fx.date),
            "round": int(fx.round),
            "girone": fx.girone.value,
            "home_team": fx.home_team,
            "away_team": fx.away_team,
            "lambda": round(p.home_rate, 4),
            "mu": round(p.away_rate, 4),
            "rho": round(p.rho, 4),
            "most_likely_score": f"{hs}-{as_}",
            "p_most_likely": round(p_top, 4),
        }
        for market in Market:
            for sel in SELECTIONS[market]:
                row[f"p_{_slug(sel)}"] = round(float(p.probabilities[market][sel]), 4)
                row[f"fair_{_slug(sel)}"] = round(float(p.fair_odds[market][sel]), 3)
        rows.append(row)
    columns = ["match_id", "date", "round", "girone", "home_team", "away_team", "lambda", "mu", "rho", "most_likely_score", "p_most_likely"]
    for market in Market:
        for sel in SELECTIONS[market]:
            columns += [f"p_{_slug(sel)}", f"fair_{_slug(sel)}"]
    frame = pd.DataFrame(rows, columns=columns)
    for c in ("match_id", "girone", "home_team", "away_team", "most_likely_score"):
        frame[c] = frame[c].astype("string")
    frame = frame.assign(date=pd.to_datetime(frame["date"]).astype("datetime64[ns]"), round=frame["round"].astype("int64"))
    return frame.reset_index(drop=True)


def opportunities_table(state: PipelineState) -> pd.DataFrame:
    """The full :meth:`ValueEngine.to_frame` of every evaluated selection (value or not)."""
    return ValueEngine().to_frame(state.opportunities)


# =============================================================================
# Picks of the matchday
# =============================================================================
def _picks(
    state: PipelineState,
    matchday: MatchdaySelector,
    girone: Girone | None,
    *,
    draws: bool,
    n: int,
) -> list[Pick]:
    """Shared body of :func:`draw_picks_table` / :func:`btts_picks_table`."""
    g = _as_girone(girone)
    preds = _restrict(state, matchday)
    if g is not None:
        preds = [p for p in preds if p.fixture.girone == g]
    chooser = draw_picks if draws else btts_picks
    return chooser(preds, n=n, matchday=None, opportunities=state.opportunities)


def draw_picks_table(
    state: PipelineState,
    matchday: MatchdaySelector = None,
    n: int = DEFAULT_N_DRAWS,
    girone: Girone | None = None,
) -> pd.DataFrame:
    """The ``n`` most probable draws (``X``) of the matchday as a :func:`picks_frame` (Italian columns).

    ``matchday`` defaults to the next one (:data:`ALL_MATCHDAYS` for the whole
    fixture list); ``girone`` optionally restricts the candidates to one
    competition, the default being all four, as the user asked.
    """
    return picks_frame(_picks(state, matchday, girone, draws=True, n=n))


def btts_picks_table(
    state: PipelineState,
    matchday: MatchdaySelector = None,
    n: int = DEFAULT_N_BTTS,
    girone: Girone | None = None,
) -> pd.DataFrame:
    """The ``n`` most probable both-teams-to-score matches (``GG``) of the matchday, see :func:`draw_picks_table`."""
    return picks_frame(_picks(state, matchday, girone, draws=False, n=n))


def picks_export_frame(draws: pd.DataFrame, btts: pd.DataFrame, matchday: Matchday | None = None) -> pd.DataFrame:
    """One frame for both lists: ``Lista`` (``'X'`` / ``'GG'``) and ``Giornata`` (the matchday label) in front of the picks columns.

    ``draws``/``btts`` are :func:`picks_frame` frames (the tables shown on
    screen); the result has exactly :data:`PICKS_EXPORT_COLUMNS` also when both
    are empty.
    """
    label = "" if matchday is None else matchday.label
    parts: list[pd.DataFrame] = []
    for name, frame in (("X", draws), ("GG", btts)):
        if frame.empty:
            continue
        parts.append(frame.assign(Lista=name, Giornata=label)[list(PICKS_EXPORT_COLUMNS)])
    if not parts:
        empty = picks_frame([]).assign(Lista=pd.Series(dtype="string"), Giornata=pd.Series(dtype="string"))
        return empty[list(PICKS_EXPORT_COLUMNS)]
    out = pd.concat(parts, ignore_index=True)
    return out.assign(Lista=out["Lista"].astype("string"), Giornata=out["Giornata"].astype("string"))


def picks_csv_bytes(draws: pd.DataFrame, btts: pd.DataFrame, matchday: Matchday | None = None) -> bytes:
    """CSV bytes (see :func:`export_csv_bytes`) of :func:`picks_export_frame`, for the download button and the console export."""
    return export_csv_bytes(picks_export_frame(draws, btts, matchday))


def export_csv_bytes(frame: pd.DataFrame) -> bytes:
    """UTF-8, comma-separated, ``.`` decimal CSV without index (dates as ISO ``YYYY-MM-DD``)."""
    return frame.to_csv(index=False, sep=",", decimal=".", date_format="%Y-%m-%d", lineterminator="\n").encode("utf-8")
