"""Pure data layer of the Streamlit dashboard (no ``streamlit`` import).

Everything the UI shows is computed here from the shared contracts, so the
dashboard can be unit-tested without a browser:

* :func:`build_state` runs the whole pipeline (load or simulate -> fit ->
  predict -> value) and returns an immutable-by-convention
  :class:`PipelineState` snapshot;
* the ``*_table`` helpers turn that snapshot into ``pandas`` frames with
  Italian column labels, ready for ``st.dataframe`` (:func:`matchday_table_for`
  is the whole matchday in one table, :func:`results_table_for` the played
  matches, both from :mod:`src.engine.tables`);
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

Sections (Serie B e Serie C / Serie D)
--------------------------------------
The fit is **one** for the whole perimeter (Serie B, Serie C and Serie D share
the Dixon-Coles estimate: promoted and relegated clubs connect the divisions,
the per-competition home advantage and the ``sum(alpha) = 0`` constraint per
competition keep the groups comparable), but the user-facing views are split by
*section*: :data:`SECTION_BC` groups the families ``'Serie C'`` and ``'Serie B'``,
:data:`SECTION_D` is the family ``'Serie D'`` alone. Every table helper takes
``section`` (``None`` = the whole perimeter, the pre-Serie-D behaviour) and
restricts itself to the competitions of that section; picks are ranked within
the section and the matchdays are grouped per section (:func:`matchdays`), so
the Serie D calendar never disturbs the Serie B / Serie C "giornate".
:func:`sections` lists the sections that have teams, :func:`section_options`
the ones the sidebar offers (Serie D also when configured but not yet
initialised, so the page can say how to initialise it, see
:attr:`PipelineState.serie_d_status` and :func:`serie_d_harvest_command`).
"""
from __future__ import annotations

import math
import re
import time
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Final, Literal

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from src.common.competitions import describe_competitions
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
from src.common.teams import DEFAULT_SEASON, TeamRegistry, load_registry, season_file_stem
from src.engine.model import DixonColesConfig, DixonColesEngine, FittedParams
from src.engine.picks import (
    DEFAULT_N_BTTS,
    DEFAULT_N_DRAWS,
    PICKS_COLUMNS,
    Matchday,
    Pick,
    btts_picks,
    draw_picks,
    picks_frame,
)
from src.engine.picks import matchdays as group_matchdays
from src.engine.tables import (
    MATCHDAY_COLUMNS,
    RESULTS_COLUMNS,
    RESULTS_MODEL_COLUMNS,
    matchday_table,
    results_table,
    team_matches,
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
    "DEFAULT_SECTION",
    "MARKETS_COLUMNS",
    "MARKET_LABELS",
    "MATCHDAY_COLUMNS",
    "MATCHDAY_TABLE_KEY_PREFIX",
    "MAX_HISTORY_SEASONS",
    "MIN_SIM_ROUNDS",
    "ODDS_COLUMNS",
    "PICKS_EXPORT_COLUMNS",
    "PRIOR_MIN_MATCHES_PER_TEAM",
    "PRIOR_THRESHOLD_MATCHES",
    "RANKING_COLUMNS",
    "RESULTS_COLUMNS",
    "RESULTS_MODEL_COLUMNS",
    "SECTION_BC",
    "SECTION_D",
    "SECTION_FAMILIES",
    "SECTION_SLUGS",
    "SERIE_D_FAMILY",
    "SERIE_D_HARVEST_SEASONS",
    "SERIE_D_STATUS_NOT_CONFIGURED",
    "SERIE_D_STATUS_NOT_INITIALISED",
    "TRAPS_COLUMNS",
    "VALUE_COLUMNS",
    "CoverageReport",
    "MatchdaySelector",
    "PipelineState",
    "Source",
    "all_sections",
    "btts_picks_table",
    "build_state",
    "competition_label",
    "competition_options",
    "configured_gironi",
    "coverage_report",
    "coverage_text",
    "current_teams",
    "draw_picks_table",
    "export_csv_bytes",
    "family_gironi",
    "family_label",
    "family_predictions",
    "find_matchday",
    "fixture_label",
    "fixtures_for",
    "heatmap_figure",
    "history_label",
    "in_section",
    "load_odds_quotes",
    "markets_table",
    "matchday_csv_bytes",
    "matchday_predictions",
    "matchday_table_for",
    "matchday_table_key",
    "matchdays",
    "matches_per_team",
    "max_sim_rounds",
    "no_odds_message",
    "opportunities_table",
    "other_season_reason",
    "outside_section",
    "perimeter_matches",
    "perimeter_text",
    "picks_csv_bytes",
    "picks_export_frame",
    "predictions_table",
    "prior_recommended",
    "quotes_for",
    "ranking_table",
    "resolve_matchday",
    "results_csv_bytes",
    "results_table_for",
    "results_team_options",
    "seasons_available",
    "section_families",
    "section_gironi",
    "section_key",
    "section_matches",
    "section_of_family",
    "section_opportunities",
    "section_options",
    "section_perimeter_label",
    "section_predictions",
    "section_ready",
    "section_slug",
    "section_team_count",
    "section_traps",
    "sections",
    "sections_for",
    "selected_row_match_id",
    "serie_d_gironi",
    "serie_d_harvest_command",
    "serie_d_status_text",
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

#: The two user-facing *sections* of the dashboard and the console. The fit is joint, the views are split:
#: :data:`SECTION_BC` groups the families ``'Serie C'`` and ``'Serie B'`` (a UI grouping, not a league),
#: :data:`SECTION_D` is the family ``'Serie D'`` on its own.
SECTION_BC: Final[str] = "Serie B e Serie C"
SECTION_D: Final[str] = "Serie D"
#: Section shown by default (the first option of the sidebar radio and the one whose widget keys carry no suffix).
DEFAULT_SECTION: Final[str] = SECTION_BC
#: Competition families covered by each section (``config/competitions.toml`` ``family`` values). A family
#: listed in no section (a league added later) is a section of its own, named after the family.
SECTION_FAMILIES: Final[dict[str, tuple[str, ...]]] = {SECTION_BC: ("Serie C", "Serie B"), SECTION_D: ("Serie D",)}
#: Short file-name / widget-key suffixes of the sections (``main.py`` exports ``<stem>_d_giornata.csv``, the
#: dashboard keys ``matchday_d`` ...); another section gets a slug derived from its name (:func:`section_slug`).
SECTION_SLUGS: Final[dict[str, str]] = {SECTION_BC: "bc", SECTION_D: "d"}
#: ``family`` of the Serie D competitions in ``config/competitions.toml``.
SERIE_D_FAMILY: Final[str] = "Serie D"
#: Seasons the FootyStats harvester collects for Serie D (:func:`serie_d_harvest_command`).
SERIE_D_HARVEST_SEASONS: Final[int] = 3
#: :attr:`PipelineState.serie_d_status` when the configuration has no Serie D competition at all.
SERIE_D_STATUS_NOT_CONFIGURED: Final[str] = "non configurata: nessuna competizione con family 'Serie D' in config/competitions.toml"
#: :attr:`PipelineState.serie_d_status` when Serie D is configured but its gironi have no teams yet (the
#: FootyStats harvester initialises them from ``/league-teams`` the first time it runs with the user's key).
SERIE_D_STATUS_NOT_INITIALISED: Final[str] = "non inizializzata: esegui la raccolta FootyStats con la chiave"

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
    next one (:attr:`next_matchday`). The per-section matchdays of the
    dashboard are :func:`matchdays` ``(state, section)``, cached on the state.

    The perimeter may span several *families* (:attr:`families`: ``'Serie C'``,
    ``'Serie B'``, ``'Serie D'``); :attr:`serie_d_status` says whether the Serie D
    section is usable (its gironi are initialised from the FootyStats API, not
    shipped with the code).
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
    #: Lazily built per-team long frame of the current-season matches (see :meth:`team_matches_frame`).
    _team_matches: pd.DataFrame | None = field(default=None, init=False, repr=False, compare=False)
    #: Lazily grouped matchdays per section (see :func:`matchdays`); a ``dataclasses.replace`` starts afresh.
    _section_matchdays: dict[str, list[Matchday]] = field(default_factory=dict, init=False, repr=False, compare=False)

    def team_matches_frame(self) -> pd.DataFrame:
        """Per-team long frame of the **current-season** matches (:func:`src.engine.tables.team_matches`), built once.

        One row per team and played match, the base of every form column of
        :func:`matchday_table_for`: the form of a team is its last matches of
        the season the table describes, so the frame is derived from
        :attr:`current_matches` and cached on the state (``dataclasses.replace``
        starts a fresh cache). Not part of the equality of the snapshot.
        """
        if self._team_matches is None:
            self._team_matches = team_matches(self.current_matches)
        return self._team_matches

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
    def perimeter_gironi(self) -> tuple[Girone, ...]:
        """Competitions of the perimeter that have **current teams**, in configuration order.

        From the registry when the state has one (a competition listed in the
        dictionary with teams, played or not; a Serie D girone not yet
        initialised has none and is left out), otherwise the competitions with
        fitted current teams.
        """
        if self.registry is not None:
            return self.registry.gironi
        fitted = self.params.current_teams_by_girone()
        return tuple(g for g in Girone if g in fitted)

    @property
    def families(self) -> tuple[str, ...]:
        """Competition families present in the perimeter (``('Serie C', 'Serie B')``, plus ``'Serie D'`` once initialised), configuration order."""
        return tuple(dict.fromkeys(g.family for g in self.perimeter_gironi))

    def family_of(self, girone: Girone | str) -> str:
        """Family of a competition (``'A'`` -> ``'Serie C'``, ``'SB'`` -> ``'Serie B'``, ``'DA'`` -> ``'Serie D'``)."""
        return Girone.parse(girone).family

    @property
    def serie_d_initialised(self) -> bool:
        """Whether at least one Serie D girone of the perimeter has teams (dictionary filled by the FootyStats harvester)."""
        return section_team_count(self, SECTION_D) > 0

    @property
    def serie_d_status(self) -> str:
        """Italian status of the Serie D section, for the sidebar notice and the console banner.

        :data:`SERIE_D_STATUS_NOT_CONFIGURED` when ``config/competitions.toml``
        has no ``'Serie D'`` family; :data:`SERIE_D_STATUS_NOT_INITIALISED` when
        its gironi have no teams yet (``'non inizializzata: esegui la raccolta
        FootyStats con la chiave'``, the command is :func:`serie_d_harvest_command`);
        otherwise ``'inizializzata: 162 squadre, 648 partite della stagione
        2026/27, 81 partite in programma previste'``, or, with teams but no
        played match in the data, ``'inizializzata: 162 squadre, 0 partite nei
        dati: esegui la raccolta FootyStats con la chiave'``.
        """
        return serie_d_status_text(
            section_team_count(self, SECTION_D),
            len(section_matches(self.current_matches, SECTION_D)),
            len(section_predictions(self, SECTION_D)),
            season_label=self.season_label,
        )

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
# Sections and families (pure: configuration only)
# =============================================================================
def section_of_family(family: str) -> str:
    """Section a competition family belongs to (:data:`SECTION_FAMILIES`).

    ``'Serie C'`` and ``'Serie B'`` -> :data:`SECTION_BC`, ``'Serie D'`` ->
    :data:`SECTION_D`; a family listed in no section (a league added later) is a
    section of its own, named after the family.
    """
    for section, members in SECTION_FAMILIES.items():
        if family in members:
            return section
    return str(family)


def family_label(family: str) -> str:
    """User-facing label of the section a family is shown under: ``family_label('Serie C') == 'Serie B e Serie C'``.

    The two families Serie B and Serie C are one *section* of the UI (a
    grouping, not a league); ``family_label('Serie D') == 'Serie D'``. Not to be
    confused with :func:`src.common.competitions.family_label`, which returns
    the canonical spelling of a family name.
    """
    return section_of_family(family)


def section_families(section: str) -> tuple[str, ...]:
    """Families covered by a section (``('Serie C', 'Serie B')`` for :data:`SECTION_BC`; an unknown section is its own family)."""
    return SECTION_FAMILIES.get(section, (str(section),))


def section_slug(section: str) -> str:
    """Short identifier of a section for file names and widget keys: ``'bc'``, ``'d'`` (:data:`SECTION_SLUGS`), else a slug of the name."""
    slug = SECTION_SLUGS.get(section)
    if slug is None:
        slug = re.sub(r"[^a-z0-9]+", "_", str(section).lower()).strip("_") or "sezione"
    return slug


def section_key(base: str, section: str | None = None) -> str:
    """``st.session_state`` key (or file stem) of ``base`` scoped to a section.

    The default section (:data:`DEFAULT_SECTION`, or ``None``) keeps the bare
    key, so the pre-Serie-D keys (``'matchday'``, ``'fixture'``, ...) are
    unchanged; another section appends its slug (``'matchday_d'``). Each section
    therefore owns its selected matchday, fixture and filters, and switching
    section starts from that section's defaults.
    """
    if section is None or section == DEFAULT_SECTION:
        return base
    return f"{base}_{section_slug(section)}"


def all_sections(gironi: Iterable[Girone] = Girone) -> tuple[str, ...]:
    """Sections of the given competitions in display order: :data:`SECTION_BC`, :data:`SECTION_D`, then any other family's section."""
    present = list(dict.fromkeys(section_of_family(g.family) for g in gironi))
    ordered = [sec for sec in (SECTION_BC, SECTION_D) if sec in present]
    ordered += [sec for sec in present if sec not in ordered]
    return tuple(ordered)


def sections_for(gironi: Iterable[Girone]) -> tuple[str, ...]:
    """The sections that have at least one of ``gironi`` (e.g. a registry's competitions with teams), display order."""
    return all_sections(gironi)


def configured_gironi(section: str | None = None) -> tuple[Girone, ...]:
    """Every configured competition of a section (``config/competitions.toml``), teams or not; all of them for ``None``."""
    if section is None:
        return tuple(Girone)
    wanted = section_families(section)
    return tuple(g for g in Girone if g.family in wanted)


def in_section(girone: Girone | str, section: str | None) -> bool:
    """Whether a competition belongs to ``section`` (``None`` = every section)."""
    return section is None or Girone.parse(girone).family in section_families(section)


def outside_section(girone: Girone | str | None, section: str | None) -> bool:
    """Whether a *requested* competition lies outside the requested section: the shared guard of the table helpers.

    ``False`` for ``girone=None`` (no competition requested) or ``section=None``
    (every section); otherwise the negation of :func:`in_section`. A helper asked
    for ``Girone.A`` within :data:`SECTION_D` returns its empty frame or list
    instead of Serie C data, exactly like :func:`fixtures_for` or
    :func:`results_table_for` already did through their row filters.
    """
    return girone is not None and not in_section(girone, section)


def serie_d_gironi() -> tuple[Girone, ...]:
    """The configured Serie D competitions (family :data:`SERIE_D_FAMILY`), ``()`` when the configuration has none."""
    return tuple(g for g in Girone if g.family == SERIE_D_FAMILY)


def serie_d_harvest_command(gironi: Iterable[Girone] | None = None, *, season: str | None = None) -> str:
    """The exact console command that initialises and harvests Serie D with the user's FootyStats key.

    ``'python -m src.ingestion.footystats --competitions DA DB DC DD DE DF DG DH DI --seasons 3'``
    with the shipped configuration (the launchers ``avvia.bat`` / ``avvia.command``
    run the harvester for every configured competition as well).

    ``season`` is the registry season the command has to initialise
    (``'2025/26'`` or ``'2025_26'``, whatever the caller holds): the harvester
    CLI writes ``config/teams_<--season>.json`` and its ``--season`` defaults to
    :data:`~src.common.teams.DEFAULT_SEASON`, so ``--season 2025_26`` is
    appended whenever the perimeter season differs from that default. A notice
    that names ``config/teams_2025_26.json`` thus prints a command that fills
    that very file; ``None`` (or the default season) keeps the short command.
    """
    codes = [g.value for g in (serie_d_gironi() if gironi is None else gironi)]
    command = f"python -m src.ingestion.footystats --competitions {' '.join(codes)} --seasons {SERIE_D_HARVEST_SEASONS}"
    stem = season_file_stem(season) if season else ""
    if stem and stem != season_file_stem(DEFAULT_SEASON):
        command += f" --season {stem}"
    return command


def serie_d_status_text(n_teams: int, n_matches: int, n_fixtures: int | None = None, *, season_label: str = "") -> str:
    """The Italian Serie D status from its counts (shared by :attr:`PipelineState.serie_d_status` and the console).

    :data:`SERIE_D_STATUS_NOT_CONFIGURED` when no Serie D competition is
    configured, :data:`SERIE_D_STATUS_NOT_INITIALISED` for ``n_teams == 0``,
    ``'inizializzata: 162 squadre, 0 partite nei dati: esegui la raccolta
    FootyStats con la chiave'`` for teams without played matches, otherwise
    ``'inizializzata: 162 squadre, 648 partite della stagione 2026/27, 81 partite
    in programma previste'`` (the fixtures part only when ``n_fixtures`` is given).
    """
    if not serie_d_gironi():
        return SERIE_D_STATUS_NOT_CONFIGURED
    if n_teams <= 0:
        return SERIE_D_STATUS_NOT_INITIALISED
    if n_matches <= 0:
        return f"inizializzata: {n_teams} squadre, 0 partite nei dati: esegui la raccolta FootyStats con la chiave"
    season = f" della stagione {season_label}" if season_label else ""
    text = f"inizializzata: {n_teams} squadre, {n_matches} partite{season}"
    if n_fixtures is not None:
        text += f", {n_fixtures} partite in programma previste"
    return text


def competition_options(section: str | None = None) -> tuple[str, ...]:
    """Entries of the *Competizione* selector: ``'Tutti'`` plus the short label of every configured competition of the section.

    ``('Tutti', 'Girone A', 'Girone B', 'Girone C', 'Serie B')`` for
    :data:`SECTION_BC`, ``('Tutti', 'D-A', ..., 'D-I')`` for :data:`SECTION_D`,
    every competition for ``None``.
    """
    return ("Tutti", *(g.short for g in configured_gironi(section)))


def section_matches(matches: pd.DataFrame, section: str | None) -> pd.DataFrame:
    """Rows of a canonical match frame whose competition belongs to ``section`` (the frame itself for ``None``)."""
    if section is None or matches.empty:
        return matches
    codes = {g.value for g in configured_gironi(section)}
    keep = matches["girone"].astype(str).isin(codes).to_numpy(dtype=bool)
    return matches.loc[keep].reset_index(drop=True)


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
        note instead of raising, so the Giornata tab and the value panel only
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
    giornate = group_matchdays(p.fixture for p in predictions)
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


def _empty_ranking() -> pd.DataFrame:
    """An empty strength ranking with the :data:`RANKING_COLUMNS` and their dtypes."""
    return pd.DataFrame({c: pd.Series(dtype="string" if c in ("Squadra", "Competizione") else "float64") for c in RANKING_COLUMNS})


def competition_label(girone: Girone | str) -> str:
    """Short label shown in the tables for a competition code: ``'A'`` -> ``'Girone A'``, ``'SB'`` -> ``'Serie B'``."""
    return Girone.parse(girone).short


# ----------------------------------------------------------------------------- sections of a state
def section_gironi(state: PipelineState, section: str | None = None) -> tuple[Girone, ...]:
    """Competitions of the perimeter **with current teams** that belong to ``section`` (every one for ``None``), configuration order.

    For filtering use :func:`configured_gironi` / :func:`in_section` (every
    configured competition of the section); this is the "has teams" view behind
    :func:`sections` and :func:`section_perimeter_label`.
    """
    return tuple(g for g in state.perimeter_gironi if in_section(g, section))


def family_gironi(state: PipelineState, family: str) -> tuple[Girone, ...]:
    """Competitions of one family (``'Serie C'``, ``'Serie D'``) that have current teams in the perimeter, configuration order."""
    return tuple(g for g in state.perimeter_gironi if g.family == family)


def section_team_count(state: PipelineState, section: str | None = None) -> int:
    """Current teams of the perimeter in a section: dictionary teams with a registry, fitted current teams otherwise."""
    gironi = configured_gironi(section)
    if state.registry is not None:
        return sum(len(state.registry.teams_by_girone.get(g, ())) for g in gironi)
    fitted = state.params.current_teams_by_girone()
    return sum(len(fitted.get(g, ())) for g in gironi)


def section_ready(state: PipelineState, section: str | None = None) -> bool:
    """Whether the section has at least one **fitted** current team, i.e. something to rank, predict and price.

    False for Serie D before its gironi are initialised, and also when they are
    initialised but no Serie D match has been harvested yet (teams without
    matches are not fitted): the dashboard then shows the status notice instead
    of empty tables.
    """
    fitted = state.params.current_teams_by_girone()
    return any(g in fitted for g in configured_gironi(section))


def sections(state: PipelineState) -> tuple[str, ...]:
    """The sections that have at least one competition with current teams, display order.

    ``('Serie B e Serie C',)`` with the shipped dictionaries (the Serie D gironi
    are empty until the FootyStats harvester initialises them), ``('Serie B e
    Serie C', 'Serie D')`` once Serie D has teams.
    """
    return sections_for(state.perimeter_gironi)


def section_options(state: PipelineState) -> tuple[str, ...]:
    """The sections the sidebar radio offers: :func:`sections` plus :data:`SECTION_D` when Serie D is configured but not initialised.

    An uninitialised Serie D is offered on purpose, so that the page can show
    :attr:`PipelineState.serie_d_status` and the command that initialises it
    instead of hiding the section. Never empty: :data:`DEFAULT_SECTION` when the
    perimeter has no competition with teams at all.
    """
    present = set(sections(state))
    if serie_d_gironi():
        present.add(SECTION_D)
    if not present:
        return (DEFAULT_SECTION,)
    ordered = [sec for sec in (SECTION_BC, SECTION_D) if sec in present]
    ordered += [sec for sec in all_sections() if sec in present and sec not in ordered]
    return tuple(ordered)


def section_predictions(state: PipelineState, section: str | None = None) -> list[MatchPrediction]:
    """Every prediction of the section's competitions (all of them for ``None``), in ``state.predictions`` order."""
    if section is None:
        return list(state.predictions)
    return [p for p in state.predictions if in_section(p.fixture.girone, section)]


def family_predictions(state: PipelineState, family: str) -> list[MatchPrediction]:
    """Every prediction whose competition belongs to ``family``, sorted by date, competition order and label."""
    order = list(Girone)
    preds = [p for p in state.predictions if p.fixture.girone.family == family]
    return sorted(preds, key=lambda p: (p.fixture.date, order.index(p.fixture.girone), p.fixture.label, p.fixture.match_id))


def matchdays(state: PipelineState, section: str | None = None) -> list[Matchday]:
    """The "giornate" of a section, grouped from **its own** predicted fixtures (``state.matchdays`` for ``None``).

    Serie D plays 34 rounds on its own calendar: grouped together with Serie B
    and Serie C its rounds would rename every shared weekend and add midweek
    matchdays nobody in the other section asked for, so each section groups its
    fixtures separately (:func:`src.engine.picks.matchdays` with the section's
    competitions) and its keys, labels and rounds only ever mention that
    section. Grouped once per section and cached on the state.
    """
    if section is None:
        return list(state.matchdays)
    cached = state._section_matchdays.get(section)
    if cached is None:
        cached = group_matchdays((p.fixture for p in state.predictions), gironi=configured_gironi(section))
        state._section_matchdays[section] = cached
        log.debug("section matchdays grouped", extra={"section": section, "n_matchdays": len(cached)})
    return list(cached)


def section_perimeter_label(state: PipelineState, section: str | None = None) -> str:
    """The perimeter of one section as the configuration describes it: ``'Serie D 2026/27 Gironi A-I (162 squadre)'``.

    Built with :func:`src.common.competitions.describe_competitions` on the
    section's competitions that have teams (``'Serie C 2026/27 Gironi A/B/C +
    Serie B (80 squadre)'`` for :data:`SECTION_BC`, identical to
    :attr:`PipelineState.perimeter_label` when the perimeter has no other
    section); the whole perimeter for ``None``. A section without teams reads
    ``'Serie D 2026/27 (nessuna squadra)'``.
    """
    if section is None:
        return state.perimeter_label
    gironi = section_gironi(state, section)
    n_teams = section_team_count(state, section)
    if not gironi or n_teams == 0:
        return f"{section} {state.season_label} (nessuna squadra)".strip()
    text = describe_competitions((g.competition for g in gironi), season=state.season_label)
    return f"{text} ({n_teams} squadre)"


def perimeter_text(state: PipelineState, section: str | None = None) -> str:
    """Italian one-liner for badges and subtitles, restricted to a section when given.

    ``'Perimetro: Serie D 2026/27 Gironi A-I (162 squadre) · 648 partite stagione
    corrente · 0 partite storiche'``; with ``None`` exactly
    :attr:`PipelineState.perimeter_text` (the whole perimeter). The match counts
    are the rows of the section's competitions in the current season and in the
    history, the history label spans the section's history seasons.
    """
    if section is None:
        return state.perimeter_text
    current = section_matches(state.current_matches, section)
    history = section_matches(state.history_matches, section)
    text = (
        f"Perimetro: {section_perimeter_label(state, section)} · {len(current)} partite stagione corrente · "
        f"{len(history)} partite storiche"
    )
    if len(history):
        text += f" ({history_label(history)})"
    return text


def coverage_text(state: PipelineState, section: str | None = None) -> str:
    """``'Copertura 2026/27: 8.0 partite per squadra nella competizione meno coperta'`` for the section's competitions.

    Read from ``state.coverage`` restricted to the section (``''`` without a
    coverage report or when none of the section's competitions is covered); the
    ignored competitions of the section, if any, follow after ``' · '``.
    """
    report = state.coverage
    if report is None:
        return ""
    per = {g: v for g, v in report.per_competition.items() if in_section(g, section)}
    ignored = tuple(g for g in report.ignored if in_section(g, section))
    if not per and not ignored:
        return ""
    restricted = CoverageReport(per_competition=per, ignored=ignored, season=report.season)
    text = f"Copertura {state.season_label}: {restricted.thinnest:.1f} partite per squadra nella competizione meno coperta" if per else f"Copertura {state.season_label}: nessuna competizione coperta"
    if ignored:
        text += f" · {restricted.describe_ignored()}"
    return text


# ----------------------------------------------------------------------------- ranking
def ranking_table(state: PipelineState, girone: Girone | None, section: str | None = None) -> pd.DataFrame:
    """Strength ranking (alpha desc) with Italian labels, rounded to 3 decimals.

    Only the current teams of the perimeter are listed and ``Competizione`` is
    the short label of their registry competition (historical opponents never
    appear). ``girone=None`` returns every current team of ``section`` (every
    section when ``None``); a competition with no fitted teams, or one outside
    ``section`` (:func:`outside_section`), yields an empty frame with the same
    columns instead of raising.
    """
    g = _as_girone(girone)
    if g is not None and (g not in state.params.teams_by_girone or outside_section(g, section)):
        return _empty_ranking()
    ranking = state.engine.ranking(g)
    if g is None and section is not None:
        codes = {x.value for x in configured_gironi(section)}
        ranking = ranking.loc[ranking["girone"].astype(str).isin(codes)]
    frame = ranking[list(_RANKING_RENAME)].rename(columns=_RANKING_RENAME)
    numeric = [c for c in RANKING_COLUMNS if c not in ("Squadra", "Competizione")]
    frame = frame.assign(**{c: frame[c].astype("float64").round(3) for c in numeric})
    frame = frame.assign(
        Squadra=frame["Squadra"].astype("string"),
        Competizione=frame["Competizione"].astype(str).map(competition_label).astype("string"),
    )
    return frame[list(RANKING_COLUMNS)].reset_index(drop=True)


def find_matchday(state: PipelineState, key: str | None, section: str | None = None) -> Matchday | None:
    """The matchday of ``state`` (of ``section``, when given) with that :attr:`Matchday.key`, ``None`` when unknown (or ``key`` is ``None``)."""
    if key is None:
        return None
    return next((m for m in matchdays(state, section) if m.key == key), None)


def resolve_matchday(state: PipelineState, matchday: MatchdaySelector | str = None, section: str | None = None) -> Matchday | None:
    """Turn a selector into a :class:`Matchday` (or ``None`` meaning "no restriction").

    ``None`` -> the next matchday of the section (``None`` when it has none,
    which then means every prediction, i.e. none); :data:`ALL_MATCHDAYS` ->
    ``None`` (no restriction); a :class:`Matchday` -> itself; a key string ->
    the matchday with that key among the section's matchdays, falling back to
    the next one when the key is stale.
    """
    if matchday is ALL_MATCHDAYS or (isinstance(matchday, str) and matchday == ALL_MATCHDAYS):
        return None
    days = matchdays(state, section)
    first = days[0] if days else None
    if matchday is None:
        return first
    if isinstance(matchday, Matchday):
        return matchday
    found = next((m for m in days if m.key == str(matchday)), None)
    if found is None:
        log.debug("unknown matchday key, using the next matchday", extra={"key": str(matchday), "section": section})
        return first
    return found


def _restrict(state: PipelineState, matchday: MatchdaySelector, section: str | None = None) -> list[MatchPrediction]:
    """Predictions of the selected matchday within a section (every prediction of the section for :data:`ALL_MATCHDAYS`).

    With ``None`` and no matchday in the section (no predicted fixture) there is
    nothing to show, so the result is empty, consistent with the selectors.
    """
    preds = section_predictions(state, section)
    if matchday is ALL_MATCHDAYS or (isinstance(matchday, str) and matchday == ALL_MATCHDAYS):
        return preds
    chosen = resolve_matchday(state, matchday, section)
    if chosen is None:
        return []
    return [p for p in preds if chosen.contains(p)]


def matchday_predictions(state: PipelineState, matchday: MatchdaySelector = None, section: str | None = None) -> list[MatchPrediction]:
    """Predictions of one matchday (the next by default, all with :data:`ALL_MATCHDAYS`) of a section, sorted by date, competition, label."""
    preds = _restrict(state, matchday, section)
    order = list(Girone)
    return sorted(preds, key=lambda p: (p.fixture.date, order.index(p.fixture.girone), p.fixture.label, p.fixture.match_id))


def fixtures_for(
    state: PipelineState,
    girone: Girone | None,
    matchday: MatchdaySelector = None,
    section: str | None = None,
) -> list[MatchPrediction]:
    """Predictions of the upcoming fixtures (optionally one girone, one section), sorted by date then label.

    ``matchday`` defaults to the **next** matchday of the section, so with a
    season-long fixture list the match selector shows one "giornata" instead of
    hundreds of fixtures; pass a :class:`Matchday` to pick another one or
    :data:`ALL_MATCHDAYS` for every predicted fixture (the exports do).
    """
    g = _as_girone(girone)
    preds = [p for p in _restrict(state, matchday, section) if g is None or p.fixture.girone == g]
    return sorted(preds, key=lambda p: (p.fixture.date, p.fixture.label))


def quotes_for(state: PipelineState, match_id: str) -> list[OddsQuote]:
    """Every bookmaker quote of one fixture."""
    return [q for q in state.quotes if q.match_id == match_id]


# =============================================================================
# Matchday table and played results
# =============================================================================
def matchday_table_for(
    state: PipelineState,
    matchday: MatchdaySelector = None,
    girone: Girone | None = None,
    section: str | None = None,
) -> pd.DataFrame:
    """The whole matchday as one table (:func:`src.engine.tables.matchday_table`), one row per predicted fixture.

    ``matchday`` follows the selector semantics of :func:`fixtures_for` (the
    next matchday of the section by default, :data:`ALL_MATCHDAYS` for every
    predicted fixture), ``girone`` keeps one competition, ``section`` one
    section (every competition when ``None``). Strengths come from
    ``state.params``, the form columns from the current-season matches
    (:meth:`PipelineState.team_matches_frame`, built once per state), the
    ``Book`` / ``EV max`` / ``Flag`` columns from ``state.opportunities`` and
    ``state.traps``. Columns exactly :data:`MATCHDAY_COLUMNS`, ``match_id`` last.
    """
    preds = _restrict(state, matchday, section)
    return matchday_table(
        preds,
        params=state.params,
        matches=state.current_matches,
        opportunities=state.opportunities,
        traps=state.traps,
        girone=_as_girone(girone),
        long=state.team_matches_frame(),
    )


#: Prefix of the ``st.dataframe`` widget key of the *Giornata* table (see :func:`matchday_table_key`).
MATCHDAY_TABLE_KEY_PREFIX: Final[str] = "matchday_table"


def matchday_table_key(matchday: Matchday | None, girone: Girone | None = None, section: str | None = None) -> str:
    """``st.session_state`` key of the *Giornata* table widget, scoped to the matchday, competition and section it shows.

    Streamlit derives a ``st.dataframe`` element id from its ``key`` and
    selection mode only, not from the data, so a row selection made on one
    table would survive a change of matchday or sidebar competition and be
    mapped onto the new table (a fixture the user never clicked). Scoping the
    key -- ``'matchday_table_2026-10-18_A'``, ``'matchday_table_tutte_all'`` for
    every predicted fixture of every competition, ``'matchday_table_2026-11-08_all_d'``
    for the Serie D section (whose matchday may share its key with a Serie B /
    Serie C one played the same weekend; the default section adds no suffix,
    see :func:`section_key`) -- makes each scope its own widget, which starts
    without a selection. The prefix is :data:`MATCHDAY_TABLE_KEY_PREFIX`, so a
    test can still find the table by ``key.startswith(...)``.
    """
    scope_matchday = "tutte" if matchday is None else str(matchday.key)
    scope_girone = "all" if girone is None else str(_as_girone(girone).value)
    return section_key(f"{MATCHDAY_TABLE_KEY_PREFIX}_{scope_matchday}_{scope_girone}", section)


def fixture_label(prediction: MatchPrediction) -> str:
    """``'[A] 2026-11-08 · Home - Away'``: the entry of the dashboard's *Partita* selectbox for one prediction."""
    fx = prediction.fixture
    return f"[{fx.girone.value}] {fx.date.isoformat()} · {fx.label}"


def selected_row_match_id(event: object, table: pd.DataFrame) -> str | None:
    """``match_id`` of the row selected in a ``st.dataframe`` rendered with ``selection_mode='single-row'``.

    ``event`` is what ``st.dataframe(..., on_select="rerun")`` returns: its
    ``selection.rows`` holds the positional index of the selected row of
    ``table`` (a :func:`matchday_table_for` frame, whose last column is
    ``match_id``). ``None`` when nothing is selected, when the position is out
    of range (the table changed under the selection) or when ``table`` has no
    ``match_id`` column. Pure, so the mapping is unit-tested without a browser.
    """
    selection = getattr(event, "selection", None)
    rows = getattr(selection, "rows", None) if selection is not None else None
    if not rows or "match_id" not in table.columns:
        return None
    try:
        position = int(list(rows)[0])
    except (TypeError, ValueError):
        return None
    if not 0 <= position < len(table):
        return None
    return str(table["match_id"].iloc[position])


def matchday_csv_bytes(
    state: PipelineState,
    matchday: MatchdaySelector = None,
    girone: Girone | None = None,
    section: str | None = None,
) -> bytes:
    """CSV bytes (see :func:`export_csv_bytes`) of :func:`matchday_table_for`, for the download button and the console export."""
    return export_csv_bytes(matchday_table_for(state, matchday, girone, section))


def seasons_available(state: PipelineState, section: str | None = None) -> list[str]:
    """Season labels present in ``state.matches`` (of the section's competitions, when given): the registry season first, then the others most recent first.

    The registry season is listed even when it has no match yet (the season has
    not started), so the *Risultati* selector always offers the current season.
    """
    frame = section_matches(state.matches, section)
    present = sorted(set(pd.to_datetime(frame["date"]).map(season_of)), reverse=True) if len(frame) else []
    current = str(state.season_label).replace("_", "/")
    if current:
        return [current, *[s for s in present if s != current]]
    return present


def current_teams(state: PipelineState, girone: Girone | None = None, section: str | None = None) -> list[str]:
    """Sorted current teams of the perimeter (one competition when ``girone`` is given, one section with ``section``).

    From the registry when the state has one (every team of the dictionary,
    with or without matches), otherwise the fitted current teams. A competition
    the registry lists without teams (an uninitialised Serie D girone) yields
    ``[]``, and so does a competition outside ``section`` (:func:`outside_section`).
    """
    g = _as_girone(girone)
    if outside_section(g, section):
        return []
    gironi = (g,) if g is not None else configured_gironi(section)
    if state.registry is not None:
        names = [t for x in gironi for t in state.registry.teams_by_girone.get(x, ())]
        return sorted(str(t) for t in names)
    by_girone = state.params.current_teams_by_girone()
    names = [t for x in gironi for t in by_girone.get(x, ())]
    return sorted(str(t) for t in names)


def results_team_options(
    state: PipelineState,
    girone: Girone | None = None,
    season: str | None = None,
    section: str | None = None,
) -> list[str]:
    """Teams selectable in the *Risultati* tab for one season, competition and section: every option has at least one row.

    The options are the clubs that **played** that season in that competition
    (home or away in ``state.matches``, filtered exactly like
    :func:`results_table_for` does: by the season of the match date and by the
    competition the match was played in), historical teams included. The
    registry teams of the competition (:func:`current_teams`) are added only
    for the current season (``state.season_label``), where a team may not have
    played yet, or when the season has no match at all. A current team that
    played a past season elsewhere (promoted or relegated since) is therefore
    *not* offered under its present competition for that season, where it would
    have an empty table. Sorted, without duplicates; ``[]`` for a competition
    outside ``section`` (:func:`outside_section`), whose table is empty too.
    """
    g = _as_girone(girone)
    if outside_section(g, section):
        return []
    label = str(state.season_label if season is None else season).replace("_", "/")
    current_label = str(state.season_label).replace("_", "/")
    names: set[str] = set()
    frame = section_matches(state.matches, section)
    if len(frame) and label:
        mask = (pd.to_datetime(frame["date"]).map(season_of) == label).to_numpy(dtype=bool)
        if g is not None:
            mask = mask & (frame["girone"] == g.value).to_numpy(dtype=bool)  # a new array: to_numpy() may be read-only
        subset = frame.loc[mask]
        names |= set(subset["home_team"].astype(str)) | set(subset["away_team"].astype(str))
    if not names or label == current_label:
        names |= set(current_teams(state, g, section))
    return sorted(names)


def results_table_for(
    state: PipelineState,
    girone: Girone | None = None,
    team: str | None = None,
    season: str | None = None,
    section: str | None = None,
) -> pd.DataFrame:
    """Played matches of one season (the registry season by default) with the model's reading of each.

    :func:`src.engine.tables.results_table` on ``state.matches`` (every season
    the fit saw, so a past season can be chosen) with ``state.engine``:
    columns :data:`RESULTS_MODEL_COLUMNS`. ``girone`` keeps one competition,
    ``section`` one section, ``team`` the matches of one team (home or away,
    historical teams included).
    """
    label = str(state.season_label if season is None else season).replace("_", "/") or None
    gironi = None if section is None else configured_gironi(section)
    return results_table(state.matches, engine=state.engine, season=label, girone=_as_girone(girone), team=team, gironi=gironi)


def results_csv_bytes(
    state: PipelineState,
    girone: Girone | None = None,
    team: str | None = None,
    season: str | None = None,
    section: str | None = None,
) -> bytes:
    """CSV bytes (see :func:`export_csv_bytes`) of :func:`results_table_for`, for the download button and the console export."""
    return export_csv_bytes(results_table_for(state, girone, team, season, section))


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
    section: str | None = None,
) -> list[ValueOpportunity]:
    """The rows of the value panel as objects: ``ev >= min_ev``, one competition when given, one matchday, one section.

    Exactly the selection and the order of :func:`value_table` (EV descending,
    then match label, market and selection), so a caller that needs the panel's
    first row -- the worked example of the *Legenda* tab -- reads it from here
    instead of re-filtering ``state.opportunities``. ``matchday`` defaults to
    the next matchday of the section like :func:`fixtures_for`;
    :data:`ALL_MATCHDAYS` lifts the restriction.
    """
    g = _as_girone(girone)
    allowed = {p.fixture.match_id for p in _restrict(state, matchday, section)}
    rows = [o for o in state.opportunities if o.ev >= min_ev and (g is None or o.girone == g) and o.match_id in allowed]
    # the table sorts on the EV rounded to 2 decimals in percent; do the same so the two orders agree
    rows.sort(key=lambda o: (-round(o.ev * 100.0, 2), f"{o.home_team} - {o.away_team}", MARKET_LABELS[o.market], o.selection))
    return rows


def value_table(
    state: PipelineState,
    min_ev: float,
    girone: Girone | None = None,
    matchday: MatchdaySelector = None,
    section: str | None = None,
) -> pd.DataFrame:
    """Selections with ``ev >= min_ev`` (optionally one competition, one section), EV descending, Italian labels.

    Like :func:`fixtures_for`, ``matchday`` defaults to the next matchday of the
    section; pass :data:`ALL_MATCHDAYS` to list the value selections of the
    whole fixture list. The rows are those of :func:`value_opportunities`, in
    the same order.
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
        for o in value_opportunities(state, min_ev, girone, matchday, section)
    ]
    frame = pd.DataFrame(rows, columns=list(VALUE_COLUMNS))
    frame = frame.assign(Data=pd.to_datetime(frame["Data"]).astype("datetime64[ns]"))
    for c in ("Competizione", "Partita", "Mercato", "Esito", "Bookmaker", "Flag"):
        frame[c] = frame[c].astype("string")
    for c in ("Quota Book", "Quota Fair", "P. Modello (%)", "EV (%)", "Stake (unità)"):
        frame[c] = frame[c].astype("float64")
    frame = frame.sort_values(["EV (%)", "Partita", "Mercato", "Esito"], ascending=[False, True, True, True], kind="stable")
    return frame.reset_index(drop=True)


def traps_table(state: PipelineState, section: str | None = None) -> pd.DataFrame:
    """REGRESSION_TRAP teams (severity descending) with Italian labels, of one section when given."""
    rows = [
        {
            "Squadra": t.team,
            "Competizione": t.girone.short,
            "Gol Subiti": int(t.goals_against),
            "xGA": round(t.xg_against, 2),
            "Severità": round(t.severity, 2),
            "Ultima Gara": pd.Timestamp(t.last_date),
        }
        for t in sorted(section_traps(state, section), key=lambda t: (-t.severity, t.team))
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


def section_traps(state: PipelineState, section: str | None = None) -> list[RegressionTrap]:
    """The REGRESSION_TRAP teams of one section (every trap for ``None``), in ``state.traps`` order."""
    return [t for t in state.traps if in_section(t.girone, section)]


def section_opportunities(state: PipelineState, section: str | None = None) -> list[ValueOpportunity]:
    """Every evaluated selection of one section (every one for ``None``), in ``state.opportunities`` order."""
    return [o for o in state.opportunities if in_section(o.girone, section)]


def _slug(selection: str) -> str:
    return selection.strip().lower().replace(" ", "_")


def predictions_table(state: PipelineState, section: str | None = None) -> pd.DataFrame:
    """Export frame: one row per predicted fixture (of one section, when given) with rates, every market probability and fair odds.

    Columns: ``match_id, date, round, girone, home_team, away_team, lambda, mu, rho,
    most_likely_score, p_most_likely`` followed by ``p_<selection>`` and
    ``fair_<selection>`` for every selection in :data:`SELECTIONS`
    (``p_1, p_x, p_2, p_over_1.5, ...``).
    """
    rows: list[dict[str, object]] = []
    for p in fixtures_for(state, None, matchday=ALL_MATCHDAYS, section=section):
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


def opportunities_table(state: PipelineState, section: str | None = None) -> pd.DataFrame:
    """The full :meth:`ValueEngine.to_frame` of every evaluated selection (value or not), of one section when given."""
    return ValueEngine().to_frame(section_opportunities(state, section))


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
    section: str | None = None,
) -> list[Pick]:
    """Shared body of :func:`draw_picks_table` / :func:`btts_picks_table`: ranked within the section (and competition)."""
    g = _as_girone(girone)
    preds = _restrict(state, matchday, section)
    if g is not None:
        preds = [p for p in preds if p.fixture.girone == g]
    chooser = draw_picks if draws else btts_picks
    return chooser(preds, n=n, matchday=None, opportunities=state.opportunities)


def draw_picks_table(
    state: PipelineState,
    matchday: MatchdaySelector = None,
    n: int = DEFAULT_N_DRAWS,
    girone: Girone | None = None,
    section: str | None = None,
) -> pd.DataFrame:
    """The ``n`` most probable draws (``X``) of the matchday as a :func:`picks_frame` (Italian columns).

    ``matchday`` defaults to the next one of the section (:data:`ALL_MATCHDAYS`
    for the whole fixture list); ``section`` ranks the candidates within one
    section (Serie B and Serie C together, or Serie D alone) and ``girone``
    optionally within one competition; with neither every competition of the
    perimeter competes, as the user asked.
    """
    return picks_frame(_picks(state, matchday, girone, draws=True, n=n, section=section))


def btts_picks_table(
    state: PipelineState,
    matchday: MatchdaySelector = None,
    n: int = DEFAULT_N_BTTS,
    girone: Girone | None = None,
    section: str | None = None,
) -> pd.DataFrame:
    """The ``n`` most probable both-teams-to-score matches (``GG``) of the matchday, see :func:`draw_picks_table`."""
    return picks_frame(_picks(state, matchday, girone, draws=False, n=n, section=section))


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
