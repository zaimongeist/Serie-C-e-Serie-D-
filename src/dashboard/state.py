"""Pure data layer of the Streamlit dashboard (no ``streamlit`` import).

Everything the UI shows is computed here from the shared contracts, so the
dashboard can be unit-tested without a browser:

* :func:`build_state` runs the whole pipeline (load -> fit -> predict ->
  value) and returns an immutable-by-convention :class:`PipelineState`
  snapshot. It works on **real data only** unless the demo is asked for
  explicitly (``simulate=True``, the dashboard's ``SERIEC_DEMO=1``): without
  usable files in ``data/processed`` it raises :class:`NoDataError`, which the
  dashboard renders as a setup page, and it never falls back to the simulator
  silently (the synthetic league invents teams' results, dates and pairings:
  showing it as if it were real is the worst failure a betting tool can have);
* the calendar views follow the **official rounds** (:mod:`src.engine.calendar`):
  :func:`matchdays` groups the current round of every competition (with its
  matches already played) and the rounds after it, :func:`state_at` regroups a
  cached state at another instant, :func:`round_view_for` /
  :func:`competition_rounds` give any round of one competition and
  :func:`recoveries_for` the postponed matches and isolated recoveries;
* the ``*_table`` helpers turn that snapshot into ``pandas`` frames with
  Italian column labels, ready for ``st.dataframe`` (:func:`matchday_table_for`
  is the whole matchday in one table, :func:`results_table_for` the played
  matches, both from :mod:`src.engine.tables`);
* :func:`heatmap_figure` builds the score-matrix Plotly figure (whole matrix
  by default, or a compact view whose last row/column aggregate the tail);
* :func:`export_csv_bytes` serialises any frame for ``st.download_button``;
  with ``demo=True`` (the demo state, :attr:`PipelineState.is_demo`) every
  export carries the first column ``AVVISO`` = :data:`DEMO_EXPORT_TEXT` and
  :func:`demo_file_name` prefixes the file name with ``DEMO_``, so invented
  numbers never leave the dashboard looking like real ones.

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
:func:`build_state` raises :class:`NoDataError`; the demo (``simulate=True``)
simulates a league on the registry season's real calendar (2026/27 by
default; played rounds derived from ``as_of``, today by default, see
:func:`simulate_current_season`) together with ``history_seasons`` synthetic
seasons *preceding* it (:func:`src.simulation.generator.history_seasons_for`),
and says so in every note (:data:`DEMO_WARNING`).

Sections and fit groups
-----------------------
The user-facing views are split by *section*, read from
``config/competitions.toml`` (:data:`SECTIONS`: ``'Serie C'``, ``'Serie B'``,
``'Serie D'`` with the shipped file; Serie A and the European leagues become
sections of their own once switched on with ``enabled = true``). The fit is one
per *fit group* -- the competitions of one country, fitted jointly by
:func:`src.engine.fitgroups.fit_by_group` (L-BFGS-B by default, SLSQP on
request): Serie B, Serie C and Serie D share one Dixon-Coles estimate (promoted
and relegated clubs connect the divisions; a home advantage and a ``sum(alpha)
= 0`` constraint per competition keep them comparable), a foreign league is
fitted on its own. Every table helper takes ``section`` (``None`` = the whole
perimeter) and restricts itself to the competitions of that section; picks are
ranked within the section and the matchdays are grouped per section
(:func:`matchdays`). The legacy name :data:`SECTION_BC` (the old joint "Serie B
e Serie C" page) is still accepted as an alias of both sections.
:func:`sections` lists the sections that have teams, :func:`section_options`
the ones the sidebar offers (every enabled section: one without data says how
to initialise it, see :attr:`PipelineState.serie_d_status` and
:func:`serie_d_harvest_command`).

Reliability, recalibration and slips
------------------------------------
:func:`build_reliability` runs the **walk-forward** backtest
(:func:`src.engine.backtest.walk_forward`: every past matchday predicted by a
model fitted only on the matches before it) and fits the validated
recalibration on it; :func:`metrics_table_for`, :func:`reliability_table_for`,
:func:`competition_metrics_table`, :func:`picks_record_table` and
:func:`slips_record_table` turn it into the Italian tables of the *Affidabilità
modello* tab. :func:`apply_recalibration` / :func:`raw_state` switch the
production probabilities between the model's own and the recalibrated ones
(reversible; only the markets whose out-of-sample log loss improved).
:func:`slips_for` builds the three slips of a matchday
(:mod:`src.engine.slips`).
"""
from __future__ import annotations

import hashlib
import math
import re
import time
from collections.abc import Iterable, Mapping, Sequence
import dataclasses
import os
from dataclasses import dataclass, field
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Final, Literal

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from src.common.competitions import competitions_of_section, describe_competitions
from src.common.competitions import sections as configured_sections
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
from src.engine.calendar import ROME as ROME_TZ
from src.engine.calendar import CompetitionCalendar, RoundMatch, RoundView, kicked_off, rome_midnight
from src.engine.backtest import ALL_SECTIONS as BACKTEST_ALL_SECTIONS
from src.engine.backtest import (
    DEFAULT_BLOCK,
    DEFAULT_MARKETS,
    BacktestResult,
    PicksTrackRecord,
    SlipsTrackRecord,
    ValueTrackRecord,
    market_comparison,
    metrics,
    picks_track_record,
    reliability,
    slips_track_record,
    value_track_record,
    walk_forward,
)
from src.engine.calibration import (
    DEFAULT_MIN_EVAL,
    DEFAULT_MIN_GAIN,
    DEFAULT_TEST_FRACTION,
    DEFAULT_TUNING_GRID,
    DEFAULT_TUNING_MARKETS,
    RecalibrationModel,
    RollingRecalibration,
    fit_recalibration,
    rolling_recalibration,
    time_split,
)
from src.engine.fitgroups import DEFAULT_FIT_METHOD, GroupedFit, fit_by_group
from src.engine.model import DixonColesConfig, DixonColesEngine, FittedParams
from src.engine.slips import (
    DEFAULT_N_GG as DEFAULT_SLIP_N_GG,
    DEFAULT_N_MIXED as DEFAULT_SLIP_N_MIXED,
    DEFAULT_N_X as DEFAULT_SLIP_N_X,
    INDEPENDENCE_NOTE,
    Slip,
    build_slips,
    make_slip_builder,
    slips_frame,
    slips_summary,
)
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
    RECOVERIES_COLUMNS,
    RESULTS_COLUMNS,
    RESULTS_MODEL_COLUMNS,
    matchday_table,
    recoveries_table,
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
    DEMO_WARNING,
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
    "DEMO_BADGE",
    "DEMO_ENV_VAR",
    "DEMO_EXPORT_COLUMN",
    "DEMO_EXPORT_TEXT",
    "DEMO_FILE_PREFIX",
    "DEMO_WARNING",
    "LEGACY_DEMO_ENV_VAR",
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
    "RECOVERIES_COLUMNS",
    "RESULTS_COLUMNS",
    "RESULTS_MODEL_COLUMNS",
    "ACCUMULATORS_COLUMNS",
    "ALL_SECTIONS_LABEL",
    "BACKTEST_LOGIC_VERSION",
    "BACKTEST_MARKET_LABELS",
    "COMPETITION_METRICS_COLUMNS",
    "DEFAULT_BACKTEST_SEASONS",
    "EVIDENCE_NOTE",
    "INDEPENDENCE_NOTE",
    "PICKS_BY_MATCHDAY_COLUMNS",
    "PICKS_RECORD_COLUMNS",
    "PRIOR_SETTINGS",
    "QUOTED_BRIER_COLUMN",
    "QUOTED_LOG_LOSS_COLUMN",
    "RAW_COMPARE_COLUMNS",
    "RECALIBRATION_COLUMNS",
    "RECALIBRATION_NOTE_PREFIX",
    "RELIABILITY_BINS",
    "RELIABILITY_EXPORT_COLUMNS",
    "RELIABILITY_METRICS_COLUMNS",
    "RELIABILITY_SELECTIONS",
    "SECTIONS",
    "SECTION_ALIASES",
    "SECTION_B",
    "SECTION_BC",
    "SECTION_C",
    "SECTION_D",
    "SECTION_FAMILIES",
    "SECTION_SLUGS",
    "SLIPS_BY_MATCHDAY_COLUMNS",
    "SLIPS_EXPORT_COLUMNS",
    "slips_by_matchday_table",
    "SLIPS_RECORD_COLUMNS",
    "SLIP_TITLES",
    "TUNING_COLUMNS",
    "ReliabilityReport",
    "ValidatedTuning",
    "accumulators_table",
    "apply_recalibration",
    "backtest_config_rule",
    "backtest_seasons",
    "build_reliability",
    "competition_metrics_table",
    "config_fingerprint",
    "default_slips_matchday",
    "headline_for",
    "known_competition_rows",
    "known_result",
    "market_card",
    "metrics_table_for",
    "model_config",
    "picks_by_matchday_table",
    "picks_record_table",
    "picks_track_for",
    "prior_setting",
    "raw_column_name",
    "raw_state",
    "recalibration_table",
    "reliability_export_frame",
    "reliability_figure",
    "reliability_for_state",
    "reliability_key",
    "reliability_table_for",
    "resolve_section",
    "rolling_frame",
    "section_members",
    "section_of_girone",
    "sections_with_data",
    "slips_csv_bytes",
    "slips_export_frame",
    "grouped_results_table",
    "slips_for",
    "slips_gironi",
    "slips_matchdays",
    "slips_record_table",
    "slips_summary_for",
    "slips_track_for",
    "tuning_disclosure",
    "tuning_table",
    "validated_tuning",
    "value_engine_for",
    "value_track_for",
    "with_raw_columns",
    "with_rolling",
    "SERIE_D_FAMILY",
    "SERIE_D_HARVEST_SEASONS",
    "SERIE_D_STATUS_NOT_CONFIGURED",
    "SERIE_D_STATUS_NOT_INITIALISED",
    "TRAPS_COLUMNS",
    "VALUE_COLUMNS",
    "CoverageReport",
    "MatchdaySelector",
    "NoDataError",
    "PipelineState",
    "Source",
    "all_sections",
    "btts_picks_table",
    "build_state",
    "competition_label",
    "competition_options",
    "competition_rounds",
    "configured_gironi",
    "coverage_report",
    "coverage_text",
    "current_round_for",
    "current_teams",
    "demo_file_name",
    "demo_requested",
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
    "load_enabled_csv",
    "load_odds_quotes",
    "markets_table",
    "matchday_csv_bytes",
    "matchday_predictions",
    "matchday_table_for",
    "matchday_table_key",
    "matchdays",
    "mark_demo_frame",
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
    "recoveries_for",
    "recoveries_table_for",
    "resolve_matchday",
    "results_csv_bytes",
    "results_table_for",
    "results_team_options",
    "round_table_for",
    "round_view_for",
    "season_fixtures",
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
    "state_at",
    "strength_figure",
    "traps_table",
    "value_opportunities",
    "value_table",
]

log = get_logger(__name__)

Source = Literal["processed", "simulated"]

#: Environment variable that switches the dashboard (and only it) to the synthetic demo league.
DEMO_ENV_VAR: Final[str] = "SERIEC_DEMO"
#: Deprecated alias of :data:`DEMO_ENV_VAR`, still honoured (with a logged deprecation warning).
LEGACY_DEMO_ENV_VAR: Final[str] = "SERIEC_FORCE_SIMULATE"
#: Text of the source badge in demo mode.
DEMO_BADGE: Final[str] = "Demo"
#: Text of the first column of every CSV exported from the demo (dashboard downloads and ``main.py`` files).
DEMO_EXPORT_TEXT: Final[str] = "DATI SIMULATI - NON USARE PER SCOMMETTERE"
#: Name of that first column.
DEMO_EXPORT_COLUMN: Final[str] = "AVVISO"
#: Prefix of the file name of every CSV exported from the demo.
DEMO_FILE_PREFIX: Final[str] = "DEMO_"
_TRUTHY: Final[frozenset[str]] = frozenset({"1", "true", "yes", "on", "si", "sì"})


def demo_requested(environ: Mapping[str, str] | None = None) -> bool:
    """Whether the synthetic demo was asked for: ``SERIEC_DEMO=1`` (or the deprecated ``SERIEC_FORCE_SIMULATE=1``).

    ``1/true/yes/on/si`` count as yes, case-insensitively. The demo is never
    the default: without one of the two variables the dashboard shows real
    data or the setup page, never invented numbers.
    """
    env = os.environ if environ is None else environ
    if str(env.get(DEMO_ENV_VAR, "")).strip().lower() in _TRUTHY:
        return True
    if str(env.get(LEGACY_DEMO_ENV_VAR, "")).strip().lower() in _TRUTHY:
        log.warning(
            "deprecated environment variable: use the demo switch instead",
            extra={"variable": LEGACY_DEMO_ENV_VAR, "replacement": DEMO_ENV_VAR},
        )
        return True
    return False


class NoDataError(RuntimeError):
    """No usable real data: :func:`build_state` refuses to invent it.

    ``reasons`` lists, in Italian, what is missing or unusable (``matches.csv``
    absent, empty or unreadable); ``processed_dir`` is the directory that was
    read. The dashboard renders it as the setup page (how to provide the
    FootyStats key, the button that downloads the data) instead of tabs with
    simulated numbers; ``main.py`` prints it and exits.
    """

    def __init__(self, reasons: Sequence[str], *, processed_dir: Path) -> None:
        self.reasons: tuple[str, ...] = tuple(reasons) or ("nessun dato reale disponibile",)
        self.processed_dir = Path(processed_dir)
        super().__init__("Nessun dato reale disponibile: " + "; ".join(self.reasons))

#: Selector value meaning "every matchday" for :func:`fixtures_for`, :func:`value_table` and
#: :func:`matchday_predictions`; ``None`` means "the next matchday" and a :class:`Matchday` that one.
ALL_MATCHDAYS: Final = "all"
MatchdaySelector = Matchday | Literal["all"] | None

#: Columns of :func:`picks_export_frame`: ``Lista`` (``'X'`` / ``'GG'``) and ``Giornata`` in front of :data:`PICKS_COLUMNS`.
PICKS_EXPORT_COLUMNS: Final[tuple[str, ...]] = ("Lista", "Giornata", *PICKS_COLUMNS)

#: The user-facing *sections* of the dashboard and the console come from ``config/competitions.toml`` (the
#: ``section`` key of every enabled competition, default its family): ``'Serie C'`` (gironi A/B/C), ``'Serie B'``
#: and ``'Serie D'`` (gironi A-I) with the shipped file, plus every league switched on later (``'Serie A'``,
#: ``'Premier League'`` ...), each its own page. The fit is one per fit group (country), the views are split.
SECTIONS: Final[tuple[str, ...]] = tuple(configured_sections())
SECTION_C: Final[str] = "Serie C"
SECTION_B: Final[str] = "Serie B"
SECTION_D: Final[str] = "Serie D"
#: Legacy name of the old joint "Serie B e Serie C" page (Serie B and Serie C are now two sections): still accepted
#: by every helper as an *alias* of both sections (``configured_gironi(SECTION_BC)`` = the gironi A/B/C and Serie B,
#: ``main.py --section bc``), never offered by the sidebar.
SECTION_BC: Final[str] = "Serie B e Serie C"
#: Aliases of several sections (helpers accept them wherever a section is expected): the legacy :data:`SECTION_BC`.
SECTION_ALIASES: Final[dict[str, tuple[str, ...]]] = {SECTION_BC: tuple(s for s in (SECTION_C, SECTION_B) if s in SECTIONS)}
#: Section shown by default (the first option of the sidebar radio and the one whose widget keys carry no suffix):
#: the first section of the configuration ('Serie C' with the shipped file; 'Serie A' once it is enabled).
DEFAULT_SECTION: Final[str] = SECTIONS[0] if SECTIONS else SECTION_C
#: Option of the reliability tab, the slips tab and the console meaning every section together.
ALL_SECTIONS_LABEL: Final[str] = BACKTEST_ALL_SECTIONS
#: Competition families covered by each configured section (``config/competitions.toml`` ``family`` values).
#: A family listed in no section is a section of its own, named after the family.
SECTION_FAMILIES: Final[dict[str, tuple[str, ...]]] = {
    sec: tuple(dict.fromkeys(c.family for c in competitions_of_section(sec))) for sec in SECTIONS
}
#: Short file-name / widget-key suffixes of the sections (``main.py`` exports ``<stem>_d_giornata.csv``, the
#: dashboard keys ``matchday_d`` ...): ``'Serie X'`` -> ``'x'``; another section gets a slug of its name
#: (:func:`section_slug`, ``'Premier League'`` -> ``'premier_league'``).
SECTION_SLUGS: Final[dict[str, str]] = {SECTION_BC: "bc"}
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
    (``'processed'``) or from the synthetic generator (``'simulated'``, the demo
    only: :attr:`is_demo`, :attr:`demo_warning`); ``notes`` is the
    human-readable log of what happened (data source, injected trap, fit
    diagnostics, skipped quotes...) shown in the *Dettagli pipeline* expander.
    ``now`` is the reference instant of the calendar (the official-round rule
    of :mod:`src.engine.calendar`; ``None`` = the fixture list already is the
    upcoming schedule, as in the demo): :func:`state_at` regroups a state at
    another instant without refitting.

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
    the calendar into "giornate" (:func:`src.engine.picks.matchdays` on the
    current-season fixtures and played matches at ``now``): the first one is
    the current round of every competition, with its matches already played,
    so the match selector, the value panel and the picks default to
    ``matchdays[0]`` (:attr:`next_matchday`). The per-section matchdays of the
    dashboard are :func:`matchdays` ``(state, section)``, cached on the state.

    The perimeter may span several *families* (:attr:`families`: ``'Serie C'``,
    ``'Serie B'``, ``'Serie D'``); :attr:`serie_d_status` says whether the Serie D
    section is usable (its gironi are initialised from the FootyStats API, not
    shipped with the code).
    """

    matches: pd.DataFrame
    fixtures: list[Fixture]
    quotes: list[OddsQuote]
    engine: "DixonColesEngine | GroupedFit"
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
    now: datetime | None = None
    #: Value-engine settings the opportunities were evaluated with (re-used when the probabilities are recalibrated).
    min_ev: float = 0.035
    kelly_fraction: float = 0.125
    bankroll_units: float = 100.0
    #: Model settings of the fit: time decay and prior (``'auto'`` = coverage rule, ``'on'``, ``'off'`` or a
    #: ``sigma_strength``), the backtest replays them (:func:`run_backtest`).
    xi: float = 0.0018
    prior: str = "auto"
    #: The model's own probabilities and opportunities when :attr:`predictions` / :attr:`opportunities` hold
    #: recalibrated ones (:func:`apply_recalibration`), ``None`` otherwise.
    raw_predictions: list[MatchPrediction] | None = None
    raw_opportunities: list[ValueOpportunity] | None = None
    #: The recalibration applied to :attr:`predictions` (``None`` = raw probabilities).
    recalibration: RecalibrationModel | None = None
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
        """The current matchday (``matchdays[0]``: the current round of every competition), ``None`` without a round to play."""
        return self.matchdays[0] if self.matchdays else None

    @property
    def is_demo(self) -> bool:
        """Whether the state is the synthetic demo (``source == 'simulated'``): invented teams' results, dates and odds."""
        return self.source == "simulated"

    @property
    def demo_warning(self) -> str:
        """:data:`DEMO_WARNING` for the demo, ``''`` for real data."""
        return DEMO_WARNING if self.is_demo else ""

    @property
    def calendar_fixtures(self) -> list[Fixture]:
        """The loaded fixtures dated in the registry season (the calendar of the current rounds), in load order."""
        return season_fixtures(self.fixtures, self.season_label)

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

    @property
    def calibrated(self) -> bool:
        """Whether :attr:`predictions` carry recalibrated probabilities (:func:`apply_recalibration` with an enabled model)."""
        return self.raw_predictions is not None and self.recalibration is not None and self.recalibration.enabled

    @property
    def fit_groups(self) -> tuple[str, ...]:
        """The fit groups of the estimate (``('Italy',)`` with the shipped configuration)."""
        engine = self.engine
        return engine.groups if isinstance(engine, GroupedFit) else ("",)

    @property
    def fit_method(self) -> str:
        """The optimiser of the fit (``'L-BFGS-B'`` by default, ``'SLSQP'`` on request)."""
        return str(self.engine.config.method)


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


def season_fixtures(fixtures: Iterable[Fixture], season_label: str) -> list[Fixture]:
    """The fixtures dated in ``season_label`` (``'2026/27'`` or ``'2026_27'``), in input order: the calendar of the current season."""
    label = str(season_label).replace("_", "/")
    return [f for f in fixtures if season_of(f.date) == label]


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
def section_members(section: str) -> tuple[str, ...]:
    """The configured sections a section name stands for: itself, or the members of an alias (:data:`SECTION_ALIASES`)."""
    return SECTION_ALIASES.get(section, (str(section),))


def section_of_family(family: str) -> str:
    """Section a competition family belongs to (:data:`SECTION_FAMILIES`).

    ``'Serie C'`` -> ``'Serie C'``, ``'Serie B'`` -> ``'Serie B'``, ``'Serie D'``
    -> :data:`SECTION_D` with the shipped configuration; a family listed in no
    section (a league added later) is a section of its own, named after the family.
    """
    for section, members in SECTION_FAMILIES.items():
        if family in members:
            return section
    return str(family)


def family_label(family: str) -> str:
    """User-facing label of the section a family is shown under: ``family_label('Serie C') == 'Serie C'``.

    Not to be confused with :func:`src.common.competitions.family_label`, which
    returns the canonical spelling of a family name.
    """
    return section_of_family(family)


def section_families(section: str) -> tuple[str, ...]:
    """Families covered by a section (``('Serie C', 'Serie B')`` for the alias :data:`SECTION_BC`; an unknown section is its own family)."""
    families: list[str] = []
    for member in section_members(section):
        families.extend(SECTION_FAMILIES.get(member, (str(member),)))
    return tuple(dict.fromkeys(families))


def section_slug(section: str) -> str:
    """Short identifier of a section for file names and widget keys.

    ``'Serie C'`` -> ``'c'``, ``'Serie B'`` -> ``'b'``, ``'Serie D'`` -> ``'d'``,
    the legacy alias :data:`SECTION_BC` -> ``'bc'`` (:data:`SECTION_SLUGS`),
    any other name a slug of it (``'Premier League'`` -> ``'premier_league'``).
    """
    slug = SECTION_SLUGS.get(section)
    if slug is None:
        match = re.fullmatch(r"Serie ([A-Z])", str(section).strip())
        if match:
            return match.group(1).lower()
        slug = re.sub(r"[^a-z0-9]+", "_", str(section).lower()).strip("_") or "sezione"
    return slug


def resolve_section(value: str) -> str:
    """The configured section (or alias) a user-typed name refers to, case- and separator-insensitive.

    ``'serie-b'``, ``'Serie_B'``, ``'b'`` (the slug) -> ``'Serie B'``; ``'bc'``
    -> :data:`SECTION_BC`; :class:`ValueError` for an unknown name. Used by
    ``main.py --section``.
    """
    wanted = re.sub(r"[\s_\-]+", " ", str(value).strip().lower())
    for section in (*SECTIONS, *SECTION_ALIASES):
        if wanted in (section.lower(), section_slug(section), section_slug(section).replace("_", " ")):
            return section
    raise ValueError(f"unknown section {value!r}; known: {', '.join(SECTIONS)} (aliases: {', '.join(SECTION_SLUGS.values())})")


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


def _section_order(section: str) -> int:
    return SECTIONS.index(section) if section in SECTIONS else len(SECTIONS)


def section_of_girone(girone: Girone | str) -> str:
    """The configured section of a competition (``'A'`` -> ``'Serie C'``, ``'SB'`` -> ``'Serie B'``, ``'DA'`` -> ``'Serie D'``)."""
    return str(Girone.parse(girone).section)


def all_sections(gironi: Iterable[Girone] = Girone) -> tuple[str, ...]:
    """Sections of the given competitions in configuration order (``('Serie C', 'Serie B', 'Serie D')`` for every competition)."""
    present = list(dict.fromkeys(section_of_girone(g) for g in gironi))
    return tuple(sorted(present, key=lambda sec: (_section_order(sec), present.index(sec))))


def sections_for(gironi: Iterable[Girone]) -> tuple[str, ...]:
    """The sections that have at least one of ``gironi`` (e.g. a registry's competitions with teams), display order."""
    return all_sections(gironi)


def configured_gironi(section: str | None = None) -> tuple[Girone, ...]:
    """Every configured competition of a section (``config/competitions.toml``), teams or not; all of them for ``None``.

    An alias (:data:`SECTION_BC`) yields the competitions of its member
    sections; a name that is no configured section is read as a family.
    """
    if section is None:
        return tuple(Girone)
    members = section_members(section)
    found = tuple(g for g in Girone if g.section in members)
    if found or any(m in SECTIONS for m in members):
        return found
    wanted = section_families(section)
    return tuple(g for g in Girone if g.family in wanted)


def in_section(girone: Girone | str, section: str | None) -> bool:
    """Whether a competition belongs to ``section`` (``None`` = every section; an alias covers its member sections)."""
    if section is None:
        return True
    g = Girone.parse(girone)
    members = section_members(section)
    if g.section in members:
        return True
    return not any(m in SECTIONS for m in members) and g.family in section_families(section)


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


#: Values of the ``prior`` model setting besides a number (the ``sigma_strength`` of the Gaussian prior).
PRIOR_SETTINGS: Final[tuple[str, ...]] = ("auto", "on", "off")


def prior_setting(value: object) -> str:
    """Normalise a prior setting: ``'auto'`` (coverage rule), ``'on'``, ``'off'`` or a positive ``sigma_strength`` as text.

    ``None``/``'auto'`` -> ``'auto'``, ``True``/``'on'`` -> ``'on'``,
    ``False``/``'off'``/``0`` -> ``'off'``, ``0.4``/``'0.4'`` -> ``'0.4'``;
    :class:`ValueError` otherwise.
    """
    if value is None:
        return "auto"
    if isinstance(value, bool):
        return "on" if value else "off"
    text = str(value).strip().lower()
    if text in PRIOR_SETTINGS:
        return text
    try:
        number = float(text)
    except ValueError as exc:
        raise ValueError(f"prior must be 'auto', 'on', 'off' or a positive sigma, got {value!r}") from exc
    if number == 0.0:
        return "off"
    if not (math.isfinite(number) and number > 0.0):
        raise ValueError(f"prior sigma must be > 0, got {value!r}")
    return f"{number:g}"


def model_config(
    matches: pd.DataFrame,
    *,
    xi: float,
    prior: object = "auto",
    registry: TeamRegistry | None = None,
    season: str | None = None,
    method: str = DEFAULT_FIT_METHOD,
) -> tuple[DixonColesConfig, bool]:
    """The production :class:`DixonColesConfig` for ``matches``; returns ``(config, prior_on)``.

    ``prior`` (:func:`prior_setting`): ``'auto'`` applies :func:`prior_recommended`
    on the current-season coverage, ``'on'``/``'off'`` force it, a number is the
    ``sigma_strength`` of an always-on prior (the value the walk-forward tuning
    of :func:`src.engine.calibration.tune_hyperparameters` may recommend).
    ``method`` is the optimiser (``'L-BFGS-B'`` by default, same optimum as
    ``'SLSQP'`` about 40 times faster).
    """
    setting = prior_setting(prior)
    if setting == "auto":
        prior_on = prior_recommended(matches, registry=registry, season=season)
    else:
        prior_on = setting != "off"
    if not prior_on:
        return DixonColesConfig(xi=xi, method=method), False  # type: ignore[arg-type]
    extra: dict[str, float] = {} if setting in PRIOR_SETTINGS else {"sigma_strength": float(setting)}
    return DixonColesConfig.with_gaussian_prior(xi=xi, method=method, **extra), True  # type: ignore[arg-type]


def _model_config(
    matches: pd.DataFrame,
    *,
    xi: float,
    use_prior: bool | None,
    registry: TeamRegistry | None = None,
    season: str | None = None,
    method: str = DEFAULT_FIT_METHOD,
) -> tuple[DixonColesConfig, bool]:
    """Resolve the ``use_prior`` tri-state into a config; returns ``(config, prior_on)`` (see :func:`model_config`)."""
    return model_config(matches, xi=xi, prior=prior_setting(use_prior), registry=registry, season=season, method=method)


def load_enabled_csv(path: Path, loader, *, notes: list[str] | None = None, what: str = "righe"):  # noqa: ANN001, ANN201 - a loader and its result
    """``loader(path)`` that skips the rows of competitions no longer enabled instead of refusing the whole file.

    A league switched off in ``config/competitions.toml`` (``enabled = false``)
    after its data was downloaded leaves rows with an unknown code in
    ``matches.csv`` / ``fixtures.csv``; the strict loaders of
    :mod:`src.ingestion.harvester` refuse such a file, which would turn the
    dashboard into the setup page. Here those rows are dropped (logged, and an
    Italian line appended to ``notes``) and the rest is loaded; the next data
    refresh rewrites the files with the enabled leagues only. Any other error is
    raised unchanged.
    """
    try:
        return loader(Path(path))
    except ValueError:
        try:
            raw = pd.read_csv(path, dtype=str, keep_default_na=False)
        except (OSError, pd.errors.ParserError, pd.errors.EmptyDataError, UnicodeDecodeError):
            raise
        if "girone" not in raw.columns:
            raise

        def enabled(value: str) -> bool:
            try:
                Girone.parse(value)
            except ValueError:
                return False
            return True

        mask = raw["girone"].map(enabled).to_numpy(dtype=bool)
        if mask.all():
            raise
        dropped = sorted(set(raw.loc[~mask, "girone"].astype(str)))
        import tempfile  # noqa: PLC0415 - only on this rare path

        with tempfile.TemporaryDirectory() as tmp:
            target = Path(tmp) / Path(path).name
            raw.loc[mask].to_csv(target, index=False)
            result = loader(target)
        message = (
            f"{int((~mask).sum())} {what} di competizioni non abilitate ({', '.join(dropped)}) ignorate in {Path(path).name}: "
            "il campionato è stato disattivato in config/competitions.toml dopo il download; il prossimo aggiornamento dei dati le rimuove."
        )
        log.warning("rows of competitions not enabled skipped", extra={"file": str(path), "codes": dropped, "rows": int((~mask).sum())})
        if notes is not None:
            notes.append(message)
        return result


def _load_processed_inputs(
    processed_dir: Path, odds_dir: Path, notes: list[str], reasons: list[str]
) -> tuple[pd.DataFrame, list[Fixture], list[OddsQuote]] | None:
    """Load matches/fixtures/odds from disk; ``None`` (with the Italian ``reasons``) when there is nothing usable.

    ``matches.csv`` missing, unreadable or empty means no real data; a missing
    or unreadable ``fixtures.csv`` and missing odds are noted but do not stop
    the pipeline (the played matches still give the ranking and the results).
    """
    matches_path = Path(processed_dir) / "matches.csv"
    fixtures_path = Path(processed_dir) / "fixtures.csv"
    if not matches_path.is_file():
        reasons.append(f"il file {matches_path} non esiste: i dati FootyStats non sono ancora stati scaricati")
        log.warning("processed matches missing", extra={"path": str(matches_path)})
        return None
    try:
        matches = load_enabled_csv(matches_path, load_processed, notes=notes, what="partite")
    except (ValueError, OSError, pd.errors.ParserError) as exc:
        log.warning("processed matches unreadable", extra={"path": str(matches_path), "error": str(exc)})
        reasons.append(f"il file {matches_path} non è leggibile ({exc})")
        return None
    if matches.empty:
        reasons.append(f"il file {matches_path} non contiene partite giocate")
        log.warning("processed matches empty", extra={"path": str(matches_path)})
        return None
    fixtures: list[Fixture] = []
    if fixtures_path.exists():
        try:
            fixtures = load_enabled_csv(fixtures_path, load_fixtures, notes=notes, what="partite in calendario")
        except (ValueError, OSError, pd.errors.ParserError) as exc:
            log.warning("fixtures unreadable", extra={"path": str(fixtures_path), "error": str(exc)})
            notes.append(f"File {fixtures_path} non leggibile ({exc}): nessuna partita in programma.")
    else:
        notes.append(f"Nessun file {fixtures_path.name}: nessuna partita in programma da prevedere.")
    quotes = load_odds_quotes(Path(odds_dir))
    notes.append(f"Dati reali da {matches_path}: {len(matches)} partite, {len(fixtures)} partite in calendario, {len(quotes)} quote da {odds_dir}.")
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

    Shared by the dashboard demo and ``main.py --mode full`` so both read
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
    simulate: bool | None = False,
    use_prior: bool | None = None,
    season: str = DEFAULT_SEASON,
    history_seasons: int = DEFAULT_HISTORY_SEASONS,
    as_of: date | None = None,
    registry: TeamRegistry | None = None,
    now: datetime | None = None,
    method: str = DEFAULT_FIT_METHOD,
    prior: object = None,
) -> PipelineState:
    """Run load -> fit -> predict -> value and return a :class:`PipelineState`.

    Perimeter
        ``season`` selects the registry (``config/teams_<season>.json``, 2026/27
        by default; ``registry`` overrides it, e.g. for a simulated composition).
        Its teams -- in every competition it holds, the three Serie C gironi and
        Serie B with the shipped files -- are the only ones ranked, predicted and
        priced: the fit sees every match of the frame, but teams outside the
        registry are fitted as *historical* opponents only. The rows of the
        registry season are the current matches; older rows are history.
    Data source
        Real data only, unless ``simulate=True``. ``processed_dir/matches.csv``
        (several seasons allowed) with ``fixtures.csv`` alongside it when
        present and the odds of ``odds_dir/*.csv`` (:func:`load_odds_quotes`
        format). When ``matches.csv`` is missing, unreadable or empty
        :class:`NoDataError` is raised: there is **no silent fallback** to the
        simulator (``simulate=None`` is the same as ``False``). ``simulate=True``
        is the explicit demo: a league simulated with
        :func:`simulate_current_season` on the registry season's real calendar
        (with ``n_rounds`` the played rounds are fixed, with ``n_rounds=None``
        they are the rounds dated before ``as_of``, today by default) plus
        ``history_seasons`` (0-2) synthetic seasons preceding it from
        :func:`simulate_history`, a REGRESSION_TRAP injected on one team of the
        first competition, and :data:`DEMO_WARNING` as the first note.
    Calendar
        The matchdays follow the official rounds at the reference instant
        ``now`` (:func:`src.engine.picks.matchdays` with the current-season
        played matches): for real data ``now`` defaults to midnight of
        ``as_of`` in ``Europe/Rome`` when ``as_of`` is given, else to the
        current instant; for the demo it defaults to ``None`` (the simulated
        fixture list is the next round by construction). A naive ``now`` is
        read as UTC. :func:`state_at` regroups an existing state at another
        instant.
    Model
        ``use_prior=None`` enables the Gaussian prior while the least-covered
        girone has fewer than :data:`PRIOR_MIN_MATCHES_PER_TEAM` played
        *current-season* matches per current team (see
        :func:`prior_recommended`; history never counts as coverage); ``True``
        forces :meth:`DixonColesConfig.with_gaussian_prior`, ``False`` the plain
        MLE; ``prior`` (when given) overrides ``use_prior`` with a
        :func:`prior_setting` (``'auto'``, ``'on'``, ``'off'`` or a
        ``sigma_strength``). ``xi`` is the time-decay rate per day. The
        competitions are fitted per *fit group* (their country,
        :func:`src.engine.fitgroups.fit_by_group`: one joint model for the
        Italian perimeter, one per foreign league), with ``method``
        (``'L-BFGS-B'`` by default, ``'SLSQP'`` reaches the same optimum about
        40 times slower).
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
    if now is not None and now.tzinfo is None:
        now = now.replace(tzinfo=UTC)  # a naive instant is UTC, as in src.engine.calendar
    if simulate is True:
        source: Source = "simulated"
        notes.append(DEMO_WARNING)
        current, history, fixtures, quotes = _simulated_inputs(
            seed, n_rounds, notes, registry=reg, history_seasons=int(history_seasons), as_of=as_of
        )
        reference = now
    else:
        reasons: list[str] = []
        loaded = _load_processed_inputs(Path(processed_dir), Path(odds_dir), notes, reasons)
        if loaded is None:
            log.warning("no real data: the pipeline refuses to simulate silently", extra={"processed_dir": str(processed_dir), "reasons": reasons})
            raise NoDataError(reasons, processed_dir=Path(processed_dir))
        source = "processed"
        all_matches, fixtures, quotes = loaded
        current, history = split_by_season(all_matches, season_label)
        reference = now if now is not None else (rome_midnight(as_of) if as_of is not None else datetime.now(UTC))
    matches = validate_matches_df(pd.concat([history, current], ignore_index=True)) if not history.empty else current
    if matches.empty:
        raise ValueError(f"no played matches available for the {season_label} perimeter")

    coverage = coverage_report(current, registry=reg, season=season_label)
    setting = prior_setting(use_prior if prior is None else prior)
    config, prior_on = model_config(current, xi=xi, prior=setting, registry=reg, season=season_label, method=method)
    engine = fit_by_group(matches, registry=reg, config=config, fixtures=season_fixtures(fixtures, season_label), method=None)
    params = engine.params_
    notes.extend(engine.notes)
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
    if len(engine.groups) > 1:
        notes.extend(engine.describe())
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
            "prior_setting": setting,
            "fit_groups": list(engine.groups),
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
    giornate = group_matchdays(season_fixtures(fixtures, season_label), played=current, now=reference)
    notes.append(_calendar_note(giornate, reference))
    log.info(
        "matchdays grouped",
        extra={
            "n_matchdays": len(giornate),
            "n_predicted": len(predictions),
            "next": giornate[0].key if giornate else None,
            "now": None if reference is None else reference.isoformat(),
        },
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
        now=reference,
        min_ev=float(min_ev),
        kelly_fraction=float(kelly_fraction),
        bankroll_units=float(bankroll_units),
        xi=float(xi),
        prior=setting,
    )


def _calendar_note(giornate: Sequence[Matchday], reference: datetime | None) -> str:
    """The *Dettagli pipeline* line about the calendar: the current matchday and how many follow."""
    when = "" if reference is None else f" al {reference.astimezone(ROME_TZ).strftime('%d/%m/%Y %H:%M')}"
    if not giornate:
        return f"Calendario{when}: nessun turno da giocare nelle partite in calendario."
    first = giornate[0]
    played = f", {first.n_played} già giocate" if first.n_played else ""
    played += f", {first.n_in_play} in corso o in attesa del risultato" if first.n_in_play else ""
    return (
        f"Calendario{when}: giornata corrente {first.label} ({first.n_fixtures} partite da giocare{played}); "
        f"{len(giornate) - 1} giornate successive in calendario. Selettore partite, pannello valore e selezioni di giornata "
        "partono dalla giornata corrente."
    )


def state_at(state: PipelineState, now: datetime | None) -> PipelineState:
    """``state`` with its calendar regrouped at the instant ``now`` (no refit): same matches, predictions and prices.

    The official-round rule depends on the time (a round stops being current
    when its last matches have kicked off), while the fit does not: the
    dashboard caches the fitted state and calls this on every run. Returns
    ``state`` itself when ``now`` equals :attr:`PipelineState.now`; otherwise a
    copy (``dataclasses.replace``, so the per-state caches start afresh).
    """
    if now == state.now:
        return state
    giornate = group_matchdays(state.calendar_fixtures, played=state.current_matches, now=now)
    log.debug("calendar regrouped", extra={"now": None if now is None else now.isoformat(), "n_matchdays": len(giornate)})
    return dataclasses.replace(state, matchdays=giornate, now=now)


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
    """The sections that have at least one competition with current teams, configuration order.

    ``('Serie C', 'Serie B')`` with a dictionary whose Serie D gironi are still
    empty (the FootyStats harvester initialises them), ``('Serie C', 'Serie B',
    'Serie D')`` once Serie D has teams.
    """
    return sections_for(state.perimeter_gironi)


def section_options(state: PipelineState) -> tuple[str, ...]:
    """The sections the sidebar radio offers: every **enabled** section of ``config/competitions.toml``, configuration order.

    The sections with data come with their competitions; a section configured
    but without teams or matches yet (Serie D before its first harvest, a league
    just switched on with ``enabled = true``) is offered too, so that its page
    can say how to initialise it (:attr:`PipelineState.serie_d_status`, the
    harvest command) instead of hiding it. Never empty: :data:`DEFAULT_SECTION`
    when the configuration has no section at all.
    """
    present = list(dict.fromkeys([*SECTIONS, *sections(state)]))
    if not present:
        return (DEFAULT_SECTION,)
    return tuple(sorted(present, key=lambda sec: (_section_order(sec), present.index(sec))))


def sections_with_data(state: PipelineState) -> tuple[str, ...]:
    """The sections with at least one **fitted** current team (:func:`section_ready`), configuration order.

    The default choice of the slips tab and of the console's *SCHEDINE*
    section: a section without matches in the data has nothing to propose.
    """
    return tuple(sec for sec in section_options(state) if section_ready(state, sec))


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
    """The "giornate" of a section, built from **its own** competitions' official rounds (``state.matchdays`` for ``None``).

    The first one is the current round of every competition of the section
    (:func:`src.engine.picks.matchdays` on :attr:`PipelineState.calendar_fixtures`
    and the current-season played matches, at :attr:`PipelineState.now`).
    Serie D plays 34 rounds on its own calendar: grouped together with Serie B
    and Serie C its rounds would rename every shared weekend, so each section
    groups its competitions separately and its keys, labels and rounds only
    ever mention that section. Grouped once per section and cached on the state.
    """
    if section is None:
        return list(state.matchdays)
    cached = state._section_matchdays.get(section)
    if cached is None:
        cached = group_matchdays(
            state.calendar_fixtures, gironi=configured_gironi(section), played=state.current_matches, now=state.now
        )
        state._section_matchdays[section] = cached
        log.debug("section matchdays grouped", extra={"section": section, "n_matchdays": len(cached)})
    return list(cached)


# ----------------------------------------------------------------------------- official rounds of one competition
def _calendar(state: PipelineState, girone: Girone | str) -> CompetitionCalendar:
    """The :class:`~src.engine.calendar.CompetitionCalendar` of one competition (current-season played matches and fixtures)."""
    return CompetitionCalendar.build(girone, state.current_matches, state.calendar_fixtures)


def competition_rounds(state: PipelineState, girone: Girone | str) -> list[int]:
    """Every round of a competition with at least one match in the current season (played, to play, postponed), ascending.

    The options of the *Giornata n.* selector of the Giornata tab (one
    competition selected in the sidebar); ``[]`` for a competition without
    matches.
    """
    return _calendar(state, girone).rounds()


def current_round_for(state: PipelineState, girone: Girone | str) -> RoundView | None:
    """The current round of a competition at :attr:`PipelineState.now` (see :mod:`src.engine.calendar`), ``None`` when none is pending."""
    return _calendar(state, girone).current(state.now)


def round_view_for(state: PipelineState, girone: Girone | str, round_number: int) -> RoundView:
    """Every match of one round of a competition at :attr:`PipelineState.now` (played with the score, in play, to play, postponed); empty for an unknown round."""
    return _calendar(state, girone).view(int(round_number), state.now)


def recoveries_for(state: PipelineState, section: str | None = None, girone: Girone | None = None) -> list[RoundMatch]:
    """Postponed matches and recoveries outside the current rounds, of one competition or of every competition of a section.

    :meth:`src.engine.calendar.CompetitionCalendar.recoveries` at
    :attr:`PipelineState.now` for each competition (the *Recuperi e rinvii*
    expander of the Giornata tab), chronological; ``[]`` for a competition
    outside ``section`` (:func:`outside_section`).
    """
    g = _as_girone(girone)
    if outside_section(g, section):
        return []
    gironi = (g,) if g is not None else configured_gironi(section)
    fixtures = state.calendar_fixtures
    codes = {f.girone for f in fixtures}
    out: list[RoundMatch] = []
    for x in gironi:
        if x in codes:
            out.extend(CompetitionCalendar.build(x, state.current_matches, fixtures).recoveries(state.now))
    out.sort(key=RoundMatch.sort_key)
    return out


def recoveries_table_for(state: PipelineState, section: str | None = None, girone: Girone | None = None) -> pd.DataFrame:
    """:func:`src.engine.tables.recoveries_table` of :func:`recoveries_for`, with the model's 1X2 / GG for the scheduled ones."""
    return recoveries_table(recoveries_for(state, section, girone), state.predictions)


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
    """Predictions of the selected matchday within a section (every pre-match prediction of the section for :data:`ALL_MATCHDAYS`).

    Only fixtures that have not kicked off at :attr:`PipelineState.now`
    (:func:`src.engine.calendar.kicked_off`) are returned: a matchday holds
    only its matches still to play, and :data:`ALL_MATCHDAYS` drops the
    fixtures already under way (or postponed, their kick-off passed) too, so
    the value panel, the picks and the predictions export never price a match
    that has started. With ``None`` and no matchday in the section (no
    predicted fixture) there is nothing to show, so the result is empty,
    consistent with the selectors.
    """
    preds = section_predictions(state, section)
    if matchday is ALL_MATCHDAYS or (isinstance(matchday, str) and matchday == ALL_MATCHDAYS):
        return [p for p in preds if not kicked_off(p.fixture, state.now)]
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
    *,
    round_number: int | None = None,
    compare_raw: bool = False,
) -> pd.DataFrame:
    """The table of the Giornata tab (:func:`src.engine.tables.matchday_table`): every match of the selected rounds.

    ``matchday`` follows the selector semantics of :func:`fixtures_for` (the
    current matchday of the section by default): the table lists **every**
    match of its rounds -- played (with the score), to play (with the model's
    numbers), postponed -- restricted to ``girone`` / ``section``.
    :data:`ALL_MATCHDAYS` lists every predicted fixture instead (the exports).
    ``round_number`` with ``girone`` shows that official round of that
    competition, whatever the matchday (the *Giornata n.* selector). Strengths
    come from ``state.params``, the form columns from the current-season matches
    (:meth:`PipelineState.team_matches_frame`, built once per state), the
    ``Book`` / ``EV max`` / ``Flag`` columns from ``state.opportunities`` and
    ``state.traps``. Columns exactly :data:`MATCHDAY_COLUMNS`, ``match_id`` last;
    with ``compare_raw`` and a recalibrated state (:func:`apply_recalibration`)
    each recalibrated probability is followed by the model's own one
    (``'PX orig. (%)'``, :func:`with_raw_columns`).
    """
    table = _matchday_table_for(state, matchday, girone, section, round_number=round_number)
    return with_raw_columns(table, state) if compare_raw else table


def _matchday_table_for(
    state: PipelineState,
    matchday: MatchdaySelector,
    girone: Girone | None,
    section: str | None,
    *,
    round_number: int | None,
) -> pd.DataFrame:
    g = _as_girone(girone)
    common = dict(
        params=state.params,
        matches=state.current_matches,
        opportunities=state.opportunities,
        traps=state.traps,
        girone=g,
        long=state.team_matches_frame(),
    )
    if round_number is not None and g is not None:
        if outside_section(g, section):
            return matchday_table([], **common)
        return matchday_table(section_predictions(state, section), rounds=[round_view_for(state, g, round_number)], **common)
    if matchday is ALL_MATCHDAYS or (isinstance(matchday, str) and matchday == ALL_MATCHDAYS):
        return matchday_table(_restrict(state, ALL_MATCHDAYS, section), **common)
    chosen = resolve_matchday(state, matchday, section)
    if chosen is None or outside_section(g, section):
        return matchday_table([], **common)
    gironi = None if section is None else configured_gironi(section)
    return matchday_table(section_predictions(state, section), matchday=chosen, gironi=gironi, **common)


def round_table_for(state: PipelineState, view: RoundView) -> pd.DataFrame:
    """The Giornata table of one :class:`~src.engine.calendar.RoundView` (every match of the round, model numbers for the ones to play)."""
    return matchday_table(
        state.predictions,
        params=state.params,
        matches=state.current_matches,
        opportunities=state.opportunities,
        traps=state.traps,
        long=state.team_matches_frame(),
        rounds=[view],
    )


#: Prefix of the ``st.dataframe`` widget key of the *Giornata* table (see :func:`matchday_table_key`).
MATCHDAY_TABLE_KEY_PREFIX: Final[str] = "matchday_table"


def matchday_table_key(
    matchday: Matchday | None,
    girone: Girone | None = None,
    section: str | None = None,
    *,
    round_number: int | None = None,
) -> str:
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
    test can still find the table by ``key.startswith(...)``. ``round_number``
    (one competition's official round chosen in the *Giornata n.* selector)
    scopes the key to that round instead of the matchday: ``'matchday_table_g13_A'``.
    """
    if round_number is not None:
        scope_matchday = f"g{int(round_number)}"
    else:
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
    """CSV bytes (see :func:`export_csv_bytes`) of :func:`matchday_table_for`, for the download button and the console export (marked in the demo)."""
    return export_csv_bytes(matchday_table_for(state, matchday, girone, section), demo=state.is_demo)


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
    return grouped_results_table(state.matches, state.engine, season=label, girone=girone, team=team, gironi=gironi)


def grouped_results_table(
    matches: pd.DataFrame,
    engine: "DixonColesEngine | GroupedFit",
    *,
    season: str | None = None,
    girone: Girone | str | None = None,
    team: str | None = None,
    gironi: Iterable[Girone] | None = None,
) -> pd.DataFrame:
    """:func:`src.engine.tables.results_table` with the right engine per fit group (each group's own ``rho``).

    One group (the shipped configuration): exactly ``results_table`` with the
    engine. Several groups: one table per group with that group's engine, then
    the shared order (date descending, competition order, home team).
    """
    label = season
    gironi = None if gironi is None else tuple(gironi)
    if isinstance(engine, GroupedFit) and len(engine.groups) > 1:
        # one table per fit group, each with its own engine (its own rho), then the shared order
        parts = []
        for group, group_engine in engine.engines.items():
            codes = set(engine.group_codes.get(group, ()))
            wanted = [g for g in (gironi if gironi is not None else tuple(Girone)) if g.value in codes]
            if wanted:
                parts.append(results_table(matches, engine=group_engine, season=label, girone=_as_girone(girone), team=team, gironi=wanted))
        if not parts:
            return results_table(matches.iloc[0:0], engine=next(iter(engine.engines.values())), season=label)
        frame = pd.concat(parts, ignore_index=True)
        order = frame["Competizione"].map({g.short: i for i, g in enumerate(Girone)}).astype("int64")
        return (
            frame.assign(_order=order)
            .sort_values(["Data", "_order", "Casa"], ascending=[False, True, True], kind="stable")
            .drop(columns="_order")
            .reset_index(drop=True)
        )
    return results_table(matches, engine=engine, season=label, girone=_as_girone(girone), team=team, gironi=gironi)


def results_csv_bytes(
    state: PipelineState,
    girone: Girone | None = None,
    team: str | None = None,
    season: str | None = None,
    section: str | None = None,
) -> bytes:
    """CSV bytes (see :func:`export_csv_bytes`) of :func:`results_table_for`, for the download button and the console export (marked in the demo)."""
    return export_csv_bytes(results_table_for(state, girone, team, season, section), demo=state.is_demo)


def no_odds_message(state: PipelineState) -> str:
    """Italian hint shown by the value panel when ``state.quotes`` is empty.

    The text depends on the data source: with harvested data FootyStats may not
    quote the fixtures yet (or the user can add odds CSVs to the odds
    directory); with the synthetic generator of the demo the odds come
    from the simulated bookmaker, which quotes only the upcoming fixtures, so
    the only way to have no quotes is to have no fixtures left.
    """
    if state.source == "processed":
        try:
            odds_dir = ODDS_DIR.relative_to(ROOT).as_posix()
        except ValueError:
            odds_dir = str(ODDS_DIR)
        return (
            "Nessuna quota bookmaker disponibile: FootyStats non quota ancora le partite in calendario (aggiorna i dati "
            f"più tardi) oppure aggiungi file CSV in {odds_dir} (colonne match_id, bookmaker, market, selection, odds) "
            "per calcolare l'EV."
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


def picks_csv_bytes(draws: pd.DataFrame, btts: pd.DataFrame, matchday: Matchday | None = None, *, demo: bool = False) -> bytes:
    """CSV bytes (see :func:`export_csv_bytes`) of :func:`picks_export_frame`, for the download button and the console export.

    ``demo`` marks the file as simulated (:func:`mark_demo_frame`): pass
    :attr:`PipelineState.is_demo`.
    """
    return export_csv_bytes(picks_export_frame(draws, btts, matchday), demo=demo)


def mark_demo_frame(frame: pd.DataFrame) -> pd.DataFrame:
    """``frame`` with a first column :data:`DEMO_EXPORT_COLUMN` = :data:`DEMO_EXPORT_TEXT` on every row.

    An empty frame gets one row holding only the warning, so even a file with
    no data says it comes from the demo. The input is not modified.
    """
    columns = [c for c in frame.columns if c != DEMO_EXPORT_COLUMN]
    if frame.empty:
        return pd.DataFrame([{DEMO_EXPORT_COLUMN: DEMO_EXPORT_TEXT}], columns=[DEMO_EXPORT_COLUMN, *columns])
    marked = frame[columns].copy()
    marked.insert(0, DEMO_EXPORT_COLUMN, DEMO_EXPORT_TEXT)
    return marked


def demo_file_name(name: str, demo: bool) -> str:
    """``'DEMO_<name>'`` for a demo export (:data:`DEMO_FILE_PREFIX`), ``name`` unchanged otherwise (never prefixed twice)."""
    if not demo or name.startswith(DEMO_FILE_PREFIX):
        return name
    return f"{DEMO_FILE_PREFIX}{name}"


def export_csv_bytes(frame: pd.DataFrame, *, demo: bool = False) -> bytes:
    """UTF-8, comma-separated, ``.`` decimal CSV without index (dates as ISO ``YYYY-MM-DD``).

    ``demo=True`` (the demo state: pass :attr:`PipelineState.is_demo`) writes
    :func:`mark_demo_frame` instead: first column ``AVVISO`` with
    ``'DATI SIMULATI - NON USARE PER SCOMMETTERE'`` on every row.
    """
    data = mark_demo_frame(frame) if demo else frame
    return data.to_csv(index=False, sep=",", decimal=".", date_format="%Y-%m-%d", lineterminator="\n").encode("utf-8")


# =============================================================================
# Recalibration: the walk-forward differential applied to the production probabilities
# =============================================================================
#: Prefix of the *Dettagli pipeline* note written by :func:`apply_recalibration` (removed by :func:`raw_state`).
RECALIBRATION_NOTE_PREFIX: Final[str] = "Probabilità ricalibrate"
#: Probability columns of the Giornata table that get an "original" twin when the recalibration is on.
RAW_COMPARE_COLUMNS: Final[tuple[tuple[str, Market, str], ...]] = (
    ("P1 (%)", Market.MATCH_ODDS, "1"),
    ("PX (%)", Market.MATCH_ODDS, "X"),
    ("P2 (%)", Market.MATCH_ODDS, "2"),
    ("O2.5 (%)", Market.OVER_UNDER_25, "Over 2.5"),
    ("GG (%)", Market.BTTS, "GG"),
)


def raw_column_name(column: str) -> str:
    """``'PX (%)'`` -> ``'PX orig. (%)'``: the column of the model's own (not recalibrated) probability."""
    return column.replace(" (%)", " orig. (%)")


def value_engine_for(state: PipelineState) -> ValueEngine:
    """The :class:`ValueEngine` the state's opportunities were evaluated with (its EV threshold and Kelly settings)."""
    return ValueEngine(min_ev=state.min_ev, kelly=KellyConfig(fraction=state.kelly_fraction, bankroll_units=state.bankroll_units))


def raw_state(state: PipelineState) -> PipelineState:
    """``state`` with the model's own probabilities (the inverse of :func:`apply_recalibration`; itself when already raw)."""
    if state.raw_predictions is None:
        return state
    return dataclasses.replace(
        state,
        predictions=list(state.raw_predictions),
        opportunities=list(state.raw_opportunities or []),
        raw_predictions=None,
        raw_opportunities=None,
        recalibration=None,
        notes=[n for n in state.notes if not n.startswith(RECALIBRATION_NOTE_PREFIX)],
    )


def apply_recalibration(state: PipelineState, model: RecalibrationModel | None) -> PipelineState:
    """``state`` whose predictions (and value opportunities) use the **validated** recalibrated probabilities.

    Only the markets the model enabled (out-of-sample log-loss gain above its
    threshold, see :func:`src.engine.calibration.fit_recalibration`) change;
    the raw predictions and opportunities stay in :attr:`PipelineState.raw_predictions`
    / :attr:`PipelineState.raw_opportunities`, so the switch is reversible
    (:func:`raw_state`) and the Giornata table can show both
    (:func:`matchday_table_for` ``compare_raw=True``). ``None`` or a model with
    no enabled market returns the raw state. The value engine is re-run with
    the state's own settings (EV threshold, Kelly), so EV, stake, picks and
    slips all follow the corrected probabilities. No refit: rates, score
    matrices and strengths are the model's.
    """
    base = raw_state(state)
    if model is None or not model.enabled:
        return base
    predictions = [model.apply_prediction(p) for p in base.predictions]
    opportunities = value_engine_for(base).evaluate(predictions, base.quotes, matches=base.current_matches, traps=base.traps)
    log.info(
        "recalibration applied to the production predictions",
        extra={"markets": [m.value for m in model.enabled_markets], "n_predictions": len(predictions), "n_opportunities": len(opportunities)},
    )
    return dataclasses.replace(
        base,
        predictions=predictions,
        opportunities=list(opportunities),
        raw_predictions=list(base.predictions),
        raw_opportunities=list(base.opportunities),
        recalibration=model,
        notes=[*base.notes, f"{RECALIBRATION_NOTE_PREFIX}: {model.describe()}"],
    )


def with_raw_columns(table: pd.DataFrame, state: PipelineState) -> pd.DataFrame:
    """A Giornata table with, next to each recalibrated probability (:data:`RAW_COMPARE_COLUMNS`), the model's own one.

    ``'PX orig. (%)'`` follows ``'PX (%)'`` and so on, filled only on the rows
    that carry a probability (the matches to play); the table itself is
    returned unchanged for a raw state.
    """
    if not state.calibrated or table.empty or "match_id" not in table.columns:
        return table
    raw = {p.fixture.match_id: p for p in state.raw_predictions or ()}
    out = table.copy()
    ids = out["match_id"].astype(str).tolist()
    for column, market, selection in RAW_COMPARE_COLUMNS:
        if column not in out.columns:
            continue
        shown = out[column].to_numpy(dtype=float)
        values = [
            round(float(raw[m].probabilities[market][selection]) * 100.0, 1) if m in raw and np.isfinite(v) else float("nan")
            for m, v in zip(ids, shown)
        ]
        out.insert(int(out.columns.get_loc(column)) + 1, raw_column_name(column), pd.Series(values, index=out.index, dtype="float64"))
    return out


# =============================================================================
# Walk-forward reliability ("Affidabilità modello")
# =============================================================================
#: Seasons the walk-forward evaluates by default: the registry season and the one before it (more matches, a
#: steadier verdict; the per-season breakdown stays available).
DEFAULT_BACKTEST_SEASONS: Final[int] = 2
#: Number of uniform probability bins of the reliability chart.
RELIABILITY_BINS: Final[int] = 10
#: Selections offered by the reliability selector, ``label -> (market, selection)``.
RELIABILITY_SELECTIONS: Final[dict[str, tuple[Market, str]]] = {
    "1 (vittoria casa)": (Market.MATCH_ODDS, "1"),
    "X (pareggio)": (Market.MATCH_ODDS, "X"),
    "2 (vittoria ospiti)": (Market.MATCH_ODDS, "2"),
    "GG (entrambe segnano)": (Market.BTTS, "GG"),
    "Over 1.5": (Market.OVER_UNDER_15, "Over 1.5"),
    "Over 2.5": (Market.OVER_UNDER_25, "Over 2.5"),
    "Over 3.5": (Market.OVER_UNDER_35, "Over 3.5"),
}
#: Italian name of every market of the backtest tables.
BACKTEST_MARKET_LABELS: Final[dict[str, str]] = {m.value: MARKET_LABELS[m] for m in DEFAULT_MARKETS}
#: Italian titles of the slips (by kind) in the track-record tables.
SLIP_TITLES: Final[dict[str, str]] = {"X": "Schedina X · pareggi", "GG": "Schedina GG · goal", "MISTA": "Schedina mista", "Mista": "Schedina mista"}
#: Model scores restricted to the matches that have pre-match odds: the only model figures comparable with the
#: market's (``Log loss modello`` / ``Brier modello`` cover every evaluated match, the market only the quoted ones).
QUOTED_LOG_LOSS_COLUMN: Final[str] = "Log loss modello (partite con quote)"
QUOTED_BRIER_COLUMN: Final[str] = "Brier modello (partite con quote)"
#: Columns of :func:`metrics_table_for`: scores on every evaluated match, then the comparison with the bookmaker on
#: the quoted matches only (model and market on the very same matches).
RELIABILITY_METRICS_COLUMNS: Final[tuple[str, ...]] = (
    "Mercato",
    "Partite",
    "Log loss modello",
    "Log loss base",
    "Skill vs base (%)",
    "Brier modello",
    "Accuratezza (%)",
    "P media (%)",
    "Frequenza (%)",
    "Partite con quote",
    QUOTED_LOG_LOSS_COLUMN,
    "Log loss mercato",
    "Skill vs mercato (%)",
    QUOTED_BRIER_COLUMN,
    "Brier mercato",
    "Margine medio (%)",
)
#: Columns of :func:`competition_metrics_table`.
COMPETITION_METRICS_COLUMNS: Final[tuple[str, ...]] = (
    "Sezione",
    "Competizione",
    "Partite",
    "Log loss modello",
    "Log loss base",
    "Skill vs base (%)",
    "Brier modello",
    "Accuratezza (%)",
    "Partite con quote",
    QUOTED_LOG_LOSS_COLUMN,
    "Log loss mercato",
    "Skill vs mercato (%)",
)
#: Columns of :func:`picks_record_table`.
PICKS_RECORD_COLUMNS: Final[tuple[str, ...]] = (
    "Lista",
    "Giornate",
    "Selezioni",
    "Vinte",
    "Attese",
    "Hit rate (%)",
    "IC 95% (%)",
    "P media modello (%)",
    "Con quota",
    "ROI (%)",
    "IC 95% ROI (%)",
)
#: Columns of :func:`slips_record_table`.
SLIPS_RECORD_COLUMNS: Final[tuple[str, ...]] = (
    "Schedina",
    "Giornate",
    "Vinte",
    "Vincite attese",
    "Win rate (%)",
    "IC 95% (%)",
    "P media (%)",
    "Partite (gambe)",
    "Gambe vinte (%)",
    "IC 95% gambe (%)",
    "P media gamba (%)",
    "Con quota",
    "ROI (%)",
    "IC 95% ROI (%)",
)
#: Columns of :func:`recalibration_table`.
RECALIBRATION_COLUMNS: Final[tuple[str, ...]] = (
    "Mercato",
    "Applicata",
    "Partite stima",
    "Partite verifica",
    "Log loss prima",
    "Log loss dopo",
    "Miglioramento (%)",
    "Brier prima",
    "Brier dopo",
    "Parametri",
    "Motivo",
)
#: Columns of :func:`reliability_export_frame` (the ``<stem>_affidabilita.csv`` file and the dashboard download).
RELIABILITY_EXPORT_COLUMNS: Final[tuple[str, ...]] = (
    "Sezione",
    "Mercato",
    "Esito di riferimento",
    "Partite",
    "Log loss modello",
    "Log loss base",
    "Skill vs base",
    "Brier modello",
    "Accuratezza",
    "P media",
    "Frequenza",
    "Partite con quote",
    QUOTED_LOG_LOSS_COLUMN,
    "Log loss mercato",
    "Skill vs mercato",
    QUOTED_BRIER_COLUMN,
    "Brier mercato",
)
#: Honest reading of a track record, shown under every hit rate / ROI table.
EVIDENCE_NOTE: Final[str] = (
    "Poche vincite non sono una prova: una schedina da 5 X o due GG azzeccati in una domenica capitano anche a un modello "
    "senza alcun vantaggio. Contano il numero di giornate e di partite valutate e l'intervallo di confidenza al 95% "
    "(Wilson per le percentuali, approssimazione normale per il ROI): finché l'intervallo del ROI contiene lo zero il "
    "guadagno non è distinguibile dalla fortuna, e un hit rate vicino alla probabilità media del modello dice solo che "
    "il modello è tarato, non che batte il bookmaker."
)


@dataclass(frozen=True)
class ReliabilityReport:
    """The walk-forward backtest of a perimeter with the recalibration it supports (the *Affidabilità modello* data).

    ``result`` holds the out-of-sample probabilities of every evaluated match
    (:func:`src.engine.backtest.walk_forward`, never the production fit);
    ``recalibration`` the per-market correction learnt on it with its
    out-of-sample verdict (:func:`src.engine.calibration.fit_recalibration`);
    ``odds_history`` the pre-match bookmaker odds of the completed matches
    (``data/odds/history/odds_history.csv``; empty when FootyStats sent none),
    for the comparison with the market and the ROI; ``key`` identifies the data
    it was computed on (:func:`reliability_key`) and ``from_cache`` says
    whether it was read back from disk. ``rolling`` is the matchday-by-matchday
    out-of-sample recalibration of the frame (:func:`rolling_frame`), attached
    once when the recalibration enabled a market (``None`` otherwise). Rows of a
    competition that is no longer enabled are dropped on construction
    (:func:`known_competition_rows`).
    """

    result: BacktestResult
    recalibration: RecalibrationModel
    odds_history: pd.DataFrame = field(default_factory=pd.DataFrame)
    key: str = ""
    from_cache: bool = False
    rolling: RollingRecalibration | None = None

    def __post_init__(self) -> None:
        # a backtest of a competition switched off since (enabled = false) must never reach the views:
        # its code is no Girone member any more and every track-record helper would raise on it
        clean = known_result(self.result)
        if clean is not self.result:
            object.__setattr__(self, "result", clean)
        if self.rolling is not None:
            frame = known_competition_rows(self.rolling.frame)
            if frame is not self.rolling.frame:
                object.__setattr__(self, "rolling", RollingRecalibration(frame, self.rolling.steps))

    @property
    def frame(self) -> pd.DataFrame:
        return self.result.frame

    @property
    def empty(self) -> bool:
        return self.result.empty

    @property
    def has_odds(self) -> bool:
        return not self.odds_history.empty

    def section_frame(self, section: str | None = None, *, frame: pd.DataFrame | None = None) -> pd.DataFrame:
        """The backtest rows of one section (every row for ``None`` / :data:`ALL_SECTIONS_LABEL`; an alias covers its members)."""
        data = self.result.frame if frame is None else known_competition_rows(frame)
        if section is None or section == ALL_SECTIONS_LABEL or data.empty:
            return data
        members = section_members(section)
        return data.loc[data["section"].astype(str).isin(members).to_numpy(dtype=bool)].reset_index(drop=True)


def backtest_seasons(matches: pd.DataFrame, season_label: str, n: int = DEFAULT_BACKTEST_SEASONS) -> tuple[str, ...]:
    """The ``n`` seasons up to ``season_label`` that the walk-forward evaluates, chronological (``('2025/26', '2026/27')``)."""
    current = str(season_label).replace("_", "/")
    present = set(pd.to_datetime(matches["date"]).map(season_of)) if len(matches) else set()
    chosen = sorted(s for s in present | {current} if s <= current)
    return tuple(chosen[-max(int(n), 1):])


def backtest_config_rule(*, xi: float, prior: object, registry: TeamRegistry | None):  # noqa: ANN201 - a walk_forward ConfigRule or None
    """The walk-forward config rule that replays the production settings (``None`` = the production rule itself).

    With the default settings (``xi`` 0.0018, prior ``'auto'``) the backtest
    uses :func:`src.engine.backtest.production_config`; other settings (a
    tuned ``xi``, a forced or stronger prior) are replayed block by block with
    :func:`model_config`, so the recalibration is learnt on the very model the
    dashboard publishes.
    """
    setting = prior_setting(prior)
    if setting == "auto" and math.isclose(float(xi), 0.0018):
        return None

    def rule(train: pd.DataFrame, season: str) -> DixonColesConfig:
        reg = registry if registry is not None and registry.season_label.replace("_", "/") == season else None
        return model_config(train, xi=float(xi), prior=setting, registry=reg, season=season, method="L-BFGS-B")[0]

    return rule


#: Version of the backtest logic baked into every reliability key: bump it whenever the walk-forward, its
#: metrics, the recalibration or the rolling correction change, so a backtest saved by an older version of the
#: code (``data/backtest.*``) is recomputed instead of read back.
BACKTEST_LOGIC_VERSION: Final[str] = "v2"


def config_fingerprint(competitions: Iterable[object] | None = None) -> str:
    """Short hash of the competition configuration a backtest depends on: code, section, family and fit group of every enabled one.

    Switching a league on or off (``enabled`` in ``config/competitions.toml``),
    moving it to another section or changing its country changes the
    fingerprint, so the saved backtest is not read back for a configuration it
    was not computed on (``competitions`` defaults to the ``Girone`` enum's).
    """
    comps = Girone.competitions() if competitions is None else tuple(competitions)
    payload = "|".join(f"{c.code}:{c.section}:{c.family}:{c.fit_group}" for c in comps)  # type: ignore[attr-defined]
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()[:12]


def reliability_key(*parts: object) -> str:
    """Cache key of a reliability report: logic version, competition configuration, then the data version, seasons and model settings.

    The first two parts (:data:`BACKTEST_LOGIC_VERSION`, :func:`config_fingerprint`)
    make a backtest persisted before a code update or a change of
    ``config/competitions.toml`` miss the cache.
    """
    return "|".join(str(p) for p in (BACKTEST_LOGIC_VERSION, f"cfg={config_fingerprint()}", *parts))


def known_competition_rows(frame: pd.DataFrame) -> pd.DataFrame:
    """The rows of a backtest frame whose competition is still enabled (the others dropped and logged; the same frame when none is).

    Like the CSV loaders (:func:`load_enabled_csv`): a league switched off after
    its backtest was computed leaves rows with a code that is no ``Girone``
    member any more, which the track-record helpers cannot rebuild.
    """
    if frame is None or frame.empty or "girone" not in frame.columns:
        return frame
    codes = {g.value for g in Girone}
    mask = frame["girone"].astype(str).isin(codes).to_numpy(dtype=bool)
    if mask.all():
        return frame
    dropped = sorted(set(frame.loc[~mask, "girone"].astype(str)))
    log.warning("backtest rows of competitions not enabled skipped", extra={"codes": dropped, "rows": int((~mask).sum())})
    return frame.loc[mask].reset_index(drop=True)


def known_result(result: BacktestResult) -> BacktestResult:
    """:func:`known_competition_rows` applied to a :class:`BacktestResult` (the same object when nothing is dropped)."""
    frame = known_competition_rows(result.frame)
    return result if frame is result.frame else dataclasses.replace(result, frame=frame)


def build_reliability(
    matches: pd.DataFrame,
    *,
    registry: TeamRegistry | None,
    season_label: str,
    n_seasons: int = DEFAULT_BACKTEST_SEASONS,
    xi: float = 0.0018,
    prior: object = "auto",
    odds_history: pd.DataFrame | None = None,
    block: str = DEFAULT_BLOCK,
    progress=None,  # noqa: ANN001 - ProgressCallback
    key: str = "",
) -> ReliabilityReport:
    """Run the walk-forward backtest of ``matches`` and fit the validated recalibration on it.

    Every evaluated block is predicted by a model fitted only on the matches
    played before it (history included, production time decay and prior rule,
    L-BFGS-B warm-started from the previous block), one model per fit group;
    ``n_seasons`` seasons up to ``season_label`` are evaluated
    (:func:`backtest_seasons`). The recalibration is fitted on the earlier
    matchdays and judged on the later ones (it is applied only when it improves
    the out-of-sample log loss by the minimum margin).
    """
    seasons = backtest_seasons(matches, season_label, n_seasons)
    result = walk_forward(
        matches,
        registry=registry,
        seasons=seasons,
        config=backtest_config_rule(xi=xi, prior=prior, registry=registry),
        block=block,  # type: ignore[arg-type]
        progress=progress,
    )
    model = fit_recalibration(result.frame) if not result.empty else RecalibrationModel.identity()
    log.info(
        "reliability report built",
        extra={"seasons": list(seasons), "n_matches": result.n_matches, "recalibration": [m.value for m in model.enabled_markets], "key": key},
    )
    return ReliabilityReport(result=result, recalibration=model, odds_history=pd.DataFrame() if odds_history is None else odds_history, key=key)


def reliability_for_state(
    state: PipelineState,
    *,
    n_seasons: int = DEFAULT_BACKTEST_SEASONS,
    odds_history: pd.DataFrame | None = None,
    progress=None,  # noqa: ANN001
    key: str = "",
) -> ReliabilityReport:
    """:func:`build_reliability` on the state's matches, registry and model settings."""
    return build_reliability(
        state.matches,
        registry=state.registry,
        season_label=state.season_label,
        n_seasons=n_seasons,
        xi=state.xi,
        prior=state.prior,
        odds_history=odds_history,
        progress=progress,
        key=key,
    )


def rolling_frame(report: ReliabilityReport) -> RollingRecalibration:
    """The backtest frame recalibrated matchday by matchday with what was known before each (an honest "con calibrazione" track record).

    The report's own :attr:`ReliabilityReport.rolling` when it carries one
    (computed once with the backtest and persisted next to it by the
    dashboard), else computed now (one validated recalibration per matchday:
    seconds to tens of seconds).
    """
    if report.rolling is not None:
        return report.rolling
    return rolling_recalibration(report.frame)


def with_rolling(report: ReliabilityReport, rolling: RollingRecalibration | None = None) -> ReliabilityReport:
    """The report carrying its rolling recalibration (``rolling`` when given, else :func:`rolling_frame`); unchanged when it already has one."""
    if report.rolling is not None and rolling is None:
        return report
    return dataclasses.replace(report, rolling=rolling if rolling is not None else rolling_frame(report))


# =============================================================================
# Walk-forward tuning of xi and prior, selected on earlier matchdays and confirmed on later ones
# =============================================================================
#: Columns of :func:`tuning_table`.
TUNING_COLUMNS: Final[tuple[str, ...]] = (
    "xi",
    "Prior",
    "Partite scelta",
    "Log loss scelta",
    "Partite conferma",
    "Log loss conferma",
    "Produzione",
    "Scelta",
    "Secondi",
)
#: The production point of the tuning grid.
PRODUCTION_TUNING_POINT: Final[tuple[float, str]] = (0.0018, "auto")


@dataclass(frozen=True)
class ValidatedTuning:
    """The xi / prior tuning, selected on the earlier matchdays and confirmed on the later ones (:func:`validated_tuning`).

    Every grid point is backtested walk-forward on the same matches; the point
    with the lowest mean log loss on the **selection** part (matchdays before
    ``split_date``) is chosen, then the chosen point and the production one are
    compared on the **confirmation** part (``split_date`` onwards), which the
    choice never saw. ``accepted`` only when that held-out gain reaches
    ``min_gain`` (the recalibration's own threshold) on at least ``min_eval``
    matches: picking the best of 9 points always "wins" on the matches it was
    picked on, so the selection gain alone proves nothing.
    """

    table: pd.DataFrame
    best: dict[str, object]
    seasons: tuple[str, ...]
    split_date: date | None
    n_select: int
    n_confirm: int
    production_select: float
    best_select: float
    production_confirm: float
    best_confirm: float
    min_gain: float = DEFAULT_MIN_GAIN
    min_eval: int = DEFAULT_MIN_EVAL
    seconds: float = 0.0

    @property
    def is_production(self) -> bool:
        return math.isclose(float(self.best.get("xi", PRODUCTION_TUNING_POINT[0])), PRODUCTION_TUNING_POINT[0]) and str(self.best.get("prior")) == PRODUCTION_TUNING_POINT[1]

    @property
    def selection_gain(self) -> float:
        """Relative gain of the chosen point on the selection part (optimistic by construction: it was chosen there)."""
        if not (np.isfinite(self.production_select) and self.production_select > 0 and np.isfinite(self.best_select)):
            return float("nan")
        return (self.production_select - self.best_select) / self.production_select

    @property
    def improvement(self) -> float:
        """Relative gain of the chosen point over production on the held-out confirmation part (``NaN`` without one)."""
        if not (np.isfinite(self.production_confirm) and self.production_confirm > 0 and np.isfinite(self.best_confirm)):
            return float("nan")
        return (self.production_confirm - self.best_confirm) / self.production_confirm

    @property
    def accepted(self) -> bool:
        """Whether the tuned settings may be applied: another point than production, confirmed on enough held-out matches by ``min_gain``."""
        return (
            not self.table.empty
            and not self.is_production
            and self.n_confirm >= int(self.min_eval)
            and bool(np.isfinite(self.improvement))
            and self.improvement >= float(self.min_gain)
        )

    @property
    def settings(self) -> tuple[float, str]:
        """``(xi, prior)`` of the chosen point."""
        return float(self.best.get("xi", PRODUCTION_TUNING_POINT[0])), str(self.best.get("prior", PRODUCTION_TUNING_POINT[1]))

    def describe(self) -> str:
        """Italian summary: the chosen point, where it was chosen, its held-out gain and whether it may be applied."""
        if self.table.empty:
            return "Ottimizzazione di xi e prior: nessun risultato (backtest vuoto)."
        xi, prior = self.settings
        when = f"fino al {self.split_date:%d/%m/%Y}" if self.split_date is not None else "su tutte le giornate"
        text = (
            f"Scelta sulle {self.n_select} partite delle giornate {when}: migliore xi={xi:g}, prior {prior} "
            f"({_signed_pct(self.selection_gain)} rispetto alla produzione su quelle stesse partite, ottimistico perché scelto lì)."
        )
        if self.is_production:
            return text + " La configurazione di produzione resta la migliore: niente da applicare."
        if self.n_confirm < int(self.min_eval):
            return text + f" Conferma impossibile: solo {self.n_confirm} partite dopo il taglio (ne servono {int(self.min_eval)}); resta la configurazione di produzione."
        verdict = (
            "supera la soglia: si può applicare."
            if self.accepted
            else f"sotto la soglia del {self.min_gain * 100:.1f}%: è rumore, resta la configurazione di produzione."
        )
        return text + f" Conferma sulle {self.n_confirm} partite successive, mai viste nella scelta: {_signed_pct(self.improvement)}, {verdict}"


def _signed_pct(value: float) -> str:
    return "n.d." if value is None or not np.isfinite(value) else f"{value * 100:+.2f}%".replace(".", ",")


def _tuning_objective(frame: pd.DataFrame, markets: Sequence[Market]) -> float:
    """Mean over ``markets`` of the mean log loss of a backtest frame (``NaN`` when empty)."""
    if frame.empty:
        return float("nan")
    scores = metrics(frame, by=(), markets=markets)
    values = scores["log_loss"].to_numpy(dtype=float) if len(scores) else np.empty(0)
    return float(values.mean()) if len(values) == len(markets) and np.isfinite(values).all() else float("nan")


def validated_tuning(
    matches: pd.DataFrame,
    registry: TeamRegistry | None,
    *,
    seasons: Sequence[str],
    grid: Mapping[str, Sequence[object]] | None = None,
    test_fraction: float = DEFAULT_TEST_FRACTION,
    min_gain: float = DEFAULT_MIN_GAIN,
    min_eval: int = DEFAULT_MIN_EVAL,
    markets: Sequence[Market] = DEFAULT_TUNING_MARKETS,
    block: str = DEFAULT_BLOCK,
    progress=None,  # noqa: ANN001 - ProgressCallback
) -> ValidatedTuning:
    """Tune the time decay ``xi`` and the prior on a time split of the walk-forward (see :class:`ValidatedTuning`).

    Every point of ``grid`` (default :data:`src.engine.calibration.DEFAULT_TUNING_GRID`,
    9 points; the production one is always added) runs one walk-forward over
    ``seasons`` with the dashboard's own config rule (:func:`backtest_config_rule`).
    The frame is split on a date (the latest ``test_fraction`` of the matches of
    the production run, whole days, as the recalibration does): the point is
    chosen on the earlier part and confirmed on the later one. Same cost as
    ranking the grid on every match (one walk-forward per point).
    """
    chosen = dict(DEFAULT_TUNING_GRID if grid is None else grid)
    xis = [float(x) for x in chosen.get("xi", (PRODUCTION_TUNING_POINT[0],))]
    priors = [prior_setting(p) for p in chosen.get("prior", (PRODUCTION_TUNING_POINT[1],))]
    if any(x < 0 for x in xis):
        raise ValueError(f"xi values must be >= 0, got {xis}")
    points = list(dict.fromkeys([PRODUCTION_TUNING_POINT, *((xi, prior) for xi in dict.fromkeys(xis) for prior in dict.fromkeys(priors))]))
    report = progress if progress is not None else (lambda _fraction, _message: None)
    started = time.perf_counter()
    frames: list[tuple[float, str, pd.DataFrame, float]] = []
    for position, (xi, prior) in enumerate(points, start=1):
        t0 = time.perf_counter()
        result = walk_forward(matches, registry=registry, seasons=list(seasons), config=backtest_config_rule(xi=xi, prior=prior, registry=registry), block=block)  # type: ignore[arg-type]
        frames.append((xi, prior, known_competition_rows(result.frame), time.perf_counter() - t0))
        report(position / len(points), f"Ottimizzazione: {position}/{len(points)} (xi={xi:g}, prior {prior})")
    production_frame = frames[0][2]
    _early, _late, cut = time_split(production_frame, test_fraction=test_fraction) if len(production_frame) else (production_frame, production_frame, None)
    rows: list[dict[str, object]] = []
    for xi, prior, frame, seconds in frames:
        early, late = (frame, frame.iloc[0:0]) if cut is None else time_split(frame, split=cut)[:2]
        rows.append(
            {
                "xi": xi,
                "prior": prior,
                "n_select": int(len(early)),
                "objective_select": _tuning_objective(early, markets),
                "n_confirm": int(len(late)),
                "objective_confirm": _tuning_objective(late, markets),
                "seconds": float(seconds),
            }
        )
    table = pd.DataFrame(rows, columns=["xi", "prior", "n_select", "objective_select", "n_confirm", "objective_confirm", "seconds"])
    table["production"] = [math.isclose(float(x), PRODUCTION_TUNING_POINT[0]) and p == PRODUCTION_TUNING_POINT[1] for x, p in zip(table["xi"], table["prior"])]
    # rank on the selection part only; on a tie production comes first (no change without a reason)
    table = table.sort_values(["objective_select", "production"], ascending=[True, False], kind="stable", na_position="last").reset_index(drop=True)
    best_row = table.iloc[0] if len(table) and np.isfinite(table["objective_select"].iloc[0]) else table.loc[table["production"]].iloc[0]
    production_row = table.loc[table["production"]].iloc[0]
    out = ValidatedTuning(
        table=table,
        best={"xi": float(best_row["xi"]), "prior": str(best_row["prior"])},
        seasons=tuple(str(s) for s in seasons),
        split_date=cut,
        n_select=int(production_row["n_select"]),
        n_confirm=int(production_row["n_confirm"]),
        production_select=float(production_row["objective_select"]),
        best_select=float(best_row["objective_select"]),
        production_confirm=float(production_row["objective_confirm"]),
        best_confirm=float(best_row["objective_confirm"]),
        min_gain=float(min_gain),
        min_eval=int(min_eval),
        seconds=time.perf_counter() - started,
    )
    log.info(
        "hyper-parameters tuned on a time split",
        extra={"best": out.best, "split": str(cut), "selection_gain": out.selection_gain, "confirmed_gain": out.improvement, "accepted": out.accepted},
    )
    return out


def tuning_table(tuned: ValidatedTuning) -> pd.DataFrame:
    """The grid of a :class:`ValidatedTuning` for the dashboard (:data:`TUNING_COLUMNS`): selection and confirmation log loss per point."""
    rows = [
        {
            "xi": float(r["xi"]),
            "Prior": str(r["prior"]),
            "Partite scelta": int(r["n_select"]),
            "Log loss scelta": _round(r["objective_select"]),
            "Partite conferma": int(r["n_confirm"]),
            "Log loss conferma": _round(r["objective_confirm"]),
            "Produzione": "sì" if bool(r["production"]) else "",
            "Scelta": "sì" if math.isclose(float(r["xi"]), tuned.settings[0]) and str(r["prior"]) == tuned.settings[1] else "",
            "Secondi": round(float(r["seconds"]), 1),
        }
        for r in tuned.table.to_dict(orient="records")
    ]
    return _typed(pd.DataFrame(rows, columns=list(TUNING_COLUMNS)), ("Prior", "Produzione", "Scelta"))


def tuning_disclosure(tuned: ValidatedTuning) -> str:
    """What the *Affidabilità* tab says while tuned settings are applied: where they were chosen, so those matches are not fully out of sample."""
    xi, prior = tuned.settings
    seasons = ", ".join(tuned.seasons)
    until = f"fino al {tuned.split_date:%d/%m/%Y}" if tuned.split_date is not None else ""
    return (
        f"Stima con xi={xi:g} e prior {prior} ottimizzati: scelti sulle {tuned.n_select} partite della stagione {seasons} {until}. "
        "Su quelle partite le metriche qui sotto non sono del tutto fuori campione (la scelta ha visto i loro risultati), e da "
        f"questa configurazione parte anche il giudizio della ricalibrazione. Verifica indipendente: le {tuned.n_confirm} partite "
        f"successive, miglioramento {_signed_pct(tuned.improvement)}."
    ).replace("  ", " ")


def _pct(value: float, digits: int = 1) -> float:
    return round(float(value) * 100.0, digits) if value is not None and np.isfinite(value) else float("nan")


def _round(value: float, digits: int = 4) -> float:
    return round(float(value), digits) if value is not None and np.isfinite(value) else float("nan")


def _interval(low: float, high: float) -> str:
    if low is None or high is None or not (np.isfinite(low) and np.isfinite(high)):
        return "-"
    return f"{low * 100:.1f} – {high * 100:.1f}"


def _market_rows(model: pd.DataFrame, market_cmp: pd.DataFrame) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    cmp = {str(r["market"]): r for r in market_cmp.to_dict(orient="records")} if len(market_cmp) else {}
    for r in model.to_dict(orient="records"):
        c = cmp.get(str(r["market"]))
        rows.append(
            {
                "Mercato": BACKTEST_MARKET_LABELS.get(str(r["market"]), str(r["market"])),
                "Partite": int(r["n"]),
                "Log loss modello": _round(r["log_loss"]),
                "Log loss base": _round(r["log_loss_naive"]),
                "Skill vs base (%)": _pct(r["skill_naive"], 2),
                "Brier modello": _round(r["brier"]),
                "Accuratezza (%)": _pct(r["accuracy"]),
                "P media (%)": _pct(r["mean_pred"]),
                "Frequenza (%)": _pct(r["observed"]),
                "Partite con quote": int(c["n"]) if c is not None else 0,
                QUOTED_LOG_LOSS_COLUMN: _round(c["log_loss_model"]) if c is not None else float("nan"),
                "Log loss mercato": _round(c["log_loss_market"]) if c is not None else float("nan"),
                "Skill vs mercato (%)": _pct(c["skill"], 2) if c is not None else float("nan"),
                QUOTED_BRIER_COLUMN: _round(c["brier_model"]) if c is not None else float("nan"),
                "Brier mercato": _round(c["brier_market"]) if c is not None else float("nan"),
                "Margine medio (%)": _pct(c["mean_overround"]) if c is not None else float("nan"),
            }
        )
    return rows


def _typed(frame: pd.DataFrame, text_columns: Iterable[str]) -> pd.DataFrame:
    text = set(text_columns)
    for column in frame.columns:
        if column in text:
            frame[column] = frame[column].astype("string")
        elif frame[column].dtype == object:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")
    return frame


def metrics_table_for(report: ReliabilityReport, section: str | None = None, *, frame: pd.DataFrame | None = None) -> pd.DataFrame:
    """Per-market scores of one section (:data:`RELIABILITY_METRICS_COLUMNS`): model vs naive base rates vs bookmaker.

    ``Log loss`` / ``Brier``: lower is better; ``Skill vs base`` = ``1 -
    logloss_modello / logloss_base`` (positive = the model beats the league
    base rates), ``Skill vs mercato`` on the matches with pre-match odds
    (positive = the model beat the de-vigged bookmaker probabilities, rare).
    ``Log loss modello`` / ``Brier modello`` cover every evaluated match; the
    market is scored only on the ``Partite con quote``, so the model's own
    scores on those very matches (:data:`QUOTED_LOG_LOSS_COLUMN`,
    :data:`QUOTED_BRIER_COLUMN`) sit next to the market's: only those two
    pairs are comparable (odds coverage differs by league).
    ``P media`` / ``Frequenza`` compare the mean predicted probability of the
    reference selection (1, GG, Over) with how often it happened. ``frame``
    overrides the backtest rows (e.g. the recalibrated ones).
    """
    data = report.section_frame(section, frame=frame)
    model = metrics(data, by=()) if len(data) else pd.DataFrame()
    cmp = market_comparison(data, report.odds_history, by=()) if len(data) and report.has_odds else pd.DataFrame()
    rows = _market_rows(model, cmp) if len(model) else []
    return _typed(pd.DataFrame(rows, columns=list(RELIABILITY_METRICS_COLUMNS)), ("Mercato",))


def headline_for(report: ReliabilityReport, section: str | None = None, market: Market = Market.MATCH_ODDS, *, frame: pd.DataFrame | None = None) -> dict[str, float]:
    """The headline numbers of one section and market.

    ``n``, ``log_loss``, ``log_loss_base``, ``brier``, ``accuracy`` cover
    every evaluated match; ``n_market`` is the number of matches with
    pre-match odds and ``log_loss_market`` / ``log_loss_model_market`` (and
    ``brier_market`` / ``brier_model_market``) are the bookmaker's and the
    model's scores on **those same matches** (``NaN`` without odds): compare
    the market only with the latter, never with ``log_loss``.
    """
    table = metrics_table_for(report, section, frame=frame)
    label = MARKET_LABELS[market]
    row = table.loc[table["Mercato"] == label]
    if row.empty:
        nan = float("nan")
        return {
            "n": 0,
            "log_loss": nan,
            "log_loss_base": nan,
            "log_loss_market": nan,
            "log_loss_model_market": nan,
            "n_market": 0,
            "brier": nan,
            "brier_market": nan,
            "brier_model_market": nan,
            "accuracy": nan,
        }
    r = row.iloc[0]
    return {
        "n": int(r["Partite"]),
        "log_loss": float(r["Log loss modello"]),
        "log_loss_base": float(r["Log loss base"]),
        "log_loss_market": float(r["Log loss mercato"]),
        "log_loss_model_market": float(r[QUOTED_LOG_LOSS_COLUMN]),
        "n_market": int(r["Partite con quote"]),
        "brier": float(r["Brier modello"]),
        "brier_market": float(r["Brier mercato"]),
        "brier_model_market": float(r[QUOTED_BRIER_COLUMN]),
        "accuracy": float(r["Accuratezza (%)"]),
    }


def market_card(head: Mapping[str, float]) -> dict[str, object]:
    """``st.metric`` arguments of the *modello vs mercato* card: both log losses on the same quoted matches.

    The value reads ``'<modello> vs <mercato>'`` and the delta is ``modello −
    mercato`` (negative = the model did better, shown green); ``n.d.``
    without an odds history. Never the all-match model log loss against the
    quoted-match market one (different matches).
    """
    model, book, n_market = head.get("log_loss_model_market"), head.get("log_loss_market"), int(head.get("n_market", 0) or 0)
    label = f"Modello vs mercato · {n_market} partite con quote"
    available = model is not None and book is not None and pd.notna(model) and pd.notna(book)
    value = f"{float(model):.4f} vs {float(book):.4f}" if available else "n.d."
    delta = f"{float(model) - float(book):+.4f}" if available else None
    text = (
        f"Log loss del modello e del bookmaker senza margine calcolate sulle stesse {n_market} partite con quote pre-partita "
        f"(su {int(head.get('n', 0) or 0)} valutate): il delta è modello − mercato, negativo = il modello ha fatto meglio. "
        "n.d. senza storico quote."
    )
    return {"label": label, "value": value, "delta": delta, "delta_color": "inverse", "help": text}


def competition_metrics_table(report: ReliabilityReport, section: str | None = None, market: Market = Market.MATCH_ODDS) -> pd.DataFrame:
    """Scores of one market per competition of the section (:data:`COMPETITION_METRICS_COLUMNS`), configuration order."""
    data = report.section_frame(section)
    columns = list(COMPETITION_METRICS_COLUMNS)
    if data.empty:
        return _typed(pd.DataFrame(columns=columns), ("Sezione", "Competizione"))
    key = market.value
    model = metrics(data, by=("section", "girone"), markets=(market,))
    cmp = market_comparison(data, report.odds_history, by=("girone",), markets=(market,)) if report.has_odds else pd.DataFrame()
    by_girone = {str(r["girone"]): r for r in cmp.to_dict(orient="records")} if len(cmp) else {}
    rows = []
    for r in model.loc[model["market"] == key].to_dict(orient="records"):
        c = by_girone.get(str(r["girone"]))
        rows.append(
            {
                "Sezione": str(r["section"]),
                "Competizione": competition_label(str(r["girone"])),
                "Partite": int(r["n"]),
                "Log loss modello": _round(r["log_loss"]),
                "Log loss base": _round(r["log_loss_naive"]),
                "Skill vs base (%)": _pct(r["skill_naive"], 2),
                "Brier modello": _round(r["brier"]),
                "Accuratezza (%)": _pct(r["accuracy"]),
                "Partite con quote": int(c["n"]) if c is not None else 0,
                QUOTED_LOG_LOSS_COLUMN: _round(c["log_loss_model"]) if c is not None else float("nan"),
                "Log loss mercato": _round(c["log_loss_market"]) if c is not None else float("nan"),
                "Skill vs mercato (%)": _pct(c["skill"], 2) if c is not None else float("nan"),
            }
        )
    return _typed(pd.DataFrame(rows, columns=columns), ("Sezione", "Competizione"))


def reliability_table_for(
    report: ReliabilityReport,
    section: str | None,
    market: Market,
    selection: str,
    *,
    bins: int = RELIABILITY_BINS,
    frame: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Reliability bins of one selection in one section (:func:`src.engine.backtest.reliability`, Wilson intervals)."""
    data = report.section_frame(section, frame=frame)
    return reliability(data, market, selection, bins=bins)


def reliability_figure(table: pd.DataFrame, title: str = "", *, calibrated: pd.DataFrame | None = None) -> go.Figure:
    """Reliability chart: mean predicted probability (x) against observed frequency (y) per bin, with 95% Wilson bars.

    The dashed diagonal is perfect calibration: points above it are selections
    that happened more often than the model said (under-confident), below it
    less often (over-confident). Marker hover carries the number of matches of
    the bin; ``calibrated`` adds the out-of-sample recalibrated series.
    """
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=[0.0, 1.0],
            y=[0.0, 1.0],
            mode="lines",
            name="Taratura perfetta",
            line={"color": "#9aa0a6", "dash": "dash", "width": 1},
            hoverinfo="skip",
        )
    )
    for frame, name, color in ((table, "Modello (fuori campione)", _SERIES_ATTACK), (calibrated, "Ricalibrato (fuori campione)", _SERIES_DEFENCE)):
        if frame is None or frame.empty:
            continue
        observed = frame["observed"].to_numpy(dtype=float)
        fig.add_trace(
            go.Scatter(
                x=frame["mean_pred"].to_numpy(dtype=float),
                y=observed,
                mode="markers+lines",
                name=name,
                marker={"color": color, "size": 9},
                line={"color": color, "width": 1},
                error_y={
                    "type": "data",
                    "symmetric": False,
                    "array": (frame["wilson_high"].to_numpy(dtype=float) - observed),
                    "arrayminus": (observed - frame["wilson_low"].to_numpy(dtype=float)),
                    "color": color,
                    "thickness": 1,
                },
                customdata=np.stack([frame["n"].to_numpy(dtype=float), frame["hits"].to_numpy(dtype=float)], axis=1),
                hovertemplate="previsto %{x:.1%}<br>osservato %{y:.1%}<br>%{customdata[1]:.0f} su %{customdata[0]:.0f} partite<extra>" + name + "</extra>",
            )
        )
    fig.update_layout(
        template="plotly",
        title={"text": title} if title else None,
        xaxis={"title": {"text": "Probabilità prevista dal modello"}, "range": [0, 1], "tickformat": ".0%"},
        yaxis={"title": {"text": "Frequenza osservata"}, "range": [0, 1], "tickformat": ".0%"},
        height=460,
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.0, "xanchor": "left", "x": 0.0, "title": None},
        margin={"l": 60, "r": 20, "t": 70 if title else 40, "b": 50},
    )
    return fig


def picks_track_for(report: ReliabilityReport, *, n_draws: int = DEFAULT_N_DRAWS, n_btts: int = DEFAULT_N_BTTS, frame: pd.DataFrame | None = None) -> PicksTrackRecord:
    """Track record of the top-``n_draws`` X and top-``n_btts`` GG lists of every past matchday (across and within sections)."""
    data = report.frame if frame is None else known_competition_rows(frame)
    return picks_track_record(data, n_draws=n_draws, n_btts=n_btts, odds_history=report.odds_history if report.has_odds else None, by_section=True)


def picks_record_table(
    track: PicksTrackRecord,
    section: str | None = None,
    *,
    n_draws: int = DEFAULT_N_DRAWS,
    n_btts: int = DEFAULT_N_BTTS,
) -> pd.DataFrame:
    """The lists' hit rate (Wilson 95%) against the model's mean probability and the ROI at real odds, of one scope (:data:`PICKS_RECORD_COLUMNS`).

    ``Lista`` reads ``'Top 5 X'`` / ``'Top 12 GG'`` (the sizes the track record was built with).
    """
    scope = ALL_SECTIONS_LABEL if section is None else section
    sizes = {"X": int(n_draws), "GG": int(n_btts)}
    rows = []
    summary = track.summary
    if len(summary):
        for r in summary.loc[summary["scope"].astype(str) == scope].to_dict(orient="records"):
            rows.append(
                {
                    "Lista": f"Top {sizes.get(str(r['list']), '')} {r['list']}".replace("Top  ", ""),
                    "Giornate": int(r["n_matchdays"]),
                    "Selezioni": int(r["n_picks"]),
                    "Vinte": int(r["hits"]),
                    "Attese": round(float(r["expected_hits"]), 1),
                    "Hit rate (%)": _pct(r["hit_rate"]),
                    "IC 95% (%)": _interval(r["wilson_low"], r["wilson_high"]),
                    "P media modello (%)": _pct(r["mean_p"]),
                    "Con quota": int(r["n_with_odds"]),
                    "ROI (%)": _pct(r["roi"]),
                    "IC 95% ROI (%)": _interval(r["roi_low"], r["roi_high"]),
                }
            )
    return _typed(pd.DataFrame(rows, columns=list(PICKS_RECORD_COLUMNS)), ("Lista", "IC 95% (%)", "IC 95% ROI (%)"))


#: Columns of :func:`accumulators_table`.
ACCUMULATORS_COLUMNS: Final[tuple[str, ...]] = (
    "Multipla",
    "Lista",
    "Partite",
    "Giornate",
    "Vinte",
    "Attese",
    "Win rate (%)",
    "IC 95% (%)",
    "P media (%)",
    "Con quota",
    "ROI (%)",
    "IC 95% ROI (%)",
)
#: Columns of :func:`picks_by_matchday_table`.
PICKS_BY_MATCHDAY_COLUMNS: Final[tuple[str, ...]] = (
    "Giornata",
    "Lista",
    "Selezioni",
    "Vinte",
    "Vincenti attese",
    "Tutte vinte",
)


def accumulators_table(track: PicksTrackRecord, section: str | None = None) -> pd.DataFrame:
    """How often the top-``k`` picks of a matchday **all** won (a multiple of the first ``k`` of the list), one scope (:data:`ACCUMULATORS_COLUMNS`).

    One row per list (X, GG) and size ``k`` (2 to 5): matchdays played, wins
    against the expected wins (sum over matchdays of the product of the ``k``
    probabilities, independence assumed), the win rate with its Wilson 95%
    interval and the ROI at stake 1 per multiple at the product of the best
    pre-match prices (only matchdays where every leg was quoted). This is the
    bet "top 2 GG of Serie C" or "top 4 GG of Serie D" replayed on every past
    matchday of the section.
    """
    scope = ALL_SECTIONS_LABEL if section is None else section
    acc = track.accumulators
    rows = []
    if len(acc):
        for r in acc.loc[acc["scope"].astype(str) == scope].to_dict(orient="records"):
            rows.append(
                {
                    "Multipla": f"Top {int(r['k'])} {r['list']}",
                    "Lista": str(r["list"]),
                    "Partite": int(r["k"]),
                    "Giornate": int(r["n_matchdays"]),
                    "Vinte": int(r["wins"]),
                    "Attese": round(float(r["expected_wins"]), 2),
                    "Win rate (%)": _pct(r["win_rate"]),
                    "IC 95% (%)": _interval(r["wilson_low"], r["wilson_high"]),
                    "P media (%)": _pct(r["mean_p"], 2),
                    "Con quota": int(r["n_with_odds"]),
                    "ROI (%)": _pct(r["roi"]),
                    "IC 95% ROI (%)": _interval(r["roi_low"], r["roi_high"]),
                }
            )
    frame = pd.DataFrame(rows, columns=list(ACCUMULATORS_COLUMNS))
    if len(frame):
        frame = frame.sort_values(["Lista", "Partite"], key=lambda c: c.map({"X": 0, "GG": 1}) if c.name == "Lista" else c, kind="stable").reset_index(drop=True)
    return _typed(frame, ("Multipla", "Lista", "IC 95% (%)", "IC 95% ROI (%)"))


def picks_by_matchday_table(track: PicksTrackRecord, section: str | None = None) -> pd.DataFrame:
    """Per past matchday and list, what the model would have listed and how it went, one scope (:data:`PICKS_BY_MATCHDAY_COLUMNS`), most recent first.

    Counts only (no match, no team): selections, how many won against the
    expected number (sum of the probabilities) and whether all of them won.
    """
    scope = ALL_SECTIONS_LABEL if section is None else section
    data = track.by_matchday
    rows = []
    if len(data):
        for r in data.loc[data["scope"].astype(str) == scope].to_dict(orient="records"):
            rows.append(
                {
                    "Giornata": pd.Timestamp(str(r["matchday"])),
                    "Lista": f"Top {int(r['n_picks'])} {r['list']}",
                    "Selezioni": int(r["n_picks"]),
                    "Vinte": int(r["hits"]),
                    "Vincenti attese": round(float(r["expected_hits"]), 2),
                    "Tutte vinte": "sì" if bool(r["all_won"]) else "no",
                }
            )
    frame = pd.DataFrame(rows, columns=list(PICKS_BY_MATCHDAY_COLUMNS))
    frame["Giornata"] = pd.to_datetime(frame["Giornata"]).astype("datetime64[ns]")
    frame = _typed(frame, ("Lista", "Tutte vinte"))
    return frame.sort_values(["Giornata", "Lista"], ascending=[False, True], kind="stable").reset_index(drop=True)


def slips_track_for(
    report: ReliabilityReport,
    sections: Sequence[str] | None = None,
    *,
    n_x: int = DEFAULT_SLIP_N_X,
    n_gg: int = DEFAULT_SLIP_N_GG,
    n_mixed: int = DEFAULT_SLIP_N_MIXED,
    min_probability: float = 0.0,
    frame: pd.DataFrame | None = None,
) -> SlipsTrackRecord:
    """The three slips every past matchday would have had (same builder as the live tab), scored at the real outcome and odds.

    ``sections`` restricts the candidate matches (``None`` = every section);
    ``frame`` replaces the raw backtest rows (the out-of-sample recalibrated ones).
    """
    data = report.frame if frame is None else known_competition_rows(frame)
    members: list[str] | None = None
    if sections is not None:
        members = list(dict.fromkeys(m for s in sections for m in section_members(s)))
        data = data.loc[data["section"].astype(str).isin(members).to_numpy(dtype=bool)] if len(data) else data
    builder = make_slip_builder(n_x=n_x, n_gg=n_gg, n_mixed=n_mixed, min_probability=min_probability, sections=members)
    return slips_track_record(data, odds_history=report.odds_history if report.has_odds else None, slip_builder=builder)


def slips_record_table(track: SlipsTrackRecord) -> pd.DataFrame:
    """Per slip: matchdays, wins against the expected wins, win rate (Wilson 95%), leg hit rate, ROI at real odds (:data:`SLIPS_RECORD_COLUMNS`)."""
    rows = []
    for r in track.summary.to_dict(orient="records"):
        rows.append(
            {
                "Schedina": SLIP_TITLES.get(str(r["slip"]), str(r["slip"])),
                "Giornate": int(r["n_slips"]),
                "Vinte": int(r["wins"]),
                "Vincite attese": round(float(r["expected_wins"]), 2),
                "Win rate (%)": _pct(r["win_rate"]),
                "IC 95% (%)": _interval(r["wilson_low"], r["wilson_high"]),
                "P media (%)": _pct(r["mean_p"], 2),
                "Partite (gambe)": int(r["n_legs"]),
                "Gambe vinte (%)": _pct(r["leg_hit_rate"]),
                "IC 95% gambe (%)": _interval(r["leg_wilson_low"], r["leg_wilson_high"]),
                "P media gamba (%)": _pct(r["mean_leg_p"]),
                "Con quota": int(r["n_with_odds"]),
                "ROI (%)": _pct(r["roi"]),
                "IC 95% ROI (%)": _interval(r["roi_low"], r["roi_high"]),
            }
        )
    return _typed(pd.DataFrame(rows, columns=list(SLIPS_RECORD_COLUMNS)), ("Schedina", "IC 95% (%)", "IC 95% gambe (%)", "IC 95% ROI (%)"))


#: Columns of :func:`slips_by_matchday_table`.
SLIPS_BY_MATCHDAY_COLUMNS: Final[tuple[str, ...]] = (
    "Giornata",
    "Schedina",
    "Partite",
    "Partite vinte",
    "Vincenti attese",
    "P schedina (%)",
    "Vinta",
    "Quota book",
    "Profitto (unità)",
)


def slips_by_matchday_table(track: SlipsTrackRecord) -> pd.DataFrame:
    """One row per past matchday and slip: legs won against the expected ones, the slip probability, won or not, the real price.

    ``Giornata`` is the first day of the football-week block of the matchday;
    most recent first. Your own local data only (the dashboard), never the CI log.
    """
    rows = [
        {
            "Giornata": pd.Timestamp(str(r["matchday"])),
            "Schedina": SLIP_TITLES.get(str(r["slip"]), str(r["slip"])),
            "Partite": int(r["n_legs"]),
            "Partite vinte": int(r["legs_won"]),
            "Vincenti attese": round(float(r["expected_legs"]), 2),
            "P schedina (%)": _pct(r["p_slip"], 2),
            "Vinta": "sì" if bool(r["won"]) else "no",
            "Quota book": _round(r["book_odds"], 2),
            "Profitto (unità)": _round(r["profit"], 2),
        }
        for r in track.slips.to_dict(orient="records")
    ]
    frame = pd.DataFrame(rows, columns=list(SLIPS_BY_MATCHDAY_COLUMNS))
    frame["Giornata"] = pd.to_datetime(frame["Giornata"]).astype("datetime64[ns]")
    frame = _typed(frame, ("Schedina", "Vinta"))
    return frame.sort_values(["Giornata", "Schedina"], ascending=[False, True], kind="stable").reset_index(drop=True)


def value_track_for(report: ReliabilityReport, section: str | None = None, *, min_ev: float = 0.035, frame: pd.DataFrame | None = None) -> pd.DataFrame:
    """The value singles the model would have flagged (EV >= ``min_ev`` at the best pre-match price): hit rate and ROI per market, one scope."""
    columns = ["Mercato", "Giocate", "Vinte", "Hit rate (%)", "IC 95% (%)", "P media modello (%)", "Quota media", "ROI (%)", "IC 95% ROI (%)"]
    if not report.has_odds:
        return _typed(pd.DataFrame(columns=columns), ("Mercato", "IC 95% (%)", "IC 95% ROI (%)"))
    track: ValueTrackRecord = value_track_record(report.frame if frame is None else known_competition_rows(frame), report.odds_history, min_ev=min_ev, by_section=True)
    scope = ALL_SECTIONS_LABEL if section is None else section
    rows = []
    for r in track.summary.loc[track.summary["scope"].astype(str) == scope].to_dict(orient="records") if len(track.summary) else []:
        rows.append(
            {
                "Mercato": "Tutti" if r["market"] == "Tutti" else BACKTEST_MARKET_LABELS.get(str(r["market"]), str(r["market"])),
                "Giocate": int(r["n_bets"]),
                "Vinte": int(r["hits"]),
                "Hit rate (%)": _pct(r["hit_rate"]),
                "IC 95% (%)": _interval(r["wilson_low"], r["wilson_high"]),
                "P media modello (%)": _pct(r["mean_p"]),
                "Quota media": _round(r["mean_odds"], 2),
                "ROI (%)": _pct(r["roi"]),
                "IC 95% ROI (%)": _interval(r["roi_low"], r["roi_high"]),
            }
        )
    return _typed(pd.DataFrame(rows, columns=columns), ("Mercato", "IC 95% (%)", "IC 95% ROI (%)"))


def recalibration_table(model: RecalibrationModel) -> pd.DataFrame:
    """Per market: whether the correction is applied, its out-of-sample log loss before/after and why (:data:`RECALIBRATION_COLUMNS`)."""
    summary = model.summary_frame()
    rows = [
        {
            "Mercato": BACKTEST_MARKET_LABELS.get(str(r["market"]), str(r["market"])),
            "Applicata": "sì" if bool(r["enabled"]) else "no",
            "Partite stima": int(r["n_fit"]),
            "Partite verifica": int(r["n_eval"]),
            "Log loss prima": _round(r["log_loss_before"]),
            "Log loss dopo": _round(r["log_loss_after"]),
            "Miglioramento (%)": _pct(r["gain"], 2),
            "Brier prima": _round(r["brier_before"]),
            "Brier dopo": _round(r["brier_after"]),
            "Parametri": str(r["parameters"]),
            "Motivo": str(r["reason"]) or ("miglioramento sopra la soglia" if bool(r["enabled"]) else ""),
        }
        for r in summary.to_dict(orient="records")
    ]
    return _typed(pd.DataFrame(rows, columns=list(RECALIBRATION_COLUMNS)), ("Mercato", "Applicata", "Parametri", "Motivo"))


def reliability_export_frame(report: ReliabilityReport, sections: Sequence[str] | None = None) -> pd.DataFrame:
    """Aggregated scores per section and market (:data:`RELIABILITY_EXPORT_COLUMNS`): every section plus *Tutte le sezioni*.

    Aggregates only (no match, no odds): the ``<stem>_affidabilita.csv`` of
    ``main.py`` and the dashboard download.
    """
    present = list(dict.fromkeys(report.frame["section"].astype(str))) if len(report.frame) else []
    scopes = [ALL_SECTIONS_LABEL, *(present if sections is None else [s for s in sections if s in present])]
    rows: list[dict[str, object]] = []
    for scope in scopes:
        data = report.section_frame(scope)
        if data.empty:
            continue
        model = metrics(data, by=())
        cmp = market_comparison(data, report.odds_history, by=()) if report.has_odds else pd.DataFrame()
        by_market = {str(r["market"]): r for r in cmp.to_dict(orient="records")} if len(cmp) else {}
        for r in model.to_dict(orient="records"):
            c = by_market.get(str(r["market"]))
            rows.append(
                {
                    "Sezione": scope,
                    "Mercato": str(r["market"]),
                    "Esito di riferimento": str(r["selection"]),
                    "Partite": int(r["n"]),
                    "Log loss modello": _round(r["log_loss"], 5),
                    "Log loss base": _round(r["log_loss_naive"], 5),
                    "Skill vs base": _round(r["skill_naive"], 5),
                    "Brier modello": _round(r["brier"], 5),
                    "Accuratezza": _round(r["accuracy"], 5),
                    "P media": _round(r["mean_pred"], 5),
                    "Frequenza": _round(r["observed"], 5),
                    "Partite con quote": int(c["n"]) if c is not None else 0,
                    QUOTED_LOG_LOSS_COLUMN: _round(c["log_loss_model"], 5) if c is not None else float("nan"),
                    "Log loss mercato": _round(c["log_loss_market"], 5) if c is not None else float("nan"),
                    "Skill vs mercato": _round(c["skill"], 5) if c is not None else float("nan"),
                    QUOTED_BRIER_COLUMN: _round(c["brier_model"], 5) if c is not None else float("nan"),
                    "Brier mercato": _round(c["brier_market"], 5) if c is not None else float("nan"),
                }
            )
    return _typed(pd.DataFrame(rows, columns=list(RELIABILITY_EXPORT_COLUMNS)), ("Sezione", "Mercato", "Esito di riferimento"))


# =============================================================================
# Slips (schedine) of a matchday
# =============================================================================
#: Columns of :func:`slips_export_frame`: the matchday, the leg table and the slip's own figures.
SLIPS_EXPORT_COLUMNS: Final[tuple[str, ...]] = (
    "Giornata",
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
    "Probabilità schedina (%)",
    "Quota fair schedina",
    "Quota book schedina",
    "EV schedina (%)",
    "Partite vincenti attese",
    "Stake suggerito",
)


def _union_key(sections: Sequence[str]) -> str:
    return "slips:" + "|".join(sections)


def slips_gironi(sections: Sequence[str] | None) -> tuple[Girone, ...]:
    """The competitions of a set of sections (every competition for ``None``), configuration order."""
    if sections is None:
        return tuple(Girone)
    wanted = {m for s in sections for m in section_members(s)}
    return tuple(g for g in Girone if any(in_section(g, m) for m in wanted))


def slips_matchdays(state: PipelineState, sections: Sequence[str] | None = None) -> list[Matchday]:
    """The "giornate" of a set of sections taken together (the current round of each of their competitions first).

    One section gives exactly :func:`matchdays` of that section (the selector
    is then shared with the Giornata, Valore and Selezioni tabs); several
    sections are grouped together from their own official rounds. Cached on
    the state per set of sections.
    """
    chosen = None if sections is None else tuple(dict.fromkeys(sections))
    if chosen is not None and len(chosen) == 1:
        return matchdays(state, chosen[0])
    if chosen is None:
        return matchdays(state, None)
    key = _union_key(chosen)
    cached = state._section_matchdays.get(key)
    if cached is None:
        cached = group_matchdays(state.calendar_fixtures, gironi=slips_gironi(chosen), played=state.current_matches, now=state.now)
        state._section_matchdays[key] = cached
    return list(cached)


def default_slips_matchday(days: Sequence[Matchday], reference: Matchday | None = None) -> Matchday | None:
    """The matchday of ``days`` that overlaps ``reference`` (the matchday selected in the other tabs), else the first one."""
    if not days:
        return None
    if reference is not None:
        for day in days:
            if day.key == reference.key:
                return day
        for day in days:
            if day.start <= reference.end and reference.start <= day.end:
                return day
    return days[0]


def slips_for(
    state: PipelineState,
    matchday: Matchday | None,
    sections: Sequence[str] | None = None,
    *,
    n_x: int = DEFAULT_SLIP_N_X,
    n_gg: int = DEFAULT_SLIP_N_GG,
    n_mixed: int = DEFAULT_SLIP_N_MIXED,
    min_probability: float = 0.0,
    only_quoted: bool = False,
) -> list[Slip]:
    """The three slips of a matchday (:func:`src.engine.slips.build_slips`): ``[X, GG, MISTA]``.

    Candidates are the state's predictions of the chosen sections that belong
    to ``matchday`` and have not kicked off at :attr:`PipelineState.now`; the
    probabilities are the state's (recalibrated when the state is, see
    :func:`apply_recalibration`), the book prices the best ones of the value
    opportunities and of the raw quotes. ``None`` matchday: three empty slips.
    """
    chosen = None if sections is None else list(dict.fromkeys(m for s in sections for m in section_members(s)))
    preds = [p for p in state.predictions if chosen is None or any(in_section(p.fixture.girone, m) for m in chosen)]
    if matchday is None:
        preds = []
    return build_slips(
        preds,
        now=state.now,
        opportunities=state.opportunities,
        quotes=state.quotes,
        n_x=n_x,
        n_gg=n_gg,
        n_mixed=n_mixed,
        min_probability=min_probability,
        only_quoted=only_quoted,
        sections=chosen,
        matchday=matchday,
    )


def slips_summary_for(slips: Sequence[Slip], state: PipelineState) -> pd.DataFrame:
    """:func:`src.engine.slips.slips_summary` with the state's bankroll and Kelly fraction (stake capped at 1 unit)."""
    return slips_summary(slips, bankroll_units=state.bankroll_units, fraction=state.kelly_fraction)


def slips_export_frame(
    slips: Sequence[Slip],
    matchday: Matchday | None,
    state: PipelineState | None = None,
    *,
    bankroll_units: float = 100.0,
    fraction: float = 0.125,
) -> pd.DataFrame:
    """One row per leg with the matchday and the figures of its slip (:data:`SLIPS_EXPORT_COLUMNS`).

    The stake follows the state's bankroll and Kelly fraction (``bankroll_units`` /
    ``fraction`` without a state, as in ``main.py``).
    """
    legs = slips_frame(slips)
    if state is not None:
        bankroll_units, fraction = state.bankroll_units, state.kelly_fraction
    summary = slips_summary(slips, bankroll_units=bankroll_units, fraction=fraction).rename(
        columns={"Probabilità (%)": "Probabilità schedina (%)", "Quota fair": "Quota fair schedina", "Quota book": "Quota book schedina", "EV (%)": "EV schedina (%)"}
    )
    merged = legs.merge(summary.drop(columns=["Partite"]), on="Schedina", how="left")
    merged.insert(0, "Giornata", "" if matchday is None else matchday.label)
    merged["Giornata"] = merged["Giornata"].astype("string")
    return merged[list(SLIPS_EXPORT_COLUMNS)].reset_index(drop=True)


def slips_csv_bytes(slips: Sequence[Slip], matchday: Matchday | None, state: PipelineState) -> bytes:
    """CSV bytes (see :func:`export_csv_bytes`) of :func:`slips_export_frame` (marked in the demo)."""
    return export_csv_bytes(slips_export_frame(slips, matchday, state), demo=state.is_demo)
