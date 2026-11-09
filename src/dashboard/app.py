"""Serie C Quant Lab -- Streamlit dashboard (UI in Italian).

Run from the repository root::

    streamlit run src/dashboard/app.py

Data
----
The dashboard shows **real FootyStats data only**, downloaded at runtime with
the user's own key and kept locally (``data/processed``, ``data/odds``,
git-ignored: FootyStats data is never committed to the public repository).
At start, when a key is configured (``FOOTYSTATS_API_KEY`` in the environment,
in ``.env`` or in the Streamlit secrets -- on Streamlit Cloud *App settings ->
Secrets*: ``FOOTYSTATS_API_KEY = "..."``) and the local data is missing or
older than 6 hours, it is downloaded again (:mod:`src.dashboard.data_refresh`,
once per 6 hours per server, with a progress bar); the sidebar says when the
data was last downloaded and offers *Aggiorna dati ora*. While the data stays
older than 6 hours (no key, a failed download) a warning with its age is shown
on every run, and so is one per competition that still carries the data of an
earlier download. Without data the page is a **setup page** (what is missing,
how to provide the key, *Scarica i dati da FootyStats adesso* when a key
exists) and no tab is shown: the dashboard never invents numbers silently.

Environment variables
---------------------
``SERIEC_DEMO=1``
    Demo mode: the synthetic league (invented results, dates, pairings and
    odds) instead of real data, with the red banner :data:`DEMO_WARNING` at the
    top of the page and of every tab and the sidebar badge *Demo*; every CSV
    download is named ``DEMO_...`` and starts with the column ``AVVISO`` =
    ``DATI SIMULATI - NON USARE PER SCOMMETTERE``. Never on by default.
    ``SERIEC_FORCE_SIMULATE=1`` is a deprecated alias.
``SERIEC_PROCESSED_DIR`` / ``SERIEC_ODDS_DIR`` (default ``data/processed`` / ``data/odds``)
    Where the real data is read from and downloaded to.
``SERIEC_SEASON`` (default ``2026_27``)
    Registry season (``config/teams_<season>.json``): the perimeter of teams that
    are ranked, predicted and priced, in every competition the file holds (the
    competition selector, the tables and the perimeter banner follow
    ``config/competitions.toml``). Older seasons in the data are history only.
``SERIEC_AS_OF`` (ISO date, default today)
    Calendar position: for real data the official-round rule is evaluated at
    midnight of that day (without the variable: at the current instant, on
    every run); in the demo the rounds dated before it are played and the next
    round is the fixture list. Resolved on every run (:func:`resolve_as_of`).
``SERIEC_HISTORY`` (int 0-2, default 2), ``SERIEC_SIM_ROUNDS`` (int >= 1, optional), ``SERIEC_SEED`` (int >= 0, default 7)
    Demo only: synthetic past seasons, fixed number of played rounds (clamped so
    one round of fixtures remains) and random seed of the simulated league.
``SERIEC_BACKTEST`` (``auto`` default, ``manual``, ``off``)
    When the walk-forward backtest of *Affidabilità modello* (and so the
    recalibration) is computed: at start, on the tab's button, or never.

Invalid or out-of-range values (non-integers, a negative seed, zero rounds, an
unparseable date, an unknown season file) are logged and replaced by the
defaults instead of crashing the page; should the pipeline itself refuse a
valid-looking combination (``ValueError``), the page shows the error and falls
back to the default season, calendar and seed.

Giornata
--------
The *Giornata* tab lists **every match** of the selected rounds
(:func:`src.dashboard.state.matchday_table_for`): the current round of each
competition with the matches already played (``Stato`` ``giocata 2-1``, no
prediction), the ones that have kicked off without a result in the data
(``in corso / risultato in attesa``, no prediction: never a pre-match bet), the
ones to play with their kick-off in ``Europe/Rome`` and the model's numbers,
and the postponed ones (``rinviata``). With one competition
selected in the sidebar a *Giornata n.* selector shows any official round of it;
the *Recuperi e rinvii* expander lists the postponed matches and the isolated
recoveries with their original date.

Sections
--------
The sidebar radio *Sezione* (key ``'section'``) has one page per league of
``config/competitions.toml`` (:data:`src.dashboard.state.SECTIONS`): *Serie C*
(default, gironi A/B/C), *Serie B*, *Serie D* and every league switched on with
``enabled = true`` (Serie A, the European top flights). The Dixon-Coles fit is
one per *country* (:func:`src.engine.fitgroups.fit_by_group`, L-BFGS-B by
default, SLSQP from *Opzioni del modello*); every tab, selector, table, pick
list, export and the perimeter text are restricted to the selected section, the
picks (5 X / 12 GG) are ranked within it and the matchdays are grouped from its
own fixtures. The *Competizione* selector lists *Tutti* plus the section's
competitions; the matchday, fixture and filter widgets have per-section keys
(:func:`src.dashboard.state.section_key`), so switching section starts from that
section's defaults. A section configured but without data (Serie D before its
first harvest, a league just enabled) shows one notice per tab with its status
and how to download it.

Schedine and Affidabilità modello
---------------------------------
*Schedine* proposes, for the selected matchday and the chosen sections (default:
every section with data), three slips -- 5 X, 10 GG and a mixed one (best sign
among 1, 2, GG, Over 2.5) -- with probability, fair and book odds, EV, expected
winning legs, suggested stake, the accumulator-margin warning, a CSV and the
past record of the same slips. *Affidabilità modello* shows the **walk-forward**
backtest (every past matchday predicted with the matches before it only, never
the production fit): scores per market against the league base rates and the
de-vigged bookmaker, the reliability chart with Wilson intervals, the scores per
competition, the track record of the lists, slips and value singles with their
intervals, and the validated recalibration with the *Usa probabilità
ricalibrate* toggle, which switches the probabilities of Giornata, Opportunità,
Selezioni and Schedine (read before any tab renders). The backtest runs once per
data version (``SERIEC_BACKTEST=auto``, the default; ``manual`` = on the tab's
button, ``off`` = never), is shared by every session and persisted in
``data/backtest.parquet`` / ``data/backtest.json`` (git-ignored).

The heavy work (load/simulate, fit, predict, value) lives in
:mod:`src.dashboard.state` and is cached with ``st.cache_resource`` keyed on
the sidebar parameters; this module only composes the widgets. Full-width
elements use :func:`src.dashboard.container_width_kwargs`, which picks
``width="stretch"`` or the legacy ``use_container_width=True`` according to
the installed Streamlit (the project declares ``streamlit>=1.38``).
"""
from __future__ import annotations

import os
import sys
import threading
import time
from dataclasses import dataclass, field
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Final

import pandas as pd

# -- bootstrap: make ``src.*`` importable when launched as ``streamlit run src/dashboard/app.py``
_REPO_ROOT: Final[Path] = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import streamlit as st

st.set_page_config(page_title="Serie C Quant Lab", page_icon="⚽", layout="wide")

from src.common.competitions import describe_competitions
from src.common.logging_config import get_logger
from src.common.paths import ODDS_DIR, PROCESSED_DIR, ROOT
from src.common.schemas import Girone, Market, MatchPrediction
from src.common.teams import DEFAULT_SEASON, list_seasons, season_file_stem
from src.dashboard import container_width_kwargs
from src.dashboard import data_refresh
from src.dashboard.legend import (
    POOR_EXAMPLE_FLAGS,
    SYNTHETIC_REASON_NO_QUOTES,
    SYNTHETIC_REASON_NO_VALUE,
    WorkedExample,
    legend_markdown,
    pick_example_row,
    worked_example_markdown,
)
from src.dashboard.state import (
    ALL_MATCHDAYS,
    ALL_SECTIONS_LABEL,
    DEFAULT_BACKTEST_SEASONS,
    DEFAULT_HISTORY_SEASONS,
    DEFAULT_SECTION,
    DEMO_BADGE,
    DEMO_WARNING,
    EVIDENCE_NOTE,
    INDEPENDENCE_NOTE,
    MATCHDAY_TABLE_KEY_PREFIX,
    MAX_HISTORY_SEASONS,
    MIN_SIM_ROUNDS,
    QUOTED_BRIER_COLUMN,
    QUOTED_LOG_LOSS_COLUMN,
    RAW_COMPARE_COLUMNS,
    RELIABILITY_SELECTIONS,
    SECTION_D,
    Matchday,
    NoDataError,
    PipelineState,
    ReliabilityReport,
    ValidatedTuning,
    accumulators_table,
    apply_recalibration,
    btts_picks_table,
    build_state,
    competition_metrics_table,
    default_slips_matchday,
    headline_for,
    market_card,
    metrics_table_for,
    picks_by_matchday_table,
    picks_record_table,
    picks_track_for,
    raw_column_name,
    recalibration_table,
    reliability_export_frame,
    reliability_figure,
    reliability_for_state,
    reliability_key,
    reliability_table_for,
    rolling_frame,
    section_members,
    sections_with_data,
    slips_csv_bytes,
    slips_for,
    slips_matchdays,
    slips_by_matchday_table,
    slips_record_table,
    slips_summary_for,
    slips_track_for,
    tuning_disclosure,
    tuning_table,
    validated_tuning,
    value_track_for,
    with_rolling,
    competition_options,
    competition_rounds,
    coverage_text,
    current_round_for,
    demo_file_name,
    demo_requested,
    draw_picks_table,
    export_csv_bytes,
    find_matchday,
    fixture_label,
    heatmap_figure,
    in_section,
    markets_table,
    matchday_table_for,
    matchday_table_key,
    matchdays,
    no_odds_message,
    opportunities_table,
    perimeter_text,
    picks_csv_bytes,
    predictions_table,
    quotes_for,
    ranking_table,
    recoveries_table_for,
    results_table_for,
    results_team_options,
    seasons_available,
    section_key,
    section_opportunities,
    section_options,
    section_predictions,
    section_ready,
    section_slug,
    section_traps,
    selected_row_match_id,
    serie_d_gironi,
    serie_d_harvest_command,
    state_at,
    strength_figure,
    traps_table,
    value_opportunities,
    value_table,
)
from src.engine.calendar import IN_PLAY_TEXT, STATUS_TO_PLAY, rome_midnight
from src.engine.calibration import DEFAULT_MIN_GAIN
from src.engine.fitgroups import DEFAULT_FIT_METHOD
from src.engine.picks import DEFAULT_N_BTTS, DEFAULT_N_DRAWS
from src.engine.slips import DEFAULT_N_GG, DEFAULT_N_MIXED, DEFAULT_N_X, MIXED_MAX_LEGS, MIXED_MIN_LEGS, slips_frame
from src.engine.tables import DEFAULT_FORM_N, DEFAULT_XG_N, RECOVERIES_COLUMNS

log = get_logger(__name__)

#: Competition selector of the whole perimeter: *Tutti* plus the short label of every configured competition
#: (``'Girone A'``, ..., ``'Serie B'``, ``'D-A'``, ..., ``'D-I'`` with the shipped ``config/competitions.toml``). The
#: sidebar shows the subset of the selected section, :func:`src.dashboard.state.competition_options`.
GIRONE_OPTIONS: Final[tuple[str, ...]] = competition_options(None)
#: Competitions named in the sidebar caption, e.g. ``'Serie C Gironi A/B/C + Serie B + Serie D Gironi A-I'``.
COMPETITIONS_TEXT: Final[str] = describe_competitions()
#: ``st.session_state`` keys of the sidebar *Sezione* radio and *Competizione* selectbox (shared by the sections:
#: a competition of another section is not a valid option, so switching section falls back to *Tutti*).
SECTION_STATE_KEY: Final[str] = "section"
GIRONE_STATE_KEY: Final[str] = "girone"
#: Launchers that run the FootyStats harvester with the key in ``.env`` (named in the Serie D notice).
LAUNCHERS_HINT: Final[str] = "avvia.bat (Windows) / avvia.command (macOS)"
KELLY_OPTIONS: Final[dict[str, float]] = {"1/8": 0.125, "1/4": 0.25}
#: Played rounds the dashboard simulated before the calendar-anchored league; kept for compatibility,
#: the default is now ``None`` (rounds derived from ``SERIEC_AS_OF``/today).
DEFAULT_SIM_ROUNDS: Final[int] = 15
DEFAULT_SEED: Final[int] = 7
#: Smallest "goals per team" the heatmap slider allows (a compact view folds the tail into an ``N+`` cell).
HEATMAP_MIN_GOALS: Final[int] = 3
#: Full-width keyword per Streamlit command, resolved once for the installed release.
FULL_WIDTH: Final[dict[str, dict[str, object]]] = {
    name: container_width_kwargs(getattr(st, name)) for name in ("button", "download_button", "dataframe", "plotly_chart")
}
#: ``st.session_state`` key holding the selected matchday's :attr:`Matchday.key`, shared by the Match, Value and Picks
#: tabs. Like every per-section key below it is the bare name for the default section and gets the section slug
#: appended for the others (``'matchday_d'``, see :func:`src.dashboard.state.section_key`).
MATCHDAY_STATE_KEY: Final[str] = "matchday"
#: ``st.session_state`` key of the value panel's "every matchday" checkbox, read again by the Legenda tab.
VALUE_ALL_MATCHDAYS_KEY: Final[str] = "value_all_matchdays"
#: Largest list size the picks tab accepts (a matchday never has more fixtures than this).
MAX_PICKS: Final[int] = 60
#: ``st.session_state`` keys of the *Giornata* tab: the prefix of the matchday table widget key (the full key is
#: scoped to the matchday and competition shown, :func:`matchday_table_key`, so a row selection never outlives
#: the table it was made on), the *Partita* selectbox and the ``(table key, match_id)`` of the last table row
#: applied to that selectbox (so a later manual choice is not overridden).
MATCHDAY_TABLE_KEY: Final[str] = MATCHDAY_TABLE_KEY_PREFIX
FIXTURE_STATE_KEY: Final[str] = "fixture"
APPLIED_ROW_STATE_KEY: Final[str] = "matchday_table_row"
#: ``st.session_state`` keys of the *Risultati* tab filters and table.
RESULTS_TEAM_KEY: Final[str] = "results_team"
RESULTS_SEASON_KEY: Final[str] = "results_season"
RESULTS_TABLE_KEY: Final[str] = "results_table"
#: Option of the *Squadra* filter meaning every team.
ALL_TEAMS_LABEL: Final[str] = "Tutte"
#: Height (pixels) of the matchday and results tables: about 17 rows, a whole Serie C round of a girone and more.
TABLE_HEIGHT: Final[int] = 600
#: ``st.session_state`` key prefix of the *Giornata n.* selector of the Giornata tab (one per competition and section).
ROUND_STATE_KEY: Final[str] = "round"
#: Labels and keys of the data controls: the sidebar refresh button, the setup page's download button, the expander
#: of the recoveries and the session key holding the last :class:`~src.dashboard.data_refresh.RefreshResult`.
REFRESH_NOW_LABEL: Final[str] = "Aggiorna dati ora"
REFRESH_BUTTON_KEY: Final[str] = "refresh_now"
SETUP_DOWNLOAD_LABEL: Final[str] = "Scarica i dati da FootyStats adesso"
SETUP_DOWNLOAD_KEY: Final[str] = "setup_download"
RECOVERIES_LABEL: Final[str] = "Recuperi e rinvii"
LAST_REFRESH_STATE_KEY: Final[str] = "last_refresh"
#: The line to paste in Streamlit Cloud, App settings -> Secrets.
SECRETS_LINE: Final[str] = 'FOOTYSTATS_API_KEY = "..."'
#: Title of the page (the sidebar title repeats it).
PAGE_TITLE: Final[str] = "⚽ Serie C Quant Lab"
#: Tab labels, in order.
TAB_LABELS: Final[tuple[str, ...]] = (
    "Classifica Forza",
    "Giornata",
    "Risultati",
    "Opportunità di Valore",
    "Selezioni di giornata",
    "Schedine",
    "Affidabilità modello",
    "Export",
    "Legenda",
)
#: Environment variable of the walk-forward backtest behind *Affidabilità modello* and the recalibration:
#: ``auto`` (default: computed at start once per data version, cached and persisted in data/backtest.*),
#: ``manual`` (only on the button of the tab, or read back from disk), ``off`` (never).
BACKTEST_ENV_VAR: Final[str] = "SERIEC_BACKTEST"
BACKTEST_MODES: Final[tuple[str, ...]] = ("auto", "manual", "off")
#: ``st.session_state`` keys of the reliability tab: the recalibration toggle (read before any tab renders), the
#: request of a manual backtest, the evaluated seasons, the scope (section or every section), the reliability
#: selection, the tuning button and the switch that applies the tuned xi / prior.
RECALIBRATION_STATE_KEY: Final[str] = "use_recalibration"
BACKTEST_REQUEST_KEY: Final[str] = "backtest_requested"
BACKTEST_BUTTON_KEY: Final[str] = "run_backtest"
RELIABILITY_SEASONS_KEY: Final[str] = "reliability_seasons"
RELIABILITY_SCOPE_KEY: Final[str] = "reliability_scope"
RELIABILITY_SELECTION_KEY: Final[str] = "reliability_selection"
TUNING_BUTTON_KEY: Final[str] = "run_tuning"
USE_TUNED_KEY: Final[str] = "use_tuned"
#: Options of the evaluated-seasons selector (number of seasons up to the current one).
RELIABILITY_SEASON_OPTIONS: Final[dict[int, str]] = {1: "Stagione corrente", 2: "Stagione corrente e precedente"}
#: ``st.session_state`` keys of the slips tab.
SLIPS_SECTIONS_KEY: Final[str] = "slips_sections"
SLIPS_MATCHDAY_KEY: Final[str] = "slips_matchday"
SLIPS_N_X_KEY: Final[str] = "slips_n_x"
SLIPS_N_GG_KEY: Final[str] = "slips_n_gg"
SLIPS_N_MIXED_KEY: Final[str] = "slips_n_mixed"
SLIPS_MIN_P_KEY: Final[str] = "slips_min_probability"
SLIPS_ONLY_QUOTED_KEY: Final[str] = "slips_only_quoted"
#: Largest legs of the X and GG slips the sliders allow.
SLIPS_MAX_X: Final[int] = 15
SLIPS_MAX_GG: Final[int] = 20
#: ``st.session_state`` key of the optimiser selector (sidebar, *Opzioni del modello*).
FIT_METHOD_KEY: Final[str] = "fit_method"
FIT_METHODS: Final[tuple[str, ...]] = (DEFAULT_FIT_METHOD, "SLSQP")


# =============================================================================
# Environment
# =============================================================================
def env_flag(name: str) -> bool:
    """``True`` when the environment variable is ``1/true/yes/on`` (case-insensitive)."""
    return os.getenv(name, "").strip().lower() in {"1", "true", "yes", "on"}


def env_int(name: str, default: int, *, minimum: int | None = None) -> int:
    """Integer environment variable with a safe fallback.

    Non-integers and values below ``minimum`` (when given) are logged and
    replaced by ``default``, so a bad ``SERIEC_SEED=-1`` or ``SERIEC_SIM_ROUNDS=0``
    never reaches the pipeline and the page still renders.
    """
    raw = os.getenv(name)
    if raw is None or not raw.strip():
        return default
    try:
        value = int(raw.strip())
    except ValueError:
        log.warning("invalid integer environment variable, using default", extra={"variable": name, "value": raw, "default": default})
        return default
    if minimum is not None and value < minimum:
        log.warning(
            "environment variable below minimum, using default",
            extra={"variable": name, "value": value, "minimum": minimum, "default": default},
        )
        return default
    return value


def env_optional_int(name: str, *, minimum: int | None = None, maximum: int | None = None) -> int | None:
    """Integer environment variable that may be absent: ``None`` when unset, blank or invalid.

    Out-of-range values (below ``minimum`` or above ``maximum``) are logged and
    treated as unset, so ``SERIEC_SIM_ROUNDS=0`` falls back to the calendar.
    """
    raw = os.getenv(name)
    if raw is None or not raw.strip():
        return None
    try:
        value = int(raw.strip())
    except ValueError:
        log.warning("invalid integer environment variable, ignored", extra={"variable": name, "value": raw})
        return None
    if (minimum is not None and value < minimum) or (maximum is not None and value > maximum):
        log.warning(
            "environment variable out of range, ignored",
            extra={"variable": name, "value": value, "minimum": minimum, "maximum": maximum},
        )
        return None
    return value


def env_date(name: str) -> date | None:
    """ISO date (``YYYY-MM-DD``) environment variable; ``None`` when unset or unparseable (logged)."""
    raw = os.getenv(name)
    if raw is None or not raw.strip():
        return None
    try:
        return date.fromisoformat(raw.strip()[:10])
    except ValueError:
        log.warning("invalid date environment variable, using today", extra={"variable": name, "value": raw})
        return None


def env_season(name: str, default: str = DEFAULT_SEASON) -> str:
    """Registry season from the environment, validated against the ``config/teams_*.json`` files."""
    raw = os.getenv(name)
    if raw is None or not raw.strip():
        return default
    stem = season_file_stem(raw)
    if stem not in list_seasons():
        log.warning("unknown season file, using default", extra={"variable": name, "value": raw, "available": list_seasons(), "default": default})
        return default
    return stem


def env_path(name: str, default: Path) -> Path:
    """Directory from the environment (``~`` expanded), ``default`` when unset or blank."""
    raw = os.getenv(name)
    if raw is None or not raw.strip():
        return default
    return Path(raw.strip()).expanduser()


def processed_dir() -> Path:
    """Directory of the real matches / fixtures CSVs: ``SERIEC_PROCESSED_DIR`` or ``data/processed``."""
    return env_path("SERIEC_PROCESSED_DIR", PROCESSED_DIR)


def odds_dir() -> Path:
    """Directory of the odds CSVs: ``SERIEC_ODDS_DIR`` or ``data/odds``."""
    return env_path("SERIEC_ODDS_DIR", ODDS_DIR)


def data_stamp(directory: Path) -> float | None:
    """Modification time of the newest local data file (``matches.csv``, ``fixtures.csv``, the refresh stamp): the cache key of the data.

    A refresh writes new files, so the cached pipeline state of the old data is
    never reused; ``None`` when there is no data.
    """
    stamps: list[float] = []
    for name in ("matches.csv", "fixtures.csv", data_refresh.STAMP_FILE):
        try:
            stamps.append((Path(directory) / name).stat().st_mtime)
        except OSError:
            continue
    return max(stamps) if stamps else None


def calendar_now() -> datetime:
    """Reference instant of the official-round rule for real data: midnight of ``SERIEC_AS_OF`` in Rome, else now (to the minute)."""
    day = env_date("SERIEC_AS_OF")
    if day is not None:
        return rome_midnight(day)
    return datetime.now(UTC).replace(second=0, microsecond=0)


def resolve_as_of() -> date:
    """Calendar position of the simulated league: ``SERIEC_AS_OF`` when valid, otherwise today.

    Resolved on every script run and passed to :func:`load_state` explicitly, so
    the cache key changes every day and the played rounds follow the real
    calendar while the server runs (a ``None`` key would freeze the day the
    state was first computed).
    """
    return env_date("SERIEC_AS_OF") or date.today()


# =============================================================================
# Cached pipeline
# =============================================================================
@st.cache_resource(show_spinner="Stima del modello Dixon-Coles in corso…", max_entries=8)
def load_state(
    *,
    min_ev: float,
    kelly_fraction: float,
    bankroll_units: float,
    simulate: bool | None,
    seed: int,
    n_rounds: int | None = None,
    season: str = DEFAULT_SEASON,
    history_seasons: int = DEFAULT_HISTORY_SEASONS,
    as_of: date | None = None,
    processed: str | None = None,
    odds: str | None = None,
    stamp: float | None = None,
    method: str = DEFAULT_FIT_METHOD,
    xi: float = 0.0018,
    prior: str = "auto",
) -> PipelineState:
    """Build (and cache) the pipeline state for one combination of sidebar parameters and one version of the local data.

    ``processed`` / ``odds`` are the data directories (default ``data/processed``
    / ``data/odds``) and ``stamp`` the data version (:func:`data_stamp`): it is
    only part of the cache key, so new data downloaded by a refresh yields a new
    state. Raises :class:`~src.dashboard.state.NoDataError` without real data
    (``simulate`` not ``True``): the exception is not cached, the next run tries
    again. ``method`` is the optimiser of the per-country fit (L-BFGS-B by
    default, SLSQP from *Opzioni del modello*), ``xi`` / ``prior`` the model
    settings (the defaults, or the walk-forward tuned ones when the user applies
    them in *Affidabilità modello*).
    """
    return build_state(
        method=method,
        xi=xi,
        prior=prior,
        processed_dir=Path(processed) if processed else PROCESSED_DIR,
        odds_dir=Path(odds) if odds else ODDS_DIR,
        min_ev=min_ev,
        kelly_fraction=kelly_fraction,
        bankroll_units=bankroll_units,
        simulate=simulate,
        seed=seed,
        n_rounds=n_rounds,
        season=season,
        history_seasons=history_seasons,
        as_of=as_of,
    )


@st.cache_resource(max_entries=16)
def calendar_state(_state: PipelineState, built_at: str, now: datetime, variant: str = "raw") -> PipelineState:
    """The cached state with its calendar regrouped at ``now`` (:func:`src.dashboard.state.state_at`), cached per minute.

    ``_state`` is not hashed (Streamlit skips arguments with a leading
    underscore): ``built_at`` identifies it and ``variant`` tells its raw and
    recalibrated versions apart (they share ``built_at``).
    """
    return state_at(_state, now)


@st.cache_resource(max_entries=8)
def calibrated_state(_state: PipelineState, built_at: str, model_key: str, _report: ReliabilityReport) -> PipelineState:
    """The state with the validated recalibration applied (:func:`src.dashboard.state.apply_recalibration`), cached per fit and backtest."""
    return apply_recalibration(_state, _report.recalibration)


# =============================================================================
# Data download (FootyStats, runtime)
# =============================================================================
@dataclass
class RefreshGuard:
    """Server-wide record of the automatic refreshes: one attempt per data directory per :data:`~src.dashboard.data_refresh.DEFAULT_MAX_AGE`.

    Shared by every session (``st.cache_resource``), so a failing download is
    not retried on every rerun and two sessions never download at once.
    ``results`` keeps the last :class:`~src.dashboard.data_refresh.RefreshResult`
    of each data directory, so every session -- not only the one that ran the
    download -- can say why the data was not renewed.
    """

    lock: threading.Lock = field(default_factory=threading.Lock)
    attempts: dict[str, float] = field(default_factory=dict)
    results: dict[str, data_refresh.RefreshResult] = field(default_factory=dict)


@st.cache_resource
def refresh_guard() -> RefreshGuard:
    """The process-wide :class:`RefreshGuard`."""
    return RefreshGuard()


def run_refresh(key: str, *, data_dir: Path, quotes_dir: Path, force: bool) -> data_refresh.RefreshResult | None:
    """Download the FootyStats data now, with a spinner and a progress bar; ``None`` when skipped by the guard.

    Without ``force`` the download runs at most once per 6 hours per data
    directory (:class:`RefreshGuard`); the button *Aggiorna dati ora* and the
    setup page pass ``force=True``. The result is kept in
    ``st.session_state[LAST_REFRESH_STATE_KEY]`` for the *Dettagli pipeline*
    expander; when files were written the cached pipeline states are dropped
    (the new data has a new cache key anyway). The key is passed to the
    downloader only, never shown.
    """
    guard = refresh_guard()
    slot = str(Path(data_dir).resolve())
    if not guard.lock.acquire(blocking=False):
        st.info("Un aggiornamento dei dati FootyStats è già in corso: riprova tra poco.", icon="⏳")
        return None
    try:
        last = guard.attempts.get(slot)
        if not force and last is not None and time.time() - last < data_refresh.DEFAULT_MAX_AGE.total_seconds():
            return None
        guard.attempts[slot] = time.time()
        bar = st.progress(0.0, text="Aggiornamento dei dati FootyStats…")

        def progress(fraction: float, message: str) -> None:
            bar.progress(min(max(float(fraction), 0.0), 1.0), text=message)

        with st.spinner("Scarico i dati da FootyStats con la tua chiave…"):
            result = data_refresh.refresh(data_dir, quotes_dir, key=key, progress=progress)
        bar.empty()
        guard.results[slot] = result
    finally:
        guard.lock.release()
    st.session_state[LAST_REFRESH_STATE_KEY] = result
    if result.written:
        load_state.clear()
        calendar_state.clear()
    log.info("dashboard refresh", extra={"written": result.written, "errors": list(result.errors), "forced": force})
    return result


def render_refresh_outcome(result: data_refresh.RefreshResult | None) -> None:
    """``st.success`` / ``st.warning`` for a refresh that just ran: what was saved and which competitions were skipped."""
    if result is None:
        return
    if result.written:
        st.success(result.summary_text(), icon="✅")
    for line in result.warnings():
        st.warning(line, icon="⚠️")


def render_data_panel(container: st.delta_generator.DeltaGenerator, *, data_dir: Path, quotes_dir: Path) -> data_refresh.RefreshResult | None:
    """Sidebar data box: automatic refresh when the data is missing or old, the timestamp and *Aggiorna dati ora*.

    The automatic refresh runs only with a key and at most once per 6 hours
    (:func:`run_refresh`); its progress and outcome are shown in the main area.
    Returns the result of a refresh that ran in this script run (``None``
    otherwise).
    """
    location = data_refresh.locate_key()
    result: data_refresh.RefreshResult | None = None
    if location is not None and data_refresh.needs_refresh(data_dir):
        result = run_refresh(location.value, data_dir=data_dir, quotes_dir=quotes_dir, force=False)
    with container:
        label = data_refresh.refresh_label(data_dir)
        st.caption(label or "Dati FootyStats non ancora scaricati.")
        if location is not None:
            pressed = st.button(
                REFRESH_NOW_LABEL,
                key=REFRESH_BUTTON_KEY,
                help="Scarica di nuovo partite, calendario e quote da FootyStats con la tua chiave (i dati restano solo su questo computer o server).",
                **FULL_WIDTH["button"],
            )
        else:
            pressed = False
            st.caption("Chiave FootyStats non configurata: i dati non si aggiornano.")
    if pressed and location is not None:
        result = run_refresh(location.value, data_dir=data_dir, quotes_dir=quotes_dir, force=True)
    render_refresh_outcome(result)
    render_data_warnings(data_dir, key_available=location is not None, refreshed_now=result is not None)
    return result


def render_data_warnings(data_dir: Path, *, key_available: bool, refreshed_now: bool) -> None:
    """The warnings shown on **every** run while the local data is old or partly old (not only after a download).

    * data older than :data:`~src.dashboard.data_refresh.DEFAULT_MAX_AGE`:
      ``'Dati vecchi di 2 giorni (...): aggiornamento non riuscito / chiave
      mancante'`` (:func:`~src.dashboard.data_refresh.stale_warning`), with the
      error of the last attempt of any session (:class:`RefreshGuard`);
    * a competition that still carries the data of an earlier download (its
      download, or its current season, failed last time:
      :func:`~src.dashboard.data_refresh.kept_data_warnings`), unless a
      download ran in this script run (its own warnings already say so).
    """
    last = refresh_guard().results.get(str(Path(data_dir).resolve()))
    error = last.error if last is not None and not last.written else None
    stale = data_refresh.stale_warning(data_dir, key_available=key_available, last_error=error)
    if stale:
        st.warning(stale, icon="⏰")
    if not refreshed_now:
        for line in data_refresh.kept_data_warnings(data_dir):
            st.warning(line, icon="⚠️")


def render_setup_page(error: NoDataError, *, data_dir: Path, quotes_dir: Path) -> None:
    """The page shown without real data: what is missing, how to provide the key, the download button. No tabs, no numbers."""
    location = data_refresh.locate_key()
    st.title(PAGE_TITLE)
    st.error("Nessun dato reale disponibile: la dashboard non mostra numeri inventati.", icon="🛑")
    st.markdown("**Cosa manca**\n\n" + "\n".join(f"- {reason}" for reason in error.reasons))
    try:
        shown_dir = Path(data_dir).resolve().relative_to(ROOT).as_posix()
    except ValueError:
        shown_dir = str(data_dir)
    st.markdown(
        "**Come fornire la chiave FootyStats** (la chiave non va mai pubblicata né scritta nel codice)\n\n"
        "- **In locale**: copia `.env.example` in `.env` nella cartella del progetto e scrivi la riga "
        "`FOOTYSTATS_API_KEY=la_tua_chiave` (il file `.env` è escluso da git).\n"
        f"- **Su Streamlit Cloud**: *App settings → Secrets*, incolla `{SECRETS_LINE}` con la tua chiave e salva.\n"
        "- Su footystats.org aggiungi all'account le leghe delle competizioni (*Choose leagues*): una lega non scelta "
        "viene saltata e segnalata.\n\n"
        f"I dati scaricati restano solo in `{shown_dir}` e nella cartella delle quote, mai nel repository pubblico."
    )
    if location is None:
        st.info("Nessuna chiave FootyStats trovata (ambiente, file .env, secrets di Streamlit): configurala e ricarica la pagina.", icon="🔑")
    else:
        st.success(f"Chiave FootyStats trovata ({location.source}).", icon="🔑")
        if st.button(SETUP_DOWNLOAD_LABEL, key=SETUP_DOWNLOAD_KEY, type="primary", **FULL_WIDTH["button"]):
            result = run_refresh(location.value, data_dir=data_dir, quotes_dir=quotes_dir, force=True)
            render_refresh_outcome(result)
            if result is not None and result.written:
                st.rerun()
    st.caption(
        "Vuoi solo provare l'interfaccia? Avvia con SERIEC_DEMO=1: lega simulata (squadre vere, calendari e risultati "
        "inventati) con un avviso rosso su ogni pagina."
    )


def demo_banner(state: PipelineState) -> None:
    """The red :data:`~src.dashboard.state.DEMO_WARNING` banner, for the demo only (called at the top of the page and of every tab)."""
    if state.is_demo:
        st.error(DEMO_WARNING, icon="🚨")


def selected_girone(label: str) -> Girone | None:
    """Map the sidebar label (``'Girone A'``, ``'Serie B'``, or a bare code) to a :class:`Girone`; ``None`` for *Tutti*."""
    return None if label == "Tutti" else Girone.parse(label)


def selected_matchday(state: PipelineState, section: str | None = None) -> Matchday | None:
    """The matchday of ``section`` the three matchday-aware tabs agree on, read from (and written to) ``st.session_state``.

    The *Selezioni di giornata* tab owns the ``Giornata`` selectbox (key
    :data:`MATCHDAY_STATE_KEY` scoped to the section), but *Giornata* and
    *Opportunità di Valore* are rendered before it in the same script run, so
    the choice is read from the session state here, once, before any tab
    renders: a missing or stale key (the state was recomputed and that matchday
    no longer exists) is replaced by the next matchday's key, which also seeds
    the selectbox without passing it an ``index``. The matchdays are the
    section's own (:func:`src.dashboard.state.matchdays`). ``None`` when the
    section has no matchday.
    """
    key = section_key(MATCHDAY_STATE_KEY, section)
    days = matchdays(state, section)
    if not days:
        st.session_state.pop(key, None)
        return None
    chosen = find_matchday(state, st.session_state.get(key), section)
    if chosen is None:
        chosen = days[0]
        st.session_state[key] = chosen.key
    return chosen


# =============================================================================
# Sidebar
# =============================================================================
def render_sidebar() -> tuple[st.delta_generator.DeltaGenerator, float, float, float]:
    """Parameter controls; returns ``(section controls container, min_ev_fraction, kelly_fraction, bankroll_units)``.

    The *Sezione* radio and the *Competizione* selectbox need the pipeline state
    (which sections have teams), which in turn needs the EV / Kelly / bankroll
    parameters rendered here; so this function only reserves their place at the
    top of the sidebar (an empty container) and :func:`render_section_controls`
    fills it once the state is known. Streamlit renders containers where they
    were created, so the two selectors stay above the sliders.
    """
    with st.sidebar:
        st.title("⚽ Serie C Quant Lab")
        st.caption(f"Modello Dixon-Coles per {COMPETITIONS_TEXT} e ricerca di valore atteso (+EV) contro le quote dei bookmaker.")
        controls = st.container()
        ev_pct = st.slider("Soglia EV minima (%)", min_value=0.0, max_value=15.0, value=3.5, step=0.5, key="min_ev_pct", help="Una selezione è di valore quando P_modello × Quota − 1 supera questa soglia.")
        kelly_label = st.radio("Frazione di Kelly", list(KELLY_OPTIONS), index=0, horizontal=True, key="kelly", help="Quota del Kelly pieno da puntare.")
        bankroll = st.number_input("Bankroll (unità)", min_value=1.0, max_value=1_000_000.0, value=100.0, step=10.0, key="bankroll", help="100 unità = ogni unità vale l'1% del bankroll.")
        if st.button("Ricalcola", key="recompute", help="Svuota la cache e riesegue la pipeline.", **FULL_WIDTH["button"]):
            st.cache_resource.clear()
            st.rerun()
    return controls, float(ev_pct) / 100.0, KELLY_OPTIONS[kelly_label], float(bankroll)


def render_section_controls(container: st.delta_generator.DeltaGenerator, state: PipelineState) -> tuple[str, Girone | None]:
    """The *Sezione* radio and the *Competizione* selectbox; returns ``(section, girone)`` (``girone=None`` for *Tutti*).

    The radio (key :data:`SECTION_STATE_KEY`) offers
    :func:`src.dashboard.state.section_options`: *Serie B e Serie C* first and
    by default, *Serie D* when it is configured (also before its gironi are
    initialised, so the page can say how to initialise it). The selectbox (key
    :data:`GIRONE_STATE_KEY`) lists *Tutti* plus the section's competitions; a
    stored choice that is not an option any more (another section's
    competition) is dropped first, so the widget falls back to *Tutti* instead
    of raising.
    """
    options = list(section_options(state))
    with container:
        if st.session_state.get(SECTION_STATE_KEY) not in options:
            st.session_state.pop(SECTION_STATE_KEY, None)
        section = str(
            st.radio(
                "Sezione",
                options,
                index=0,
                key=SECTION_STATE_KEY,
                horizontal=True,
                help=(
                    "Una sezione per campionato (Serie C gironi A/B/C, Serie B, Serie D gironi A-I e i campionati abilitati in "
                    "config/competitions.toml): ogni tab, selettore, tabella, lista e export mostra solo la sezione scelta. "
                    "La stima Dixon-Coles è una per paese (Serie B, C e D insieme)."
                ),
            )
        )
        comp_options = list(competition_options(section))
        if st.session_state.get(GIRONE_STATE_KEY) not in comp_options:
            st.session_state.pop(GIRONE_STATE_KEY, None)
        girone_label = str(
            st.selectbox(
                "Competizione",
                comp_options,
                index=0,
                key=GIRONE_STATE_KEY,
                help="Filtra classifica, partite e opportunità per competizione della sezione (un girone di Serie C o di Serie D, oppure il campionato intero).",
            )
        )
    return section, selected_girone(girone_label)


def render_source_badge(container: st.delta_generator.DeltaGenerator, state: PipelineState, section: str | None = None) -> None:
    """Data-source badge: *Dati reali FootyStats* or *Demo* (red, with the warning), the section's perimeter, fit diagnostics, Serie D status."""
    params = state.params
    groups = ", ".join(g for g in state.fit_groups if g) or "unico"
    calibration = (
        "ricalibrate (" + ", ".join(m.value for m in state.recalibration.enabled_markets) + ")"
        if state.calibrated and state.recalibration is not None
        else "originali del modello"
    )
    body = (
        f"**{perimeter_text(state, section)}**  \n"
        f"{state.n_matches} partite nel fit (una stima per paese: {groups}) · "
        f"**{len(section_predictions(state, section))}** partite in programma · "
        f"{state.n_historical_teams} squadre storiche escluse  \n"
        f"Fit: {params.method} · LL {params.log_likelihood:.1f}  \n"
        f"Convergenza: {'sì' if params.converged else 'no'} · {params.fit_seconds:.2f} s  \n"
        f"Prior gaussiana: {'attiva' if state.prior_enabled else 'disattivata'} · xi {state.xi:g}  \n"
        f"Probabilità: {calibration}"
    )
    coverage = coverage_text(state, section)
    if coverage:
        body += f"  \n{coverage}"
    if serie_d_gironi():
        body += f"  \nSerie D: {state.serie_d_status}"
    if state.is_demo:
        container.error(f"**{DEMO_BADGE}** · {DEMO_WARNING}  \n{body}", icon="🧪")
    else:
        container.success(f"**Dati reali FootyStats** (scaricati con la tua chiave)  \n{body}", icon="✅")


def render_section_notice(state: PipelineState, section: str) -> bool:
    """One ``st.info`` when the section has nothing fitted; returns ``True`` when the tab should stop there.

    For Serie D the notice carries :attr:`~src.dashboard.state.PipelineState.serie_d_status`
    and the exact command that initialises the gironi and harvests their data
    (:func:`src.dashboard.state.serie_d_harvest_command` for the perimeter
    season, so that it fills the ``config/teams_<season>.json`` the notice
    names), or the launchers.
    """
    if section_ready(state, section):
        return False
    if section == SECTION_D:
        st.info(
            f"Serie D {state.serie_d_status}. Comando: `{serie_d_harvest_command(season=state.season_label)}` — oppure {LAUNCHERS_HINT} con la chiave "
            "FootyStats in .env, o «Aggiorna dati ora» nella sidebar. La raccolta allinea i nove gironi (18 squadre ciascuno, 34 giornate) in "
            f"config/teams_{state.season_label.replace('/', '_')}.json alle liste squadre dell'API e ne scarica partite e calendario; poi la sezione mostra "
            "classifica, giornate, risultati, valore e selezioni come quella di Serie B e Serie C.",
            icon="ℹ️",
        )
    else:
        st.info(
            f"Sezione {section}: nessuna squadra del perimetro {state.season_label} con partite nei dati. Se il campionato è "
            "appena stato abilitato in config/competitions.toml (enabled = true), sceglilo anche sull'account footystats.org "
            "(Choose leagues) e premi «Aggiorna dati ora»: la raccolta riempie le sue squadre nel dizionario e ne scarica partite, "
            "calendario e quote; poi la sezione mostra classifica, giornate, risultati, valore, selezioni e schedine.",
            icon="ℹ️",
        )
    return True


# =============================================================================
# Tabs
# =============================================================================
def render_ranking_tab(state: PipelineState, girone: Girone | None, section: str = DEFAULT_SECTION) -> None:
    """*Classifica Forza*: the strength ranking of the section (one competition with ``girone``) and its α/β charts."""
    demo_banner(state)
    st.subheader(f"Classifica di forza · {section}")
    if render_section_notice(state, section):
        return
    st.caption(
        "Parametri Dixon-Coles: α = forza offensiva, β = solidità difensiva (più alto = subisce meno), γ = vantaggio casa della competizione; indici = exp(parametro). "
        f"Solo le squadre del perimetro {state.season_label} ({perimeter_text(state, section).removeprefix('Perimetro: ').split(' · ')[0]}) nella competizione del dizionario corrente; "
        f"le squadre delle stagioni passate ({state.n_historical_teams}) entrano nella stima solo come avversarie. "
        "α e β sono centrati per competizione: confronta le squadre dentro la stessa competizione."
    )
    table = ranking_table(state, girone, section)
    if table.empty:
        st.info("Nessuna squadra stimata per la competizione selezionata.")
        return
    st.dataframe(
        table,
        hide_index=True,
        **FULL_WIDTH["dataframe"],
        column_config={
            "Squadra": st.column_config.TextColumn("Squadra"),
            "Competizione": st.column_config.TextColumn("Competizione", width="small"),
            "Forza Offensiva (α)": st.column_config.NumberColumn("Forza Offensiva (α)", format="%+.3f"),
            "Solidità Difensiva (β)": st.column_config.NumberColumn("Solidità Difensiva (β)", format="%+.3f"),
            "Vantaggio Casa (γ)": st.column_config.NumberColumn("Vantaggio Casa (γ)", format="%+.3f"),
            "Indice Attacco": st.column_config.NumberColumn("Indice Attacco", format="%.3f"),
            "Indice Difesa": st.column_config.NumberColumn("Indice Difesa", format="%.3f"),
        },
    )
    if girone is None:
        gironi = [g for g in state.gironi if in_section(g, section)]
        per_row = len(gironi) if len(gironi) <= 4 else 3  # nine Serie D gironi: three charts per row
        for start in range(0, len(gironi), max(per_row, 1)):
            row = gironi[start : start + per_row]
            columns = st.columns(len(row) or 1)
            for col, g in zip(columns, row):
                with col:
                    st.plotly_chart(strength_figure(ranking_table(state, g), title=g.short), **FULL_WIDTH["plotly_chart"], key=f"strength_{g.value}")
    else:
        st.plotly_chart(strength_figure(table, title=f"{girone.short}: α e β per squadra"), **FULL_WIDTH["plotly_chart"], key=f"strength_{girone.value}")


def heatmap_goals_control(full_goals: int) -> int:
    """Slider choosing how many goals per team the heatmap draws; returns the chosen maximum.

    The range is ``[HEATMAP_MIN_GOALS, full_goals]`` (``full_goals`` = the
    engine's ``max_goals``, i.e. the whole ``prediction.matrix``) and the default
    is ``full_goals``, so the complete matrix is shown unless the user asks for
    a compact view; :func:`heatmap_figure` then folds the tail into an ``N+``
    row/column so the cells keep summing to 100%. A matrix too small for a
    range yields ``full_goals`` without rendering a widget.
    """
    low = min(HEATMAP_MIN_GOALS, full_goals)
    if low >= full_goals:
        return full_goals
    return int(
        st.slider(
            "Gol massimi nella matrice",
            min_value=low,
            max_value=full_goals,
            value=full_goals,
            step=1,
            key="heatmap_max_goals",
            help=(
                f"{full_goals} = matrice completa del modello. Un valore più basso mostra una vista compatta: "
                "l'ultima riga/colonna (N+) accorpa la probabilità di N o più gol, "
                "così le celle sommano sempre al 100%."
            ),
        )
    )


def matchday_column_config() -> dict[str, object]:
    """``column_config`` of the matchday table: number formats, tooltips, the key columns pinned on the left."""
    cc = st.column_config
    form_n, xg_n = DEFAULT_FORM_N, DEFAULT_XG_N
    config: dict[str, object] = {
        "Competizione": cc.TextColumn("Competizione", width="small", pinned=True),
        "Data": cc.DateColumn("Data", format="DD/MM/YYYY", width="small"),
        "Calcio d'inizio": cc.TextColumn(
            "Calcio d'inizio", width="small", help="Giorno e ora del calcio d'inizio (ora italiana); solo il giorno quando la fonte non dà l'orario."
        ),
        "Giornata": cc.NumberColumn("Giornata", format="%d", width="small", help="Giornata ufficiale della competizione (game_week di FootyStats)."),
        "Casa": cc.TextColumn("Casa", pinned=True),
        "Ospiti": cc.TextColumn("Ospiti", pinned=True),
        "Stato": cc.TextColumn(
            "Stato",
            width="small",
            help=(
                "giocata con il risultato finale (niente previsione) · da giocare (previsione del modello) · in corso / risultato "
                "in attesa (calcio d'inizio passato, risultato non ancora nei dati: niente previsione né selezioni) · rinviata "
                "(nessun risultato 48 ore dopo il calcio d'inizio, recupero da fissare)."
            ),
        ),
        "Tot. att.": cc.NumberColumn("Tot. att.", format="%.2f", help="λ + μ: gol totali attesi della partita."),
        "Ris. prob.": cc.TextColumn("Ris. prob.", help="Risultato esatto più probabile e sua probabilità (la cella più alta della matrice)."),
        "EV max (%)": cc.NumberColumn("EV max (%)", format="%+.2f %%", help="Miglior valore atteso fra tutte le selezioni quotate della partita; vuoto senza quote."),
        "Miglior EV": cc.TextColumn("Miglior EV", help="La selezione con l'EV più alto e la sua quota, es. X @3.40."),
        "Flag": cc.TextColumn(
            "Flag",
            help="Unione dei flag delle selezioni quotate (VALUE, LONGSHOT, SUSPICIOUS_EDGE) più REGRESSION_TRAP / TRAP:<squadra> quando una squadra è in trappola di regressione, anche senza quote.",
        ),
    }
    for side in ("Casa", "Ospiti"):
        config[f"α {side}"] = cc.NumberColumn(f"α {side}", format="%+.2f", help="Forza offensiva (0 = media della competizione).")
        config[f"β {side}"] = cc.NumberColumn(f"β {side}", format="%+.2f", help="Solidità difensiva (più alto = subisce meno).")
        config[f"Forma {side}"] = cc.TextColumn(
            f"Forma {side}", help=f"Ultime {form_n} gare della stagione, dalla più recente: V vittoria, N pareggio (nullo), P sconfitta."
        )
        config[f"Pt {side} ({form_n})"] = cc.NumberColumn(f"Pt {side} ({form_n})", format="%d", help=f"Punti nelle ultime {form_n} gare (massimo {3 * form_n}).")
        config[f"GF/GS {side} ({form_n})"] = cc.TextColumn(f"GF/GS {side} ({form_n})", help=f"Gol fatti / subiti nelle ultime {form_n} gare.")
        config[f"xG F/S {side} ({xg_n})"] = cc.TextColumn(
            f"xG F/S {side} ({xg_n})", help=f"xG prodotti / concessi nelle ultime {xg_n} gare ('-' quando un dato manca)."
        )
    config["λ"] = cc.NumberColumn("λ", format="%.2f", help="Gol attesi della squadra di casa.")
    config["μ"] = cc.NumberColumn("μ", format="%.2f", help="Gol attesi della squadra ospite.")
    for column in ("P1 (%)", "PX (%)", "P2 (%)", "O1.5 (%)", "O2.5 (%)", "O3.5 (%)", "GG (%)"):
        config[column] = cc.NumberColumn(column, format="%.1f %%")
    for column, _market, _selection in RAW_COMPARE_COLUMNS:
        raw = raw_column_name(column)
        config[raw] = cc.NumberColumn(raw, format="%.1f %%", help="Probabilità originale del modello, prima della ricalibrazione validata fuori campione.")
    for column in ("Fair 1", "Fair X", "Fair 2", "Fair O2.5", "Fair GG"):
        config[column] = cc.NumberColumn(column, format="%.2f", help="Quota fair = 1 / probabilità del modello.")
    for column in ("Book 1", "Book X", "Book 2", "Book O2.5", "Book GG"):
        config[column] = cc.NumberColumn(column, format="%.2f", help="Miglior quota fra i bookmaker nei dati; vuota se nessuno quota l'esito.")
    return config


def results_column_config() -> dict[str, object]:
    """``column_config`` of the results table."""
    cc = st.column_config
    return {
        "Competizione": cc.TextColumn("Competizione", width="small", pinned=True),
        "Data": cc.DateColumn("Data", format="DD/MM/YYYY", width="small"),
        "Giornata": cc.NumberColumn("Giornata", format="%d", width="small"),
        "Casa": cc.TextColumn("Casa", pinned=True),
        "Ospiti": cc.TextColumn("Ospiti", pinned=True),
        "Risultato": cc.TextColumn("Risultato", width="small", help="Gol casa-gol ospiti."),
        "Esito": cc.TextColumn("Esito", width="small", help="1 vittoria casa, X pareggio, 2 vittoria ospiti."),
        "Tiri C/O": cc.TextColumn("Tiri C/O", help="Tiri totali casa / ospiti ('-' quando il dato manca)."),
        "In porta C/O": cc.TextColumn("In porta C/O", help="Tiri in porta casa / ospiti."),
        "Angoli C/O": cc.TextColumn("Angoli C/O", help="Calci d'angolo casa / ospiti."),
        "xG Casa": cc.NumberColumn("xG Casa", format="%.2f"),
        "xG Ospiti": cc.NumberColumn("xG Ospiti", format="%.2f"),
        "λ": cc.NumberColumn("λ", format="%.2f", help="Gol attesi casa secondo il modello stimato a fine campione."),
        "μ": cc.NumberColumn("μ", format="%.2f", help="Gol attesi ospiti secondo il modello stimato a fine campione."),
        "P esito (%)": cc.NumberColumn("P esito (%)", format="%.1f %%", help="Probabilità che il modello assegna all'esito 1X2 avvenuto (lettura a posteriori)."),
        "Sorpresa": cc.TextColumn("Sorpresa", width="small", help="sorpresa = P esito sotto il 20%; atteso = P esito sopra il 50%."),
    }


def round_selector(state: PipelineState, girone: Girone, matchday: Matchday | None, section: str) -> int | None:
    """The *Giornata n.* selectbox of one competition (every official round with matches); returns the chosen round.

    The default is the competition's round in the selected matchday (its
    current round when it is not in it, the last round otherwise); the choice
    follows the matchday until the user picks another round (the key is
    scoped to the competition and the section). ``None`` when the competition
    has no match at all.
    """
    rounds = competition_rounds(state, girone)
    if not rounds:
        return None
    current = current_round_for(state, girone)
    current_no = None if current is None else current.round
    default = matchday.rounds.get(girone) if matchday is not None else None
    if default is None:
        default = current_no if current_no is not None else rounds[-1]
    if default not in rounds:
        default = rounds[-1]
    key = section_key(f"{ROUND_STATE_KEY}_{girone.value}", section)
    seed_key = f"{key}_default"
    if st.session_state.get(key) not in rounds or st.session_state.get(key) == st.session_state.get(seed_key):
        st.session_state[key] = default
    st.session_state[seed_key] = default
    return int(
        st.selectbox(
            "Giornata n.",
            rounds,
            key=key,
            format_func=lambda r: f"Giornata {r}" + (" (corrente)" if r == current_no else ""),
            help=(
                f"Ogni giornata ufficiale di {girone.short} con almeno una partita: la tabella mostra tutte le sue partite, "
                "le giocate con il risultato, quelle iniziate senza risultato nei dati (in corso / risultato in attesa), "
                "le da giocare con la previsione, le rinviate."
            ),
        )
    )


def render_recoveries(state: PipelineState, girone: Girone | None, section: str) -> None:
    """The *Recuperi e rinvii* expander: postponed matches and recoveries outside the current rounds, with their original date."""
    table = recoveries_table_for(state, section, girone)
    with st.expander(f"{RECOVERIES_LABEL} ({len(table)})", expanded=False):
        if table.empty:
            st.caption("Nessuna partita rinviata o da recuperare fuori dalle giornate correnti.")
            return
        st.caption(
            "Partite rinviate (data originale passata da oltre 48 ore senza risultato, recupero non ancora fissato) e recuperi "
            "ancora da giocare fissati fuori dalla giornata corrente della loro competizione: non spostano la giornata e non "
            "entrano nelle selezioni di giornata. Data originale = il giorno in cui si giocava il resto del turno, solo per una "
            "partita spostata davvero. Una partita già iniziata senza risultato resta nel suo turno (in corso / risultato in attesa)."
        )
        st.dataframe(
            table,
            hide_index=True,
            column_order=[c for c in RECOVERIES_COLUMNS if c != "match_id"],
            column_config={
                "Giornata": st.column_config.NumberColumn("Giornata", format="%d", width="small"),
                "Data originale": st.column_config.TextColumn("Data originale", help="Il giorno in cui la partita era in calendario."),
                "Nuova data": st.column_config.TextColumn("Nuova data", help="Calcio d'inizio del recupero (ora italiana); da definire per un rinvio."),
                **{c: st.column_config.NumberColumn(c, format="%.1f %%") for c in ("P1 (%)", "PX (%)", "P2 (%)", "GG (%)")},
            },
            key=section_key(f"recoveries_{'all' if girone is None else girone.value}", section),
            **FULL_WIDTH["dataframe"],
        )


def render_match_tab(state: PipelineState, girone: Girone | None, matchday: Matchday | None, section: str = DEFAULT_SECTION) -> None:
    """*Giornata*: every match of the selected rounds in one table, the recoveries, then the detail of the selected fixture.

    Without a sidebar competition the table is the selected matchday (the
    current round of every competition of the section by default); with one
    competition the *Giornata n.* selector (:func:`round_selector`) picks any of
    its official rounds. The table (:func:`matchday_table_for`) lists every
    match: played (``Stato`` *giocata 2-1*, no prediction), kicked off without a
    result in the data (*in corso / risultato in attesa*, no prediction), to play
    (kick-off and the model's numbers) and postponed. Clicking the row of a match to play
    (``selection_mode='single-row'``, ``on_select='rerun'``) selects it: the
    *Partita* selectbox (key :data:`FIXTURE_STATE_KEY`) follows the click and
    stays available as the fallback; a choice made in the selectbox afterwards
    is kept until the next click (:data:`APPLIED_ROW_STATE_KEY`); both keys are
    scoped to the section. The widget key is scoped to the matchday (or round),
    competition and section (:func:`matchday_table_key`), so a selection never
    outlives its table. Below: the metrics row, the score-matrix heatmap and
    the markets table of the selected fixture.
    """
    demo_banner(state)
    st.subheader(f"Giornata: tutte le partite del turno · {section}")
    if render_section_notice(state, section):
        return
    fixture_key = section_key(FIXTURE_STATE_KEY, section)
    applied_key = section_key(APPLIED_ROW_STATE_KEY, section)
    round_number: int | None = None
    compare = state.calibrated
    if girone is not None:
        selector, _ = st.columns([1, 3])
        with selector:
            round_number = round_selector(state, girone, matchday, section)
        if round_number is not None:
            table = matchday_table_for(state, None, girone, section, round_number=round_number, compare_raw=compare)
        else:
            table = matchday_table_for(state, None, girone, section, compare_raw=compare)
        scope_label = f"{girone.short} · Giornata {round_number}" if round_number is not None else girone.short
    else:
        table = matchday_table_for(state, matchday, None, section, compare_raw=compare)
        scope_label = matchday.label if matchday is not None else f"tutte le competizioni di {section}"
    if table.empty:
        message = f"Nessuna partita in calendario per {scope_label}."
        if state.skipped_fixtures:
            message += (
                f" {len(state.skipped_fixtures)} partite caricate sono state saltate perché di un'altra stagione o con squadre "
                f"fuori dal perimetro {state.season_label} (dettagli in *Dettagli pipeline*)."
            )
        st.info(message)
        render_recoveries(state, girone, section)
        return
    status = table["Stato"].astype(str)
    n_played = int(status.str.startswith("giocata").sum())
    n_postponed = int((status == "rinviata").sum())
    n_in_play = int((status == IN_PLAY_TEXT).sum())
    to_play_ids = set(table.loc[(status == STATUS_TO_PLAY).to_numpy(dtype=bool), "match_id"].astype(str))
    preds = sorted(
        (p for p in section_predictions(state, section) if p.fixture.match_id in to_play_ids),
        key=lambda p: (p.fixture.date, p.fixture.kickoff or datetime.min.replace(tzinfo=UTC), p.fixture.label),
    )
    n_matchdays = len(matchdays(state, section))
    st.caption(
        f"{scope_label} · {len(table)} partite: {n_played} giocate, {len(to_play_ids)} da giocare"
        + (f", {n_in_play} in corso o in attesa del risultato (senza previsione)" if n_in_play else "")
        + (f", {n_postponed} rinviate" if n_postponed else "")
        + f" · orari di calcio d'inizio in ora italiana · per le partite da giocare forza (α/β), forma delle ultime {DEFAULT_FORM_N} gare, "
        f"xG delle ultime {DEFAULT_XG_N}" + (" ('-' quando il dato manca: in Serie D gli xG sono spesso assenti)" if section == SECTION_D else "")
        + ", gol attesi, probabilità, quote fair, miglior quota book, EV massimo e flag; le giocate mostrano solo il risultato. "
        "Clicca una partita da giocare per aprirla qui sotto"
        + (
            f"; la giornata si cambia nel tab *Selezioni di giornata* ({n_matchdays} in programma)."
            if girone is None and matchday is not None
            else "; scegli un'altra giornata dal menù «Giornata n.»." if girone is not None else "."
        )
    )
    if compare and state.recalibration is not None:
        st.caption(
            "Probabilità ricalibrate attive (tab *Affidabilità modello*): accanto a P1, PX, P2, O2.5 e GG la colonna «orig.» "
            "riporta la probabilità originale del modello; quote fair, EV, liste e schedine usano quelle ricalibrate. "
            + state.recalibration.describe()
        )
    table_key = matchday_table_key(matchday if girone is None else None, girone, section, round_number=round_number)
    event = st.dataframe(
        table,
        hide_index=True,
        height=TABLE_HEIGHT,
        key=table_key,
        on_select="rerun",
        selection_mode="single-row",
        column_order=[c for c in table.columns if c != "match_id"],
        column_config=matchday_column_config(),
        **FULL_WIDTH["dataframe"],
    )
    if round_number is not None and girone is not None:
        stem = f"g{round_number}_{girone.value}"
    else:
        stem = ("tutte" if matchday is None else matchday.key.replace("#", "_")) + ("" if girone is None else "_" + girone.value)
    st.download_button(
        "Scarica tabella giornata (CSV)",
        data=export_csv_bytes(table, demo=state.is_demo),
        file_name=demo_file_name(f"giornata_{section_slug(section)}_{stem}.csv", state.is_demo),
        mime="text/csv",
        key="dl_matchday",
        help="Tutte le colonne della tabella più match_id.",
        **FULL_WIDTH["download_button"],
    )
    render_recoveries(state, girone, section)
    if not preds:
        st.info(
            "Nessuna partita da giocare con una previsione in questa giornata: le partite sono giocate, già iniziate "
            "(in corso / risultato in attesa) o rinviate.",
            icon="ℹ️",
        )
        return
    labels = [fixture_label(p) for p in preds]
    label_by_id = {p.fixture.match_id: label for p, label in zip(preds, labels)}
    clicked = selected_row_match_id(event, table)
    applied = (table_key, clicked)  # the scope travels with the applied row: a selection of another table is never reused
    if clicked is None:
        st.session_state.pop(applied_key, None)
    elif clicked in label_by_id and st.session_state.get(applied_key) != applied:
        # a new click on a match to play: the selectbox follows the table (a later manual choice is then respected)
        st.session_state[applied_key] = applied
        st.session_state[fixture_key] = label_by_id[clicked]
    if st.session_state.get(fixture_key) not in labels:
        st.session_state.pop(fixture_key, None)  # stale choice (another competition or matchday): back to the first fixture
    st.markdown("**Dettaglio partita**")
    choice = st.selectbox(
        "Partita",
        labels,
        key=fixture_key,
        help="La partita aperta qui sotto: segue la riga cliccata nella tabella, oppure si sceglie da questo menù.",
    )
    pred: MatchPrediction = preds[labels.index(choice)]
    hs, as_, p_top = pred.most_likely_score()
    p_over = pred.probabilities[Market.OVER_UNDER_25]["Over 2.5"]
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("λ casa (gol attesi)", f"{pred.home_rate:.2f}", help=f"Gol attesi di {pred.fixture.home_team}")
    c2.metric("μ ospiti (gol attesi)", f"{pred.away_rate:.2f}", help=f"Gol attesi di {pred.fixture.away_team}")
    c3.metric("Risultato più probabile", f"{hs}-{as_}", delta=f"{p_top:.1%}", delta_color="off")
    c4.metric("P(Over 2.5)", f"{p_over:.1%}")
    full_goals = pred.matrix.shape[0] - 1
    control, _ = st.columns([1, 2])
    with control:
        shown = heatmap_goals_control(full_goals)
    st.plotly_chart(heatmap_figure(pred, max_goals_shown=shown), **FULL_WIDTH["plotly_chart"], key="heatmap")
    if shown < full_goals:
        st.caption(
            f"Matrice compatta 0-{shown} gol: l'ultima riga e l'ultima colonna ({shown}+) accorpano i punteggi con "
            f"{shown} o più gol, così le celle sommano sempre al 100%."
        )
    else:
        st.caption(f"Matrice completa 0-{full_goals} gol per squadra: le celle sommano al 100%.")
    st.markdown("**Mercati: probabilità del modello, quota fair e quota del bookmaker**")
    quotes = quotes_for(state, pred.match_id)
    st.dataframe(
        markets_table(pred, quotes),
        hide_index=True,
        **FULL_WIDTH["dataframe"],
        column_config={
            "Probabilità (%)": st.column_config.NumberColumn("Probabilità (%)", format="%.1f %%"),
            "Quota Fair": st.column_config.NumberColumn("Quota Fair", format="%.2f"),
            "Quota Book": st.column_config.NumberColumn("Quota Book", format="%.2f"),
            "EV (%)": st.column_config.NumberColumn("EV (%)", format="%+.2f %%"),
        },
    )
    if not quotes:
        st.caption("Nessuna quota del bookmaker per questa partita: le colonne Quota Book ed EV restano vuote.")


def render_results_tab(state: PipelineState, girone: Girone | None, section: str = DEFAULT_SECTION) -> None:
    """*Risultati*: the played matches of one season, section, competition (sidebar) and team, with the model's reading of each.

    The season selectbox (key :data:`RESULTS_SEASON_KEY`, scoped to the section)
    offers the registry season first and the history seasons after it; the team
    selectbox (key :data:`RESULTS_TEAM_KEY`, scoped likewise) offers *Tutte*
    plus the teams of the sidebar competition in that season
    (:func:`results_team_options`). The table is :func:`results_table_for`,
    sortable by clicking a header, with a CSV download.
    """
    demo_banner(state)
    st.subheader(f"Risultati: partite giocate · {section}")
    if render_section_notice(state, section):
        return
    season_key = section_key(RESULTS_SEASON_KEY, section)
    team_key = section_key(RESULTS_TEAM_KEY, section)
    seasons = seasons_available(state, section)
    if not seasons or state.matches.empty:
        st.info("Nessuna partita giocata nei dati.")
        return
    c_team, c_season = st.columns([3, 1])
    with c_season:
        if st.session_state.get(season_key) not in seasons:
            st.session_state.pop(season_key, None)
        season = str(
            st.selectbox(
                "Stagione",
                seasons,
                key=season_key,
                help="La stagione corrente per prima; le precedenti sono lo storico su cui il modello è stimato.",
            )
        )
    team_options = [ALL_TEAMS_LABEL, *results_team_options(state, girone, season, section)]
    with c_team:
        if st.session_state.get(team_key) not in team_options:
            st.session_state.pop(team_key, None)
        team_label = str(
            st.selectbox(
                "Squadra",
                team_options,
                key=team_key,
                help=(
                    "Le squadre che hanno giocato la stagione scelta nella competizione della sidebar (Tutte = ogni partita): "
                    "nelle stagioni passate anche quelle non più nel perimetro, e non le attuali che allora giocavano altrove."
                ),
            )
        )
    team = None if team_label == ALL_TEAMS_LABEL else team_label
    table = results_table_for(state, girone=girone, team=team, season=season, section=section)
    scope = ", ".join(part for part in (season, f"tutte le competizioni di {section}" if girone is None else girone.short, team or "") if part)
    if table.empty:
        st.info(f"Nessuna partita giocata per {scope}.")
        return
    st.caption(f"{len(table)} partite giocate · {scope} · dalla più recente; clicca l'intestazione di una colonna per ordinare.")
    st.dataframe(
        table,
        hide_index=True,
        height=TABLE_HEIGHT,
        key=RESULTS_TABLE_KEY,
        column_order=[c for c in table.columns if c != "match_id"],
        column_config=results_column_config(),
        **FULL_WIDTH["dataframe"],
    )
    st.caption(
        "P esito (%) = probabilità che il modello assegna all'esito 1X2 realmente avvenuto, con λ e μ di quella partita "
        "(la stessa matrice dei punteggi del tab Giornata). Il modello è stimato su tutte le partite giocate, compresa questa: "
        "è una lettura a posteriori, non una previsione. Sorpresa = «sorpresa» sotto il 20%, «atteso» sopra il 50%, vuoto nel mezzo."
    )
    st.download_button(
        "Scarica risultati (CSV)",
        data=export_csv_bytes(table, demo=state.is_demo),
        file_name=demo_file_name(
            f"risultati_{section_slug(section)}_{season.replace('/', '_')}{'' if girone is None else '_' + girone.value}{'' if team is None else '_' + team.replace(' ', '_')}.csv",
            state.is_demo,
        ),
        mime="text/csv",
        key="dl_results",
        help="Tutte le colonne della tabella più match_id.",
        **FULL_WIDTH["download_button"],
    )


def render_value_tab(state: PipelineState, girone: Girone | None, min_ev: float, matchday: Matchday | None, section: str = DEFAULT_SECTION) -> None:
    """*Opportunità di Valore*: the value selections of the section's matchday and its REGRESSION_TRAP alerts."""
    demo_banner(state)
    st.subheader(f"Opportunità di valore · {section}")
    if render_section_notice(state, section):
        return
    if not state.quotes:
        st.info(no_odds_message(state))
    elif not section_opportunities(state, section):
        st.info(
            f"Nessuna quota bookmaker per le partite di {section}: il pannello EV resta vuoto per questa sezione"
            + (" (su FootyStats la copertura delle quote di Serie D è parziale: molte partite non sono quotate)." if section == SECTION_D else ".")
        )
    show_all = False
    if matchday is not None:
        show_all = bool(
            st.checkbox(
                "Mostra tutte le giornate",
                value=False,
                key=section_key(VALUE_ALL_MATCHDAYS_KEY, section),
                help="Di default il pannello mostra solo la giornata selezionata nel tab Selezioni di giornata.",
            )
        )
        scope = "Tutte le partite in programma" if show_all else matchday.label
        st.caption(f"{scope} · la giornata si cambia nel tab *Selezioni di giornata* ({len(matchdays(state, section))} in programma).")
    table = value_table(state, min_ev, girone, matchday=ALL_MATCHDAYS if show_all else matchday, section=section)
    if table.empty:
        if section_opportunities(state, section):
            st.info(f"Nessuna selezione con EV ≥ {min_ev:.1%} per il filtro corrente.")
    else:
        ev_max = float(max(table["EV (%)"].max(), min_ev * 100.0, 1.0))
        st.dataframe(
            table,
            hide_index=True,
            **FULL_WIDTH["dataframe"],
            column_config={
                "Data": st.column_config.DateColumn("Data", format="DD/MM/YYYY"),
                "Competizione": st.column_config.TextColumn("Competizione", width="small"),
                "Bookmaker": st.column_config.TextColumn("Bookmaker", help="Fonte della quota (es. FootyStats, SimBook o il book del CSV in data/odds)."),
                "Quota Book": st.column_config.NumberColumn("Quota Book", format="%.2f"),
                "Quota Fair": st.column_config.NumberColumn("Quota Fair", format="%.2f"),
                "P. Modello (%)": st.column_config.NumberColumn("P. Modello (%)", format="%.1f %%"),
                "EV (%)": st.column_config.ProgressColumn("EV (%)", format="%.2f %%", min_value=0.0, max_value=ev_max),
                "Stake (unità)": st.column_config.NumberColumn("Stake (unità)", format="%.1f"),
                "Flag": st.column_config.TextColumn("Flag", help="VALUE = sopra soglia; REGRESSION_TRAP = Over/GG contro una difesa in sovra-rendimento; LONGSHOT = quota > 6; SUSPICIOUS_EDGE = EV > 25% (quote probabilmente stantie)."),
            },
        )
        st.caption(f"{len(table)} selezioni · stake complessivo {table['Stake (unità)'].sum():.1f} unità")
    st.caption("EV = P_modello × Quota − 1; stake = Kelly frazionario")

    st.markdown("---")
    st.subheader(f"Alert REGRESSION_TRAP · {section}")
    st.caption(
        "Squadre con pochi gol subiti nelle ultime 3 gare ma xGA elevato: la difesa è in sovra-rendimento e il mercato sottoprezza i gol contro di loro."
        + (" In Serie D gli xG mancano spesso: senza xG il filtro non può scattare e l'alert resta vuoto." if section == SECTION_D else "")
    )
    traps = section_traps(state, section)
    if traps:
        for trap in traps:
            st.warning(trap.description(), icon="⚠️")
        st.dataframe(
            traps_table(state, section),
            hide_index=True,
            **FULL_WIDTH["dataframe"],
            column_config={
                "Competizione": st.column_config.TextColumn("Competizione", width="small"),
                "xGA": st.column_config.NumberColumn("xGA", format="%.2f"),
                "Severità": st.column_config.NumberColumn("Severità", format="%+.2f"),
                "Ultima Gara": st.column_config.DateColumn("Ultima Gara", format="DD/MM/YYYY"),
            },
        )
        flagged = sum(1 for o in section_opportunities(state, section) if o.is_trap)
        st.caption(f"Selezioni Over/GG segnalate dal filtro: {flagged}")
    else:
        st.success("Nessuna squadra in trappola di regressione nella finestra corrente.", icon="✅")


def render_picks_tab(state: PipelineState, matchday: Matchday | None, section: str = DEFAULT_SECTION) -> None:
    """*Selezioni di giornata*: the section's matchday selector and its top-N draw / both-teams-to-score lists."""
    demo_banner(state)
    st.subheader(f"Selezioni di giornata · {section}")
    if render_section_notice(state, section):
        return
    days = matchdays(state, section)
    if matchday is None or not days:
        message = "Nessuna partita in programma: nessuna giornata da selezionare."
        if state.skipped_fixtures:
            message += f" {len(state.skipped_fixtures)} partite caricate sono state saltate (dettagli in *Dettagli pipeline*)."
        st.info(message)
        return
    keys = [m.key for m in days]
    labels = {m.key: m.label for m in days}
    c_day, c_draws, c_btts = st.columns([3, 1, 1])
    with c_day:
        chosen_key = st.selectbox(
            "Giornata",
            keys,
            format_func=lambda k: labels.get(k, k),
            key=section_key(MATCHDAY_STATE_KEY, section),
            help=(
                f"La prima è la giornata ufficiale corrente di ogni competizione di {section} (con le partite già giocate); le "
                "successive raggruppano i turni seguenti giocati nello stesso blocco della settimana (fine settimana ven-lun o "
                "infrasettimanale mar-gio). Rinvii e recuperi fuori turno non spostano le giornate. Vale anche per Giornata e "
                "Opportunità di Valore."
            ),
        )
    with c_draws:
        n_draws = int(st.number_input("N risultati X", min_value=1, max_value=MAX_PICKS, value=DEFAULT_N_DRAWS, step=1, key="n_draws"))
    with c_btts:
        n_btts = int(st.number_input("N partite GG", min_value=1, max_value=MAX_PICKS, value=DEFAULT_N_BTTS, step=1, key="n_btts"))
    chosen = find_matchday(state, str(chosen_key), section) or matchday
    st.caption(
        f"{chosen.label} · {chosen.n_fixtures} partite da giocare"
        + (f" ({chosen.n_played} già giocate, escluse dalle liste)" if chosen.n_played else "")
        + f" · {chosen.start.strftime('%d/%m/%Y')}"
        + (f" - {chosen.end.strftime('%d/%m/%Y')}" if chosen.end != chosen.start else "")
        + f". Le liste considerano tutte le competizioni di {section}, qualunque sia il filtro Competizione della sidebar."
    )
    column_config = {
        "#": st.column_config.NumberColumn("#", width="small", format="%d"),
        "Competizione": st.column_config.TextColumn("Competizione", width="small"),
        "Data": st.column_config.DateColumn("Data", format="DD/MM/YYYY"),
        "P. Modello (%)": st.column_config.NumberColumn("P. Modello (%)", format="%.1f %%"),
        "Quota Fair": st.column_config.NumberColumn("Quota Fair", format="%.2f"),
        "Quota Book": st.column_config.NumberColumn("Quota Book", format="%.2f"),
        "EV (%)": st.column_config.NumberColumn("EV (%)", format="%+.2f %%"),
        "Flag": st.column_config.TextColumn("Flag", help="VALUE = EV sopra soglia; REGRESSION_TRAP = GG contro una difesa in sovra-rendimento; LONGSHOT = quota > 6; SUSPICIOUS_EDGE = EV > 25%."),
        "Nota": st.column_config.TextColumn("Nota", help="Indizio del modello: gol attesi vicini / pochi gol attesi per la X, xG attese alte / entrambi gli attacchi sopra 1.2 per il GG."),
    }
    draws = draw_picks_table(state, chosen, n=n_draws, section=section)
    btts = btts_picks_table(state, chosen, n=n_btts, section=section)

    st.markdown(f"**Le {len(draws)} partite più probabili da X (tutte le competizioni di {section})**")
    if draws.empty:
        st.info("Nessuna partita prevista nella giornata selezionata.")
    else:
        st.dataframe(draws, hide_index=True, **FULL_WIDTH["dataframe"], column_config=column_config, key="picks_draws")
    st.caption(
        "Ordinate solo per probabilità di pareggio del modello, non per valore: Quota Fair = 1/P; Quota Book ed EV "
        "dicono se il prezzo del bookmaker vale la puntata (EV > 0 = quota sopra la fair). Vuote quando nessun book quota la X."
    )

    st.markdown(f"**Le {len(btts)} partite più probabili da GG (tutte le competizioni di {section})**")
    if btts.empty:
        st.info("Nessuna partita prevista nella giornata selezionata.")
    else:
        st.dataframe(btts, hide_index=True, **FULL_WIDTH["dataframe"], column_config=column_config, key="picks_btts")
    st.caption(
        "Ordinate solo per probabilità che entrambe segnino, non per valore: un GG molto probabile pagato sotto la "
        "quota fair perde nel lungo periodo. Incrocia sempre con Quota Book ed EV; REGRESSION_TRAP segnala il GG "
        "contro una difesa in sovra-rendimento."
    )
    st.download_button(
        "Scarica le due liste (CSV)",
        data=picks_csv_bytes(draws, btts, chosen, demo=state.is_demo),
        file_name=demo_file_name(f"selezioni_{section_slug(section)}_{chosen.key.replace('#', '_')}.csv", state.is_demo),
        mime="text/csv",
        key="dl_picks",
        help="Un solo CSV: colonna Lista = X oppure GG, poi le colonne delle tabelle.",
        **FULL_WIDTH["download_button"],
    )


def value_scope_text(girone: Girone | None, matchday: Matchday | None, show_all: bool, section: str | None = None) -> str:
    """``'Giornata del 7-9 nov 2026 · ..., Girone A'``: the value panel's current scope, for captions (the section when no competition is chosen)."""
    days = "tutte le partite in programma" if show_all or matchday is None else matchday.label
    if girone is not None:
        competitions = girone.short
    else:
        competitions = "tutte le competizioni" if section is None else f"tutte le competizioni di {section}"
    return f"{days}, {competitions}"


def render_legend_tab(
    state: PipelineState,
    girone: Girone | None,
    min_ev: float,
    matchday: Matchday | None,
    kelly_fraction: float,
    bankroll: float,
    section: str = DEFAULT_SECTION,
) -> None:
    """*Legenda*: the glossary (``legend_markdown``) plus a worked example on a row of the value panel as shown.

    The example row comes from the same filtered list the *Opportunità di
    Valore* tab displays (selected matchday or every matchday, sidebar
    competition, EV threshold): the first row without ``LONGSHOT`` /
    ``SUSPICIOUS_EDGE`` (:func:`pick_example_row`). Without such a row the
    example is synthetic and its header says why (no quotes at all, or no value
    row with the current filters).
    """
    demo_banner(state)
    st.markdown(legend_markdown())
    if section == SECTION_D:
        st.info(f"Sezione Serie D: {state.serie_d_status}.", icon="ℹ️")
    show_all = bool(st.session_state.get(section_key(VALUE_ALL_MATCHDAYS_KEY, section), False))
    rows = value_opportunities(state, min_ev, girone, matchday=ALL_MATCHDAYS if show_all else matchday, section=section)
    chosen = pick_example_row(rows)
    scope = value_scope_text(girone, matchday, show_all, section)
    with st.expander("Come leggere una riga del pannello valore", expanded=False):
        if chosen is None:
            reason = SYNTHETIC_REASON_NO_QUOTES if not state.quotes else f"{SYNTHETIC_REASON_NO_VALUE} ({scope}, soglia EV {min_ev:.1%})"
            example = WorkedExample.synthetic(kelly_fraction=kelly_fraction, bankroll_units=bankroll, reason=reason)
            st.caption(f"Nessuna riga nel pannello Opportunità di Valore con i filtri correnti ({scope}, soglia EV {min_ev:.1%}): esempio costruito.")
        else:
            example = WorkedExample.from_opportunity(chosen, bankroll_units=bankroll)
            position = rows.index(chosen) + 1
            if position == 1 and POOR_EXAMPLE_FLAGS.isdisjoint(chosen.flags):
                which = "la prima riga"
            elif position == 1:
                which = "la prima riga (tutte le righe sono LONGSHOT o SUSPICIOUS_EDGE: leggile con la cautela che la legenda consiglia)"
            else:
                which = f"la riga {position} di {len(rows)}, la prima senza flag LONGSHOT / SUSPICIOUS_EDGE (quote lunghe o EV sospetti insegnano poco)"
            st.caption(f"Esempio preso dal pannello Opportunità di Valore con i filtri correnti ({scope}, soglia EV {min_ev:.1%}): {which}.")
        st.markdown(worked_example_markdown(example))


def render_export_tab(state: PipelineState, section: str = DEFAULT_SECTION) -> None:
    """*Export*: the value report and the full predictions of the section as CSV."""
    demo_banner(state)
    st.subheader(f"Export CSV · {section}")
    if render_section_notice(state, section):
        return
    today = datetime.now(UTC).astimezone().date().isoformat()
    slug = section_slug(section)
    value_frame = opportunities_table(state, section)
    preds_frame = predictions_table(state, section)
    c1, c2 = st.columns(2)
    with c1:
        st.markdown(f"**Report valore** · {len(value_frame)} selezioni valutate (EV, edge, Kelly, flag) · {section}")
        st.download_button(
            "Scarica report valore (CSV)",
            data=export_csv_bytes(value_frame, demo=state.is_demo),
            file_name=demo_file_name(f"report_valore_{slug}_{today}.csv", state.is_demo),
            mime="text/csv",
            key="dl_value",
            **FULL_WIDTH["download_button"],
        )
    with c2:
        st.markdown(f"**Previsioni complete** · {len(preds_frame)} partite (λ, μ, probabilità e quote fair di ogni mercato) · {section}")
        st.download_button(
            "Scarica previsioni (CSV)",
            data=export_csv_bytes(preds_frame, demo=state.is_demo),
            file_name=demo_file_name(f"previsioni_{slug}_{today}.csv", state.is_demo),
            mime="text/csv",
            key="dl_predictions",
            **FULL_WIDTH["download_button"],
        )
    st.caption(
        "CSV UTF-8, separatore virgola, decimale punto, date ISO (AAAA-MM-GG)."
        + (
            " Modalità demo: i file si chiamano DEMO_... e la prima colonna AVVISO dice DATI SIMULATI - NON USARE PER SCOMMETTERE."
            if state.is_demo
            else " Le previsioni coprono solo le partite non ancora iniziate."
        )
    )
    with st.expander("Anteprima previsioni"):
        st.dataframe(preds_frame, hide_index=True, **FULL_WIDTH["dataframe"])


# =============================================================================
# Reliability (walk-forward backtest) and recalibration
# =============================================================================
@dataclass
class ReliabilityStore:
    """Process-wide memo of the reliability reports and of the views derived from them (track records, tuning).

    Shared by every session (``st.cache_resource``): a backtest computed for
    one data version is reused by every rerun and every session until the data
    changes (its key, :func:`reliability_cache_key`, carries the data version).
    ``values`` holds the derived views keyed by ``(report key, parameters)``;
    both maps keep only the most recent entries.
    """

    lock: threading.Lock = field(default_factory=threading.Lock)
    reports: dict[str, ReliabilityReport] = field(default_factory=dict)
    values: dict[tuple, object] = field(default_factory=dict)
    tuning: dict[str, ValidatedTuning] = field(default_factory=dict)

    def remember(self, key: str, report: ReliabilityReport, *, keep: int = 4) -> None:
        with self.lock:
            self.reports[key] = report
            while len(self.reports) > keep:
                self.reports.pop(next(iter(self.reports)))

    def memo(self, key: tuple, compute):  # noqa: ANN001, ANN201 - a callable and its result
        with self.lock:
            if key in self.values:
                return self.values[key]
        value = compute()
        with self.lock:
            self.values[key] = value
            while len(self.values) > 64:
                self.values.pop(next(iter(self.values)))
        return value


@st.cache_resource
def reliability_store() -> ReliabilityStore:
    """The process-wide :class:`ReliabilityStore`."""
    return ReliabilityStore()


def backtest_mode() -> str:
    """``SERIEC_BACKTEST``: ``auto`` (default), ``manual`` or ``off``; an unknown value is logged and read as ``auto``."""
    raw = os.getenv(BACKTEST_ENV_VAR, "").strip().lower()
    if not raw:
        return "auto"
    if raw not in BACKTEST_MODES:
        log.warning("unknown backtest mode, using auto", extra={"variable": BACKTEST_ENV_VAR, "value": raw})
        return "auto"
    return raw


def data_version(*, demo: bool, data_dir: Path, seed: int, n_rounds: int | None, season: str, history: int, as_of: date) -> str:
    """Identifier of the data a state is built on: the local data stamp, or the demo's simulation parameters."""
    if demo:
        return f"demo:{seed}:{n_rounds}:{season}:{history}:{as_of.isoformat()}"
    return f"data:{data_stamp(data_dir)}"


def reliability_cache_key(state: PipelineState, version: str, n_seasons: int) -> str:
    """Key of the backtest of a state: the logic version and the competition configuration (:func:`reliability_key`), the data version, the season, the evaluated seasons and the model settings.

    The configuration part makes a league switched on or off, or moved to another
    section, miss the saved ``data/backtest.*`` even when the data files did not change.
    """
    return reliability_key(version, state.season_label, n_seasons, f"xi={state.xi:g}", f"prior={state.prior}")


def selected_backtest_seasons() -> int:
    """The evaluated seasons chosen in *Affidabilità modello* (:data:`DEFAULT_BACKTEST_SEASONS` by default)."""
    value = st.session_state.get(RELIABILITY_SEASONS_KEY, DEFAULT_BACKTEST_SEASONS)
    return int(value) if value in RELIABILITY_SEASON_OPTIONS else DEFAULT_BACKTEST_SEASONS


def obtain_reliability(
    state: PipelineState,
    *,
    version: str,
    n_seasons: int,
    data_dir: Path,
    quotes_dir: Path,
    compute: bool,
) -> ReliabilityReport | None:
    """The reliability report of ``state``: from memory, else from ``data/backtest.*``, else computed now (when ``compute``).

    The walk-forward fits one model per past week block, so it runs once per
    data version with a progress bar and is then persisted (real data only)
    and shared by every session. ``None`` when nothing is available and
    ``compute`` is false (manual mode before the button, mode ``off``). The
    matchday-by-matchday recalibration of the backtest is attached here too
    (:func:`attach_rolling`), never on a tab's first render.
    """
    key = reliability_cache_key(state, version, n_seasons)
    store = reliability_store()
    with store.lock:
        found = store.reports.get(key)
    if found is not None:
        return found
    report: ReliabilityReport | None = None
    odds_history = pd.DataFrame() if state.is_demo else data_refresh.load_odds_history_frame(quotes_dir)
    if not state.is_demo:
        loaded = data_refresh.load_backtest(data_dir, key=key)
        if loaded is not None:
            report = ReliabilityReport(result=loaded[0], recalibration=loaded[1], odds_history=odds_history, key=key, from_cache=True)
    if report is None and compute:
        bar = st.progress(0.0, text="Verifica fuori campione del modello (backtest walk-forward)…")

        def progress(fraction: float, message: str) -> None:
            bar.progress(min(max(float(fraction), 0.0), 1.0), text=message)

        with st.spinner("Backtest walk-forward: ogni giornata passata viene prevista con un modello stimato solo sulle partite precedenti…"):
            report = reliability_for_state(state, n_seasons=n_seasons, odds_history=odds_history, progress=progress, key=key)
        bar.empty()
        if not state.is_demo:
            data_refresh.save_backtest(report.result, report.recalibration, key=key, processed_dir=data_dir)
    if report is not None:
        report = attach_rolling(report, state=state, data_dir=data_dir)
        store.remember(key, report)
        log.info("reliability report ready", extra={"key": key, "n_matches": report.result.n_matches, "from_cache": report.from_cache})
    return report


#: Errors a view built on the backtest may raise on a stale or inconsistent report (a saved backtest of a league
#: switched off since, a regrouped section): caught per view so the other tabs still render.
REPORT_ERRORS: Final[tuple[type[Exception], ...]] = (ValueError, KeyError, RuntimeError)


def report_error_notice(what: str, exc: BaseException) -> None:
    """Log a failed backtest view and say so on the page instead of stopping the script (the next tabs still render)."""
    log.warning("reliability view failed", extra={"view": what, "error": str(exc)})
    st.warning(
        f"{what}: vista non disponibile ({exc}). Il backtest non corrisponde ai dati o alla configurazione attuale: "
        "aggiorna i dati (il backtest viene ricalcolato) oppure, se il problema resta, cancella data/backtest.parquet. "
        "Le altre schede funzionano normalmente.",
        icon="⚠️",
    )


def attach_rolling(report: ReliabilityReport, *, state: PipelineState, data_dir: Path) -> ReliabilityReport:
    """The report with its matchday-by-matchday out-of-sample recalibration, read from disk or computed now (with a spinner) and saved.

    Needed only when the recalibration enabled a market (otherwise the
    probabilities are never recalibrated and nothing is computed). It costs one
    validated recalibration per past matchday (tens of seconds on two real
    seasons), so it runs once per backtest, here on the loading path rather than
    inside a tab, and is persisted next to ``data/backtest.parquet`` under the
    same key (real data only): a restart reads it back.
    """
    if report.empty or report.rolling is not None or not report.recalibration.enabled:
        return report
    rolling = None if state.is_demo else data_refresh.load_rolling(data_dir, key=report.key, frame=report.frame)
    if rolling is None:
        with st.spinner("Ricalibrazione fuori campione giornata per giornata (ogni giornata corretta solo con quelle precedenti)…"):
            rolling = rolling_frame(report)
        if not state.is_demo:
            data_refresh.save_rolling(rolling, key=report.key, processed_dir=data_dir)
    return with_rolling(report, rolling)


def recalibration_wanted(report: ReliabilityReport | None) -> bool:
    """The *Usa probabilità ricalibrate* choice, read before any tab renders (default: on only when the model enabled a market)."""
    if report is None:
        return False
    if RECALIBRATION_STATE_KEY not in st.session_state:
        st.session_state[RECALIBRATION_STATE_KEY] = bool(report.recalibration.enabled)
    return bool(st.session_state.get(RECALIBRATION_STATE_KEY)) and report.recalibration.enabled


def _metric_value(value: float, fmt: str = "{:.4f}") -> str:
    return fmt.format(value) if value is not None and pd.notna(value) else "n.d."


def render_recalibration_controls(state: PipelineState, report: ReliabilityReport) -> None:
    """The recalibration status (*Calibrazione applicata: sì/no*), its per-market table and the toggle."""
    model = report.recalibration
    st.markdown("**Imparare dal differenziale: ricalibrazione validata**")
    st.caption(
        "La correzione (Platt per GG/NG e Over/Under, temperatura multinomiale per 1X2, con penalità verso «nessuna correzione») "
        "è stimata sulle giornate passate più vecchie e giudicata su quelle più recenti: un mercato viene corretto solo se la "
        "log loss fuori campione migliora almeno della soglia, altrimenti restano le probabilità originali. È reversibile: il "
        "modello non viene mai sovrascritto."
    )
    if model.enabled:
        st.success(model.describe(), icon="✅")
    else:
        st.info(model.describe(), icon="ℹ️")
    st.toggle(
        "Usa probabilità ricalibrate",
        key=RECALIBRATION_STATE_KEY,
        disabled=not model.enabled,
        help=(
            "Acceso: Giornata, Opportunità di Valore, Selezioni di giornata e Schedine usano le probabilità ricalibrate (solo i "
            "mercati che hanno superato la verifica fuori campione); nella tabella Giornata compare accanto la probabilità originale. "
            "Spento: le probabilità originali del modello."
        ),
    )
    st.dataframe(recalibration_table(model), hide_index=True, **FULL_WIDTH["dataframe"])


def render_tuning(state: PipelineState, report: ReliabilityReport, version: str) -> None:
    """Optional walk-forward tuning of the time decay xi and of the prior strength (button, cached per data version).

    The point is chosen on the earlier matchdays of the current season and
    confirmed on the later ones (:func:`validated_tuning`); the switch that
    applies it is enabled only when the held-out gain reaches the
    recalibration's own threshold.
    """
    store = reliability_store()
    key = f"{version}|{state.season_label}"
    with st.expander("Ottimizza decadimento (xi) e prior (facoltativo, lento)", expanded=False):
        st.caption(
            "Ripete il backtest della stagione corrente per 9 combinazioni (xi 0.001 / 0.0018 / 0.003 per giorno × prior "
            "automatica / forza 0.2 / 0.4). La combinazione viene **scelta** sulle giornate più vecchie (log loss fuori campione "
            "media di 1X2, GG/NG e Over/Under 2.5) e **confermata** sul 30% più recente, che la scelta non ha visto: si può "
            f"applicare solo se lì batte la configurazione di produzione di almeno lo {DEFAULT_MIN_GAIN * 100:.1f}% (la stessa "
            "soglia della ricalibrazione). Richiede qualche minuto sui dati reali; il risultato resta in memoria finché i dati "
            "non cambiano."
        )
        tuned = store.tuning.get(key)
        if st.button("Esegui l'ottimizzazione", key=TUNING_BUTTON_KEY):
            bar = st.progress(0.0, text="Ottimizzazione walk-forward…")
            seasons = (state.season_label.replace("_", "/"),)
            tuned = validated_tuning(
                state.matches,
                state.registry,
                seasons=seasons,
                progress=lambda f, m: bar.progress(min(max(float(f), 0.0), 1.0), text=m),
            )
            bar.empty()
            with store.lock:
                store.tuning[key] = tuned
        if tuned is None:
            return
        (st.success if tuned.accepted else st.info)(tuned.describe())
        st.dataframe(tuning_table(tuned), hide_index=True, **FULL_WIDTH["dataframe"])
        if not tuned.accepted:
            st.session_state[USE_TUNED_KEY] = False
        st.toggle(
            "Usa xi e prior ottimizzati nella stima",
            key=USE_TUNED_KEY,
            disabled=not tuned.accepted,
            help=(
                "Ristima il modello con la combinazione scelta (e ripete il backtest con la stessa combinazione: le giornate su "
                "cui è stata scelta non sono più del tutto fuori campione, la tab lo segnala). Attivabile solo se la conferma "
                "sulle giornate più recenti supera la soglia. Spegnilo per tornare alla configurazione standard."
            ),
        )


def active_tuning(version: str) -> ValidatedTuning | None:
    """The validated tuning applied to the fit: switched on by the user and confirmed on held-out matchdays (``None`` otherwise)."""
    if not st.session_state.get(USE_TUNED_KEY):
        return None
    store = reliability_store()
    for key, tuned in store.tuning.items():
        if key.startswith(f"{version}|") and tuned.accepted:
            return tuned
    return None


def tuned_settings(version: str, season: str) -> tuple[float, str]:
    """``(xi, prior)`` for the fit: the tuned ones when the user switched them on and the held-out check confirmed them, else the defaults."""
    tuned = active_tuning(version)
    return tuned.settings if tuned is not None else (0.0018, "auto")


#: Why the market is compared only with the model's scores on the quoted matches (under the per-market table).
QUOTED_SUBSET_NOTE: Final[str] = (
    "«Log loss modello» e «Brier modello» coprono tutte le partite valutate; il bookmaker si misura solo sulle «Partite con "
    f"quote». Per il confronto con il mercato usa le colonne «{QUOTED_LOG_LOSS_COLUMN}» e «{QUOTED_BRIER_COLUMN}», "
    "calcolate sulle stesse partite del mercato (come «Skill vs mercato»): la copertura delle quote cambia da campionato "
    "a campionato, e confrontare numeri su partite diverse può far sembrare il modello migliore del mercato quando non lo è."
)


def market_column_config() -> dict[str, object]:
    """Formats and tooltips of the reliability score tables (per market and per competition)."""
    return {
        "Skill vs base (%)": st.column_config.NumberColumn("Skill vs base (%)", format="%+.2f %%", help="1 − log loss modello / log loss base: positivo = meglio delle frequenze di base."),
        "Skill vs mercato (%)": st.column_config.NumberColumn(
            "Skill vs mercato (%)",
            format="%+.2f %%",
            help="1 − log loss modello / log loss mercato sulle stesse partite con quote: positivo = meglio del bookmaker (raro e fragile su campioni piccoli).",
        ),
        QUOTED_LOG_LOSS_COLUMN: st.column_config.NumberColumn(QUOTED_LOG_LOSS_COLUMN, help="Log loss del modello sulle sole partite con quote: è il numero da confrontare con «Log loss mercato»."),
        QUOTED_BRIER_COLUMN: st.column_config.NumberColumn(QUOTED_BRIER_COLUMN, help="Brier del modello sulle sole partite con quote: è il numero da confrontare con «Brier mercato»."),
        "Margine medio (%)": st.column_config.NumberColumn("Margine medio (%)", format="%.1f %%", help="Sovrapprezzo medio del bookmaker sul mercato."),
    }


def render_reliability_tab(
    state: PipelineState,
    report: ReliabilityReport | None,
    section: str = DEFAULT_SECTION,
    *,
    mode: str = "auto",
    version: str = "",
) -> None:
    """*Affidabilità modello*: how well the out-of-sample probabilities matched the results, and the recalibration they support."""
    demo_banner(state)
    st.subheader(f"Affidabilità del modello · {section}")
    st.caption(
        "Ogni giornata passata (blocco della settimana calcistica) è prevista da un modello stimato **solo** sulle partite "
        "giocate prima del suo calcio d'inizio, con lo stesso decadimento e la stessa prior della dashboard: sono le probabilità "
        "che il modello avrebbe pubblicato allora, confrontate con quello che è successo. Le probabilità della stima di produzione "
        "(che ha già visto quei risultati) non vengono mai usate per giudicare la precisione."
    )
    c_seasons, _ = st.columns([1, 2])
    with c_seasons:
        if st.session_state.get(RELIABILITY_SEASONS_KEY) not in RELIABILITY_SEASON_OPTIONS:
            st.session_state[RELIABILITY_SEASONS_KEY] = DEFAULT_BACKTEST_SEASONS
        st.selectbox(
            "Stagioni valutate",
            list(RELIABILITY_SEASON_OPTIONS),
            format_func=lambda n: RELIABILITY_SEASON_OPTIONS[n],
            key=RELIABILITY_SEASONS_KEY,
            help="Più stagioni = più partite valutate e un giudizio più stabile (il backtest viene ricalcolato).",
        )
    if report is None:
        if mode == "off":
            st.info(f"Backtest disattivato ({BACKTEST_ENV_VAR}=off): nessuna verifica fuori campione e nessuna ricalibrazione.", icon="ℹ️")
            return
        st.info(
            "Il backtest walk-forward non è ancora stato calcolato per questi dati: richiede una stima per ogni settimana passata "
            "(da qualche secondo a un minuto).",
            icon="⏳",
        )
        if st.button("Calcola l'affidabilità del modello", key=BACKTEST_BUTTON_KEY, type="primary"):
            st.session_state[BACKTEST_REQUEST_KEY] = True
            st.rerun()
        return
    result = report.result
    st.caption(result.summary() + (" Letto dal backtest salvato in data/backtest.parquet." if report.from_cache else ""))
    tuned = active_tuning(version)
    if tuned is not None:
        st.warning(tuning_disclosure(tuned), icon="🎛️")
    if report.empty:
        st.info("Nessuna partita valutabile: servono almeno 150 partite giocate prima della prima giornata da valutare.")
        render_recalibration_controls(state, report)
        return
    store = reliability_store()
    present = [s for s in dict.fromkeys(report.frame["section"].astype(str))]
    scopes = [section, ALL_SECTIONS_LABEL] if section in present else [ALL_SECTIONS_LABEL]
    scope_key = section_key(RELIABILITY_SCOPE_KEY, section)
    if st.session_state.get(scope_key) not in scopes:
        st.session_state.pop(scope_key, None)
    c_scope, c_sel = st.columns([2, 1])
    with c_scope:
        scope = str(st.radio("Ambito", scopes, key=scope_key, horizontal=True, help="La sezione scelta nella sidebar oppure tutte le sezioni insieme."))
    with c_sel:
        if st.session_state.get(RELIABILITY_SELECTION_KEY) not in RELIABILITY_SELECTIONS:
            st.session_state.pop(RELIABILITY_SELECTION_KEY, None)
        label = str(st.selectbox("Esito", list(RELIABILITY_SELECTIONS), key=RELIABILITY_SELECTION_KEY, help="Esito del grafico di affidabilità e dei riquadri."))
    market, selection = RELIABILITY_SELECTIONS[label]
    scope_arg = None if scope == ALL_SECTIONS_LABEL else scope
    rolled = None
    if state.calibrated:
        rolled = store.memo((report.key, "rolling"), lambda: rolling_frame(report)).frame
    head = headline_for(report, scope_arg, market)
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Partite valutate", f"{head['n']}", help="Partite previste fuori campione nell'ambito scelto.")
    c2.metric(
        f"Log loss modello · {market.value}",
        _metric_value(head["log_loss"]),
        delta=None if pd.isna(head["log_loss_base"]) else f"{head['log_loss'] - head['log_loss_base']:+.4f} vs base",
        delta_color="inverse",
        help=(
            f"−ln della probabilità data all'esito avvenuto, in media su tutte le {head['n']} partite valutate: più bassa è "
            "meglio. Il delta è contro le frequenze di base del campionato sulle stesse partite. Non confrontarla con il "
            "mercato: il bookmaker si misura solo sulle partite con quote (riquadro accanto)."
        ),
    )
    c3.metric(**market_card(head))
    c4.metric("Brier", _metric_value(head["brier"]), help="Errore quadratico medio delle probabilità (1X2 da 0 a 2, mercati a due esiti da 0 a 1): più basso è meglio.")
    c5.metric("Accuratezza", _metric_value(head["accuracy"], "{:.1f}%"), help="Quante volte l'esito più probabile secondo il modello è avvenuto.")
    st.markdown("**Punteggi per mercato** (modello, frequenze di base del campionato, bookmaker senza margine)")
    st.dataframe(
        metrics_table_for(report, scope_arg),
        hide_index=True,
        **FULL_WIDTH["dataframe"],
        column_config=market_column_config(),
    )
    st.caption(QUOTED_SUBSET_NOTE)
    table = reliability_table_for(report, scope_arg, market, selection)
    calibrated_table = reliability_table_for(report, scope_arg, market, selection, frame=rolled) if rolled is not None else None
    st.plotly_chart(
        reliability_figure(table, f"Affidabilità · {label} · {scope}", calibrated=calibrated_table),
        **FULL_WIDTH["plotly_chart"],
        key="reliability_chart",
    )
    st.caption(
        "Ogni punto è un gruppo di partite con probabilità prevista simile: sull'asse orizzontale la probabilità media prevista, "
        "su quello verticale quante volte l'esito è avvenuto, con l'intervallo di confidenza al 95% (Wilson). Sulla diagonale "
        "tratteggiata il modello è tarato; sopra la diagonale l'esito avviene più spesso di quanto previsto, sotto meno spesso. "
        "Le barre larghe dicono che il gruppo ha poche partite."
    )
    st.markdown(f"**Per competizione** · {MARKET_LABELS_UI.get(market, market.value)}")
    st.dataframe(competition_metrics_table(report, scope_arg, market), hide_index=True, **FULL_WIDTH["dataframe"], column_config=market_column_config())

    st.markdown("**Track record delle liste e delle schedine** (quello che il modello avrebbe proposto in ogni giornata passata)")
    frame_key = "rolling" if rolled is not None else "raw"
    picks = store.memo((report.key, "picks", frame_key), lambda: picks_track_for(report, frame=rolled))
    st.dataframe(picks_record_table(picks, scope_arg), hide_index=True, **FULL_WIDTH["dataframe"])
    st.caption(
        f"Top {DEFAULT_N_DRAWS} X e top {DEFAULT_N_BTTS} GG di ogni giornata ({'tutte le sezioni insieme' if scope_arg is None else 'dentro la sezione'}): "
        "hit rate con intervallo di Wilson al 95% contro la probabilità media del modello (se coincidono il modello è tarato), ROI "
        "a puntata fissa sulla miglior quota pre-partita quando c'è."
    )
    st.markdown("**Multiple top-k delle liste** (le prime k partite della lista di ogni giornata giocate insieme)")
    st.dataframe(
        accumulators_table(picks, scope_arg),
        hide_index=True,
        **FULL_WIDTH["dataframe"],
        column_config={
            "Attese": st.column_config.NumberColumn("Attese", help="Somma, giornata per giornata, del prodotto delle k probabilità (indipendenza ipotizzata)."),
            "ROI (%)": st.column_config.NumberColumn("ROI (%)", help="Puntata 1 per multipla al prodotto delle migliori quote pre-partita, solo nelle giornate con tutte le partite quotate."),
        },
        key="accumulators_record",
    )
    st.caption(
        "È la giocata «le prime 2 GG della Serie C» o «le prime 4 della Serie D» ripetuta su ogni giornata passata della sezione: "
        "conta quante giornate sono state giocate, quante vinte contro quante il modello se ne aspettava e l'intervallo di "
        "confidenza, non la singola domenica vinta."
    )
    with st.expander("Liste giornata per giornata", expanded=False):
        st.dataframe(
            picks_by_matchday_table(picks, scope_arg),
            hide_index=True,
            **FULL_WIDTH["dataframe"],
            column_config={"Giornata": st.column_config.DateColumn("Giornata", format="DD/MM/YYYY", help="Primo giorno del blocco della settimana calcistica.")},
            key="picks_history",
        )
    sections = None if scope_arg is None else section_members(scope_arg)
    slips_track = store.memo(
        (report.key, "slips", frame_key, tuple(sections or ()), DEFAULT_N_X, DEFAULT_N_GG, DEFAULT_N_MIXED, 0.0),
        lambda: slips_track_for(report, sections, frame=rolled),
    )
    st.dataframe(slips_record_table(slips_track), hide_index=True, **FULL_WIDTH["dataframe"])
    with st.expander("Schedine giornata per giornata", expanded=False):
        st.dataframe(
            slips_by_matchday_table(slips_track),
            hide_index=True,
            **FULL_WIDTH["dataframe"],
            column_config={"Giornata": st.column_config.DateColumn("Giornata", format="DD/MM/YYYY", help="Primo giorno del blocco della settimana calcistica.")},
        )
    if report.has_odds:
        st.markdown(f"**Singole di valore** (EV ≥ {state.min_ev:.1%} sulla miglior quota pre-partita)")
        st.dataframe(
            store.memo((report.key, "value", frame_key, scope, state.min_ev), lambda: value_track_for(report, scope_arg, min_ev=state.min_ev, frame=rolled)),
            hide_index=True,
            **FULL_WIDTH["dataframe"],
        )
    else:
        st.caption("Nessuno storico delle quote pre-partita delle partite giocate: confronto con il mercato, ROI e singole di valore non disponibili.")
    st.warning(EVIDENCE_NOTE, icon="⚖️")
    render_recalibration_controls(state, report)
    render_tuning(state, report, version)
    st.download_button(
        "Scarica i punteggi aggregati (CSV)",
        data=export_csv_bytes(reliability_export_frame(report), demo=state.is_demo),
        file_name=demo_file_name(f"affidabilita_{datetime.now(UTC).astimezone().date().isoformat()}.csv", state.is_demo),
        mime="text/csv",
        key="dl_reliability",
        help="Solo punteggi aggregati per sezione e mercato (nessuna partita e nessuna quota).",
        **FULL_WIDTH["download_button"],
    )


#: Italian market names of the reliability tab (the value panel's).
MARKET_LABELS_UI: Final[dict[Market, str]] = {
    Market.MATCH_ODDS: "Esito finale 1X2",
    Market.BTTS: "Gol/No Gol",
    Market.OVER_UNDER_15: "Over/Under 1.5",
    Market.OVER_UNDER_25: "Over/Under 2.5",
    Market.OVER_UNDER_35: "Over/Under 3.5",
}


# =============================================================================
# Slips (schedine)
# =============================================================================
def _slip_number(value: float | None, fmt: str) -> str:
    return fmt.format(value) if value is not None and pd.notna(value) else "n.d."


def render_slips_tab(
    state: PipelineState,
    report: ReliabilityReport | None,
    section: str = DEFAULT_SECTION,
    matchday: Matchday | None = None,
) -> None:
    """*Schedine*: the three slips of a matchday (5 X, 10 GG, mista 1 / 2 / GG / Over 2.5) over the chosen sections, and their past record."""
    demo_banner(state)
    st.subheader("Schedine della giornata")
    options = list(section_options(state))
    defaults = list(sections_with_data(state)) or [section]
    stored = st.session_state.get(SLIPS_SECTIONS_KEY)
    if not isinstance(stored, list) or any(s not in options for s in stored):
        st.session_state[SLIPS_SECTIONS_KEY] = [s for s in defaults if s in options]
    chosen = list(
        st.multiselect(
            "Sezioni",
            options,
            key=SLIPS_SECTIONS_KEY,
            help="Le partite candidate: di default tutte le sezioni con dati. Ogni partita entra al massimo una volta per schedina.",
        )
    )
    if not chosen:
        st.info("Scegli almeno una sezione.")
        return
    days = slips_matchdays(state, chosen)
    if not days:
        st.info("Nessuna giornata con partite da giocare nelle sezioni scelte.")
        return
    keys = [d.key for d in days]
    labels = {d.key: d.label for d in days}
    default = default_slips_matchday(days, matchday)
    seed_key = f"{SLIPS_MATCHDAY_KEY}_default"
    if st.session_state.get(SLIPS_MATCHDAY_KEY) not in keys or st.session_state.get(SLIPS_MATCHDAY_KEY) == st.session_state.get(seed_key):
        st.session_state[SLIPS_MATCHDAY_KEY] = default.key if default is not None else keys[0]
    st.session_state[seed_key] = default.key if default is not None else keys[0]
    c_day, c_min, c_quoted = st.columns([3, 1, 1])
    with c_day:
        day_key = st.selectbox(
            "Giornata",
            keys,
            format_func=lambda k: labels.get(k, k),
            key=SLIPS_MATCHDAY_KEY,
            help="Segue la giornata scelta negli altri tab finché non ne scegli un'altra qui.",
        )
    with c_min:
        min_pct = st.slider("P minima per partita (%)", 0, 90, 0, step=5, key=SLIPS_MIN_P_KEY, help="Esclude le partite il cui segno ha probabilità più bassa.")
    with c_quoted:
        only_quoted = bool(st.checkbox("Solo partite quotate", value=False, key=SLIPS_ONLY_QUOTED_KEY, help="Solo selezioni con una quota del bookmaker nei dati."))
    c_x, c_gg, c_mix = st.columns(3)
    with c_x:
        n_x = int(st.slider("Partite schedina X", 1, SLIPS_MAX_X, DEFAULT_N_X, key=SLIPS_N_X_KEY))
    with c_gg:
        n_gg = int(st.slider("Partite schedina GG", 1, SLIPS_MAX_GG, DEFAULT_N_GG, key=SLIPS_N_GG_KEY))
    with c_mix:
        n_mixed = int(st.slider("Partite schedina mista", MIXED_MIN_LEGS, MIXED_MAX_LEGS, DEFAULT_N_MIXED, key=SLIPS_N_MIXED_KEY))
    day = next((d for d in days if d.key == day_key), days[0])
    slips = slips_for(state, day, chosen, n_x=n_x, n_gg=n_gg, n_mixed=n_mixed, min_probability=min_pct / 100.0, only_quoted=only_quoted)
    summary = slips_summary_for(slips, state)
    st.caption(
        f"{day.label} · partite non ancora iniziate delle sezioni {', '.join(chosen)} · probabilità "
        + ("ricalibrate (tab Affidabilità modello)" if state.calibrated else "originali del modello")
        + ". Schedina mista: per ogni partita il segno più probabile fra 1, 2, GG e Over 2.5, poi le partite con quel segno più probabile."
    )
    for slip, row in zip(slips, summary.to_dict(orient="records")):
        with st.container(border=True):
            st.markdown(f"**{slip.title}** · {slip.n_legs} partite")
            if slip.is_empty:
                st.info(slip.coverage_note or "Nessuna partita idonea.")
                continue
            m1, m2, m3, m4, m5, m6 = st.columns(6)
            m1.metric("Probabilità", _slip_number(slip.probability * 100.0, "{:.2f}%"), help=INDEPENDENCE_NOTE)
            m2.metric("Quota fair", _slip_number(slip.fair_odds, "{:.2f}"), help="1 / probabilità della schedina.")
            m3.metric("Quota book", _slip_number(slip.book_odds, "{:.2f}"), help="Prodotto delle migliori quote, solo se ogni partita è quotata.")
            m4.metric("EV", _slip_number(None if slip.ev is None else slip.ev * 100.0, "{:+.1f}%"), help="Probabilità × quota book − 1.")
            m5.metric("Partite vincenti attese", f"{slip.expected_winning_legs:.2f}", help="Somma delle probabilità delle singole partite.")
            m6.metric("Stake suggerito", str(row["Stake suggerito"]), help="Kelly frazionario sulla schedina, con tetto di 1 unità; «non consigliata» con EV ≤ 0 o senza quote.")
            st.dataframe(
                slips_frame([slip]).drop(columns=["Schedina"]),
                hide_index=True,
                **FULL_WIDTH["dataframe"],
                column_config={
                    "P. Modello (%)": st.column_config.NumberColumn("P. Modello (%)", format="%.1f %%"),
                    "Quota Fair": st.column_config.NumberColumn("Quota Fair", format="%.2f"),
                    "Quota Book": st.column_config.NumberColumn("Quota Book", format="%.2f"),
                },
                key=f"slip_legs_{slip.kind}",
            )
            if slip.coverage_note:
                st.caption(slip.coverage_note)
            st.caption(f"⚠️ {slip.margin_note}")
    st.caption(
        INDEPENDENCE_NOTE + " Una multipla vince solo se vincono tutte le partite: anche con probabilità alte per partita la "
        "probabilità della schedina crolla, e il margine del bookmaker si moltiplica a ogni partita aggiunta."
    )
    st.download_button(
        "Scarica le tre schedine (CSV)",
        data=slips_csv_bytes(slips, day, state),
        file_name=demo_file_name(f"schedine_{day.key.replace('#', '_')}.csv", state.is_demo),
        mime="text/csv",
        key="dl_slips",
        help="Una riga per partita con le cifre della sua schedina.",
        **FULL_WIDTH["download_button"],
    )
    st.markdown("**Come sarebbero andate le stesse schedine nelle giornate passate**")
    if report is None or report.empty:
        st.caption("Track record non disponibile: il backtest walk-forward non è stato calcolato (tab Affidabilità modello).")
        return
    try:
        render_slips_track_record(state, report, chosen, n_x=n_x, n_gg=n_gg, n_mixed=n_mixed, min_pct=min_pct)
    except REPORT_ERRORS as exc:  # a bad report never takes the rest of the page down
        report_error_notice("Track record delle schedine", exc)


def render_slips_track_record(state: PipelineState, report: ReliabilityReport, chosen: list[str], *, n_x: int, n_gg: int, n_mixed: int, min_pct: int) -> None:
    """The past record of the three slips over the chosen sections (walk-forward probabilities, recalibrated out of sample when on)."""
    store = reliability_store()
    rolled = store.memo((report.key, "rolling"), lambda: rolling_frame(report)).frame if state.calibrated else None
    members = tuple(dict.fromkeys(m for s in chosen for m in section_members(s)))
    track = store.memo(
        (report.key, "slips", "rolling" if rolled is not None else "raw", members, n_x, n_gg, n_mixed, min_pct / 100.0),
        lambda: slips_track_for(report, list(members), n_x=n_x, n_gg=n_gg, n_mixed=n_mixed, min_probability=min_pct / 100.0, frame=rolled),
    )
    st.dataframe(slips_record_table(track), hide_index=True, **FULL_WIDTH["dataframe"])
    with st.expander("Giornata per giornata", expanded=False):
        st.dataframe(
            slips_by_matchday_table(track),
            hide_index=True,
            **FULL_WIDTH["dataframe"],
            column_config={"Giornata": st.column_config.DateColumn("Giornata", format="DD/MM/YYYY")},
            key="slips_history",
        )
    st.caption(
        "Stesse regole e stesse lunghezze, applicate a ogni giornata passata con le probabilità fuori campione (il filtro «solo "
        "partite quotate» non si applica allo storico). " + EVIDENCE_NOTE
    )


def render_model_options() -> str:
    """Sidebar *Opzioni del modello*: the optimiser of the fit (L-BFGS-B by default, SLSQP kept as an option)."""
    with st.sidebar.expander("Opzioni del modello", expanded=False):
        if st.session_state.get(FIT_METHOD_KEY) not in FIT_METHODS:
            st.session_state.pop(FIT_METHOD_KEY, None)
        method = str(
            st.selectbox(
                "Ottimizzatore",
                list(FIT_METHODS),
                key=FIT_METHOD_KEY,
                help="L-BFGS-B (predefinito) e SLSQP raggiungono lo stesso ottimo; L-BFGS-B è circa 40 volte più veloce.",
            )
        )
    return method


# =============================================================================
# Page
# =============================================================================
def main() -> None:
    """Compose the page: demo banner or data refresh, cached state (or the setup page), section controls, badge, tabs.

    Real data by default: the sidebar data box refreshes it from FootyStats
    when it is missing or older than 6 hours (with a key) and without data the
    setup page replaces the tabs (:func:`render_setup_page`). ``SERIEC_DEMO=1``
    switches to the simulated league, flagged by a red banner on the page and
    on every tab.
    """
    demo = demo_requested()
    controls, min_ev, kelly_fraction, bankroll = render_sidebar()
    method = render_model_options()
    badge = st.sidebar.empty()
    data_box = st.sidebar.container()
    data_dir, quotes_dir = processed_dir(), odds_dir()
    if demo:
        st.error(DEMO_WARNING, icon="🚨")
        with data_box:
            st.caption("Modalità demo (SERIEC_DEMO=1): nessun dato FootyStats viene letto o scaricato.")
    else:
        render_data_panel(data_box, data_dir=data_dir, quotes_dir=quotes_dir)

    seed = env_int("SERIEC_SEED", DEFAULT_SEED, minimum=0)
    n_rounds = env_optional_int("SERIEC_SIM_ROUNDS", minimum=MIN_SIM_ROUNDS)
    season = env_season("SERIEC_SEASON")
    history_seasons = env_optional_int("SERIEC_HISTORY", minimum=0, maximum=MAX_HISTORY_SEASONS)
    history = DEFAULT_HISTORY_SEASONS if history_seasons is None else history_seasons
    as_of = resolve_as_of()
    version = data_version(demo=demo, data_dir=data_dir, seed=seed, n_rounds=n_rounds, season=season, history=history, as_of=as_of)
    xi, prior = tuned_settings(version, season)
    common = dict(
        min_ev=min_ev,
        kelly_fraction=kelly_fraction,
        bankroll_units=bankroll,
        simulate=True if demo else False,
        as_of=as_of,
        processed=str(data_dir),
        odds=str(quotes_dir),
        stamp=None if demo else data_stamp(data_dir),
        method=method,
        xi=xi,
        prior=prior,
    )
    try:
        state = load_state(seed=seed, n_rounds=n_rounds, season=season, history_seasons=history, **common)
    except NoDataError as exc:
        log.warning("no real data: setup page", extra={"reasons": list(exc.reasons), "processed_dir": str(exc.processed_dir)})
        badge.warning("**Nessun dato reale**: configura la chiave FootyStats e scarica i dati.", icon="🛑")
        render_setup_page(exc, data_dir=data_dir, quotes_dir=quotes_dir)
        return
    except ValueError as exc:
        # the page never crashes on a configuration the data layer refuses: say so and use the defaults
        log.warning(
            "pipeline failed with the requested parameters, falling back to the defaults",
            extra={"season": season, "history_seasons": history, "n_rounds": n_rounds, "seed": seed, "error": str(exc)},
        )
        st.error(
            f"Pipeline non eseguibile con i parametri richiesti (stagione {season_file_stem(season).replace('_', '/')}, "
            f"storico {history}, giornate {n_rounds if n_rounds is not None else 'da calendario'}, seed {seed}): {exc}. "
            f"Uso la configurazione predefinita: stagione {DEFAULT_SEASON.replace('_', '/')}, calendario al {as_of.isoformat()}, seed {DEFAULT_SEED}.",
            icon="⚠️",
        )
        try:
            state = load_state(seed=DEFAULT_SEED, n_rounds=None, season=DEFAULT_SEASON, history_seasons=DEFAULT_HISTORY_SEASONS, **common)
        except NoDataError as again:
            render_setup_page(again, data_dir=data_dir, quotes_dir=quotes_dir)
            return
    mode = backtest_mode()
    report: ReliabilityReport | None = None
    if mode != "off":
        compute = mode == "auto" or bool(st.session_state.get(BACKTEST_REQUEST_KEY))
        try:
            report = obtain_reliability(
                state, version=version, n_seasons=selected_backtest_seasons(), data_dir=data_dir, quotes_dir=quotes_dir, compute=compute
            )
        except (ValueError, KeyError, RuntimeError) as exc:  # the reliability view must never take the page down
            log.warning("backtest failed", extra={"error": str(exc)})
            st.warning(f"Backtest walk-forward non riuscito ({exc}): la tab Affidabilità modello resta vuota e si usano le probabilità originali.", icon="⚠️")
    variant = "raw"
    if recalibration_wanted(report) and report is not None:
        state = calibrated_state(state, state.built_at.isoformat(), report.key, report)
        variant = f"cal:{report.key}"
    if not state.is_demo:
        state = calendar_state(state, state.built_at.isoformat(), calendar_now(), variant)
    section, girone = render_section_controls(controls, state)
    render_source_badge(badge, state, section)

    st.title(PAGE_TITLE)
    scope = f"tutte le competizioni di {section}" if girone is None else girone.short
    st.caption(
        f"{perimeter_text(state, section)} · Dixon-Coles su {state.n_matches} partite (una stima per paese) · "
        f"{len(section_predictions(state, section))} previsioni · soglia EV {min_ev:.1%} · Kelly x{kelly_fraction:g} · "
        f"bankroll {bankroll:g} unità · sezione: {section} · vista: {scope} · probabilità "
        + ("ricalibrate" if state.calibrated else "originali del modello")
    )
    with st.expander("Dettagli pipeline", expanded=False):
        notes = list(state.notes)
        if serie_d_gironi():
            notes.append(f"Sezione Serie D: {state.serie_d_status}.")
        notes += registry_sync_notes(data_dir) if not state.is_demo else []
        st.markdown("\n".join(f"- {note}" for note in notes))
        st.caption(f"Stato calcolato alle {state.built_at.astimezone().strftime('%d/%m/%Y %H:%M:%S')} (fonte: {state.source}).")

    matchday = selected_matchday(state, section)  # before any tab renders, so the three matchday-aware tabs agree
    tab_rank, tab_match, tab_results, tab_value, tab_picks, tab_slips, tab_reliability, tab_export, tab_legend = st.tabs(list(TAB_LABELS))
    with tab_rank:
        render_ranking_tab(state, girone, section)
    with tab_match:
        render_match_tab(state, girone, matchday, section)
    with tab_results:
        render_results_tab(state, girone, section)
    with tab_value:
        render_value_tab(state, girone, min_ev, matchday, section)
    with tab_picks:
        render_picks_tab(state, matchday, section)
    with tab_slips:
        render_slips_tab(state, report, section, matchday)
    with tab_reliability:
        try:
            render_reliability_tab(state, report, section, mode=mode, version=version)
        except REPORT_ERRORS as exc:  # a bad report never takes the Esporta and Legenda tabs down
            report_error_notice("Affidabilità modello", exc)
    with tab_export:
        render_export_tab(state, section)
    with tab_legend:
        render_legend_tab(state, girone, min_ev, matchday, kelly_fraction, bankroll, section)


def registry_sync_notes(data_dir: Path) -> list[str]:
    """The *Dettagli pipeline* lines about the last FootyStats download: registry differences, competitions skipped.

    From the refresh run in this session when there was one, else from the
    stamp file the last refresh wrote (:func:`src.dashboard.data_refresh.last_refresh`).
    """
    result = st.session_state.get(LAST_REFRESH_STATE_KEY)
    if isinstance(result, data_refresh.RefreshResult):
        lines = [result.summary_text()]
        lines += [f"Dizionario squadre: {line}" for line in result.registry_sync]
        lines += result.warnings()
        return lines
    stamp = data_refresh.last_refresh(data_dir)
    if stamp is None:
        return []
    lines = [data_refresh.refresh_label(data_dir) or "Dati FootyStats scaricati."]
    lines += [f"Dizionario squadre: {line}" for line in stamp.get("registry_sync", []) or []]
    for code, reason in (stamp.get("errors") or {}).items():
        lines.append(f"Competizione {code} non scaricata all'ultimo aggiornamento: {reason}.")
    return lines

main()
