"""Serie C Quant Lab -- Streamlit dashboard (UI in Italian).

Run from the repository root::

    streamlit run src/dashboard/app.py

Environment variables
---------------------
``SERIEC_FORCE_SIMULATE=1``
    Always use the synthetic league, even when ``data/processed/matches.csv`` exists.
``SERIEC_SEASON`` (default ``2026_27``)
    Registry season (``config/teams_<season>.json``): the perimeter of teams that
    are ranked, predicted and priced, in every competition the file holds (the
    three Serie C gironi and Serie B; the competition selector, the tables and
    the perimeter banner follow ``config/competitions.toml``). Older seasons in
    the data are history only; the simulated league of another season runs on
    that season's calendar.
``SERIEC_HISTORY`` (int 0-2, default 2)
    Synthetic seasons before the registry season (2024/25 and 2025/26 for
    2026/27) prepended to the simulated league.
``SERIEC_AS_OF`` (ISO date, default today)
    Calendar position of the simulated league: rounds dated before it are
    played, the next round is the fixture list. It is resolved on every run
    (:func:`resolve_as_of`), so without the variable the position advances
    with the real calendar while the server runs instead of being cached.
``SERIEC_SIM_ROUNDS`` (int >= 1, optional)
    Fixed number of played rounds of the simulated league instead of the
    calendar; values above the season length are clamped by the data layer so
    one round of fixtures always remains.
``SERIEC_SEED`` (int >= 0, default 7)
    Random seed of the simulated league.

Invalid or out-of-range values (non-integers, a negative seed, zero rounds, an
unparseable date, an unknown season file) are logged and replaced by the
defaults instead of crashing the page; should the pipeline itself refuse a
valid-looking combination (``ValueError``), the page shows the error and falls
back to the default season, calendar and seed.

Sections
--------
The sidebar radio *Sezione* (key ``'section'``) splits the page into the
sections of :mod:`src.dashboard.state`: *Serie B e Serie C* (default) and
*Serie D*. The Dixon-Coles fit is one for the whole perimeter; every tab,
selector, table, pick list, export and the perimeter text are restricted to the
selected section, the picks (5 X / 12 GG) are ranked within it and the
matchdays are grouped from its own fixtures. The *Competizione* selector lists
*Tutti* plus the section's competitions; the matchday, fixture and filter
widgets have per-section keys (:func:`src.dashboard.state.section_key`), so
switching section starts from that section's defaults. When Serie D is
configured but not yet initialised (its gironi are filled by the FootyStats
harvester from the API, never shipped with the code) every tab shows one notice
with :attr:`~src.dashboard.state.PipelineState.serie_d_status` and the exact
command to run.

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
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Final


# -- bootstrap: make ``src.*`` importable when launched as ``streamlit run src/dashboard/app.py``
_REPO_ROOT: Final[Path] = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import streamlit as st

st.set_page_config(page_title="Serie C Quant Lab", page_icon="⚽", layout="wide")

from src.common.competitions import describe_competitions
from src.common.logging_config import get_logger
from src.common.schemas import Girone, Market, MatchPrediction
from src.common.teams import DEFAULT_SEASON, list_seasons, season_file_stem
from src.dashboard import container_width_kwargs
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
    DEFAULT_HISTORY_SEASONS,
    DEFAULT_SECTION,
    MATCHDAY_TABLE_KEY_PREFIX,
    MAX_HISTORY_SEASONS,
    MIN_SIM_ROUNDS,
    SECTION_D,
    Matchday,
    PipelineState,
    btts_picks_table,
    build_state,
    competition_options,
    coverage_text,
    draw_picks_table,
    export_csv_bytes,
    find_matchday,
    fixture_label,
    fixtures_for,
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
    strength_figure,
    traps_table,
    value_opportunities,
    value_table,
)
from src.engine.picks import DEFAULT_N_BTTS, DEFAULT_N_DRAWS
from src.engine.tables import DEFAULT_FORM_N, DEFAULT_XG_N

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
#: Tab labels, in order.
TAB_LABELS: Final[tuple[str, ...]] = (
    "Classifica Forza",
    "Giornata",
    "Risultati",
    "Opportunità di Valore",
    "Selezioni di giornata",
    "Export",
    "Legenda",
)


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
@st.cache_resource(show_spinner="Stima del modello Dixon-Coles in corso…")
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
) -> PipelineState:
    """Build (and cache) the pipeline state for one combination of sidebar parameters."""
    return build_state(
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
                    "Serie B e Serie C (gironi A/B/C) oppure Serie D (gironi A-I): ogni tab, selettore, tabella, lista e "
                    "export mostra solo la sezione scelta. La stima Dixon-Coles resta unica per tutto il perimetro."
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
                help="Filtra classifica, partite e opportunità per competizione della sezione (girone di Serie C, Serie B o girone di Serie D).",
            )
        )
    return section, selected_girone(girone_label)


def render_source_badge(container: st.delta_generator.DeltaGenerator, state: PipelineState, section: str | None = None) -> None:
    """Data-source badge: the section's perimeter (season, teams, current vs historical matches), fit diagnostics, Serie D status."""
    params = state.params
    body = (
        f"**{perimeter_text(state, section)}**  \n"
        f"{state.n_matches} partite nel fit (stima congiunta di tutto il perimetro) · "
        f"**{len(section_predictions(state, section))}** partite in programma · "
        f"{state.n_historical_teams} squadre storiche escluse  \n"
        f"Fit: {params.method} · LL {params.log_likelihood:.1f}  \n"
        f"Convergenza: {'sì' if params.converged else 'no'} · {params.fit_seconds:.2f} s  \n"
        f"Prior gaussiana: {'attiva' if state.prior_enabled else 'disattivata'}"
    )
    coverage = coverage_text(state, section)
    if coverage:
        body += f"  \n{coverage}"
    if serie_d_gironi():
        body += f"  \nSerie D: {state.serie_d_status}"
    if state.source == "processed":
        container.success(f"**Dati reali** (data/processed)  \n{body}", icon="✅")
    else:
        container.info(f"**Dati simulati** (generatore sintetico)  \n{body}", icon="🧪")


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
            "FootyStats in .env. La prima raccolta inizializza i nove gironi (18 squadre ciascuno, 34 giornate) in "
            f"config/teams_{state.season_label.replace('/', '_')}.json dalle liste squadre dell'API; poi la sezione mostra "
            "classifica, giornate, risultati, valore e selezioni come quella di Serie B e Serie C.",
            icon="ℹ️",
        )
    else:
        st.info(f"Sezione {section}: nessuna squadra del perimetro {state.season_label} con partite nei dati.", icon="ℹ️")
    return True


# =============================================================================
# Tabs
# =============================================================================
def render_ranking_tab(state: PipelineState, girone: Girone | None, section: str = DEFAULT_SECTION) -> None:
    """*Classifica Forza*: the strength ranking of the section (one competition with ``girone``) and its α/β charts."""
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
        "Giornata": cc.NumberColumn("Giornata", format="%d", width="small", help="Numero di turno della competizione."),
        "Casa": cc.TextColumn("Casa", pinned=True),
        "Ospiti": cc.TextColumn("Ospiti", pinned=True),
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


def render_match_tab(state: PipelineState, girone: Girone | None, matchday: Matchday | None, section: str = DEFAULT_SECTION) -> None:
    """*Giornata*: the whole matchday of the section as one table, then the detail of the selected fixture.

    The table (:func:`matchday_table_for`) has one row per fixture of the
    selected matchday and sidebar competition. Clicking a row
    (``selection_mode='single-row'``, ``on_select='rerun'``) selects that fixture:
    the *Partita* selectbox (key :data:`FIXTURE_STATE_KEY`) follows the click and
    stays available as the fallback; a choice made in the selectbox afterwards
    is kept until the next click (:data:`APPLIED_ROW_STATE_KEY`); both keys are
    scoped to the section (:func:`src.dashboard.state.section_key`). The widget
    key is scoped to the matchday, competition and section (:func:`matchday_table_key`):
    Streamlit keeps a selection per widget key regardless of the data, so a
    fixed key would carry the clicked row position onto the table of another
    competition or matchday and silently open a fixture never clicked; a new
    scope is a new widget, without a selection, and the detail falls back to
    the first fixture. Below the table: the metrics row, the score-matrix
    heatmap and the markets table of the selected fixture, as before.
    """
    st.subheader(f"Giornata: tutte le partite in tabella · {section}")
    if render_section_notice(state, section):
        return
    fixture_key = section_key(FIXTURE_STATE_KEY, section)
    applied_key = section_key(APPLIED_ROW_STATE_KEY, section)
    preds = fixtures_for(state, girone, matchday=matchday, section=section)
    if not preds:
        message = "Nessuna partita in programma per la competizione selezionata"
        message += "." if matchday is None else f" nella {matchday.label}."
        if state.skipped_fixtures:
            message += (
                f" {len(state.skipped_fixtures)} partite caricate sono state saltate perché di un'altra stagione o con squadre "
                f"fuori dal perimetro {state.season_label} (dettagli in *Dettagli pipeline*)."
            )
        st.info(message)
        return
    table = matchday_table_for(state, matchday, girone, section)
    scope = f"tutte le competizioni di {section}" if girone is None else girone.short
    n_matchdays = len(matchdays(state, section))
    st.caption(
        (f"{matchday.label} · " if matchday is not None else "")
        + f"{len(table)} partite ({scope}) · una riga per partita: forza (α/β), forma delle ultime {DEFAULT_FORM_N} gare, "
        f"xG delle ultime {DEFAULT_XG_N}" + (" ('-' quando il dato manca: in Serie D gli xG sono spesso assenti)" if section == SECTION_D else "")
        + ", gol attesi, probabilità, quote fair, miglior quota book, EV massimo e flag. Clicca una riga per aprire la partita qui sotto"
        + (f"; la giornata si cambia nel tab *Selezioni di giornata* ({n_matchdays} in programma)." if matchday is not None else ".")
    )
    table_key = matchday_table_key(matchday, girone, section)
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
    stem = "tutte" if matchday is None else matchday.key.replace("#", "_")
    st.download_button(
        "Scarica tabella giornata (CSV)",
        data=export_csv_bytes(table),
        file_name=f"giornata_{section_slug(section)}_{stem}{'' if girone is None else '_' + girone.value}.csv",
        mime="text/csv",
        key="dl_matchday",
        help="Tutte le colonne della tabella più match_id.",
        **FULL_WIDTH["download_button"],
    )
    labels = [fixture_label(p) for p in preds]
    label_by_id = {p.fixture.match_id: label for p, label in zip(preds, labels)}
    clicked = selected_row_match_id(event, table)
    applied = (table_key, clicked)  # the scope travels with the applied row: a selection of another table is never reused
    if clicked is None:
        st.session_state.pop(applied_key, None)
    elif clicked in label_by_id and st.session_state.get(applied_key) != applied:
        # a new click: the selectbox follows the table (a later manual choice in the selectbox is then respected)
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
        data=export_csv_bytes(table),
        file_name=f"risultati_{section_slug(section)}_{season.replace('/', '_')}{'' if girone is None else '_' + girone.value}{'' if team is None else '_' + team.replace(' ', '_')}.csv",
        mime="text/csv",
        key="dl_results",
        help="Tutte le colonne della tabella più match_id.",
        **FULL_WIDTH["download_button"],
    )


def render_value_tab(state: PipelineState, girone: Girone | None, min_ev: float, matchday: Matchday | None, section: str = DEFAULT_SECTION) -> None:
    """*Opportunità di Valore*: the value selections of the section's matchday and its REGRESSION_TRAP alerts."""
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
                f"I turni delle competizioni di {section} giocati nello stesso blocco della settimana (fine settimana ven-lun o "
                "infrasettimanale mar-gio), con i recuperi aggregati alla giornata più vicina; vale anche per Giornata e Opportunità di Valore."
            ),
        )
    with c_draws:
        n_draws = int(st.number_input("N risultati X", min_value=1, max_value=MAX_PICKS, value=DEFAULT_N_DRAWS, step=1, key="n_draws"))
    with c_btts:
        n_btts = int(st.number_input("N partite GG", min_value=1, max_value=MAX_PICKS, value=DEFAULT_N_BTTS, step=1, key="n_btts"))
    chosen = find_matchday(state, str(chosen_key), section) or matchday
    st.caption(
        f"{chosen.label} · {chosen.n_fixtures} partite · {chosen.start.strftime('%d/%m/%Y')}"
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
        data=picks_csv_bytes(draws, btts, chosen),
        file_name=f"selezioni_{section_slug(section)}_{chosen.key.replace('#', '_')}.csv",
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
            data=export_csv_bytes(value_frame),
            file_name=f"report_valore_{slug}_{today}.csv",
            mime="text/csv",
            key="dl_value",
            **FULL_WIDTH["download_button"],
        )
    with c2:
        st.markdown(f"**Previsioni complete** · {len(preds_frame)} partite (λ, μ, probabilità e quote fair di ogni mercato) · {section}")
        st.download_button(
            "Scarica previsioni (CSV)",
            data=export_csv_bytes(preds_frame),
            file_name=f"previsioni_{slug}_{today}.csv",
            mime="text/csv",
            key="dl_predictions",
            **FULL_WIDTH["download_button"],
        )
    st.caption("CSV UTF-8, separatore virgola, decimale punto, date ISO (AAAA-MM-GG).")
    with st.expander("Anteprima previsioni"):
        st.dataframe(preds_frame, hide_index=True, **FULL_WIDTH["dataframe"])


# =============================================================================
# Page
# =============================================================================
def main() -> None:
    """Compose the page: sidebar parameters, cached state, section controls, badge, subtitle, tabs of the section."""
    controls, min_ev, kelly_fraction, bankroll = render_sidebar()
    badge = st.sidebar.empty()

    simulate: bool | None = True if env_flag("SERIEC_FORCE_SIMULATE") else None
    seed = env_int("SERIEC_SEED", DEFAULT_SEED, minimum=0)
    n_rounds = env_optional_int("SERIEC_SIM_ROUNDS", minimum=MIN_SIM_ROUNDS)
    season = env_season("SERIEC_SEASON")
    history_seasons = env_optional_int("SERIEC_HISTORY", minimum=0, maximum=MAX_HISTORY_SEASONS)
    history = DEFAULT_HISTORY_SEASONS if history_seasons is None else history_seasons
    as_of = resolve_as_of()
    try:
        state = load_state(
            min_ev=min_ev,
            kelly_fraction=kelly_fraction,
            bankroll_units=bankroll,
            simulate=simulate,
            seed=seed,
            n_rounds=n_rounds,
            season=season,
            history_seasons=history,
            as_of=as_of,
        )
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
        state = load_state(
            min_ev=min_ev,
            kelly_fraction=kelly_fraction,
            bankroll_units=bankroll,
            simulate=simulate,
            seed=DEFAULT_SEED,
            n_rounds=None,
            season=DEFAULT_SEASON,
            history_seasons=DEFAULT_HISTORY_SEASONS,
            as_of=as_of,
        )
    section, girone = render_section_controls(controls, state)
    render_source_badge(badge, state, section)

    st.title("⚽ Serie C Quant Lab")
    scope = f"tutte le competizioni di {section}" if girone is None else girone.short
    st.caption(
        f"{perimeter_text(state, section)} · Dixon-Coles su {state.n_matches} partite (stima congiunta di tutto il perimetro) · "
        f"{len(section_predictions(state, section))} previsioni · soglia EV {min_ev:.1%} · Kelly x{kelly_fraction:g} · "
        f"bankroll {bankroll:g} unità · sezione: {section} · vista: {scope}"
    )
    with st.expander("Dettagli pipeline", expanded=False):
        notes = list(state.notes)
        if serie_d_gironi():
            notes.append(f"Sezione Serie D: {state.serie_d_status}.")
        st.markdown("\n".join(f"- {note}" for note in notes))
        st.caption(f"Stato calcolato alle {state.built_at.astimezone().strftime('%d/%m/%Y %H:%M:%S')} (fonte: {state.source}).")

    matchday = selected_matchday(state, section)  # before any tab renders, so the three matchday-aware tabs agree
    tab_rank, tab_match, tab_results, tab_value, tab_picks, tab_export, tab_legend = st.tabs(list(TAB_LABELS))
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
    with tab_export:
        render_export_tab(state, section)
    with tab_legend:
        render_legend_tab(state, girone, min_ev, matchday, kelly_fraction, bankroll, section)


main()
