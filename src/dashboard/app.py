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
    MAX_HISTORY_SEASONS,
    MIN_SIM_ROUNDS,
    Matchday,
    PipelineState,
    btts_picks_table,
    build_state,
    draw_picks_table,
    export_csv_bytes,
    find_matchday,
    fixtures_for,
    heatmap_figure,
    markets_table,
    no_odds_message,
    opportunities_table,
    picks_csv_bytes,
    predictions_table,
    quotes_for,
    ranking_table,
    strength_figure,
    traps_table,
    value_opportunities,
    value_table,
)
from src.engine.picks import DEFAULT_N_BTTS, DEFAULT_N_DRAWS

log = get_logger(__name__)

#: Competition selector: *Tutti* plus the short label of every configured competition
#: (``'Girone A'``, ``'Girone B'``, ``'Girone C'``, ``'Serie B'`` with the shipped ``config/competitions.toml``).
GIRONE_OPTIONS: Final[tuple[str, ...]] = ("Tutti", *(g.short for g in Girone))
#: Competitions named in the sidebar caption, e.g. ``'Serie C Gironi A/B/C + Serie B'``.
COMPETITIONS_TEXT: Final[str] = describe_competitions()
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
#: ``st.session_state`` key holding the selected matchday's :attr:`Matchday.key`, shared by the Match, Value and Picks tabs.
MATCHDAY_STATE_KEY: Final[str] = "matchday"
#: ``st.session_state`` key of the value panel's "every matchday" checkbox, read again by the Legenda tab.
VALUE_ALL_MATCHDAYS_KEY: Final[str] = "value_all_matchdays"
#: Largest list size the picks tab accepts (a matchday never has more fixtures than this).
MAX_PICKS: Final[int] = 60
#: Tab labels, in order.
TAB_LABELS: Final[tuple[str, ...]] = (
    "Classifica Forza",
    "Match & Heatmap",
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


def selected_matchday(state: PipelineState) -> Matchday | None:
    """The matchday the three matchday-aware tabs agree on, read from (and written to) ``st.session_state``.

    The *Selezioni di giornata* tab owns the ``Giornata`` selectbox (key
    :data:`MATCHDAY_STATE_KEY`), but *Match & Heatmap* and *Opportunità di
    Valore* are rendered before it in the same script run, so the choice is
    read from the session state here, once, before any tab renders: a missing
    or stale key (the state was recomputed and that matchday no longer exists)
    is replaced by the next matchday's key, which also seeds the selectbox
    without passing it an ``index``. ``None`` when the state has no matchday.
    """
    if not state.matchdays:
        st.session_state.pop(MATCHDAY_STATE_KEY, None)
        return None
    chosen = find_matchday(state, st.session_state.get(MATCHDAY_STATE_KEY))
    if chosen is None:
        chosen = state.matchdays[0]
        st.session_state[MATCHDAY_STATE_KEY] = chosen.key
    return chosen


# =============================================================================
# Sidebar
# =============================================================================
def render_sidebar() -> tuple[Girone | None, float, float, float]:
    """Controls; returns ``(girone, min_ev_fraction, kelly_fraction, bankroll_units)``."""
    with st.sidebar:
        st.title("⚽ Serie C Quant Lab")
        st.caption(f"Modello Dixon-Coles per {COMPETITIONS_TEXT} e ricerca di valore atteso (+EV) contro le quote dei bookmaker.")
        girone_label = st.selectbox("Competizione", GIRONE_OPTIONS, index=0, key="girone", help="Filtra classifica, partite e opportunità per competizione (girone di Serie C o Serie B).")
        ev_pct = st.slider("Soglia EV minima (%)", min_value=0.0, max_value=15.0, value=3.5, step=0.5, key="min_ev_pct", help="Una selezione è di valore quando P_modello × Quota − 1 supera questa soglia.")
        kelly_label = st.radio("Frazione di Kelly", list(KELLY_OPTIONS), index=0, horizontal=True, key="kelly", help="Quota del Kelly pieno da puntare.")
        bankroll = st.number_input("Bankroll (unità)", min_value=1.0, max_value=1_000_000.0, value=100.0, step=10.0, key="bankroll", help="100 unità = ogni unità vale l'1% del bankroll.")
        if st.button("Ricalcola", key="recompute", help="Svuota la cache e riesegue la pipeline.", **FULL_WIDTH["button"]):
            st.cache_resource.clear()
            st.rerun()
    return selected_girone(girone_label), float(ev_pct) / 100.0, KELLY_OPTIONS[kelly_label], float(bankroll)


def render_source_badge(container: st.delta_generator.DeltaGenerator, state: PipelineState) -> None:
    """Data-source badge: perimeter (season, teams, current vs historical matches) and fit diagnostics."""
    params = state.params
    body = (
        f"**{state.perimeter_text}**  \n"
        f"{state.n_matches} partite nel fit · **{state.n_fixtures}** partite in programma · "
        f"{state.n_historical_teams} squadre storiche escluse  \n"
        f"Fit: {params.method} · LL {params.log_likelihood:.1f}  \n"
        f"Convergenza: {'sì' if params.converged else 'no'} · {params.fit_seconds:.2f} s  \n"
        f"Prior gaussiana: {'attiva' if state.prior_enabled else 'disattivata'}"
    )
    if state.coverage is not None:
        body += f"  \nCopertura {state.season_label}: {state.coverage.thinnest:.1f} partite per squadra nella competizione meno coperta"
        if state.coverage.ignored:
            body += f" · {state.coverage.describe_ignored()}"
    if state.source == "processed":
        container.success(f"**Dati reali** (data/processed)  \n{body}", icon="✅")
    else:
        container.info(f"**Dati simulati** (generatore sintetico)  \n{body}", icon="🧪")


# =============================================================================
# Tabs
# =============================================================================
def render_ranking_tab(state: PipelineState, girone: Girone | None) -> None:
    st.subheader("Classifica di forza (parametri Dixon-Coles)")
    st.caption(
        "α = forza offensiva, β = solidità difensiva (più alto = subisce meno), γ = vantaggio casa della competizione; indici = exp(parametro). "
        f"Solo le squadre del perimetro {state.season_label} ({state.perimeter_label}) nella competizione del dizionario corrente; "
        f"le squadre delle stagioni passate ({state.n_historical_teams}) entrano nella stima solo come avversarie. "
        "α e β sono centrati per competizione: confronta le squadre dentro la stessa competizione."
    )
    table = ranking_table(state, girone)
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
        columns = st.columns(len(state.gironi) or 1)
        for col, g in zip(columns, state.gironi):
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


def render_match_tab(state: PipelineState, girone: Girone | None, matchday: Matchday | None) -> None:
    st.subheader("Partita e matrice dei punteggi")
    preds = fixtures_for(state, girone, matchday=matchday)
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
    if matchday is not None:
        st.caption(
            f"{matchday.label} · {len(preds)} partite"
            + (" (tutte le competizioni)" if girone is None else f" ({girone.short})")
            + f" · la giornata si cambia nel tab *Selezioni di giornata* ({len(state.matchdays)} in programma)."
        )
    labels = [f"[{p.fixture.girone.value}] {p.fixture.date.isoformat()} · {p.fixture.label}" for p in preds]
    choice = st.selectbox("Partita della giornata", labels, index=0, key="fixture")
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


def render_value_tab(state: PipelineState, girone: Girone | None, min_ev: float, matchday: Matchday | None) -> None:
    st.subheader("Opportunità di valore")
    if not state.quotes:
        st.info(no_odds_message(state))
    show_all = False
    if matchday is not None:
        show_all = bool(
            st.checkbox(
                "Mostra tutte le giornate",
                value=False,
                key=VALUE_ALL_MATCHDAYS_KEY,
                help="Di default il pannello mostra solo la giornata selezionata nel tab Selezioni di giornata.",
            )
        )
        scope = "Tutte le partite in programma" if show_all else matchday.label
        st.caption(f"{scope} · la giornata si cambia nel tab *Selezioni di giornata* ({len(state.matchdays)} in programma).")
    table = value_table(state, min_ev, girone, matchday=ALL_MATCHDAYS if show_all else matchday)
    if table.empty:
        if state.quotes:
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
    st.subheader("Alert REGRESSION_TRAP")
    st.caption("Squadre con pochi gol subiti nelle ultime 3 gare ma xGA elevato: la difesa è in sovra-rendimento e il mercato sottoprezza i gol contro di loro.")
    if state.traps:
        for trap in state.traps:
            st.warning(trap.description(), icon="⚠️")
        st.dataframe(
            traps_table(state),
            hide_index=True,
            **FULL_WIDTH["dataframe"],
            column_config={
                "Competizione": st.column_config.TextColumn("Competizione", width="small"),
                "xGA": st.column_config.NumberColumn("xGA", format="%.2f"),
                "Severità": st.column_config.NumberColumn("Severità", format="%+.2f"),
                "Ultima Gara": st.column_config.DateColumn("Ultima Gara", format="DD/MM/YYYY"),
            },
        )
        flagged = sum(1 for o in state.opportunities if o.is_trap)
        st.caption(f"Selezioni Over/GG segnalate dal filtro: {flagged}")
    else:
        st.success("Nessuna squadra in trappola di regressione nella finestra corrente.", icon="✅")


def render_picks_tab(state: PipelineState, matchday: Matchday | None) -> None:
    """*Selezioni di giornata*: the matchday selector and the top-N draw / both-teams-to-score lists."""
    st.subheader("Selezioni di giornata")
    if matchday is None or not state.matchdays:
        message = "Nessuna partita in programma: nessuna giornata da selezionare."
        if state.skipped_fixtures:
            message += f" {len(state.skipped_fixtures)} partite caricate sono state saltate (dettagli in *Dettagli pipeline*)."
        st.info(message)
        return
    keys = [m.key for m in state.matchdays]
    labels = {m.key: m.label for m in state.matchdays}
    c_day, c_draws, c_btts = st.columns([3, 1, 1])
    with c_day:
        chosen_key = st.selectbox(
            "Giornata",
            keys,
            format_func=lambda k: labels.get(k, k),
            key=MATCHDAY_STATE_KEY,
            help="I turni delle competizioni giocati nello stesso blocco della settimana (fine settimana ven-lun o infrasettimanale mar-gio), con i recuperi aggregati alla giornata più vicina; vale anche per Match & Heatmap e Opportunità di Valore.",
        )
    with c_draws:
        n_draws = int(st.number_input("N risultati X", min_value=1, max_value=MAX_PICKS, value=DEFAULT_N_DRAWS, step=1, key="n_draws"))
    with c_btts:
        n_btts = int(st.number_input("N partite GG", min_value=1, max_value=MAX_PICKS, value=DEFAULT_N_BTTS, step=1, key="n_btts"))
    chosen = find_matchday(state, str(chosen_key)) or matchday
    st.caption(
        f"{chosen.label} · {chosen.n_fixtures} partite · {chosen.start.strftime('%d/%m/%Y')}"
        + (f" - {chosen.end.strftime('%d/%m/%Y')}" if chosen.end != chosen.start else "")
        + ". Le liste considerano tutte le competizioni, qualunque sia il filtro della sidebar."
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
    draws = draw_picks_table(state, chosen, n=n_draws)
    btts = btts_picks_table(state, chosen, n=n_btts)

    st.markdown(f"**Le {len(draws)} partite più probabili da X (tutte le competizioni)**")
    if draws.empty:
        st.info("Nessuna partita prevista nella giornata selezionata.")
    else:
        st.dataframe(draws, hide_index=True, **FULL_WIDTH["dataframe"], column_config=column_config, key="picks_draws")
    st.caption(
        "Ordinate solo per probabilità di pareggio del modello, non per valore: Quota Fair = 1/P; Quota Book ed EV "
        "dicono se il prezzo del bookmaker vale la puntata (EV > 0 = quota sopra la fair). Vuote quando nessun book quota la X."
    )

    st.markdown(f"**Le {len(btts)} partite più probabili da GG (tutte le competizioni)**")
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
        file_name=f"selezioni_{chosen.key.replace('#', '_')}.csv",
        mime="text/csv",
        key="dl_picks",
        help="Un solo CSV: colonna Lista = X oppure GG, poi le colonne delle tabelle.",
        **FULL_WIDTH["download_button"],
    )


def value_scope_text(girone: Girone | None, matchday: Matchday | None, show_all: bool) -> str:
    """``'Giornata del 7-9 nov 2026 · ..., Girone A'``: the value panel's current scope, for captions."""
    days = "tutte le partite in programma" if show_all or matchday is None else matchday.label
    return f"{days}, {'tutte le competizioni' if girone is None else girone.short}"


def render_legend_tab(
    state: PipelineState,
    girone: Girone | None,
    min_ev: float,
    matchday: Matchday | None,
    kelly_fraction: float,
    bankroll: float,
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
    show_all = bool(st.session_state.get(VALUE_ALL_MATCHDAYS_KEY, False))
    rows = value_opportunities(state, min_ev, girone, matchday=ALL_MATCHDAYS if show_all else matchday)
    chosen = pick_example_row(rows)
    scope = value_scope_text(girone, matchday, show_all)
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


def render_export_tab(state: PipelineState) -> None:
    st.subheader("Export CSV")
    today = datetime.now(UTC).astimezone().date().isoformat()
    value_frame = opportunities_table(state)
    preds_frame = predictions_table(state)
    c1, c2 = st.columns(2)
    with c1:
        st.markdown(f"**Report valore** · {len(value_frame)} selezioni valutate (EV, edge, Kelly, flag)")
        st.download_button(
            "Scarica report valore (CSV)",
            data=export_csv_bytes(value_frame),
            file_name=f"serie_c_value_report_{today}.csv",
            mime="text/csv",
            key="dl_value",
            **FULL_WIDTH["download_button"],
        )
    with c2:
        st.markdown(f"**Previsioni complete** · {len(preds_frame)} partite (λ, μ, probabilità e quote fair di ogni mercato)")
        st.download_button(
            "Scarica previsioni (CSV)",
            data=export_csv_bytes(preds_frame),
            file_name=f"serie_c_previsioni_{today}.csv",
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
    """Compose the page."""
    girone, min_ev, kelly_fraction, bankroll = render_sidebar()
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
    render_source_badge(badge, state)

    st.title("⚽ Serie C Quant Lab")
    scope = "tutte le competizioni" if girone is None else girone.short
    st.caption(
        f"{state.perimeter_text} · Dixon-Coles su {state.n_matches} partite · {len(state.predictions)} previsioni · "
        f"soglia EV {min_ev:.1%} · Kelly x{kelly_fraction:g} · bankroll {bankroll:g} unità · vista: {scope}"
    )
    with st.expander("Dettagli pipeline", expanded=False):
        st.markdown("\n".join(f"- {note}" for note in state.notes))
        st.caption(f"Stato calcolato alle {state.built_at.astimezone().strftime('%d/%m/%Y %H:%M:%S')} (fonte: {state.source}).")

    matchday = selected_matchday(state)  # before any tab renders, so the three matchday-aware tabs agree
    tab_rank, tab_match, tab_value, tab_picks, tab_export, tab_legend = st.tabs(list(TAB_LABELS))
    with tab_rank:
        render_ranking_tab(state, girone)
    with tab_match:
        render_match_tab(state, girone, matchday)
    with tab_value:
        render_value_tab(state, girone, min_ev, matchday)
    with tab_picks:
        render_picks_tab(state, matchday)
    with tab_export:
        render_export_tab(state)
    with tab_legend:
        render_legend_tab(state, girone, min_ev, matchday, kelly_fraction, bankroll)


main()
