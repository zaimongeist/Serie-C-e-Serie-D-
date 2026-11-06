"""Serie C Quant Lab -- Streamlit dashboard (UI in Italian).

Run from the repository root::

    streamlit run src/dashboard/app.py

Environment variables
---------------------
``SERIEC_FORCE_SIMULATE=1``
    Always use the synthetic league, even when ``data/processed/matches.csv`` exists.
``SERIEC_SIM_ROUNDS`` (int >= 1, default 15)
    Played rounds of the simulated league; values above the season length are
    clamped by the data layer so one round of fixtures always remains.
``SERIEC_SEED`` (int >= 0, default 7)
    Random seed of the simulated league.

Invalid or out-of-range values (non-integers, a negative seed, zero rounds)
are logged and replaced by the defaults instead of crashing the page.

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
from datetime import UTC, datetime
from pathlib import Path
from typing import Final

# -- bootstrap: make ``src.*`` importable when launched as ``streamlit run src/dashboard/app.py``
_REPO_ROOT: Final[Path] = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import streamlit as st

st.set_page_config(page_title="Serie C Quant Lab", page_icon="⚽", layout="wide")

from src.common.logging_config import get_logger
from src.common.schemas import Girone, Market, MatchPrediction
from src.dashboard import container_width_kwargs
from src.dashboard.state import (
    MIN_SIM_ROUNDS,
    PipelineState,
    build_state,
    export_csv_bytes,
    fixtures_for,
    heatmap_figure,
    markets_table,
    no_odds_message,
    opportunities_table,
    predictions_table,
    quotes_for,
    ranking_table,
    strength_figure,
    traps_table,
    value_table,
)

log = get_logger(__name__)

GIRONE_OPTIONS: Final[tuple[str, ...]] = ("Tutti", "A", "B", "C")
KELLY_OPTIONS: Final[dict[str, float]] = {"1/8": 0.125, "1/4": 0.25}
DEFAULT_SIM_ROUNDS: Final[int] = 15
DEFAULT_SEED: Final[int] = 7
#: Smallest "goals per team" the heatmap slider allows (a compact view folds the tail into an ``N+`` cell).
HEATMAP_MIN_GOALS: Final[int] = 3
#: Full-width keyword per Streamlit command, resolved once for the installed release.
FULL_WIDTH: Final[dict[str, dict[str, object]]] = {
    name: container_width_kwargs(getattr(st, name)) for name in ("button", "download_button", "dataframe", "plotly_chart")
}


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
    n_rounds: int,
) -> PipelineState:
    """Build (and cache) the pipeline state for one combination of sidebar parameters."""
    return build_state(
        min_ev=min_ev,
        kelly_fraction=kelly_fraction,
        bankroll_units=bankroll_units,
        simulate=simulate,
        seed=seed,
        n_rounds=n_rounds,
    )


def selected_girone(label: str) -> Girone | None:
    """Map the sidebar label to a :class:`Girone` (``None`` for *Tutti*)."""
    return None if label == "Tutti" else Girone.parse(label)


# =============================================================================
# Sidebar
# =============================================================================
def render_sidebar() -> tuple[Girone | None, float, float, float]:
    """Controls; returns ``(girone, min_ev_fraction, kelly_fraction, bankroll_units)``."""
    with st.sidebar:
        st.title("⚽ Serie C Quant Lab")
        st.caption("Modello Dixon-Coles per i Gironi A/B/C e ricerca di valore atteso (+EV) contro le quote dei bookmaker.")
        girone_label = st.selectbox("Girone", GIRONE_OPTIONS, index=0, key="girone", help="Filtra classifica, partite e opportunità.")
        ev_pct = st.slider("Soglia EV minima (%)", min_value=0.0, max_value=15.0, value=3.5, step=0.5, key="min_ev_pct", help="Una selezione è di valore quando P_modello × Quota − 1 supera questa soglia.")
        kelly_label = st.radio("Frazione di Kelly", list(KELLY_OPTIONS), index=0, horizontal=True, key="kelly", help="Quota del Kelly pieno da puntare.")
        bankroll = st.number_input("Bankroll (unità)", min_value=1.0, max_value=1_000_000.0, value=100.0, step=10.0, key="bankroll", help="100 unità = ogni unità vale l'1% del bankroll.")
        if st.button("Ricalcola", key="recompute", help="Svuota la cache e riesegue la pipeline.", **FULL_WIDTH["button"]):
            st.cache_resource.clear()
            st.rerun()
    return selected_girone(girone_label), float(ev_pct) / 100.0, KELLY_OPTIONS[kelly_label], float(bankroll)


def render_source_badge(container: st.delta_generator.DeltaGenerator, state: PipelineState) -> None:
    """Data-source badge with the fit diagnostics."""
    params = state.params
    body = (
        f"**{state.n_matches}** partite · **{state.n_fixtures}** partite in programma  \n"
        f"Fit: {params.method} · LL {params.log_likelihood:.1f}  \n"
        f"Convergenza: {'sì' if params.converged else 'no'} · {params.fit_seconds:.2f} s  \n"
        f"Prior gaussiana: {'attiva' if state.prior_enabled else 'disattivata'}"
    )
    if state.source == "processed":
        container.success(f"**Dati reali** (data/processed)  \n{body}", icon="✅")
    else:
        container.info(f"**Dati simulati** (generatore sintetico)  \n{body}", icon="🧪")


# =============================================================================
# Tabs
# =============================================================================
def render_ranking_tab(state: PipelineState, girone: Girone | None) -> None:
    st.subheader("Classifica di forza (parametri Dixon-Coles)")
    st.caption("α = forza offensiva, β = solidità difensiva (più alto = subisce meno), γ = vantaggio casa del girone; indici = exp(parametro).")
    table = ranking_table(state, girone)
    if table.empty:
        st.info("Nessuna squadra stimata per il girone selezionato.")
        return
    st.dataframe(
        table,
        hide_index=True,
        **FULL_WIDTH["dataframe"],
        column_config={
            "Squadra": st.column_config.TextColumn("Squadra"),
            "Girone": st.column_config.TextColumn("Girone", width="small"),
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
                st.plotly_chart(strength_figure(ranking_table(state, g), title=f"Girone {g.value}"), **FULL_WIDTH["plotly_chart"], key=f"strength_{g.value}")
    else:
        st.plotly_chart(strength_figure(table, title=f"Girone {girone.value}: α e β per squadra"), **FULL_WIDTH["plotly_chart"], key=f"strength_{girone.value}")


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


def render_match_tab(state: PipelineState, girone: Girone | None) -> None:
    st.subheader("Partita e matrice dei punteggi")
    preds = fixtures_for(state, girone)
    if not preds:
        st.info("Nessuna partita in programma per il girone selezionato.")
        return
    labels = [f"[{p.fixture.girone.value}] {p.fixture.date.isoformat()} · {p.fixture.label}" for p in preds]
    choice = st.selectbox("Partita del prossimo turno", labels, index=0, key="fixture")
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


def render_value_tab(state: PipelineState, girone: Girone | None, min_ev: float) -> None:
    st.subheader("Opportunità di valore")
    if not state.quotes:
        st.info(no_odds_message(state))
    table = value_table(state, min_ev, girone)
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
                "xGA": st.column_config.NumberColumn("xGA", format="%.2f"),
                "Severità": st.column_config.NumberColumn("Severità", format="%+.2f"),
                "Ultima Gara": st.column_config.DateColumn("Ultima Gara", format="DD/MM/YYYY"),
            },
        )
        flagged = sum(1 for o in state.opportunities if o.is_trap)
        st.caption(f"Selezioni Over/GG segnalate dal filtro: {flagged}")
    else:
        st.success("Nessuna squadra in trappola di regressione nella finestra corrente.", icon="✅")


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
    n_rounds = env_int("SERIEC_SIM_ROUNDS", DEFAULT_SIM_ROUNDS, minimum=MIN_SIM_ROUNDS)
    state = load_state(
        min_ev=min_ev,
        kelly_fraction=kelly_fraction,
        bankroll_units=bankroll,
        simulate=simulate,
        seed=seed,
        n_rounds=n_rounds,
    )
    render_source_badge(badge, state)

    st.title("⚽ Serie C Quant Lab")
    scope = "tutti i gironi" if girone is None else f"Girone {girone.value}"
    st.caption(
        f"Dixon-Coles su {state.n_matches} partite · {len(state.predictions)} previsioni · soglia EV {min_ev:.1%} · "
        f"Kelly x{kelly_fraction:g} · bankroll {bankroll:g} unità · vista: {scope}"
    )
    with st.expander("Dettagli pipeline", expanded=False):
        st.markdown("\n".join(f"- {note}" for note in state.notes))
        st.caption(f"Stato calcolato alle {state.built_at.astimezone().strftime('%d/%m/%Y %H:%M:%S')} (fonte: {state.source}).")

    tab_rank, tab_match, tab_value, tab_export = st.tabs(["Classifica Forza", "Match & Heatmap", "Opportunità di Valore", "Export"])
    with tab_rank:
        render_ranking_tab(state, girone)
    with tab_match:
        render_match_tab(state, girone)
    with tab_value:
        render_value_tab(state, girone, min_ev)
    with tab_export:
        render_export_tab(state)


main()
