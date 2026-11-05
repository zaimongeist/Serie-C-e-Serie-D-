"""Pure data layer of the Streamlit dashboard (no ``streamlit`` import).

Everything the UI shows is computed here from the shared contracts, so the
dashboard can be unit-tested without a browser:

* :func:`build_state` runs the whole pipeline (load or simulate -> fit ->
  predict -> value) and returns an immutable-by-convention
  :class:`PipelineState` snapshot;
* the ``*_table`` helpers turn that snapshot into ``pandas`` frames with
  Italian column labels, ready for ``st.dataframe``;
* :func:`heatmap_figure` builds the score-matrix Plotly figure;
* :func:`export_csv_bytes` serialises any frame for ``st.download_button``.

Bookmaker odds are read from ``odds_dir/*.csv`` in the same format
``main.load_odds_dir`` understands, see :func:`load_odds_quotes`.
"""
from __future__ import annotations

import math
import time
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Final, Literal

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from src.common.logging_config import get_logger
from src.common.paths import ODDS_DIR, PROCESSED_DIR
from src.common.schemas import SELECTIONS, Fixture, Girone, Market, MatchPrediction, OddsQuote
from src.engine.model import DixonColesConfig, DixonColesEngine, FittedParams
from src.engine.value import (
    KellyConfig,
    RegressionTrap,
    ValueEngine,
    ValueOpportunity,
    detect_regression_traps,
)
from src.ingestion.harvester import load_fixtures, load_processed
from src.simulation.generator import inject_regression_trap, simulate_league

__all__ = [
    "MARKETS_COLUMNS",
    "MARKET_LABELS",
    "ODDS_COLUMNS",
    "PRIOR_THRESHOLD_MATCHES",
    "RANKING_COLUMNS",
    "TRAPS_COLUMNS",
    "VALUE_COLUMNS",
    "PipelineState",
    "Source",
    "build_state",
    "export_csv_bytes",
    "fixtures_for",
    "heatmap_figure",
    "load_odds_quotes",
    "markets_table",
    "opportunities_table",
    "predictions_table",
    "quotes_for",
    "ranking_table",
    "strength_figure",
    "traps_table",
    "value_table",
]

log = get_logger(__name__)

Source = Literal["processed", "simulated"]

#: Below this many played matches the Gaussian (MAP) prior is switched on automatically.
PRIOR_THRESHOLD_MATCHES: Final[int] = 100

#: Required columns of a bookmaker odds CSV (one row per quoted selection).
ODDS_COLUMNS: Final[tuple[str, ...]] = ("match_id", "bookmaker", "market", "selection", "odds")

#: Italian labels of :func:`ranking_table`, in order.
RANKING_COLUMNS: Final[tuple[str, ...]] = (
    "Squadra",
    "Girone",
    "Forza Offensiva (α)",
    "Solidità Difensiva (β)",
    "Vantaggio Casa (γ)",
    "Indice Attacco",
    "Indice Difesa",
)
_RANKING_RENAME: Final[dict[str, str]] = {
    "team": "Squadra",
    "girone": "Girone",
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
    "Girone",
    "Partita",
    "Mercato",
    "Esito",
    "Quota Book",
    "Quota Fair",
    "P. Modello (%)",
    "EV (%)",
    "Stake (unità)",
    "Flag",
)

#: Italian labels of :func:`traps_table`, in order.
TRAPS_COLUMNS: Final[tuple[str, ...]] = ("Squadra", "Girone", "Gol Subiti", "xGA", "Severità", "Ultima Gara")

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

    @property
    def n_matches(self) -> int:
        """Number of played matches used by the fit."""
        return len(self.matches)

    @property
    def n_fixtures(self) -> int:
        """Number of upcoming fixtures loaded (predicted or not)."""
        return len(self.fixtures)

    @property
    def gironi(self) -> tuple[Girone, ...]:
        """Gironi present in the fitted parameters, in A/B/C order."""
        return tuple(g for g in Girone if g in self.params.teams_by_girone)

    @property
    def prior_enabled(self) -> bool:
        """Whether the fit used the Gaussian (MAP) prior."""
        cfg = self.engine.config
        return bool(cfg.l2_penalty > 0.0 or cfg.l2_penalty_home > 0.0 or cfg.l2_penalty_rho > 0.0)


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


def _model_config(n_matches: int, *, xi: float, use_prior: bool | None) -> tuple[DixonColesConfig, bool]:
    """Resolve the ``use_prior`` tri-state into a config; returns ``(config, prior_on)``."""
    prior_on = (n_matches < PRIOR_THRESHOLD_MATCHES) if use_prior is None else bool(use_prior)
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


def _simulated_inputs(seed: int, n_rounds: int, notes: list[str]) -> tuple[pd.DataFrame, list[Fixture], list[OddsQuote]]:
    """Synthetic league with a REGRESSION_TRAP injected on a Girone A team of the next round."""
    league = simulate_league(seed=seed, n_rounds=n_rounds)
    matches = league.matches
    notes.append(
        f"Dataset simulato (seed={seed}, {n_rounds} giornate giocate): {len(matches)} partite, "
        f"{len(league.fixtures)} partite in programma, {len(league.odds)} quote del bookmaker sintetico."
    )
    trap_team = _pick_trap_team(matches, league.fixtures, Girone.A)
    if trap_team is not None:
        matches = inject_regression_trap(matches, trap_team)
        notes.append(
            f"Trappola di regressione iniettata su {trap_team} (Girone A): ultime 3 gare riscritte con "
            f"1 gol subito contro 4.20 xGA, così il filtro REGRESSION_TRAP compare nel pannello valore."
        )
        log.info("regression trap injected", extra={"team": trap_team})
    else:
        notes.append("Nessuna squadra del Girone A con 3 gare giocate e una partita in programma: trappola non iniettata.")
    return matches, league.fixtures, league.odds


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
    n_rounds: int = 15,
    simulate: bool | None = None,
    use_prior: bool | None = None,
) -> PipelineState:
    """Run load/simulate -> fit -> predict -> value and return a :class:`PipelineState`.

    Data source
        ``processed_dir/matches.csv`` is used when it exists and ``simulate`` is
        not ``True`` (``fixtures.csv`` alongside it when present, odds from
        ``odds_dir/*.csv`` in the :func:`load_odds_quotes` format). Otherwise
        a league is simulated with :func:`simulate_league` (``seed``,
        ``n_rounds`` played rounds, the next round as fixtures) and a
        REGRESSION_TRAP is injected on one Girone A team of the next round.
    Model
        ``use_prior=None`` enables the Gaussian prior when fewer than
        :data:`PRIOR_THRESHOLD_MATCHES` matches are available; ``True`` forces
        :meth:`DixonColesConfig.with_gaussian_prior`, ``False`` the plain MLE.
        ``xi`` is the time-decay rate per day.
    Value
        ``ValueEngine(min_ev, KellyConfig(kelly_fraction, bankroll_units))`` is
        run on the predictions and the quotes; without quotes the
        opportunities simply stay empty. Fixtures with teams unknown to the fit
        are skipped with a note instead of raising.
    """
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
        matches, fixtures, quotes = loaded
    else:
        source = "simulated"
        matches, fixtures, quotes = _simulated_inputs(seed, n_rounds, notes)

    config, prior_on = _model_config(len(matches), xi=xi, use_prior=use_prior)
    engine = DixonColesEngine(config)
    params = engine.fit(matches)
    notes.append(
        f"Stima Dixon-Coles: metodo {params.method}, convergenza {'sì' if params.converged else 'no'} "
        f"({params.message}), log-likelihood {params.log_likelihood:.2f}, {params.n_iter} iterazioni in "
        f"{params.fit_seconds:.2f} s, xi={params.xi}, prior gaussiana {'attiva' if prior_on else 'disattivata'}"
        + (f" (penalità {params.penalty:.3f})" if prior_on else "")
        + f", rho={params.rho:+.4f}."
    )
    log.info(
        "dashboard fit",
        extra={
            "source": source,
            "n_matches": params.n_matches,
            "method": params.method,
            "converged": params.converged,
            "log_likelihood": params.log_likelihood,
            "fit_seconds": params.fit_seconds,
            "prior": prior_on,
        },
    )

    predictions: list[MatchPrediction] = []
    skipped: list[str] = []
    for fx in fixtures:
        try:
            predictions.append(engine.predict(fx))
        except (KeyError, ValueError) as exc:
            skipped.append(fx.label)
            log.warning("fixture not predictable", extra={"match_id": fx.match_id, "error": str(exc)})
    if skipped:
        notes.append(f"{len(skipped)} partite in programma saltate (squadre non presenti nello storico): {', '.join(skipped[:5])}" + (" ..." if len(skipped) > 5 else "") + ".")

    value_engine = ValueEngine(min_ev=min_ev, kelly=KellyConfig(fraction=kelly_fraction, bankroll_units=bankroll_units))
    traps = detect_regression_traps(
        matches,
        window=value_engine.trap_window,
        max_goals_against=value_engine.trap_max_goals_against,
        min_xg_against=value_engine.trap_min_xg_against,
    )
    opportunities = value_engine.evaluate(predictions, quotes, matches=matches, traps=traps)
    n_value = sum(1 for o in opportunities if o.value)
    n_trap_flags = sum(1 for o in opportunities if o.is_trap)
    notes.append(
        f"Motore valore: {len(opportunities)} selezioni valutate, {n_value} sopra soglia EV {min_ev:.1%}, "
        f"Kelly x{kelly_fraction:g} su {bankroll_units:g} unità, {len(traps)} squadre in trappola di regressione, "
        f"{n_trap_flags} selezioni Over/GG segnalate."
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
    )


# =============================================================================
# Views
# =============================================================================
def _as_girone(girone: Girone | str | None) -> Girone | None:
    return None if girone is None else Girone.parse(girone)


def ranking_table(state: PipelineState, girone: Girone | None) -> pd.DataFrame:
    """Strength ranking (alpha desc) with Italian labels, rounded to 3 decimals.

    ``girone=None`` returns every fitted team; a girone with no fitted teams
    yields an empty frame with the same columns instead of raising.
    """
    g = _as_girone(girone)
    if g is not None and g not in state.params.teams_by_girone:
        return pd.DataFrame({c: pd.Series(dtype="string" if c in ("Squadra", "Girone") else "float64") for c in RANKING_COLUMNS})
    frame = state.engine.ranking(g)[list(_RANKING_RENAME)].rename(columns=_RANKING_RENAME)
    numeric = [c for c in RANKING_COLUMNS if c not in ("Squadra", "Girone")]
    frame = frame.assign(**{c: frame[c].astype("float64").round(3) for c in numeric})
    frame = frame.assign(Squadra=frame["Squadra"].astype("string"), Girone=frame["Girone"].astype("string"))
    return frame[list(RANKING_COLUMNS)].reset_index(drop=True)


def fixtures_for(state: PipelineState, girone: Girone | None) -> list[MatchPrediction]:
    """Predictions of the upcoming fixtures (optionally one girone), sorted by date then label."""
    g = _as_girone(girone)
    preds = [p for p in state.predictions if g is None or p.fixture.girone == g]
    return sorted(preds, key=lambda p: (p.fixture.date, p.fixture.label))


def quotes_for(state: PipelineState, match_id: str) -> list[OddsQuote]:
    """Every bookmaker quote of one fixture."""
    return [q for q in state.quotes if q.match_id == match_id]


def heatmap_figure(prediction: MatchPrediction, max_goals_shown: int = 6) -> go.Figure:
    """Score-matrix heatmap: home goals on the y axis, away goals on the x axis.

    Exactly one :class:`plotly.graph_objects.Heatmap` trace holding
    ``matrix[:max_goals_shown + 1, :max_goals_shown + 1] * 100`` (percentages),
    annotated with one decimal, single-hue ``Blues`` colourscale, the most
    likely score outlined with a rectangle. The ``plotly`` template is used so
    Streamlit can re-theme it for light and dark modes; fixed height 520.
    """
    if max_goals_shown < 0:
        raise ValueError("max_goals_shown must be >= 0")
    fx = prediction.fixture
    n = min(max_goals_shown + 1, prediction.matrix.shape[0])
    z = np.asarray(prediction.matrix[:n, :n], dtype=float) * 100.0
    goals = list(range(n))
    text = [[f"{z[i, j]:.1f}%" for j in goals] for i in goals]
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
        hovertemplate=f"{fx.home_team} %{{y}} - %{{x}} {fx.away_team}<br>P = %{{z:.2f}}%<extra></extra>",
        xgap=2,
        ygap=2,
    )
    fig = go.Figure(data=[heat])
    home_goals, away_goals, p_top = prediction.most_likely_score()
    if home_goals < n and away_goals < n:
        fig.add_shape(
            type="rect",
            x0=away_goals - 0.5,
            x1=away_goals + 0.5,
            y0=home_goals - 0.5,
            y1=home_goals + 0.5,
            line={"color": _HIGHLIGHT_COLOR, "width": 3},
            fillcolor="rgba(0,0,0,0)",
        )
    fig.update_layout(
        template="plotly",
        title={
            "text": (
                f"{fx.label} · Girone {fx.girone.value} · {fx.date.isoformat()}<br>"
                f"<sup>λ casa = {prediction.home_rate:.2f} · μ ospiti = {prediction.away_rate:.2f} · "
                f"risultato più probabile {home_goals}-{away_goals} ({p_top:.1%})</sup>"
            )
        },
        xaxis={"title": {"text": f"Gol {fx.away_team}"}, "tickmode": "array", "tickvals": goals, "side": "top", "showgrid": False},
        yaxis={"title": {"text": f"Gol {fx.home_team}"}, "tickmode": "array", "tickvals": goals, "autorange": "reversed", "showgrid": False},
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


def value_table(state: PipelineState, min_ev: float, girone: Girone | None = None) -> pd.DataFrame:
    """Selections with ``ev >= min_ev`` (optionally one girone), EV descending, Italian labels."""
    g = _as_girone(girone)
    rows = [
        {
            "Data": pd.Timestamp(o.date),
            "Girone": o.girone.value,
            "Partita": f"{o.home_team} - {o.away_team}",
            "Mercato": MARKET_LABELS[o.market],
            "Esito": o.selection,
            "Quota Book": round(o.book_odds, 2),
            "Quota Fair": round(o.fair_odds, 2),
            "P. Modello (%)": round(o.p_model * 100.0, 1),
            "EV (%)": round(o.ev * 100.0, 2),
            "Stake (unità)": round(o.stake_units, 1),
            "Flag": " | ".join(o.flags),
        }
        for o in state.opportunities
        if o.ev >= min_ev and (g is None or o.girone == g)
    ]
    frame = pd.DataFrame(rows, columns=list(VALUE_COLUMNS))
    frame = frame.assign(Data=pd.to_datetime(frame["Data"]).astype("datetime64[ns]"))
    for c in ("Girone", "Partita", "Mercato", "Esito", "Flag"):
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
            "Girone": t.girone.value,
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
        Girone=frame["Girone"].astype("string"),
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
    for p in fixtures_for(state, None):
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


def export_csv_bytes(frame: pd.DataFrame) -> bytes:
    """UTF-8, comma-separated, ``.`` decimal CSV without index (dates as ISO ``YYYY-MM-DD``)."""
    return frame.to_csv(index=False, sep=",", decimal=".", date_format="%Y-%m-%d", lineterminator="\n").encode("utf-8")
