"""Serie C Quant Lab -- end-to-end demonstration pipeline.

Runs the whole chain on a dataset and prints every intermediate result to the
console:

1. dataset summary                       (src.simulation.generator or harvested CSV)
2. Dixon-Coles fit                       (src.engine.model.DixonColesEngine)
3. strength ranking per girone           (alpha / beta / gamma)
4. next-round predictions                (score matrix -> 1X2, O/U, GG/NG, fair odds)
5. de-vigged bookmaker odds, EV, Kelly   (src.engine.value.ValueEngine)
6. REGRESSION_TRAP alerts                (xGA vs goals conceded anomaly filter)
7. CSV export of the value report

Examples::

    python main.py                       # 20-match simulated mini dataset (default)
    python main.py --mode full           # full simulated season, with parameter-recovery report
    python main.py --processed data/processed/matches.csv --fixtures data/processed/fixtures.csv --odds data/odds
    python main.py --kelly 0.25 --min-ev 0.05 --export reports/value.csv
"""
from __future__ import annotations

import argparse
import sys
import time
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd

from src.common.logging_config import configure_logging, get_logger
from src.common.paths import ROOT
from src.common.schemas import SELECTIONS, Fixture, Girone, Market, MatchPrediction, OddsQuote, validate_matches_df
from src.engine.model import DixonColesConfig, DixonColesEngine, FittedParams
from src.engine.value import KellyConfig, RegressionTrap, ValueEngine, ValueOpportunity, detect_regression_traps
from src.simulation.generator import SimulatedLeague, inject_regression_trap, simulate_league, simulate_mini_dataset

log = get_logger("main")

pd.set_option("display.width", 160)
pd.set_option("display.max_columns", 30)
pd.set_option("display.max_rows", 200)
pd.set_option("display.float_format", lambda v: f"{v:,.3f}")


# ----------------------------------------------------------------------------- io
def _banner(title: str) -> None:
    line = "=" * 92
    print(f"\n{line}\n{title}\n{line}")


def load_odds_dir(odds_dir: Path) -> list[OddsQuote]:
    """Load bookmaker odds from ``odds_dir/*.csv``.

    Expected columns: ``match_id, bookmaker, market, selection, odds`` -- one row
    per selection. ``market`` uses the :class:`Market` values (``1X2``, ``OU2.5``,
    ``GG/NG`` ...). Rows belonging to the same (match, bookmaker, market) are
    grouped into one :class:`OddsQuote`; incomplete groups are skipped with a
    warning.
    """
    quotes: list[OddsQuote] = []
    for csv in sorted(Path(odds_dir).glob("*.csv")):
        df = pd.read_csv(csv)
        required = {"match_id", "bookmaker", "market", "selection", "odds"}
        if not required.issubset(df.columns):
            log.warning("odds file skipped: missing columns", extra={"file": str(csv), "columns": list(df.columns)})
            continue
        for (match_id, bookmaker, market), grp in df.groupby(["match_id", "bookmaker", "market"], sort=False):
            try:
                quotes.append(
                    OddsQuote(
                        match_id=str(match_id),
                        bookmaker=str(bookmaker),
                        market=Market(str(market)),
                        odds={str(r.selection): float(r.odds) for r in grp.itertuples(index=False)},
                    )
                )
            except ValueError as exc:
                log.warning("odds group skipped", extra={"match_id": match_id, "market": market, "error": str(exc)})
    return quotes


def load_real_data(processed: Path, fixtures: Path | None, odds_dir: Path | None) -> tuple[pd.DataFrame, list[Fixture], list[OddsQuote]]:
    """Load harvested data (see ``src/ingestion/harvester.py`` for the writers)."""
    from src.ingestion.harvester import load_fixtures, load_processed

    matches = load_processed(Path(processed))
    fx = load_fixtures(Path(fixtures)) if fixtures and Path(fixtures).exists() else []
    quotes = load_odds_dir(Path(odds_dir)) if odds_dir and Path(odds_dir).exists() else []
    return matches, fx, quotes


# ------------------------------------------------------------------------ pipeline
@dataclass
class PipelineOutput:
    matches: pd.DataFrame
    fixtures: list[Fixture]
    quotes: list[OddsQuote]
    params: FittedParams
    engine: DixonColesEngine
    predictions: list[MatchPrediction]
    opportunities: list[ValueOpportunity]
    traps: list[RegressionTrap]
    value_frame: pd.DataFrame
    export_path: Path | None = None
    recovery: pd.DataFrame | None = None
    notes: list[str] = field(default_factory=list)


def run_pipeline(
    matches: pd.DataFrame,
    fixtures: Sequence[Fixture],
    quotes: Sequence[OddsQuote],
    *,
    config: DixonColesConfig,
    value_engine: ValueEngine,
    export: Path | None = None,
    verbose: bool = True,
) -> PipelineOutput:
    """Fit -> predict -> value, printing each step when ``verbose``."""
    matches = validate_matches_df(matches)
    say = print if verbose else (lambda *a, **k: None)

    # 1. dataset ---------------------------------------------------------------
    if verbose:
        _banner("1) DATASET")
    teams = sorted(set(matches["home_team"]) | set(matches["away_team"]))
    say(f"partite: {len(matches)}   squadre: {len(teams)}   periodo: {matches['date'].min().date()} -> {matches['date'].max().date()}")
    say(matches.groupby("girone", observed=True).agg(partite=("match_id", "count"), gol_casa=("home_goals", "mean"), gol_ospiti=("away_goals", "mean"), xg_casa=("home_xg", "mean"), xg_ospiti=("away_xg", "mean")).to_string())
    say("\nultime partite:")
    say(matches.tail(8)[["date", "girone", "round", "home_team", "away_team", "home_goals", "away_goals", "home_xg", "away_xg"]].to_string(index=False))

    # 2. fit -------------------------------------------------------------------
    if verbose:
        _banner("2) STIMA DIXON-COLES")
    engine = DixonColesEngine(config)
    t0 = time.perf_counter()
    params = engine.fit(matches)
    say(f"metodo: {params.method}   convergenza: {params.converged} ({params.message})   iterazioni: {params.n_iter}   tempo: {params.fit_seconds:.2f}s")
    say(f"log-likelihood pesata: {params.log_likelihood:.3f}   n={params.n_matches}   n_eff={params.effective_n:.1f}   xi={params.xi}   penalità L2={config.l2_penalty}")
    say(f"rho (correlazione bassi punteggi): {params.rho:+.4f}")
    for g, gamma in params.gamma.items():
        alphas = [params.alpha[t] for t in params.teams_by_girone.get(g, ())]
        say(f"girone {g.value}: gamma (fattore campo) = {gamma:+.4f}  -> moltiplicatore casa x{np.exp(gamma):.3f} | somma alpha = {sum(alphas):+.2e} | squadre = {len(alphas)}")

    # 3. ranking ---------------------------------------------------------------
    if verbose:
        _banner("3) RANKING FORZA PER GIRONE (alpha = attacco, beta = solidità difensiva, gamma = fattore campo)")
    for g in params.teams_by_girone:
        rk = engine.ranking(g)
        say(f"\n--- Girone {g.value} ---")
        say(rk[["team", "alpha", "beta", "gamma", "attack_index", "defence_index"]].to_string(index=False))

    # 4. predictions -----------------------------------------------------------
    if verbose:
        _banner("4) PREVISIONI PROSSIMO TURNO")
    predictions = engine.predict_fixtures(fixtures)
    for p in predictions:
        fx = p.fixture
        hs, as_, ps = p.most_likely_score()
        p1x2 = p.probabilities[Market.MATCH_ODDS]
        pou = p.probabilities[Market.OVER_UNDER_25]
        pgg = p.probabilities[Market.BTTS]
        fo = p.fair_odds
        say(
            f"[{fx.girone.value}] {fx.date} {fx.home_team} - {fx.away_team}: "
            f"lambda={p.home_rate:.2f} mu={p.away_rate:.2f} | "
            f"1={p1x2['1']:.1%} X={p1x2['X']:.1%} 2={p1x2['2']:.1%} "
            f"(fair {fo[Market.MATCH_ODDS]['1']:.2f}/{fo[Market.MATCH_ODDS]['X']:.2f}/{fo[Market.MATCH_ODDS]['2']:.2f}) | "
            f"O2.5={pou['Over 2.5']:.1%} GG={pgg['GG']:.1%} | risultato più probabile {hs}-{as_} ({ps:.1%})"
        )
    if not predictions:
        say("nessuna partita in programma nel dataset.")

    # 5-6. value + traps -------------------------------------------------------
    if verbose:
        _banner("5) QUOTE BOOKMAKER DE-VIGGATE, EV E KELLY FRAZIONARIO")
    traps = detect_regression_traps(
        matches,
        window=value_engine.trap_window,
        max_goals_against=value_engine.trap_max_goals_against,
        min_xg_against=value_engine.trap_min_xg_against,
    )
    opportunities = value_engine.evaluate(predictions, quotes, traps=traps)
    frame = value_engine.to_frame(opportunities)
    value = value_engine.value_bets(opportunities)
    say(f"quote valutate: {len(opportunities)}   soglia EV: {value_engine.min_ev:+.1%}   Kelly frazionario: x{value_engine.kelly.fraction}   bankroll: {value_engine.kelly.bankroll_units:.0f} unità")
    if value:
        vf = value_engine.to_frame(value)
        show = vf[["home_team", "away_team", "market", "selection", "book_odds", "fair_odds", "p_model", "p_implied_devig", "ev", "stake_units", "flags"]].copy()
        show["ev"] = (show["ev"] * 100).round(2).astype(str) + "%"
        show["p_model"] = (show["p_model"] * 100).round(1).astype(str) + "%"
        show["p_implied_devig"] = (show["p_implied_devig"] * 100).round(1).astype(str) + "%"
        say(show.to_string(index=False))
    else:
        say("nessuna opportunità sopra soglia." if quotes else "nessuna quota bookmaker fornita: salto il calcolo EV.")
    summary = value_engine.summary(opportunities)
    say(f"\nriepilogo: {summary}")

    if verbose:
        _banner("6) ALERT REGRESSION_TRAP (clean sheet anomali vs xGA)")
    if traps:
        for t in traps:
            say(" * " + t.description())
        flagged = [o for o in opportunities if "REGRESSION_TRAP" in o.flags]
        say(f"selezioni Over/GG segnalate dal filtro: {len(flagged)}")
    else:
        say("nessuna squadra in trappola di regressione nella finestra corrente.")

    # 7. export ----------------------------------------------------------------
    export_path: Path | None = None
    if export is not None:
        if verbose:
            _banner("7) EXPORT")
        export_path = value_engine.export_csv(opportunities, Path(export))
        say(f"report CSV scritto in: {export_path}")

    return PipelineOutput(matches, list(fixtures), list(quotes), params, engine, predictions, opportunities, traps, frame, export_path)


def recovery_report(league: SimulatedLeague, params: FittedParams) -> pd.DataFrame:
    """Compare fitted parameters with the generator's ground truth."""
    rows = []
    for g, teams in params.teams_by_girone.items():
        a_true = np.array([league.true_params.alpha[t] for t in teams])
        b_true = np.array([league.true_params.beta[t] for t in teams])
        a_hat = np.array([params.alpha[t] for t in teams])
        b_hat = np.array([params.beta[t] for t in teams])
        rows.append(
            {
                "girone": g.value,
                "n_teams": len(teams),
                "corr_alpha": float(np.corrcoef(a_true, a_hat)[0, 1]) if len(teams) > 2 else np.nan,
                "corr_beta": float(np.corrcoef(b_true, b_hat)[0, 1]) if len(teams) > 2 else np.nan,
                "gamma_true": league.true_params.gamma[g],
                "gamma_hat": params.gamma[g],
                "rho_true": league.true_params.rho,
                "rho_hat": params.rho,
            }
        )
    return pd.DataFrame(rows)


# ------------------------------------------------------------------------------ cli
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Serie C Quant Lab -- Dixon-Coles + EV pipeline demo")
    p.add_argument("--mode", choices=["mini", "full"], default="mini", help="mini: 20 simulated matches (default); full: whole simulated season")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--processed", type=Path, help="harvested matches CSV (overrides simulation)")
    p.add_argument("--fixtures", type=Path, help="fixtures CSV (with --processed)")
    p.add_argument("--odds", type=Path, help="directory of bookmaker odds CSVs (with --processed)")
    p.add_argument("--method", choices=["SLSQP", "L-BFGS-B"], default="SLSQP")
    p.add_argument("--xi", type=float, default=None, help="time-decay per day (default 0 for simulated data, 0.0018 for real data)")
    p.add_argument("--l2", type=float, default=None, help="ridge penalty (default 0.1 for the mini dataset, 0 otherwise)")
    p.add_argument("--kelly", type=float, choices=[0.125, 0.25], default=0.125, help="Kelly fraction")
    p.add_argument("--bankroll", type=float, default=100.0, help="bankroll in units")
    p.add_argument("--min-ev", type=float, default=0.035, help="EV threshold (fraction)")
    p.add_argument("--export", type=Path, default=ROOT / "reports" / "value_report.csv")
    p.add_argument("--no-export", action="store_true")
    p.add_argument("--trap-team", default=None, help="team whose last 3 matches are rewritten to showcase REGRESSION_TRAP (simulated modes only; default: automatic)")
    p.add_argument("--log-level", default="WARNING")
    p.add_argument("--log-json", action="store_true")
    return p


def _pick_trap_team(league: SimulatedLeague) -> str | None:
    """A team with >= 3 played matches that also plays in the next round."""
    played = pd.concat([league.matches["home_team"], league.matches["away_team"]]).value_counts()
    for fx in league.fixtures:
        for team in (fx.home_team, fx.away_team):
            if played.get(team, 0) >= 3:
                return team
    return None


def main(argv: Sequence[str] | None = None) -> PipelineOutput:
    args = build_parser().parse_args(argv)
    configure_logging(args.log_level, json_output=args.log_json, force=True)

    league: SimulatedLeague | None = None
    if args.processed:
        matches, fixtures, quotes = load_real_data(args.processed, args.fixtures, args.odds)
        xi = 0.0018 if args.xi is None else args.xi
        l2 = 0.0 if args.l2 is None else args.l2
        source = f"dati reali: {args.processed}"
    else:
        league = simulate_mini_dataset(seed=args.seed) if args.mode == "mini" else simulate_league(seed=args.seed, n_rounds=30)
        trap_team = args.trap_team or _pick_trap_team(league)
        if trap_team:
            league.matches = inject_regression_trap(league.matches, trap_team)
        matches, fixtures, quotes = league.matches, league.fixtures, league.odds
        xi = 0.0 if args.xi is None else args.xi
        l2 = (0.1 if args.mode == "mini" else 0.0) if args.l2 is None else args.l2
        source = f"dataset simulato ({args.mode}, seed={args.seed})" + (f", trappola di regressione iniettata su {trap_team}" if trap_team else "")

    _banner(f"SERIE C QUANT LAB -- {source}")
    config = DixonColesConfig(xi=xi, method=args.method, l2_penalty=l2)
    value_engine = ValueEngine(min_ev=args.min_ev, kelly=KellyConfig(fraction=args.kelly, bankroll_units=args.bankroll))
    out = run_pipeline(matches, fixtures, quotes, config=config, value_engine=value_engine, export=None if args.no_export else args.export)

    if league is not None and args.mode == "full":
        _banner("VERIFICA: RECUPERO DEI PARAMETRI VERI DEL SIMULATORE")
        out.recovery = recovery_report(league, out.params)
        print(out.recovery.to_string(index=False))
    return out


if __name__ == "__main__":
    main()
