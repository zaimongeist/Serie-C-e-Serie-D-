"""Serie C Quant Lab -- end-to-end demonstration pipeline.

Runs the whole chain on a dataset and prints every intermediate result to the
console:

1. dataset summary per season              (src.simulation.generator or harvested CSV)
2. Dixon-Coles fit                         (src.engine.model.DixonColesEngine)
3. strength ranking per girone             (alpha / beta / gamma, current teams only)
4. next-round predictions                  (score matrix -> 1X2, O/U, GG/NG, fair odds)
5. de-vigged bookmaker odds, EV, Kelly     (src.engine.value.ValueEngine)
6. REGRESSION_TRAP alerts                  (xGA vs goals conceded anomaly filter)
7. CSV export of the value report

Perimeter and history
---------------------
The *perimeter* is the registry of ``--season`` (``config/teams_2026_27.json``
by default): only its teams are ranked, predicted and priced. The model may be
fitted on up to three seasons: the rows of the registry season are the current
matches, older rows are history that sharpens the strength estimates (time
decay down-weights them). Teams that appear only in history are fitted as
*historical* opponents and reported as such. In the simulated modes
``--history N`` prepends ``N`` synthetic seasons *before* the registry season
(2024/25 and 2025/26 for 2026/27) whose composition differs in part from the
current one. Fixtures dated in another season than the registry's (a stale
``fixtures.csv`` of last season) are never predicted: they are listed as skipped.

Calendar of the simulated modes: ``--mode full`` plays the rounds dated before
``--as-of`` (today by default) and predicts the first round on/after it, like
the dashboard; ``--rounds N`` fixes the played rounds instead (e.g. 30 for the
parameter-recovery report). ``--mode mini`` is a fixed early-season snapshot
(the first 20 matches, rounds 1-3, and round 4 as fixtures) and says so.

Examples::

    python main.py                       # 20-match simulated mini dataset + 2 synthetic past seasons (default)
    python main.py --mode full --history 0   # simulated 2026/27 season as of today, with parameter-recovery report
    python main.py --mode full --rounds 30   # 30 played rounds whatever the date
    python main.py --mode full --as-of 2026-12-20 --history 1
    python main.py --processed data/processed/matches.csv --fixtures data/processed/fixtures.csv --odds data/odds
    python main.py --season 2026_27 --kelly 0.25 --min-ev 0.05 --export reports/value.csv
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass, field, replace
from datetime import date
from pathlib import Path
from typing import Final, Sequence

import numpy as np
import pandas as pd

from src.common.logging_config import configure_logging, get_logger
from src.common.paths import ROOT
from src.common.schemas import (
    Fixture,
    Market,
    MatchPrediction,
    OddsQuote,
    empty_matches_frame,
    season_of,
    validate_matches_df,
)
from src.common.teams import DEFAULT_SEASON, TeamRegistry, list_seasons, load_registry
from src.engine.model import DixonColesConfig, DixonColesEngine, FittedParams
from src.engine.value import KellyConfig, RegressionTrap, ValueEngine, ValueOpportunity, detect_regression_traps
from src.simulation.generator import (
    SimulatedLeague,
    history_seasons_for,
    inject_regression_trap,
    simulate_history,
    simulate_mini_dataset,
)

log = get_logger("main")

pd.set_option("display.width", 160)
pd.set_option("display.max_columns", 30)
pd.set_option("display.max_rows", 200)
pd.set_option("display.float_format", lambda v: f"{v:,.3f}")


# ----------------------------------------------------------------------------- io
def _banner(title: str) -> None:
    line = "=" * 92
    print(f"\n{line}\n{title}\n{line}")


#: Key columns of a bookmaker odds row; two rows sharing all four quote the same selection twice.
ODDS_KEY_COLUMNS: Final[tuple[str, ...]] = ("match_id", "bookmaker", "market", "selection")


def _audit_odds_file(csv_path: Path, notes: list[str]) -> None:
    """Warn about rows the odds parser has to drop or collapse silently.

    Rows with an empty ``match_id``/``bookmaker``/``market``/``selection`` cell
    cannot be attributed to a quote and are dropped; two rows with the same
    ``(match_id, bookmaker, market, selection)`` quote one selection twice and
    only the last price survives. Both are almost always export mistakes, so
    they are reported (log + console note) instead of vanishing. Unreadable
    files and missing columns are left to the parser, which skips them with its
    own warning.
    """
    try:
        frame = pd.read_csv(csv_path, dtype="string")
    except (pd.errors.EmptyDataError, pd.errors.ParserError, OSError, UnicodeDecodeError):
        return
    if any(c not in frame.columns for c in ODDS_KEY_COLUMNS):
        return
    keys = frame[list(ODDS_KEY_COLUMNS)]
    blank = keys.isna().any(axis=1)
    if blank.any():
        n_blank = int(blank.sum())
        log.warning("odds rows with empty key cells dropped", extra={"file": str(csv_path), "rows": n_blank})
        notes.append(f"{csv_path.name}: {n_blank} righe con match_id/bookmaker/market/selection vuoti ignorate.")
    duplicated = keys[~blank].duplicated(keep=False)
    if duplicated.any():
        dup_keys = keys[~blank][duplicated].drop_duplicates()
        sample = ", ".join("/".join(map(str, row)) for row in dup_keys.head(3).itertuples(index=False))
        log.warning(
            "duplicate odds rows: only the last price per selection is kept",
            extra={"file": str(csv_path), "rows": int(duplicated.sum()), "selections": len(dup_keys)},
        )
        notes.append(
            f"{csv_path.name}: {len(dup_keys)} selezioni quotate più volte dallo stesso bookmaker "
            f"(tengo l'ultima quota): {sample}" + (" ..." if len(dup_keys) > 3 else "")
        )


def load_odds_dir(odds_dir: Path, *, notes: list[str] | None = None) -> list[OddsQuote]:
    """Load bookmaker odds from ``odds_dir/*.csv``.

    Expected columns: ``match_id, bookmaker, market, selection, odds`` -- one row
    per selection. ``market`` uses the :class:`Market` values (``1X2``, ``OU2.5``,
    ``GG/NG`` ...). Rows belonging to the same (match, bookmaker, market) are
    grouped into one :class:`OddsQuote`.

    The parser is the dashboard's :func:`src.dashboard.state.load_odds_quotes`,
    so the console and the Streamlit app read exactly the same files the same
    way: an empty or unreadable CSV, a file with missing columns, an unknown
    market, an incomplete selection set or odds ``<= 1`` are skipped with a
    warning instead of aborting the run. On top of that, rows with empty key
    cells and duplicated ``(match_id, bookmaker, market, selection)`` rows are
    reported (see :func:`_audit_odds_file`); ``notes`` collects those messages
    for the console.
    """
    from src.dashboard.state import load_odds_quotes

    directory = Path(odds_dir)
    notes = [] if notes is None else notes
    for csv_path in sorted(directory.glob("*.csv")):
        _audit_odds_file(csv_path, notes)
    return load_odds_quotes(directory)


def _missing_input(option: str, path: Path, fallback: str, notes: list[str]) -> None:
    """A path the user typed explicitly does not exist: say so instead of pretending the input is empty."""
    log.warning("input path not found", extra={"option": option, "path": str(path)})
    notes.append(f"{option} {path} non trovato: {fallback}.")


def load_real_data(
    processed: Path,
    fixtures: Path | None,
    odds_dir: Path | None,
    *,
    notes: list[str] | None = None,
) -> tuple[pd.DataFrame, list[Fixture], list[OddsQuote]]:
    """Load harvested data (see ``src/ingestion/harvester.py`` for the writers).

    The processed CSV may hold several seasons (the harvester writes the last
    ``--seasons N`` seasons in one file); the season of a row is
    :func:`src.common.schemas.season_of` of its date. ``fixtures``/``odds_dir``
    are optional, but when a path *is* given and does not exist the run must not
    look like a legitimate empty season: a warning is logged and appended to
    ``notes`` (printed by :func:`main`) and the missing input is treated as empty.
    """
    from src.ingestion.harvester import load_fixtures, load_processed

    notes = [] if notes is None else notes
    matches = load_processed(Path(processed))
    fx: list[Fixture] = []
    if fixtures is not None:
        if Path(fixtures).is_file():
            fx = load_fixtures(Path(fixtures))
        else:
            _missing_input("--fixtures", Path(fixtures), "nessuna partita in programma da prevedere", notes)
    quotes: list[OddsQuote] = []
    if odds_dir is not None:
        if Path(odds_dir).is_dir():
            quotes = load_odds_dir(Path(odds_dir), notes=notes)
        else:
            _missing_input("--odds", Path(odds_dir), "nessuna quota bookmaker, salto il calcolo EV", notes)
    return matches, fx, quotes


# ------------------------------------------------------------------ diagnostics
#: ``--prior auto`` keeps the Gaussian prior on while the least-covered girone averages
#: fewer than this many played matches per team (``2 * matches / teams``). A threshold on
#: the *total* number of matches ignores the 60-team league: 100 matches are < 2 per team
#: and the pure MLE then drives gamma and rho onto their bounds. In a parameter-recovery
#: study on ``simulate_league`` (10 seeds per season length) the MLE still ended on a bound
#: in 3/10 fits at 8 matches per team, 2/10 at 10 and 1/10 at 12; from 15 matches per team
#: no fit hit a bound and the largest home-advantage error fell below 0.45.
# ``--prior auto`` rule: shared with the dashboard so console and UI agree. Coverage is
# counted on the current-season matches of the current teams: history never counts.
from src.dashboard.state import (  # noqa: E402
    PRIOR_MIN_MATCHES_PER_TEAM,
    history_label,
    matches_per_team,
    other_season_reason,
    perimeter_matches,
    prior_recommended,
    simulate_current_season,
    split_by_season,
)


def parameters_at_bounds(params: FittedParams, config: DixonColesConfig, *, tol: float = 1e-3) -> list[str]:
    """Names of the fitted parameters sitting on an optimiser bound.

    A bound-constrained estimate (``alpha``/``beta`` at ``param_bounds``,
    ``gamma`` at ``home_bounds``, ``rho`` at ``rho_bounds``) is the signature of
    an under-determined fit: the data want to push the parameter further, the
    reported value is just where the optimiser stopped, and every probability
    derived from it (and every EV) inherits that bias.
    """

    def on_bound(value: float, bounds: tuple[float, float]) -> bool:
        lo, hi = bounds
        return value <= lo + tol or value >= hi - tol

    hits = [f"alpha[{team}]" for team, v in params.alpha.items() if on_bound(v, config.param_bounds)]
    hits += [f"beta[{team}]" for team, v in params.beta.items() if on_bound(v, config.param_bounds)]
    hits += [f"gamma[{g.value}]" for g, v in params.gamma.items() if on_bound(v, config.home_bounds)]
    if on_bound(params.rho, config.rho_bounds):
        hits.append("rho")
    return hits


def season_summary(matches: pd.DataFrame, registry: TeamRegistry | None = None) -> pd.DataFrame:
    """One row per season in ``matches``: ``[stagione, partite, squadre, attuali, storiche, dal, al]``.

    ``attuali``/``storiche`` count the distinct teams of the season that are /
    are not in ``registry`` (every team is current without a registry).
    """
    columns = ["stagione", "partite", "squadre", "attuali", "storiche", "dal", "al"]
    if matches.empty:
        return pd.DataFrame(columns=columns)
    current = set(registry.canonical_names) if registry is not None else None
    rows = []
    seasons = pd.to_datetime(matches["date"]).map(season_of)
    for label, group in matches.groupby(seasons, sort=True):
        teams = set(group["home_team"].astype(str)) | set(group["away_team"].astype(str))
        n_current = len(teams) if current is None else len(teams & current)
        rows.append(
            {
                "stagione": str(label),
                "partite": int(len(group)),
                "squadre": len(teams),
                "attuali": n_current,
                "storiche": len(teams) - n_current,
                "dal": pd.Timestamp(group["date"].min()).date().isoformat(),
                "al": pd.Timestamp(group["date"].max()).date().isoformat(),
            }
        )
    return pd.DataFrame(rows, columns=columns)


def perimeter_banner(registry: TeamRegistry, current: pd.DataFrame, history: pd.DataFrame) -> str:
    """``'Perimetro: Serie C 2026/27 (60 squadre) · storico: 2280 partite 2024/25-2025/26'``."""
    text = f"Perimetro: Serie C {registry.season_label} ({registry.n_teams} squadre) · stagione corrente: {len(current)} partite"
    if len(history):
        text += f" · storico: {len(history)} partite {history_label(history)}"
    else:
        text += " · storico: nessuna partita di stagioni precedenti"
    return text


# ------------------------------------------------------------------------ pipeline
@dataclass
class PipelineOutput:
    """Everything :func:`run_pipeline` computed.

    ``matches`` is the frame the model was fitted on (history included);
    ``current_matches``/``history_matches`` split it by the registry season;
    ``skipped_fixtures`` names the fixtures that were not predicted (dated in
    another season, teams outside the perimeter or unknown to the fit).
    """

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
    registry: TeamRegistry | None = None
    season_label: str = ""
    current_matches: pd.DataFrame = field(default_factory=empty_matches_frame)
    history_matches: pd.DataFrame = field(default_factory=empty_matches_frame)
    skipped_fixtures: list[str] = field(default_factory=list)


def run_pipeline(
    matches: pd.DataFrame,
    fixtures: Sequence[Fixture],
    quotes: Sequence[OddsQuote],
    *,
    config: DixonColesConfig,
    value_engine: ValueEngine,
    export: Path | None = None,
    verbose: bool = True,
    registry: TeamRegistry | None = None,
    season: str | None = None,
) -> PipelineOutput:
    """Fit -> predict -> value, printing each step when ``verbose``.

    ``registry`` defines the perimeter (its teams are current, the others in
    ``matches`` are historical opponents); ``season`` is the label of the current
    season (defaults to the registry's, else to the season of the latest match).
    Only fixtures dated in that season and between current teams are predicted;
    the others are listed as skipped. Without a registry every fitted team is
    current, as in earlier releases.
    """
    matches = validate_matches_df(matches)
    say = print if verbose else (lambda *a, **k: None)
    notes: list[str] = []
    if season is None:
        season = registry.season_label if registry is not None else (season_of(matches["date"].max()) if len(matches) else "")
    label = str(season).replace("_", "/")
    current, history = split_by_season(matches, label)
    coverage_frame = perimeter_matches(matches, registry=registry, season=label)

    # 1. dataset ---------------------------------------------------------------
    if verbose:
        _banner("1) DATASET")
    teams = sorted(set(matches["home_team"]) | set(matches["away_team"]))
    current_names = set(registry.canonical_names) if registry is not None else set(teams)
    n_current_teams = len(set(teams) & current_names)
    say(
        f"partite: {len(matches)}   squadre: {len(teams)} ({n_current_teams} del perimetro {label}"
        + (f" + {len(teams) - n_current_teams} storiche" if len(teams) > n_current_teams else "")
        + f")   periodo: {matches['date'].min().date()} -> {matches['date'].max().date()}"
    )
    say(f"stagione corrente {label}: {len(current)} partite   storico: {len(history)} partite" + (f" ({history_label(history)})" if len(history) else ""))
    say("\nper stagione (attuali = squadre del dizionario corrente, storiche = squadre di stagioni passate):")
    say(season_summary(matches, registry).to_string(index=False))
    if registry is not None:
        without = sorted(current_names - set(teams))
        if without:
            say(f"squadre del dizionario {label} senza partite nei dati (non stimate): {', '.join(without)}")
    per_girone = current if len(current) else matches
    say(f"\nper girone ({'stagione corrente ' + label if len(current) else 'tutte le partite'}):")
    say(per_girone.groupby("girone", observed=True).agg(partite=("match_id", "count"), gol_casa=("home_goals", "mean"), gol_ospiti=("away_goals", "mean"), xg_casa=("home_xg", "mean"), xg_ospiti=("away_xg", "mean")).to_string())
    say("\nultime partite:")
    say(matches.tail(8)[["date", "girone", "round", "home_team", "away_team", "home_goals", "away_goals", "home_xg", "away_xg"]].to_string(index=False))

    # 2. fit -------------------------------------------------------------------
    if verbose:
        _banner("2) STIMA DIXON-COLES")
    engine = DixonColesEngine(config)
    params = engine.fit(matches, registry=registry)
    say(f"metodo: {params.method}   convergenza: {params.converged} ({params.message})   iterazioni: {params.n_iter}   tempo: {params.fit_seconds:.2f}s")
    say(
        f"log-likelihood pesata: {params.log_likelihood:.3f}   n={params.n_matches}   n_eff={params.effective_n:.1f}   xi={params.xi}   "
        f"prior: l2(alpha,beta)={config.l2_penalty:.2f} l2(gamma->{config.home_prior})={config.l2_penalty_home:.2f} l2(rho)={config.l2_penalty_rho:.2f}   penalità={params.penalty:.3f}"
    )
    say(
        f"squadre stimate: {len(params.teams)} = {len(params.current_teams)} del perimetro {label} + {len(params.historical_teams)} storiche"
        + (f" ({', '.join(params.historical_teams[:8])}{', ...' if len(params.historical_teams) > 8 else ''})" if params.historical_teams else "")
    )
    say(f"rho (correlazione bassi punteggi): {params.rho:+.4f}")
    current_by_girone = params.current_teams_by_girone()
    for g, gamma in params.gamma.items():
        group = params.teams_by_girone.get(g, ())
        alphas = [params.alpha[t] for t in group]
        say(
            f"girone {g.value}: gamma (fattore campo) = {gamma:+.4f}  -> moltiplicatore casa x{np.exp(gamma):.3f} | "
            f"somma alpha (gruppo di vincolo) = {sum(alphas):+.2e} | squadre = {len(current_by_girone.get(g, ()))} attuali + {len(group) - len(current_by_girone.get(g, ()))} storiche"
        )
    at_bounds = parameters_at_bounds(params, config)
    if at_bounds:
        shown = ", ".join(at_bounds[:6]) + (" ..." if len(at_bounds) > 6 else "")
        warning = (
            f"ATTENZIONE: {len(at_bounds)} parametri sui limiti dell'ottimizzatore ({shown}): stima poco affidabile "
            f"({matches_per_team(coverage_frame):.1f} partite per squadra nel girone meno coperto della stagione {label}); usa --prior on o più partite."
        )
        say(warning)
        log.warning("parameters at optimiser bounds", extra={"parameters": at_bounds, "n_matches": params.n_matches})
        notes.append(warning)

    # 3. ranking ---------------------------------------------------------------
    if verbose:
        _banner(f"3) RANKING FORZA PER GIRONE {label} (alpha = attacco, beta = solidità difensiva, gamma = fattore campo; solo squadre del perimetro)")
    for g in current_by_girone:
        rk = engine.ranking(g)
        say(f"\n--- Girone {g.value} ({len(rk)} squadre) ---")
        say(rk[["team", "alpha", "beta", "gamma", "attack_index", "defence_index"]].to_string(index=False))
    if params.historical_teams:
        say(f"\n{len(params.historical_teams)} squadre storiche escluse dal ranking (stimate solo come avversarie): {', '.join(params.historical_teams)}")

    # 4. predictions -----------------------------------------------------------
    if verbose:
        _banner(f"4) PREVISIONI PROSSIMO TURNO (solo partite tra squadre della Serie C {label})")
    predictions: list[MatchPrediction] = []
    skipped: list[str] = []
    skipped_outside: list[str] = []
    skipped_season: list[str] = []
    for fx in fixtures:
        fixture_season = season_of(fx.date)
        if fixture_season != label:
            skipped_season.append(fx.label)
            log.warning(
                "fixture of another season skipped",
                extra={"match_id": fx.match_id, "date": fx.date.isoformat(), "fixture_season": fixture_season, "season": label},
            )
            say(f"[{fx.girone.value}] {fx.date} {fx.label}: partita saltata, {other_season_reason(fixture_season, label)}")
            continue
        outside = [t for t in (fx.home_team, fx.away_team) if t in params.alpha and not params.is_current(t)]
        if outside:
            skipped_outside.append(fx.label)
            log.warning("fixture outside the perimeter skipped", extra={"match_id": fx.match_id, "season": label, "historical_teams": outside})
            say(f"[{fx.girone.value}] {fx.date} {fx.label}: partita saltata, squadra fuori dal perimetro {label} ({', '.join(outside)})")
            continue
        try:
            predictions.append(engine.predict(fx))
        except (KeyError, ValueError) as exc:
            # a team (or girone) without played matches in the dataset: early season, hand-edited
            # fixtures.csv, girone not harvested ... skip the fixture instead of aborting the run.
            skipped.append(fx.label)
            unknown = [t for t in (fx.home_team, fx.away_team) if t not in params.alpha]
            if fx.girone not in params.gamma:
                unknown.append(f"girone {fx.girone.value}")
            reason = ", ".join(unknown) if unknown else str(exc).splitlines()[0][:120]
            log.warning("fixture not predictable", extra={"match_id": fx.match_id, "error": str(exc)})
            say(f"[{fx.girone.value}] {fx.date} {fx.label}: partita saltata, squadra non presente nello storico ({reason})")
    if skipped_season:
        notes.append(f"{len(skipped_season)} partite in programma saltate (partite di un'altra stagione, non della {label}): {', '.join(skipped_season[:5])}" + (" ..." if len(skipped_season) > 5 else "") + ".")
    if skipped_outside:
        notes.append(f"{len(skipped_outside)} partite in programma saltate (squadre fuori dal perimetro {label}): {', '.join(skipped_outside[:5])}" + (" ..." if len(skipped_outside) > 5 else "") + ".")
    if skipped:
        notes.append(f"{len(skipped)} partite in programma saltate (squadre non presenti nello storico): {', '.join(skipped[:5])}" + (" ..." if len(skipped) > 5 else "") + ".")
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
        say("nessuna partita in programma nel dataset." if not (skipped or skipped_outside or skipped_season) else "nessuna partita in programma prevedibile.")

    # 5-6. value + traps -------------------------------------------------------
    if verbose:
        _banner("5) QUOTE BOOKMAKER DE-VIGGATE, EV E KELLY FRAZIONARIO")
    trap_frame = current if len(current) else matches
    traps = [
        t
        for t in detect_regression_traps(
            trap_frame,
            window=value_engine.trap_window,
            max_goals_against=value_engine.trap_max_goals_against,
            min_xg_against=value_engine.trap_min_xg_against,
        )
        if params.is_current(t.team)
    ]
    opportunities = value_engine.evaluate(predictions, quotes, matches=trap_frame, traps=traps)
    frame = value_engine.to_frame(opportunities)
    value = value_engine.value_bets(opportunities)
    say(f"quote valutate: {len(opportunities)}   soglia EV: {value_engine.min_ev:+.1%}   Kelly frazionario: x{value_engine.kelly.fraction}   bankroll: {value_engine.kelly.bankroll_units:.0f} unità")
    if value:
        vf = value_engine.to_frame(value)
        show = vf[["home_team", "away_team", "market", "selection", "bookmaker", "book_odds", "fair_odds", "p_model", "p_implied_devig", "ev", "stake_units", "flags"]].copy()
        show["ev"] = (show["ev"] * 100).round(2).astype(str) + "%"
        show["p_model"] = (show["p_model"] * 100).round(1).astype(str) + "%"
        show["p_implied_devig"] = (show["p_implied_devig"] * 100).round(1).astype(str) + "%"
        say(show.to_string(index=False))
    else:
        say("nessuna opportunità sopra soglia." if quotes else "nessuna quota bookmaker fornita: salto il calcolo EV.")
    summary = value_engine.summary(opportunities)
    say(f"\nriepilogo: {summary}")

    if verbose:
        _banner(f"6) ALERT REGRESSION_TRAP (clean sheet anomali vs xGA, stagione {label})")
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

    return PipelineOutput(
        matches,
        list(fixtures),
        list(quotes),
        params,
        engine,
        predictions,
        opportunities,
        traps,
        frame,
        export_path,
        notes=notes,
        registry=registry,
        season_label=label,
        current_matches=current,
        history_matches=history,
        skipped_fixtures=skipped_season + skipped_outside + skipped,
    )


def recovery_report(league: SimulatedLeague, params: FittedParams) -> pd.DataFrame:
    """Compare the fitted parameters of the *current* teams with the generator's ground truth."""
    rows = []
    for g, teams in params.current_teams_by_girone().items():
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
def _iso_date(text: str) -> date:
    """``argparse`` type for ``--as-of``: an ISO date (``AAAA-MM-GG``)."""
    try:
        return date.fromisoformat(text.strip()[:10])
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"data non valida {text!r}: usa il formato AAAA-MM-GG") from exc


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Serie C Quant Lab -- Dixon-Coles + EV pipeline demo")
    p.add_argument(
        "--mode",
        choices=["mini", "full"],
        default="mini",
        help="mini: fixed early-season snapshot of 20 simulated matches (default); full: simulated season positioned on the calendar (--as-of) or with --rounds played rounds",
    )
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--as-of",
        type=_iso_date,
        default=None,
        metavar="AAAA-MM-GG",
        help="simulated full mode: calendar position (default today): rounds dated before it are played, the first round on/after it is predicted",
    )
    p.add_argument(
        "--rounds",
        type=int,
        default=None,
        metavar="N",
        help="simulated full mode: fixed number of played rounds instead of the calendar (>= 1; clamped so one round of fixtures remains, 37 with 20-team gironi)",
    )
    p.add_argument(
        "--season",
        default=DEFAULT_SEASON,
        help=f"registry season = perimeter of the teams that are ranked, predicted and priced (config/teams_<season>.json; default {DEFAULT_SEASON})",
    )
    p.add_argument(
        "--history",
        type=int,
        choices=[0, 1, 2],
        default=2,
        help="simulated modes: number of synthetic seasons before the registry season (2024/25, 2025/26 for 2026/27) prepended to the current one (default 2)",
    )
    p.add_argument("--processed", type=Path, help="harvested matches CSV, possibly several seasons (overrides simulation)")
    p.add_argument("--fixtures", type=Path, help="fixtures CSV (with --processed)")
    p.add_argument("--odds", type=Path, help="directory of bookmaker odds CSVs (with --processed)")
    p.add_argument("--method", choices=["SLSQP", "L-BFGS-B"], default="SLSQP")
    p.add_argument("--xi", type=float, default=None, help="time-decay per day (default 0 for simulated data, which is stationary across seasons, 0.0018 for real data)")
    p.add_argument(
        "--prior",
        choices=["auto", "on", "off"],
        default="auto",
        help=(
            "Gaussian (MAP) prior on alpha/beta, home advantage and rho: auto (default) = on while the least-covered "
            f"girone has fewer than {PRIOR_MIN_MATCHES_PER_TEAM:g} played current-season matches per team, on = always, off = pure MLE"
        ),
    )
    p.add_argument("--l2", type=float, default=None, help="override the alpha/beta ridge strength (default from the prior)")
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
    try:
        registry = load_registry(args.season)
    except FileNotFoundError:
        build_parser().error(f"--season {args.season}: nessun file config/teams_<stagione>.json (disponibili: {', '.join(list_seasons())})")

    if args.rounds is not None and args.rounds < 1:
        build_parser().error(f"--rounds {args.rounds}: servono almeno 1 giornata giocata")
    as_of: date = args.as_of if args.as_of is not None else date.today()

    league: SimulatedLeague | None = None
    history_matches = empty_matches_frame()
    input_notes: list[str] = []
    sim_notes: list[str] = []
    if args.processed:
        if not Path(args.processed).is_file():
            build_parser().error(f"--processed {args.processed}: file non trovato")
        matches, fixtures, quotes = load_real_data(args.processed, args.fixtures, args.odds, notes=input_notes)
        xi = 0.0018 if args.xi is None else args.xi
        source = f"dati reali: {args.processed}"
    else:
        if args.mode == "mini":
            league = simulate_mini_dataset(seed=args.seed, season=registry.season_label)
            swap = 1
            played_rounds = sorted(int(r) for r in league.matches["round"].unique())
            next_round = league.fixtures[0] if league.fixtures else None
            sim_notes.append(
                f"Demo mini: istantanea fissa di inizio stagione {registry.season_label} (giornate "
                f"{played_rounds[0]}-{played_rounds[-1]} giocate, {len(league.matches)} partite"
                + (f"; turno da prevedere: {next_round.round}ª giornata del {next_round.date.isoformat()}" if next_round is not None else "")
                + "), indipendente dalla data di oggi: --as-of e --rounds valgono solo per --mode full."
            )
            calendar = "istantanea fissa di inizio stagione"
        else:
            league, played = simulate_current_season(args.seed, registry, n_rounds=args.rounds, as_of=as_of, notes=sim_notes)
            swap = 3
            calendar = f"{played} giornate giocate" + (f" al {as_of.isoformat()}" if args.rounds is None else " (--rounds)")
        registry = league.registry
        trap_team = args.trap_team or _pick_trap_team(league)
        if trap_team:
            league.matches = inject_regression_trap(league.matches, trap_team)
        matches, fixtures, quotes = league.matches, league.fixtures, league.odds
        if args.history > 0:
            past = simulate_history(seed=args.seed, registry=registry, seasons=history_seasons_for(registry, args.history), swap_per_girone=swap)
            history_matches = past.matches
            matches = validate_matches_df(pd.concat([history_matches, matches], ignore_index=True))
        xi = 0.0 if args.xi is None else args.xi
        source = (
            f"dataset simulato ({args.mode}, seed={args.seed}, {calendar}"
            + (f", +{args.history} stagioni storiche sintetiche" if args.history else "")
            + ")"
            + (f", trappola di regressione iniettata su {trap_team}" if trap_team else "")
        )
    season_label = registry.season_label
    current, history = split_by_season(validate_matches_df(matches), season_label)

    use_prior = prior_recommended(matches, registry=registry, season=season_label) if args.prior == "auto" else (args.prior == "on")
    if use_prior:
        # Thin sample (few current-season matches per current team, not few matches in total:
        # 100 matches over 60 teams are < 2 each, and three seasons of history do not make the
        # current season any longer): a Gaussian prior (alpha/beta sd 0.30, home advantage
        # 0.25 +- 0.15, rho sd 0.10) keeps gamma and rho off their bounds. With a full season
        # the data dominate and the MAP estimate is practically the MLE.
        config = DixonColesConfig.with_gaussian_prior(xi=xi, method=args.method)
        source += (
            f" | stima MAP con prior gaussiana ({matches_per_team(matches, registry=registry, season=season_label):.1f} partite per squadra nel girone "
            f"meno coperto della stagione {season_label}, soglia {PRIOR_MIN_MATCHES_PER_TEAM:g})"
        )
    else:
        config = DixonColesConfig(xi=xi, method=args.method)
    if args.l2 is not None:
        config = replace(config, l2_penalty=args.l2)

    _banner(f"SERIE C QUANT LAB -- {source}")
    print(perimeter_banner(registry, current, history))
    for note in sim_notes:
        print(note)
    for note in input_notes:
        print(f"ATTENZIONE: {note}")
    value_engine = ValueEngine(min_ev=args.min_ev, kelly=KellyConfig(fraction=args.kelly, bankroll_units=args.bankroll))
    out = run_pipeline(
        matches,
        fixtures,
        quotes,
        config=config,
        value_engine=value_engine,
        export=None if args.no_export else args.export,
        registry=registry,
        season=season_label,
    )
    out.notes = input_notes + sim_notes + out.notes

    if league is not None and args.mode == "full":
        _banner(f"VERIFICA: RECUPERO DEI PARAMETRI VERI DEL SIMULATORE (squadre {season_label})")
        out.recovery = recovery_report(league, out.params)
        print(out.recovery.to_string(index=False))
    return out


if __name__ == "__main__":
    main()
