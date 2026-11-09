"""Serie C Quant Lab -- end-to-end demonstration pipeline.

Runs the whole chain on a dataset and prints every intermediate result to the
console:

1. dataset summary per season              (src.simulation.generator or harvested CSV)
2. Dixon-Coles fit                         (src.engine.model.DixonColesEngine)
3. strength ranking per girone             (alpha / beta / gamma, current teams only)
4. next-matchday table                     (src.engine.tables.matchday_table: one row per fixture with form,
                                           rates, probabilities, fair and book odds, EV max and flags)
5. de-vigged bookmaker odds, EV, Kelly     (src.engine.value.ValueEngine)
6. REGRESSION_TRAP alerts                  (xGA vs goals conceded anomaly filter)
8. picks of the matchday                   (src.engine.picks: top-N draws and both-teams-to-score, printed before 7)
7. CSV export of the value report, of the picks (``<stem>_selezioni.csv``), of the matchday table
   (``<stem>_giornata.csv``) and of the played results (``<stem>_risultati.csv``)

plus, after the fit, **AFFIDABILITÀ MODELLO** (``--backtest``, on by default with real data): the
walk-forward backtest (every past matchday predicted by a model fitted only on the matches before it,
src.engine.backtest), its aggregated scores per section and market against the league base rates and
the bookmaker (the model's own scores on the quoted matches next to the market's), the track record of
the lists, of their top-k multiples and of the slips and the validated recalibration
(src.engine.calibration; applied to the probabilities of the following sections only when it improves
the out-of-sample log loss, ``--no-recalibration`` keeps the raw ones), exported to
``<stem>_affidabilita.csv``; and, at the end, **SCHEDINE** (``--slips``, on by default): the three
slips of the next matchday over the chosen sections (5 X, 10 GG, mista 1 / 2 / GG / Over 2.5,
src.engine.slips), exported to ``<stem>_schedine.csv``.

Perimeter and history
---------------------
The *perimeter* is the registry of ``--season`` (``config/teams_2026_27.json``
by default: the three Serie C gironi and Serie B, every competition of
``config/competitions.toml`` the file holds, 80 teams): only its teams are
ranked, predicted and priced. The banner spells it out from the configuration
(``Perimetro: Serie C 2026/27 Gironi A/B/C + Serie B (80 squadre) ...``). The model may be
fitted on up to three seasons: the rows of the registry season are the current
matches, older rows are history that sharpens the strength estimates (time
decay down-weights them). Teams that appear only in history are fitted as
*historical* opponents and reported as such. In the simulated modes
``--history N`` prepends ``N`` synthetic seasons *before* the registry season
(2024/25 and 2025/26 for 2026/27) whose composition differs in part from the
current one. Fixtures dated in another season than the registry's (a stale
``fixtures.csv`` of last season) are never predicted: they are listed as skipped.

Sections (``--section``, default ``all``)
-----------------------------------------
The fit (steps 1-2) is one per fit group (country: Serie B, C and D together,
:func:`src.engine.fitgroups.fit_by_group`, L-BFGS-B unless ``--method SLSQP``);
steps 3-8 are printed once per *section* -- the sections of
``config/competitions.toml``, the same split as the dashboard sidebar: ``Serie
C``, ``Serie B``, ``Serie D`` and every league switched on later -- each with its
own banner, ranking per competition, matchday table, value panel, trap alerts and
picks (5 X / 12 GG ranked within the section). ``--section`` takes a section
name, its slug or a comma-separated list (``serie-b``, ``"Serie C"``, ``c,d``),
``all`` (every section with teams) or the legacy aliases ``bc`` (Serie C and
Serie B) and ``d``. Only the sections that have teams run: the Serie D gironi
ship empty (the FootyStats harvester initialises them from the API), so until
then the console prints the Serie D status and the command that initialises it.
With more than one section the per-section CSVs carry the section slug
(``<stem>_c_giornata.csv``, ``<stem>_b_giornata.csv``, ``<stem>_d_giornata.csv``,
...); the value report at ``--export`` always covers the whole perimeter. In the
simulated full mode ``--serie-d-demo`` adds a synthetic Serie D registry (nine
gironi of 18 clubs) so the section can be shown offline.

Real data and demo
------------------
``--footystats`` downloads the FootyStats data first with the user's key
(:func:`src.dashboard.data_refresh.refresh`: ``FOOTYSTATS_API_KEY`` from the
environment or ``.env``; the data is saved only locally in ``data/processed`` /
``data/odds``, never committed) and then runs on it; ``--processed`` runs on CSVs
already harvested. Without either the console runs a **simulated** league
(``--mode mini`` / ``--mode full``): its first and last output line is the
warning :data:`~src.simulation.generator.DEMO_WARNING` (invented results, dates
and odds: never to be used for betting), and every CSV it writes with
``--export`` is named ``DEMO_<name>`` and starts with the column ``AVVISO`` =
``DATI SIMULATI - NON USARE PER SCOMMETTERE``.

Rounds: the matchday table of section 4 and the picks of section 8 follow the
**official rounds** (:mod:`src.engine.calendar`): the current round of every
competition with every one of its matches -- played ones with the score
(``Stato`` ``giocata 2-1``), the ones that have kicked off without a result in
the data (``in corso / risultato in attesa``, no prediction and no pick), the
ones to play with the kick-off in Europe/Rome and the model's numbers, the
postponed ones -- evaluated at ``--as-of`` (midnight, Europe/Rome) or now for
real data.

Calendar of the simulated modes: ``--mode full`` plays the rounds dated before
``--as-of`` (today by default) and predicts the first round on/after it, like
the dashboard demo; ``--rounds N`` fixes the played rounds instead (e.g. 30 for the
parameter-recovery report). ``--mode mini`` is a fixed early-season snapshot
(the first 20 matches, rounds 1-3, and round 4 as fixtures) and says so.

Examples::

    python main.py --footystats          # download the real data with the key in .env, then run on it
    python main.py                       # DEMO: 20-match simulated mini dataset + 2 synthetic past seasons (default)
    python main.py --mode full --history 0   # simulated 2026/27 season as of today, with parameter-recovery report
    python main.py --mode full --rounds 30   # 30 played rounds whatever the date
    python main.py --mode full --as-of 2026-12-20 --history 1
    python main.py --processed data/processed/matches.csv --fixtures data/processed/fixtures.csv --odds data/odds
    python main.py --season 2026_27 --kelly 0.25 --min-ev 0.05 --export reports/value.csv
    python main.py --n-draws 8 --n-btts 15        # longer matchday pick lists (section 8)
    python main.py --mode full --rounds 6 --history 0 --serie-d-demo --section d   # the Serie D section on synthetic data
    python main.py --processed data/processed/matches.csv --section serie-b --no-backtest   # Serie B only, no reliability section
    python main.py --mode full --history 1 --backtest --slip-mixed 8   # demo with the reliability section and a longer mixed slip
"""
from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass, field, replace
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Callable, Final, Sequence

import numpy as np
import pandas as pd

from src.common.logging_config import configure_logging, get_logger
from src.common.paths import ODDS_DIR, PROCESSED_DIR, ROOT
from src.common.competitions import describe_competitions
from src.common.schemas import (
    Fixture,
    Girone,
    MatchPrediction,
    OddsQuote,
    empty_matches_frame,
    season_of,
    validate_matches_df,
)
from src.common.teams import DEFAULT_SEASON, TeamRegistry, list_seasons, load_registry
from src.engine.calendar import IN_PLAY_TEXT, STATUS_TO_PLAY, kicked_off, rome_midnight
from src.engine.fitgroups import DEFAULT_FIT_METHOD, GroupedFit, fit_by_group
from src.engine.model import DixonColesConfig, DixonColesEngine, FittedParams
from src.engine.picks import DEFAULT_N_BTTS, DEFAULT_N_DRAWS, Matchday, Pick, btts_picks, draw_picks, matchdays, picks_frame
from src.engine.slips import DEFAULT_N_GG, DEFAULT_N_MIXED, DEFAULT_N_X, MIXED_MAX_LEGS, MIXED_MIN_LEGS, Slip, build_slips, slips_frame, slips_summary
from src.engine.tables import matchday_table
from src.engine.value import KellyConfig, RegressionTrap, ValueEngine, ValueOpportunity, detect_regression_traps
from src.simulation.generator import (
    DEMO_WARNING,
    SimulatedLeague,
    history_seasons_for,
    inject_regression_trap,
    simulate_history,
    simulate_mini_dataset,
    simulate_serie_d_registry,
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
    from src.dashboard.state import load_enabled_csv
    from src.ingestion.harvester import load_fixtures, load_processed

    notes = [] if notes is None else notes
    # rows of a league switched off after its download are skipped with a note instead of aborting the run
    matches = load_enabled_csv(Path(processed), load_processed, notes=notes, what="partite")
    fx: list[Fixture] = []
    if fixtures is not None:
        if Path(fixtures).is_file():
            fx = load_enabled_csv(Path(fixtures), load_fixtures, notes=notes, what="partite in calendario")
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
#: ``--prior auto`` keeps the Gaussian prior on while the least-covered competition averages
#: fewer than this many played matches per team (``2 * matches / teams``). A threshold on
#: the *total* number of matches ignores the size of the 80-team perimeter (three Serie C
#: gironi plus Serie B): 100 matches are barely more than 2 per team and the pure MLE then
#: drives gamma and rho onto their bounds. In a parameter-recovery
#: study on ``simulate_league`` (10 seeds per season length) the MLE still ended on a bound
#: in 3/10 fits at 8 matches per team, 2/10 at 10 and 1/10 at 12; from 15 matches per team
#: no fit hit a bound and the largest home-advantage error fell below 0.45.
# ``--prior auto`` rule: shared with the dashboard so console and UI agree. Coverage is
# counted on the current-season matches of the current teams: history never counts, and a
# registry competition that was never harvested (no match in any season) is ignored by the
# rule instead of pinning it to zero coverage (``CoverageReport.ignored`` names it).
from src.dashboard.state import (  # noqa: E402
    ALL_SECTIONS_LABEL,
    DEFAULT_BACKTEST_SEASONS,
    EVIDENCE_NOTE,
    INDEPENDENCE_NOTE,
    PRIOR_MIN_MATCHES_PER_TEAM,
    QUOTED_LOG_LOSS_COLUMN,
    SECTION_ALIASES,
    SECTION_BC,
    SECTION_D,
    SECTIONS,
    CoverageReport,
    ReliabilityReport,
    accumulators_table,
    build_reliability,
    configured_gironi,
    coverage_report,
    demo_file_name,
    export_csv_bytes,
    grouped_results_table,
    history_label,
    in_section,
    matches_per_team,
    metrics_table_for,
    other_season_reason,
    perimeter_matches,
    picks_csv_bytes,
    picks_record_table,
    picks_track_for,
    prior_recommended,
    prior_setting,
    recalibration_table,
    reliability_export_frame,
    resolve_section,
    section_matches,
    section_members,
    season_fixtures,
    section_slug,
    sections_for,
    serie_d_gironi,
    serie_d_harvest_command,
    serie_d_status_text,
    simulate_current_season,
    slips_export_frame,
    slips_gironi,
    slips_record_table,
    slips_track_for,
    split_by_season,
)

#: ``--section`` shortcuts -> the sections of :mod:`src.dashboard.state` they ask for (``all`` = every section with
#: teams); any section name or slug is accepted too (:func:`parse_sections`).
SECTION_CHOICES: Final[dict[str, tuple[str, ...]]] = {
    "all": tuple(SECTIONS),
    "bc": tuple(SECTION_ALIASES[SECTION_BC]),
    "d": (SECTION_D,),
}


def parse_sections(value: str) -> tuple[str, ...]:
    """The sections asked for by ``--section``: ``all``, a section name or slug, an alias, or a comma-separated list.

    ``'all'`` -> every configured section; ``'bc'`` -> ``('Serie C', 'Serie B')``;
    ``'serie-b'`` / ``'Serie B'`` / ``'b'`` -> ``('Serie B',)``; ``'c,d'`` ->
    ``('Serie C', 'Serie D')``. :class:`argparse.ArgumentTypeError` for an
    unknown name. Configuration order, without duplicates.
    """
    text = str(value).strip()
    if text.lower() in SECTION_CHOICES:
        return SECTION_CHOICES[text.lower()]
    found: list[str] = []
    for part in (p for p in text.split(",") if p.strip()):
        try:
            found.extend(section_members(resolve_section(part)))
        except ValueError as exc:
            raise argparse.ArgumentTypeError(
                f"sezione sconosciuta {part.strip()!r}: usa all, {', '.join(SECTIONS)} (o le sigle {', '.join(section_slug(s) for s in SECTIONS)}, bc)"
            ) from exc
    if not found:
        raise argparse.ArgumentTypeError("--section vuota: usa all o il nome di una sezione")
    ordered = [s for s in SECTIONS if s in found] + [s for s in dict.fromkeys(found) if s not in SECTIONS]
    return tuple(dict.fromkeys(ordered))

#: Console pointer to the glossary, printed with the matchday picks.
LEGEND_POINTER: Final[str] = "Legenda: tab Legenda della dashboard oppure README"


#: Columns of the matchday table printed by section 4 (the full table goes to ``<stem>_giornata.csv``).
CONSOLE_MATCHDAY_COLUMNS: Final[tuple[str, ...]] = (
    "Competizione",
    "Data",
    "Calcio d'inizio",
    "Casa",
    "Ospiti",
    "Stato",
    "λ",
    "μ",
    "P1 (%)",
    "PX (%)",
    "P2 (%)",
    "O2.5 (%)",
    "GG (%)",
    "Fair X",
    "Book X",
    "EV max (%)",
    "Flag",
)
#: One-line reading guide of the compact table, printed under it.
MATCHDAY_CONSOLE_LEGEND: Final[str] = (
    "colonne: Calcio d'inizio in ora italiana · Stato giocata (con il risultato, senza previsione) / da giocare / "
    "in corso / risultato in attesa (iniziata, risultato non ancora nei dati: senza previsione) / rinviata · "
    "λ/μ gol attesi casa/ospiti · P1 PX P2 O2.5 GG probabilità % del modello · Fair X quota fair del pareggio · "
    "Book X miglior quota book della X · EV max miglior valore atteso % fra le selezioni quotate · Flag unione dei flag "
    "della partita (tabella completa con forma, α/β, tutte le quote e match_id nel CSV <export>_giornata.csv)"
)


def section_export_stem(export: Path, section: str | None, *, multi: bool) -> Path:
    """The base path of a section's CSVs: ``export`` itself for one section, ``<stem>_<slug>.csv`` when several sections run.

    ``reports/value_report.csv`` -> ``reports/value_report_d.csv`` for Serie D
    with ``multi=True``, so :func:`picks_export_path` & co. then yield
    ``value_report_d_selezioni.csv``, ``value_report_d_giornata.csv``,
    ``value_report_d_risultati.csv``.
    """
    export = Path(export)
    if section is None or not multi:
        return export
    return export.with_name(f"{export.stem}_{section_slug(section)}{export.suffix}")


def demo_export_path(export: Path) -> Path:
    """``export`` with its file name prefixed by ``DEMO_`` (the base path of every CSV of the simulated modes)."""
    export = Path(export)
    return export.with_name(demo_file_name(export.name, True))


def picks_export_path(export: Path) -> Path:
    """``reports/value_report.csv`` -> ``reports/value_report_selezioni.csv`` (next to the value report)."""
    export = Path(export)
    return export.with_name(f"{export.stem}_selezioni.csv")


def matchday_export_path(export: Path) -> Path:
    """``reports/value_report.csv`` -> ``reports/value_report_giornata.csv`` (the matchday table of section 4)."""
    export = Path(export)
    return export.with_name(f"{export.stem}_giornata.csv")


def reliability_export_path(export: Path) -> Path:
    """``reports/value_report.csv`` -> ``reports/value_report_affidabilita.csv`` (aggregated walk-forward scores)."""
    export = Path(export)
    return export.with_name(f"{export.stem}_affidabilita.csv")


def slips_export_path(export: Path) -> Path:
    """``reports/value_report.csv`` -> ``reports/value_report_schedine.csv`` (the three slips of the next matchday)."""
    export = Path(export)
    return export.with_name(f"{export.stem}_schedine.csv")


def results_export_path(export: Path) -> Path:
    """``reports/value_report.csv`` -> ``reports/value_report_risultati.csv`` (the played matches of the current season)."""
    export = Path(export)
    return export.with_name(f"{export.stem}_risultati.csv")


def format_matchday_console(frame: pd.DataFrame, columns: Sequence[str] = CONSOLE_MATCHDAY_COLUMNS) -> str:
    """Compact text view of a :func:`~src.engine.tables.matchday_table` frame for the console.

    Only ``columns`` are shown (:data:`CONSOLE_MATCHDAY_COLUMNS` by default);
    percentages with one decimal, odds and rates with two, the EV with its sign,
    ``-`` for a missing value (an unquoted selection). ``'nessuna partita.'`` for
    an empty frame.
    """
    if frame.empty:
        return "nessuna partita."
    view = frame[list(columns)].copy()
    if "Data" in view.columns:
        view["Data"] = pd.to_datetime(view["Data"]).dt.strftime("%Y-%m-%d")

    def pct(value: float) -> str:
        return "-" if pd.isna(value) else f"{value:.1f}"

    def odds(value: float) -> str:
        return "-" if pd.isna(value) else f"{value:.2f}"

    def signed(value: float) -> str:
        return "-" if pd.isna(value) else f"{value:+.2f}"

    formatters = {c: pct for c in view.columns if c.endswith("(%)") and not c.startswith("EV")}
    formatters.update({c: odds for c in view.columns if c in ("λ", "μ", "Tot. att.") or c.startswith(("Fair ", "Book "))})
    formatters.update({c: signed for c in view.columns if c.startswith("EV")})
    return view.to_string(index=False, formatters=formatters, na_rep="-")


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


def perimeter_banner(registry: TeamRegistry, current: pd.DataFrame, history: pd.DataFrame, section: str | None = None) -> str:
    """``'Perimetro: Serie C 2026/27 Gironi A/B/C + Serie B (80 squadre) · stagione corrente: 440 partite · storico: 3040 partite 2024/25-2025/26'``.

    The competitions come from the registry (:attr:`TeamRegistry.perimeter_label`,
    built on ``config/competitions.toml``), never from a hard-coded string. With
    ``section`` only that section's competitions with teams are named
    (``'Perimetro: Serie D 2026/27 Gironi A-I (162 squadre) · ...'``) and the
    match counts are restricted to them.
    """
    if section is None:
        label = registry.perimeter_label
    else:
        gironi = [g for g in registry.gironi if in_section(g, section)]
        n_teams = sum(len(registry.teams_by_girone.get(g, ())) for g in gironi)
        label = (
            f"{describe_competitions((g.competition for g in gironi), season=registry.season_label)} ({n_teams} squadre)"
            if gironi
            else f"{section} {registry.season_label} (nessuna squadra)"
        )
        current, history = section_matches(current, section), section_matches(history, section)
    text = f"Perimetro: {label} · stagione corrente: {len(current)} partite"
    if len(history):
        text += f" · storico: {len(history)} partite {history_label(history)}"
    else:
        text += " · storico: nessuna partita di stagioni precedenti"
    return text


def registry_serie_d_status(registry: TeamRegistry, current: pd.DataFrame) -> str:
    """The Serie D status (:func:`src.dashboard.state.serie_d_status_text`) from a registry and the current-season matches."""
    n_teams = sum(len(registry.teams_by_girone.get(g, ())) for g in serie_d_gironi())
    return serie_d_status_text(n_teams, len(section_matches(current, SECTION_D)), season_label=registry.season_label)


def serie_d_demo_registry(base: TeamRegistry, *, seed: int = 7, n_teams: int = 18) -> TeamRegistry:
    """``base`` (the real perimeter) plus the synthetic Serie D gironi of :func:`simulate_serie_d_registry` (``--serie-d-demo``).

    The merged registry keeps the base season and every base team under its
    competition and adds ``n_teams`` synthetic clubs to each configured Serie D
    girone (``'Squadra DA 01'`` ...), so the simulated league and the console
    demo can show the Serie D section offline. Aliases are not needed by the
    simulator and are left out.
    """
    extra = simulate_serie_d_registry(seed=seed, n_teams=n_teams, season=base.season)
    teams_by_code: dict[str, tuple[str, ...]] = {g.value: tuple(base.teams_by_girone[g]) for g in base.gironi}
    for g in extra.gironi:
        teams_by_code[g.value] = tuple(extra.teams_by_girone[g])
    merged = TeamRegistry.from_teams(base.season, teams_by_code)
    log.info("synthetic Serie D registry merged", extra={"serie_d_gironi": [g.value for g in extra.gironi], "n_teams": merged.n_teams})
    return merged


# ------------------------------------------------------------------------ pipeline
@dataclass
class SectionOutput:
    """What :func:`run_pipeline` computed for one section (``section`` is ``None`` for the whole perimeter).

    ``gironi`` are the configured competitions of the section; ``predictions``,
    ``opportunities`` and ``traps`` the perimeter-wide ones restricted to them;
    ``matchdays`` the section's own "giornate" (grouped from its fixtures only),
    ``matchday_frame`` the table of its next matchday, ``results_frame`` its
    played matches of the current season, ``draw_picks``/``btts_picks`` the top-N
    lists ranked within the section, and the ``*_export_path`` the CSVs written
    with ``--export`` (``None`` otherwise).
    """

    section: str | None
    gironi: tuple[Girone, ...]
    predictions: list[MatchPrediction] = field(default_factory=list)
    opportunities: list[ValueOpportunity] = field(default_factory=list)
    traps: list[RegressionTrap] = field(default_factory=list)
    matchdays: list[Matchday] = field(default_factory=list)
    matchday_frame: pd.DataFrame = field(default_factory=pd.DataFrame)
    results_frame: pd.DataFrame = field(default_factory=pd.DataFrame)
    draw_picks: list[Pick] = field(default_factory=list)
    btts_picks: list[Pick] = field(default_factory=list)
    picks_export_path: Path | None = None
    matchday_export_path: Path | None = None
    results_export_path: Path | None = None

    @property
    def next_matchday(self) -> Matchday | None:
        """The first upcoming matchday of the section, ``None`` without predicted fixtures."""
        return self.matchdays[0] if self.matchdays else None


@dataclass
class PipelineOutput:
    """Everything :func:`run_pipeline` computed.

    ``matches`` is the frame the model was fitted on (history included);
    ``current_matches``/``history_matches`` split it by the registry season;
    ``skipped_fixtures`` names the fixtures that were not predicted (dated in
    another season, teams outside the perimeter or unknown to the fit).
    ``matchdays`` groups the predicted fixtures into "giornate"
    (:func:`src.engine.picks.matchdays`); ``draw_picks``/``btts_picks`` are the
    section-8 lists of the next matchday and ``picks_export_path`` their CSV
    (``None`` without ``--export``). ``matchday_frame`` is the section-4 table of
    the next matchday (:func:`src.engine.tables.matchday_table`, every column)
    and ``results_frame`` the played matches of the current season with the
    model's reading (:func:`src.engine.tables.results_table`); with ``--export``
    they are written to ``matchday_export_path`` / ``results_export_path``.

    ``reliability`` is the walk-forward report of the *AFFIDABILITÀ MODELLO*
    section (``None`` without ``backtest``), ``recalibrated`` whether its
    validated recalibration was applied to the predictions, ``slips`` the three
    slips of the *SCHEDINE* section for ``slips_matchday``; their CSVs are
    ``reliability_export_path`` / ``slips_export_path`` with ``--export``.

    ``sections`` holds the per-section results (:class:`SectionOutput`, keyed by
    section name) when :func:`run_pipeline` ran sections. The top-level
    ``matchdays`` / frames / picks / export paths mirror the **single** section
    when exactly one ran (so ``--section d`` yields Serie D's tables) and the
    **whole perimeter** when several ran (whose CSVs then carry a section
    suffix each, see :func:`section_export_stem`, and the top-level paths are
    ``None``); ``export_path`` is always the whole-perimeter value report.
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
    matchdays: list[Matchday] = field(default_factory=list)
    draw_picks: list[Pick] = field(default_factory=list)
    btts_picks: list[Pick] = field(default_factory=list)
    picks_export_path: Path | None = None
    matchday_frame: pd.DataFrame = field(default_factory=pd.DataFrame)
    results_frame: pd.DataFrame = field(default_factory=pd.DataFrame)
    matchday_export_path: Path | None = None
    results_export_path: Path | None = None
    sections: dict[str, SectionOutput] = field(default_factory=dict)
    reliability: ReliabilityReport | None = None
    reliability_export_path: Path | None = None
    recalibrated: bool = False
    slips: list[Slip] = field(default_factory=list)
    slips_matchday: Matchday | None = None
    slips_export_path: Path | None = None

    @property
    def next_matchday(self) -> Matchday | None:
        """The first upcoming matchday, ``None`` without predicted fixtures."""
        return self.matchdays[0] if self.matchdays else None


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
    n_draws: int = DEFAULT_N_DRAWS,
    n_btts: int = DEFAULT_N_BTTS,
    sections: Sequence[str] | None = None,
    now: datetime | None = None,
    demo: bool = False,
    backtest: bool = False,
    backtest_seasons: int = DEFAULT_BACKTEST_SEASONS,
    prior: str = "auto",
    odds_history: pd.DataFrame | None = None,
    recalibrate: bool = True,
    slips: bool = True,
    n_slip_x: int = DEFAULT_N_X,
    n_slip_gg: int = DEFAULT_N_GG,
    n_slip_mixed: int = DEFAULT_N_MIXED,
) -> PipelineOutput:
    """Fit -> predict -> value, printing each step when ``verbose``.

    ``registry`` defines the perimeter (its teams are current, the others in
    ``matches`` are historical opponents); ``season`` is the label of the current
    season (defaults to the registry's, else to the season of the latest match).
    Only fixtures dated in that season and between current teams are predicted;
    the others are listed as skipped. Without a registry every fitted team is
    current, as in earlier releases. ``n_draws``/``n_btts`` size the section-8
    lists (most probable ``X`` and ``GG`` of the next matchday); with ``export``
    they are also written to :func:`picks_export_path`. Section 4 prints the
    next matchday as one table (:func:`format_matchday_console`) instead of one
    line per fixture; with ``export`` the full table and the played results of
    the current season go to :func:`matchday_export_path` /
    :func:`results_export_path`.

    ``sections`` (:data:`~src.dashboard.state.SECTION_BC`,
    :data:`~src.dashboard.state.SECTION_D`) prints steps 3-8 once per section,
    each under its own banner and restricted to the section's competitions
    (picks ranked within the section, matchdays grouped from its fixtures);
    with several sections the CSVs carry the section slug
    (:func:`section_export_stem`). ``None`` (the default) is one pass over the
    whole perimeter, the pre-Serie-D behaviour; an empty sequence prints only
    steps 1-2 (and writes the value report). The value engine runs once for the
    whole perimeter whatever the sections.

    The matchdays follow the official rounds (:func:`src.engine.picks.matchdays`
    on the fixtures of the current season and its played matches) at the
    reference instant ``now`` (aware; ``None`` = the fixture list already is the
    upcoming schedule, the simulated modes): the section-4 table lists every
    match of the current rounds, played ones with their score.

    ``demo`` marks the data as simulated: every CSV written with ``export``
    goes to :func:`demo_export_path` (``DEMO_<name>``) and starts with the
    column ``AVVISO`` = ``DATI SIMULATI - NON USARE PER SCOMMETTERE``
    (:func:`src.dashboard.state.mark_demo_frame`).

    ``backtest`` prints the *AFFIDABILITÀ MODELLO* section right after the fit
    (:func:`src.dashboard.state.build_reliability` over ``backtest_seasons``
    seasons with the ``config`` time decay and the ``prior`` setting;
    ``odds_history`` adds the comparison with the bookmaker) and, with
    ``recalibrate``, applies its validated recalibration to every probability
    that follows (only the markets that improved out of sample). ``slips``
    prints the *SCHEDINE* section at the end: the three slips of the next
    matchday of the sections run (``n_slip_x`` X, ``n_slip_gg`` GG,
    ``n_slip_mixed`` mixed legs).
    """
    if demo and export is not None:
        export = demo_export_path(Path(export))
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
    engine = fit_by_group(matches, registry=registry, config=config, fixtures=season_fixtures(fixtures, label), method=None)
    params = engine.params_
    say(f"metodo: {params.method}   convergenza: {params.converged} ({params.message})   iterazioni: {params.n_iter}   tempo: {params.fit_seconds:.2f}s")
    if len(engine.groups) > 1 or engine.missing_groups:
        for line in engine.describe():
            say(line)
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
            f"{g.short} [{g.value}]: gamma (fattore campo) = {gamma:+.4f}  -> moltiplicatore casa x{np.exp(gamma):.3f} | "
            f"somma alpha (gruppo di vincolo) = {sum(alphas):+.2e} | squadre = {len(current_by_girone.get(g, ()))} attuali + {len(group) - len(current_by_girone.get(g, ()))} storiche"
        )
    for g in params.teams_by_girone:
        if g not in params.gamma:
            say(
                f"{g.short} [{g.value}]: nessuna partita nei dati, fattore campo non stimato; le sue {len(current_by_girone.get(g, ()))} squadre "
                "del perimetro sono vincolate con la competizione in cui hanno giocato (raccogli la competizione per prevederne le partite)"
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

    # AFFIDABILITÀ MODELLO: the walk-forward backtest and the validated recalibration (applied to what follows)
    reliability: ReliabilityReport | None = None
    reliability_path: Path | None = None
    if backtest:
        reliability = _reliability_section(
            matches,
            say=say,
            verbose=verbose,
            registry=registry,
            label=label,
            n_seasons=backtest_seasons,
            xi=float(config.xi),
            prior=prior,
            odds_history=odds_history,
            sections=sections,
            recalibrate=recalibrate,
        )
        if export is not None:
            reliability_path = reliability_export_path(Path(export))
            reliability_path.parent.mkdir(parents=True, exist_ok=True)
            reliability_path.write_bytes(export_csv_bytes(reliability_export_frame(reliability), demo=demo))
            say(f"punteggi aggregati scritti in: {reliability_path}")
    model = reliability.recalibration if reliability is not None and recalibrate else None
    recalibrated = model is not None and model.enabled

    # 3-8 are printed per section (3. ranking, 4. next matchday, 5-6. value + traps, 8. picks, 7. export); the
    # predictions and the value engine run once for the whole perimeter first, so every section shares one fit.
    predictions: list[MatchPrediction] = []
    skipped: list[str] = []
    skipped_outside: list[str] = []
    skipped_season: list[str] = []
    skip_lines: list[tuple[Girone, str]] = []
    for fx in fixtures:
        fixture_season = season_of(fx.date)
        if fixture_season != label:
            skipped_season.append(fx.label)
            log.warning(
                "fixture of another season skipped",
                extra={"match_id": fx.match_id, "date": fx.date.isoformat(), "fixture_season": fixture_season, "season": label},
            )
            skip_lines.append((fx.girone, f"[{fx.girone.value}] {fx.date} {fx.label}: partita saltata, {other_season_reason(fixture_season, label)}"))
            continue
        outside = [t for t in (fx.home_team, fx.away_team) if t in params.alpha and not params.is_current(t)]
        if outside:
            skipped_outside.append(fx.label)
            log.warning("fixture outside the perimeter skipped", extra={"match_id": fx.match_id, "season": label, "historical_teams": outside})
            skip_lines.append((fx.girone, f"[{fx.girone.value}] {fx.date} {fx.label}: partita saltata, squadra fuori dal perimetro {label} ({', '.join(outside)})"))
            continue
        try:
            prediction = engine.predict(fx)
            predictions.append(model.apply_prediction(prediction) if recalibrated and model is not None else prediction)
        except (KeyError, ValueError) as exc:
            # a team (or girone) without played matches in the dataset: early season, hand-edited
            # fixtures.csv, girone not harvested ... skip the fixture instead of aborting the run.
            skipped.append(fx.label)
            unknown = [t for t in (fx.home_team, fx.away_team) if t not in params.alpha]
            if fx.girone not in params.gamma:
                unknown.append(f"girone {fx.girone.value}")
            reason = ", ".join(unknown) if unknown else str(exc).splitlines()[0][:120]
            log.warning("fixture not predictable", extra={"match_id": fx.match_id, "error": str(exc)})
            skip_lines.append((fx.girone, f"[{fx.girone.value}] {fx.date} {fx.label}: partita saltata, squadra non presente nello storico ({reason})"))
    if skipped_season:
        notes.append(f"{len(skipped_season)} partite in programma saltate (partite di un'altra stagione, non della {label}): {', '.join(skipped_season[:5])}" + (" ..." if len(skipped_season) > 5 else "") + ".")
    if skipped_outside:
        notes.append(f"{len(skipped_outside)} partite in programma saltate (squadre fuori dal perimetro {label}): {', '.join(skipped_outside[:5])}" + (" ..." if len(skipped_outside) > 5 else "") + ".")
    if skipped:
        notes.append(f"{len(skipped)} partite in programma saltate (squadre non presenti nello storico): {', '.join(skipped[:5])}" + (" ..." if len(skipped) > 5 else "") + ".")
    # the value engine runs here, once for the whole perimeter (its Book / EV / Flag columns belong to the matchday tables)
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
    calendar = season_fixtures(fixtures, label)
    giornate = matchdays(calendar, played=current, now=now)
    log.info("predictions and value computed", extra={"n_predictions": len(predictions), "n_opportunities": len(opportunities), "n_traps": len(traps), "n_matchdays": len(giornate)})

    export_path: Path | None = None
    if export is not None and demo:
        export_path = Path(export)
        export_path.parent.mkdir(parents=True, exist_ok=True)
        export_path.write_bytes(export_csv_bytes(frame, demo=True))
        log.info("value report exported (demo)", extra={"path": str(export_path), "rows": len(frame)})
    elif export is not None:
        export_path = value_engine.export_csv(opportunities, Path(export))
    passes: list[str | None] = [None] if sections is None else [str(sec) for sec in sections]
    multi = len(passes) > 1
    section_outputs: dict[str, SectionOutput] = {}
    for position, sec in enumerate(passes):
        result = _run_section(
            sec,
            say=say,
            verbose=verbose,
            label=label,
            registry=registry,
            current=current,
            history=history,
            matches=matches,
            engine=engine,
            params=params,
            predictions=predictions,
            opportunities=opportunities,
            traps=traps,
            quotes=quotes,
            value_engine=value_engine,
            skip_lines=skip_lines,
            n_draws=n_draws,
            n_btts=n_btts,
            calendar=calendar,
            now=now,
            export=None if export is None else section_export_stem(Path(export), sec, multi=multi),
            value_report=export_path if position == 0 else None,
            first=position == 0,
            demo=demo,
        )
        if sec is not None:
            section_outputs[sec] = result

    if len(passes) == 1:
        head = result
    else:  # several sections: the top-level fields describe the whole perimeter, nothing unsuffixed is written
        next_day = giornate[0] if giornate else None
        head = SectionOutput(
            section=None,
            gironi=tuple(Girone),
            predictions=list(predictions),
            opportunities=list(opportunities),
            traps=list(traps),
            matchdays=giornate,
            matchday_frame=matchday_table(predictions, params=params, matches=trap_frame, opportunities=opportunities, traps=traps, matchday=next_day),
            results_frame=grouped_results_table(matches, engine, season=label),
            draw_picks=draw_picks(predictions, n=n_draws, matchday=next_day, opportunities=opportunities) if next_day else [],
            btts_picks=btts_picks(predictions, n=n_btts, matchday=next_day, opportunities=opportunities) if next_day else [],
        )

    slip_list: list[Slip] = []
    slip_day: Matchday | None = None
    slip_path: Path | None = None
    if slips and (sections is None or len(passes) > 0):
        slip_list, slip_day, slip_path = _slips_section(
            say=say,
            verbose=verbose,
            sections=None if sections is None else [str(sec) for sec in sections],
            predictions=predictions,
            opportunities=opportunities,
            quotes=quotes,
            calendar=calendar,
            current=current,
            now=now,
            value_engine=value_engine,
            n_x=n_slip_x,
            n_gg=n_slip_gg,
            n_mixed=n_slip_mixed,
            export=None if export is None else Path(export),
            demo=demo,
            reliability=reliability,
            recalibrated=recalibrated,
        )

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
        matchdays=head.matchdays,
        draw_picks=head.draw_picks,
        btts_picks=head.btts_picks,
        picks_export_path=head.picks_export_path,
        matchday_frame=head.matchday_frame,
        results_frame=head.results_frame,
        matchday_export_path=head.matchday_export_path,
        results_export_path=head.results_export_path,
        sections=section_outputs,
        reliability=reliability,
        reliability_export_path=reliability_path,
        recalibrated=recalibrated,
        slips=slip_list,
        slips_matchday=slip_day,
        slips_export_path=slip_path,
    )


#: Columns of the compact per-market table of the console's *AFFIDABILITÀ MODELLO* section.
CONSOLE_RELIABILITY_COLUMNS: Final[tuple[str, ...]] = (
    "Mercato",
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


def _reliability_section(
    matches: pd.DataFrame,
    *,
    say: "Callable[..., object]",
    verbose: bool,
    registry: TeamRegistry | None,
    label: str,
    n_seasons: int,
    xi: float,
    prior: str,
    odds_history: pd.DataFrame | None,
    sections: Sequence[str] | None,
    recalibrate: bool,
) -> ReliabilityReport:
    """The console's *AFFIDABILITÀ MODELLO*: walk-forward scores per section and market, track records, recalibration.

    Aggregates only (no match-level line): what the dashboard tab shows, for
    the whole perimeter and each section that runs.
    """
    if verbose:
        _banner("AFFIDABILITÀ MODELLO (backtest walk-forward: ogni giornata prevista solo con le partite precedenti)")
    report = build_reliability(matches, registry=registry, season_label=label, n_seasons=n_seasons, xi=xi, prior=prior, odds_history=odds_history)
    say(report.result.summary())
    if report.empty:
        say("nessuna partita valutabile fuori campione: niente punteggi, niente ricalibrazione.")
        return report
    present = list(dict.fromkeys(report.frame["section"].astype(str)))
    scopes: list[str | None] = [None, *[sec for sec in (sections if sections is not None else present) if sec in present]]
    for scope in scopes:
        table = metrics_table_for(report, scope)
        say(f"\n--- {ALL_SECTIONS_LABEL if scope is None else scope}: punteggi per mercato (log loss e Brier: più bassi = meglio) ---")
        say(table[list(CONSOLE_RELIABILITY_COLUMNS)].to_string(index=False, na_rep="-"))
    if not report.has_odds:
        say("nessuno storico delle quote pre-partita: confronto con il mercato e ROI non disponibili.")
    else:
        say(
            f"il mercato è misurato solo sulle partite con quote: confrontalo con «{QUOTED_LOG_LOSS_COLUMN}» (stesse partite), "
            "non con «Log loss modello» (tutte le partite valutate)."
        )
    picks = picks_track_for(report)
    say(f"\ntrack record delle liste (top {DEFAULT_N_DRAWS} X e top {DEFAULT_N_BTTS} GG di ogni giornata passata, {ALL_SECTIONS_LABEL.lower()}):")
    say(picks_record_table(picks, None).to_string(index=False, na_rep="-"))
    for scope in scopes:
        table = accumulators_table(picks, scope)
        if table.empty:
            continue
        say(f"multiple top-k delle liste ({ALL_SECTIONS_LABEL if scope is None else scope}: le prime k partite di ogni giornata giocate insieme):")
        say(table.drop(columns=["Lista"]).to_string(index=False, na_rep="-"))
    track = slips_track_for(report, None if sections is None else list(sections))
    say("track record delle tre schedine (stesse regole della sezione SCHEDINE):")
    say(slips_record_table(track).to_string(index=False, na_rep="-"))
    say(EVIDENCE_NOTE)
    model = report.recalibration
    say("\n" + model.describe())
    say(recalibration_table(model)[["Mercato", "Applicata", "Partite verifica", "Log loss prima", "Log loss dopo", "Miglioramento (%)", "Motivo"]].to_string(index=False, na_rep="-"))
    if model.enabled:
        say(
            "probabilità usate nelle sezioni seguenti: "
            + ("ricalibrate (" + ", ".join(m.value for m in model.enabled_markets) + ")" if recalibrate else "originali (--no-recalibration)")
        )
    log.info("reliability section", extra={"n_matches": report.result.n_matches, "recalibration": [m.value for m in model.enabled_markets]})
    return report


def _slips_section(
    *,
    say: "Callable[..., object]",
    verbose: bool,
    sections: Sequence[str] | None,
    predictions: Sequence[MatchPrediction],
    opportunities: Sequence[ValueOpportunity],
    quotes: Sequence[OddsQuote],
    calendar: Sequence[Fixture],
    current: pd.DataFrame,
    now: datetime | None,
    value_engine: ValueEngine,
    n_x: int,
    n_gg: int,
    n_mixed: int,
    export: Path | None,
    demo: bool,
    reliability: ReliabilityReport | None,
    recalibrated: bool,
) -> tuple[list[Slip], Matchday | None, Path | None]:
    """The console's *SCHEDINE*: the three slips of the next matchday of the sections run, with the CSV."""
    scope = "tutte le sezioni" if sections is None else ", ".join(sections)
    if verbose:
        _banner(f"SCHEDINE ({scope}: {n_x} X, {n_gg} GG, mista 1 / 2 / GG / Over 2.5 da {n_mixed})")
    gironi = slips_gironi(sections)
    days = matchdays(calendar, gironi=None if sections is None else gironi, played=current, now=now)
    day = days[0] if days else None
    chosen = None if sections is None else list(dict.fromkeys(m for sec in sections for m in section_members(sec)))
    candidates = [p for p in predictions if p.fixture.girone in gironi] if day is not None else []
    slip_list = build_slips(
        candidates,
        now=now,
        opportunities=opportunities,
        quotes=quotes,
        n_x=n_x,
        n_gg=n_gg,
        n_mixed=n_mixed,
        sections=chosen,
        matchday=day,
    )
    bankroll, fraction = value_engine.kelly.bankroll_units, value_engine.kelly.fraction
    if day is None:
        say("nessuna partita in programma prevedibile: nessuna schedina.")
    else:
        say(f"{day.label} · probabilità {'ricalibrate' if recalibrated else 'originali del modello'}")
        say(slips_summary(slip_list, bankroll_units=bankroll, fraction=fraction).to_string(index=False, na_rep="-"))
        for slip in slip_list:
            say(f"\n{slip.title} ({slip.n_legs} partite){' · ' + slip.coverage_note if slip.coverage_note else ''}")
            legs = slips_frame([slip]).drop(columns=["Schedina"])
            say(legs.to_string(index=False, na_rep="-") if not legs.empty else "nessuna partita.")
            if slip.margin_note:
                say(slip.margin_note)
        say("\n" + INDEPENDENCE_NOTE)
        if reliability is not None and not reliability.empty:
            track = slips_track_for(reliability, sections, n_x=n_x, n_gg=n_gg, n_mixed=n_mixed)
            say("stesse schedine nelle giornate passate (backtest fuori campione):")
            say(slips_record_table(track).to_string(index=False, na_rep="-"))
    path: Path | None = None
    if export is not None:
        path = slips_export_path(export)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(export_csv_bytes(slips_export_frame(slip_list, day, bankroll_units=bankroll, fraction=fraction), demo=demo))
        say(f"schedine scritte in: {path}")
    log.info("slips section", extra={"matchday": None if day is None else day.key, "legs": {s.kind: s.n_legs for s in slip_list}})
    return slip_list, day, path


def _run_section(
    section: str | None,
    *,
    say: "Callable[..., object]",
    verbose: bool,
    label: str,
    registry: TeamRegistry | None,
    current: pd.DataFrame,
    history: pd.DataFrame,
    matches: pd.DataFrame,
    engine: "DixonColesEngine | GroupedFit",
    params: FittedParams,
    predictions: Sequence[MatchPrediction],
    opportunities: Sequence[ValueOpportunity],
    traps: Sequence[RegressionTrap],
    quotes: Sequence[OddsQuote],
    value_engine: ValueEngine,
    skip_lines: Sequence[tuple[Girone, str]],
    n_draws: int,
    n_btts: int,
    export: Path | None,
    value_report: Path | None,
    first: bool = True,
    calendar: Sequence[Fixture] = (),
    now: datetime | None = None,
    demo: bool = False,
) -> SectionOutput:
    """Steps 3-8 of :func:`run_pipeline` for one section (``None`` = the whole perimeter); see :class:`SectionOutput`.

    ``export`` is the base path of the section's CSVs (already suffixed by the
    caller when several sections run); ``value_report`` is the whole-perimeter
    value report written by the caller, named in the export step of the first
    section only; ``first`` marks the first pass, the only one that prints the
    perimeter-wide list of historical teams excluded from the rankings.
    ``calendar`` (the current-season fixtures) and ``now`` build the section's
    matchdays from the official rounds. ``demo`` marks every CSV written as
    simulated (first column ``AVVISO``, see :func:`run_pipeline`).
    """
    gironi = configured_gironi(section)
    suffix = "" if section is None else f" · {section}"
    preds = [p for p in predictions if in_section(p.fixture.girone, section)]
    opps = [o for o in opportunities if in_section(o.girone, section)]
    section_traps = [t for t in traps if in_section(t.girone, section)]
    section_quotes = [q for q in quotes if q.match_id in {p.fixture.match_id for p in preds}] if section is not None else list(quotes)
    if section is not None and verbose:
        banner_perimeter = perimeter_banner(registry, current, history, section) if registry is not None else f"Sezione {section}"
        _banner(f"SEZIONE {section.upper()} -- {banner_perimeter}")
        if section == SECTION_D:
            say(f"Serie D {registry_serie_d_status(registry, current) if registry is not None else 'senza dizionario'}.")

    # 3. ranking ---------------------------------------------------------------
    current_by_girone = params.current_teams_by_girone()
    if verbose:
        _banner(f"3) RANKING FORZA PER COMPETIZIONE {label}{suffix} (alpha = attacco, beta = solidità difensiva, gamma = fattore campo; solo squadre del perimetro)")
    for g in current_by_girone:
        if not in_section(g, section):
            continue
        rk = engine.ranking(g)
        say(f"\n--- {g.short} ({len(rk)} squadre) ---")
        say(rk[["team", "alpha", "beta", "gamma", "attack_index", "defence_index"]].to_string(index=False))
    if first and params.historical_teams:
        say(f"\n{len(params.historical_teams)} squadre storiche escluse dal ranking (stimate solo come avversarie): {', '.join(params.historical_teams)}")

    # 4. predictions -----------------------------------------------------------
    if verbose:
        _banner(f"4) PREVISIONI PROSSIMO TURNO{suffix} (solo partite tra squadre del perimetro {label})")
    for g, line in skip_lines:
        if in_section(g, section):
            say(line)
    giornate = matchdays(calendar, gironi=gironi if section is not None else None, played=current, now=now)
    next_day = giornate[0] if giornate else None
    trap_frame = current if len(current) else matches
    matchday_frame = matchday_table(preds, params=params, matches=trap_frame, opportunities=opps, traps=section_traps, matchday=next_day)
    if next_day is None:
        say("nessuna partita in programma nel dataset." if not any(in_section(g, section) for g, _ in skip_lines) else "nessuna partita in programma prevedibile.")
    else:
        status = matchday_frame["Stato"].astype(str)
        n_played = int(status.str.startswith("giocata").sum())
        n_to_play = int((status == STATUS_TO_PLAY).sum())
        n_in_play = int((status == IN_PLAY_TEXT).sum())
        n_postponed = len(matchday_frame) - n_played - n_to_play - n_in_play
        others = sum(1 for p in preds if not next_day.contains(p))
        say(
            f"{next_day.label} · {len(matchday_frame)} partite ({n_played} giocate, {n_to_play} da giocare"
            + (f", {n_in_play} in corso o in attesa del risultato" if n_in_play else "")
            + (f", {n_postponed} rinviate" if n_postponed else "")
            + ")"
            + (f" · altre {others} partite fuori dalla giornata corrente (solo nel CSV delle previsioni)" if others else "")
        )
        say(format_matchday_console(matchday_frame))
        say(MATCHDAY_CONSOLE_LEGEND)
    results_frame = grouped_results_table(matches, engine, season=label, gironi=gironi if section is not None else None)
    log.info(
        "matchday table",
        extra={"section": section, "matchday": None if next_day is None else next_day.key, "n_rows": len(matchday_frame), "n_results": len(results_frame)},
    )

    # 5-6. value + traps -------------------------------------------------------
    if verbose:
        _banner(f"5) QUOTE BOOKMAKER DE-VIGGATE, EV E KELLY FRAZIONARIO{suffix}")
    started = {p.fixture.match_id for p in preds if kicked_off(p.fixture, now)}
    pre_match = [o for o in opps if o.match_id not in started]
    value = value_engine.value_bets(pre_match)
    say(f"quote valutate: {len(pre_match)}   soglia EV: {value_engine.min_ev:+.1%}   Kelly frazionario: x{value_engine.kelly.fraction}   bankroll: {value_engine.kelly.bankroll_units:.0f} unità")
    if len(pre_match) < len(opps):
        say(f"{len(opps) - len(pre_match)} quote di partite già iniziate (o rinviate) escluse: nessuna giocata pre-partita su una partita in corso.")
    if value:
        vf = value_engine.to_frame(value)
        show = vf[["home_team", "away_team", "market", "selection", "bookmaker", "book_odds", "fair_odds", "p_model", "p_implied_devig", "ev", "stake_units", "flags"]].copy()
        show["ev"] = (show["ev"] * 100).round(2).astype(str) + "%"
        show["p_model"] = (show["p_model"] * 100).round(1).astype(str) + "%"
        show["p_implied_devig"] = (show["p_implied_devig"] * 100).round(1).astype(str) + "%"
        say(show.to_string(index=False))
    elif section_quotes:
        say("nessuna opportunità sopra soglia.")
    elif section == SECTION_D and quotes:
        say("nessuna quota bookmaker per le partite di Serie D (su FootyStats la copertura delle quote di Serie D è parziale): salto il calcolo EV.")
    else:
        say("nessuna quota bookmaker fornita: salto il calcolo EV.")
    summary = value_engine.summary(pre_match)
    say(f"\nriepilogo: {summary}")

    if verbose:
        _banner(f"6) ALERT REGRESSION_TRAP{suffix} (clean sheet anomali vs xGA, stagione {label})")
    if section_traps:
        for t in section_traps:
            say(" * " + t.description())
        flagged = [o for o in opps if "REGRESSION_TRAP" in o.flags]
        say(f"selezioni Over/GG segnalate dal filtro: {len(flagged)}")
    else:
        say(
            "nessuna squadra in trappola di regressione nella finestra corrente."
            + (" In Serie D gli xG mancano spesso: senza xG il filtro non può scattare." if section == SECTION_D else "")
        )

    # 8. picks of the matchday (printed before the export, which writes them too) -----
    if verbose:
        _banner(f"8) SELEZIONI DI GIORNATA{suffix} (solo probabilità del modello: la quota e l'EV dicono se il prezzo vale)")
    top_draws = draw_picks(preds, n=n_draws, matchday=next_day, opportunities=opps) if next_day else []
    top_btts = btts_picks(preds, n=n_btts, matchday=next_day, opportunities=opps) if next_day else []
    draws_frame, btts_frame = picks_frame(top_draws), picks_frame(top_btts)
    scope = "tutte le competizioni" if section is None else f"tutte le competizioni di {section}"
    if next_day is None:
        say("nessuna partita in programma prevedibile: nessuna giornata da selezionare.")
    else:
        say(f"{next_day.label} · {next_day.n_fixtures} partite" + (f" · {len(giornate)} giornate in programma" if len(giornate) > 1 else ""))
        say(f"\nle {len(top_draws)} partite più probabili da X ({scope}):")
        say(draws_frame.to_string(index=False) if not draws_frame.empty else "nessuna partita.")
        say(f"\nle {len(top_btts)} partite più probabili da GG ({scope}):")
        say(btts_frame.to_string(index=False) if not btts_frame.empty else "nessuna partita.")
    say(f"\n{LEGEND_POINTER}")
    log.info(
        "matchday picks",
        extra={"section": section, "matchday": None if next_day is None else next_day.key, "n_draws": len(top_draws), "n_btts": len(top_btts), "n_matchdays": len(giornate)},
    )

    # 7. export ----------------------------------------------------------------
    picks_path: Path | None = None
    matchday_path: Path | None = None
    results_path: Path | None = None
    if export is not None:
        if verbose:
            _banner(f"7) EXPORT{suffix}")
        if value_report is not None:
            say(f"report CSV scritto in: {value_report}")
        picks_path = picks_export_path(Path(export))
        picks_path.parent.mkdir(parents=True, exist_ok=True)
        picks_path.write_bytes(picks_csv_bytes(draws_frame, btts_frame, next_day, demo=demo))
        log.info("matchday picks exported", extra={"section": section, "path": str(picks_path), "rows": len(draws_frame) + len(btts_frame)})
        say(f"selezioni di giornata scritte in: {picks_path}")
        matchday_path = matchday_export_path(Path(export))
        matchday_path.write_bytes(export_csv_bytes(matchday_frame, demo=demo))
        log.info("matchday table exported", extra={"section": section, "path": str(matchday_path), "rows": len(matchday_frame)})
        say(f"tabella giornata scritta in: {matchday_path}")
        results_path = results_export_path(Path(export))
        results_path.write_bytes(export_csv_bytes(results_frame, demo=demo))
        log.info("results table exported", extra={"section": section, "path": str(results_path), "rows": len(results_frame)})
        say(f"risultati giocati scritti in: {results_path}")

    return SectionOutput(
        section=section,
        gironi=gironi,
        predictions=preds,
        opportunities=opps,
        traps=section_traps,
        matchdays=giornate,
        matchday_frame=matchday_frame,
        results_frame=results_frame,
        draw_picks=top_draws,
        btts_picks=top_btts,
        picks_export_path=picks_path,
        matchday_export_path=matchday_path,
        results_export_path=results_path,
    )


def recovery_report(league: SimulatedLeague, params: FittedParams) -> pd.DataFrame:
    """Compare the fitted parameters of the *current* teams with the generator's ground truth.

    One row per competition of the perimeter (``girone`` = code, ``competizione``
    = short label); ``gamma_hat`` is ``NaN`` for a competition without matches in
    the fit (no home advantage estimated).
    """
    rows = []
    for g, teams in params.current_teams_by_girone().items():
        a_true = np.array([league.true_params.alpha[t] for t in teams])
        b_true = np.array([league.true_params.beta[t] for t in teams])
        a_hat = np.array([params.alpha[t] for t in teams])
        b_hat = np.array([params.beta[t] for t in teams])
        rows.append(
            {
                "girone": g.value,
                "competizione": g.short,
                "n_teams": len(teams),
                "corr_alpha": float(np.corrcoef(a_true, a_hat)[0, 1]) if len(teams) > 2 else np.nan,
                "corr_beta": float(np.corrcoef(b_true, b_hat)[0, 1]) if len(teams) > 2 else np.nan,
                "gamma_true": league.true_params.gamma.get(g, np.nan),
                "gamma_hat": params.gamma.get(g, np.nan),
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
        help="simulated full mode: calendar position (default today): rounds dated before it are played, the first round on/after it is predicted; "
        "real data: the current rounds are those of that day (midnight Europe/Rome, default now)",
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
    p.add_argument(
        "--footystats",
        action="store_true",
        help="download the real data from FootyStats first (key FOOTYSTATS_API_KEY in the environment or .env; saved only "
        "locally in data/processed and data/odds), then run on it instead of a simulated league",
    )
    p.add_argument("--processed", type=Path, help="harvested matches CSV, possibly several seasons (overrides simulation)")
    p.add_argument("--fixtures", type=Path, help="fixtures CSV (with --processed)")
    p.add_argument("--odds", type=Path, help="directory of bookmaker odds CSVs (with --processed)")
    p.add_argument(
        "--method",
        choices=["SLSQP", "L-BFGS-B"],
        default=DEFAULT_FIT_METHOD,
        help=f"optimiser of the per-country Dixon-Coles fit (default {DEFAULT_FIT_METHOD}: same optimum as SLSQP, about 40 times faster)",
    )
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
    p.add_argument("--export", type=Path, default=ROOT / "reports" / "value_report.csv", help="value report CSV; the matchday picks go next to it as <stem>_selezioni.csv")
    p.add_argument("--no-export", action="store_true")
    p.add_argument("--n-draws", type=int, default=DEFAULT_N_DRAWS, metavar="N", help=f"section 8: most probable draws (X) of the next matchday to list (default {DEFAULT_N_DRAWS})")
    p.add_argument("--n-btts", type=int, default=DEFAULT_N_BTTS, metavar="N", help=f"section 8: most probable both-teams-to-score (GG) matches to list (default {DEFAULT_N_BTTS})")
    p.add_argument("--trap-team", default=None, help="team whose last 3 matches are rewritten to showcase REGRESSION_TRAP (simulated modes only; default: automatic)")
    p.add_argument(
        "--section",
        default="all",
        help=(
            "which section(s) to print and export after the fit: all (default) = every section that has teams, one after the "
            f"other with its own banner, ranking, matchday table, value, picks and CSVs (suffixed _<slug> when several run); or a "
            f"section name / slug / comma-separated list ({', '.join(SECTIONS)}; slugs {', '.join(section_slug(x) for x in SECTIONS)}); "
            "bc = Serie C and Serie B (legacy alias), d = Serie D"
        ),
    )
    p.add_argument(
        "--backtest",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="print the AFFIDABILITÀ MODELLO section (walk-forward backtest, scores against base rates and bookmaker, track "
        "records, validated recalibration) and export <stem>_affidabilita.csv; default: on with real data, off in the simulated modes",
    )
    p.add_argument(
        "--backtest-seasons",
        type=int,
        choices=[1, 2, 3],
        default=DEFAULT_BACKTEST_SEASONS,
        help=f"seasons up to the current one evaluated by the backtest (default {DEFAULT_BACKTEST_SEASONS})",
    )
    p.add_argument(
        "--recalibration",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="apply the validated recalibration of the backtest to the probabilities (only the markets that improved out of sample; default on)",
    )
    p.add_argument(
        "--slips",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="print the SCHEDINE section (three slips of the next matchday) and export <stem>_schedine.csv (default on)",
    )
    p.add_argument("--slip-x", type=int, default=DEFAULT_N_X, metavar="N", help=f"legs of the X slip (default {DEFAULT_N_X})")
    p.add_argument("--slip-gg", type=int, default=DEFAULT_N_GG, metavar="N", help=f"legs of the GG slip (default {DEFAULT_N_GG})")
    p.add_argument(
        "--slip-mixed",
        type=int,
        default=DEFAULT_N_MIXED,
        metavar="N",
        help=f"legs of the mixed slip 1 / 2 / GG / Over 2.5 ({MIXED_MIN_LEGS}-{MIXED_MAX_LEGS}, default {DEFAULT_N_MIXED})",
    )
    p.add_argument(
        "--serie-d-demo",
        action="store_true",
        help="simulated full mode: add a synthetic Serie D registry (nine gironi of 18 clubs) to the perimeter so the Serie D section can be shown offline",
    )
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


def refresh_footystats(*, processed_dir: Path | None = None, odds_dir: Path | None = None) -> tuple[Path, Path, Path]:
    """``--footystats``: download the real data with the user's key, print the outcome; returns ``(matches, fixtures, odds dir)``.

    ``processed_dir`` / ``odds_dir`` default to ``data/processed`` / ``data/odds``
    (the module's :data:`PROCESSED_DIR` / :data:`ODDS_DIR`, read at call time).

    A competition that could not be downloaded is printed as a warning (the
    others are saved, its previous data is kept). When nothing could be
    downloaded and no earlier data exists the run stops with exit code 2 and
    the instructions to configure the key: the console never falls back to a
    simulated league silently. The key is never printed.
    """
    from src.dashboard import data_refresh

    processed_dir = PROCESSED_DIR if processed_dir is None else Path(processed_dir)
    odds_dir = ODDS_DIR if odds_dir is None else Path(odds_dir)

    def progress(fraction: float, message: str) -> None:
        print(f"  [{fraction:4.0%}] {message}", flush=True)

    print("Aggiornamento dati FootyStats (chiave da FOOTYSTATS_API_KEY o .env; i dati restano solo in locale)...")
    result = data_refresh.refresh(processed_dir, odds_dir, progress=progress)
    print(result.summary_text())
    for line in result.warnings():
        print(f"ATTENZIONE: {line}")
    for line in result.registry_sync:
        print(f"Dizionario squadre: {line}")
    matches = Path(processed_dir) / "matches.csv"
    if not matches.is_file():
        print(
            "Nessun dato reale disponibile: configura la chiave (copia .env.example in .env e scrivi FOOTYSTATS_API_KEY=...) "
            "e scegli le leghe sull'account footystats.org, poi riprova con --footystats.",
            file=sys.stderr,
        )
        raise SystemExit(2)
    if not result.written:
        print(f"Uso i dati FootyStats già scaricati in {processed_dir} ({data_refresh.refresh_label(processed_dir) or 'data sconosciuta'}).")
    return matches, Path(processed_dir) / "fixtures.csv", Path(odds_dir)


def main(argv: Sequence[str] | None = None) -> PipelineOutput:
    args = build_parser().parse_args(argv)
    configure_logging(args.log_level, json_output=args.log_json, force=True)
    if args.footystats and args.processed:
        build_parser().error("--footystats scarica i dati in data/processed: non va usato insieme a --processed")
    demo = not args.footystats and not args.processed
    if demo:
        print(DEMO_WARNING)
    try:
        registry = load_registry(args.season)
    except FileNotFoundError:
        build_parser().error(f"--season {args.season}: nessun file config/teams_<stagione>.json (disponibili: {', '.join(list_seasons())})")

    if args.rounds is not None and args.rounds < 1:
        build_parser().error(f"--rounds {args.rounds}: servono almeno 1 giornata giocata")
    for option, value in (("--n-draws", args.n_draws), ("--n-btts", args.n_btts), ("--slip-x", args.slip_x), ("--slip-gg", args.slip_gg)):
        if value < 1:
            build_parser().error(f"{option} {value}: serve almeno 1 partita")
    if not MIXED_MIN_LEGS <= args.slip_mixed <= MIXED_MAX_LEGS:
        build_parser().error(f"--slip-mixed {args.slip_mixed}: la schedina mista va da {MIXED_MIN_LEGS} a {MIXED_MAX_LEGS} partite")
    section_text = str(args.section).strip()
    try:
        wanted_sections = parse_sections(section_text)
    except argparse.ArgumentTypeError as exc:
        build_parser().error(f"--section {section_text}: {exc}")
    if section_text.lower() in SECTION_CHOICES:
        section_text = section_text.lower()
    if args.serie_d_demo and (args.processed or args.footystats or args.mode != "full"):
        build_parser().error("--serie-d-demo vale solo per la lega simulata di --mode full (senza --processed)")
    if args.serie_d_demo and not serie_d_gironi():
        build_parser().error("--serie-d-demo: nessuna competizione con family 'Serie D' in config/competitions.toml")
    as_of: date = args.as_of if args.as_of is not None else date.today()

    league: SimulatedLeague | None = None
    history_matches = empty_matches_frame()
    input_notes: list[str] = []
    sim_notes: list[str] = []
    now: datetime | None = None
    if args.footystats:
        processed, fixtures_path, odds_path = refresh_footystats()
        args.processed, args.fixtures, args.odds = processed, args.fixtures or fixtures_path, args.odds or odds_path
    if args.processed:
        if not Path(args.processed).is_file():
            build_parser().error(f"--processed {args.processed}: file non trovato")
        matches, fixtures, quotes = load_real_data(args.processed, args.fixtures, args.odds, notes=input_notes)
        xi = 0.0018 if args.xi is None else args.xi
        source = f"dati reali{' FootyStats' if args.footystats else ''}: {args.processed}"
        now = rome_midnight(args.as_of) if args.as_of is not None else datetime.now(UTC)
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
            if args.serie_d_demo:
                registry = serie_d_demo_registry(registry, seed=args.seed)
                sim_notes.append(
                    f"Demo Serie D: dizionario sintetico di {len(serie_d_gironi())} gironi x 18 squadre aggiunto al perimetro "
                    f"({registry.n_teams} squadre in tutto); le composizioni reali arrivano dalla raccolta FootyStats."
                )
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

    coverage: CoverageReport = coverage_report(matches, registry=registry, season=season_label)
    use_prior = prior_recommended(matches, registry=registry, season=season_label) if args.prior == "auto" else (args.prior == "on")
    if use_prior:
        # Thin sample (few current-season matches per current team, not few matches in total:
        # 100 matches over the 80-team perimeter are barely 2 each, and three seasons of history
        # do not make the current season any longer): a Gaussian prior (alpha/beta sd 0.30, home
        # advantage 0.25 +- 0.15, rho sd 0.10) keeps gamma and rho off their bounds. With a full
        # season the data dominate and the MAP estimate is practically the MLE. Competitions of
        # the registry without any match (never harvested) do not count as thin: the banner
        # names them as ignored.
        config = DixonColesConfig.with_gaussian_prior(xi=xi, method=args.method)
        source += (
            f" | stima MAP con prior gaussiana ({coverage.thinnest:.1f} partite per squadra nel girone "
            f"meno coperto della stagione {season_label}, soglia {PRIOR_MIN_MATCHES_PER_TEAM:g}"
            + (f"; {coverage.describe_ignored()}" if coverage.ignored else "")
            + ")"
        )
    else:
        config = DixonColesConfig(xi=xi, method=args.method)
    if args.l2 is not None:
        config = replace(config, l2_penalty=args.l2)

    available = sections_for(registry.gironi)  # the sections with teams, display order
    wanted = wanted_sections
    runnable = [sec for sec in available if sec in wanted]
    unavailable = [sec for sec in wanted if sec not in available and (section_text != "all" or sec == SECTION_D)]
    backtest = (not demo) if args.backtest is None else bool(args.backtest)
    odds_history = None
    if backtest and args.odds is not None:
        from src.dashboard import data_refresh  # noqa: PLC0415

        odds_history = data_refresh.load_odds_history_frame(Path(args.odds))

    _banner(f"SERIE C QUANT LAB -- {source}")
    print(perimeter_banner(registry, current, history))
    print(f"Sezioni: {', '.join(runnable) if runnable else 'nessuna'} (--section {section_text})")
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
        n_draws=args.n_draws,
        n_btts=args.n_btts,
        sections=runnable,
        now=now,
        demo=demo,
        backtest=backtest,
        backtest_seasons=args.backtest_seasons,
        prior=prior_setting(None if args.prior == "auto" else args.prior == "on"),
        odds_history=odds_history,
        recalibrate=bool(args.recalibration),
        slips=bool(args.slips),
        n_slip_x=args.slip_x,
        n_slip_gg=args.slip_gg,
        n_slip_mixed=args.slip_mixed,
    )
    out.notes = input_notes + sim_notes + out.notes
    for sec in unavailable:
        _banner(f"SEZIONE {sec.upper()} -- non disponibile")
        if sec == SECTION_D and serie_d_gironi():
            status = registry_serie_d_status(registry, current)
            print(f"Serie D {status}.")
            # the command carries --season when the perimeter season is not the harvester default, so it fills the file named here
            print(
                f"Comando: {serie_d_harvest_command(season=registry.season)} (oppure avvia.bat / avvia.command con la chiave FootyStats in .env): "
                f"la prima raccolta inizializza i {len(serie_d_gironi())} gironi in config/teams_{season_label.replace('/', '_')}.json dalle liste squadre dell'API."
            )
        elif sec == SECTION_D:
            print("Serie D non configurata: nessuna competizione con family 'Serie D' in config/competitions.toml.")
        else:
            print(f"Nessuna squadra del dizionario {season_label} nelle competizioni di {sec}.")
        # expected with the shipped dictionaries (Serie D is initialised by the harvester): not a warning, not a note
        log.info("section not available", extra={"section": sec, "season": season_label})

    if league is not None and args.mode == "full":
        _banner(f"VERIFICA: RECUPERO DEI PARAMETRI VERI DEL SIMULATORE (squadre {season_label})")
        out.recovery = recovery_report(league, out.params)
        print(out.recovery.to_string(index=False))
    if demo:
        print(DEMO_WARNING)
    return out


if __name__ == "__main__":
    main()
