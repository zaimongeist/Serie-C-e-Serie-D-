"""End-to-end tests for the console pipeline in ``main.py``."""
from __future__ import annotations

import csv
from pathlib import Path

import pandas as pd
import pytest

import main as pipeline
from src.common.schemas import MATCH_COLUMNS, Market, OddsQuote
from src.engine.value import ValueOpportunity
from src.simulation.generator import simulate_league


def test_mini_mode_runs_end_to_end(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    export = tmp_path / "value.csv"
    out = pipeline.main(["--mode", "mini", "--export", str(export), "--log-level", "ERROR"])
    captured = capsys.readouterr().out

    # 20 current-season matches plus the two synthetic past seasons of the 14 mini teams (default --history 2)
    assert len(out.current_matches) == 20 and len(out.history_matches) == 108
    assert len(out.matches) == 128 == out.params.n_matches
    assert list(out.matches.columns) == list(MATCH_COLUMNS)
    assert out.season_label == "2026/27" and "Perimetro: Serie C 2026/27 Gironi A/B/C (14 squadre)" in captured  # the mini demo has no Serie B
    assert out.params.converged
    assert len(out.predictions) == len(out.fixtures) > 0
    # every quote selection was evaluated
    n_selections = sum(len(q.odds) for q in out.quotes)
    assert len(out.opportunities) == n_selections
    assert all(isinstance(o, ValueOpportunity) for o in out.opportunities)
    # the injected regression trap is detected and shown
    assert out.traps, "the mini demo injects a REGRESSION_TRAP; it must be detected"
    assert "REGRESSION_TRAP" in captured
    # all console sections are printed; section 8 (matchday picks) comes after the trap alerts and before the export
    for section in ("1) DATASET", "2) STIMA DIXON-COLES", "3) RANKING", "4) PREVISIONI", "5) QUOTE BOOKMAKER", "6) ALERT REGRESSION_TRAP", "8) SELEZIONI DI GIORNATA", "7) EXPORT"):
        assert section in captured, section
    assert captured.index("6) ALERT REGRESSION_TRAP") < captured.index("8) SELEZIONI DI GIORNATA") < captured.index("7) EXPORT")
    assert pipeline.LEGEND_POINTER == "Legenda: tab Legenda della dashboard oppure README" and pipeline.LEGEND_POINTER in captured
    # export is a well-formed CSV with one row per evaluated selection
    assert export.exists()
    with export.open(newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    assert len(rows) == n_selections
    assert {"match_id", "market", "selection", "book_odds", "fair_odds", "ev", "stake_units"} <= set(rows[0])
    # the matchday picks: 5 draws and up to 12 GG of the next matchday (the mini demo has 7 fixtures), written next to the report
    assert len(out.matchdays) == 1 and out.next_matchday is not None and out.next_matchday.n_fixtures == len(out.predictions) == 7
    assert len(out.draw_picks) == 5 and len(out.btts_picks) == 7
    assert all(p.selection == "X" for p in out.draw_picks) and all(p.selection == "GG" for p in out.btts_picks)
    assert out.next_matchday.label in captured and "le 5 partite più probabili da X" in captured and "le 7 partite più probabili da GG" in captured
    assert out.picks_export_path == tmp_path / "value_selezioni.csv" == pipeline.picks_export_path(export) and out.picks_export_path.exists()
    picks = pd.read_csv(out.picks_export_path)
    assert list(picks.columns)[:3] == ["Lista", "Giornata", "#"] and picks["Lista"].tolist() == ["X"] * 5 + ["GG"] * 7
    assert picks["Giornata"].eq(out.next_matchday.label).all() and {"Partita", "P. Modello (%)", "Quota Fair", "Quota Book", "EV (%)", "Flag", "Nota"} <= set(picks.columns)
    assert f"selezioni di giornata scritte in: {out.picks_export_path}" in captured


def test_pick_list_sizes_are_flags(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    out = pipeline.main(["--mode", "mini", "--no-export", "--log-level", "ERROR", "--n-draws", "2", "--n-btts", "3"])
    captured = capsys.readouterr().out
    assert len(out.draw_picks) == 2 and len(out.btts_picks) == 3 and out.picks_export_path is None
    assert "le 2 partite più probabili da X" in captured and "le 3 partite più probabili da GG" in captured
    assert out.draw_picks[0].p_model >= out.draw_picks[1].p_model
    with pytest.raises(SystemExit) as exc:
        pipeline.main(["--mode", "mini", "--no-export", "--n-draws", "0"])
    assert exc.value.code == 2 and "--n-draws 0" in capsys.readouterr().err


def test_mini_mode_predictions_are_sane() -> None:
    """A 20-match sample must not yield absurd rates: the ridge prior keeps the fit regular."""
    out = pipeline.main(["--mode", "mini", "--no-export", "--log-level", "ERROR"])
    rates = [r for p in out.predictions for r in (p.home_rate, p.away_rate)]
    assert max(rates) < 4.0 and min(rates) > 0.2
    for p in out.predictions:
        for market, probs in p.probabilities.items():
            assert abs(sum(probs.values()) - 1.0) < 1e-9, market
    assert -0.2 <= out.params.rho <= 0.2


def test_full_mode_reports_parameter_recovery(capsys: pytest.CaptureFixture[str]) -> None:
    # --rounds fixes 30 played rounds whatever today's date (the default follows the calendar as of today)
    out = pipeline.main(["--mode", "full", "--no-export", "--log-level", "ERROR", "--seed", "3", "--rounds", "30"])
    console = capsys.readouterr().out
    assert out.recovery is not None and len(out.recovery) == 4  # three Serie C gironi + Serie B
    assert out.recovery["competizione"].tolist() == ["Girone A", "Girone B", "Girone C", "Serie B"]
    assert len(out.current_matches) == 1200 and "30 giornate giocate (--rounds)" in console
    assert "Perimetro: Serie C 2026/27 Gironi A/B/C + Serie B (80 squadre)" in console
    assert (out.recovery["corr_alpha"] > 0.8).all()
    assert (out.recovery["corr_beta"] > 0.7).all()
    assert ((out.recovery["gamma_hat"] - out.recovery["gamma_true"]).abs() < 0.15).all()
    assert "RECUPERO DEI PARAMETRI" in console


def test_kelly_and_threshold_flags_are_applied() -> None:
    quarter = pipeline.main(["--mode", "mini", "--no-export", "--log-level", "ERROR", "--kelly", "0.25", "--min-ev", "0.05"])
    eighth = pipeline.main(["--mode", "mini", "--no-export", "--log-level", "ERROR", "--kelly", "0.125", "--min-ev", "0.05"])
    value_q = [o for o in quarter.opportunities if o.value]
    value_e = [o for o in eighth.opportunities if o.value]
    assert value_q and all(o.ev >= 0.05 for o in value_q)
    assert len(value_q) == len(value_e)
    # a quarter Kelly stakes at least as much as an eighth Kelly on the same (uncapped) bet
    by_key_e = {(o.match_id, o.market, o.selection): o.stake_units for o in value_e}
    assert all(o.stake_units >= by_key_e[(o.match_id, o.market, o.selection)] - 1e-9 for o in value_q)


def test_real_data_path_loads_processed_csv_and_odds(tmp_path: Path) -> None:
    """``--processed/--fixtures/--odds`` must consume the harvester's CSV layout."""
    from src.ingestion.harvester import save_fixtures, save_processed

    league = simulate_league(seed=9, n_rounds=12)
    processed = tmp_path / "matches.csv"
    fixtures = tmp_path / "fixtures.csv"
    odds_dir = tmp_path / "odds"
    odds_dir.mkdir()
    save_processed(league.matches, processed)
    save_fixtures(league.fixtures, fixtures)
    rows = [
        {"match_id": q.match_id, "bookmaker": q.bookmaker, "market": q.market.value, "selection": sel, "odds": o}
        for q in league.odds
        for sel, o in q.odds.items()
    ]
    pd.DataFrame(rows).to_csv(odds_dir / "simbook.csv", index=False)

    out = pipeline.main(
        ["--processed", str(processed), "--fixtures", str(fixtures), "--odds", str(odds_dir), "--no-export", "--log-level", "ERROR"]
    )
    assert len(out.matches) == len(league.matches)
    assert len(out.fixtures) == len(league.fixtures)
    assert len(out.quotes) == len(league.odds)
    assert all(isinstance(q, OddsQuote) for q in out.quotes)
    assert len(out.opportunities) == sum(len(q.odds) for q in league.odds)
    assert out.params.xi == pytest.approx(0.0018)


def test_load_odds_dir_skips_malformed_groups(tmp_path: Path) -> None:
    good = {"match_id": "M1", "bookmaker": "B", "market": Market.MATCH_ODDS.value}
    rows = [dict(good, selection=s, odds=o) for s, o in (("1", 2.1), ("X", 3.3), ("2", 3.4))]
    rows.append({"match_id": "M2", "bookmaker": "B", "market": "1X2", "selection": "1", "odds": 1.9})  # incomplete
    rows.append({"match_id": "M3", "bookmaker": "B", "market": "OU2.5", "selection": "Over 2.5", "odds": 0.9})  # invalid odds
    rows.append({"match_id": "M3", "bookmaker": "B", "market": "OU2.5", "selection": "Under 2.5", "odds": 1.9})
    pd.DataFrame(rows).to_csv(tmp_path / "odds.csv", index=False)
    (tmp_path / "junk.csv").write_text("a,b\n1,2\n", encoding="utf-8")
    quotes = pipeline.load_odds_dir(tmp_path)
    assert [q.match_id for q in quotes] == ["M1"]
    assert quotes[0].odds == {"1": 2.1, "X": 3.3, "2": 3.4}


# ============================================================================ review findings (robustness of the console pipeline)
def _save_league(tmp_path: Path, league, *, odds: bool = False) -> tuple[Path, Path, Path]:
    """Write a simulated league in the harvester's CSV layout; returns (matches, fixtures, odds_dir)."""
    from src.ingestion.harvester import save_fixtures, save_processed

    processed, fixtures, odds_dir = tmp_path / "matches.csv", tmp_path / "fixtures.csv", tmp_path / "odds"
    save_processed(league.matches, processed)
    save_fixtures(league.fixtures, fixtures)
    odds_dir.mkdir()
    if odds:
        rows = [
            {"match_id": q.match_id, "bookmaker": q.bookmaker, "market": q.market.value, "selection": sel, "odds": o}
            for q in league.odds
            for sel, o in q.odds.items()
        ]
        pd.DataFrame(rows).to_csv(odds_dir / "simbook.csv", index=False)
    return processed, fixtures, odds_dir


def test_auto_prior_is_based_on_matches_per_team_not_total_matches(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """120 matches over 60 teams are 2 per team: the old ``len(matches) < 100`` rule switched the
    prior off and the pure MLE pinned rho/gamma on their bounds (home multiplier x3). The rule now
    looks at the matches per team of the least-covered girone."""
    league = simulate_league(seed=7, n_rounds=4)
    assert len(league.matches) == 160  # 4 rounds x (30 Serie C + 10 Serie B) matches
    assert pipeline.matches_per_team(league.matches) == pytest.approx(4.0)
    assert pipeline.prior_recommended(league.matches)
    # a full half season (15 per team) is where the MLE takes over; a single thin girone keeps the prior on
    half = simulate_league(seed=7, n_rounds=15).matches
    assert pipeline.matches_per_team(half) == pytest.approx(15.0) and not pipeline.prior_recommended(half)
    thin = half[(half["girone"] != "C") | (half["round"] <= 2)]
    assert pipeline.matches_per_team(thin) == pytest.approx(2.0) and pipeline.prior_recommended(thin)
    assert pipeline.matches_per_team(half.iloc[0:0]) == 0.0

    processed, fixtures, _ = _save_league(tmp_path, league)
    out = pipeline.main(["--processed", str(processed), "--fixtures", str(fixtures), "--no-export", "--log-level", "ERROR"])
    banner = capsys.readouterr().out
    assert out.params.penalty > 0.0, "auto prior must be on with 4 matches per team"
    assert "stima MAP con prior gaussiana (4.0 partite per squadra" in banner
    assert pipeline.parameters_at_bounds(out.params, out.engine.config) == []
    assert -0.15 < out.params.rho < 0.15
    for g, gamma in out.params.gamma.items():
        assert abs(gamma - league.true_params.gamma[g]) < 0.25, (g, gamma)
    assert max(r for p in out.predictions for r in (p.home_rate, p.away_rate)) < 4.0

    # 30 played rounds (30 per team) still run the pure MLE under 'auto'; as of 2026-11-06 only 11 rounds are
    # played, so the calendar-anchored run keeps the prior on (the synthetic history never counts as coverage)
    full = pipeline.main(["--mode", "full", "--no-export", "--log-level", "ERROR", "--seed", "3", "--rounds", "30"])
    assert full.params.penalty == 0.0 and not pipeline.prior_recommended(full.matches)
    today = pipeline.main(["--mode", "full", "--no-export", "--log-level", "ERROR", "--seed", "3", "--as-of", "2026-11-06"])
    assert len(today.current_matches) == 440 and today.params.penalty > 0.0
    assert pipeline.prior_recommended(today.matches, registry=today.registry, season=today.season_label)


def test_parameters_on_optimiser_bounds_are_reported(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """With ``--prior off`` on 2 matches per team the MLE is degenerate: the console must say so."""
    league = simulate_league(seed=7, n_rounds=4)
    processed, fixtures, _ = _save_league(tmp_path, league)
    out = pipeline.main(["--processed", str(processed), "--fixtures", str(fixtures), "--prior", "off", "--no-export", "--log-level", "ERROR"])
    captured = capsys.readouterr().out
    hits = pipeline.parameters_at_bounds(out.params, out.engine.config)
    assert len(hits) >= 2, hits  # which parameters hit a bound depends on the draw (beta/gamma/rho); the report must name them
    assert "ATTENZIONE" in captured and "parametri sui limiti dell'ottimizzatore" in captured
    assert all(hit in captured for hit in hits[:6]) and f"{len(hits)} parametri" in captured
    assert any("limiti dell'ottimizzatore" in n for n in out.notes)
    # the helper itself
    from dataclasses import replace as dc_replace

    inside = dc_replace(out.params, rho=0.0, alpha={t: 0.1 for t in out.params.alpha}, beta={t: -0.1 for t in out.params.beta}, gamma={g: 0.3 for g in out.params.gamma})
    assert pipeline.parameters_at_bounds(inside, out.engine.config) == []
    on_gamma = dc_replace(inside, gamma={g: out.engine.config.home_bounds[1] for g in inside.gamma})
    assert pipeline.parameters_at_bounds(on_gamma, out.engine.config) == [f"gamma[{g.value}]" for g in inside.gamma]


def test_fixture_with_unknown_team_is_skipped_not_fatal(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """A fixture naming a team (or girone) absent from the fit used to kill the run with KeyError
    after the ranking was printed; like the dashboard, the console now skips it with a note.
    (``main()`` re-installs the log handler on ``sys.stderr``, so warnings are read from ``capsys``.)"""
    from src.common.schemas import Fixture, Girone

    league = simulate_league(seed=9, n_rounds=12)
    first = league.fixtures[0]
    league.fixtures.append(Fixture("HAND-1", first.date, first.round, Girone.A, "Nowhere FC", first.away_team))  # not in any registry
    girone_c_free = league.matches[league.matches["girone"] != "C"].reset_index(drop=True)
    league.matches = girone_c_free  # girone C never harvested, but its fixtures are still listed
    processed, fixtures, odds_dir = _save_league(tmp_path, league, odds=True)

    out = pipeline.main(["--processed", str(processed), "--fixtures", str(fixtures), "--odds", str(odds_dir), "--no-export", "--log-level", "WARNING"])
    captured, err = capsys.readouterr()
    n_c = sum(1 for f in league.fixtures if f.girone == Girone.C)
    assert n_c > 0
    assert len(out.fixtures) == len(league.fixtures)
    assert len(out.predictions) == len(league.fixtures) - n_c - 1
    assert all(p.fixture.match_id != "HAND-1" and p.fixture.girone != Girone.C for p in out.predictions)
    assert "Nowhere FC - " in captured and "partita saltata, squadra non presente nello storico (Nowhere FC)" in captured
    assert "girone C" in captured
    assert any(n.startswith(f"{n_c + 1} partite in programma saltate") for n in out.notes)
    assert err.count("WARNING main: fixture not predictable") == n_c + 1 and "match_id=HAND-1" in err
    # the remaining fixtures are still evaluated against the odds
    assert out.opportunities and all(o.match_id != "HAND-1" for o in out.opportunities)
    # the odds for the skipped fixtures are simply not evaluated (the value engine has no prediction for them)
    assert "5) QUOTE BOOKMAKER" in captured and "6) ALERT REGRESSION_TRAP" in captured


def test_missing_explicit_fixtures_or_odds_path_warns(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """A typo in --fixtures/--odds used to be indistinguishable from an empty season (exit 0, silence)."""
    league = simulate_league(seed=9, n_rounds=12)
    processed, _, _ = _save_league(tmp_path, league)
    missing_fx, missing_odds = tmp_path / "nope" / "fixtures.csv", tmp_path / "nope" / "odds"
    out = pipeline.main(["--processed", str(processed), "--fixtures", str(missing_fx), "--odds", str(missing_odds), "--no-export", "--log-level", "WARNING"])
    captured, err = capsys.readouterr()
    assert out.fixtures == [] and out.quotes == [] and out.predictions == []
    assert f"ATTENZIONE: --fixtures {missing_fx} non trovato" in captured
    assert f"ATTENZIONE: --odds {missing_odds} non trovato" in captured
    assert [n for n in out.notes if "non trovato" in n] == [n for n in out.notes[:2]]
    assert err.count("WARNING main: input path not found") == 2 and "option=--fixtures" in err and "option=--odds" in err
    # ... while not passing them at all stays silent, as before
    quiet = pipeline.main(["--processed", str(processed), "--no-export", "--log-level", "WARNING"])
    quiet_out, quiet_err = capsys.readouterr()
    assert "non trovato" not in quiet_out and "input path not found" not in quiet_err and quiet.notes == []
    # a missing --processed is a hard error, not a traceback
    with pytest.raises(SystemExit) as exc:
        pipeline.main(["--processed", str(tmp_path / "nope.csv"), "--no-export"])
    assert exc.value.code == 2 and "file non trovato" in capsys.readouterr().err


def test_load_odds_dir_tolerates_empty_file_and_reports_blank_and_duplicate_rows(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    """``pd.read_csv`` on a zero-byte export raised EmptyDataError and aborted the run; the shared
    loader skips it. Blank key cells and duplicated selections are reported instead of vanishing."""
    import logging

    good = {"match_id": "M1", "bookmaker": "Bet365", "market": "1X2"}
    rows = [dict(good, selection=s, odds=o) for s, o in (("1", 2.1), ("X", 3.3), ("2", 3.4))]
    rows.append(dict(good, selection="1", odds=2.2))  # same selection quoted twice: last price wins
    rows.append({"match_id": "M1", "bookmaker": None, "market": "OU2.5", "selection": "Over 2.5", "odds": 1.9})  # blank bookmaker
    rows.append({"match_id": "M1", "bookmaker": None, "market": "OU2.5", "selection": "Under 2.5", "odds": 1.9})
    pd.DataFrame(rows).to_csv(tmp_path / "book.csv", index=False)
    (tmp_path / "aborted_export.csv").write_bytes(b"")
    notes: list[str] = []
    with caplog.at_level(logging.WARNING):
        quotes = pipeline.load_odds_dir(tmp_path, notes=notes)
    assert [(q.match_id, q.bookmaker, q.market) for q in quotes] == [("M1", "Bet365", Market.MATCH_ODDS)]
    assert quotes[0].odds["1"] == 2.2
    messages = [r.getMessage() for r in caplog.records]
    assert "odds file unreadable, skipped" in messages  # the empty CSV
    assert "odds rows with empty key cells dropped" in messages
    assert "duplicate odds rows: only the last price per selection is kept" in messages
    assert len(notes) == 2 and "2 righe" in notes[0] and "M1/Bet365/1X2/1" in notes[1]
    # same result through the CLI path, and the notes reach the console
    assert pipeline.load_odds_dir(tmp_path) == quotes


def test_console_value_table_names_the_bookmaker(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """With several books quoting the same selection the rows were indistinguishable."""
    league = simulate_league(seed=9, n_rounds=12)
    processed, fixtures, odds_dir = _save_league(tmp_path, league, odds=True)
    second = pd.read_csv(odds_dir / "simbook.csv")
    second["bookmaker"] = "OtherBook"
    second["odds"] = (second["odds"] * 1.08).round(2)  # a better price everywhere
    second.to_csv(odds_dir / "otherbook.csv", index=False)

    out = pipeline.main(["--processed", str(processed), "--fixtures", str(fixtures), "--odds", str(odds_dir), "--no-export", "--log-level", "ERROR"])
    captured = capsys.readouterr().out
    value = [o for o in out.opportunities if o.value]
    assert value and {o.bookmaker for o in value} >= {"OtherBook"}
    table = captured.split("5) QUOTE BOOKMAKER")[1].split("riepilogo:")[0]
    header = next(line for line in table.splitlines() if "book_odds" in line)
    assert header.split().index("bookmaker") == header.split().index("book_odds") - 1
    assert "OtherBook" in table
