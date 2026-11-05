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

    assert len(out.matches) == 20
    assert list(out.matches.columns) == list(MATCH_COLUMNS)
    assert out.params.converged
    assert len(out.predictions) == len(out.fixtures) > 0
    # every quote selection was evaluated
    n_selections = sum(len(q.odds) for q in out.quotes)
    assert len(out.opportunities) == n_selections
    assert all(isinstance(o, ValueOpportunity) for o in out.opportunities)
    # the injected regression trap is detected and shown
    assert out.traps, "the mini demo injects a REGRESSION_TRAP; it must be detected"
    assert "REGRESSION_TRAP" in captured
    # all seven console sections are printed
    for section in ("1) DATASET", "2) STIMA DIXON-COLES", "3) RANKING", "4) PREVISIONI", "5) QUOTE BOOKMAKER", "6) ALERT REGRESSION_TRAP", "7) EXPORT"):
        assert section in captured, section
    # export is a well-formed CSV with one row per evaluated selection
    assert export.exists()
    with export.open(newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    assert len(rows) == n_selections
    assert {"match_id", "market", "selection", "book_odds", "fair_odds", "ev", "stake_units"} <= set(rows[0])


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
    out = pipeline.main(["--mode", "full", "--no-export", "--log-level", "ERROR", "--seed", "3"])
    assert out.recovery is not None and len(out.recovery) == 3
    assert (out.recovery["corr_alpha"] > 0.8).all()
    assert (out.recovery["corr_beta"] > 0.7).all()
    assert ((out.recovery["gamma_hat"] - out.recovery["gamma_true"]).abs() < 0.15).all()
    assert "RECUPERO DEI PARAMETRI" in capsys.readouterr().out


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
