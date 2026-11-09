"""Integration of the per-league sections, the per-country fit, the walk-forward reliability, the recalibration and the slips.

Covers the data layer (:mod:`src.dashboard.state`), the backtest cache of :mod:`src.dashboard.data_refresh`, the
dashboard (``AppTest`` on the demo and on processed real-data fixtures: the *Affidabilità modello* and *Schedine*
tabs, the recalibration toggle and its effect on the Giornata probabilities), the console (*AFFIDABILITÀ MODELLO*,
*SCHEDINE*, ``--section`` names, exports) and the aggregated CI summary of :mod:`src.ingestion.calendar_snapshot`.
"""
from __future__ import annotations

import io
import json
import re
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import main as pipeline
from src.common.schemas import Girone, Market, OddsQuote
from src.common.teams import DEFAULT_SEASON, TeamRegistry, load_registry
from src.dashboard import data_refresh
from src.dashboard.state import (
    ALL_SECTIONS_LABEL,
    COMPETITION_METRICS_COLUMNS,
    DEMO_EXPORT_COLUMN,
    MATCHDAY_COLUMNS,
    PICKS_RECORD_COLUMNS,
    RECALIBRATION_COLUMNS,
    RELIABILITY_EXPORT_COLUMNS,
    RELIABILITY_METRICS_COLUMNS,
    SECTION_B,
    SECTION_BC,
    SECTION_C,
    SLIPS_BY_MATCHDAY_COLUMNS,
    SLIPS_EXPORT_COLUMNS,
    SLIPS_RECORD_COLUMNS,
    PipelineState,
    ReliabilityReport,
    apply_recalibration,
    backtest_seasons,
    build_reliability,
    build_state,
    competition_metrics_table,
    default_slips_matchday,
    headline_for,
    matchday_table_for,
    matchdays,
    metrics_table_for,
    picks_record_table,
    picks_track_for,
    raw_column_name,
    raw_state,
    recalibration_table,
    reliability_export_frame,
    reliability_figure,
    reliability_table_for,
    slips_by_matchday_table,
    slips_csv_bytes,
    slips_export_frame,
    slips_for,
    slips_matchdays,
    slips_record_table,
    slips_track_for,
    value_track_for,
)
from src.engine.calibration import RecalibrationModel, fit_recalibration
from src.engine.fitgroups import GroupedFit
from src.engine.slips import INDEPENDENCE_NOTE, SLIPS_COLUMNS
from src.ingestion.harvester import save_fixtures, save_odds_history, save_processed
from src.simulation.generator import simulate_league

REPO_ROOT = Path(__file__).resolve().parents[1]
APP_PATH = REPO_ROOT / "src" / "dashboard" / "app.py"
BC_FAMILIES = ("Serie C", "Serie B")


def bc_registry(season: str = DEFAULT_SEASON) -> TeamRegistry:
    """The Serie B and Serie C part of the shipped dictionary (Serie D left out: a fast simulated perimeter)."""
    full = load_registry(season)
    return TeamRegistry.from_teams(full.season, {g.value: full.teams_by_girone[g] for g in full.gironi if g.family in BC_FAMILIES})


BC = bc_registry()


def forced_recalibration(frame: pd.DataFrame) -> RecalibrationModel:
    """A recalibration that enables every market (threshold below zero): the switch has something to switch."""
    return fit_recalibration(frame, min_gain=-1.0, min_eval=1)


def synthetic_odds_history(matches: pd.DataFrame, *, bookmaker: str = "HistBook") -> list[OddsQuote]:
    """Pre-match prices for every played match: a flat book with a ~5% margin (aggregates only are ever printed)."""
    quotes: list[OddsQuote] = []
    for mid in matches["match_id"].astype(str):
        quotes.append(OddsQuote(mid, bookmaker, Market.MATCH_ODDS, {"1": 2.25, "X": 3.25, "2": 3.20}))
        quotes.append(OddsQuote(mid, bookmaker, Market.BTTS, {"GG": 1.85, "NG": 1.92}))
        quotes.append(OddsQuote(mid, bookmaker, Market.OVER_UNDER_25, {"Over 2.5": 2.05, "Under 2.5": 1.75}))
    return quotes


@pytest.fixture(scope="module")
def demo_state() -> PipelineState:
    """A simulated Serie B + Serie C perimeter: 8 played rounds, one synthetic past season (one fit per country)."""
    return build_state(simulate=True, seed=13, n_rounds=8, history_seasons=1, registry=BC)


@pytest.fixture(scope="module")
def demo_report(demo_state: PipelineState) -> ReliabilityReport:
    """The walk-forward of the demo with a synthetic odds history for the comparison with the market."""
    odds = pd.DataFrame(
        [
            {"match_id": q.match_id, "bookmaker": q.bookmaker, "market": q.market.value, "selection": s, "odds": o}
            for q in synthetic_odds_history(demo_state.matches)
            for s, o in q.odds.items()
        ]
    )
    return build_reliability(demo_state.matches, registry=demo_state.registry, season_label=demo_state.season_label, odds_history=odds, key="test")


# ============================================================================ data layer
def test_the_production_fit_is_one_grouped_fit_per_country_with_lbfgsb_and_slsqp_as_an_option(demo_state: PipelineState) -> None:
    assert isinstance(demo_state.engine, GroupedFit) and demo_state.fit_groups == ("Italy",) and demo_state.fit_method == "L-BFGS-B"
    assert demo_state.params is demo_state.engine.params_, "one group: the engine's own parameters"
    slsqp = build_state(simulate=True, seed=13, n_rounds=8, history_seasons=1, registry=BC, method="SLSQP")
    assert slsqp.fit_method == "SLSQP" and slsqp.params.method == "SLSQP"
    a = {p.match_id: p.probabilities[Market.MATCH_ODDS]["X"] for p in demo_state.predictions}
    b = {p.match_id: p.probabilities[Market.MATCH_ODDS]["X"] for p in slsqp.predictions}
    assert a.keys() == b.keys() and max(abs(a[k] - b[k]) for k in a) < 5e-3, "same optimum, whatever the optimiser"
    # every view of a section only sees its own league
    assert matchday_table_for(demo_state, None, None, SECTION_B)["Competizione"].eq("Serie B").all()
    assert not matchday_table_for(demo_state, None, None, SECTION_C)["Competizione"].eq("Serie B").any()


def test_the_backtest_is_out_of_sample_and_its_tables_are_aggregates(demo_state: PipelineState, demo_report: ReliabilityReport) -> None:
    result = demo_report.result
    assert backtest_seasons(demo_state.matches, demo_state.season_label) == ("2025/26", "2026/27") == result.seasons
    assert result.n_matches > 500 and (result.frame["train_end"] < result.frame["date"]).all(), "never fitted on the future"
    assert set(result.frame["section"]) == {SECTION_C, SECTION_B}
    table = metrics_table_for(demo_report, SECTION_C)
    assert list(table.columns) == list(RELIABILITY_METRICS_COLUMNS) and len(table) == 5
    quoted = table.set_index("Mercato")
    assert quoted.loc["Esito finale 1X2", "Partite con quote"] > 0 and np.isfinite(quoted.loc["Esito finale 1X2", "Log loss mercato"])
    assert quoted.loc["Over/Under 1.5", "Partite con quote"] == 0, "no price, no market column"
    head = headline_for(demo_report, None)
    assert head["n"] == result.n_matches and 0.9 < head["log_loss"] < 1.2 and head["n_market"] > 0
    comp = competition_metrics_table(demo_report, None, Market.BTTS)
    assert list(comp.columns) == list(COMPETITION_METRICS_COLUMNS) and set(comp["Competizione"]) == {"Girone A", "Girone B", "Girone C", "Serie B"}
    bins = reliability_table_for(demo_report, SECTION_B, Market.MATCH_ODDS, "X")
    assert bins["n"].sum() == metrics_table_for(demo_report, SECTION_B)["Partite"].iloc[0] and (bins["wilson_low"] <= bins["observed"]).all()
    fig = reliability_figure(bins, "x", calibrated=bins)
    assert [t.name for t in fig.data] == ["Taratura perfetta", "Modello (fuori campione)", "Ricalibrato (fuori campione)"]
    picks = picks_track_for(demo_report)
    record = picks_record_table(picks, SECTION_C)
    assert list(record.columns) == list(PICKS_RECORD_COLUMNS) and record["Lista"].tolist() == ["Top 5 X", "Top 12 GG"]
    assert (record["Giornate"] > 0).all() and record["IC 95% (%)"].str.contains(" – ").all() and (record["Con quota"] > 0).all()
    slips_track = slips_track_for(demo_report, [SECTION_C, SECTION_B])
    slips = slips_record_table(slips_track)
    assert list(slips.columns) == list(SLIPS_RECORD_COLUMNS) and slips["Schedina"].tolist() == ["Schedina X · pareggi", "Schedina GG · goal", "Schedina mista"]
    by_day = slips_by_matchday_table(slips_track)
    assert list(by_day.columns) == list(SLIPS_BY_MATCHDAY_COLUMNS) and len(by_day) == 3 * slips["Giornate"].iloc[0]
    assert by_day["Vinta"].isin(["sì", "no"]).all() and by_day["Giornata"].is_monotonic_decreasing
    assert (by_day["Partite vinte"] <= by_day["Partite"]).all() and by_day["Quota book"].notna().any()
    value = value_track_for(demo_report, None, min_ev=0.0)
    assert value["Mercato"].iloc[0] == "Tutti" and value["Giocate"].iloc[0] > 0
    export = reliability_export_frame(demo_report)
    assert list(export.columns) == list(RELIABILITY_EXPORT_COLUMNS) and export["Sezione"].iloc[0] == ALL_SECTIONS_LABEL
    assert set(export["Sezione"]) == {ALL_SECTIONS_LABEL, SECTION_C, SECTION_B}
    text = export.to_csv(index=False)
    assert not any(mid in text for mid in result.frame["match_id"].head(50)), "aggregates only: no match id"
    assert not any(team in text for team in BC.canonical_names), "aggregates only: no team"
    assert list(recalibration_table(demo_report.recalibration).columns) == list(RECALIBRATION_COLUMNS)


def test_the_recalibration_switch_changes_the_probabilities_and_is_reversible(demo_state: PipelineState, demo_report: ReliabilityReport) -> None:
    model = forced_recalibration(demo_report.frame)
    assert model.enabled and "Calibrazione applicata: sì" in model.describe()
    calibrated = apply_recalibration(demo_state, model)
    assert calibrated.calibrated and not demo_state.calibrated and calibrated.raw_predictions == demo_state.predictions
    raw_x = {p.match_id: p.probabilities[Market.MATCH_ODDS]["X"] for p in demo_state.predictions}
    new_x = {p.match_id: p.probabilities[Market.MATCH_ODDS]["X"] for p in calibrated.predictions}
    assert any(abs(raw_x[k] - new_x[k]) > 1e-4 for k in raw_x), "the enabled markets move"
    assert all(abs(sum(p.probabilities[Market.MATCH_ODDS].values()) - 1.0) < 1e-9 for p in calibrated.predictions)
    assert len(calibrated.opportunities) == len(demo_state.opportunities) and calibrated.opportunities != demo_state.opportunities
    # the Giornata table: recalibrated probabilities with the original ones next to them
    table = matchday_table_for(calibrated, None, None, SECTION_C, compare_raw=True)
    raw_table = matchday_table_for(demo_state, None, None, SECTION_C, compare_raw=True)
    assert list(raw_table.columns) == list(MATCHDAY_COLUMNS), "a raw state adds nothing"
    assert table.columns.get_loc(raw_column_name("PX (%)")) == table.columns.get_loc("PX (%)") + 1
    to_play = table["PX (%)"].notna()
    assert (table.loc[to_play, "PX orig. (%)"].to_numpy() == raw_table.loc[to_play, "PX (%)"].to_numpy()).all()
    assert not np.allclose(table.loc[to_play, "PX (%)"].to_numpy(), raw_table.loc[to_play, "PX (%)"].to_numpy())
    # reversible, and a model without enabled markets changes nothing
    back = raw_state(calibrated)
    assert not back.calibrated and back.predictions == demo_state.predictions and back.opportunities == demo_state.opportunities
    assert apply_recalibration(demo_state, model.disabled()).predictions == demo_state.predictions
    assert apply_recalibration(calibrated, None).predictions == demo_state.predictions


def test_slips_of_a_matchday_over_chosen_sections(demo_state: PipelineState) -> None:
    c_days = matchdays(demo_state, SECTION_C)
    assert slips_matchdays(demo_state, [SECTION_C]) == c_days, "one section: the very matchdays of the other tabs"
    both = slips_matchdays(demo_state, [SECTION_C, SECTION_B])
    assert both and both[0].n_fixtures == c_days[0].n_fixtures + matchdays(demo_state, SECTION_B)[0].n_fixtures
    assert default_slips_matchday(both, c_days[0]) == both[0] and default_slips_matchday([], c_days[0]) is None
    assert [d.key for d in slips_matchdays(demo_state, [SECTION_BC])] == [d.key for d in both], "the legacy alias covers both sections"
    slips = slips_for(demo_state, both[0], [SECTION_C, SECTION_B])
    assert [s.kind for s in slips] == ["X", "GG", "MISTA"] and [s.n_legs for s in slips] == [5, 10, 6]
    x, gg, mixed = slips
    assert all(leg.selection == "X" for leg in x.legs) and all(leg.selection == "GG" for leg in gg.legs)
    assert {leg.selection for leg in mixed.legs} <= {"1", "2", "GG", "Over 2.5"}
    assert x.probability == pytest.approx(float(np.prod([leg.p_model for leg in x.legs])))
    assert mixed.expected_winning_legs == pytest.approx(sum(leg.p_model for leg in mixed.legs))
    only_b = slips_for(demo_state, both[0], [SECTION_B], n_x=3, n_gg=4, n_mixed=3)
    assert [s.n_legs for s in only_b] == [3, 4, 3] and all(leg.competition == Girone.SB for s in only_b for leg in s.legs)
    strict = slips_for(demo_state, both[0], [SECTION_C], min_probability=0.6)
    assert all(leg.p_model >= 0.6 for s in strict for leg in s.legs)
    assert [s.n_legs for s in slips_for(demo_state, None, [SECTION_C])] == [0, 0, 0]
    frame = slips_export_frame(slips, both[0], demo_state)
    assert list(frame.columns) == list(SLIPS_EXPORT_COLUMNS) and len(frame) == 21 and frame["Giornata"].eq(both[0].label).all()
    assert set(SLIPS_COLUMNS) <= set(frame.columns)
    demo_csv = pd.read_csv(io.BytesIO(slips_csv_bytes(slips, both[0], demo_state)))
    assert list(demo_csv.columns)[0] == DEMO_EXPORT_COLUMN, "a demo export is marked"
    assert "indipendenza" in INDEPENDENCE_NOTE


def test_the_backtest_cache_roundtrip_is_keyed_and_tolerant(tmp_path: Path, demo_report: ReliabilityReport) -> None:
    processed = tmp_path / "processed"
    processed.mkdir()
    path = data_refresh.save_backtest(demo_report.result, demo_report.recalibration, key="k1", processed_dir=processed)
    frame_path, meta_path = data_refresh.backtest_paths(processed)
    assert path == frame_path == tmp_path / "backtest.parquet" and meta_path == tmp_path / "backtest.json"
    loaded = data_refresh.load_backtest(processed, key="k1")
    assert loaded is not None
    result, model = loaded
    pd.testing.assert_frame_equal(result.frame, demo_report.result.frame)
    assert result.seasons == demo_report.result.seasons and model.to_dict()["markets"] == demo_report.recalibration.to_dict()["markets"]
    assert data_refresh.load_backtest(processed, key="other data") is None, "new data: recompute"
    meta_path.write_text("{not json", encoding="utf-8")
    assert data_refresh.load_backtest(processed, key="k1") is None, "unreadable: recompute, never raise"
    assert data_refresh.load_odds_history_frame(tmp_path / "missing").empty
    # the default location is next to data/processed, and git-ignored (the repository is public)
    import shutil
    import subprocess

    assert data_refresh.backtest_paths()[0].parent.name == "data"
    if shutil.which("git") is not None and (REPO_ROOT / ".git").exists():
        for ignored in ("data/backtest.parquet", "data/backtest.json"):
            assert subprocess.run(["git", "check-ignore", "--no-index", "-q", ignored], cwd=REPO_ROOT).returncode == 0, ignored


# ============================================================================ dashboard (AppTest)
def _demo_env(monkeypatch: pytest.MonkeyPatch, *, backtest: str, seed: str) -> None:
    """The demo on the Serie B + Serie C perimeter, 10 played rounds and no synthetic history (a fast walk-forward)."""
    import src.dashboard.state as state_module

    monkeypatch.setenv("SERIEC_DEMO", "1")
    monkeypatch.delenv("SERIEC_FORCE_SIMULATE", raising=False)
    monkeypatch.setenv("SERIEC_BACKTEST", backtest)
    monkeypatch.setenv("SERIEC_SIM_ROUNDS", "10")
    monkeypatch.setenv("SERIEC_HISTORY", "0")
    monkeypatch.setenv("SERIEC_SEED", seed)  # a cache key no other test uses
    monkeypatch.setenv("SERIEC_AS_OF", "2026-11-09")
    monkeypatch.setattr(data_refresh, "locate_key", lambda **_: None)
    original = state_module.build_state

    def wrapper(**kwargs):  # noqa: ANN003, ANN202
        kwargs.setdefault("registry", BC)
        return original(**kwargs)

    monkeypatch.setattr(state_module, "build_state", wrapper)


def _giornata_table(at) -> pd.DataFrame:  # noqa: ANN001
    return next(d.value for d in at.dataframe if "match_id" in d.value.columns and "Calcio d'inizio" in d.value.columns)


def test_app_reliability_tab_and_the_recalibration_toggle_drive_the_giornata_probabilities(monkeypatch: pytest.MonkeyPatch) -> None:
    from streamlit.testing.v1 import AppTest

    import src.dashboard.state as state_module

    _demo_env(monkeypatch, backtest="auto", seed="61")
    monkeypatch.setattr(state_module, "fit_recalibration", forced_recalibration)  # the model enables every market
    at = AppTest.from_file(str(APP_PATH), default_timeout=300)
    at.run()
    assert not at.exception, [e.value for e in at.exception]
    labels = [t.label for t in at.tabs]
    assert labels[5:7] == ["Schedine", "Affidabilità modello"]
    reliability = at.tabs[6]
    assert any(h.value == "Affidabilità del modello · Serie C" for h in reliability.subheader)
    assert any(m.label == "Partite valutate" and int(m.value) > 0 for m in reliability.metric)
    assert any(m.label.startswith("Log loss modello") for m in reliability.metric)
    assert any(list(d.value.columns) == list(RELIABILITY_METRICS_COLUMNS) for d in reliability.dataframe)
    assert any(list(d.value.columns) == list(PICKS_RECORD_COLUMNS) for d in reliability.dataframe)
    assert any(list(d.value.columns) == list(SLIPS_RECORD_COLUMNS) for d in reliability.dataframe)
    assert any("Poche vincite non sono una prova" in w.value for w in reliability.warning)
    assert any("Calibrazione applicata: sì" in s.value for s in reliability.success)
    assert at.selectbox(key="reliability_seasons").value == 2 and at.radio(key="reliability_scope").options == ["Serie C", ALL_SECTIONS_LABEL]
    # default ON (the model enabled markets): the Giornata table carries the original probabilities next to the new ones
    toggle = at.toggle(key="use_recalibration")
    assert toggle.value is True and not toggle.disabled
    on_table = _giornata_table(at)
    assert raw_column_name("PX (%)") in on_table.columns and raw_column_name("GG (%)") in on_table.columns
    assert any("Probabilità ricalibrate attive" in c.value for c in at.caption)
    assert any("Probabilità: ricalibrate" in e.value for e in at.sidebar.error)
    # OFF: the model's own probabilities, no extra column; they equal the 'orig.' column shown before
    toggle.set_value(False).run()
    assert not at.exception, [e.value for e in at.exception]
    off_table = _giornata_table(at)
    assert list(off_table.columns) == list(MATCHDAY_COLUMNS)
    merged = on_table.merge(off_table, on="match_id", suffixes=("_on", "_off"))
    to_play = merged["PX (%)_on"].notna()
    assert np.allclose(merged.loc[to_play, "PX orig. (%)"], merged.loc[to_play, "PX (%)_off"])
    assert not np.allclose(merged.loc[to_play, "PX (%)_on"], merged.loc[to_play, "PX (%)_off"])
    # every section together, another selection of the reliability chart (one rerun)
    at.radio(key="reliability_scope").set_value(ALL_SECTIONS_LABEL)
    at.selectbox(key="reliability_selection").set_value("GG (entrambe segnano)")
    at.run()
    assert not at.exception, [e.value for e in at.exception]
    assert any("Affidabilità del modello" in h.value for h in at.subheader)


def test_app_slips_tab_interactions_and_the_section_radio(monkeypatch: pytest.MonkeyPatch) -> None:
    from streamlit.testing.v1 import AppTest

    _demo_env(monkeypatch, backtest="manual", seed="67")
    at = AppTest.from_file(str(APP_PATH), default_timeout=300)
    at.run()
    assert not at.exception, [e.value for e in at.exception]
    radio = at.radio(key="section")
    assert radio.options[:2] == ["Serie C", "Serie B"] and radio.value == "Serie C"
    slips_tab = at.tabs[5]
    assert any(h.value == "Schedine della giornata" for h in slips_tab.subheader)
    sections = at.multiselect(key="slips_sections")
    assert sections.value == ["Serie C", "Serie B"], "default: every section with data"
    titles = [m.value for m in slips_tab.markdown if m.value.startswith("**Schedina")]
    assert titles == ["**Schedina X · pareggi** · 5 partite", "**Schedina GG · goal** · 10 partite", "**Schedina mista · 1 / 2 / GG / Over 2.5** · 6 partite"]
    legs = [d.value for d in slips_tab.dataframe if "Esito" in d.value.columns and "Nota" in d.value.columns]
    assert [len(t) for t in legs] == [5, 10, 6] and legs[0]["Esito"].eq("X").all() and legs[1]["Esito"].eq("GG").all()
    assert any(m.label == "Stake suggerito" for m in slips_tab.metric) and any(m.label == "Partite vincenti attese" for m in slips_tab.metric)
    assert any("Le multiple moltiplicano il margine del bookmaker" in c.value for c in slips_tab.caption)
    assert any("Track record non disponibile" in c.value for c in slips_tab.caption), "manual mode: no backtest until the button"
    assert any(b.label == "Scarica le tre schedine (CSV)" for b in at.get("download_button"))
    # lengths and sections in one rerun: Serie B alone, 3 X, 10 GG, 8 mixed legs
    at.slider(key="slips_n_x").set_value(3)
    at.slider(key="slips_n_mixed").set_value(8)
    at.multiselect(key="slips_sections").set_value(["Serie B"])
    at.run()
    assert not at.exception, [e.value for e in at.exception]
    legs = [d.value for d in at.tabs[5].dataframe if "Esito" in d.value.columns and "Nota" in d.value.columns]
    assert legs and all(t["Competizione"].eq("Serie B").all() for t in legs) and [len(t) for t in legs] == [3, 10, 8]
    # the minimum probability, and the manual backtest button that computes the track record
    at.slider(key="slips_min_probability").set_value(60)
    at.button(key="run_backtest").click()
    at.run()
    assert not at.exception, [e.value for e in at.exception]
    legs = [d.value for d in at.tabs[5].dataframe if "Esito" in d.value.columns and "Nota" in d.value.columns]
    assert all((t["P. Modello (%)"] >= 60.0).all() for t in legs)
    assert any(list(d.value.columns) == list(SLIPS_RECORD_COLUMNS) for d in at.tabs[5].dataframe)
    # the section radio: Serie B is its own page
    at.radio(key="section").set_value("Serie B").run()
    assert not at.exception, [e.value for e in at.exception]
    assert at.selectbox(key="girone").options == ["Tutti", "Serie B"]
    assert any(h.value == "Classifica di forza · Serie B" for h in at.subheader)


def test_app_reliability_on_processed_data_is_persisted_and_read_back(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    import streamlit as st
    from streamlit.testing.v1 import AppTest

    import src.dashboard.state as state_module

    league = simulate_league(seed=71, registry=BC, n_rounds=12)
    processed, odds = tmp_path / "processed", tmp_path / "odds"
    save_processed(league.matches, processed / "matches.csv")
    save_fixtures(league.fixtures, processed / "fixtures.csv")
    rows = [{"match_id": q.match_id, "bookmaker": q.bookmaker, "market": q.market.value, "selection": s, "odds": o} for q in league.odds for s, o in q.odds.items()]
    odds.mkdir()
    pd.DataFrame(rows).to_csv(odds / "simbook.csv", index=False)
    save_odds_history(synthetic_odds_history(league.matches), data_refresh.odds_history_file(odds))
    for name in ("SERIEC_DEMO", "SERIEC_FORCE_SIMULATE", "SERIEC_SIM_ROUNDS"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("SERIEC_BACKTEST", "auto")
    monkeypatch.setenv("SERIEC_PROCESSED_DIR", str(processed))
    monkeypatch.setenv("SERIEC_ODDS_DIR", str(odds))
    monkeypatch.setenv("SERIEC_AS_OF", "2026-11-01")
    monkeypatch.setattr(data_refresh, "locate_key", lambda **_: None)
    original = state_module.build_state
    monkeypatch.setattr(state_module, "build_state", lambda **kwargs: original(**{"registry": BC, **kwargs}))
    at = AppTest.from_file(str(APP_PATH), default_timeout=300)
    at.run()
    assert not at.exception, [e.value for e in at.exception]
    assert (tmp_path / "backtest.parquet").is_file() and (tmp_path / "backtest.json").is_file(), "persisted next to the data folder"
    meta = json.loads((tmp_path / "backtest.json").read_text(encoding="utf-8"))
    assert meta["seasons"] == ["2026/27"] and "recalibration" in meta
    metrics = next(d.value for d in at.tabs[6].dataframe if list(d.value.columns) == list(RELIABILITY_METRICS_COLUMNS))
    assert (metrics.set_index("Mercato").loc["Esito finale 1X2", "Partite con quote"] > 0), "the odds history feeds the market comparison"
    assert any("Singole di valore" in m.value for m in at.tabs[6].markdown)
    assert not any("Letto dal backtest salvato" in c.value for c in at.tabs[6].caption)
    # a new process (empty memory caches) reads the persisted backtest instead of refitting every week
    st.cache_resource.clear()
    again = AppTest.from_file(str(APP_PATH), default_timeout=300)
    again.run()
    assert not again.exception, [e.value for e in again.exception]
    assert any("Letto dal backtest salvato in data/backtest.parquet" in c.value for c in again.tabs[6].caption)


# ============================================================================ console
def test_console_sections_reliability_and_slips_with_exports(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    league = simulate_league(seed=73, registry=BC, n_rounds=12)
    matches, fixtures, odds = tmp_path / "matches.csv", tmp_path / "fixtures.csv", tmp_path / "odds"
    save_processed(league.matches, matches)
    save_fixtures(league.fixtures, fixtures)
    odds.mkdir()
    pd.DataFrame(
        [{"match_id": q.match_id, "bookmaker": q.bookmaker, "market": q.market.value, "selection": s, "odds": o} for q in league.odds for s, o in q.odds.items()]
    ).to_csv(odds / "simbook.csv", index=False)
    save_odds_history(synthetic_odds_history(league.matches), data_refresh.odds_history_file(odds))
    export = tmp_path / "reports" / "value.csv"
    base = ["--processed", str(matches), "--fixtures", str(fixtures), "--odds", str(odds), "--as-of", "2026-11-01", "--log-level", "ERROR"]
    out = pipeline.main([*base, "--section", "serie-b", "--export", str(export), "--slip-mixed", "4"])
    console = capsys.readouterr().out
    assert "Sezioni: Serie B (--section serie-b)" in console and list(out.sections) == [SECTION_B]
    assert console.index("2) STIMA DIXON-COLES") < console.index("AFFIDABILITÀ MODELLO") < console.index("3) RANKING") < console.index("SCHEDINE (Serie B")
    reliability_block = console.split("AFFIDABILITÀ MODELLO")[1].split("3) RANKING")[0]
    assert "--- Tutte le sezioni: punteggi per mercato" in reliability_block and "--- Serie B: punteggi per mercato" in reliability_block
    assert "Calibrazione applicata:" in reliability_block and "Poche vincite non sono una prova" in reliability_block
    # regressions (review): the model next to the market on the same quoted matches, and the top-k multiples per scope
    assert "Log loss modello (partite con quote)" in reliability_block and "confrontalo con «Log loss modello (partite con quote)»" in reliability_block
    assert "multiple top-k delle liste (Tutte le sezioni" in reliability_block and "multiple top-k delle liste (Serie B" in reliability_block
    assert "Top 2 GG" in reliability_block and "Top 5 X" in reliability_block
    assert not any(mid in reliability_block for mid in league.matches["match_id"].head(30)), "aggregates only"
    assert out.reliability is not None and out.reliability.has_odds and isinstance(out.engine, GroupedFit)
    names = {p.name for p in export.parent.iterdir()}
    assert {"value_affidabilita.csv", "value_schedine.csv", "value_selezioni.csv"} <= names, "one section: no slug"
    affidabilita = pd.read_csv(export.parent / "value_affidabilita.csv")
    assert list(affidabilita.columns) == list(RELIABILITY_EXPORT_COLUMNS) and (affidabilita["Partite con quote"] > 0).any()
    schedine = pd.read_csv(export.parent / "value_schedine.csv")
    assert list(schedine.columns) == list(SLIPS_EXPORT_COLUMNS) and set(schedine["Competizione"]) == {"Serie B"}
    assert [s.n_legs for s in out.slips] == [5, 10, 4] and out.slips_export_path == export.parent / "value_schedine.csv"
    # off switches, names and aliases of --section
    out = pipeline.main([*base, "--no-export", "--no-backtest", "--no-slips", "--section", "c,b"])
    console = capsys.readouterr().out
    assert "AFFIDABILITÀ MODELLO" not in console and "SCHEDINE (" not in console and out.reliability is None and out.slips == []
    assert list(out.sections) == [SECTION_C, SECTION_B] and "Sezioni: Serie C, Serie B (--section c,b)" in console
    assert pipeline.parse_sections("bc") == (SECTION_C, SECTION_B) == pipeline.parse_sections("Serie C,serie_b")
    assert pipeline.parse_sections("all")[:2] == (SECTION_C, SECTION_B)
    with pytest.raises(SystemExit) as exc:
        pipeline.main([*base, "--no-export", "--section", "eccellenza"])
    assert exc.value.code == 2 and "sezione sconosciuta" in capsys.readouterr().err
    with pytest.raises(SystemExit):
        pipeline.main([*base, "--no-export", "--slip-mixed", "11"])


def test_ci_reliability_summary_prints_aggregates_only(tmp_path: Path) -> None:
    from src.ingestion import calendar_snapshot as cs

    league = simulate_league(seed=79, registry=BC, n_rounds=14)
    matches = tmp_path / "matches.csv"
    save_processed(league.matches, matches)
    history = tmp_path / "history" / "odds_history.csv"
    save_odds_history(synthetic_odds_history(league.matches), history)
    text = cs.run_reliability(matches, season=DEFAULT_SEASON, odds_history_path=history)
    assert text.startswith("# Affidabilità del modello (backtest walk-forward, solo metriche aggregate)")
    for heading in ("## Punteggi per sezione e mercato", "## Taratura (tutte le sezioni)", "## Liste di giornata", "## Multiple top-k delle liste", "## Schedine", "## Ricalibrazione"):
        assert heading in text, heading
    assert "Calibrazione applicata:" in text and "| Tutte le sezioni | 1X2 |" in text
    assert not any(mid in text for mid in league.matches["match_id"]), "no match id in a public log"
    assert not any(team in text for team in BC.canonical_names), "no team name in a public log"
    assert not re.search(r"\b(2\.25|3\.25|3\.20|1\.85|1\.92|2\.05|1\.75)\b", text), "no raw price in a public log"
    summary = tmp_path / "ci" / "affidabilita.md"
    assert cs.main(["--reliability", "--matches", str(matches), "--odds-history", str(history), "--summary", str(summary), "--log-level", "ERROR"]) == 0
    assert summary.read_text(encoding="utf-8").startswith("# Affidabilità del modello")
    assert cs.main(["--reliability", "--matches", str(tmp_path / "missing.csv"), "--log-level", "ERROR"]) == 1


def test_rows_of_a_league_switched_off_never_take_the_dashboard_or_the_console_down(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """Regression (sections builder open issue): a league enabled, downloaded and then switched off again
    (``enabled = false``) leaves rows with an unknown code in the CSVs; the strict loaders refused the whole file and
    the dashboard fell back to the setup page. Those rows are now skipped with a note until the next refresh."""
    league = simulate_league(seed=83, registry=BC, n_rounds=6)
    processed = tmp_path / "processed"
    save_processed(league.matches, processed / "matches.csv")
    save_fixtures(league.fixtures, processed / "fixtures.csv")
    for name, n in (("matches.csv", 5), ("fixtures.csv", 2)):
        frame = pd.read_csv(processed / name, dtype=str, keep_default_na=False)
        frame.loc[: n - 1, "girone"] = "SA"  # Serie A: configured, disabled in the shipped file
        frame.to_csv(processed / name, index=False)
    state = build_state(processed_dir=processed, odds_dir=tmp_path / "odds", registry=BC, as_of=date_of(league))
    assert state.source == "processed" and len(state.matches) == len(league.matches) - 5
    assert any("5 partite di competizioni non abilitate (SA) ignorate in matches.csv" in n for n in state.notes)
    assert any("2 partite in calendario di competizioni non abilitate (SA) ignorate in fixtures.csv" in n for n in state.notes)
    out = pipeline.main(["--processed", str(processed / "matches.csv"), "--fixtures", str(processed / "fixtures.csv"), "--no-export", "--no-backtest", "--log-level", "ERROR"])
    assert len(out.matches) == len(league.matches) - 5 and any("non abilitate (SA)" in n for n in out.notes)
    capsys.readouterr()


def date_of(league) -> date:  # noqa: ANN001
    """The day before the league's first fixture (the calendar position of the processed-data checks)."""
    return min(f.date for f in league.fixtures) - timedelta(days=1)


# ============================================================================ review fixes
def _partial_odds(report: ReliabilityReport, section: str) -> ReliabilityReport:
    """The same backtest with pre-match odds for one section only (odds coverage differs by league)."""
    import dataclasses

    quoted_ids = set(report.frame.loc[report.frame["section"] == section, "match_id"].astype(str))
    odds = report.odds_history.loc[report.odds_history["match_id"].astype(str).isin(quoted_ids)].reset_index(drop=True)
    return dataclasses.replace(report, odds_history=odds)


def test_model_and_market_scores_are_compared_on_the_same_quoted_matches(demo_report: ReliabilityReport) -> None:
    """Regression (review: model and market log loss on different matches): with odds for Serie B only, the
    all-section model log loss covers B and C while the market covers B alone; the tables, the headline, the export,
    the console and the CI summary now carry the model's own scores on the quoted matches next to the market's."""
    from src.dashboard.state import QUOTED_BRIER_COLUMN, QUOTED_LOG_LOSS_COLUMN
    from src.engine.backtest import metrics as bt_metrics
    from src.ingestion import calendar_snapshot as cs

    report = _partial_odds(demo_report, SECTION_B)
    only_b = report.section_frame(SECTION_B)
    b_scores = bt_metrics(only_b, by=()).set_index("market")
    all_scores = bt_metrics(report.frame, by=()).set_index("market")
    table = metrics_table_for(report, None).set_index("Mercato")
    assert QUOTED_LOG_LOSS_COLUMN in RELIABILITY_METRICS_COLUMNS and QUOTED_BRIER_COLUMN in RELIABILITY_METRICS_COLUMNS
    row = table.loc["Esito finale 1X2"]
    assert row["Partite"] == len(report.frame) and row["Partite con quote"] == len(only_b) < row["Partite"]
    assert row["Log loss modello"] == pytest.approx(all_scores.loc["1X2", "log_loss"], abs=1e-4), "every evaluated match"
    assert row[QUOTED_LOG_LOSS_COLUMN] == pytest.approx(b_scores.loc["1X2", "log_loss"], abs=1e-4), "the quoted matches only"
    assert row[QUOTED_BRIER_COLUMN] == pytest.approx(b_scores.loc["1X2", "brier"], abs=1e-4)
    assert row[QUOTED_LOG_LOSS_COLUMN] != pytest.approx(row["Log loss modello"], abs=1e-4), "a different set of matches"
    # the skill is the comparison of the two numbers shown side by side
    assert row["Skill vs mercato (%)"] == pytest.approx((1 - row[QUOTED_LOG_LOSS_COLUMN] / row["Log loss mercato"]) * 100, abs=0.02)
    head = headline_for(report, None)
    assert head["n_market"] == len(only_b) and head["log_loss_model_market"] == pytest.approx(row[QUOTED_LOG_LOSS_COLUMN])
    assert head["brier_model_market"] == pytest.approx(row[QUOTED_BRIER_COLUMN]) and np.isfinite(head["brier_market"])
    comp = competition_metrics_table(report, None).set_index("Competizione")
    assert QUOTED_LOG_LOSS_COLUMN in COMPETITION_METRICS_COLUMNS
    assert comp.loc["Serie B", QUOTED_LOG_LOSS_COLUMN] == pytest.approx(comp.loc["Serie B", "Log loss modello"], abs=1e-4)
    assert np.isnan(comp.loc["Girone A", QUOTED_LOG_LOSS_COLUMN]) and comp.loc["Girone A", "Partite con quote"] == 0
    export = reliability_export_frame(report)
    assert QUOTED_LOG_LOSS_COLUMN in RELIABILITY_EXPORT_COLUMNS and QUOTED_BRIER_COLUMN in RELIABILITY_EXPORT_COLUMNS
    every = export.loc[(export["Sezione"] == ALL_SECTIONS_LABEL) & (export["Mercato"] == "1X2")].iloc[0]
    assert every[QUOTED_LOG_LOSS_COLUMN] == pytest.approx(b_scores.loc["1X2", "log_loss"], abs=1e-4)
    # the console table and the CI summary carry the same-match column
    assert QUOTED_LOG_LOSS_COLUMN in pipeline.CONSOLE_RELIABILITY_COLUMNS
    text = cs.reliability_markdown(report.result, report.recalibration, report.odds_history)
    assert "| LL modello (con quote) | LL mercato | Brier modello (con quote) | Brier mercato |" in text
    every_line = next(line for line in text.splitlines() if line.startswith(f"| {ALL_SECTIONS_LABEL} | 1X2 |"))
    assert f"| {b_scores.loc['1X2', 'log_loss']:.4f} |" in every_line, "the model on the quoted matches, not on all of them"


def test_the_market_card_compares_model_and_market_on_the_same_matches() -> None:
    """Regression (review: the headline cards compared the all-match model with the quoted-match market)."""
    from src.dashboard.state import market_card

    fake = {"log_loss": 0.95, "log_loss_market": 1.06, "log_loss_model_market": 1.08, "n_market": 138, "n": 276}
    card = market_card(fake)
    assert card["label"] == "Modello vs mercato · 138 partite con quote"
    assert card["value"] == "1.0800 vs 1.0600" and card["delta"] == "+0.0200" and card["delta_color"] == "inverse"
    assert "stesse 138 partite" in card["help"] and "276 valutate" in card["help"]
    assert "0.95" not in card["value"], "never the all-match model log loss next to the market's"
    missing = market_card({**fake, "log_loss_market": float("nan"), "log_loss_model_market": float("nan"), "n_market": 0})
    assert missing["value"] == "n.d." and missing["delta"] is None
    assert "c3.metric(**market_card(head))" in (REPO_ROOT / "src" / "dashboard" / "app.py").read_text(encoding="utf-8")


def _with_disabled_league(report: ReliabilityReport, code: str = "SA", n: int = 40) -> pd.DataFrame:
    """The backtest frame with ``n`` rows relabelled to a competition that is configured but switched off (``SA``)."""
    frame = report.frame.copy()
    rows = frame.index[:n]
    frame.loc[rows, "girone"] = code
    frame.loc[rows, "section"] = "Serie A"
    return frame


def test_a_saved_backtest_never_outlives_a_configuration_change(tmp_path: Path, demo_report: ReliabilityReport) -> None:
    """Regression (review: switching a league off brought back a saved backtest that crashed the page): the key carries
    the competition configuration and the logic version, the loader and the report drop rows of competitions that are
    no longer enabled, and every track-record helper works on what is left."""
    import dataclasses

    from src.common.competitions import ALL_COMPETITIONS
    from src.common.schemas import Girone as G
    from src.dashboard.state import BACKTEST_LOGIC_VERSION, config_fingerprint, known_competition_rows, reliability_key

    enabled = G.competitions()
    key = reliability_key("data:1.0", "2026_27", 2, "xi=0.0018", "prior=auto")
    assert key.startswith(f"{BACKTEST_LOGIC_VERSION}|cfg={config_fingerprint()}|data:1.0") and BACKTEST_LOGIC_VERSION != "v1"
    # switching a league off, or moving one to another section, changes the fingerprint (and so the key)
    assert config_fingerprint(enabled[:-1]) != config_fingerprint(enabled)
    moved = (dataclasses.replace(enabled[0], section="Altra sezione"), *enabled[1:])
    assert config_fingerprint(moved) != config_fingerprint(enabled)
    disabled = [c for c in ALL_COMPETITIONS if not c.enabled]
    if disabled:
        assert config_fingerprint((*enabled, dataclasses.replace(disabled[0], enabled=True))) != config_fingerprint(enabled)
    # a frame with rows of a league switched off since: dropped by the loader of the saved backtest ...
    stale = _with_disabled_league(demo_report)
    processed = tmp_path / "processed"
    processed.mkdir()
    data_refresh.save_backtest(dataclasses.replace(demo_report.result, frame=stale), demo_report.recalibration, key=key, processed_dir=processed)
    loaded = data_refresh.load_backtest(processed, key=key)
    assert loaded is not None and len(loaded[0].frame) == len(stale) - 40 and "SA" not in set(loaded[0].frame["girone"])
    assert data_refresh.load_backtest(processed, key=reliability_key("data:1.0", "2026_27", 2, "xi=0.0018", "prior=auto") + "x") is None
    # ... and by the report itself, whatever built it, so every view works on the enabled leagues only
    report = ReliabilityReport(result=dataclasses.replace(demo_report.result, frame=stale), recalibration=demo_report.recalibration, odds_history=demo_report.odds_history, key="stale")
    assert len(report.frame) == len(stale) - 40 and set(report.frame["girone"]) <= {g.value for g in G}
    assert known_competition_rows(demo_report.frame) is demo_report.frame, "nothing to drop: the same frame"
    picks = picks_track_for(report)
    assert not picks.picks.empty and "SA" not in set(picks.picks["girone"])
    assert not picks_track_for(demo_report, frame=stale).picks.empty, "a stale override frame is filtered too"
    assert not slips_track_for(report, None).slips.empty and not slips_track_for(demo_report, None, frame=stale).slips.empty
    assert len(competition_metrics_table(report, None)) and len(value_track_for(demo_report, None, min_ev=0.0, frame=stale))
    assert ALL_SECTIONS_LABEL in set(reliability_export_frame(report)["Sezione"]) and "Serie A" not in set(reliability_export_frame(report)["Sezione"])
    assert metrics_table_for(demo_report, None, frame=stale)["Partite"].iloc[0] == len(stale) - 40


def test_the_tuning_is_chosen_on_earlier_matchdays_and_confirmed_on_later_ones(demo_state: PipelineState) -> None:
    """Regression (review: tuned xi/prior chosen and reported on the same matches; applied for any gain): the grid point
    is chosen on the earlier matchdays, its gain is measured on the later ones it never saw, and it may be applied only
    above the recalibration's own threshold."""
    from src.dashboard.state import TUNING_COLUMNS, ValidatedTuning, tuning_disclosure, tuning_table, validated_tuning
    from src.engine.backtest import metrics as bt_metrics
    from src.engine.backtest import walk_forward
    from src.engine.calibration import DEFAULT_MIN_GAIN, time_split

    season = demo_state.season_label.replace("_", "/")
    grid = {"xi": (0.0018, 0.004), "prior": ("auto",)}
    tuned = validated_tuning(demo_state.matches, demo_state.registry, seasons=(season,), grid=grid)
    assert len(tuned.table) == 2 and tuned.split_date is not None and tuned.n_select > 0 and tuned.n_confirm > 0
    # the production run, cut at the split date: selection and confirmation are disjoint, consecutive in time
    prod = walk_forward(demo_state.matches, registry=demo_state.registry, seasons=[season]).frame
    early, late, cut = time_split(prod, test_fraction=0.3)
    assert cut == tuned.split_date and (tuned.n_select, tuned.n_confirm) == (len(early), len(late))
    objective = lambda f: float(bt_metrics(f, by=()).set_index("market").loc[["1X2", "GG/NG", "OU2.5"], "log_loss"].mean())  # noqa: E731
    assert tuned.production_select == pytest.approx(objective(early)) and tuned.production_confirm == pytest.approx(objective(late))
    # the winner is the argmin of the selection part only; its reported gain is the held-out one
    assert tuned.best_select == pytest.approx(tuned.table["objective_select"].min())
    assert tuned.improvement == pytest.approx((tuned.production_confirm - tuned.best_confirm) / tuned.production_confirm)
    assert tuned.accepted == (not tuned.is_production and tuned.improvement >= DEFAULT_MIN_GAIN)
    table = tuning_table(tuned)
    assert list(table.columns) == list(TUNING_COLUMNS) and table["Scelta"].eq("sì").sum() == 1 and table["Produzione"].eq("sì").sum() == 1
    assert "Conferma sulle" in tuned.describe() or tuned.is_production
    # the gate: +0.1% on held-out matches is noise, the 0.5% threshold of the recalibration is required
    base = dict(table=tuned.table, best={"xi": 0.001, "prior": "0.2"}, seasons=(season,), split_date=date(2026, 10, 1), n_select=900, n_confirm=400, production_select=0.80, best_select=0.79)
    noise = ValidatedTuning(**base, production_confirm=0.7963, best_confirm=0.7952)
    assert not noise.accepted and "è rumore" in noise.describe()
    real = ValidatedTuning(**base, production_confirm=0.80, best_confirm=0.79)
    assert real.accepted and real.settings == (0.001, "0.2") and "si può applicare" in real.describe()
    assert not ValidatedTuning(**{**base, "n_confirm": 40}, production_confirm=0.80, best_confirm=0.79).accepted, "too few held-out matches"
    assert not ValidatedTuning(**{**base, "best": {"xi": 0.0018, "prior": "auto"}}, production_confirm=0.80, best_confirm=0.80).accepted
    assert not ValidatedTuning(**{**base, "table": tuned.table.iloc[0:0]}, production_confirm=0.80, best_confirm=0.79).accepted, "no grid, nothing to apply"
    worse = ValidatedTuning(**base, production_confirm=0.80, best_confirm=0.81)
    assert worse.selection_gain > 0 and not worse.accepted, "a selection win that does not hold out is refused"
    text = tuning_disclosure(real)
    assert "scelti sulle 900 partite" in text and "01/10/2026" in text and "non sono del tutto fuori campione" in text and "400 partite" in text


def test_top_k_multiples_and_the_lists_matchday_by_matchday(demo_report: ReliabilityReport) -> None:
    """Regression (review: the top-k accumulators were computed but never shown): the multiple of the first k picks of
    every past matchday (the user's "top 2 GG of Serie C" bet) has its own table, per scope, with the matchday detail."""
    from src.dashboard.state import ACCUMULATORS_COLUMNS, PICKS_BY_MATCHDAY_COLUMNS, accumulators_table, picks_by_matchday_table

    track = picks_track_for(demo_report)
    table = accumulators_table(track, SECTION_C)
    assert list(table.columns) == list(ACCUMULATORS_COLUMNS)
    assert table["Multipla"].tolist() == ["Top 2 X", "Top 3 X", "Top 4 X", "Top 5 X", "Top 2 GG", "Top 3 GG", "Top 4 GG", "Top 5 GG"]
    acc = track.accumulators.set_index(["scope", "list", "k"])
    row = table.set_index("Multipla").loc["Top 2 GG"]
    expected = acc.loc[(SECTION_C, "GG", 2)]
    assert row["Giornate"] == expected["n_matchdays"] and row["Vinte"] == expected["wins"] and row["Attese"] == pytest.approx(expected["expected_wins"], abs=0.01)
    assert " – " in row["IC 95% (%)"] and row["Con quota"] > 0
    # the multiple of the top 2 is won exactly on the matchdays where the first two picks both won
    picks = track.picks.loc[(track.picks["scope"] == SECTION_C) & (track.picks["list"] == "GG") & (track.picks["rank"] <= 2)]
    assert row["Vinte"] == int(picks.groupby("matchday")["won"].all().sum())
    assert set(accumulators_table(track, None)["Multipla"]) == set(table["Multipla"]), "every section together too"
    days = picks_by_matchday_table(track, SECTION_C)
    assert list(days.columns) == list(PICKS_BY_MATCHDAY_COLUMNS) and set(days["Lista"]) == {"Top 5 X", "Top 12 GG"}
    assert days["Giornata"].is_monotonic_decreasing and (days["Vinte"] <= days["Selezioni"]).all() and days["Tutte vinte"].isin(["sì", "no"]).all()
    assert len(days) == 2 * track.picks.loc[track.picks["scope"] == SECTION_C, "matchday"].nunique()
    text = days.to_csv(index=False) + table.to_csv(index=False)
    assert not any(team in text for team in BC.canonical_names), "counts only: no team"


def test_the_rolling_recalibration_is_persisted_with_the_backtest_and_reused(tmp_path: Path, demo_report: ReliabilityReport, monkeypatch: pytest.MonkeyPatch) -> None:
    """Regression (review: the rolling recalibration ran on the first render with no progress and after every restart):
    it is attached to the report once, saved next to the backtest under the same key and read back, never refitted."""
    import src.dashboard.state as state_module
    from src.dashboard.state import rolling_frame, with_rolling
    from src.engine.calibration import RollingRecalibration

    processed = tmp_path / "processed"
    processed.mkdir()
    # a stand-in for the (slow) matchday-by-matchday refit: shifted probabilities, one step per matchday
    shifted = demo_report.frame.copy()
    shifted["p_X"] = shifted["p_X"] * 0.97
    days = sorted(shifted["matchday"].astype(str).unique())
    steps = pd.DataFrame({"matchday": days, "n_history": range(len(days)), "enabled": ["", *["1X2"] * (len(days) - 1)]})
    report = with_rolling(demo_report, RollingRecalibration(shifted, steps))
    rolling = report.rolling
    assert rolling is not None and len(rolling.frame) == len(demo_report.frame) and demo_report.rolling is None
    path = data_refresh.save_rolling(rolling, key="k1", processed_dir=processed)
    assert path == tmp_path / "backtest_rolling.parquet" == data_refresh.rolling_paths(processed)[0] and (tmp_path / "backtest_rolling.json").is_file()
    stored = pd.read_parquet(path)
    assert set(stored.columns) == {"match_id", *[c for c in demo_report.frame.columns if c.startswith("p_")]}, "only the probabilities"
    loaded = data_refresh.load_rolling(processed, key="k1", frame=demo_report.frame)
    assert loaded is not None
    pd.testing.assert_frame_equal(loaded.frame, rolling.frame, check_dtype=False)
    pd.testing.assert_frame_equal(loaded.steps, rolling.steps, check_dtype=False)
    assert data_refresh.load_rolling(processed, key="other", frame=demo_report.frame) is None, "another backtest: recompute"
    assert data_refresh.load_rolling(processed, key="k1", frame=demo_report.frame.iloc[:-3]) is None, "other matches: recompute"
    # a report that carries it never refits it
    monkeypatch.setattr(state_module, "rolling_recalibration", lambda *_a, **_k: pytest.fail("refitted"))
    assert rolling_frame(report) is rolling and with_rolling(report) is report
    import shutil
    import subprocess

    if shutil.which("git") is not None and (REPO_ROOT / ".git").exists():
        for ignored in ("data/backtest_rolling.parquet", "data/backtest_rolling.json"):
            assert subprocess.run(["git", "check-ignore", "--no-index", "-q", ignored], cwd=REPO_ROOT).returncode == 0, ignored


def _processed_app_env(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, *, seed: int) -> Path:
    """Processed real-data fixtures (Serie B + C, synthetic odds history) for an AppTest; returns the processed folder."""
    import src.dashboard.state as state_module

    league = simulate_league(seed=seed, registry=BC, n_rounds=12)
    processed, odds = tmp_path / "processed", tmp_path / "odds"
    save_processed(league.matches, processed / "matches.csv")
    save_fixtures(league.fixtures, processed / "fixtures.csv")
    odds.mkdir()
    save_odds_history(synthetic_odds_history(league.matches), data_refresh.odds_history_file(odds))
    for name in ("SERIEC_DEMO", "SERIEC_FORCE_SIMULATE", "SERIEC_SIM_ROUNDS"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("SERIEC_BACKTEST", "auto")
    monkeypatch.setenv("SERIEC_PROCESSED_DIR", str(processed))
    monkeypatch.setenv("SERIEC_ODDS_DIR", str(odds))
    monkeypatch.setenv("SERIEC_AS_OF", "2026-11-01")
    monkeypatch.setattr(data_refresh, "locate_key", lambda **_: None)
    original = state_module.build_state
    monkeypatch.setattr(state_module, "build_state", lambda **kwargs: original(**{"registry": BC, **kwargs}))
    monkeypatch.setattr(state_module, "fit_recalibration", forced_recalibration)  # a recalibration to roll
    return processed


def _later_tabs_rendered(at) -> bool:  # noqa: ANN001
    """The Export and Legenda tabs (after Affidabilità modello) rendered: the script did not stop on the way."""
    export, legend = at.tabs[7], at.tabs[8]
    return any(h.value.startswith("Export CSV") for h in export.subheader) and len(legend.markdown) > 0


def test_app_restart_reads_the_rolling_recalibration_and_survives_a_stale_or_failing_backtest(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Regression (review: the rolling recalibration was recomputed on every first render; a league switched off
    brought back a saved backtest that stopped the page): the rolling correction is persisted with the backtest and read
    back on restart; rows of a disabled league in the saved file are dropped; a failing view only warns."""
    import streamlit as st
    from streamlit.testing.v1 import AppTest

    import src.dashboard.state as state_module

    _processed_app_env(monkeypatch, tmp_path, seed=89)
    at = AppTest.from_file(str(APP_PATH), default_timeout=300)
    at.run()
    assert not at.exception, [e.value for e in at.exception]
    rolling_meta = json.loads((tmp_path / "backtest_rolling.json").read_text(encoding="utf-8"))
    backtest_meta = json.loads((tmp_path / "backtest.json").read_text(encoding="utf-8"))
    assert rolling_meta["key"] == backtest_meta["key"] and backtest_meta["key"].startswith("v2|cfg="), "same key, with the configuration"
    assert at.toggle(key="use_recalibration").value is True and _later_tabs_rendered(at)
    # restart: backtest and rolling correction both read from disk, nothing refitted on the first render
    st.cache_resource.clear()

    def refitted(*_a, **_k):  # noqa: ANN002, ANN003, ANN202
        raise AssertionError("rolling recalibration refitted on restart")

    with monkeypatch.context() as m:
        m.setattr(state_module, "rolling_recalibration", refitted)
        again = AppTest.from_file(str(APP_PATH), default_timeout=300)
        again.run()
    assert not again.exception, [e.value for e in again.exception]
    assert any("Letto dal backtest salvato" in c.value for c in again.tabs[6].caption) and _later_tabs_rendered(again)
    # a saved backtest with rows of a league switched off since (as written before the configuration change)
    frame = pd.read_parquet(tmp_path / "backtest.parquet")
    frame.loc[frame.index[:40], "girone"] = "SA"
    frame.loc[frame.index[:40], "section"] = "Serie A"
    frame.to_parquet(tmp_path / "backtest.parquet", index=False)
    st.cache_resource.clear()
    stale = AppTest.from_file(str(APP_PATH), default_timeout=300)
    stale.run()
    assert not stale.exception, [e.value for e in stale.exception]
    assert any(list(d.value.columns) == list(PICKS_RECORD_COLUMNS) for d in stale.tabs[6].dataframe) and _later_tabs_rendered(stale)
    # a view that fails on a bad report warns in its own tab; the next tabs still render
    def broken(*_a, **_k):  # noqa: ANN002, ANN003, ANN202
        raise ValueError("Unknown girone: 'SA'")

    monkeypatch.setattr(state_module, "picks_track_for", broken)
    monkeypatch.setattr(state_module, "slips_track_for", broken)
    st.cache_resource.clear()
    failing = AppTest.from_file(str(APP_PATH), default_timeout=300)
    failing.run()
    assert not failing.exception, [e.value for e in failing.exception]
    assert any("Affidabilità modello: vista non disponibile" in w.value for w in failing.tabs[6].warning)
    assert any("Track record delle schedine: vista non disponibile" in w.value for w in failing.tabs[5].warning)
    assert _later_tabs_rendered(failing)


def test_app_tuning_toggle_needs_a_confirmed_gain_and_the_tab_discloses_it(monkeypatch: pytest.MonkeyPatch) -> None:
    """Regression (review: the tuned xi/prior switch was enabled for any in-sample gain and the tab then reported the
    selection score as out of sample): the switch follows the held-out verdict and the tab says where the settings were
    chosen; the top-k multiples table is in the tab."""
    from streamlit.testing.v1 import AppTest

    import src.dashboard.state as state_module
    from src.dashboard.state import ACCUMULATORS_COLUMNS, ValidatedTuning

    _demo_env(monkeypatch, backtest="auto", seed="97")
    grid = pd.DataFrame(
        [
            {"xi": 0.003, "prior": "0.2", "n_select": 300, "objective_select": 0.82, "n_confirm": 150, "objective_confirm": 0.81, "seconds": 1.0, "production": False},
            {"xi": 0.0018, "prior": "auto", "n_select": 300, "objective_select": 0.83, "n_confirm": 150, "objective_confirm": 0.83, "seconds": 1.0, "production": True},
        ]
    )
    base = dict(table=grid, best={"xi": 0.003, "prior": "0.2"}, seasons=("2026/27",), split_date=date(2026, 10, 18), n_select=300, n_confirm=150, production_select=0.83, best_select=0.82)
    noise = ValidatedTuning(**base, production_confirm=0.8206, best_confirm=0.8199)
    confirmed = ValidatedTuning(**base, production_confirm=0.83, best_confirm=0.81)
    monkeypatch.setattr(state_module, "validated_tuning", lambda *_a, **_k: noise)
    at = AppTest.from_file(str(APP_PATH), default_timeout=300)
    at.run()
    assert not at.exception, [e.value for e in at.exception]
    assert any(list(d.value.columns) == list(ACCUMULATORS_COLUMNS) for d in at.tabs[6].dataframe), "top-k multiples shown"
    at.button(key="run_tuning").click().run()
    assert not at.exception, [e.value for e in at.exception]
    assert at.toggle(key="use_tuned").disabled, "a gain within noise cannot be applied"
    assert any("è rumore" in i.value for i in at.tabs[6].info)
    monkeypatch.setattr(state_module, "validated_tuning", lambda *_a, **_k: confirmed)
    at.button(key="run_tuning").click().run()
    assert not at.exception, [e.value for e in at.exception]
    toggle = at.toggle(key="use_tuned")
    assert not toggle.disabled
    toggle.set_value(True).run()
    assert not at.exception, [e.value for e in at.exception]
    assert any("scelti sulle 300 partite" in w.value and "non sono del tutto fuori campione" in w.value for w in at.tabs[6].warning)
