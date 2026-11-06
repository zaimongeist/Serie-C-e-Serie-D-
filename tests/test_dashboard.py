"""Tests for the Streamlit dashboard: data layer (``src.dashboard.state``) and UI (``src.dashboard.app``)."""
from __future__ import annotations

import base64
import dataclasses
import inspect
import io
import json
import os
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
from datetime import date
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
import plotly.colors as pc
import plotly.graph_objects as go
import plotly.io as pio
import pytest

from src.common.schemas import SELECTIONS, Girone, Market, OddsQuote
from src.dashboard import LEGACY_FULL_WIDTH, MODERN_FULL_WIDTH, container_width_kwargs
from src.dashboard.state import (
    MARKETS_COLUMNS,
    MIN_SIM_ROUNDS,
    RANKING_COLUMNS,
    TRAPS_COLUMNS,
    VALUE_COLUMNS,
    PipelineState,
    build_state,
    export_csv_bytes,
    fixtures_for,
    heatmap_figure,
    load_odds_quotes,
    markets_table,
    max_sim_rounds,
    no_odds_message,
    opportunities_table,
    predictions_table,
    quotes_for,
    ranking_table,
    strength_figure,
    traps_table,
    value_table,
)
from src.ingestion.harvester import save_fixtures, save_processed
from src.simulation.generator import simulate_league, simulate_mini_dataset

if TYPE_CHECKING:
    from streamlit.testing.v1 import AppTest

REPO_ROOT = Path(__file__).resolve().parents[1]
APP_PATH = REPO_ROOT / "src" / "dashboard" / "app.py"


# ----------------------------------------------------------------------------- helpers
def write_odds_csv(quotes: list[OddsQuote], path: Path) -> Path:
    """Write quotes in the documented odds format (one row per selection)."""
    rows = [
        {"match_id": q.match_id, "bookmaker": q.bookmaker, "market": q.market.value, "selection": sel, "odds": odds}
        for q in quotes
        for sel, odds in q.odds.items()
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows, columns=["match_id", "bookmaker", "market", "selection", "odds"]).to_csv(path, index=False)
    return path


def plotly_array(value: object) -> np.ndarray:
    """Decode a Plotly JSON array (plain list or the ``{dtype, bdata, shape}`` typed-array encoding)."""
    if isinstance(value, dict) and "bdata" in value:
        arr = np.frombuffer(base64.b64decode(value["bdata"]), dtype=np.dtype(value["dtype"]))
        if "shape" in value:
            arr = arr.reshape([int(s) for s in str(value["shape"]).split(",")])
        return arr
    return np.asarray(value)


def rendered_heatmap(at: AppTest) -> tuple[np.ndarray, dict]:
    """``(z in %, layout)`` of the ``key="heatmap"`` chart as serialised to the Streamlit front end."""
    charts = [c for c in at.get("plotly_chart") if c.proto.id.endswith("-heatmap")]
    assert len(charts) == 1, [c.proto.id for c in at.get("plotly_chart")]
    spec = json.loads(charts[0].proto.spec)
    (trace,) = spec["data"]
    assert trace["type"] == "heatmap"
    return plotly_array(trace["z"]).astype(float), spec["layout"]


@pytest.fixture(scope="module")
def sim_state() -> PipelineState:
    """One simulated pipeline run shared by the view tests (seed 7, 15 rounds, stationary)."""
    return build_state(simulate=True, seed=7, xi=0.0, n_rounds=15)


# ----------------------------------------------------------------------------- build_state
def test_build_state_simulated_injects_trap(sim_state: PipelineState) -> None:
    state = sim_state
    assert state.source == "simulated"
    assert len(state.predictions) == len(state.fixtures) > 0
    assert state.params.converged
    assert state.n_matches == len(state.matches) > 0
    assert state.traps, "the simulated run injects a REGRESSION_TRAP"
    trapped = [o for o in state.opportunities if "REGRESSION_TRAP" in o.flags]
    assert trapped, "at least one Over/GG selection must carry REGRESSION_TRAP"
    assert all(o.selection.startswith("Over") or o.selection == "GG" for o in trapped), "only Over/GG selections are trapped"
    # the injected team is a Girone A team that plays in the next round and is mentioned in the notes
    trap_teams = {t.team for t in state.traps}
    fixture_teams_a = {t for f in state.fixtures if f.girone == Girone.A for t in (f.home_team, f.away_team)}
    injected = trap_teams & fixture_teams_a
    assert injected, "the injected Girone A team must be detected as a trap"
    joined_notes = "\n".join(state.notes)
    assert any(team in joined_notes for team in injected)
    assert "Trappola di regressione iniettata" in joined_notes
    # fit diagnostics are recorded
    assert "SLSQP" in joined_notes or "L-BFGS-B" in joined_notes
    assert "log-likelihood" in joined_notes and "prior gaussiana" in joined_notes
    assert not state.prior_enabled  # 450 matches -> pure MLE


def test_build_state_processed_with_odds(tmp_path: Path) -> None:
    league = simulate_league(seed=11, n_rounds=8)
    processed = tmp_path / "processed"
    odds_dir = tmp_path / "odds"
    save_processed(league.matches, processed / "matches.csv")
    save_fixtures(league.fixtures, processed / "fixtures.csv")
    write_odds_csv(league.odds, odds_dir / "simbook.csv")

    state = build_state(processed_dir=processed, odds_dir=odds_dir, xi=0.0)
    assert state.source == "processed"
    assert state.n_matches == len(league.matches)
    assert len(state.fixtures) == len(league.fixtures)
    assert len(state.predictions) == len(league.fixtures)
    assert len(state.quotes) == len(league.odds)
    assert state.opportunities, "odds were provided, so selections must be evaluated"
    assert len(state.opportunities) == sum(len(q.odds) for q in league.odds)
    assert any("Dati reali" in n for n in state.notes)


def test_build_state_processed_without_odds(tmp_path: Path) -> None:
    league = simulate_league(seed=11, n_rounds=8)
    processed = tmp_path / "processed"
    save_processed(league.matches, processed / "matches.csv")
    save_fixtures(league.fixtures, processed / "fixtures.csv")

    state = build_state(processed_dir=processed, odds_dir=tmp_path / "missing_odds", xi=0.0)
    assert state.source == "processed"
    assert state.quotes == []
    assert state.opportunities == []
    assert len(state.predictions) == len(league.fixtures) > 0
    assert value_table(state, 0.0).empty
    assert opportunities_table(state).empty
    assert any("Nessuna quota" in n for n in state.notes)


def test_build_state_processed_without_fixtures(tmp_path: Path) -> None:
    league = simulate_league(seed=11, n_rounds=8)
    processed = tmp_path / "processed"
    save_processed(league.matches, processed / "matches.csv")
    state = build_state(processed_dir=processed, odds_dir=tmp_path / "odds", xi=0.0)
    assert state.source == "processed"
    assert state.fixtures == [] and state.predictions == [] and state.opportunities == []
    assert predictions_table(state).empty


def test_small_sample_uses_prior(tmp_path: Path) -> None:
    mini = simulate_mini_dataset(seed=42, n_matches=20)
    processed = tmp_path / "processed"
    save_processed(mini.matches, processed / "matches.csv")
    save_fixtures(mini.fixtures, processed / "fixtures.csv")
    write_odds_csv(mini.odds, tmp_path / "odds" / "book.csv")

    state = build_state(processed_dir=processed, odds_dir=tmp_path / "odds", xi=0.0)
    assert state.source == "processed"
    assert state.n_matches == 20
    assert state.prior_enabled
    assert state.params.penalty > 0.0
    for g, gamma in state.params.gamma.items():
        assert -0.3 < gamma < 1.0, (g, gamma)
    assert state.params.converged
    assert any("prior gaussiana attiva" in n for n in state.notes)

    # explicit override: no prior even on a small sample
    mle = build_state(processed_dir=processed, odds_dir=tmp_path / "odds", xi=0.0, use_prior=False)
    assert not mle.prior_enabled and mle.params.penalty == 0.0


def test_load_odds_quotes_skips_invalid_groups(tmp_path: Path) -> None:
    odds_dir = tmp_path / "odds"
    odds_dir.mkdir()
    (odds_dir / "mixed.csv").write_text(
        "match_id,bookmaker,market,selection,odds\n"
        "M1,Book,1X2,1,2.10\nM1,Book,1X2,X,3.30\nM1,Book,1X2,2,3.60\n"
        "M1,Book,OU2.5,Over 2.5,1.95\n"  # incomplete group -> skipped
        "M2,Book,GG/NG,GG,1.80\nM2,Book,GG/NG,NG,abc\n"  # non-numeric odds -> skipped
        "M3,Book,ZZZ,1,2.00\n",  # unknown market -> skipped
        encoding="utf-8",
    )
    (odds_dir / "bad_columns.csv").write_text("foo,bar\n1,2\n", encoding="utf-8")
    quotes = load_odds_quotes(odds_dir)
    assert len(quotes) == 1
    assert quotes[0].match_id == "M1" and quotes[0].market is Market.MATCH_ODDS
    assert load_odds_quotes(tmp_path / "nope") == []


# ----------------------------------------------------------------------------- views
def test_ranking_table_labels_and_counts(sim_state: PipelineState) -> None:
    full = ranking_table(sim_state, None)
    assert list(full.columns) == list(RANKING_COLUMNS)
    # only the 80 current teams (Serie C A/B/C + Serie B) are ranked; the historical opponents of the synthetic past
    # seasons are fitted but hidden
    assert len(full) == len(sim_state.params.current_teams) == 80 < len(sim_state.params.teams)
    assert not set(full["Squadra"]) & set(sim_state.params.historical_teams)
    current_by_girone = sim_state.params.current_teams_by_girone()
    for g in Girone:
        table = ranking_table(sim_state, g)
        assert list(table.columns) == list(RANKING_COLUMNS)
        assert len(table) == len(current_by_girone[g]) == 20
        assert set(table["Squadra"]) == set(sim_state.registry.teams(g)), "Competizione = competizione del dizionario 2026/27"
        assert set(table["Competizione"]) == {g.short}  # the short label, not the code
        alphas = table["Forza Offensiva (α)"].to_numpy()
        assert np.all(np.diff(alphas) <= 1e-9), "sorted by alpha descending"
        # rounded to 3 decimals
        for col in RANKING_COLUMNS[2:]:
            assert np.allclose(table[col], table[col].round(3))
    assert ranking_table(sim_state, "B")["Competizione"].eq("Girone B").all()
    assert ranking_table(sim_state, "Serie B")["Competizione"].eq("Serie B").all() and len(ranking_table(sim_state, Girone.SB)) == 20


def test_strength_figure(sim_state: PipelineState) -> None:
    fig = strength_figure(ranking_table(sim_state, Girone.C), title="Girone C")
    assert isinstance(fig, go.Figure)
    assert len(fig.data) == 2 and all(trace.type == "bar" and trace.orientation == "h" for trace in fig.data)
    assert len(fig.data[0].y) == 20
    assert fig.layout.barmode == "group"


def test_fixtures_for_sorted_and_filtered(sim_state: PipelineState) -> None:
    all_preds = fixtures_for(sim_state, None)
    assert len(all_preds) == len(sim_state.predictions)
    keys = [(p.fixture.date, p.fixture.label) for p in all_preds]
    assert keys == sorted(keys)
    only_a = fixtures_for(sim_state, Girone.A)
    assert only_a and all(p.fixture.girone == Girone.A for p in only_a)
    assert sum(len(fixtures_for(sim_state, g)) for g in Girone) == len(all_preds) and len(fixtures_for(sim_state, Girone.SB)) == 10


def test_value_table_threshold_and_order(sim_state: PipelineState) -> None:
    for min_ev in (0.0, 0.035, 0.10):
        table = value_table(sim_state, min_ev)
        assert list(table.columns) == list(VALUE_COLUMNS)
        expected = sum(1 for o in sim_state.opportunities if o.ev >= min_ev)
        assert len(table) == expected
        ev = table["EV (%)"].to_numpy()
        assert np.all(ev >= min_ev * 100.0 - 1e-6)
        assert np.all(np.diff(ev) <= 1e-9), "sorted by EV descending"
    by_girone = value_table(sim_state, 0.035, Girone.B)
    expected_b = sum(1 for o in sim_state.opportunities if o.ev >= 0.035 and o.girone == Girone.B)
    assert len(by_girone) == expected_b > 0, "the girone filter must keep every Girone B value selection"
    assert set(by_girone["Competizione"]) == {"Girone B"}
    assert set(value_table(sim_state, 0.035, Girone.SB)["Competizione"]) == {"Serie B"}
    assert len(by_girone) < len(value_table(sim_state, 0.035)), "other gironi are filtered out"
    assert str(table["Data"].dtype).startswith("datetime64")
    flagged = value_table(sim_state, -1.0)
    assert flagged["Flag"].str.contains("REGRESSION_TRAP").any()


def test_heatmap_figure_shape(sim_state: PipelineState) -> None:
    pred = sim_state.predictions[0]
    fig = heatmap_figure(pred)
    assert isinstance(fig, go.Figure)
    assert len(fig.data) == 1 and fig.data[0].type == "heatmap"
    z = np.asarray(fig.data[0].z)
    assert z.shape == pred.matrix.shape == (11, 11)
    assert np.allclose(z, pred.matrix * 100.0)
    assert [list(stop) for stop in fig.data[0].colorscale] == [list(stop) for stop in pc.get_colorscale("Blues")]
    assert fig.layout.height == 520
    assert fig.layout.template == pio.templates["plotly"], "the 'plotly' template lets Streamlit re-theme the figure"
    assert f"Gol {pred.fixture.home_team}" == fig.layout.yaxis.title.text
    assert f"Gol {pred.fixture.away_team}" == fig.layout.xaxis.title.text
    assert pred.fixture.label in fig.layout.title.text
    hs, as_, _ = pred.most_likely_score()
    assert len(fig.layout.shapes) == 1
    shape = fig.layout.shapes[0]
    assert shape.x0 == as_ - 0.5 and shape.y0 == hs - 0.5
    small = heatmap_figure(pred, max_goals_shown=3)
    assert np.asarray(small.data[0].z).shape == (4, 4)
    with pytest.raises(ValueError):
        heatmap_figure(pred, max_goals_shown=-1)


def test_heatmap_figure_shows_whole_matrix_by_default(sim_state: PipelineState) -> None:
    """Regression: the default view used to be a 7x7 corner (0..6 goals) of the 11x11 matrix.

    The spec asks for a heatmap *of the matrix* (P(X, Y) up to 10 goals), so the
    whole ``prediction.matrix`` must be drawn and the cells must add up to 100%,
    also for the highest-scoring fixture, where the dropped tail was largest.
    """
    busiest = max(sim_state.predictions, key=lambda p: p.home_rate + p.away_rate)
    for pred in (sim_state.predictions[0], busiest):
        fig = heatmap_figure(pred)
        z = np.asarray(fig.data[0].z, dtype=float)
        assert z.shape == pred.matrix.shape
        assert np.allclose(z, pred.matrix * 100.0)
        assert abs(float(z.sum()) - 100.0) < 1e-6, "the complete distribution is displayed"
        n = pred.matrix.shape[0]
        assert list(fig.data[0].x) == list(range(n)) and list(fig.data[0].y) == list(range(n))
        assert list(fig.layout.xaxis.ticktext) == [str(g) for g in range(n)]
        assert list(fig.layout.yaxis.ticktext) == [str(g) for g in range(n)]
        assert "+" not in "".join(fig.layout.xaxis.ticktext) and "o più gol" not in fig.layout.title.text
        # the corner the old default showed is the same data, just no longer the whole story
        assert np.allclose(z[:7, :7], pred.matrix[:7, :7] * 100.0)
    # the old 7x7 corner really was incomplete for the busiest fixture
    assert busiest.matrix[:7, :7].sum() < 1.0 - 1e-4
    # explicit values at or above the matrix size are the same full view
    for shown in (busiest.matrix.shape[0] - 1, busiest.matrix.shape[0] + 5):
        full = heatmap_figure(busiest, max_goals_shown=shown)
        assert np.allclose(np.asarray(full.data[0].z, dtype=float), busiest.matrix * 100.0)
        assert "+" not in "".join(full.layout.xaxis.ticktext)


def test_heatmap_figure_compact_view_folds_tail_into_last_cell(sim_state: PipelineState) -> None:
    """A compact view aggregates the probability beyond N into an ``N+`` row/column (cells still sum to 100%)."""
    pred = max(sim_state.predictions, key=lambda p: p.home_rate + p.away_rate)
    m = pred.matrix
    fig = heatmap_figure(pred, max_goals_shown=6)
    z = np.asarray(fig.data[0].z, dtype=float)
    assert z.shape == (7, 7)
    assert abs(float(z.sum()) - 100.0) < 1e-6
    assert np.allclose(z[:6, :6], m[:6, :6] * 100.0), "exact scores are untouched"
    assert np.allclose(z[6, :6], m[6:, :6].sum(axis=0) * 100.0), "last row = P(home >= 6, away = j)"
    assert np.allclose(z[:6, 6], m[:6, 6:].sum(axis=1) * 100.0), "last column = P(home = i, away >= 6)"
    assert np.isclose(z[6, 6], m[6:, 6:].sum() * 100.0), "corner = P(home >= 6, away >= 6)"
    assert z[6, 6] > m[6, 6] * 100.0, "the fold adds the dropped tail mass"
    labels = ["0", "1", "2", "3", "4", "5", "6+"]
    assert list(fig.layout.xaxis.ticktext) == labels and list(fig.layout.yaxis.ticktext) == labels
    assert list(fig.layout.xaxis.tickvals) == list(range(7))
    assert "6+ = 6 o più gol" in fig.layout.title.text
    hover = np.asarray(fig.data[0].hovertext)
    assert hover.shape == (7, 7)
    assert hover[6, 0].startswith(f"{pred.fixture.home_team} 6+ - 0 {pred.fixture.away_team}")
    assert hover[0, 6].startswith(f"{pred.fixture.home_team} 0 - 6+ {pred.fixture.away_team}")
    assert fig.data[0].hovertemplate == "%{hovertext}<extra></extra>"
    text = np.asarray(fig.data[0].text)
    assert text[6, 6] == f"{z[6, 6]:.1f}%"
    hs, as_, _ = pred.most_likely_score()
    assert len(fig.layout.shapes) == (1 if hs < 6 and as_ < 6 else 0), "no outline on an aggregated cell"
    # degenerate compact view: one '0+' cell holding everything, no misleading outline
    single = heatmap_figure(pred, max_goals_shown=0)
    z1 = np.asarray(single.data[0].z, dtype=float)
    assert z1.shape == (1, 1) and np.isclose(z1[0, 0], 100.0)
    assert list(single.layout.xaxis.ticktext) == ["0+"] and len(single.layout.shapes) == 0


def test_markets_table_rows(sim_state: PipelineState) -> None:
    pred = sim_state.predictions[0]
    quotes = quotes_for(sim_state, pred.match_id)
    assert quotes
    table = markets_table(pred, quotes)
    assert list(table.columns) == list(MARKETS_COLUMNS)
    n_selections = sum(len(sels) for sels in SELECTIONS.values())
    assert len(table) == n_selections == 11
    assert table["Quota Book"].notna().all() and table["EV (%)"].notna().all()
    # EV = p * odds - 1 (in %), computed from the bookmaker's quote (best price when several books), not from the table
    first = table.iloc[0]
    p = pred.probabilities[Market.MATCH_ODDS]["1"]
    quoted = [q.odds["1"] for q in quotes if q.market is Market.MATCH_ODDS]
    assert quoted, "the synthetic bookmaker quotes the 1X2 market of every fixture"
    book = max(quoted)
    assert first["Esito"] == "1"
    assert first["Quota Book"] == round(book, 2)
    assert abs(first["EV (%)"] - round((p * book - 1.0) * 100.0, 2)) < 1e-9
    assert first["Quota Fair"] == round(pred.fair_odds[Market.MATCH_ODDS]["1"], 2)
    # without quotes the book columns are NaN but the model columns are filled
    bare = markets_table(pred, [])
    assert len(bare) == n_selections
    assert bare["Quota Book"].isna().all() and bare["EV (%)"].isna().all()
    assert bare["Probabilità (%)"].between(0.0, 100.0).all()


def test_traps_table(sim_state: PipelineState) -> None:
    table = traps_table(sim_state)
    assert list(table.columns) == list(TRAPS_COLUMNS)
    assert len(table) == len(sim_state.traps) > 0
    assert np.all(np.diff(table["Severità"].to_numpy()) <= 1e-9)
    assert str(table["Ultima Gara"].dtype).startswith("datetime64")


def test_predictions_and_opportunities_tables(sim_state: PipelineState) -> None:
    preds = predictions_table(sim_state)
    assert len(preds) == len(sim_state.predictions)
    assert {"match_id", "lambda", "mu", "rho", "p_1", "fair_1", "p_over_2.5", "fair_under_2.5", "p_gg", "fair_ng"} <= set(preds.columns)
    row = preds.iloc[0]
    assert abs(row["p_1"] + row["p_x"] + row["p_2"] - 1.0) < 1e-3
    opps = opportunities_table(sim_state)
    assert len(opps) == len(sim_state.opportunities)
    assert {"match_id", "market", "selection", "book_odds", "fair_odds", "ev", "stake_units", "flags"} <= set(opps.columns)


def test_export_csv_bytes_roundtrip(sim_state: PipelineState) -> None:
    table = value_table(sim_state, 0.035)
    payload = export_csv_bytes(table)
    assert isinstance(payload, bytes)
    text = payload.decode("utf-8")
    assert text.splitlines()[0] == ",".join(VALUE_COLUMNS)
    parsed = pd.read_csv(io.BytesIO(payload))
    assert parsed.shape == table.shape
    assert list(parsed.columns) == list(VALUE_COLUMNS)
    assert np.allclose(parsed["EV (%)"].to_numpy(), table["EV (%)"].to_numpy())
    assert parsed["Data"].iloc[0] == table["Data"].iloc[0].strftime("%Y-%m-%d")


# ----------------------------------------------------------------------------- Streamlit UI
def test_app_renders_with_apptest(monkeypatch: pytest.MonkeyPatch) -> None:
    from streamlit.testing.v1 import AppTest

    monkeypatch.setenv("SERIEC_FORCE_SIMULATE", "1")
    monkeypatch.setenv("SERIEC_SIM_ROUNDS", "8")
    at = AppTest.from_file(str(APP_PATH), default_timeout=240)
    at.run()
    assert not at.exception, [e.value for e in at.exception]
    assert [tab.label for tab in at.tabs] == ["Classifica Forza", "Match & Heatmap", "Opportunità di Valore", "Export"]
    assert len(at.get("download_button")) >= 1
    assert any("Dati simulati" in i.value for i in at.sidebar.info)
    # the badge and the subtitle state the perimeter: season, teams, current-season vs historical matches
    perimeter = "Perimetro: Serie C 2026/27 Gironi A/B/C + Serie B (80 squadre) · 320 partite stagione corrente · 3040 partite storiche (2024/25-2025/26)"
    assert any(perimeter in i.value for i in at.sidebar.info)
    assert any(perimeter in c.value for c in at.caption)
    assert any("squadre storiche escluse" in i.value for i in at.sidebar.info)
    assert at.metric, "the match tab shows the metrics row"
    assert any("EV = P_modello" in c.value for c in at.caption)

    assert at.selectbox(key="girone").options == ["Tutti", "Girone A", "Girone B", "Girone C", "Serie B"]  # from config/competitions.toml
    at.selectbox(key="girone").select("Girone A").run()
    assert not at.exception, [e.value for e in at.exception]
    assert len(at.selectbox(key="fixture").options) == 10  # Girone A: 20 teams -> 10 fixtures
    assert at.dataframe, "tables are rendered"
    at.selectbox(key="girone").select("Serie B").run()
    assert not at.exception, [e.value for e in at.exception]
    assert len(at.selectbox(key="fixture").options) == 10 and all(o.startswith("[SB]") for o in at.selectbox(key="fixture").options)
    assert any("vista: Serie B" in c.value for c in at.caption)


def free_port() -> int:
    """A TCP port that is free right now (so a busy 8599 cannot fail the server test)."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def test_streamlit_server_healthz() -> None:
    """Launch the real server once (headless) on a free port and check /healthz, then stop it."""
    port = free_port()
    env = {**os.environ, "SERIEC_FORCE_SIMULATE": "1", "SERIEC_SIM_ROUNDS": "6"}
    proc = subprocess.Popen(
        [sys.executable, "-m", "streamlit", "run", str(APP_PATH), "--server.headless", "true", "--server.port", str(port), "--browser.gatherUsageStats", "false"],
        cwd=str(REPO_ROOT),
        env=env,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    try:
        health = b""
        deadline = time.monotonic() + 60.0
        while time.monotonic() < deadline:
            try:
                with urllib.request.urlopen(f"http://localhost:{port}/_stcore/health", timeout=2) as resp:
                    health = resp.read()
                    break
            except (urllib.error.URLError, ConnectionError, OSError):
                time.sleep(0.5)
        assert health.strip().lower() == b"ok", health
        # the legacy /healthz path is served by the SPA in Streamlit >= 1.3x: it must answer 200 with the app shell
        with urllib.request.urlopen(f"http://localhost:{port}/healthz", timeout=5) as resp:
            assert resp.status == 200
            shell = resp.read()
        assert shell.strip().lower() == b"ok" or b'id="root"' in shell
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=15)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait(timeout=15)


# ----------------------------------------------------------------------------- regression tests (review findings)
def test_build_state_rejects_negative_seed_and_zero_rounds() -> None:
    """A negative seed or a non-positive number of rounds fails fast with an actionable message."""
    with pytest.raises(ValueError, match=r"seed must be >= 0 \(got -1\)"):
        build_state(simulate=True, seed=-1, xi=0.0)
    with pytest.raises(ValueError, match=r"n_rounds must be >= 1 \(got 0\)"):
        build_state(simulate=True, n_rounds=0, xi=0.0)
    with pytest.raises(ValueError, match="n_rounds must be >= 1"):
        build_state(simulate=True, n_rounds=-5, xi=0.0)


def test_build_state_clamps_full_season_rounds() -> None:
    """Asking for the whole season (or more) still leaves one round of fixtures, with a note."""
    upper = max_sim_rounds()
    assert upper == 37  # 20-team gironi -> 38 double round-robin rounds; the last one stays as fixtures
    assert MIN_SIM_ROUNDS == 1
    state = build_state(simulate=True, seed=3, xi=0.0, n_rounds=50)
    assert state.source == "simulated"
    assert state.n_current_matches == upper * 40  # 4 competitions (3 Serie C gironi + Serie B) x 10 matches per round
    assert state.n_matches == state.n_current_matches + state.n_history_matches
    assert len(state.fixtures) == 40 and len(state.predictions) == 40
    assert state.quotes and state.opportunities, "the synthetic bookmaker still quotes the last round"
    assert state.traps and any(o.is_trap for o in state.opportunities), "the trap is still injected"
    joined = "\n".join(state.notes)
    assert "richieste: 50" in joined and f"limitato a {upper}" in joined
    # a value inside the range is used verbatim and leaves no clamp note
    plain = build_state(simulate=True, seed=3, xi=0.0, n_rounds=upper)
    assert plain.n_matches == state.n_matches
    assert not any("limitato a" in n for n in plain.notes)
    # the lower bound is valid too: one played round, prior on (history does not count as coverage), next round predicted
    tiny = build_state(simulate=True, seed=3, xi=0.0, n_rounds=MIN_SIM_ROUNDS)
    assert tiny.n_current_matches == 40 and tiny.n_history_matches > 0 and len(tiny.predictions) == 40 and tiny.prior_enabled


def test_no_odds_message_is_source_aware(sim_state: PipelineState) -> None:
    """The value panel hint must not tell a simulated run to add odds CSVs."""
    empty_sim = dataclasses.replace(sim_state, quotes=[], fixtures=[], predictions=[], opportunities=[])
    msg = no_odds_message(empty_sim)
    assert "sintetico" in msg and "partite in programma" in msg
    assert "data/odds" not in msg and "CSV" not in msg
    with_fixtures = dataclasses.replace(sim_state, quotes=[], opportunities=[])
    msg_fx = no_odds_message(with_fixtures)
    assert "sintetico" in msg_fx and "CSV" not in msg_fx and msg_fx != msg
    processed = dataclasses.replace(sim_state, source="processed", quotes=[], opportunities=[])
    msg_proc = no_odds_message(processed)
    assert "data/odds" in msg_proc and "match_id, bookmaker, market, selection, odds" in msg_proc


def test_container_width_kwargs_matches_streamlit_api() -> None:
    """The full-width adapter picks the keyword the installed Streamlit actually accepts."""
    import streamlit as st

    def legacy(data=None, width=None, height=None, *, use_container_width: bool = False) -> None:  # noqa: ARG001
        """Streamlit <= 1.4x signature: ``width`` is a pixel integer."""

    def deprecated_tristate(label, *, width="content", use_container_width=None) -> None:  # noqa: ARG001
        """Recent signature: ``use_container_width`` deprecated, forwards to ``width``."""

    def future(label, *, width="content") -> None:  # noqa: ARG001
        """Signature once ``use_container_width`` is removed."""

    assert container_width_kwargs(legacy) == LEGACY_FULL_WIDTH == {"use_container_width": True}
    assert container_width_kwargs(deprecated_tristate) == MODERN_FULL_WIDTH == {"width": "stretch"}
    assert container_width_kwargs(future) == MODERN_FULL_WIDTH
    assert container_width_kwargs(len) in (LEGACY_FULL_WIDTH, MODERN_FULL_WIDTH)  # builtins never raise
    for name in ("button", "download_button", "dataframe", "plotly_chart"):
        command = getattr(st, name)
        kwargs = container_width_kwargs(command)
        (key,) = kwargs
        assert key in inspect.signature(command).parameters, (name, kwargs)
    fresh = container_width_kwargs(legacy)
    fresh["extra"] = 1
    assert "extra" not in LEGACY_FULL_WIDTH, "the adapter returns a copy"


def test_app_survives_out_of_range_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Negative seed and full-season rounds render a complete page instead of a stack trace."""
    from streamlit.testing.v1 import AppTest

    monkeypatch.setenv("SERIEC_FORCE_SIMULATE", "1")
    monkeypatch.setenv("SERIEC_SEED", "-1")  # negative seed -> default seed, no crash
    monkeypatch.setenv("SERIEC_SIM_ROUNDS", "99")  # full season -> clamped, fixtures still present
    monkeypatch.setenv("SERIEC_AS_OF", "2026-11-06")  # deterministic calendar position
    at = AppTest.from_file(str(APP_PATH), default_timeout=240)
    at.run()
    assert not at.exception, [e.value for e in at.exception]
    assert any("Dati simulati" in i.value for i in at.sidebar.info)
    assert not any("0 partite in programma" in i.value for i in at.sidebar.info)
    assert at.metric, "fixtures exist, so the match tab shows its metrics"
    assert len(at.selectbox(key="fixture").options) == 40
    assert any("limitato a 37" in m.value for m in at.markdown), "the clamp is reported in 'Dettagli pipeline'"
    assert not any("aggiungi file CSV" in i.value for i in at.info)

    monkeypatch.setenv("SERIEC_SIM_ROUNDS", "0")  # below the minimum -> ignored, calendar as of SERIEC_AS_OF
    monkeypatch.setenv("SERIEC_SEED", "abc")  # not an integer -> default seed
    monkeypatch.setenv("SERIEC_HISTORY", "7")  # out of range -> default (2 synthetic seasons)
    at.run()
    assert not at.exception, [e.value for e in at.exception]
    assert len(at.selectbox(key="fixture").options) == 40
    assert any("seed=7, 11 giornate giocate" in m.value for m in at.markdown)  # 2026-08-23 .. 2026-11-01 played
    assert any("Perimetro: Serie C 2026/27 Gironi A/B/C + Serie B (80 squadre) · 440 partite stagione corrente · 3040 partite storiche (2024/25-2025/26)" in i.value for i in at.sidebar.info)

    monkeypatch.setenv("SERIEC_AS_OF", "not-a-date")  # unparseable -> today, page still renders
    monkeypatch.setenv("SERIEC_SEASON", "1999_00")  # unknown season file -> default 2026/27
    monkeypatch.setenv("SERIEC_HISTORY", "0")
    at.run()
    assert not at.exception, [e.value for e in at.exception]
    assert any("Perimetro: Serie C 2026/27 Gironi A/B/C + Serie B (80 squadre)" in i.value and "0 partite storiche" in i.value for i in at.sidebar.info)


def test_app_heatmap_renders_complete_matrix(monkeypatch: pytest.MonkeyPatch) -> None:
    """Regression: the rendered heatmap used to be a 7x7 corner summing to < 100%.

    Decodes the Plotly spec actually sent to the front end: by default the full
    11x11 matrix is drawn (cells sum to 100%); the 'Gol massimi nella matrice'
    slider gives a compact view whose last row/column aggregate the tail, so the
    cells still sum to 100%.
    """
    from streamlit.testing.v1 import AppTest

    monkeypatch.setenv("SERIEC_FORCE_SIMULATE", "1")
    monkeypatch.setenv("SERIEC_SIM_ROUNDS", "8")
    at = AppTest.from_file(str(APP_PATH), default_timeout=240)
    at.run()
    assert not at.exception, [e.value for e in at.exception]
    slider = at.slider(key="heatmap_max_goals")
    assert slider.value == 10 and slider.min == 3 and slider.max == 10
    z, layout = rendered_heatmap(at)
    assert z.shape == (11, 11), "the whole score matrix (0..10 goals) is drawn"
    assert abs(float(z.sum()) - 100.0) < 1e-6
    assert list(layout["xaxis"]["tickvals"]) == list(range(11))
    assert list(layout["xaxis"]["ticktext"]) == [str(g) for g in range(11)]
    assert any("Matrice completa 0-10 gol" in c.value for c in at.caption)

    slider.set_value(6).run()
    assert not at.exception, [e.value for e in at.exception]
    z6, layout6 = rendered_heatmap(at)
    assert z6.shape == (7, 7)
    assert abs(float(z6.sum()) - 100.0) < 1e-6, "the compact view folds the tail instead of dropping it"
    assert np.allclose(z6[:6, :6], z[:6, :6])
    assert np.isclose(z6[6, 6], z[6:, 6:].sum()) and np.allclose(z6[6, :6], z[6:, :6].sum(axis=0))
    assert list(layout6["xaxis"]["ticktext"]) == ["0", "1", "2", "3", "4", "5", "6+"]
    assert any("(6+)" in c.value and "100%" in c.value for c in at.caption)


def test_app_non_default_season_renders_on_its_own_calendar(monkeypatch: pytest.MonkeyPatch) -> None:
    """Regression: ``SERIEC_SEASON=2025_26`` (a documented value) crashed the page with 'HISTORICAL_TEAMS pool
    exhausted' and, had it not, would have simulated the 2025/26 perimeter on the 2026/27 calendar."""
    from streamlit.testing.v1 import AppTest

    monkeypatch.setenv("SERIEC_FORCE_SIMULATE", "1")
    monkeypatch.setenv("SERIEC_SEASON", "2025_26")
    monkeypatch.setenv("SERIEC_AS_OF", "2025-11-06")
    at = AppTest.from_file(str(APP_PATH), default_timeout=240)
    at.run()
    assert not at.exception, [e.value for e in at.exception]
    assert not at.error, [e.value for e in at.error]
    perimeter = "Perimetro: Serie C 2025/26 Gironi A/B/C + Serie B (80 squadre) · 440 partite stagione corrente · 3040 partite storiche (2023/24-2024/25)"
    assert any(perimeter in i.value for i in at.sidebar.info)
    options = at.selectbox(key="fixture").options
    assert len(options) == 40 and all("2025-11-09" in o for o in options)
    assert any("Calendario simulato al 2025-11-06" in m.value for m in at.markdown)
    assert any("Storica 01" in m.value or "Caldiero Terme" in m.value for m in at.markdown), "the padded history is reported"


def test_app_resolves_today_on_every_run_and_survives_a_pipeline_error(monkeypatch: pytest.MonkeyPatch) -> None:
    """Regression: without ``SERIEC_AS_OF`` the cache key was ``as_of=None`` and ``date.today()`` was evaluated once
    inside the cached state, so the calendar position never advanced while the server ran. The app now resolves
    today itself and passes it down; a ``ValueError`` from the data layer shows an error and falls back to the
    default season instead of crashing the page."""
    import src.dashboard.state as state_module
    from streamlit.testing.v1 import AppTest

    original = state_module.build_state
    received: list[date | None] = []

    def recording_build_state(**kwargs):
        received.append(kwargs.get("as_of"))
        if kwargs.get("n_rounds") == 5:
            raise ValueError("configurazione rifiutata (test)")
        return original(**kwargs)

    monkeypatch.setattr(state_module, "build_state", recording_build_state)
    monkeypatch.setenv("SERIEC_FORCE_SIMULATE", "1")
    monkeypatch.setenv("SERIEC_SEED", "11")  # a seed no other test uses: the cache cannot answer for us
    monkeypatch.delenv("SERIEC_AS_OF", raising=False)
    monkeypatch.delenv("SERIEC_SIM_ROUNDS", raising=False)
    at = AppTest.from_file(str(APP_PATH), default_timeout=240)
    at.run()
    assert not at.exception, [e.value for e in at.exception]
    assert received and received[-1] == date.today(), "today is resolved by the app and is part of the cache key"
    assert any(f"Calendario simulato al {date.today().isoformat()}" in m.value for m in at.markdown)

    received.clear()
    monkeypatch.setenv("SERIEC_SIM_ROUNDS", "5")  # the wrapper refuses this combination
    at.run()
    assert not at.exception, [e.value for e in at.exception]
    assert any("Pipeline non eseguibile con i parametri richiesti" in e.value and "stagione 2026/27" in e.value for e in at.error)
    assert received and received[0] == date.today()  # the refused call (the fallback may be served by the cache)
    assert any("Perimetro: Serie C 2026/27 Gironi A/B/C + Serie B (80 squadre)" in i.value for i in at.sidebar.info)
    assert at.metric, "the fallback state renders the match tab"
