"""Tests for the Streamlit dashboard: data layer (``src.dashboard.state``) and UI (``src.dashboard.app``)."""
from __future__ import annotations

import base64
import dataclasses
import functools
import inspect
import io
import json
import os
import re
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
from datetime import date
from pathlib import Path
from types import SimpleNamespace
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
import plotly.colors as pc
import plotly.graph_objects as go
import plotly.io as pio
import pytest

from src.common.schemas import SELECTIONS, Fixture, Girone, Market, OddsQuote, season_of
from src.dashboard.legend import POOR_EXAMPLE_FLAGS, SYNTHETIC_REASON_NO_QUOTES, SYNTHETIC_REASON_NO_VALUE, pick_example_row
from src.dashboard import LEGACY_FULL_WIDTH, MODERN_FULL_WIDTH, container_width_kwargs
from src.dashboard.legend import legend_terms
from src.dashboard.state import (
    ALL_MATCHDAYS,
    DEFAULT_SECTION,
    MARKETS_COLUMNS,
    MATCHDAY_COLUMNS,
    MATCHDAY_TABLE_KEY_PREFIX,
    MIN_SIM_ROUNDS,
    PICKS_EXPORT_COLUMNS,
    RANKING_COLUMNS,
    RESULTS_MODEL_COLUMNS,
    SECTION_BC,
    SECTION_D,
    SECTION_FAMILIES,
    SERIE_D_STATUS_NOT_CONFIGURED,
    SERIE_D_STATUS_NOT_INITIALISED,
    TRAPS_COLUMNS,
    VALUE_COLUMNS,
    PipelineState,
    all_sections,
    btts_picks_table,
    build_state,
    competition_options,
    configured_gironi,
    coverage_text,
    current_teams,
    draw_picks_table,
    export_csv_bytes,
    family_gironi,
    family_label,
    family_predictions,
    find_matchday,
    fixture_label,
    fixtures_for,
    heatmap_figure,
    in_section,
    load_odds_quotes,
    markets_table,
    matchday_csv_bytes,
    matchday_predictions,
    matchday_table_for,
    matchday_table_key,
    matchdays,
    max_sim_rounds,
    no_odds_message,
    opportunities_table,
    perimeter_text,
    picks_csv_bytes,
    picks_export_frame,
    predictions_table,
    quotes_for,
    ranking_table,
    resolve_matchday,
    results_csv_bytes,
    results_table_for,
    results_team_options,
    seasons_available,
    section_families,
    section_gironi,
    section_key,
    section_matches,
    section_of_family,
    section_opportunities,
    section_options,
    section_perimeter_label,
    section_predictions,
    section_ready,
    section_slug,
    section_team_count,
    section_traps,
    sections,
    sections_for,
    selected_row_match_id,
    serie_d_gironi,
    serie_d_harvest_command,
    serie_d_status_text,
    strength_figure,
    traps_table,
    value_opportunities,
    value_table,
)
from src.common.teams import TeamRegistry, load_registry
from src.engine.picks import PICKS_COLUMNS
from src.engine.tables import team_form
from src.ingestion.harvester import save_fixtures, save_processed
from src.simulation.generator import simulate_league, simulate_mini_dataset, simulate_serie_d_registry

if TYPE_CHECKING:
    from streamlit.testing.v1 import AppTest

REPO_ROOT = Path(__file__).resolve().parents[1]
APP_PATH = REPO_ROOT / "src" / "dashboard" / "app.py"
#: The competitions of the shipped 2026/27 dictionary (Serie C A/B/C and Serie B): the Serie D gironi of
#: ``config/competitions.toml`` ship without teams, so the simulated perimeter never contains them.
PERIMETER_GIRONI = tuple(load_registry().gironi)
#: The configured Serie D competitions (empty when the configuration has no 'Serie D' family).
SERIE_D = serie_d_gironi()
needs_serie_d = pytest.mark.skipif(not SERIE_D, reason="no competition of family 'Serie D' in config/competitions.toml")


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


def matchday_table_element(at: AppTest):  # noqa: ANN201 - streamlit.testing element type is private
    """The *Giornata* table element: the one ``st.dataframe`` whose key starts with :data:`MATCHDAY_TABLE_KEY_PREFIX` (the key is scoped per matchday and competition)."""
    tables = [d for d in at.dataframe if d.key is not None and str(d.key).startswith(MATCHDAY_TABLE_KEY_PREFIX)]
    assert len(tables) == 1, [d.key for d in at.dataframe]
    return tables[0]


def matchday_table_shown(at: AppTest) -> pd.DataFrame:
    """The *Giornata* table as rendered."""
    return matchday_table_element(at).value


def results_table_shown(at: AppTest) -> pd.DataFrame:
    """The *Risultati* table as rendered (found by its columns: a plain ``st.dataframe`` is not a widget, so AppTest has no key for it)."""
    tables = [d.value for d in at.dataframe if list(d.value.columns) == list(RESULTS_MODEL_COLUMNS)]
    assert len(tables) == 1, [list(d.value.columns)[:3] for d in at.dataframe]
    return tables[0]


EXAMPLE_HEADER = re.compile(r"^\*\*(?P<partita>.+?)\*\* \((?P<comp>[^)]+)\) · mercato (?P<mercato>.+?), esito \*\*(?P<esito>[^*]+)\*\* · quota (?P<quota>[0-9.]+) di (?P<book>.+)$")


def worked_example_shown(at: AppTest) -> tuple[dict[str, str], str]:
    """``(header fields, whole markdown)`` of the Legenda tab's worked example (the markdown that holds 'Kelly pieno')."""
    text = next(m.value for m in at.markdown if "Kelly pieno" in m.value and "**Stake**" in m.value)
    match = EXAMPLE_HEADER.match(text.splitlines()[0])
    assert match is not None, text.splitlines()[0]
    return match.groupdict(), text


def assert_example_is_the_panel_row(at: AppTest) -> None:
    """The worked example is the first row of the value table shown without LONGSHOT / SUSPICIOUS_EDGE (the first row when all are flagged)."""
    panels = [d.value for d in at.dataframe if list(d.value.columns) == list(VALUE_COLUMNS)]
    fields, text = worked_example_shown(at)
    if not panels or panels[0].empty:
        assert "Esempio costruito" in text and SYNTHETIC_REASON_NO_VALUE in text and SYNTHETIC_REASON_NO_QUOTES not in text
        assert any(c.value.startswith("Nessuna riga nel pannello Opportunità di Valore") for c in at.caption)
        return
    panel = panels[0]
    ordinary = panel[~panel["Flag"].str.contains("LONGSHOT|SUSPICIOUS_EDGE", regex=True, na=False)]
    row = (ordinary if not ordinary.empty else panel).iloc[0]
    assert "Esempio costruito" not in text
    assert fields["partita"] == row["Partita"] and fields["esito"] == row["Esito"] and float(fields["quota"]) == pytest.approx(row["Quota Book"], abs=0.005)
    assert fields["comp"] == row["Competizione"] and fields["book"] == row["Bookmaker"]
    assert any(c.value.startswith("Esempio preso dal pannello Opportunità di Valore con i filtri correnti (") for c in at.caption)


@pytest.fixture(scope="module")
def sim_state() -> PipelineState:
    """One simulated pipeline run shared by the view tests (seed 7, 15 rounds, stationary)."""
    return build_state(simulate=True, seed=7, xi=0.0, n_rounds=15)


def multi_round_processed_dir(tmp_path: Path, *, seed: int = 11, played: int = 8, sb_rounds: int = 2) -> tuple[Path, Path, dict[int, list[Fixture]]]:
    """A processed dir whose fixtures span three future rounds per Serie C girone (``sb_rounds`` for Serie B), with odds for all.

    ``simulate_league(seed, n_rounds=k)`` always draws the same schedule, so the
    fixture round of the ``played``, ``played + 1`` and ``played + 2`` leagues are
    rounds ``played + 1 .. played + 3`` of one calendar; the matches come from the
    first league only. Returns ``(processed_dir, odds_dir, fixtures_by_round)``.
    """
    base = simulate_league(seed=seed, n_rounds=played)
    processed, odds_dir = tmp_path / "processed", tmp_path / "odds"
    save_processed(base.matches, processed / "matches.csv")
    fixtures: list[Fixture] = []
    quotes: list[OddsQuote] = []
    by_round: dict[int, list[Fixture]] = {}
    for k in range(3):
        league = simulate_league(seed=seed, n_rounds=played + k)
        keep = [f for f in league.fixtures if f.girone != Girone.SB or k < sb_rounds]
        by_round[played + 1 + k] = keep
        fixtures += keep
        kept_ids = {f.match_id for f in keep}
        quotes += [q for q in league.odds if q.match_id in kept_ids]
    save_fixtures(fixtures, processed / "fixtures.csv")
    write_odds_csv(quotes, odds_dir / "simbook.csv")
    return processed, odds_dir, by_round


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
    # no fixtures: no matchday, and every matchday-aware view is empty instead of raising
    assert state.matchdays == [] and state.next_matchday is None and resolve_matchday(state, None) is None
    assert fixtures_for(state, None) == [] and matchday_predictions(state) == [] and value_table(state, -1.0).empty
    assert draw_picks_table(state, None).empty and btts_picks_table(state, None).empty
    assert list(draw_picks_table(state, None).columns) == list(PICKS_COLUMNS)


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
    for g in PERIMETER_GIRONI:
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
    assert len(all_preds) == len(sim_state.predictions)  # the simulated league has one round: the next matchday is everything
    keys = [(p.fixture.date, p.fixture.label) for p in all_preds]
    assert keys == sorted(keys)
    only_a = fixtures_for(sim_state, Girone.A)
    assert only_a and all(p.fixture.girone == Girone.A for p in only_a)
    assert sum(len(fixtures_for(sim_state, g)) for g in Girone) == len(all_preds) and len(fixtures_for(sim_state, Girone.SB)) == 10
    # one matchday with every competition at round 16 (15 played); the explicit forms agree with the default
    assert len(sim_state.matchdays) == 1 and sim_state.next_matchday is not None
    assert sim_state.next_matchday.rounds == {g: 16 for g in PERIMETER_GIRONI} and sim_state.next_matchday.n_fixtures == 40
    assert fixtures_for(sim_state, None, sim_state.next_matchday) == all_preds == fixtures_for(sim_state, None, ALL_MATCHDAYS)
    assert any("raggruppate in 1 giornata" in n for n in sim_state.notes)


def test_matchday_defaults_with_a_season_long_fixture_list(tmp_path: Path) -> None:
    """With FootyStats the fixture list is the whole remaining season: the selector, the value panel and the picks
    must default to the next matchday (the sum of the next rounds), while the exports keep every fixture."""
    processed, odds_dir, by_round = multi_round_processed_dir(tmp_path)
    state = build_state(processed_dir=processed, odds_dir=odds_dir, xi=0.0)
    assert state.source == "processed" and len(state.fixtures) == 110 == len(state.predictions)  # 40 + 40 + 30 (Serie B has two rounds left)
    assert [m.n_fixtures for m in state.matchdays] == [40, 40, 30]
    first, second, third = state.matchdays
    assert first.rounds == {g: 9 for g in PERIMETER_GIRONI} and second.rounds == {g: 10 for g in PERIMETER_GIRONI} and third.rounds == {g: 11 for g in (Girone.A, Girone.B, Girone.C)}
    assert first.fixture_ids == {f.match_id for f in by_round[9]} and third.fixture_ids == {f.match_id for f in by_round[11]}
    assert first.key == first.start.isoformat() and "g.9" in first.label and "Serie B g.9" in first.label and "Serie B" not in third.label
    assert state.next_matchday is first and find_matchday(state, second.key) is second and find_matchday(state, "nope") is None
    assert resolve_matchday(state, None) is first and resolve_matchday(state, ALL_MATCHDAYS) is None and resolve_matchday(state, second.key) is second
    assert resolve_matchday(state, "stale-key") is first and resolve_matchday(state, third) is third
    # the selector: next matchday by default (40, not the whole season), one girone of it, another matchday, everything
    assert len(fixtures_for(state, None)) == 40 and {p.fixture.round for p in fixtures_for(state, None)} == {9}
    assert len(fixtures_for(state, Girone.SB)) == 10 and len(fixtures_for(state, Girone.SB, third)) == 0
    assert len(fixtures_for(state, None, second)) == 40 and len(fixtures_for(state, None, third)) == 30
    assert len(fixtures_for(state, None, ALL_MATCHDAYS)) == 110 == len(predictions_table(state))
    preds_second = matchday_predictions(state, second)
    assert len(preds_second) == 40 and all(second.contains(p) for p in preds_second)
    assert len(matchday_predictions(state, ALL_MATCHDAYS)) == 110 and len(matchday_predictions(state)) == 40
    # the value panel: next matchday by default, every matchday on request
    everything = value_table(state, -1.0, matchday=ALL_MATCHDAYS)
    assert len(everything) == len(state.opportunities) > 0
    default = value_table(state, -1.0)
    assert 0 < len(default) < len(everything)
    first_labels = {f"{f.home_team} - {f.away_team}" for f in by_round[9]}
    assert set(default["Partita"]) <= first_labels
    assert len(default) == sum(1 for o in state.opportunities if o.match_id in first.fixture_ids)
    only_third = value_table(state, -1.0, matchday=third)
    assert set(only_third["Competizione"]) <= {"Girone A", "Girone B", "Girone C"} and len(only_third) == sum(1 for o in state.opportunities if o.match_id in third.fixture_ids)
    assert len(value_table(state, -1.0, Girone.SB, matchday=third)) == 0
    assert len(opportunities_table(state)) == len(state.opportunities), "the export keeps every evaluated selection"
    # value_opportunities is the panel as objects: same rows, same order (EV desc, then match, market, selection)
    rows = value_opportunities(state, -1.0)
    assert len(rows) == len(default) and [f"{o.home_team} - {o.away_team}" for o in rows] == default["Partita"].tolist()
    assert [o.selection for o in rows] == default["Esito"].tolist() and [round(o.ev * 100.0, 2) for o in rows] == default["EV (%)"].tolist()
    assert value_opportunities(state, -1.0, Girone.SB, matchday=third) == [] and len(value_opportunities(state, -1.0, matchday=ALL_MATCHDAYS)) == len(everything)
    # the Legenda example comes from the panel's own rows (next matchday), skipping long shots and suspicious edges
    example = pick_example_row(value_opportunities(state, 0.035))
    assert example is not None and first.contains(example.match_id) and POOR_EXAMPLE_FLAGS.isdisjoint(example.flags)
    assert example.ev >= 0.035 and example is value_opportunities(state, 0.035)[[POOR_EXAMPLE_FLAGS.isdisjoint(o.flags) for o in value_opportunities(state, 0.035)].index(True)]
    best_overall = max((o for o in state.opportunities if o.value), key=lambda o: o.ev)
    assert example is not best_overall or first.contains(best_overall.match_id), "the example is never a row of another matchday"
    # the picks: sized 5 / 12, restricted to the matchday, with the synthetic bookmaker's prices
    draws = draw_picks_table(state, second)
    btts = btts_picks_table(state, second)
    assert list(draws.columns) == list(PICKS_COLUMNS) == list(btts.columns) and len(draws) == 5 and len(btts) == 12
    second_labels = {f"{f.home_team} - {f.away_team}" for f in by_round[10]}
    assert set(draws["Partita"]) <= second_labels and set(btts["Partita"]) <= second_labels
    assert draws["Esito"].eq("X").all() and btts["Esito"].eq("GG").all() and draws["Quota Book"].notna().all()
    assert len(draw_picks_table(state, None)) == 5 and set(draw_picks_table(state, None)["Partita"]) <= first_labels
    assert len(btts_picks_table(state, third, n=12, girone=Girone.A)) == 10 and set(btts_picks_table(state, third, n=4)["Competizione"]) <= {"Girone A", "Girone B", "Girone C"}
    assert len(draw_picks_table(state, third, n=3, girone=Girone.SB)) == 0
    # the combined export of both lists
    export = picks_export_frame(draws, btts, second)
    assert list(export.columns) == list(PICKS_EXPORT_COLUMNS) and len(export) == 17
    assert export["Lista"].tolist() == ["X"] * 5 + ["GG"] * 12 and export["Giornata"].eq(second.label).all()
    payload = picks_csv_bytes(draws, btts, second)
    parsed = pd.read_csv(io.BytesIO(payload))
    assert list(parsed.columns) == list(PICKS_EXPORT_COLUMNS) and len(parsed) == 17 and parsed["Data"].iloc[0] == second.start.isoformat()
    empty = picks_export_frame(draw_picks_table(state, third, n=0), btts_picks_table(state, third, n=0))
    assert list(empty.columns) == list(PICKS_EXPORT_COLUMNS) and empty.empty
    assert any("raggruppate in 3 giornate" in n for n in state.notes)


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


def test_matchday_table_for_and_results_table_for(sim_state: PipelineState) -> None:
    """The two tables the Giornata and Risultati tabs show, from the state."""
    table = matchday_table_for(sim_state)
    assert list(table.columns) == list(MATCHDAY_COLUMNS) and len(table) == 40 == len(fixtures_for(sim_state, None))
    assert set(table["match_id"]) == {p.fixture.match_id for p in sim_state.predictions}
    assert table["EV max (%)"].notna().all(), "the synthetic bookmaker quotes every fixture"
    trap_teams = {t.team for t in sim_state.traps}
    trapped_rows = table[table["Casa"].isin(trap_teams) | table["Ospiti"].isin(trap_teams)]
    assert len(trapped_rows) > 0 and trapped_rows["Flag"].str.contains("REGRESSION_TRAP | TRAP:", regex=False).all()
    assert not table[~table.index.isin(trapped_rows.index)]["Flag"].str.contains("TRAP", regex=False).any()
    only_sb = matchday_table_for(sim_state, None, Girone.SB)
    assert len(only_sb) == 10 and only_sb["Competizione"].eq("Serie B").all()
    assert matchday_table_for(sim_state, ALL_MATCHDAYS).equals(table) and matchday_table_for(sim_state, sim_state.next_matchday).equals(table)
    # the form columns come from the current season only (15 rounds -> five results for every team) and are cached per state
    assert table["Forma Casa"].str.count("-").eq(4).all() and table["Forma Ospiti"].str.count("-").eq(4).all()
    first = table.iloc[0]
    assert first["Forma Casa"] == team_form(sim_state.current_matches, first["Casa"]).results_last_n
    assert first["Pt Casa (5)"] == team_form(sim_state.current_matches, first["Casa"]).points_last_n
    frame = sim_state.team_matches_frame()
    assert frame is sim_state.team_matches_frame() and len(frame) == 2 * sim_state.n_current_matches
    assert dataclasses.replace(sim_state, quotes=[])._team_matches is None, "a copy of the state starts a fresh cache"
    parsed = pd.read_csv(io.BytesIO(matchday_csv_bytes(sim_state)))
    assert list(parsed.columns) == list(MATCHDAY_COLUMNS) and len(parsed) == 40 and parsed["Data"].iloc[0] == table["Data"].iloc[0].strftime("%Y-%m-%d")
    assert len(pd.read_csv(io.BytesIO(matchday_csv_bytes(sim_state, None, Girone.C)))) == 10
    # results: the current season by default, any season of the history on request, one competition, one team
    results = results_table_for(sim_state)
    assert list(results.columns) == list(RESULTS_MODEL_COLUMNS) and len(results) == sim_state.n_current_matches == 600
    assert results["Data"].is_monotonic_decreasing and results["P esito (%)"].between(0.0, 100.0).all()
    assert seasons_available(sim_state) == ["2026/27", "2025/26", "2024/25"]
    past = results_table_for(sim_state, season="2024/25")
    assert len(past) == 1520 and past["P esito (%)"].notna().all() and len(results_table_for(sim_state, season="2025_26")) == 1520
    assert len(results_table_for(sim_state, girone=Girone.B)) == 150 and len(results_table_for(sim_state, girone="Serie B", season="2024/25")) == 380
    team = sim_state.registry.teams(Girone.A)[0]
    mine = results_table_for(sim_state, team=team)
    assert len(mine) == 15 and ((mine["Casa"] == team) | (mine["Ospiti"] == team)).all()
    assert current_teams(sim_state) == sorted(sim_state.registry.canonical_names) and len(current_teams(sim_state)) == 80
    assert current_teams(sim_state, Girone.A) == sorted(sim_state.registry.teams(Girone.A)) == results_team_options(sim_state, Girone.A)
    historical = set(sim_state.params.historical_teams)
    options_past = results_team_options(sim_state, None, "2025/26")
    past_rows = results_table_for(sim_state, season="2025/26")
    # a past season offers exactly the clubs that played it (historical ones included), not the current registry
    assert set(options_past) == set(past_rows["Casa"].astype(str)) | set(past_rows["Ospiti"].astype(str))
    assert set(options_past) & historical and set(options_past) - set(current_teams(sim_state)) <= historical
    assert set(current_teams(sim_state)) - set(options_past), "a club new to the perimeter has no 2025/26 row and is not offered"
    hist = next(t for t in options_past if t in historical)
    rows = results_table_for(sim_state, team=hist, season="2025/26")
    assert len(rows) > 0 and ((rows["Casa"] == hist) | (rows["Ospiti"] == hist)).all() and rows["λ"].notna().all()
    assert results_table_for(sim_state, team=hist).empty, "not in the current season"
    parsed = pd.read_csv(io.BytesIO(results_csv_bytes(sim_state, team=team)))
    assert len(parsed) == 15 and list(parsed.columns) == list(RESULTS_MODEL_COLUMNS)


def test_fixture_label_and_selected_row_match_id(sim_state: PipelineState) -> None:
    """The pure helpers behind the Giornata tab's row selection."""
    pred = sim_state.predictions[0]
    assert fixture_label(pred) == f"[{pred.fixture.girone.value}] {pred.fixture.date.isoformat()} · {pred.fixture.home_team} - {pred.fixture.away_team}"
    table = matchday_table_for(sim_state)
    event = SimpleNamespace(selection=SimpleNamespace(rows=[2]))
    assert selected_row_match_id(event, table) == table["match_id"].iloc[2]
    assert selected_row_match_id(SimpleNamespace(selection=SimpleNamespace(rows=[])), table) is None
    assert selected_row_match_id(SimpleNamespace(selection=SimpleNamespace(rows=None)), table) is None
    assert selected_row_match_id(None, table) is None
    assert selected_row_match_id(SimpleNamespace(selection=SimpleNamespace(rows=[99])), table) is None, "the table changed under the selection"
    assert selected_row_match_id(SimpleNamespace(selection=SimpleNamespace(rows=[-1])), table) is None
    assert selected_row_match_id(event, table.drop(columns="match_id")) is None
    assert selected_row_match_id(event, table.iloc[0:0]) is None


def test_matchday_table_key_is_scoped_to_matchday_and_competition(sim_state: PipelineState) -> None:
    """Regression: the Giornata table widget key changes with the matchday and the sidebar competition.

    Streamlit keeps a ``st.dataframe`` row selection per widget key, whatever
    the data: with one fixed key a click on row 3 of *Tutti* survived a switch
    to *Serie B* and silently opened row 3 of the Serie B table. A scoped key is
    a fresh widget per scope, so the selection dies with its table.
    """
    md = sim_state.next_matchday
    assert md is not None
    keys = {
        matchday_table_key(None, None),
        matchday_table_key(md, None),
        *(matchday_table_key(None, g) for g in Girone),
        *(matchday_table_key(md, g) for g in Girone),
    }
    assert len(keys) == 2 + 2 * len(Girone), "every (matchday, competition) pair is its own widget"
    # ... and so is every section: the default section keeps the bare key, the others carry their slug
    assert matchday_table_key(md, None, DEFAULT_SECTION) == matchday_table_key(md, None) == matchday_table_key(md, None, SECTION_BC)
    assert matchday_table_key(md, None, SECTION_D) == f"{MATCHDAY_TABLE_KEY_PREFIX}_{md.key}_all_d" != matchday_table_key(md, None)
    assert matchday_table_key(md, Girone.A, SECTION_D).endswith("_A_d")
    assert all(k.startswith(MATCHDAY_TABLE_KEY_PREFIX + "_") for k in keys)
    assert matchday_table_key(None, None) == f"{MATCHDAY_TABLE_KEY_PREFIX}_tutte_all"
    assert matchday_table_key(md, Girone.SB) == f"{MATCHDAY_TABLE_KEY_PREFIX}_{md.key}_SB" == matchday_table_key(md, "Serie B")
    assert matchday_table_key(md, "A") == matchday_table_key(md, Girone.A) != matchday_table_key(md, Girone.B)
    # Streamlit recovers the user key from the element id by splitting on the first two hyphens only, so a key
    # holding the ISO date of the matchday ('2026-10-18') round-trips untouched
    from streamlit.runtime.state.common import user_key_from_element_id

    key = matchday_table_key(md, Girone.A)
    assert "-" in key and user_key_from_element_id(f"$$ID-0123456789abcdef-{key}") == key


def test_results_team_options_offer_only_teams_with_rows(sim_state: PipelineState) -> None:
    """Regression: for a past season the *Squadra* filter offers the clubs that played it in that competition.

    The options used to start from the registry (every current team under its
    present competition), while the table filters on the competition the match
    was played in: a club promoted or relegated since was offered under its new
    competition for a season it played elsewhere, and its table was empty.
    """
    registry = sim_state.registry
    assert registry is not None
    for season in seasons_available(sim_state):
        for g in (None, *PERIMETER_GIRONI):
            table = results_table_for(sim_state, girone=g, season=season)
            with_rows = set(table["Casa"].astype(str)) | set(table["Ospiti"].astype(str))
            options = results_team_options(sim_state, g, season)
            assert options == sorted(set(options)) and options, (season, g)
            if season == "2026/27":
                # the current season also offers the registry teams of the competition, played or not
                assert set(options) == with_rows | set(current_teams(sim_state, g)), (season, g)
            else:
                assert set(options) == with_rows, (season, g)
            for team in options:
                assert team in with_rows or season == "2026/27", (season, g, team)
    # a current team that played a past season in another competition (promoted/relegated in the simulated history)
    past = "2025/26"
    frame = sim_state.matches
    in_past = frame.loc[(pd.to_datetime(frame["date"]).map(season_of) == past).to_numpy(dtype=bool)]
    then_girone = dict(zip(in_past["home_team"].astype(str), in_past["girone"].astype(str)))
    moved = sorted(
        (team, Girone.parse(then_girone[team]), registry.girone_of(team))
        for team in registry.canonical_names
        if team in then_girone and Girone.parse(then_girone[team]) != registry.girone_of(team)
    )
    assert moved, "the simulated history has clubs that changed competition"
    team, then_g, now_g = moved[0]
    assert team in results_team_options(sim_state, then_g, past) and team in results_team_options(sim_state, None, past)
    assert team not in results_team_options(sim_state, now_g, past), "offered under its registry competition it would have no row"
    assert results_table_for(sim_state, girone=now_g, team=team, season=past).empty
    assert not results_table_for(sim_state, girone=then_g, team=team, season=past).empty
    assert team in results_team_options(sim_state, now_g), "the current season still offers every registry team"
    # a season no match belongs to falls back to the registry teams, so the selectbox is never empty
    assert results_team_options(sim_state, Girone.B, "2019/20") == sorted(registry.teams(Girone.B))


# ----------------------------------------------------------------------------- Streamlit UI
def test_app_renders_with_apptest(monkeypatch: pytest.MonkeyPatch) -> None:
    from streamlit.testing.v1 import AppTest

    monkeypatch.setenv("SERIEC_FORCE_SIMULATE", "1")
    monkeypatch.setenv("SERIEC_SIM_ROUNDS", "8")
    at = AppTest.from_file(str(APP_PATH), default_timeout=240)
    at.run()
    assert not at.exception, [e.value for e in at.exception]
    assert [tab.label for tab in at.tabs] == ["Classifica Forza", "Giornata", "Risultati", "Opportunità di Valore", "Selezioni di giornata", "Export", "Legenda"]
    assert len(at.get("download_button")) >= 5  # matchday table, results, picks, value report, predictions
    assert {"Scarica tabella giornata (CSV)", "Scarica risultati (CSV)", "Scarica le due liste (CSV)"} <= {b.label for b in at.get("download_button")}
    # Giornata: the whole matchday in one table (one row per fixture, match_id last) and the Partita fallback in sync
    matchday_table = matchday_table_shown(at)
    assert list(matchday_table.columns) == list(MATCHDAY_COLUMNS) and len(matchday_table) == 40
    fixture_box = at.selectbox(key="fixture")
    assert fixture_box.label == "Partita" and len(fixture_box.options) == 40
    assert {o.split(" · ")[1].split(" - ")[0] for o in fixture_box.options} == set(matchday_table["Casa"])
    assert any("Clicca una riga per aprire la partita" in c.value for c in at.caption)
    # Risultati: season (current first) and team filters, the played matches of the current season with the model's reading
    season_box = at.selectbox(key="results_season")
    assert season_box.options == ["2026/27", "2025/26", "2024/25"] and season_box.value == "2026/27"
    team_box = at.selectbox(key="results_team")
    assert team_box.options[0] == "Tutte" and len(team_box.options) == 81 and team_box.value == "Tutte"
    results = results_table_shown(at)
    assert list(results.columns) == list(RESULTS_MODEL_COLUMNS) and len(results) == 320  # 8 rounds x 40 matches
    assert any(c.value.startswith("P esito (%) = probabilità") and "Sorpresa" in c.value for c in at.caption)
    assert any("Dati simulati" in i.value for i in at.sidebar.info)
    # Selezioni di giornata: one matchday (the simulated league has one round), N controls, two picks tables, one CSV
    matchday_box = at.selectbox(key="matchday")
    assert len(matchday_box.options) == 1 and matchday_box.options[0].startswith("Giornata del 18 ott 2026 · Girone A g.9")
    assert at.session_state["matchday"] == "2026-10-18"
    assert at.number_input(key="n_draws").value == 5 and at.number_input(key="n_btts").value == 12
    picks_tables = [d.value for d in at.dataframe if list(d.value.columns) == list(PICKS_COLUMNS)]
    assert [len(t) for t in picks_tables] == [5, 12]
    assert picks_tables[0]["Esito"].eq("X").all() and picks_tables[1]["Esito"].eq("GG").all()
    assert any("Le 5 partite più probabili da X (tutte le competizioni di Serie B e Serie C)" in m.value for m in at.markdown)
    assert any("Le 12 partite più probabili da GG (tutte le competizioni di Serie B e Serie C)" in m.value for m in at.markdown)
    # the section radio: Serie B e Serie C by default, Serie D offered as soon as it is configured (initialised or not)
    section_radio = at.radio(key="section")
    assert section_radio.value == "Serie B e Serie C" == section_radio.options[0]
    assert section_radio.options == (["Serie B e Serie C", "Serie D"] if SERIE_D else ["Serie B e Serie C"])
    assert any("sezione: Serie B e Serie C" in c.value for c in at.caption)
    assert any("solo per probabilità" in c.value for c in at.caption)
    assert any(b.label == "Scarica le due liste (CSV)" for b in at.get("download_button"))
    # Legenda: the glossary and the worked example on the first value row
    assert any(m.value.startswith("## Legenda: come leggere i numeri") for m in at.markdown)
    assert any(e.label == "Come leggere una riga del pannello valore" for e in at.expander)
    assert any("Kelly pieno" in m.value and "**Stake**" in m.value and "Esempio costruito" not in m.value for m in at.markdown)
    assert_example_is_the_panel_row(at)
    # the N controls resize the lists
    at.number_input(key="n_draws").set_value(3).run()
    assert not at.exception, [e.value for e in at.exception]
    picks_tables = [d.value for d in at.dataframe if list(d.value.columns) == list(PICKS_COLUMNS)]
    assert [len(t) for t in picks_tables] == [3, 12]
    assert any("Le 3 partite più probabili da X" in m.value for m in at.markdown)
    # the value panel's "every matchday" switch
    at.checkbox(key="value_all_matchdays").check().run()
    assert not at.exception, [e.value for e in at.exception]
    assert any(c.value.startswith("Tutte le partite in programma") for c in at.caption)
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
    sb_table = matchday_table_shown(at)
    assert len(sb_table) == 10 and set(sb_table["Competizione"]) == {"Serie B"}
    assert at.selectbox(key="results_team").options == ["Tutte", *sorted(load_registry().teams(Girone.SB))]
    assert len(results_table_shown(at)) == 80  # 8 rounds x 10 Serie B matches
    # the Partita selectbox is the fallback to the row click: the detail (heatmap) follows it
    fourth = at.selectbox(key="fixture").options[3]
    at.selectbox(key="fixture").select(fourth).run()
    assert not at.exception, [e.value for e in at.exception]
    _z, layout = rendered_heatmap(at)
    assert fourth.split(" · ")[1] in layout["title"]["text"]
    assert any(m.label.startswith("λ casa") and m.value for m in at.metric)
    # Risultati filters: one team -> its 8 matches, a past season with 'Tutte' -> the whole synthetic season, a historical team
    team = at.selectbox(key="results_team").options[1]
    at.selectbox(key="results_team").select(team).run()
    assert not at.exception, [e.value for e in at.exception]
    mine = results_table_shown(at)
    assert len(mine) == 8 and ((mine["Casa"] == team) | (mine["Ospiti"] == team)).all() and set(mine["Competizione"]) == {"Serie B"}
    assert any(b.label.startswith("Scarica risultati") for b in at.get("download_button"))
    at.selectbox(key="results_season").select("2025/26")
    at.selectbox(key="results_team").select("Tutte")
    at.run()
    assert not at.exception, [e.value for e in at.exception]
    past = results_table_shown(at)
    assert len(past) == 380 and set(past["Competizione"]) == {"Serie B"}  # one synthetic Serie B season
    current_names = set(load_registry().canonical_names)
    historical = [o for o in at.selectbox(key="results_team").options[1:] if o not in current_names]
    assert historical, "a past season offers the clubs that played it and are no longer in the perimeter"
    at.selectbox(key="results_team").select(historical[0]).run()
    assert not at.exception, [e.value for e in at.exception]
    theirs = results_table_shown(at)
    assert len(theirs) == 38 and ((theirs["Casa"] == historical[0]) | (theirs["Ospiti"] == historical[0])).all()
    # back to the current season: the historical team is no longer an option, the filter resets without an error
    at.selectbox(key="results_season").select("2026/27").run()
    assert not at.exception, [e.value for e in at.exception]
    assert at.selectbox(key="results_team").value == "Tutte" and len(results_table_shown(at)) == 80


def test_app_row_click_opens_the_fixture_and_dies_with_its_table(monkeypatch: pytest.MonkeyPatch) -> None:
    """The row-click path of the Giornata tab, driven in-process by wrapping ``streamlit.dataframe``.

    ``AppTest`` cannot click a dataframe row, but it runs the script in-process:
    a wrapper around ``streamlit.dataframe`` renders the real table and, for the
    Giornata table, returns the selection Streamlit would (``selection.rows``),
    bound to the widget key it was made on exactly like Streamlit's own widget
    state. Checks: the *Partita* selectbox, the metrics and the heatmap follow
    the clicked row; a later manual selectbox choice is kept while the click
    stays; a selection made on one competition's table does not re-target the
    detail when the sidebar competition changes (regression: with a fixed key
    the stale row position was mapped onto the new table).
    """
    import streamlit
    from streamlit.testing.v1 import AppTest

    original = streamlit.dataframe
    click: dict[str, object] = {"key": None, "rows": []}
    calls: list[str | None] = []

    @functools.wraps(original)  # keeps the signature, so container_width_kwargs still picks the right keyword
    def fake_dataframe(data=None, *args, **kwargs):  # noqa: ANN001, ANN002, ANN003, ANN202
        result = original(data, *args, **kwargs)
        key = kwargs.get("key")
        if key is not None and str(key).startswith(MATCHDAY_TABLE_KEY_PREFIX):
            calls.append(str(key))
            rows = list(click["rows"]) if key == click["key"] else []  # a selection belongs to the widget it was made on
            return SimpleNamespace(selection=SimpleNamespace(rows=rows, columns=[]))
        return result

    monkeypatch.setattr(streamlit, "dataframe", fake_dataframe)
    monkeypatch.setenv("SERIEC_FORCE_SIMULATE", "1")
    monkeypatch.setenv("SERIEC_SIM_ROUNDS", "8")
    at = AppTest.from_file(str(APP_PATH), default_timeout=240)
    at.run()
    assert not at.exception, [e.value for e in at.exception]
    element = matchday_table_element(at)
    table = element.value
    assert calls and calls[-1] == element.key == f"{MATCHDAY_TABLE_KEY_PREFIX}_2026-10-18_all" and len(table) == 40
    labels = at.selectbox(key="fixture").options
    assert at.selectbox(key="fixture").value == labels[0] and "matchday_table_row" not in at.session_state
    # the Forma tooltip uses the legend's wording for the letters (one glossary everywhere)
    columns = json.loads(element.proto.columns)
    wording = "V vittoria, N pareggio (nullo), P sconfitta"
    assert wording in columns["Forma Casa"]["help"] and wording in columns["Forma Ospiti"]["help"]
    assert wording in legend_terms()["Forma (Casa / Ospiti)"]

    # 1) click row 3: selectbox, metrics and heatmap follow the clicked fixture
    k = 3
    click.update(key=element.key, rows=[k])
    at.run()
    assert not at.exception, [e.value for e in at.exception]
    row = table.iloc[k]
    partita = f"{row['Casa']} - {row['Ospiti']}"
    chosen = at.selectbox(key="fixture").value
    assert chosen.endswith(f" · {partita}") and chosen == labels[labels.index(chosen)]
    assert at.session_state["fixture"] == chosen and at.session_state["matchday_table_row"] == (element.key, row["match_id"])
    lam = next(m.value for m in at.metric if m.label.startswith("λ casa"))
    mu = next(m.value for m in at.metric if m.label.startswith("μ ospiti"))
    assert lam == f"{row['λ']:.2f}" and mu == f"{row['μ']:.2f}"
    top = next(m for m in at.metric if m.label == "Risultato più probabile")
    assert row["Ris. prob."].startswith(f"{top.value} (")
    _z, layout = rendered_heatmap(at)
    assert partita in layout["title"]["text"]
    # 2) a later manual choice in the selectbox is respected while the click stays on row 3
    other = labels[7]
    assert other != chosen
    at.selectbox(key="fixture").select(other).run()
    assert not at.exception, [e.value for e in at.exception]
    assert at.selectbox(key="fixture").value == other
    _z, layout = rendered_heatmap(at)
    assert other.split(" · ")[1] in layout["title"]["text"]
    # 3) the sidebar competition changes: the Serie B table is another widget, without a selection, so the detail
    #    falls back to the first Serie B fixture instead of row 3 of the new table (the regression)
    at.selectbox(key="girone").select("Serie B").run()
    assert not at.exception, [e.value for e in at.exception]
    sb_element = matchday_table_element(at)
    sb_table = sb_element.value
    assert sb_element.key == f"{MATCHDAY_TABLE_KEY_PREFIX}_2026-10-18_SB" != element.key and len(sb_table) == 10
    sb_labels = at.selectbox(key="fixture").options
    assert len(sb_labels) == 10 and at.selectbox(key="fixture").value == sb_labels[0]
    assert not at.selectbox(key="fixture").value.endswith(f" · {sb_table.iloc[k]['Casa']} - {sb_table.iloc[k]['Ospiti']}")
    assert "matchday_table_row" not in at.session_state
    _z, layout = rendered_heatmap(at)
    assert sb_labels[0].split(" · ")[1] in layout["title"]["text"]
    # 4) a click on the Serie B table works on its own key
    click.update(key=sb_element.key, rows=[2])
    at.run()
    assert not at.exception, [e.value for e in at.exception]
    sb_row = sb_table.iloc[2]
    assert at.selectbox(key="fixture").value.endswith(f" · {sb_row['Casa']} - {sb_row['Ospiti']}")
    assert at.session_state["matchday_table_row"] == (sb_element.key, sb_row["match_id"])
    # 5) back to every competition: the old widget's selection is gone (Streamlit drops the state of a widget not
    #    rendered), the Serie B choice is still a valid label of the wider list and is kept
    at.selectbox(key="girone").select("Tutti").run()
    assert not at.exception, [e.value for e in at.exception]
    assert matchday_table_element(at).key == element.key and len(matchday_table_shown(at)) == 40
    assert at.selectbox(key="fixture").value.endswith(f" · {sb_row['Casa']} - {sb_row['Ospiti']}")
    assert "matchday_table_row" not in at.session_state
    # 6) no selection at all: the Partita fallback keeps working
    click.update(key=None, rows=[])
    at.run()
    assert not at.exception, [e.value for e in at.exception]
    assert at.selectbox(key="fixture").value in at.selectbox(key="fixture").options and at.metric


def test_app_matchday_selector_drives_match_value_and_picks_tabs(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """With a season-long fixture list the three matchday-aware tabs show the next matchday and follow the selector."""
    from streamlit.testing.v1 import AppTest

    import src.dashboard.state as state_module

    processed, odds_dir, by_round = multi_round_processed_dir(tmp_path, seed=23)
    original = state_module.build_state

    def processed_build_state(**kwargs):
        return original(processed_dir=processed, odds_dir=odds_dir, **kwargs)

    monkeypatch.setattr(state_module, "build_state", processed_build_state)
    monkeypatch.delenv("SERIEC_FORCE_SIMULATE", raising=False)
    monkeypatch.delenv("SERIEC_SIM_ROUNDS", raising=False)
    monkeypatch.setenv("SERIEC_SEED", "23")  # a cache key no other test uses
    at = AppTest.from_file(str(APP_PATH), default_timeout=240)
    at.run()
    assert not at.exception, [e.value for e in at.exception]
    assert any("Dati reali" in s.value for s in at.sidebar.success)
    box = at.selectbox(key="matchday")
    assert len(box.options) == 3 and all(o.startswith("Giornata ") for o in box.options)
    assert "Serie B g.9" in box.options[0] and "Serie B" not in box.options[2]
    first_key = by_round[9][0].date.isoformat()
    assert at.session_state["matchday"] == first_key
    assert len(at.selectbox(key="fixture").options) == 40, "the next matchday, not the 110 fixtures of the season"
    assert len(matchday_table_shown(at)) == 40
    value_tables = [d.value for d in at.dataframe if list(d.value.columns) == list(VALUE_COLUMNS)]
    first_labels = {f"{f.home_team} - {f.away_team}" for f in by_round[9]}
    assert value_tables and set(value_tables[0]["Partita"]) <= first_labels
    picks_tables = [d.value for d in at.dataframe if list(d.value.columns) == list(PICKS_COLUMNS)]
    assert [len(t) for t in picks_tables] == [5, 12] and all(set(t["Partita"]) <= first_labels for t in picks_tables)
    assert any("la giornata si cambia nel tab" in c.value and "(3 in programma)" in c.value for c in at.caption)
    # the Legenda example is a row of the panel shown (next matchday), not the best EV of the whole season
    assert_example_is_the_panel_row(at)
    fields, _text = worked_example_shown(at)
    assert fields["partita"] in first_labels

    # choose the third matchday: the Giornata table, the value panel and the picks follow
    box.select(box.options[2]).run()
    assert not at.exception, [e.value for e in at.exception]
    third_key = by_round[11][0].date.isoformat()
    assert at.session_state["matchday"] == third_key
    third_labels = {f"{f.home_team} - {f.away_team}" for f in by_round[11]}
    assert len(at.selectbox(key="fixture").options) == 30 and not any("[SB]" in o for o in at.selectbox(key="fixture").options)
    third_table = matchday_table_shown(at)
    assert len(third_table) == 30 and "Serie B" not in set(third_table["Competizione"])
    assert set(third_table["match_id"]) == {f.match_id for f in by_round[11]}
    assert set(third_table["Casa"].astype(str) + " - " + third_table["Ospiti"].astype(str)) == third_labels
    value_tables = [d.value for d in at.dataframe if list(d.value.columns) == list(VALUE_COLUMNS)]
    assert value_tables and set(value_tables[0]["Partita"]) <= third_labels and not set(value_tables[0]["Competizione"]) & {"Serie B"}
    picks_tables = [d.value for d in at.dataframe if list(d.value.columns) == list(PICKS_COLUMNS)]
    assert [len(t) for t in picks_tables] == [5, 12] and all(set(t["Partita"]) <= third_labels for t in picks_tables)
    # the sidebar competition filter still applies to the Giornata tab within the selected matchday
    at.selectbox(key="girone").select("Girone A").run()
    assert not at.exception, [e.value for e in at.exception]
    assert len(at.selectbox(key="fixture").options) == 10 and all(o.startswith("[A]") for o in at.selectbox(key="fixture").options)
    assert len(matchday_table_shown(at)) == 10
    assert at.session_state["matchday"] == third_key, "the matchday choice survives other interactions"
    # ... and so does the Legenda example: third matchday, Girone A only
    assert_example_is_the_panel_row(at)
    fields, _text = worked_example_shown(at)
    assert fields["partita"] in third_labels and fields["comp"] == "Girone A"
    # the EV threshold at its maximum: whatever the panel shows, the example is consistent with it (synthetic with the
    # 'no value row' reason when the panel is empty, never 'no quotes' since quotes exist)
    at.slider(key="min_ev_pct").set_value(15.0).run()
    assert not at.exception, [e.value for e in at.exception]
    assert_example_is_the_panel_row(at)
    assert not any(SYNTHETIC_REASON_NO_QUOTES in m.value for m in at.markdown)


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
    from streamlit.testing.v1 import AppTest

    import src.dashboard.state as state_module

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


# ----------------------------------------------------------------------------- sections (Serie B e Serie C / Serie D)
def serie_d_registry(seed: int = 7) -> TeamRegistry:
    """The real 2026/27 perimeter plus the synthetic Serie D gironi (18 clubs each), as the console demo builds it."""
    base = load_registry()
    extra = simulate_serie_d_registry(seed=seed, season=base.season)
    teams = {g.value: tuple(base.teams_by_girone[g]) for g in base.gironi}
    teams.update({g.value: tuple(extra.teams_by_girone[g]) for g in extra.gironi})
    return TeamRegistry.from_teams(base.season, teams)


@pytest.fixture(scope="module")
def serie_d_state() -> PipelineState:
    """A simulated perimeter with Serie D initialised: 242 teams, 4 rounds, no history (one joint fit)."""
    if not SERIE_D:
        pytest.skip("no competition of family 'Serie D' in config/competitions.toml")
    return build_state(simulate=True, seed=7, xi=0.0, n_rounds=4, history_seasons=0, registry=serie_d_registry())


def test_section_helpers_are_configuration_driven() -> None:
    """The pure section helpers: families -> sections, slugs, keys, configured competitions, options, status texts."""
    assert SECTION_BC == "Serie B e Serie C" and SECTION_D == "Serie D" and DEFAULT_SECTION == SECTION_BC
    assert SECTION_FAMILIES == {SECTION_BC: ("Serie C", "Serie B"), SECTION_D: ("Serie D",)}
    assert section_of_family("Serie C") == section_of_family("Serie B") == family_label("Serie C") == SECTION_BC
    assert section_of_family("Serie D") == family_label("Serie D") == SECTION_D
    assert section_of_family("Eccellenza") == "Eccellenza", "an unknown family is a section of its own"
    assert section_families(SECTION_BC) == ("Serie C", "Serie B") and section_families(SECTION_D) == ("Serie D",) and section_families("Eccellenza") == ("Eccellenza",)
    assert section_slug(SECTION_BC) == "bc" and section_slug(SECTION_D) == "d" and section_slug("Lega Pro Élite!") == "lega_pro_lite"
    assert section_key("matchday") == section_key("matchday", None) == section_key("matchday", SECTION_BC) == "matchday"
    assert section_key("matchday", SECTION_D) == "matchday_d" and section_key("fixture", "Eccellenza") == "fixture_eccellenza"
    # configured competitions per section, from config/competitions.toml
    assert configured_gironi(None) == tuple(Girone) and configured_gironi(SECTION_BC) == (Girone.A, Girone.B, Girone.C, Girone.SB)
    assert configured_gironi(SECTION_D) == SERIE_D and all(g.family == "Serie D" for g in SERIE_D)
    assert all(in_section(g, None) for g in Girone) and in_section("A", SECTION_BC) and in_section(Girone.SB, SECTION_BC)
    assert not in_section(Girone.A, SECTION_D) and (not SERIE_D or in_section(SERIE_D[0], SECTION_D) and not in_section(SERIE_D[0], SECTION_BC))
    assert competition_options(SECTION_BC) == ("Tutti", "Girone A", "Girone B", "Girone C", "Serie B")
    assert competition_options(None) == ("Tutti", *(g.short for g in Girone))
    assert all_sections() == ((SECTION_BC, SECTION_D) if SERIE_D else (SECTION_BC,)) and sections_for([Girone.SB]) == (SECTION_BC,)
    assert sections_for([]) == ()
    if SERIE_D:
        assert [g.value for g in SERIE_D] == ["DA", "DB", "DC", "DD", "DE", "DF", "DG", "DH", "DI"]
        assert competition_options(SECTION_D) == ("Tutti", "D-A", "D-B", "D-C", "D-D", "D-E", "D-F", "D-G", "D-H", "D-I")
        assert sections_for(SERIE_D[:1]) == (SECTION_D,) and sections_for([Girone.A, SERIE_D[0]]) == (SECTION_BC, SECTION_D)
        assert serie_d_harvest_command() == "python -m src.ingestion.footystats --competitions DA DB DC DD DE DF DG DH DI --seasons 3"
        assert serie_d_harvest_command(SERIE_D[:2]) == "python -m src.ingestion.footystats --competitions DA DB --seasons 3"
        # the status texts, by counts
        assert serie_d_status_text(0, 0) == SERIE_D_STATUS_NOT_INITIALISED == "non inizializzata: esegui la raccolta FootyStats con la chiave"
        assert serie_d_status_text(162, 0) == "inizializzata: 162 squadre, 0 partite nei dati: esegui la raccolta FootyStats con la chiave"
        assert serie_d_status_text(162, 648, season_label="2026/27") == "inizializzata: 162 squadre, 648 partite della stagione 2026/27"
        assert serie_d_status_text(162, 648, 81, season_label="2026/27") == "inizializzata: 162 squadre, 648 partite della stagione 2026/27, 81 partite in programma previste"
    else:
        assert serie_d_status_text(0, 0) == SERIE_D_STATUS_NOT_CONFIGURED


def test_sections_of_the_shipped_perimeter_without_serie_d_teams(sim_state: PipelineState) -> None:
    """With the shipped dictionaries Serie D has no teams: one section with data, Serie D offered but not ready."""
    assert sim_state.perimeter_gironi == PERIMETER_GIRONI == (Girone.A, Girone.B, Girone.C, Girone.SB)
    assert sim_state.families == ("Serie C", "Serie B") and sim_state.family_of("A") == "Serie C" and sim_state.family_of(Girone.SB) == "Serie B"
    assert sections(sim_state) == (SECTION_BC,)
    assert section_options(sim_state) == ((SECTION_BC, SECTION_D) if SERIE_D else (SECTION_BC,))
    assert not sim_state.serie_d_initialised and section_team_count(sim_state, SECTION_D) == 0 and section_team_count(sim_state, SECTION_BC) == 80 == section_team_count(sim_state)
    assert sim_state.serie_d_status == (SERIE_D_STATUS_NOT_INITIALISED if SERIE_D else SERIE_D_STATUS_NOT_CONFIGURED)
    assert section_ready(sim_state, SECTION_BC) and section_ready(sim_state) and not section_ready(sim_state, SECTION_D)
    assert section_gironi(sim_state, SECTION_BC) == PERIMETER_GIRONI and section_gironi(sim_state, SECTION_D) == () and family_gironi(sim_state, "Serie C") == (Girone.A, Girone.B, Girone.C)
    # restricting to the only section with data changes nothing; restricting to Serie D yields empty views, never errors
    assert perimeter_text(sim_state, SECTION_BC) == perimeter_text(sim_state) == sim_state.perimeter_text
    assert section_perimeter_label(sim_state, SECTION_BC) == sim_state.perimeter_label == "Serie C 2026/27 Gironi A/B/C + Serie B (80 squadre)"
    assert perimeter_text(sim_state, SECTION_D) == "Perimetro: Serie D 2026/27 (nessuna squadra) · 0 partite stagione corrente · 0 partite storiche"
    assert ranking_table(sim_state, None, SECTION_BC).equals(ranking_table(sim_state, None)) and ranking_table(sim_state, None, SECTION_D).empty
    assert list(ranking_table(sim_state, None, SECTION_D).columns) == list(RANKING_COLUMNS)
    assert matchdays(sim_state, SECTION_BC) == sim_state.matchdays == matchdays(sim_state) and matchdays(sim_state, SECTION_D) == []
    assert matchdays(sim_state, SECTION_BC) is not sim_state.matchdays, "a copy, the state's list is never handed out"
    assert fixtures_for(sim_state, None, section=SECTION_BC) == fixtures_for(sim_state, None) and fixtures_for(sim_state, None, section=SECTION_D) == []
    assert section_predictions(sim_state, SECTION_D) == [] and family_predictions(sim_state, "Serie D") == [] and len(family_predictions(sim_state, "Serie B")) == 10
    assert len(family_predictions(sim_state, "Serie C")) == 30 and len(section_predictions(sim_state, SECTION_BC)) == 40 == len(section_predictions(sim_state))
    assert resolve_matchday(sim_state, None, SECTION_D) is None and find_matchday(sim_state, sim_state.next_matchday.key, SECTION_D) is None
    assert value_table(sim_state, 0.0, section=SECTION_BC).equals(value_table(sim_state, 0.0)) and value_table(sim_state, -1.0, section=SECTION_D).empty
    assert draw_picks_table(sim_state, section=SECTION_BC).equals(draw_picks_table(sim_state)) and btts_picks_table(sim_state, section=SECTION_D).empty
    assert results_table_for(sim_state, section=SECTION_BC).equals(results_table_for(sim_state)) and results_table_for(sim_state, section=SECTION_D).empty
    assert traps_table(sim_state, SECTION_BC).equals(traps_table(sim_state)) and traps_table(sim_state, SECTION_D).empty and section_traps(sim_state, SECTION_D) == []
    assert predictions_table(sim_state, SECTION_BC).equals(predictions_table(sim_state)) and predictions_table(sim_state, SECTION_D).empty
    assert opportunities_table(sim_state, SECTION_BC).equals(opportunities_table(sim_state)) and section_opportunities(sim_state, SECTION_D) == []
    assert current_teams(sim_state, section=SECTION_BC) == current_teams(sim_state) and current_teams(sim_state, section=SECTION_D) == []
    assert results_team_options(sim_state, section=SECTION_D) == [] and seasons_available(sim_state, SECTION_D) == ["2026/27"]
    assert seasons_available(sim_state, SECTION_BC) == seasons_available(sim_state) == ["2026/27", "2025/26", "2024/25"]
    assert section_matches(sim_state.matches, SECTION_BC).equals(sim_state.matches) and section_matches(sim_state.matches, SECTION_D).empty
    assert coverage_text(sim_state, SECTION_BC) == coverage_text(sim_state) == f"Copertura 2026/27: {sim_state.coverage.thinnest:.1f} partite per squadra nella competizione meno coperta"
    assert coverage_text(sim_state, SECTION_D) == ""


@needs_serie_d
def test_serie_d_state_splits_every_view_by_section(serie_d_state: PipelineState) -> None:
    """One joint fit, two sections: every table, pick list, matchday and export is restricted to the chosen section."""
    state = serie_d_state
    assert state.registry is not None and state.registry.n_teams == 242 and len(state.params.current_teams) == 242
    assert state.perimeter_gironi == tuple(Girone) and state.families == ("Serie C", "Serie B", "Serie D")
    assert sections(state) == section_options(state) == (SECTION_BC, SECTION_D)
    assert state.serie_d_initialised and section_team_count(state, SECTION_D) == 162 and section_team_count(state) == 242
    assert state.serie_d_status == "inizializzata: 162 squadre, 324 partite della stagione 2026/27, 81 partite in programma previste"
    assert section_ready(state, SECTION_D) and section_ready(state, SECTION_BC)
    assert section_gironi(state, SECTION_D) == SERIE_D == family_gironi(state, "Serie D")
    # perimeter texts: the whole perimeter, and each section on its own
    assert state.perimeter_label == "Serie C 2026/27 Gironi A/B/C + Serie B + Serie D Gironi A-I (242 squadre)"
    assert perimeter_text(state, SECTION_BC) == "Perimetro: Serie C 2026/27 Gironi A/B/C + Serie B (80 squadre) · 160 partite stagione corrente · 0 partite storiche"
    assert perimeter_text(state, SECTION_D) == "Perimetro: Serie D 2026/27 Gironi A-I (162 squadre) · 324 partite stagione corrente · 0 partite storiche"
    assert section_perimeter_label(state, SECTION_D) == "Serie D 2026/27 Gironi A-I (162 squadre)"
    # ranking: 242 teams in all, 80 + 162 by section, one girone of Serie D
    assert len(ranking_table(state, None)) == 242 and len(ranking_table(state, None, SECTION_BC)) == 80 and len(ranking_table(state, None, SECTION_D)) == 162
    d_ranking = ranking_table(state, None, SECTION_D)
    assert set(d_ranking["Competizione"]) == {g.short for g in SERIE_D} and len(ranking_table(state, SERIE_D[2])) == 18
    assert d_ranking["Competizione"].str.startswith("D-").all() and not ranking_table(state, None, SECTION_BC)["Competizione"].str.startswith("D-").any()
    # matchdays are grouped per section: same weekend, different labels, keys shared, rounds disjoint
    bc_days, d_days, all_days = matchdays(state, SECTION_BC), matchdays(state, SECTION_D), matchdays(state)
    assert len(bc_days) == len(d_days) == len(all_days) == 1
    assert set(bc_days[0].rounds) == set(PERIMETER_GIRONI) and set(d_days[0].rounds) == set(SERIE_D) and set(all_days[0].rounds) == set(Girone)
    assert "D-A" not in bc_days[0].label and "Girone A" not in d_days[0].label and bc_days[0].key == d_days[0].key
    assert bc_days[0].n_fixtures == 40 and d_days[0].n_fixtures == 81 and all_days[0].n_fixtures == 121
    assert resolve_matchday(state, None, SECTION_D) == d_days[0] and find_matchday(state, d_days[0].key, SECTION_D) == d_days[0]
    assert matchdays(state, SECTION_D) == d_days, "cached per section and returned as a copy"
    # fixtures, matchday table, predictions
    assert len(fixtures_for(state, None, section=SECTION_BC)) == 40 and len(fixtures_for(state, None, section=SECTION_D)) == 81 and len(fixtures_for(state, None)) == 121
    assert all(p.fixture.girone.family == "Serie D" for p in fixtures_for(state, None, section=SECTION_D))
    assert len(fixtures_for(state, SERIE_D[0], section=SECTION_D)) == 9 and fixtures_for(state, Girone.A, section=SECTION_D) == []
    assert len(matchday_predictions(state, section=SECTION_D)) == 81 and len(family_predictions(state, "Serie D")) == 81
    d_table = matchday_table_for(state, section=SECTION_D)
    assert len(d_table) == 81 and d_table["Competizione"].str.startswith("D-").all() and list(d_table.columns) == list(MATCHDAY_COLUMNS)
    assert len(matchday_table_for(state, section=SECTION_BC)) == 40 and len(matchday_table_for(state)) == 121
    assert len(pd.read_csv(io.BytesIO(matchday_csv_bytes(state, section=SECTION_D)))) == 81
    assert len(predictions_table(state, SECTION_D)) == 81 and predictions_table(state, SECTION_D)["girone"].str.startswith("D").all() and len(predictions_table(state)) == 121
    # picks are ranked within the section: 5 X / 12 GG of Serie D are Serie D matches, the top P(X) of that section
    d_draws, d_btts = draw_picks_table(state, section=SECTION_D), btts_picks_table(state, section=SECTION_D)
    assert len(d_draws) == 5 and len(d_btts) == 12 and d_draws["Competizione"].str.startswith("D-").all() and d_btts["Competizione"].str.startswith("D-").all()
    top_x = sorted((round(p.probabilities[Market.MATCH_ODDS]["X"] * 100.0, 1) for p in section_predictions(state, SECTION_D)), reverse=True)[:5]
    assert d_draws["P. Modello (%)"].tolist() == top_x
    bc_draws = draw_picks_table(state, section=SECTION_BC)
    assert len(bc_draws) == 5 and not bc_draws["Competizione"].str.startswith("D-").any()
    assert len(draw_picks_table(state, section=SECTION_D, girone=SERIE_D[1])) == 5 and draw_picks_table(state, section=SECTION_D, girone=SERIE_D[1])["Competizione"].eq(SERIE_D[1].short).all()
    assert draw_picks_table(state, section=SECTION_BC, girone=SERIE_D[1]).empty, "a competition outside the section has no candidates"
    # value, traps, results, teams, exports
    n_bc, n_d, n_all = (len(value_table(state, -1.0, section=s)) for s in (SECTION_BC, SECTION_D, None))
    assert n_bc + n_d == n_all == len(state.opportunities) and n_d == len(section_opportunities(state, SECTION_D)) == 891
    assert value_table(state, -1.0, section=SECTION_D)["Competizione"].str.startswith("D-").all()
    assert len(value_opportunities(state, -1.0, section=SECTION_D, matchday=ALL_MATCHDAYS)) == 891
    assert len(opportunities_table(state, SECTION_D)) == 891 and len(opportunities_table(state)) == len(state.opportunities)
    assert len(traps_table(state, SECTION_BC)) + len(traps_table(state, SECTION_D)) == len(traps_table(state)) == len(state.traps)
    assert all(t.girone.family == "Serie D" for t in section_traps(state, SECTION_D))
    assert len(results_table_for(state, section=SECTION_D)) == 324 and results_table_for(state, section=SECTION_D)["Competizione"].str.startswith("D-").all()
    assert len(results_table_for(state, section=SECTION_BC)) == 160 and len(results_table_for(state)) == 484 == state.n_current_matches
    assert len(results_table_for(state, girone=SERIE_D[0], section=SECTION_D)) == 36 and results_table_for(state, girone=Girone.A, section=SECTION_D).empty
    assert len(current_teams(state, section=SECTION_D)) == 162 == len(results_team_options(state, section=SECTION_D)) and len(current_teams(state, SERIE_D[0])) == 18
    assert len(current_teams(state, section=SECTION_BC)) == 80 and len(current_teams(state)) == 242
    assert seasons_available(state, SECTION_D) == ["2026/27"] and len(section_matches(state.current_matches, SECTION_D)) == 324
    assert coverage_text(state, SECTION_D) == "Copertura 2026/27: 4.0 partite per squadra nella competizione meno coperta"
    # the Serie D fixture labels carry the code, the widget keys the section slug
    assert fixture_label(fixtures_for(state, None, section=SECTION_D)[0]).startswith("[D")
    assert matchday_table_key(d_days[0], None, SECTION_D) == f"{MATCHDAY_TABLE_KEY_PREFIX}_{d_days[0].key}_all_d"
    assert dataclasses.replace(state, quotes=[])._section_matchdays == {}, "a copy of the state starts a fresh matchday cache"


@needs_serie_d
def test_app_section_radio_switches_to_serie_d(monkeypatch: pytest.MonkeyPatch) -> None:
    """With Serie D initialised the sidebar offers two sections; Serie D swaps every selector, table and list to its own data."""
    from streamlit.testing.v1 import AppTest

    import src.dashboard.state as state_module

    original = state_module.build_state
    registry = serie_d_registry()

    def serie_d_build_state(**kwargs):
        return original(registry=registry, **kwargs)

    monkeypatch.setattr(state_module, "build_state", serie_d_build_state)
    monkeypatch.setenv("SERIEC_FORCE_SIMULATE", "1")
    monkeypatch.setenv("SERIEC_SIM_ROUNDS", "4")
    monkeypatch.setenv("SERIEC_HISTORY", "0")
    monkeypatch.setenv("SERIEC_SEED", "29")  # a cache key no other test uses
    at = AppTest.from_file(str(APP_PATH), default_timeout=240)
    at.run()
    assert not at.exception, [e.value for e in at.exception]
    radio = at.radio(key="section")
    assert radio.options == ["Serie B e Serie C", "Serie D"] and radio.value == "Serie B e Serie C"
    assert at.selectbox(key="girone").options == ["Tutti", "Girone A", "Girone B", "Girone C", "Serie B"]
    rankings = [d.value for d in at.dataframe if list(d.value.columns) == list(RANKING_COLUMNS)]
    assert len(rankings) == 1 and len(rankings[0]) == 80, "the default section ranks the 80 Serie B / Serie C teams only"
    assert len(matchday_table_shown(at)) == 40 and len(at.selectbox(key="fixture").options) == 40
    assert any("Perimetro: Serie C 2026/27 Gironi A/B/C + Serie B (80 squadre) · 160 partite stagione corrente" in i.value for i in at.sidebar.info)
    assert any("Serie D: inizializzata: 162 squadre, 324 partite" in i.value for i in at.sidebar.info)
    assert "matchday" in at.session_state and "matchday_d" not in at.session_state

    # switch to Serie D: competition options, ranking, matchday table, picks, perimeter, keys
    radio.set_value("Serie D").run()
    assert not at.exception, [e.value for e in at.exception]
    assert at.radio(key="section").value == "Serie D"
    assert at.selectbox(key="girone").options == ["Tutti", "D-A", "D-B", "D-C", "D-D", "D-E", "D-F", "D-G", "D-H", "D-I"] and at.selectbox(key="girone").value == "Tutti"
    rankings = [d.value for d in at.dataframe if list(d.value.columns) == list(RANKING_COLUMNS)]
    assert len(rankings) == 1 and len(rankings[0]) == 162 and rankings[0]["Competizione"].str.startswith("D-").all()
    assert any(h.value.startswith("Classifica di forza · Serie D") for h in at.subheader)
    assert any("sezione: Serie D" in c.value for c in at.caption)
    assert any("Perimetro: Serie D 2026/27 Gironi A-I (162 squadre) · 324 partite stagione corrente · 0 partite storiche" in i.value for i in at.sidebar.info)
    assert any("Perimetro: Serie D 2026/27 Gironi A-I (162 squadre)" in c.value for c in at.caption)
    d_table = matchday_table_shown(at)
    assert len(d_table) == 81 and d_table["Competizione"].str.startswith("D-").all()
    assert matchday_table_element(at).key.endswith("_all_d")
    fixture_box = at.selectbox(key="fixture_d")
    assert len(fixture_box.options) == 81 and all(o.startswith("[D") for o in fixture_box.options)
    assert at.session_state["matchday_d"] == "2026-09-20", "round 5 of Serie D on the first Sunday after four played rounds, under the section's own key"
    assert "matchday" not in at.session_state, "Streamlit drops the state of the Serie B / Serie C selector while it is not rendered"
    assert at.selectbox(key="matchday_d").options[0].startswith("Giornata del 20 set 2026 · D-A g.5")
    assert any("D-A g.5" in c.value and "D-I g.5" in c.value and "Girone A" not in c.value for c in at.caption), "the Serie D matchday label names Serie D rounds only"
    picks_tables = [d.value for d in at.dataframe if list(d.value.columns) == list(PICKS_COLUMNS)]
    assert [len(t) for t in picks_tables] == [5, 12] and all(t["Competizione"].str.startswith("D-").all() for t in picks_tables)
    assert any("Le 5 partite più probabili da X (tutte le competizioni di Serie D)" in m.value for m in at.markdown)
    assert at.selectbox(key="results_team_d").options[0] == "Tutte" and len(at.selectbox(key="results_team_d").options) == 163
    assert len(results_table_shown(at)) == 324
    assert_example_is_the_panel_row(at)
    assert not any("non inizializzata" in i.value for i in at.info)
    # one Serie D girone: 9 fixtures of 18 teams, the table widget is its own
    at.selectbox(key="girone").select("D-C").run()
    assert not at.exception, [e.value for e in at.exception]
    assert len(at.selectbox(key="fixture_d").options) == 9 and all(o.startswith("[DC]") for o in at.selectbox(key="fixture_d").options)
    assert len(matchday_table_shown(at)) == 9 and matchday_table_element(at).key.endswith("_DC_d")
    assert len(at.selectbox(key="results_team_d").options) == 19 and len(results_table_shown(at)) == 36
    assert any("vista: D-C" in c.value for c in at.caption)
    # back to Serie B e Serie C: the competition filter resets (D-C is not an option there), the tables are the old ones
    at.radio(key="section").set_value("Serie B e Serie C").run()
    assert not at.exception, [e.value for e in at.exception]
    assert at.selectbox(key="girone").value == "Tutti" and len(matchday_table_shown(at)) == 40 and len(at.selectbox(key="fixture").options) == 40
    rankings = [d.value for d in at.dataframe if list(d.value.columns) == list(RANKING_COLUMNS)]
    assert len(rankings[0]) == 80


@needs_serie_d
def test_app_serie_d_not_initialised_shows_the_notice_in_every_tab(monkeypatch: pytest.MonkeyPatch) -> None:
    """With the shipped dictionary Serie D has no teams: the section is offered and every tab shows the status and the command."""
    from streamlit.testing.v1 import AppTest

    monkeypatch.setenv("SERIEC_FORCE_SIMULATE", "1")
    monkeypatch.setenv("SERIEC_SIM_ROUNDS", "8")
    at = AppTest.from_file(str(APP_PATH), default_timeout=240)
    at.run()
    assert not at.exception, [e.value for e in at.exception]
    assert at.radio(key="section").options == ["Serie B e Serie C", "Serie D"]
    assert any("Serie D: non inizializzata: esegui la raccolta FootyStats con la chiave" in i.value for i in at.sidebar.info)
    at.radio(key="section").set_value("Serie D").run()
    assert not at.exception, [e.value for e in at.exception]
    assert at.selectbox(key="girone").options == ["Tutti", "D-A", "D-B", "D-C", "D-D", "D-E", "D-F", "D-G", "D-H", "D-I"]
    command = "python -m src.ingestion.footystats --competitions DA DB DC DD DE DF DG DH DI --seasons 3"
    notices = [i.value for i in at.info if command in i.value]
    assert len(notices) == 6, "one notice per tab (Classifica, Giornata, Risultati, Valore, Selezioni, Export); the Legenda carries its own status line"
    assert all("Serie D non inizializzata: esegui la raccolta FootyStats con la chiave" in n and "avvia.bat" in n for n in notices)
    assert any("Sezione Serie D: non inizializzata" in i.value for i in at.info)
    assert not any(list(d.value.columns) == list(RANKING_COLUMNS) for d in at.dataframe), "no Serie D table is rendered"
    assert not any(list(d.value.columns) == list(MATCHDAY_COLUMNS) for d in at.dataframe)
    assert "matchday_d" not in at.session_state and "fixture_d" not in at.session_state
    assert any("Perimetro: Serie D 2026/27 (nessuna squadra)" in i.value for i in at.sidebar.info)
    assert any("## Legenda" in m.value for m in at.markdown), "the glossary is still shown"
    # back to the default section: everything renders as before
    at.radio(key="section").set_value("Serie B e Serie C").run()
    assert not at.exception, [e.value for e in at.exception]
    assert len(matchday_table_shown(at)) == 40 and at.selectbox(key="girone").options == ["Tutti", "Girone A", "Girone B", "Girone C", "Serie B"]
