"""Tests for ``src.engine.tables``: team form, the matchday table and the results table."""
from __future__ import annotations

import dataclasses
import math
import time
from datetime import date

import numpy as np
import pandas as pd
import pytest

from src.common.schemas import Fixture, Girone, Market, OddsQuote, validate_matches_df
from src.engine.model import DixonColesConfig, DixonColesEngine, dixon_coles_pmf, markets_from_matrix
from src.engine.picks import next_matchday
from src.engine.tables import (
    DEFAULT_FORM_N,
    DEFAULT_XG_N,
    EXPECTED_ABOVE,
    EXPECTED_TAG,
    MATCHDAY_COLUMNS,
    MISSING_TEXT,
    P_OUTCOME_DECIMALS,
    RESULT_DRAW,
    RESULT_LOSS,
    RESULT_WIN,
    RESULTS_COLUMNS,
    RESULTS_MODEL_COLUMNS,
    SURPRISE_BELOW,
    SURPRISE_TAG,
    TeamForm,
    _FixtureQuotes,
    _quotes_by_fixture,
    form_table,
    matchday_columns,
    matchday_table,
    outcome_probabilities,
    p_outcome_percent,
    results_table,
    surprise_tag,
    team_form,
    team_matches,
)
from src.engine.value import FLAG_REGRESSION_TRAP, TRAP_FLAG_PREFIX, ValueEngine, detect_regression_traps
from src.simulation.generator import inject_regression_trap, simulate_history, simulate_league

AS_OF = date(2026, 11, 6)


# ----------------------------------------------------------------------------- hand-built frame
def _row(mid: str, when: str, rnd: int, home: str, away: str, hg: int, ag: int, hxg: float | None, axg: float | None) -> dict:
    return {
        "match_id": mid,
        "date": when,
        "round": rnd,
        "girone": "A",
        "home_team": home,
        "away_team": away,
        "home_goals": hg,
        "away_goals": ag,
        "home_shots": 10.0,
        "away_shots": 8.0,
        "home_shots_target": 4.0,
        "away_shots_target": float("nan"),
        "home_corners": 5.0,
        "away_corners": 3.0,
        "home_xg": float("nan") if hxg is None else hxg,
        "away_xg": float("nan") if axg is None else axg,
    }


def hand_frame() -> pd.DataFrame:
    """Alpha's 2026/27 season (six matches, one with a missing xG) plus one 2025/26 match and one match without Alpha."""
    rows = [
        _row("OLD", "2026-05-10", 38, "Alpha", "Beta", 4, 0, 3.0, 0.2),  # season 2025/26
        _row("R1", "2026-09-06", 1, "Alpha", "Beta", 2, 0, 1.5, 0.4),  # V, clean sheet
        _row("R2", "2026-09-13", 2, "Gamma", "Alpha", 1, 1, 0.9, 1.2),  # N
        _row("R3", "2026-09-20", 3, "Alpha", "Delta", 0, 3, 0.7, 2.1),  # P
        _row("R4", "2026-09-27", 4, "Beta", "Alpha", 0, 1, 1.1, 0.8),  # V, clean sheet
        _row("R5", "2026-10-04", 5, "Alpha", "Gamma", 2, 2, None, 1.3),  # N, Alpha's xG missing
        _row("R6", "2026-10-11", 6, "Delta", "Alpha", 2, 0, 1.6, 0.5),  # P
        _row("X1", "2026-10-18", 7, "Beta", "Gamma", 1, 0, 1.0, 0.6),  # no Alpha
    ]
    return validate_matches_df(pd.DataFrame(rows))


def test_team_form_points_goals_results_and_nan_rule() -> None:
    form = team_form(hand_frame(), "Alpha")  # every match, default window of 5
    assert isinstance(form, TeamForm) and form.n == DEFAULT_FORM_N == 5
    assert form.team == "Alpha" and form.played == 7 and form.n_window == 5
    assert form.results_last_n == "P-N-V-P-N"  # R6 R5 R4 R3 R2, most recent first
    assert form.points_last_n == 5 and form.goals_for_last_n == 4 and form.goals_against_last_n == 8
    assert form.clean_sheets_last_n == 1 and form.last_date == date(2026, 10, 11)
    assert math.isnan(form.xg_for_last_n) and math.isnan(form.xg_against_last_n), "R5 has no xG for Alpha: the window xG is NaN"
    assert form.goals_text == "4/8" and form.xg_text == MISSING_TEXT
    # a window without the missing value carries the sums
    three = team_form(hand_frame(), "Alpha", n=3, as_of=date(2026, 10, 4))  # R2 R3 R4 (date < as_of)
    assert three.results_last_n == "V-P-N" and three.points_last_n == 4 and three.goals_text == "2/4"
    assert three.xg_for_last_n == pytest.approx(1.2 + 0.7 + 0.8) and three.xg_against_last_n == pytest.approx(0.9 + 2.1 + 1.1)
    assert three.xg_text == "2.7/4.1" and three.clean_sheets_last_n == 1 and three.played == 5 and three.last_date == date(2026, 9, 27)


def test_team_form_as_of_and_season_filters() -> None:
    frame = hand_frame()
    before = team_form(frame, "Alpha", as_of=date(2026, 10, 4))  # OLD R1 R2 R3 R4
    assert before.played == 5 and before.results_last_n == "V-P-N-V-V" and before.points_last_n == 10
    assert before.goals_text == "8/4" and before.clean_sheets_last_n == 3
    assert before.xg_for_last_n == pytest.approx(7.2) and before.xg_against_last_n == pytest.approx(4.7)
    season = team_form(frame, "Alpha", as_of=date(2026, 10, 4), season="2026/27")  # R1 R2 R3 R4 only
    assert season.played == 4 == season.n_window and season.results_last_n == "V-P-N-V" and season.points_last_n == 7
    assert season.goals_text == "4/4" and season.xg_text == "4.2/4.5" and season.clean_sheets_last_n == 2
    assert team_form(frame, "Alpha", season="2026_27").played == 6  # underscore spelling accepted
    old = team_form(frame, "Alpha", season="2025/26")
    assert old.played == 1 and old.results_last_n == "V" and old.points_last_n == 3 and old.last_date == date(2026, 5, 10)
    # unknown team / nothing before as_of: the empty form, never an error
    nobody = team_form(frame, "Nobody")
    assert nobody.team == "Nobody" and nobody.played == 0 and nobody.results_last_n == "" and nobody.n_window == 0
    assert (nobody.points_last_n, nobody.goals_for_last_n, nobody.goals_against_last_n, nobody.clean_sheets_last_n) == (0, 0, 0, 0)
    assert TeamForm.empty("Nobody").played == 0 and math.isnan(TeamForm.empty("Nobody").xg_against_last_n)
    assert nobody.last_date is None and math.isnan(nobody.xg_for_last_n) and nobody.goals_text == "0/0"
    assert team_form(frame, "Alpha", as_of=date(2026, 1, 1)).played == 0
    with pytest.raises(ValueError):
        team_form(frame, "Alpha", n=0)


def test_form_table_and_team_matches_cover_every_team() -> None:
    frame = hand_frame()
    long = team_matches(frame)
    assert len(long) == 2 * len(frame) and list(long.columns)[:3] == ["team", "match_id", "date"]
    alpha = long[long["team"] == "Alpha"]
    assert alpha["date"].is_monotonic_increasing and alpha["points"].tolist() == [3, 3, 1, 0, 3, 1, 0]
    assert alpha["result"].tolist() == ["V", "V", "N", "P", "V", "N", "P"] and alpha["clean_sheet"].sum() == 3
    assert bool(alpha.iloc[1]["home"]) and not bool(alpha.iloc[2]["home"])
    table = form_table(frame, n=5)
    assert set(table.index) == {"Alpha", "Beta", "Gamma", "Delta"} and table.index.name == "team"
    assert table.loc["Alpha", "points"] == 5 and table.loc["Alpha", "results"] == "P-N-V-P-N" and table.loc["Alpha", "played"] == 7
    beta = table.loc["Beta"]
    assert beta["played"] == 4 and beta["n_window"] == 4 and beta["results"] == "V-P-P-P" and beta["points"] == 3
    assert beta["goals_for"] == 1 and beta["goals_against"] == 7
    assert beta["xg_for"] == pytest.approx(0.2 + 0.4 + 1.1 + 1.0) and beta["xg_against"] == pytest.approx(3.0 + 1.5 + 0.8 + 0.6)
    assert form_table(long=long, n=5).equals(table), "a pre-built team_matches frame gives the same table"
    assert form_table(frame.iloc[0:0], n=5).empty and team_matches(frame.iloc[0:0]).empty
    with pytest.raises(ValueError):
        form_table(n=5)


# ----------------------------------------------------------------------------- simulated league
@pytest.fixture(scope="module")
def league() -> dict:
    """Simulated 2026/27 league as of 2026-11-06 (11 rounds played, round 12 to play) plus one synthetic past season.

    One Girone A team of the next round gets the regression trap; the engine is
    fitted once; opportunities come from the value engine on the synthetic odds.
    """
    sim = simulate_league(seed=7, as_of=AS_OF)
    first_a = next(f for f in sim.fixtures if f.girone == Girone.A)
    trap_team = first_a.home_team
    current = inject_regression_trap(sim.matches, trap_team)
    history = simulate_history(seed=7, registry=sim.registry, seasons=("2025/26",))
    matches = validate_matches_df(pd.concat([history.matches, current], ignore_index=True))
    engine = DixonColesEngine(DixonColesConfig(xi=0.0))
    params = engine.fit(matches, registry=sim.registry)
    predictions = engine.predict_fixtures(sim.fixtures)
    value_engine = ValueEngine()
    traps = [t for t in detect_regression_traps(current) if params.is_current(t.team)]
    opportunities = value_engine.evaluate(predictions, sim.odds, matches=current, traps=traps)
    assert any(t.team == trap_team for t in traps)
    return {
        "sim": sim,
        "current": current,
        "history": history,
        "matches": matches,
        "engine": engine,
        "params": params,
        "predictions": predictions,
        "opportunities": opportunities,
        "traps": traps,
        "trap_team": trap_team,
        "value_engine": value_engine,
    }


def test_matchday_table_one_row_per_fixture_with_exact_columns(league: dict) -> None:
    sim = league["sim"]
    day = next_matchday(sim.fixtures)
    table = matchday_table(league["predictions"], params=league["params"], matches=league["current"], opportunities=league["opportunities"], traps=league["traps"], matchday=day)
    assert list(table.columns) == list(MATCHDAY_COLUMNS) == list(matchday_columns(DEFAULT_FORM_N, DEFAULT_XG_N))
    assert len(table) == day.n_fixtures == len(sim.fixtures) == 40 and set(table["match_id"]) == {f.match_id for f in sim.fixtures}
    assert table["Competizione"].tolist() == ["Girone A"] * 10 + ["Girone B"] * 10 + ["Girone C"] * 10 + ["Serie B"] * 10, "sorted by competition order"
    for _, group in table.groupby("Competizione", sort=False):
        assert group["Casa"].is_monotonic_increasing, "then by home team"
    assert table["Giornata"].eq(12).all() and str(table["Data"].dtype).startswith("datetime64") and table["Data"].dt.date.eq(sim.fixtures[0].date).all()
    # the model columns are the prediction's numbers, rounded
    by_id = {p.match_id: p for p in league["predictions"]}
    for _, row in table.head(5).iterrows():
        p = by_id[row["match_id"]]
        assert row["Casa"] == p.fixture.home_team and row["Ospiti"] == p.fixture.away_team
        assert row["λ"] == round(p.home_rate, 2) and row["μ"] == round(p.away_rate, 2) and row["Tot. att."] == round(p.home_rate + p.away_rate, 2)
        assert row["P1 (%)"] == round(p.probabilities[Market.MATCH_ODDS]["1"] * 100, 1) and row["GG (%)"] == round(p.probabilities[Market.BTTS]["GG"] * 100, 1)
        assert row["O2.5 (%)"] == round(p.probabilities[Market.OVER_UNDER_25]["Over 2.5"] * 100, 1)
        assert row["Fair X"] == round(p.fair_odds[Market.MATCH_ODDS]["X"], 2) and row["Fair GG"] == round(p.fair_odds[Market.BTTS]["GG"], 2)
        hs, as_, p_top = p.most_likely_score()
        assert row["Ris. prob."] == f"{hs}-{as_} ({p_top:.0%})"
        assert row["α Casa"] == round(league["params"].alpha[p.fixture.home_team], 2) and row["β Ospiti"] == round(league["params"].beta[p.fixture.away_team], 2)
        assert abs(row["P1 (%)"] + row["PX (%)"] + row["P2 (%)"] - 100.0) < 0.2
    # the form columns are team_form on the same matches
    row = table.iloc[0]
    home_form = team_form(league["current"], row["Casa"], n=5)
    assert row["Forma Casa"] == home_form.results_last_n and row["Pt Casa (5)"] == home_form.points_last_n and row["GF/GS Casa (5)"] == home_form.goals_text
    assert row["xG F/S Ospiti (3)"] == team_form(league["current"], row["Ospiti"], n=3).xg_text and "/" in row["xG F/S Ospiti (3)"]
    assert table["Forma Casa"].str.count("-").eq(4).all(), "11 rounds played: every form string has five results"
    assert table["Pt Casa (5)"].between(0, 15).all() and table["Pt Ospiti (5)"].between(0, 15).all()


def test_matchday_table_book_odds_ev_max_and_best_selection(league: dict) -> None:
    sim, preds = league["sim"], league["predictions"]
    # a second bookmaker 8% better on the draw only: the Book X column must pick it up
    better = [
        OddsQuote(q.match_id, "OtherBook", q.market, {sel: (round(o * 1.08, 2) if sel == "X" else o) for sel, o in q.odds.items()})
        for q in sim.odds
        if q.market is Market.MATCH_ODDS
    ]
    opps = league["value_engine"].evaluate(preds, [*sim.odds, *better], matches=league["current"], traps=league["traps"])
    table = matchday_table(preds, params=league["params"], matches=league["current"], opportunities=opps).set_index("match_id")
    frame = league["value_engine"].to_frame(opps)
    best_x = frame[frame["selection"] == "X"].groupby("match_id")["book_odds"].max()
    assert np.allclose(table.loc[best_x.index, "Book X"].to_numpy(), best_x.round(2).to_numpy())
    simbook_x = frame[(frame["selection"] == "X") & (frame["bookmaker"] != "OtherBook")].groupby("match_id")["book_odds"].max()
    assert (table.loc[simbook_x.index, "Book X"] > simbook_x).all(), "the better price of the second bookmaker wins"
    for suffix, selection in (("1", "1"), ("2", "2"), ("O2.5", "Over 2.5"), ("GG", "GG")):
        best = frame[frame["selection"] == selection].groupby("match_id")["book_odds"].max()
        assert np.allclose(table.loc[best.index, f"Book {suffix}"].to_numpy(), best.round(2).to_numpy())
    # EV max is the best EV of the fixture over every market and bookmaker, as ValueEngine.to_frame reports it
    ev_max = frame.groupby("match_id")["ev"].max()
    assert np.allclose(table.loc[ev_max.index, "EV max (%)"].to_numpy(), (ev_max * 100).round(2).to_numpy())
    idx = frame.groupby("match_id")["ev"].idxmax()
    for match_id, i in idx.items():
        best_row = frame.loc[i]
        assert table.loc[match_id, "Miglior EV"] == f"{best_row['selection']} @{best_row['book_odds']:.2f}"
    # an unquoted fixture: NaN prices, NaN EV, empty best selection
    quoted = {o.match_id for o in opps}
    some = next(p for p in preds if p.match_id in quoted)
    partial = [o for o in opps if o.match_id != some.match_id]
    bare = matchday_table(preds, params=league["params"], matches=league["current"], opportunities=partial).set_index("match_id").loc[some.match_id]
    assert all(math.isnan(bare[f"Book {s}"]) for s in ("1", "X", "2", "O2.5", "GG")) and math.isnan(bare["EV max (%)"]) and bare["Miglior EV"] == ""
    # without any opportunity the model columns are still filled
    none = matchday_table(preds, params=league["params"], matches=league["current"])
    assert len(none) == 40 and none["EV max (%)"].isna().all() and none["Book 1"].isna().all() and none["P1 (%)"].notna().all() and none["Flag"].eq("").any()


def test_matchday_table_flags_union_includes_the_trapped_team(league: dict) -> None:
    trap_team = league["trap_team"]
    table = matchday_table(league["predictions"], params=league["params"], matches=league["current"], opportunities=league["opportunities"], traps=league["traps"])
    row = table[(table["Casa"] == trap_team) | (table["Ospiti"] == trap_team)].iloc[0]
    flags = row["Flag"].split(" | ")
    assert FLAG_REGRESSION_TRAP in flags and f"{TRAP_FLAG_PREFIX}{trap_team}" in flags
    # the union: every flag of every quoted selection of the fixture, each once, VALUE first when present
    expected = {f for o in league["opportunities"] if o.match_id == row["match_id"] for f in o.flags}
    assert set(flags) == expected and len(flags) == len(set(flags))
    if "VALUE" in expected:
        assert flags[0] == "VALUE"
    assert flags.index(FLAG_REGRESSION_TRAP) < flags.index(f"{TRAP_FLAG_PREFIX}{trap_team}")
    # the trap is reported from ``traps`` even when the fixture has no quote at all
    unquoted = [o for o in league["opportunities"] if o.match_id != row["match_id"]]
    again = matchday_table(league["predictions"], params=league["params"], matches=league["current"], opportunities=unquoted, traps=league["traps"]).set_index("match_id").loc[row["match_id"]]
    assert again["Flag"] == f"{FLAG_REGRESSION_TRAP} | {TRAP_FLAG_PREFIX}{trap_team}" and math.isnan(again["EV max (%)"])
    # ... and without traps nor quotes there is no flag
    plain = matchday_table(league["predictions"], params=league["params"], matches=league["current"]).set_index("match_id").loc[row["match_id"]]
    assert plain["Flag"] == ""


def test_matchday_table_girone_and_matchday_filters_and_empty(league: dict) -> None:
    kwargs = dict(params=league["params"], matches=league["current"], opportunities=league["opportunities"], traps=league["traps"])
    sb = matchday_table(league["predictions"], girone=Girone.SB, **kwargs)
    assert len(sb) == 10 and sb["Competizione"].eq("Serie B").all()
    assert len(matchday_table(league["predictions"], girone="Girone A", **kwargs)) == 10
    day = next_matchday(league["sim"].fixtures)
    a_fixtures = [f for f in league["sim"].fixtures if f.girone == Girone.A]
    other = Fixture("FAR-1", date(2026, 12, 20), 18, Girone.A, a_fixtures[0].home_team, a_fixtures[1].away_team)
    far = league["engine"].predict(other)
    assert len(matchday_table([*league["predictions"], far], matchday=day, **kwargs)) == 40, "fixtures outside the matchday are left out"
    assert len(matchday_table([*league["predictions"], far], **kwargs)) == 41
    empty = matchday_table([], **kwargs)
    assert empty.empty and list(empty.columns) == list(MATCHDAY_COLUMNS) and str(empty["Data"].dtype).startswith("datetime64")
    assert list(matchday_columns(3, 2))[11:17] == ["Pt Casa (3)", "Pt Ospiti (3)", "GF/GS Casa (3)", "GF/GS Ospiti (3)", "xG F/S Casa (2)", "xG F/S Ospiti (2)"]
    custom = matchday_table(league["predictions"], form_n=3, xg_n=2, **kwargs)
    assert list(custom.columns) == list(matchday_columns(3, 2)) and custom["Forma Casa"].str.count("-").eq(2).all()
    with pytest.raises(ValueError):
        matchday_table(league["predictions"], form_n=0, **kwargs)
    # a pre-built per-team frame gives the same table (the dashboard caches it on the state)
    long = team_matches(league["current"])
    assert matchday_table(league["predictions"], long=long, **kwargs).equals(matchday_table(league["predictions"], **kwargs))


def test_results_table_rows_outcome_probabilities_and_tags(league: dict) -> None:
    engine, matches, current = league["engine"], league["matches"], league["current"]
    table = results_table(matches, engine=engine, season="2026/27")
    assert list(table.columns) == list(RESULTS_MODEL_COLUMNS) and len(table) == len(current) == 440
    assert table["Data"].is_monotonic_decreasing, "most recent first"
    first_day = table[table["Data"] == table["Data"].iloc[0]]
    assert first_day["Competizione"].tolist() == ["Girone A"] * 10 + ["Girone B"] * 10 + ["Girone C"] * 10 + ["Serie B"] * 10
    assert len(results_table(matches, engine=engine, season="2026/27", girone=Girone.A)) == 110
    # Esito and Risultato agree with the goals; the statistics read 'home/away'
    source = current.set_index("match_id")
    for _, row in table.head(20).iterrows():
        m = source.loc[row["match_id"]]
        hg, ag = int(m["home_goals"]), int(m["away_goals"])
        assert row["Risultato"] == f"{hg}-{ag}" and row["Esito"] == ("1" if hg > ag else "X" if hg == ag else "2")
        assert row["Tiri C/O"] == f"{m['home_shots']:.0f}/{m['away_shots']:.0f}" and row["Angoli C/O"] == f"{m['home_corners']:.0f}/{m['away_corners']:.0f}"
        assert row["xG Casa"] == round(float(m["home_xg"]), 2) and row["xG Ospiti"] == round(float(m["away_xg"]), 2)
    assert table["P esito (%)"].between(0.0, 100.0).all() and table["λ"].gt(0).all() and table["μ"].gt(0).all()
    # P esito equals the outcome probability of the engine's own score matrix
    for _, row in table.head(6).iterrows():
        m = source.loc[row["match_id"]]
        lam, mu = engine.rates(m["home_team"], m["away_team"], m["girone"])
        probs = markets_from_matrix(engine.score_matrix(m["home_team"], m["away_team"], m["girone"]))[Market.MATCH_ODDS]
        vector = outcome_probabilities([lam], [mu], engine.params_.rho, engine.params_.max_goals)[0]
        assert vector == pytest.approx([probs["1"], probs["X"], probs["2"]], abs=1e-9)
        assert row["λ"] == round(lam, 2) and row["μ"] == round(mu, 2)
        assert row["P esito (%)"] == round(probs[row["Esito"]] * 100.0, 1)
    # the tags follow the thresholds applied to the displayed (rounded) percentage, never contradicting it
    p = table["P esito (%)"]
    assert (table.loc[p < 20.0, "Sorpresa"] == SURPRISE_TAG).all() and (table.loc[p > 50.0, "Sorpresa"] == EXPECTED_TAG).all()
    assert (table.loc[(p >= 20.0) & (p <= 50.0), "Sorpresa"] == "").all()
    assert (table["Sorpresa"] == SURPRISE_TAG).any() and (table["Sorpresa"] == EXPECTED_TAG).any()
    assert table["Sorpresa"].tolist() == [surprise_tag(v / 100.0) for v in p]
    # without an engine: the statistics only
    bare = results_table(current)
    assert list(bare.columns) == list(RESULTS_COLUMNS) and len(bare) == 440 and "λ" not in bare.columns
    empty = results_table(current, engine=engine, season="2019/20")
    assert empty.empty and list(empty.columns) == list(RESULTS_MODEL_COLUMNS)


def test_surprise_tag_follows_the_displayed_percentage() -> None:
    """Regression: a probability that *displays* as 20.0% (or 50.0%) is not tagged, whatever its hidden decimals.

    ``results_table`` shows ``P esito (%)`` rounded to one decimal; tagging the
    raw probability made a row read ``20.0`` next to ``sorpresa``, against the
    legend ("sorpresa sotto il 20%"). The tag is now judged on the shown value.
    """
    assert P_OUTCOME_DECIMALS == 1 and SURPRISE_BELOW == 0.20 and EXPECTED_ABOVE == 0.50
    # the boundary: 0.1996 displays as 20.0 -> blank; 0.1994 displays as 19.9 -> sorpresa
    assert float(p_outcome_percent(0.1996)) == 20.0 and surprise_tag(0.1996) == ""
    assert float(p_outcome_percent(0.1994)) == 19.9 and surprise_tag(0.1994) == SURPRISE_TAG
    assert surprise_tag(0.2) == "" and surprise_tag(0.19999) == ""  # still 20.0 on screen
    assert surprise_tag(0.0) == SURPRISE_TAG and surprise_tag(0.1) == SURPRISE_TAG
    # the upper threshold: 50.0 shown -> blank, 50.1 shown -> atteso
    assert float(p_outcome_percent(0.5004)) == 50.0 and surprise_tag(0.5004) == ""
    assert float(p_outcome_percent(0.5006)) == 50.1 and surprise_tag(0.5006) == EXPECTED_TAG
    assert surprise_tag(0.5) == "" and surprise_tag(0.51) == EXPECTED_TAG and surprise_tag(1.0) == EXPECTED_TAG
    assert surprise_tag(0.35) == "" and surprise_tag(float("nan")) == ""
    # every tag agrees with the threshold applied to the displayed value, on a dense grid around both thresholds
    grid = np.concatenate([np.linspace(0.19, 0.21, 2001), np.linspace(0.49, 0.51, 2001)])
    shown = p_outcome_percent(grid)
    for value, pct in zip(grid, shown):
        tag = surprise_tag(float(value))
        expected = SURPRISE_TAG if pct < 20.0 else EXPECTED_TAG if pct > 50.0 else ""
        assert tag == expected, (value, pct, tag)
    assert p_outcome_percent([0.123456, 0.5]).tolist() == [12.3, 50.0]


def test_fixture_quotes_defaults_are_per_instance() -> None:
    """Regression: ``_FixtureQuotes`` uses ``default_factory`` containers, not a ``None`` placeholder fixed in ``__post_init__``."""
    fields = {f.name: f for f in dataclasses.fields(_FixtureQuotes)}
    assert fields["flags"].default_factory is set and fields["best_price"].default_factory is dict  # type: ignore[comparison-overlap]
    assert fields["flags"].default is dataclasses.MISSING and fields["best_price"].default is dataclasses.MISSING
    assert "__post_init__" not in vars(_FixtureQuotes)
    a, b = _FixtureQuotes(), _FixtureQuotes()
    assert a.flags == set() and a.best_price == {} and a.best is None
    assert a.flags is not b.flags and a.best_price is not b.best_price
    a.flags.add("VALUE")
    a.best_price[(Market.MATCH_ODDS, "X")] = 3.4
    assert b.flags == set() and b.best_price == {}
    assert _quotes_by_fixture([]) == {}


def test_result_letters_wording_is_shared_with_the_legend() -> None:
    """One glossary for the results string: V vittoria, N pareggio (nullo), P sconfitta, in the engine and in the legend."""
    from src.dashboard.legend import legend_terms

    assert (RESULT_WIN, RESULT_DRAW, RESULT_LOSS) == ("V", "N", "P")
    wording = "V vittoria, N pareggio (nullo), P sconfitta"
    assert wording in legend_terms()["Forma (Casa / Ospiti)"]
    import src.engine.tables as tables_module

    assert "``N`` pareggio (nullo)" in str(tables_module.__doc__) and "``P`` sconfitta" in str(tables_module.__doc__)
    assert "perdita" not in str(tables_module.__doc__) and "perdita" not in str(TeamForm.__doc__)


def test_results_table_team_filter_includes_historical_teams(league: dict) -> None:
    engine, matches, params = league["engine"], league["matches"], league["params"]
    historical = params.historical_teams[0]
    assert historical not in league["sim"].registry.canonical_names
    rows = results_table(matches, engine=engine, season="2025/26", team=historical)
    hist = league["history"].matches
    expected = int(((hist["home_team"] == historical) | (hist["away_team"] == historical)).sum())
    assert len(rows) == expected > 0 and ((rows["Casa"] == historical) | (rows["Ospiti"] == historical)).all()
    assert rows["P esito (%)"].notna().all(), "historical teams are fitted, so the model columns are filled"
    assert results_table(matches, engine=engine, season="2026/27", team=historical).empty
    current_team = league["trap_team"]  # a Girone A team of the next round
    mine = results_table(matches, engine=engine, season="2026/27", team=current_team, girone=Girone.A)
    assert len(mine) == 11 and ((mine["Casa"] == current_team) | (mine["Ospiti"] == current_team)).all()
    assert len(results_table(matches, season="2025/26")) == len(league["history"].matches) == 1520
    # a missing statistic reads '-'
    patched = league["current"].copy()
    patched.loc[0, ["away_shots", "home_corners", "away_corners"]] = float("nan")
    row = results_table(patched).set_index("match_id").loc[patched.loc[0, "match_id"]]
    assert row["Tiri C/O"].endswith(f"/{MISSING_TEXT}") and row["Angoli C/O"] == MISSING_TEXT


def test_outcome_probabilities_vectorised() -> None:
    probs = outcome_probabilities([1.3, 0.9, float("nan")], [1.1, 1.4, 1.0], -0.1)
    assert probs.shape == (3, 3) and np.allclose(probs[:2].sum(axis=1), 1.0) and np.isnan(probs[2]).all()
    direct = markets_from_matrix(dixon_coles_pmf(1.3, 1.1, -0.1))[Market.MATCH_ODDS]
    assert probs[0] == pytest.approx([direct["1"], direct["X"], direct["2"]], abs=1e-12)
    with pytest.raises(ValueError):
        outcome_probabilities([1.0], [1.0, 2.0], 0.0)
    with pytest.raises(ValueError):
        outcome_probabilities([1.0], [1.0], 0.0, max_goals=0)


def test_tables_are_fast_on_a_full_league(league: dict) -> None:
    """~3,500 played matches and a season-long fixture list of 160 fixtures: both tables in well under 3 s."""
    sim = league["sim"]
    history = simulate_history(seed=7, registry=sim.registry)  # two seasons, 3040 matches
    matches = validate_matches_df(pd.concat([history.matches, league["current"]], ignore_index=True))
    assert len(matches) == 3040 + 440
    engine = DixonColesEngine(DixonColesConfig(xi=0.0))
    params = engine.fit(matches, registry=sim.registry)
    fixtures = [f for k in range(11, 15) for f in simulate_league(seed=7, n_rounds=k).fixtures]
    assert len(fixtures) == 160 and len({f.match_id for f in fixtures}) == 160
    predictions = engine.predict_fixtures(fixtures)
    quotes = [q for k in range(11, 15) for q in simulate_league(seed=7, n_rounds=k).odds]
    opportunities = league["value_engine"].evaluate(predictions, quotes, matches=league["current"], traps=league["traps"])
    started = time.perf_counter()
    table = matchday_table(predictions, params=params, matches=matches, opportunities=opportunities, traps=league["traps"])
    elapsed = time.perf_counter() - started
    assert len(table) == 160 and table["EV max (%)"].notna().all()
    assert elapsed < 3.0, f"matchday_table took {elapsed:.2f}s"
    started = time.perf_counter()
    results = results_table(matches, engine=engine)
    elapsed = time.perf_counter() - started
    assert len(results) == 3480 and results["P esito (%)"].notna().all()
    assert elapsed < 3.0, f"results_table took {elapsed:.2f}s"


# ----------------------------------------------------------------------------- competition restriction (dashboard sections)
def test_tables_restrict_to_a_set_of_competitions(league: dict) -> None:
    """``gironi`` keeps the fixtures / matches of a set of competitions (the dashboard section); ``None`` keeps every one."""
    kwargs = dict(params=league["params"], matches=league["current"], opportunities=league["opportunities"], traps=league["traps"])
    a_and_sb = matchday_table(league["predictions"], gironi=(Girone.A, "SB"), **kwargs)
    assert len(a_and_sb) == 20 and set(a_and_sb["Competizione"]) == {"Girone A", "Serie B"}
    assert list(a_and_sb.columns) == list(MATCHDAY_COLUMNS)
    assert matchday_table(league["predictions"], gironi=None, **kwargs).equals(matchday_table(league["predictions"], **kwargs))
    only_c = matchday_table(league["predictions"], gironi=["A", "B", "C"], girone=Girone.C, **kwargs)
    assert len(only_c) == 10 and only_c["Competizione"].eq("Girone C").all(), "girone and gironi both apply"
    assert matchday_table(league["predictions"], gironi=["SB"], girone=Girone.A, **kwargs).empty, "a competition outside the set has no rows"
    empty = matchday_table(league["predictions"], gironi=[], **kwargs)
    assert empty.empty and list(empty.columns) == list(MATCHDAY_COLUMNS)
    # results: the same restriction on the played matches
    results = results_table(league["matches"], engine=league["engine"], season="2026/27", gironi=("B", "SB"))
    assert len(results) == len(league["current"][league["current"]["girone"].isin(["B", "SB"])]) and set(results["Competizione"]) == {"Girone B", "Serie B"}
    assert results_table(league["matches"], engine=league["engine"], season="2026/27", gironi=None).equals(results_table(league["matches"], engine=league["engine"], season="2026/27"))
    team = league["sim"].registry.teams(Girone.SB)[0]
    mine = results_table(league["matches"], season="2026/27", team=team, gironi=["SB"])
    assert len(mine) == 11 and ((mine["Casa"] == team) | (mine["Ospiti"] == team)).all()
    assert results_table(league["matches"], season="2026/27", team=team, gironi=["A"]).empty
    assert list(results_table(league["matches"], gironi=[]).columns) == list(RESULTS_COLUMNS)
