"""Tests for src.engine.value: de-vigging, EV, fractional Kelly, REGRESSION_TRAP, ValueEngine.

Predictions are built from the simulated league's TRUE parameters so the model
side is exact and the only noise is the one injected into the bookmaker odds.
"""
from __future__ import annotations

import logging
import math
from datetime import date, timedelta
from typing import ClassVar

import numpy as np
import pandas as pd
import pytest

from src.common.schemas import SELECTIONS, Fixture, Girone, Market, MatchPrediction, OddsQuote
from src.engine.value import (
    DEVIG_METHODS,
    FLAG_LONGSHOT,
    FLAG_REGRESSION_TRAP,
    FLAG_SUSPICIOUS_EDGE,
    FLAG_VALUE,
    KELLY_EIGHTH,
    KELLY_QUARTER,
    TRAPS_FRAME_COLUMNS,
    VALUE_FRAME_COLUMNS,
    KellyConfig,
    RegressionTrap,
    ValueEngine,
    ValueOpportunity,
    detect_regression_traps,
    devig,
    expected_value,
    implied_probabilities,
    kelly_fraction,
    overround,
    stake_units,
)
from src.simulation.generator import (
    SimulatedLeague,
    dixon_coles_matrix,
    inject_regression_trap,
    market_probabilities,
    simulate_league,
    simulate_mini_dataset,
)

TRAP_TEAM = "Vicenza"


# ----------------------------------------------------------------------------- helpers
def _prediction(fixture: Fixture, lam: float, mu: float, rho: float) -> MatchPrediction:
    matrix = dixon_coles_matrix(lam, mu, rho)
    probs = market_probabilities(matrix)
    fair = {m: {s: (1.0 / p if p > 0.0 else math.inf) for s, p in d.items()} for m, d in probs.items()}
    return MatchPrediction(
        fixture=fixture,
        home_rate=lam,
        away_rate=mu,
        rho=rho,
        matrix=matrix,
        probabilities=probs,
        fair_odds=fair,
    )


def _predictions_from_truth(league: SimulatedLeague) -> list[MatchPrediction]:
    out: list[MatchPrediction] = []
    for fx in league.fixtures:
        lam, mu = league.true_params.rates(fx.home_team, fx.away_team, fx.girone)
        out.append(_prediction(fx, lam, mu, league.true_params.rho))
    return out


def _team_window_ids(df: pd.DataFrame, team: str, window: int = 3) -> list[str]:
    mask = (df["home_team"] == team) | (df["away_team"] == team)
    return df.loc[mask].sort_values(["date", "match_id"])["match_id"].tail(window).astype(str).tolist()


# ----------------------------------------------------------------------------- fixtures
@pytest.fixture(scope="module")
def league() -> SimulatedLeague:
    return simulate_league(seed=11, n_rounds=15)


@pytest.fixture(scope="module")
def trapped_matches(league: SimulatedLeague) -> pd.DataFrame:
    return inject_regression_trap(league.matches, TRAP_TEAM)


@pytest.fixture(scope="module")
def mini() -> SimulatedLeague:
    return simulate_mini_dataset()


@pytest.fixture(scope="module")
def mini_predictions(mini: SimulatedLeague) -> list[MatchPrediction]:
    return _predictions_from_truth(mini)


@pytest.fixture(scope="module")
def engine() -> ValueEngine:
    return ValueEngine()


@pytest.fixture(scope="module")
def opportunities(engine: ValueEngine, mini_predictions: list[MatchPrediction], mini: SimulatedLeague) -> list[ValueOpportunity]:
    return engine.evaluate(mini_predictions, mini.odds)


# ============================================================================= implied / de-vig
class TestImpliedAndDevig:
    ODDS: ClassVar[dict[str, float]] = {"1": 2.0, "X": 3.5, "2": 3.6}

    def test_overround_about_six_percent(self) -> None:
        assert overround(self.ODDS) == pytest.approx(0.0635, abs=1e-3)
        assert overround(self.ODDS) == pytest.approx(1 / 2.0 + 1 / 3.5 + 1 / 3.6 - 1.0)

    def test_implied_probabilities_are_raw_reciprocals(self) -> None:
        raw = implied_probabilities(self.ODDS)
        assert list(raw) == ["1", "X", "2"]
        assert raw["1"] == pytest.approx(0.5)
        assert sum(raw.values()) == pytest.approx(1.0 + overround(self.ODDS))

    def test_proportional_sums_to_one_and_preserves_ratios(self) -> None:
        p = devig(self.ODDS)  # default method
        assert p == devig(self.ODDS, "proportional")
        assert sum(p.values()) == pytest.approx(1.0, abs=1e-9)
        assert p["1"] / p["X"] == pytest.approx((1 / 2.0) / (1 / 3.5))
        assert p["X"] / p["2"] == pytest.approx((1 / 3.5) / (1 / 3.6))
        assert list(p) == ["1", "X", "2"]

    @pytest.mark.parametrize("method", ["power", "shin"])
    def test_power_and_shin_sum_to_one(self, method: str) -> None:
        p = devig(self.ODDS, method)
        assert sum(p.values()) == pytest.approx(1.0, abs=1e-9)
        assert all(0.0 < v < 1.0 for v in p.values())
        assert sorted(p, key=p.get, reverse=True) == ["1", "X", "2"]

    def test_all_methods_agree_on_a_fair_book(self) -> None:
        fair = {"1": 1 / 0.5, "X": 1 / 0.3, "2": 1 / 0.2}
        assert overround(fair) == pytest.approx(0.0, abs=1e-12)
        prop = devig(fair, "proportional")
        for method in ("power", "shin"):
            other = devig(fair, method)
            for sel in fair:
                assert other[sel] == pytest.approx(prop[sel], abs=1e-9)
            assert other["1"] == pytest.approx(0.5, abs=1e-9)

    def test_shin_gives_longshots_less_than_proportional(self) -> None:
        book = {"1": 1.3, "X": 5.5, "2": 9.0}
        prop = devig(book, "proportional")
        shin = devig(book, "shin")
        power = devig(book, "power")
        assert shin["2"] < prop["2"]
        assert shin["X"] < prop["X"]
        assert shin["1"] > prop["1"]
        assert power["2"] < prop["2"]
        assert power["1"] > prop["1"]

    @pytest.mark.parametrize("method", DEVIG_METHODS)
    @pytest.mark.parametrize(
        "bad",
        [
            {},
            {"1": 2.0},
            {"1": 1.0, "2": 2.0},
            {"1": 0.8, "2": 2.0},
            {"1": -3.0, "2": 2.0},
            {"1": math.inf, "2": 2.0},
            {"1": math.nan, "2": 2.0},
        ],
    )
    def test_invalid_odds_raise(self, method: str, bad: dict[str, float]) -> None:
        with pytest.raises(ValueError):
            devig(bad, method)

    def test_invalid_odds_raise_in_implied_and_overround(self) -> None:
        with pytest.raises(ValueError):
            implied_probabilities({"1": 1.0, "2": 2.0})
        with pytest.raises(ValueError):
            overround({})
        with pytest.raises(ValueError):
            devig(self.ODDS, "multiplicative")  # type: ignore[arg-type]

    def test_two_way_markets_work(self) -> None:
        for method in DEVIG_METHODS:
            p = devig({"Over 2.5": 1.9, "Under 2.5": 1.9}, method)
            assert p["Over 2.5"] == pytest.approx(0.5, abs=1e-9)
            assert p["Under 2.5"] == pytest.approx(0.5, abs=1e-9)


# ============================================================================= EV / Kelly
class TestExpectedValueAndKelly:
    def test_expected_value_formula(self) -> None:
        assert expected_value(0.5, 2.2) == pytest.approx(0.10)
        assert expected_value(0.4, 2.2) < 0.0
        assert expected_value(0.0, 5.0) == pytest.approx(-1.0)

    def test_expected_value_validation(self) -> None:
        with pytest.raises(ValueError):
            expected_value(1.2, 2.0)
        with pytest.raises(ValueError):
            expected_value(0.5, 1.0)

    def test_kelly_fraction(self) -> None:
        assert kelly_fraction(0.55, 2.0) == pytest.approx(0.10)
        assert kelly_fraction(0.4, 2.0) == 0.0
        assert kelly_fraction(0.5, 2.0) == 0.0
        # b = 2, p = 0.4 -> (0.8 - 0.6) / 2 = 0.1
        assert kelly_fraction(0.4, 3.0) == pytest.approx(0.10)
        with pytest.raises(ValueError):
            kelly_fraction(-0.1, 2.0)
        with pytest.raises(ValueError):
            kelly_fraction(0.5, 0.99)

    def test_kelly_constants_and_config_validation(self) -> None:
        assert KELLY_EIGHTH == 0.125
        assert KELLY_QUARTER == 0.25
        assert KELLY_QUARTER == 2 * KELLY_EIGHTH
        assert KellyConfig().fraction == KELLY_EIGHTH
        for bad in (0.0, -0.1, 1.5, math.nan):
            with pytest.raises(ValueError):
                KellyConfig(fraction=bad)
        assert KellyConfig(fraction=1.0).fraction == 1.0
        with pytest.raises(ValueError):
            KellyConfig(min_stake_units=5.0, max_stake_units=3.0)
        with pytest.raises(ValueError):
            KellyConfig(round_to=0.0)
        with pytest.raises(ValueError):
            KellyConfig(bankroll_units=0.0)

    def test_stake_rounds_half_up_to_tenth(self) -> None:
        # kelly 0.10 * 0.125 * 100 = 1.25 -> documented rule: round half up -> 1.3
        assert stake_units(0.55, 2.0, KellyConfig(fraction=0.125)) == 1.3
        assert stake_units(0.55, 2.0) == 1.3  # default config is 1/8 Kelly
        # 1.25 with a 0.5 step -> 1.5 ; with a 0.25 step -> 1.25 exactly
        assert stake_units(0.55, 2.0, KellyConfig(round_to=0.5)) == 1.5
        assert stake_units(0.55, 2.0, KellyConfig(round_to=0.25)) == 1.25

    def test_stake_cap(self) -> None:
        # kelly(0.9, 2.0) = 0.8 -> 0.8 * 0.125 * 100 = 10 units -> capped at 3.0
        assert stake_units(0.9, 2.0) == 3.0
        assert stake_units(0.9, 2.0, KellyConfig(max_stake_units=1.5)) == 1.5
        # cap not a multiple of the step: floored to the step, never above the cap
        assert stake_units(0.9, 2.0, KellyConfig(max_stake_units=2.95, round_to=0.1)) == 2.9

    def test_stake_below_minimum_is_zero(self) -> None:
        # kelly(0.5005, 2.0) = 0.001 -> 0.0125 units -> rounds to 0.0 -> below min
        assert stake_units(0.5005, 2.0) == 0.0
        # 1.25 -> 1.3 but min 2.0 -> 0.0
        assert stake_units(0.55, 2.0, KellyConfig(min_stake_units=2.0)) == 0.0
        # no edge -> always 0
        assert stake_units(0.4, 2.0) == 0.0

    def test_quarter_kelly_doubles_eighth_kelly(self) -> None:
        eighth = KellyConfig(fraction=KELLY_EIGHTH, max_stake_units=10.0)
        quarter = KellyConfig(fraction=KELLY_QUARTER, max_stake_units=10.0)
        # kelly(0.6, 2.0) = 0.2 -> 2.5 units vs 5.0 units
        assert stake_units(0.6, 2.0, eighth) == 2.5
        assert stake_units(0.6, 2.0, quarter) == 5.0
        assert stake_units(0.6, 2.0, quarter) == 2 * stake_units(0.6, 2.0, eighth)


# ============================================================================= regression trap
class TestRegressionTraps:
    def test_injected_team_is_detected(self, trapped_matches: pd.DataFrame) -> None:
        traps = detect_regression_traps(trapped_matches)
        by_team = {t.team: t for t in traps}
        assert TRAP_TEAM in by_team
        trap = by_team[TRAP_TEAM]
        assert trap.girone is Girone.A
        assert trap.window == 3
        assert trap.goals_against <= 1
        assert trap.xg_against >= 3.5
        assert trap.xg_against == pytest.approx(4.2)
        assert trap.severity == pytest.approx(trap.xg_against - trap.goals_against)
        assert trap.xga_per_match == pytest.approx(trap.xg_against / 3)
        assert list(trap.match_ids) == _team_window_ids(trapped_matches, TRAP_TEAM)
        expected_last = trapped_matches.loc[trapped_matches["match_id"].isin(trap.match_ids), "date"].max()
        assert trap.last_date == expected_last.date()
        text = trap.description()
        assert "\n" not in text
        assert TRAP_TEAM in text and "xGA" in text

    def test_input_frame_is_not_mutated(self, trapped_matches: pd.DataFrame) -> None:
        before = trapped_matches.copy(deep=True)
        detect_regression_traps(trapped_matches)
        pd.testing.assert_frame_equal(trapped_matches, before)

    def test_nan_opponent_xg_in_window_blocks_the_flag(self, trapped_matches: pd.DataFrame) -> None:
        df = trapped_matches.copy()
        window_ids = _team_window_ids(df, TRAP_TEAM)
        row = df.index[df["match_id"] == window_ids[1]][0]
        column = "away_xg" if df.at[row, "home_team"] == TRAP_TEAM else "home_xg"
        df.loc[row, column] = np.nan
        teams = {t.team for t in detect_regression_traps(df)}
        assert TRAP_TEAM not in teams
        # sanity: the team is still detected on the untouched frame
        assert TRAP_TEAM in {t.team for t in detect_regression_traps(trapped_matches)}

    def test_as_of_before_injected_matches_excludes_team(self, trapped_matches: pd.DataFrame) -> None:
        window_ids = _team_window_ids(trapped_matches, TRAP_TEAM)
        first_injected = trapped_matches.loc[trapped_matches["match_id"].isin(window_ids), "date"].min()
        as_of = (first_injected - timedelta(days=1)).date()
        traps = detect_regression_traps(trapped_matches, as_of=as_of)
        assert TRAP_TEAM not in {t.team for t in traps}
        assert all(t.last_date <= as_of for t in traps)
        # as_of on the last injected day keeps it
        last_injected = trapped_matches.loc[trapped_matches["match_id"].isin(window_ids), "date"].max().date()
        assert TRAP_TEAM in {t.team for t in detect_regression_traps(trapped_matches, as_of=last_injected)}

    def test_thresholds_are_respected(self, trapped_matches: pd.DataFrame) -> None:
        assert detect_regression_traps(trapped_matches, min_xg_against=10.0) == []
        # the injected team conceded exactly 1 goal: a zero tolerance excludes it
        strict = detect_regression_traps(trapped_matches, max_goals_against=0)
        assert TRAP_TEAM not in {t.team for t in strict}
        assert all(t.goals_against == 0 for t in strict)
        loose = detect_regression_traps(trapped_matches, min_xg_against=0.0, max_goals_against=1)
        assert {t.team for t in detect_regression_traps(trapped_matches)} <= {t.team for t in loose}

    def test_sorted_by_severity_desc(self, trapped_matches: pd.DataFrame) -> None:
        df = inject_regression_trap(trapped_matches, "Arezzo", xga_total=6.0)
        traps = detect_regression_traps(df)
        severities = [t.severity for t in traps]
        assert severities == sorted(severities, reverse=True)
        assert traps[0].team == "Arezzo"
        assert traps[0].severity == pytest.approx(6.0 - 1)
        assert TRAP_TEAM in {t.team for t in traps}

    def test_window_parameter(self, trapped_matches: pd.DataFrame) -> None:
        traps = detect_regression_traps(trapped_matches, window=4, min_xg_against=0.0, max_goals_against=50)
        assert traps  # every team has >= 4 matches with xG
        assert all(t.window == 4 and len(t.match_ids) == 4 for t in traps)
        with pytest.raises(ValueError):
            detect_regression_traps(trapped_matches, window=0)

    def test_short_history_and_empty_frame(self, trapped_matches: pd.DataFrame) -> None:
        # a team with fewer than `window` matches is never flagged
        one_round = trapped_matches.loc[trapped_matches["round"] == 1]
        assert detect_regression_traps(one_round, min_xg_against=0.0, max_goals_against=50) == []
        assert detect_regression_traps(trapped_matches.iloc[0:0]) == []

    def test_traps_frame(self, trapped_matches: pd.DataFrame) -> None:
        traps = detect_regression_traps(trapped_matches)
        frame = ValueEngine.traps_frame(traps)
        assert list(frame.columns) == list(TRAPS_FRAME_COLUMNS)
        assert len(frame) == len(traps)
        row = frame.loc[frame["team"] == TRAP_TEAM].iloc[0]
        assert row["match_ids"] == "|".join(_team_window_ids(trapped_matches, TRAP_TEAM))
        assert row["girone"] == "A"
        empty = ValueEngine.traps_frame([])
        assert list(empty.columns) == list(TRAPS_FRAME_COLUMNS) and empty.empty

    def test_regression_trap_dataclass_consistency(self) -> None:
        trap = RegressionTrap.from_window("Lecco", "A", 3, 1, 4.5, ("m1", "m2", "m3"), "2025-10-01")
        assert trap.severity == pytest.approx(3.5)
        assert trap.xga_per_match == pytest.approx(1.5)
        assert trap.last_date == date(2025, 10, 1)
        assert trap.to_dict()["match_ids"] == "m1|m2|m3"
        with pytest.raises(ValueError):
            RegressionTrap("Lecco", Girone.A, 3, 1, 4.5, ("m1",), date(2025, 10, 1), 1.5, 9.9)
        with pytest.raises(ValueError):
            RegressionTrap("Lecco", Girone.A, 0, 1, 4.5, (), date(2025, 10, 1), 1.5, 3.5)


# ============================================================================= ValueEngine
class TestValueEngine:
    def test_every_quote_selection_yields_an_opportunity(
        self, opportunities: list[ValueOpportunity], mini: SimulatedLeague
    ) -> None:
        assert len(opportunities) == sum(len(q.odds) for q in mini.odds)
        expected = {(q.match_id, q.market, s) for q in mini.odds for s in SELECTIONS[q.market]}
        got = {(o.match_id, o.market, o.selection) for o in opportunities}
        assert got == expected

    def test_numbers_and_flags_are_consistent(
        self, opportunities: list[ValueOpportunity], engine: ValueEngine, mini_predictions: list[MatchPrediction]
    ) -> None:
        preds = {p.match_id: p for p in mini_predictions}
        for o in opportunities:
            pred = preds[o.match_id]
            assert o.p_model == pytest.approx(pred.probabilities[o.market][o.selection])
            assert o.fair_odds == pytest.approx(pred.fair_odds[o.market][o.selection])
            assert o.p_implied_raw == pytest.approx(1.0 / o.book_odds)
            assert o.ev == pytest.approx(o.p_model * o.book_odds - 1.0)
            assert o.edge == pytest.approx(o.p_model - o.p_implied_devig)
            assert o.kelly_full == pytest.approx(kelly_fraction(o.p_model, o.book_odds))
            assert o.kelly_fraction == engine.kelly.fraction
            assert (FLAG_VALUE in o.flags) == (o.ev >= engine.min_ev) == o.value
            assert (FLAG_LONGSHOT in o.flags) == (o.book_odds > engine.longshot_odds)
            assert (FLAG_SUSPICIOUS_EDGE in o.flags) == (o.ev > engine.suspicious_ev)
            assert o.home_team == pred.fixture.home_team and o.away_team == pred.fixture.away_team
            assert o.date == pred.fixture.date and o.girone is pred.fixture.girone
        # de-vigged probabilities of each quote sum to one
        by_quote: dict[tuple[str, Market, str], float] = {}
        for o in opportunities:
            key = (o.match_id, o.market, o.bookmaker)
            by_quote[key] = by_quote.get(key, 0.0) + o.p_implied_devig
        assert all(total == pytest.approx(1.0, abs=1e-9) for total in by_quote.values())
        # without matches/traps nothing can be trapped
        assert not any(FLAG_REGRESSION_TRAP in o.flags for o in opportunities)

    def test_stake_zero_unless_value(self, opportunities: list[ValueOpportunity], engine: ValueEngine) -> None:
        assert any(o.value for o in opportunities), "the noisy simulated book should produce at least one value bet"
        for o in opportunities:
            if o.value:
                assert o.stake_units == stake_units(o.p_model, o.book_odds, engine.kelly)
                assert o.stake_units <= engine.kelly.max_stake_units
            else:
                assert o.stake_units == 0.0

    def test_sorted_by_ev_desc(self, opportunities: list[ValueOpportunity]) -> None:
        evs = [o.ev for o in opportunities]
        assert evs == sorted(evs, reverse=True)

    def test_value_bets(self, opportunities: list[ValueOpportunity], engine: ValueEngine) -> None:
        value = engine.value_bets(opportunities)
        assert value
        assert all(o.value and FLAG_VALUE in o.flags for o in value)
        assert len(value) == sum(1 for o in opportunities if o.value)
        evs = [o.ev for o in value]
        assert evs == sorted(evs, reverse=True)
        assert min(evs) >= engine.min_ev

    def test_to_frame_columns(self, opportunities: list[ValueOpportunity], engine: ValueEngine) -> None:
        frame = engine.to_frame(opportunities)
        assert list(frame.columns) == list(VALUE_FRAME_COLUMNS) == [
            "match_id", "date", "girone", "home_team", "away_team", "market", "selection", "bookmaker",
            "book_odds", "fair_odds", "p_model", "p_implied_devig", "edge", "ev", "kelly_full",
            "stake_units", "flags", "value",
        ]
        assert len(frame) == len(opportunities)
        assert frame["ev"].abs().max() < 1.0  # fractions, not percentages
        assert str(frame["date"].dtype) == "datetime64[ns]"
        assert frame["value"].dtype == bool
        assert set(frame["market"]) <= {m.value for m in Market}
        empty = engine.to_frame([])
        assert list(empty.columns) == list(VALUE_FRAME_COLUMNS)
        assert empty.empty

    def test_summary(self, opportunities: list[ValueOpportunity], engine: ValueEngine) -> None:
        s = engine.summary(opportunities)
        value = engine.value_bets(opportunities)
        assert s["n_selections"] == len(opportunities)
        assert s["n_value"] == len(value)
        assert s["mean_ev_value"] == pytest.approx(float(np.mean([o.ev for o in value])))
        assert s["total_stake_units"] == pytest.approx(sum(o.stake_units for o in opportunities))
        assert s["n_trap_flags"] == sum(1 for o in opportunities if FLAG_REGRESSION_TRAP in o.flags)
        assert set(s["by_market"]) == {m.value for m in Market}
        assert sum(s["by_market"].values()) == s["n_value"]
        empty = engine.summary([])
        assert empty["n_selections"] == 0 and empty["n_value"] == 0 and empty["mean_ev_value"] == 0.0

    def test_export_csv_round_trip(self, opportunities: list[ValueOpportunity], engine: ValueEngine, tmp_path) -> None:
        path = engine.export_csv(opportunities, tmp_path / "reports" / "value.csv")
        assert path.exists()
        back = pd.read_csv(path, parse_dates=["date"])
        frame = engine.to_frame(opportunities)
        assert list(back.columns) == list(VALUE_FRAME_COLUMNS)
        assert len(back) == len(frame)
        assert back["match_id"].tolist() == frame["match_id"].tolist()
        np.testing.assert_allclose(back["ev"].to_numpy(), frame["ev"].to_numpy(), rtol=0, atol=1e-9)
        np.testing.assert_allclose(back["stake_units"].to_numpy(), frame["stake_units"].to_numpy(), atol=1e-9)
        assert back["value"].tolist() == frame["value"].tolist()
        assert back["date"].dt.date.tolist() == frame["date"].dt.date.tolist()

    def test_quote_without_prediction_is_skipped(
        self, engine: ValueEngine, mini_predictions: list[MatchPrediction], mini: SimulatedLeague, caplog
    ) -> None:
        orphan = OddsQuote(match_id="NO-SUCH-MATCH", bookmaker="SimBook", market=Market.MATCH_ODDS, odds={"1": 2.0, "X": 3.4, "2": 3.8})
        with caplog.at_level(logging.WARNING, logger="src.engine.value"):
            opps = engine.evaluate(mini_predictions, [*mini.odds, orphan])
        assert len(opps) == sum(len(q.odds) for q in mini.odds)
        assert not any(o.match_id == "NO-SUCH-MATCH" for o in opps)
        assert any("skipped" in r.getMessage() for r in caplog.records)
        # predictions without quotes produce nothing
        assert engine.evaluate(mini_predictions, []) == []

    def test_manual_trap_flags_only_goal_selections(
        self, engine: ValueEngine, mini_predictions: list[MatchPrediction], mini: SimulatedLeague
    ) -> None:
        fx = mini.fixtures[0]
        trap = RegressionTrap.from_window(fx.home_team, fx.girone, 3, 1, 4.2, ("a", "b", "c"), fx.date - timedelta(days=7))
        opps = engine.evaluate(mini_predictions, mini.odds, traps=[trap])
        tagged = [o for o in opps if o.match_id == fx.match_id]
        assert tagged
        for o in tagged:
            goals_side = o.selection.startswith("Over") or o.selection == "GG"
            assert (FLAG_REGRESSION_TRAP in o.flags) == goals_side
            assert (f"TRAP:{fx.home_team}" in o.flags) == goals_side
            assert o.is_trap == goals_side
        assert {o.selection for o in tagged if FLAG_REGRESSION_TRAP in o.flags} == {"Over 1.5", "Over 2.5", "Over 3.5", "GG"}
        # fixtures not involving the trapped team are untouched
        assert not any(FLAG_REGRESSION_TRAP in o.flags for o in opps if o.match_id != fx.match_id)
        # the trap is diagnostic: value/stake decisions are unchanged
        baseline = {(o.match_id, o.market, o.selection): (o.value, o.stake_units) for o in engine.evaluate(mini_predictions, mini.odds)}
        assert {(o.match_id, o.market, o.selection): (o.value, o.stake_units) for o in opps} == baseline

    def test_traps_detected_from_matches_when_not_given(
        self, engine: ValueEngine, mini_predictions: list[MatchPrediction], mini: SimulatedLeague
    ) -> None:
        detected = detect_regression_traps(
            mini.matches,
            window=engine.trap_window,
            max_goals_against=engine.trap_max_goals_against,
            min_xg_against=engine.trap_min_xg_against,
        )
        from_matches = engine.evaluate(mini_predictions, mini.odds, matches=mini.matches)
        from_traps = engine.evaluate(mini_predictions, mini.odds, traps=detected)
        assert [o.flags for o in from_matches] == [o.flags for o in from_traps]
        trapped_teams = {t.team for t in detected}
        for o in from_matches:
            involved = {o.home_team, o.away_team} & trapped_teams
            goals_side = o.selection.startswith("Over") or o.selection == "GG"
            assert (FLAG_REGRESSION_TRAP in o.flags) == bool(involved and goals_side)

    def test_longshot_and_suspicious_edge_flags(self) -> None:
        fx = Fixture("SYN-1", date(2025, 10, 5), 9, Girone.B, "Arezzo", "Ternana")
        pred = _prediction(fx, lam=0.9, mu=1.8, rho=-0.05)
        p_away = pred.probabilities[Market.MATCH_ODDS]["2"]
        assert p_away > 0.4
        quote = OddsQuote(fx.match_id, "SynBook", Market.MATCH_ODDS, {"1": 4.0, "X": 3.6, "2": 8.0})
        eng = ValueEngine(longshot_odds=6.0, suspicious_ev=0.25)
        opps = {o.selection: o for o in eng.evaluate([pred], [quote])}
        away = opps["2"]
        assert away.ev == pytest.approx(p_away * 8.0 - 1.0) and away.ev > 0.25
        assert FLAG_VALUE in away.flags and FLAG_LONGSHOT in away.flags and FLAG_SUSPICIOUS_EDGE in away.flags
        assert away.flags[0] == FLAG_VALUE
        assert away.stake_units > 0.0
        assert FLAG_LONGSHOT not in opps["1"].flags and FLAG_LONGSHOT not in opps["X"].flags
        assert FLAG_SUSPICIOUS_EDGE not in opps["X"].flags
        # a lower longshot threshold flags the home price too; a higher one flags nothing
        assert FLAG_LONGSHOT in {f for o in ValueEngine(longshot_odds=3.9).evaluate([pred], [quote]) for f in o.flags if o.selection == "1"}
        assert not any(FLAG_LONGSHOT in o.flags for o in ValueEngine(longshot_odds=9.0).evaluate([pred], [quote]))

    def test_devig_method_option_and_validation(self, mini_predictions: list[MatchPrediction], mini: SimulatedLeague) -> None:
        shin_engine = ValueEngine(devig_method="shin")
        opps = shin_engine.evaluate(mini_predictions, mini.odds)
        q0 = mini.odds[0]
        implied = {o.selection: o.p_implied_devig for o in opps if o.match_id == q0.match_id and o.market == q0.market}
        expected = devig(q0.odds, "shin")
        for sel in q0.odds:
            assert implied[sel] == pytest.approx(expected[sel])
        with pytest.raises(ValueError):
            ValueEngine(devig_method="magic")
        with pytest.raises(ValueError):
            ValueEngine(trap_window=0)
        with pytest.raises(TypeError):
            ValueEngine(kelly=0.125)  # type: ignore[arg-type]

    def test_opportunity_to_dict(self, opportunities: list[ValueOpportunity]) -> None:
        o = next(o for o in opportunities if o.flags)
        d = o.to_dict()
        assert d["market"] == o.market.value
        assert d["girone"] == o.girone.value
        assert d["flags"] == "|".join(o.flags)
        assert d["date"] == o.date
        assert set(VALUE_FRAME_COLUMNS) <= set(d)
