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

from src.common.schemas import (
    SELECTIONS,
    Fixture,
    Girone,
    Market,
    MatchPrediction,
    MatchRecord,
    OddsQuote,
    matches_to_frame,
)
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

TRAP_TEAM = "Cittadella"  # Girone A team of the 2026/27 registry (the default simulated composition); not a _three_match_window opponent


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


def _synthetic_prediction(fixture: Fixture, probabilities: dict[Market, dict[str, float]]) -> MatchPrediction:
    """A prediction whose market probabilities are chosen by hand (exact boundary cases)."""
    for market, probs in probabilities.items():
        assert set(probs) == set(SELECTIONS[market]) and math.isclose(sum(probs.values()), 1.0, abs_tol=1e-12)
    fair = {m: {s: 1.0 / p for s, p in d.items()} for m, d in probabilities.items()}
    return MatchPrediction(
        fixture=fixture,
        home_rate=1.3,
        away_rate=1.1,
        rho=-0.05,
        matrix=dixon_coles_matrix(1.3, 1.1, -0.05),
        probabilities=probabilities,
        fair_odds=fair,
    )


def _three_match_window(team: str, xg_against: tuple[float, float, float], goals_against: tuple[int, int, int]) -> pd.DataFrame:
    """Canonical frame where ``team`` plays exactly three matches conceding the given goals / xG."""
    opponents = ("Trento", "Lecco", "Novara")
    records: list[MatchRecord] = []
    for i, (opp, xga, ga) in enumerate(zip(opponents, xg_against, goals_against)):
        home = i != 1  # the middle match is away so both melt branches are exercised
        records.append(
            MatchRecord(
                match_id=f"SYN-A-{i + 1:02d}",
                date=date(2025, 9, 7) + timedelta(days=7 * i),
                round=i + 1,
                girone=Girone.A,
                home_team=team if home else opp,
                away_team=opp if home else team,
                home_goals=2 if home else ga,
                away_goals=ga if home else 2,
                home_xg=1.9 if home else xga,
                away_xg=xga if home else 1.9,
            )
        )
    return matches_to_frame(records)


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

    @pytest.mark.parametrize("book", [ODDS, {"1": 1.3, "X": 5.5, "2": 9.0}, {"Over 2.5": 1.72, "Under 2.5": 2.05}])
    def test_power_devig_solves_sum_of_q_to_the_k_equals_one(self, book: dict[str, float]) -> None:
        # p_i = q_i ** k  =>  k = log(p_i) / log(q_i) must be the same for every selection ...
        q = implied_probabilities(book)
        p = devig(book, "power")
        ks = [math.log(p[s]) / math.log(q[s]) for s in book]
        assert max(ks) - min(ks) < 1e-9
        k = ks[0]
        # ... and that k is the root of the defining equation sum((1 / o_i) ** k) == 1, with k > 1 for an over-round book
        assert math.fsum((1.0 / o) ** k for o in book.values()) == pytest.approx(1.0, abs=1e-9)
        assert k > 1.0
        if book is self.ODDS:
            assert k == pytest.approx(1.0618997401771182, abs=1e-9)

    def test_power_differs_from_shin_on_an_unbalanced_book(self) -> None:
        book = {"1": 1.3, "X": 5.5, "2": 9.0}
        power = devig(book, "power")
        shin = devig(book, "shin")
        prop = devig(book, "proportional")
        assert max(abs(power[s] - shin[s]) for s in book) > 1e-3
        # hard-coded reference values (k = 1.0876096588821311 for the power method)
        assert power["1"] == pytest.approx(0.7517511644113969, abs=1e-9)
        assert shin["1"] == pytest.approx(0.7422563047551191, abs=1e-9)
        # on this book the power method compresses the longshots harder than Shin, both more than proportional
        assert power["1"] > shin["1"] > prop["1"]
        assert power["X"] < shin["X"] < prop["X"]
        assert power["2"] < shin["2"] < prop["2"]

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

    def test_kelly_config_accepts_numpy_scalars(self) -> None:
        cfg = KellyConfig(
            fraction=np.float32(0.125),
            bankroll_units=np.int64(100),
            max_stake_units=np.int64(3),
            min_stake_units=np.float64(0.1),
            round_to=np.float64(0.1),
        )
        for name in ("fraction", "bankroll_units", "max_stake_units", "min_stake_units", "round_to"):
            assert type(getattr(cfg, name)) is float
        assert cfg.fraction == 0.125 and cfg.bankroll_units == 100.0 and cfg.max_stake_units == 3.0
        assert stake_units(0.55, 2.0, cfg) == stake_units(0.55, 2.0, KellyConfig())
        assert KellyConfig(fraction=np.float64(0.25)).fraction == KELLY_QUARTER
        assert KellyConfig(max_stake_units=5).max_stake_units == 5.0

    @pytest.mark.parametrize(
        "bad",
        [True, False, np.bool_(True), math.nan, np.float64("nan"), math.inf, -math.inf, np.float32("inf"), "0.125", None, [0.125], 0.125j],
        ids=repr,
    )
    @pytest.mark.parametrize("name", ["fraction", "bankroll_units", "max_stake_units", "min_stake_units", "round_to"])
    def test_kelly_config_rejects_bool_nan_inf_and_wrong_types(self, name: str, bad: object) -> None:
        with pytest.raises(ValueError, match=f"KellyConfig.{name}"):
            KellyConfig(**{name: bad})

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

    def test_nan_opponent_xg_in_window_blocks_the_flag(self, league: SimulatedLeague) -> None:
        # 6.0 xGA over the window (2.0 per match): after NaN-ing one match the two remaining
        # ones still sum to 4.0 >= 3.5, so only the "xG known in every match" rule can exclude the team
        intact = inject_regression_trap(league.matches, TRAP_TEAM, xga_total=6.0)
        before = {t.team: t for t in detect_regression_traps(intact)}
        assert TRAP_TEAM in before and before[TRAP_TEAM].xg_against == pytest.approx(6.0)
        df = intact.copy()
        window_ids = _team_window_ids(df, TRAP_TEAM)
        row = df.index[df["match_id"] == window_ids[1]][0]
        column = "away_xg" if df.at[row, "home_team"] == TRAP_TEAM else "home_xg"
        df.loc[row, column] = np.nan
        remaining = df.loc[df["match_id"].isin(window_ids)]
        xga_known = np.where(remaining["home_team"] == TRAP_TEAM, remaining["away_xg"], remaining["home_xg"])
        assert np.isnan(xga_known).sum() == 1 and np.nansum(xga_known) >= 3.5
        teams = {t.team for t in detect_regression_traps(df)}
        assert TRAP_TEAM not in teams
        # the other traps are unaffected by the edit
        assert teams == set(before) - {TRAP_TEAM}

    def test_window_summing_exactly_to_the_threshold_in_decimal_is_flagged(self) -> None:
        # 0.01 + 2.01 + 1.48 == 3.50 in decimal but 3.4999999999999996 as a binary float sum
        xga = (0.01, 2.01, 1.48)
        assert float(pd.Series(xga).sum()) < 3.5
        df = _three_match_window(TRAP_TEAM, xga, goals_against=(1, 0, 0))
        traps = {t.team: t for t in detect_regression_traps(df, min_xg_against=3.5)}
        assert TRAP_TEAM in traps
        trap = traps[TRAP_TEAM]
        assert trap.xg_against == 3.5  # stored rounded, exactly the decimal sum
        assert trap.goals_against == 1 and trap.window == 3
        assert trap.severity == 2.5 and trap.xga_per_match == pytest.approx(3.5 / 3)
        assert "3.50 xGA" in trap.description()
        assert list(trap.match_ids) == ["SYN-A-01", "SYN-A-02", "SYN-A-03"]
        # the opponents each played once: not eligible
        assert set(traps) == {TRAP_TEAM}
        # just above the decimal sum the window is (correctly) not flagged; one more goal conceded neither
        assert detect_regression_traps(df, min_xg_against=3.500001) == []
        assert detect_regression_traps(_three_match_window(TRAP_TEAM, xga, goals_against=(1, 1, 0))) == []

    def test_injected_total_is_flagged_at_a_threshold_equal_to_the_total(self, trapped_matches: pd.DataFrame) -> None:
        # inject_regression_trap(..., xga_total=4.2) writes 1.4 + 1.4 + 1.4, whose float sum is 4.199999999999999
        traps = {t.team: t for t in detect_regression_traps(trapped_matches, min_xg_against=4.2)}
        assert TRAP_TEAM in traps
        assert traps[TRAP_TEAM].xg_against == 4.2
        assert traps[TRAP_TEAM].severity == pytest.approx(3.2)

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
        df = inject_regression_trap(trapped_matches, "Perugia", xga_total=6.0)
        traps = detect_regression_traps(df)
        severities = [t.severity for t in traps]
        assert severities == sorted(severities, reverse=True)
        assert traps[0].team == "Perugia"
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

    def test_positive_ev_below_min_ev_has_zero_stake_despite_sizeable_kelly(self) -> None:
        fx = Fixture("SYN-2", date(2025, 10, 12), 10, Girone.C, "Benevento", "Catania")
        pred = _synthetic_prediction(fx, {Market.MATCH_ODDS: {"1": 0.52, "X": 0.28, "2": 0.20}})
        quote = OddsQuote(fx.match_id, "SynBook", Market.MATCH_ODDS, {"1": 2.0, "X": 3.4, "2": 4.8})
        eng = ValueEngine(min_ev=0.05)
        home = {o.selection: o for o in eng.evaluate([pred], [quote])}["1"]
        assert 0.0 < home.ev < eng.min_ev and home.ev == pytest.approx(0.04)
        # full Kelly 0.04 -> 1/8 Kelly on 100 units = 0.5 units: a sizeable stake if it were a value bet
        assert home.kelly_full == pytest.approx(0.04)
        assert stake_units(home.p_model, home.book_odds, eng.kelly) == 0.5
        assert home.value is False and FLAG_VALUE not in home.flags
        assert home.stake_units == 0.0
        # lowering min_ev below the EV turns the very same selection into a 0.5-unit value bet
        relaxed = {o.selection: o for o in ValueEngine(min_ev=0.03).evaluate([pred], [quote])}["1"]
        assert relaxed.value is True and relaxed.stake_units == 0.5

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

    def test_trap_on_away_team_flags_goal_selections_and_unrelated_trap_flags_nothing(
        self, engine: ValueEngine, mini_predictions: list[MatchPrediction], mini: SimulatedLeague
    ) -> None:
        fx = mini.fixtures[0]
        involved = {fx.home_team, fx.away_team}
        stranger = next(t for p in mini_predictions for t in (p.fixture.home_team, p.fixture.away_team) if t not in involved)
        assert stranger not in involved
        when = fx.date - timedelta(days=7)
        away_trap = RegressionTrap.from_window(fx.away_team, fx.girone, 3, 0, 4.0, ("a", "b", "c"), when)
        other_trap = RegressionTrap.from_window("Pro Patria", Girone.A, 3, 1, 5.0, ("d", "e", "f"), when)
        opps = engine.evaluate(mini_predictions, mini.odds, traps=[away_trap, other_trap])
        tagged = [o for o in opps if o.match_id == fx.match_id]
        assert tagged
        for o in tagged:
            goals_side = o.selection.startswith("Over") or o.selection == "GG"
            assert (FLAG_REGRESSION_TRAP in o.flags) == goals_side
            assert (f"TRAP:{fx.away_team}" in o.flags) == goals_side
            assert f"TRAP:{fx.home_team}" not in o.flags
        assert not any(FLAG_REGRESSION_TRAP in o.flags for o in opps if o.match_id != fx.match_id)
        assert not any("TRAP:Pro Patria" in o.flags for o in opps)
        # both teams trapped: both TRAP:<team> markers, home first, on the goals selections only
        home_trap = RegressionTrap.from_window(fx.home_team, fx.girone, 3, 1, 4.2, ("g", "h", "i"), when)
        both = [o for o in engine.evaluate(mini_predictions, mini.odds, traps=[other_trap, away_trap, home_trap]) if o.match_id == fx.match_id]
        for o in both:
            goals_side = o.selection.startswith("Over") or o.selection == "GG"
            markers = [f for f in o.flags if f.startswith("TRAP:")]
            assert markers == ([f"TRAP:{fx.home_team}", f"TRAP:{fx.away_team}"] if goals_side else [])
        # a trap on a team that plays no fixture never shows up anywhere
        assert not any(FLAG_REGRESSION_TRAP in o.flags for o in engine.evaluate(mini_predictions, mini.odds, traps=[other_trap]))

    def test_flag_boundaries_value_inclusive_longshot_and_suspicious_strict(self) -> None:
        fx = Fixture("SYN-3", date(2025, 10, 19), 11, Girone.B, "Perugia", "Ascoli")
        pred = _synthetic_prediction(fx, {Market.MATCH_ODDS: {"1": 0.3, "X": 0.2, "2": 0.5}})
        quote = OddsQuote(fx.match_id, "SynBook", Market.MATCH_ODDS, {"1": 2.6, "X": 3.6, "2": 2.5})

        def away(eng: ValueEngine) -> ValueOpportunity:
            return {o.selection: o for o in eng.evaluate([pred], [quote])}["2"]

        # p * odds - 1 == 0.5 * 2.5 - 1 == 0.25 exactly in binary floating point
        assert away(ValueEngine()).ev == 0.25
        # VALUE is inclusive: ev == min_ev flags, ev just below does not
        on = away(ValueEngine(min_ev=0.25))
        assert on.value is True and FLAG_VALUE in on.flags and on.stake_units > 0.0
        off = away(ValueEngine(min_ev=0.25 + 1e-9))
        assert off.value is False and FLAG_VALUE not in off.flags and off.stake_units == 0.0
        # LONGSHOT is strict: book_odds == longshot_odds is not a longshot, book_odds just above is
        equal = {o.selection: o for o in ValueEngine(longshot_odds=2.5).evaluate([pred], [quote])}
        assert FLAG_LONGSHOT not in equal["2"].flags
        assert FLAG_LONGSHOT in equal["1"].flags and FLAG_LONGSHOT in equal["X"].flags
        assert FLAG_LONGSHOT in away(ValueEngine(longshot_odds=2.5 - 1e-9)).flags
        # SUSPICIOUS_EDGE is strict: ev == suspicious_ev is not suspicious, ev just above is
        assert FLAG_SUSPICIOUS_EDGE not in away(ValueEngine(suspicious_ev=0.25)).flags
        assert FLAG_SUSPICIOUS_EDGE in away(ValueEngine(suspicious_ev=0.25 - 1e-9)).flags

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
