"""Tests for :mod:`src.engine.backtest`: walk-forward without leakage, scores, odds comparison, track record.

Every league here is synthetic with **known** goal rates (independent Poisson,
the Dixon-Coles model with ``rho = 0``), so the true probabilities of every
match are known and the tests can tell a calibrated forecast from a lucky one.
"""
from __future__ import annotations

import logging
import math
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import date, timedelta
from typing import Any

import numpy as np
import pandas as pd
import pytest

from src.common.schemas import Market
from src.common.teams import TeamRegistry
from src.engine import backtest as bt
from src.engine.model import DixonColesConfig, DixonColesEngine, dixon_coles_pmf, markets_from_matrix
from src.engine.picks import rank_picks

UTC_SEASONS = (date(2025, 8, 24), date(2026, 8, 30))


# ============================================================================ synthetic leagues
def _round_robin(teams: Sequence[str]) -> list[list[tuple[str, str]]]:
    """Double round robin (circle method), home and away legs."""
    arr = list(teams) + ([None] if len(teams) % 2 else [])
    n = len(arr)
    rounds: list[list[tuple[str, str]]] = []
    for r in range(n - 1):
        pairs = []
        for i in range(n // 2):
            a, b = arr[i], arr[n - 1 - i]
            if a is not None and b is not None:
                pairs.append((a, b) if r % 2 == 0 else (b, a))
        rounds.append(pairs)
        arr = [arr[0], arr[-1], *arr[1:-1]]
    return rounds + [[(b, a) for a, b in rnd] for rnd in rounds]


@dataclass
class League:
    """A synthetic league: the canonical match frame plus the true rates of every match."""

    matches: pd.DataFrame
    truth: pd.DataFrame  # match_id, lam, mu
    alpha: dict[str, float]
    beta: dict[str, float]


def make_league(
    sizes: Mapping[str, int],
    season_starts: Sequence[date] = UTC_SEASONS,
    *,
    seed: int = 1,
    current_rounds: int | None = 8,
    sigma_attack: float = 0.35,
    sigma_defence: float = 0.25,
    home: float = 0.25,
    base: float = 0.1,
    midweek_rounds: Sequence[int] = (),
    newcomers: Mapping[str, str] | None = None,
) -> League:
    """Weekly double round robins per competition (Sundays; rounds in ``midweek_rounds`` moved to the following Wednesday).

    The last season is cut after ``current_rounds`` rounds. ``newcomers`` maps a team that exists only in the
    **last** season to the team it replaces (a promoted club: unseen by every fit before its first match).
    """
    rng = np.random.default_rng(seed)
    teams = {code: [f"{code}-{i:02d}" for i in range(n)] for code, n in sizes.items()}
    alpha = {t: float(rng.normal(0.0, sigma_attack)) for ts in teams.values() for t in ts}
    beta = {t: float(rng.normal(0.0, sigma_defence)) for ts in teams.values() for t in ts}
    for new, old in (newcomers or {}).items():
        alpha[new], beta[new] = alpha[old], beta[old]
    rows: list[dict[str, Any]] = []
    truth: list[dict[str, Any]] = []
    counter = 0
    for k, start in enumerate(season_starts):
        last = k == len(season_starts) - 1
        for code, members in teams.items():
            members = [next((new for new, old in (newcomers or {}).items() if old == t), t) if last else t for t in members]
            schedule = _round_robin(members)
            if last and current_rounds is not None:
                schedule = schedule[:current_rounds]
            for r, pairs in enumerate(schedule, start=1):
                day = start + timedelta(days=7 * (r - 1)) + (timedelta(days=3) if r in midweek_rounds else timedelta(0))
                for h, a in pairs:
                    lam = math.exp(base + alpha[h] - beta[a] + home)
                    mu = math.exp(base + alpha[a] - beta[h])
                    counter += 1
                    match_id = f"S{k}-{code}-{counter:05d}"
                    rows.append(
                        {
                            "match_id": match_id,
                            "date": day,
                            "round": r,
                            "girone": code,
                            "home_team": h,
                            "away_team": a,
                            "home_goals": int(rng.poisson(lam)),
                            "away_goals": int(rng.poisson(mu)),
                        }
                    )
                    truth.append({"match_id": match_id, "lam": lam, "mu": mu})
    return League(pd.DataFrame(rows), pd.DataFrame(truth), alpha, beta)


SMALL = {"A": 10, "B": 10, "DA": 10}


@pytest.fixture(scope="module")
def small_league() -> League:
    return make_league(SMALL, seed=3)


@pytest.fixture(scope="module")
def small_result(small_league: League) -> bt.BacktestResult:
    return bt.walk_forward(small_league.matches, min_train_matches=100)


def _hand_frame(rows: Sequence[Mapping[str, Any]]) -> pd.DataFrame:
    """A minimal backtest-like frame from hand-written rows (defaults fill the columns a test does not care about)."""
    out = []
    for i, row in enumerate(rows):
        base = {
            "match_id": f"M{i}",
            "date": pd.Timestamp(2026, 9, 6),
            "season": "2026/27",
            "matchday": "2026-09-04",
            "block": "2026-09-01",
            "girone": "A",
            "family": "Serie C",
            "section": bt.section_of_girone("A"),
            "round": 3,
            "home_team": f"H{i}",
            "away_team": f"V{i}",
            "home_goals": 1,
            "away_goals": 1,
            "lam": 1.3,
            "mu": 1.1,
            "rho": 0.0,
        }
        base.update(row)
        out.append(base)
    frame = pd.DataFrame(out)
    for spec in bt.MARKET_SPECS.values():
        for prefix in ("p_", "base_"):
            for column, default in zip(spec.columns(prefix), (0.5, 0.5) if spec.is_binary else (0.45, 0.27, 0.28)):
                if column not in frame.columns:
                    frame[column] = default
    hg, ag = frame["home_goals"].to_numpy(), frame["away_goals"].to_numpy()
    frame["res_1X2"] = np.where(hg > ag, "1", np.where(hg == ag, "X", "2"))
    frame["gg"] = ((hg > 0) & (ag > 0)).astype(int)
    frame["o15"] = (hg + ag > 1.5).astype(int)
    frame["o25"] = (hg + ag > 2.5).astype(int)
    frame["o35"] = (hg + ag > 3.5).astype(int)
    return frame


# ============================================================================ math helpers
def test_vectorised_markets_match_the_engine_score_matrix() -> None:
    rng = np.random.default_rng(0)
    lam = rng.uniform(0.2, 3.5, 25)
    mu = rng.uniform(0.2, 3.0, 25)
    for rho in (-0.15, 0.0, 0.1):
        arrays = bt._market_arrays(lam, mu, rho, 10)
        for i in range(len(lam)):
            expected = markets_from_matrix(dixon_coles_pmf(float(lam[i]), float(mu[i]), rho, 10))
            for spec in bt.MARKET_SPECS.values():
                for selection, column in zip(spec.selections, spec.columns("p_")):
                    assert arrays[column][i] == pytest.approx(expected[spec.market][selection], abs=1e-12)


def test_wilson_and_roi_intervals() -> None:
    low, high = bt.wilson_interval(0, 10)
    assert low == 0.0 and high == pytest.approx(0.2775, abs=1e-4)
    low, high = bt.wilson_interval(5, 10)
    assert (low, high) == pytest.approx((0.2366, 0.7634), abs=1e-4)
    assert bt.wilson_interval(10, 10)[1] == 1.0 and all(math.isnan(v) for v in bt.wilson_interval(0, 0))
    lows, highs = bt.wilson_interval(np.array([1, 50]), np.array([2, 100]))
    assert lows.shape == (2,) and highs[1] - lows[1] < highs[0] - lows[0], "more data, narrower interval"
    roi, lo, hi = bt.roi_interval([1.0, -1.0, 1.0, -1.0])  # 2 wins at 2.0 out of 4: exact bounds of the rate x 2.0 - 1
    rate_low, rate_high = bt.exact_binomial_interval(2, 4)
    assert roi == 0.0 and (lo, hi) == pytest.approx((2.0 * rate_low - 1.0, 2.0 * rate_high - 1.0)) and lo < 0.0 < hi
    assert bt.roi_interval([0.5])[0] == 0.5 and math.isnan(bt.roi_interval([0.5])[1])
    assert all(math.isnan(v) for v in bt.roi_interval([float("nan")]))


def test_exact_binomial_interval_matches_clopper_pearson() -> None:
    from scipy import stats

    for k, n in ((0, 49), (1, 49), (2, 23), (5, 10), (10, 10)):
        low, high = bt.exact_binomial_interval(k, n)
        assert low == pytest.approx(stats.beta.ppf(0.025, k, n - k + 1) if k else 0.0)
        assert high == pytest.approx(stats.beta.ppf(0.975, k + 1, n - k) if k < n else 1.0)
    assert bt.exact_binomial_interval(2, 23)[0] < bt.wilson_interval(2, 23)[0], "conservative where Wilson is not"
    assert all(math.isnan(v) for v in bt.exact_binomial_interval(0, 0))
    with pytest.raises(ValueError):
        bt.exact_binomial_interval(3, 2)


def test_roi_interval_is_never_zero_width_nor_below_minus_100_percent() -> None:
    """Regression: the normal approximation gave [-100%, -100%] with no win and -433% after one big win."""
    # no win at all: without prices the interval cannot be determined ('n.d.'), never "certainly -100%"
    roi, lo, hi = bt.roi_interval([-1.0] * 30)
    assert roi == -1.0 and math.isnan(lo) and math.isnan(hi)
    # no win with the prices known: one-sided, from the exact upper bound of the win rate times the price
    roi, lo, hi = bt.roi_interval([-1.0] * 49, odds=[170.0] * 49)
    assert roi == -1.0 and lo == -1.0 and hi == pytest.approx(170.0 * bt.exact_binomial_interval(0, 49)[1] - 1.0) and hi > 0.0
    # one win at 170 among 49 slips: the lower bound stays above -100% and the point estimate inside
    profits = [-1.0] * 48 + [169.0]
    roi, lo, hi = bt.roi_interval(profits, odds=[170.0] * 49)
    assert roi == pytest.approx(169.0 / 49 - 48 / 49) and -1.0 <= lo < 0.0 < roi < hi
    roi, lo, hi = bt.roi_interval(profits)  # the price of the winning slip is the only one the profits reveal
    assert -1.0 <= lo < roi < hi
    # every bet won: not "certainly +50%" either
    roi, lo, hi = bt.roi_interval([0.5] * 10, odds=[1.5] * 10)
    assert roi == 0.5 and lo < 0.5 and hi == pytest.approx(0.5)
    with pytest.raises(ValueError):
        bt.roi_interval([-1.0, 1.0], odds=[2.0])


def test_roi_interval_bootstraps_when_wins_and_losses_are_plentiful() -> None:
    rng = np.random.default_rng(3)
    odds = rng.uniform(1.6, 3.5, 400)
    won = rng.random(400) < 1.0 / odds
    profits = np.where(won, odds - 1.0, -1.0)
    roi, lo, hi = bt.roi_interval(profits, odds=odds)
    assert roi == pytest.approx(profits.mean()) and -1.0 <= lo < roi < hi
    half = 1.959964 * profits.std(ddof=1) / math.sqrt(profits.size)
    assert lo == pytest.approx(roi - half, abs=0.03) and hi == pytest.approx(roi + half, abs=0.03), "close to normal for many singles"
    assert bt.roi_interval(profits, odds=odds) == (roi, lo, hi), "deterministic (fixed seed)"
    # a skewed payoff (6 wins at 50 out of 150): the percentile bootstrap is asymmetric, longer on the upside
    skewed = np.array([49.0] * 6 + [-1.0] * 144)
    roi, lo, hi = bt.roi_interval(skewed)
    assert -1.0 <= lo < roi < hi and hi - roi > roi - lo


def test_market_of_and_specs() -> None:
    assert bt.market_of("GG/NG") is Market.BTTS and bt.market_of("BTTS") is Market.BTTS and bt.market_of(Market.MATCH_ODDS) is Market.MATCH_ODDS
    with pytest.raises(ValueError):
        bt.market_of("corners")
    assert bt.PROB_COLUMNS == ("p_1", "p_X", "p_2", "p_GG", "p_NG", "p_O15", "p_U15", "p_O25", "p_U25", "p_O35", "p_U35")
    assert bt.market_spec("OU2.5").columns("book_") == ("book_O25", "book_U25")


# ============================================================================ walk-forward
def test_walk_forward_columns_dtypes_and_determinism(small_league: League, small_result: bt.BacktestResult) -> None:
    frame = small_result.frame
    assert list(frame.columns) == list(bt.BACKTEST_COLUMNS)
    current = small_league.matches[pd.to_datetime(small_league.matches["date"]) >= pd.Timestamp(2026, 7, 1)]
    assert len(frame) == len(current) and set(frame["match_id"]) == set(current["match_id"])
    assert small_result.seasons == ("2026/27",), "default: the season of the latest match"
    assert str(frame["date"].dtype) == "datetime64[ns]" and str(frame["train_size"].dtype) == "int64" and str(frame["cold_start"].dtype) == "bool"
    for spec in bt.MARKET_SPECS.values():
        assert np.allclose(frame[list(spec.columns("p_"))].sum(axis=1), 1.0)
        assert np.allclose(frame[list(spec.columns("base_"))].sum(axis=1), 1.0)
    assert set(frame["res_1X2"]) <= {"1", "X", "2"} and set(frame["gg"]) <= {0, 1}
    assert (frame["gg"] == ((frame["home_goals"] > 0) & (frame["away_goals"] > 0)).astype(int)).all()
    assert (frame["fit_method"] == "L-BFGS-B").all() and frame["fit_converged"].all() and not frame["cold_start"].any()
    assert set(frame["section"]) == {bt.section_of_girone(c) for c in SMALL} and set(frame["family"]) == {"Serie C", "Serie D"}
    assert small_result.n_blocks == 8 and small_result.n_fits == 8 and small_result.n_skipped_blocks == 0 and small_result.matchdays == tuple(sorted(frame["matchday"].unique()))
    again = bt.walk_forward(small_league.matches, min_train_matches=100)
    pd.testing.assert_frame_equal(frame, again.frame)
    assert "partite valutate fuori campione" in small_result.summary() and "8 blocchi" in small_result.summary()


def test_walk_forward_never_uses_future_matches(small_league: League, monkeypatch: pytest.MonkeyPatch) -> None:
    seen: list[pd.Timestamp] = []
    real_fit = DixonColesEngine.fit

    def spying_fit(self: DixonColesEngine, matches: pd.DataFrame, **kwargs: Any) -> Any:
        seen.append(pd.to_datetime(matches["date"]).max())
        return real_fit(self, matches, **kwargs)

    monkeypatch.setattr(DixonColesEngine, "fit", spying_fit)
    result = bt.walk_forward(small_league.matches, min_train_matches=100)
    frame = result.frame
    assert len(seen) == result.n_fits == frame["block"].nunique()
    for (block, rows), train_max in zip(frame.groupby("block", sort=True), seen):
        assert train_max < rows["date"].min(), f"block {block}: the fit saw a match of the block or later"
        assert (rows["train_end"] == train_max).all()
    assert (frame["train_end"] < frame["date"]).all()
    # training size = every match before the block (history + earlier blocks), nothing more
    dates = pd.to_datetime(small_league.matches["date"])
    for _block, rows in frame.groupby("block"):
        assert (rows["train_size"] == int((dates < rows["date"].min()).sum())).all()


def test_changing_future_results_never_changes_earlier_predictions(small_league: League, small_result: bt.BacktestResult) -> None:
    last_block = small_result.frame["block"].max()
    future = small_result.frame.loc[small_result.frame["block"] == last_block, "match_id"]
    altered = small_league.matches.copy()
    mask = altered["match_id"].isin(set(future))
    altered.loc[mask, "home_goals"] = 7
    altered.loc[mask, "away_goals"] = 0
    result = bt.walk_forward(altered, min_train_matches=100)
    before = small_result.frame.loc[small_result.frame["block"] != last_block, list(bt.PROB_COLUMNS)].reset_index(drop=True)
    after = result.frame.loc[result.frame["block"] != last_block, list(bt.PROB_COLUMNS)].reset_index(drop=True)
    pd.testing.assert_frame_equal(before, after)
    # the altered block itself is predicted identically too (its own results are never inputs of its fit)
    pd.testing.assert_frame_equal(
        small_result.frame.loc[small_result.frame["block"] == last_block, list(bt.PROB_COLUMNS)].reset_index(drop=True),
        result.frame.loc[result.frame["block"] == last_block, list(bt.PROB_COLUMNS)].reset_index(drop=True),
    )


def test_walk_forward_blocks_seasons_and_skips() -> None:
    league = make_league({"A": 8, "DA": 8}, seed=5, current_rounds=6, midweek_rounds=(4,))
    weekly = bt.walk_forward(league.matches, min_train_matches=60, block="week")
    football = bt.walk_forward(league.matches, min_train_matches=60, block="block")
    daily = bt.walk_forward(league.matches, min_train_matches=60, block="day")
    # the Wednesday round 4 shares its Tuesday-Monday week with round 5 (Sunday) but has its own football block
    assert weekly.n_blocks == 5 and football.n_blocks == 6 and daily.n_blocks == 6
    assert set(weekly.frame["matchday"]) == set(football.frame["block"]), "matchday = block of the football week"
    round5 = football.frame["round"] == 5
    assert (football.frame.loc[round5, "train_size"] > weekly.frame.loc[weekly.frame["round"] == 5, "train_size"].to_numpy()).all()
    # history seasons can be evaluated too; the very first blocks lack training data and are skipped
    both = bt.walk_forward(league.matches, seasons=["2025_26", "2026/27"], min_train_matches=60)
    assert both.seasons == ("2025/26", "2026/27") and both.skipped_matches["storico insufficiente"] > 0
    assert both.n_matches + sum(both.skipped_matches.values()) == len(league.matches)
    nothing = bt.walk_forward(league.matches, min_train_matches=10_000)
    assert nothing.empty and nothing.skipped_matches == {"storico insufficiente": 6 * 8} and "nessuna partita valutabile" in nothing.summary()
    assert list(nothing.frame.columns) == list(bt.BACKTEST_COLUMNS)
    with pytest.raises(ValueError):
        bt.walk_forward(league.matches, block="month")  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        bt.walk_forward(league.matches, seasons=["2026/28"])
    with pytest.raises(ValueError):
        bt.walk_forward(league.matches, min_train_matches=-1)
    # the registry's season is the default evaluated season
    registry = TeamRegistry.from_teams("2025/26", {"A": [f"A-{i:02d}" for i in range(8)], "DA": [f"DA-{i:02d}" for i in range(8)]})
    by_registry = bt.walk_forward(league.matches, registry=registry, min_train_matches=10)
    assert by_registry.seasons == ("2025/26",)


def test_cold_start_team_is_predicted_with_prior_strengths_and_flagged() -> None:
    league = make_league({"A": 8}, seed=7, current_rounds=5, newcomers={"A-NEW": "A-03"})
    result = bt.walk_forward(league.matches, min_train_matches=40)
    frame = result.frame
    involved = (frame["home_team"] == "A-NEW") | (frame["away_team"] == "A-NEW")
    first = frame.loc[involved].iloc[0]
    assert bool(first["cold_start"]) and np.isfinite(first[list(bt.PROB_COLUMNS)].to_numpy(dtype=float)).all()
    assert frame.loc[involved].iloc[1:]["cold_start"].sum() == 0, "seen from its second matchday on"
    assert int(frame["cold_start"].sum()) == 1 == result.n_cold_start
    assert "1 partite con una squadra mai vista prima" in result.summary()


def test_production_prior_rule_and_config() -> None:
    league = make_league({"A": 10}, seed=2, current_rounds=None)  # 18 matches per team in a full season
    matches = league.matches.assign(date=pd.to_datetime(league.matches["date"]))
    current = matches[matches["date"] >= pd.Timestamp(2026, 7, 1)]
    early = current[current["round"] <= 3]
    assert bt.production_prior_on(early, season="2026/27") is True, "3 rounds: thin coverage, prior on"
    assert bt.production_prior_on(current, season="2026/27") is False, "a full season: plain MLE"
    assert bt.production_prior_on(matches[matches["date"] < pd.Timestamp(2026, 7, 1)], season="2026/27") is True, "history never counts"
    registry = TeamRegistry.from_teams("2026/27", {"A": [f"A-{i:02d}" for i in range(10)]})
    assert bt.production_prior_on(current, registry=registry, season="2026/27") is False
    assert bt.production_prior_on(early, registry=registry, season="2026/27") is True
    config, prior_on = bt.production_config(early, season="2026/27")
    assert prior_on and config.method == "L-BFGS-B" and config.xi == bt.PRODUCTION_XI and config.l2_penalty > 0
    config, prior_on = bt.production_config(current, season="2026/27", method="SLSQP")
    assert not prior_on and config.method == "SLSQP" and config.l2_penalty == 0
    # a fixed config or a rule are honoured block by block
    short = make_league({"A": 10}, seed=2, current_rounds=4)
    fixed = bt.walk_forward(short.matches, seasons=["2026/27"], config=DixonColesConfig(xi=0.0, method="L-BFGS-B"), min_train_matches=50)
    assert not fixed.frame["prior"].any()
    calls: list[str] = []

    def rule(train: pd.DataFrame, season: str) -> DixonColesConfig:
        calls.append(season)
        return DixonColesConfig.with_gaussian_prior(xi=0.001, method="L-BFGS-B")

    ruled = bt.walk_forward(short.matches, seasons=["2026/27"], config=rule, min_train_matches=50)
    assert ruled.frame["prior"].all() and calls == ["2026/27"] * ruled.n_fits == ["2026/27"] * 4


def test_walk_forward_quiet_mode_silences_per_fit_logs(small_league: League, caplog: pytest.LogCaptureFixture) -> None:
    caplog.set_level(logging.INFO)
    bt.walk_forward(small_league.matches, min_train_matches=100)
    assert not [r for r in caplog.records if r.name == "src.engine.model"]
    assert any(r.getMessage() == "walk-forward backtest complete" for r in caplog.records)
    caplog.clear()
    bt.walk_forward(small_league.matches, min_train_matches=100, quiet=False)
    assert [r for r in caplog.records if r.name == "src.engine.model"]


def test_walk_forward_runtime_on_the_full_perimeter() -> None:
    """Runtime guard: 13 competitions (242 teams), two full seasons of history plus 11 rounds, well under 60 s."""
    sizes = {"A": 20, "B": 20, "C": 20, "SB": 20, **{f"D{x}": 18 for x in "ABCDEFGHI"}}
    league = make_league(sizes, (date(2024, 8, 25), date(2025, 8, 24), date(2026, 8, 30)), seed=11, current_rounds=11)
    assert len(league.matches) > 9_000
    started = time.perf_counter()
    result = bt.walk_forward(league.matches)
    seconds = time.perf_counter() - started
    assert result.n_blocks == 11 and result.n_matches == 11 * (3 * 10 + 10 + 9 * 9)
    assert seconds < 30.0, f"walk-forward of the full perimeter took {seconds:.1f} s"
    assert result.frame["prior"].all(), "11 rounds of 38: the production rule keeps the prior on"


# ============================================================================ scores
def test_metrics_math_on_hand_made_examples() -> None:
    frame = _hand_frame(
        [
            {"home_goals": 2, "away_goals": 0, "p_1": 0.6, "p_X": 0.3, "p_2": 0.1, "p_GG": 0.4, "p_NG": 0.6, "base_1": 0.4, "base_X": 0.3, "base_2": 0.3},
            {"home_goals": 1, "away_goals": 1, "p_1": 0.5, "p_X": 0.2, "p_2": 0.3, "p_GG": 0.7, "p_NG": 0.3, "base_1": 0.4, "base_X": 0.3, "base_2": 0.3},
        ]
    )
    table = bt.metrics(frame, by=None, markets=["1X2", "GG/NG"])
    one = table.loc[table["market"] == "1X2"].iloc[0]
    assert one["n"] == 2 and one["selection"] == "1"
    assert one["brier"] == pytest.approx(((0.4**2 + 0.3**2 + 0.1**2) + (0.5**2 + 0.8**2 + 0.3**2)) / 2)
    assert one["log_loss"] == pytest.approx(-(math.log(0.6) + math.log(0.2)) / 2)
    assert one["accuracy"] == 0.5 and one["mean_pred"] == pytest.approx(0.55) and one["observed"] == 0.5
    assert one["log_loss_naive"] == pytest.approx(-(math.log(0.4) + math.log(0.3)) / 2)
    assert one["skill_naive"] == pytest.approx(1 - one["log_loss"] / one["log_loss_naive"])
    gg = table.loc[table["market"] == "GG/NG"].iloc[0]
    assert gg["brier"] == pytest.approx((0.4**2 + 0.3**2) / 2), "binary Brier on GG"
    assert gg["log_loss"] == pytest.approx(-(math.log(0.6) + math.log(0.7)) / 2) and gg["accuracy"] == 1.0
    assert gg["observed"] == 0.5 and gg["mean_pred"] == pytest.approx(0.55)
    # grouping: one row per group and market, configuration order
    two = pd.concat([frame, frame.assign(girone="DA", section=bt.section_of_girone("DA"), match_id=["N0", "N1"])], ignore_index=True)
    grouped = bt.metrics(two, by=("section", "girone"), markets=["1X2"])
    assert list(grouped["girone"]) == ["A", "DA"] and (grouped["n"] == 2).all()
    assert bt.metrics(frame.iloc[0:0], by=None).empty


def test_selection_calibration_and_reliability_of_a_calibrated_forecast() -> None:
    rng = np.random.default_rng(42)
    n = 20_000
    p = rng.uniform(0.05, 0.95, n)
    gg = (rng.uniform(size=n) < p).astype(int)
    frame = pd.DataFrame({"p_GG": p, "p_NG": 1 - p, "gg": gg, "section": "S"})
    table = bt.reliability(frame, "GG/NG", "GG", bins=10)
    assert len(table) == 10 and table["n"].sum() == n
    inside = (table["wilson_low"] <= table["mean_pred"]) & (table["mean_pred"] <= table["wilson_high"])
    assert inside.sum() >= 9, "a calibrated forecast sits inside the 95% intervals (one miss allowed by chance)"
    assert (table["observed"] - table["mean_pred"]).abs().max() < 0.03
    quantile = bt.reliability(frame, "GG/NG", "NG", bins=5, strategy="quantile")
    assert len(quantile) == 5 and abs(int(quantile["n"].max()) - int(quantile["n"].min())) <= 1
    # a deliberately overconfident forecast is visibly off the diagonal
    sharp = frame.assign(p_GG=np.clip(0.5 + 1.8 * (p - 0.5), 0.01, 0.99))
    sharp = sharp.assign(p_NG=1 - sharp["p_GG"])
    off = bt.reliability(sharp, "GG/NG", "GG", bins=10)
    assert ((off["wilson_low"] > off["mean_pred"]) | (off["mean_pred"] > off["wilson_high"])).sum() >= 4
    with pytest.raises(ValueError):
        bt.reliability(frame, "GG/NG", "Over 2.5")
    calibration = bt.selection_calibration(frame, by=None, markets=["GG/NG"])
    assert list(calibration["selection"]) == ["GG", "NG"] and abs(calibration["gap"].iloc[0]) < 0.01


def test_walk_forward_model_beats_the_naive_base_rates(small_league: League, small_result: bt.BacktestResult) -> None:
    overall = bt.metrics(small_result.frame, by=None, markets=["1X2"]).iloc[0]
    assert overall["skill_naive"] > 0, "strong team differences: the model must beat the league base rates"
    # and it is close to the truth: the true probabilities score better than the fitted ones, by a little
    truth = small_league.truth.set_index("match_id").loc[small_result.frame["match_id"]]
    true_probs = bt._market_arrays(truth["lam"].to_numpy(), truth["mu"].to_numpy(), 0.0, 10)
    oracle = small_result.frame.assign(**true_probs)
    oracle_ll = bt.metrics(oracle, by=None, markets=["1X2"]).iloc[0]["log_loss"]
    assert oracle_ll <= overall["log_loss"] + 0.01 and overall["log_loss"] - oracle_ll < 0.08


# ============================================================================ market comparison
def _book(frame: pd.DataFrame, *, prefix: str = "p_", margin: float = 0.06, noise: float = 0.0, seed: int = 0, bookmaker: str = "FootyStats") -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for _, r in frame.iterrows():
        for spec in bt.MARKET_SPECS.values():
            q = np.array([r[c] for c in spec.columns(prefix)], dtype=float) * np.exp(rng.normal(0.0, noise, len(spec.selections)))
            q = np.minimum(q / q.sum() * (1.0 + margin), 0.99)  # a bookmaker never quotes below 1.01
            rows += [{"match_id": r["match_id"], "bookmaker": bookmaker, "market": spec.key, "selection": s, "odds": 1.0 / qq} for s, qq in zip(spec.selections, q)]
    return pd.DataFrame(rows)


def _oracle_frame(n: int, *, seed: int) -> pd.DataFrame:
    """``n`` matches whose ``p_*`` columns are the true probabilities and whose scores are drawn from them."""
    rng = np.random.default_rng(seed)
    lam, mu = rng.uniform(0.6, 2.4, n), rng.uniform(0.4, 1.9, n)
    rows = [{"home_goals": int(h), "away_goals": int(a), "lam": float(lh), "mu": float(la)} for h, a, lh, la in zip(rng.poisson(lam), rng.poisson(mu), lam, mu)]
    frame = _hand_frame(rows)
    return frame.assign(**bt._market_arrays(lam, mu, 0.0, 10))


def test_odds_table_devigs_the_sharpest_book_and_takes_the_best_price() -> None:
    frame = _hand_frame([{"p_1": 0.5, "p_X": 0.3, "p_2": 0.2}])
    soft = _book(frame, margin=0.10, bookmaker="Soft")
    sharp = _book(frame, margin=0.02, bookmaker="Sharp")
    long = pd.concat([soft, sharp], ignore_index=True)
    long.loc[(long["bookmaker"] == "Soft") & (long["selection"] == "X"), "odds"] = 4.0  # an outlier price on X
    incomplete = pd.DataFrame([{"match_id": "M9", "bookmaker": "B", "market": "1X2", "selection": "1", "odds": 2.0}])
    table = bt.odds_table(pd.concat([long, incomplete], ignore_index=True))
    assert list(table["match_id"]) == ["M0"], "an incomplete book is ignored"
    row = table.iloc[0]
    assert row["odds_X"] == 4.0, "best price across bookmakers"
    assert (row["book_1"], row["book_X"], row["book_2"]) == pytest.approx((0.5, 0.3, 0.2)), "probabilities from the lowest over-round book"
    assert row["overround_1X2"] == pytest.approx(0.02)
    # OddsQuote input and the power method work as well
    from src.common.schemas import OddsQuote

    quotes = [OddsQuote("M0", "B", Market.MATCH_ODDS, {"1": 1.9, "X": 3.4, "2": 4.6})]
    power = bt.odds_table(quotes, method="power").iloc[0]
    assert power[["book_1", "book_X", "book_2"]].sum() == pytest.approx(1.0) and math.isnan(power["book_GG"])
    assert bt.odds_table(None).empty and bt.odds_table([]).empty


def test_market_comparison_scores_model_and_market_on_the_same_matches(small_league: League, small_result: bt.BacktestResult) -> None:
    frame = small_result.frame
    truth = small_league.truth.set_index("match_id").loc[frame["match_id"]]
    true_frame = frame.assign(**bt._market_arrays(truth["lam"].to_numpy(), truth["mu"].to_numpy(), 0.0, 10))
    # the bookmaker quotes the truth with a 6% margin on half the matches only
    quoted = true_frame.iloc[::2]
    odds = _book(quoted)
    table = bt.market_comparison(frame, odds, by=None)
    assert set(table["market"]) == {m.value for m in bt.DEFAULT_MARKETS} and (table["n"] == len(quoted)).all()
    assert (table["skill"] < 0.02).all(), "the model cannot beat a market quoting the true probabilities"
    assert table["mean_overround"].to_numpy() == pytest.approx(np.full(len(table), 0.06), abs=0.01)
    # a model equal to the de-vigged market has zero skill; the oracle beats a noisy market
    as_market = bt.market_comparison(quoted, odds, by=None)
    assert as_market["skill"].to_numpy() == pytest.approx(np.zeros(len(as_market)), abs=1e-3), "zero up to the books capped at 1.01"
    oracle = _oracle_frame(3000, seed=8)
    noisy = bt.market_comparison(oracle, _book(oracle, noise=0.3, seed=4), by=("section",), markets=["1X2", "OU2.5"])
    assert (noisy["skill"] > 0).all() and list(noisy.columns)[:2] == ["section", "market"] and (noisy["n"] == 3000).all()
    assert bt.market_comparison(frame, None).empty


# ============================================================================ track record
def test_picks_track_record_counts_hits_roi_and_accumulators() -> None:
    rows = []
    # matchday 1: six matches in two sections; draws on M0, M2; GG on M0, M1, M3
    spec = [
        ("A", 0.34, 1, 1, 0.70),
        ("A", 0.30, 2, 1, 0.65),
        ("A", 0.28, 0, 0, 0.40),
        ("DA", 0.33, 2, 2, 0.60),
        ("DA", 0.25, 1, 0, 0.55),
        ("DA", 0.20, 3, 0, 0.30),
    ]
    for i, (code, px, hg, ag, pgg) in enumerate(spec):
        rows.append(
            {
                "girone": code,
                "section": bt.section_of_girone(code),
                "family": "Serie C" if code == "A" else "Serie D",
                "home_goals": hg,
                "away_goals": ag,
                "p_X": px,
                "p_1": (1 - px) / 2,
                "p_2": (1 - px) / 2,
                "p_GG": pgg,
                "p_NG": 1 - pgg,
            }
        )
    frame = _hand_frame(rows)
    odds = pd.DataFrame(
        [
            {"match_id": f"M{i}", "bookmaker": "B", "market": "1X2", "selection": s, "odds": o}
            for i in range(6)
            for s, o in (("1", 2.5), ("X", 3.0), ("2", 2.8))
        ]
    )
    record = bt.picks_track_record(frame, n_draws=2, n_btts=3, odds_history=odds, accumulator_sizes=(1, 2))
    picks = record.picks
    overall_x = picks[(picks["scope"] == bt.ALL_SECTIONS) & (picks["list"] == "X")]
    assert list(overall_x["match_id"]) == ["M0", "M3"] and list(overall_x["won"]) == [True, True]
    assert list(overall_x["odds"]) == [3.0, 3.0] and list(overall_x["profit"]) == [2.0, 2.0]
    # the ranking is the dashboard's: rank_picks on the same predictions
    expected = [p.match_id for p in rank_picks(bt._predictions(frame, "p_"), market=Market.MATCH_ODDS, selection="X", n=2)]
    assert expected == list(overall_x["match_id"])
    overall_gg = picks[(picks["scope"] == bt.ALL_SECTIONS) & (picks["list"] == "GG")]
    assert list(overall_gg["match_id"]) == ["M0", "M1", "M3"] and overall_gg["won"].all() and overall_gg["odds"].isna().all()
    section_a = picks[(picks["scope"] == bt.section_of_girone("A")) & (picks["list"] == "X")]
    assert list(section_a["match_id"]) == ["M0", "M1"] and list(section_a["won"]) == [True, False]
    summary = record.summary.set_index(["scope", "list"])
    row = summary.loc[(bt.ALL_SECTIONS, "X")]
    assert row["n_picks"] == 2 and row["hits"] == 2 and row["hit_rate"] == 1.0 and row["mean_p"] == pytest.approx(0.335)
    assert row["wilson_high"] == 1.0 and row["wilson_low"] == pytest.approx(bt.wilson_interval(2, 2)[0])
    assert row["roi"] == 2.0 and row["n_with_odds"] == 2 and math.isnan(summary.loc[(bt.ALL_SECTIONS, "GG"), "roi"])
    assert list(record.summary["list"][:2]) == ["X", "GG"] and record.summary["scope"].iloc[0] == bt.ALL_SECTIONS
    acc = record.accumulators.set_index(["scope", "list", "k"])
    assert acc.loc[(bt.ALL_SECTIONS, "X", 2), "wins"] == 1 and acc.loc[(bt.ALL_SECTIONS, "X", 2), "mean_p"] == pytest.approx(0.34 * 0.33)
    assert acc.loc[(bt.ALL_SECTIONS, "X", 2), "profit"] == pytest.approx(3.0 * 3.0 - 1.0)
    assert acc.loc[(bt.section_of_girone("A"), "X", 2), "wins"] == 0
    md = record.by_matchday.set_index(["scope", "list"])
    assert bool(md.loc[(bt.ALL_SECTIONS, "GG"), "all_won"]) and md.loc[(bt.ALL_SECTIONS, "GG"), "expected_hits"] == pytest.approx(0.70 + 0.65 + 0.60)
    empty = bt.picks_track_record(frame.iloc[0:0])
    assert empty.picks.empty and empty.summary.empty and "roi" in empty.summary.columns


def test_slips_track_record_default_and_injected_builders() -> None:
    rows = [
        {"girone": "A", "home_goals": 1, "away_goals": 1, "p_X": 0.33, "p_1": 0.34, "p_2": 0.33, "p_GG": 0.62, "p_NG": 0.38, "p_O25": 0.55, "p_U25": 0.45},
        {"girone": "A", "home_goals": 2, "away_goals": 0, "p_X": 0.22, "p_1": 0.60, "p_2": 0.18, "p_GG": 0.45, "p_NG": 0.55, "p_O25": 0.48, "p_U25": 0.52},
        {"girone": "DA", "home_goals": 0, "away_goals": 3, "p_X": 0.24, "p_1": 0.20, "p_2": 0.56, "p_GG": 0.50, "p_NG": 0.50, "p_O25": 0.66, "p_U25": 0.34},
    ]
    frame = _hand_frame([{**r, "section": bt.section_of_girone(r["girone"])} for r in rows])
    slips = bt.default_slip_builder(frame, n_draws=2, n_btts=2, n_mixed=3)
    assert [s.name for s in slips] == ["X", "GG", "Mista"]
    assert [leg.match_id for leg in slips[0].legs] == ["M0", "M2"] and all(leg.selection == "X" for leg in slips[0].legs)
    mixed = {leg.match_id: (leg.selection, leg.p) for leg in slips[2].legs}
    assert mixed == {"M0": ("GG", 0.62), "M1": ("1", 0.60), "M2": ("Over 2.5", 0.66)}
    assert slips[2].p == pytest.approx(0.62 * 0.60 * 0.66) and len({leg.match_id for leg in slips[2].legs}) == 3
    odds = pd.DataFrame(
        [{"match_id": m, "bookmaker": "B", "market": "1X2", "selection": s, "odds": o} for m in ("M0", "M1", "M2") for s, o in (("1", 2.0), ("X", 3.2), ("2", 3.5))]
    )
    record = bt.slips_track_record(frame, odds_history=odds, slip_builder=lambda pre: bt.default_slip_builder(pre, n_draws=2, n_btts=2, n_mixed=3))
    by_name = record.slips.set_index("slip")
    assert not by_name.loc["X", "won"] and by_name.loc["X", "legs_won"] == 1 and by_name.loc["X", "book_odds"] == pytest.approx(3.2 * 3.2)
    assert by_name.loc["X", "profit"] == -1.0 and by_name.loc["X", "expected_legs"] == pytest.approx(0.33 + 0.24)
    assert bool(by_name.loc["Mista", "won"]) and math.isnan(by_name.loc["Mista", "book_odds"]), "GG and Over 2.5 not quoted"
    assert record.legs.groupby("slip").size().to_dict() == {"GG": 2, "Mista": 3, "X": 2}
    summary = record.summary.set_index("slip")
    assert summary.loc["Mista", "wins"] == 1 and summary.loc["X", "leg_hit_rate"] == 0.5 and summary.loc["X", "n_with_odds"] == 1
    # an injected builder sees no result and can use mappings; a leg outside the matchday skips its slip
    seen_columns: list[set[str]] = []

    def builder(pre: pd.DataFrame) -> list[dict[str, Any]]:
        seen_columns.append(set(pre.columns))
        return [
            {"name": "Mia", "legs": [{"match_id": "M1", "market": "1X2", "selection": "1", "p": 0.6}]},
            {"name": "Rotta", "legs": [{"match_id": "ALTRO", "market": "1X2", "selection": "1", "p": 0.6}]},
        ]

    custom = bt.slips_track_record(frame, slip_builder=builder)
    assert not seen_columns[0] & {"home_goals", "away_goals", "res_1X2", "gg", "o15", "o25", "o35"}, "no outcome reaches the builder"
    assert {"p_X", "p_GG", "lam", "section"} <= seen_columns[0]
    assert list(custom.slips["slip"]) == ["Mia"] and bool(custom.slips["won"].iloc[0]) and math.isnan(custom.slips["profit"].iloc[0])


def test_track_records_never_show_a_zero_width_roi_interval_for_losing_slips_and_accumulators() -> None:
    """Regression: 20 matchdays of X slips and X accumulators, none won, showed 'IC 95% ROI -100.0 - -100.0'."""
    rows = []
    for day in range(20):
        matchday = (pd.Timestamp(2026, 9, 4) + pd.Timedelta(days=7 * day)).date().isoformat()
        for i in range(3):
            rows.append(
                {
                    "match_id": f"D{day}M{i}",
                    "matchday": matchday,
                    "date": pd.Timestamp(matchday) + pd.Timedelta(days=2),
                    "home_goals": 2,
                    "away_goals": 0,  # no draw ever: every X slip and accumulator loses
                    "p_1": 0.40,
                    "p_X": 0.31 - 0.01 * i,
                    "p_2": 0.29 + 0.01 * i,
                }
            )
    frame = _hand_frame(rows)
    odds = pd.DataFrame([{"match_id": r["match_id"], "bookmaker": "B", "market": "1X2", "selection": s, "odds": o} for r in rows for s, o in (("1", 2.3), ("X", 3.2), ("2", 3.4))])
    record = bt.slips_track_record(frame, odds_history=odds, slip_builder=lambda pre: bt.default_slip_builder(pre, n_draws=2, n_btts=0, n_mixed=0))
    x = record.summary.set_index("slip").loc["X"]
    assert x["n_slips"] == 20 and x["wins"] == 0 and x["n_with_odds"] == 20 and x["roi"] == -1.0
    expected_high = 3.2 * 3.2 * bt.exact_binomial_interval(0, 20)[1] - 1.0
    assert x["roi_low"] == -1.0 and x["roi_high"] == pytest.approx(expected_high) and x["roi_high"] > 0.0, "20 lost slips cannot exclude a profit"
    picks = bt.picks_track_record(frame, n_draws=2, n_btts=1, odds_history=odds, by_section=False, accumulator_sizes=(2,))
    acc = picks.accumulators.set_index(["list", "k"]).loc[("X", 2)]
    assert acc["wins"] == 0 and acc["roi_low"] == -1.0 and acc["roi_high"] == pytest.approx(expected_high)
    singles = picks.summary.set_index("list").loc["X"]
    assert singles["hits"] == 0 and singles["roi_low"] == -1.0 and singles["roi_high"] == pytest.approx(3.2 * bt.exact_binomial_interval(0, 40)[1] - 1.0)


def test_value_track_record_flags_positive_ev_at_the_best_price() -> None:
    frame = _hand_frame(
        [
            {"home_goals": 2, "away_goals": 1, "p_1": 0.55, "p_X": 0.25, "p_2": 0.20},
            {"home_goals": 0, "away_goals": 0, "p_1": 0.40, "p_X": 0.30, "p_2": 0.30},
        ]
    )
    odds = pd.DataFrame(
        [
            {"match_id": "M0", "bookmaker": "B", "market": "1X2", "selection": s, "odds": o}
            for s, o in (("1", 2.0), ("X", 3.5), ("2", 4.5))  # EV(1) = 0.10, EV(X) = -0.125, EV(2) = -0.10
        ]
        + [{"match_id": "M1", "bookmaker": "B", "market": "1X2", "selection": s, "odds": o} for s, o in (("1", 2.4), ("X", 3.0), ("2", 3.1))]
    )
    record = bt.value_track_record(frame, odds, min_ev=0.035, markets=["1X2"])
    bets = record.bets
    assert list(zip(bets["match_id"], bets["selection"])) == [("M0", "1")]
    assert bets["ev"].iloc[0] == pytest.approx(0.10) and bool(bets["won"].iloc[0]) and bets["profit"].iloc[0] == pytest.approx(1.0)
    total = record.summary.set_index(["scope", "market"]).loc[(bt.ALL_SECTIONS, "Tutti")]
    assert total["n_bets"] == 1 and total["roi"] == pytest.approx(1.0) and math.isnan(total["roi_low"])
    assert bt.value_track_record(frame, None).bets.empty
