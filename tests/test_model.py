"""Tests for the Dixon-Coles engine (``src/engine/model.py``).

All fits use synthetic data with known true parameters from
``src.simulation.generator``. Module-scoped fixtures share the expensive fits
so the whole file runs in a few seconds.
"""
from __future__ import annotations

import logging
import math
from dataclasses import replace
from datetime import date, timedelta

import numpy as np
import pandas as pd
import pytest
from scipy.optimize import check_grad
from scipy.stats import poisson

from src.common.schemas import SELECTIONS, Fixture, Girone, Market, MatchPrediction
from src.engine.model import (
    DixonColesConfig,
    DixonColesEngine,
    FittedParams,
    LikelihoodData,
    NotFittedError,
    TeamIndex,
    build_team_index,
    dixon_coles_pmf,
    markets_from_matrix,
    negative_log_likelihood,
    negative_log_likelihood_and_grad,
    negative_log_likelihood_grad,
    pack_theta,
    prepare_likelihood_data,
    tau,
    time_weights,
    unpack_theta,
)
from src.simulation.generator import dixon_coles_matrix, simulate_league, simulate_mini_dataset

RANKING_COLUMNS = [
    "team",
    "girone",
    "alpha",
    "beta",
    "gamma",
    "attack_index",
    "defence_index",
    "xg_for_vs_avg",
    "xg_against_vs_avg",
]


# =============================================================================
# Fixtures (module scoped: the fits are shared)
# =============================================================================
@pytest.fixture(scope="module")
def full_league():
    return simulate_league(seed=3)


@pytest.fixture(scope="module")
def full_fit(full_league):
    engine = DixonColesEngine(DixonColesConfig(xi=0.0))
    params = engine.fit(full_league.matches)
    return engine, params


@pytest.fixture(scope="module")
def multi_season(full_league):
    """Three independent seasons generated from the seed-3 ``TrueParams`` (~3330 matches).

    ``match_id`` is prefixed per season because ``validate_matches_df`` keeps the
    last duplicate id; the seasons are dated consecutively.
    """
    frames = []
    for k, seed in enumerate((300, 301, 302)):
        season = simulate_league(seed=seed, true_params=full_league.true_params, start_date=date(2022 + k, 8, 24), with_odds=False)
        frames.append(season.matches.assign(match_id=f"S{k}-" + season.matches["match_id"]))
    return pd.concat(frames, ignore_index=True)


@pytest.fixture(scope="module")
def league12():
    return simulate_league(seed=5, n_rounds=12)


@pytest.fixture(scope="module")
def fit12_slsqp(league12):
    engine = DixonColesEngine(DixonColesConfig(xi=0.0, method="SLSQP"))
    return engine, engine.fit(league12.matches)


@pytest.fixture(scope="module")
def fit12_lbfgsb(league12):
    engine = DixonColesEngine(DixonColesConfig(xi=0.0, method="L-BFGS-B"))
    return engine, engine.fit(league12.matches)


@pytest.fixture(scope="module")
def mini():
    return simulate_mini_dataset()


def _small_problem(seed: int = 0, n_matches: int = 30) -> tuple[pd.DataFrame, dict[str, list[str]]]:
    """6 teams in 2 gironi, random scores and dates."""
    rng = np.random.default_rng(seed)
    teams = {"A": ["Alpha", "Bravo", "Charlie"], "B": ["Delta", "Echo", "Foxtrot"]}
    start = pd.Timestamp("2025-09-01")
    rows = []
    for i in range(n_matches):
        g = str(rng.choice(["A", "B"]))
        home, away = rng.choice(teams[g], 2, replace=False)
        rows.append(
            {
                "match_id": f"S{i:03d}",
                "date": start + pd.Timedelta(days=int(rng.integers(0, 90))),
                "round": 1 + i // 3,
                "girone": g,
                "home_team": str(home),
                "away_team": str(away),
                "home_goals": int(rng.poisson(1.4)),
                "away_goals": int(rng.poisson(1.0)),
            }
        )
    return pd.DataFrame(rows), teams


def _true_vectors(params: FittedParams, true_params, girone: Girone | None = None):
    teams = params.teams_by_girone[girone] if girone is not None else params.teams
    a_hat = np.array([params.alpha[t] for t in teams])
    a_true = np.array([true_params.alpha[t] for t in teams])
    b_hat = np.array([params.beta[t] for t in teams])
    b_true = np.array([true_params.beta[t] for t in teams])
    return a_hat, a_true, b_hat, b_true


def _corr(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.corrcoef(a, b)[0, 1])


# =============================================================================
# tau / pmf / markets
# =============================================================================
def test_tau_special_cases():
    lam, mu, rho = 1.4, 0.9, -0.07
    assert tau(0, 0, lam, mu, rho) == pytest.approx(1.0 - lam * mu * rho)
    assert tau(0, 1, lam, mu, rho) == pytest.approx(1.0 + lam * rho)
    assert tau(1, 0, lam, mu, rho) == pytest.approx(1.0 + mu * rho)
    assert tau(1, 1, lam, mu, rho) == pytest.approx(1.0 - rho)
    assert tau(2, 0, lam, mu, rho) == pytest.approx(1.0)
    assert tau(0, 2, lam, mu, rho) == pytest.approx(1.0)
    assert isinstance(tau(0, 0, lam, mu, rho), float)


def test_tau_vectorised_and_broadcasting():
    x = np.array([0, 0, 1, 1, 2, 0, 3])
    y = np.array([0, 1, 0, 1, 0, 2, 3])
    lam = np.full(7, 1.2)
    mu = np.linspace(0.5, 1.5, 7)
    rho = -0.05
    out = tau(x, y, lam, mu, rho)
    assert isinstance(out, np.ndarray) and out.shape == (7,)
    expected = np.array(
        [1 - 1.2 * mu[0] * rho, 1 + 1.2 * rho, 1 + mu[2] * rho, 1 - rho, 1.0, 1.0, 1.0]
    )
    np.testing.assert_allclose(out, expected)
    # scalar rates broadcast against a score grid
    grid = tau(*np.meshgrid(np.arange(3), np.arange(3), indexing="ij"), 1.0, 1.0, 0.1)
    assert grid.shape == (3, 3) and grid[2, 2] == 1.0 and grid[1, 1] == pytest.approx(0.9)


@pytest.mark.parametrize("lam,mu,rho", [(1.3, 1.0, -0.06), (0.4, 2.2, 0.12), (2.5, 0.3, -0.18), (1.0, 1.0, 0.0)])
def test_dixon_coles_pmf_is_a_distribution_and_matches_generator(lam, mu, rho):
    m = dixon_coles_pmf(lam, mu, rho, max_goals=10)
    assert m.shape == (11, 11)
    assert np.all(m >= 0.0)
    assert m.sum() == pytest.approx(1.0, abs=1e-12)
    np.testing.assert_allclose(m, dixon_coles_matrix(lam, mu, rho, max_goals=10), rtol=1e-10, atol=1e-14)


def test_dixon_coles_pmf_rho_zero_is_product_of_poissons():
    lam, mu = 1.7, 0.8
    ks = np.arange(8)
    product = np.outer(poisson.pmf(ks, lam), poisson.pmf(ks, mu))
    product /= product.sum()
    np.testing.assert_allclose(dixon_coles_pmf(lam, mu, 0.0, max_goals=7), product, rtol=1e-10)


def test_dixon_coles_pmf_rejects_bad_inputs():
    with pytest.raises(ValueError):
        dixon_coles_pmf(1.0, 1.0, 0.0, max_goals=0)
    with pytest.raises(ValueError):
        dixon_coles_pmf(-1.0, 1.0, 0.0)


def test_markets_from_matrix_hand_computed():
    m = np.array([[0.10, 0.10, 0.05], [0.20, 0.15, 0.05], [0.15, 0.10, 0.10]])
    assert m.sum() == pytest.approx(1.0)
    probs = markets_from_matrix(m)
    assert set(probs) == set(SELECTIONS)
    for market, sels in probs.items():
        assert tuple(sels) == SELECTIONS[market]
        assert sum(sels.values()) == pytest.approx(1.0, abs=1e-12)
    assert probs[Market.MATCH_ODDS]["1"] == pytest.approx(0.45)
    assert probs[Market.MATCH_ODDS]["X"] == pytest.approx(0.35)
    assert probs[Market.MATCH_ODDS]["2"] == pytest.approx(0.20)
    assert probs[Market.OVER_UNDER_15]["Over 1.5"] == pytest.approx(0.60)
    assert probs[Market.OVER_UNDER_15]["Under 1.5"] == pytest.approx(0.40)
    assert probs[Market.OVER_UNDER_25]["Over 2.5"] == pytest.approx(0.25)
    assert probs[Market.OVER_UNDER_35]["Over 3.5"] == pytest.approx(0.10)
    assert probs[Market.BTTS]["GG"] == pytest.approx(0.40)
    assert probs[Market.BTTS]["NG"] == pytest.approx(0.60)


def test_markets_from_matrix_sums_to_one_for_model_matrix():
    probs = markets_from_matrix(dixon_coles_pmf(1.6, 1.1, -0.08))
    for market in Market:
        assert sum(probs[market].values()) == pytest.approx(1.0, abs=1e-12)
    with pytest.raises(ValueError):
        markets_from_matrix(np.ones((2, 3)))


# =============================================================================
# time weights
# =============================================================================
def test_time_weights_behaviour():
    ref = date(2026, 1, 31)
    dates = pd.Series(pd.to_datetime([ref - timedelta(days=d) for d in (0, 1, 7, 30, 120)]))
    ones = time_weights(dates, ref, xi=0.0)
    np.testing.assert_array_equal(ones, np.ones(5))
    w = time_weights(dates, ref, xi=0.0018)
    assert w[0] == pytest.approx(1.0)
    assert np.all(np.diff(w) < 0), "weights must strictly decrease with age"
    np.testing.assert_allclose(w, np.exp(-0.0018 * np.array([0, 1, 7, 30, 120])))
    # Matches after the reference date are clipped at age 0 -> weight 1 (documented behaviour)
    future = time_weights(pd.Series(pd.to_datetime([ref + timedelta(days=10)])), ref, xi=0.01)
    assert future[0] == pytest.approx(1.0)
    # numpy datetime64 input and plain lists are accepted too
    arr = np.array(["2026-01-31", "2026-01-01"], dtype="datetime64[D]")
    np.testing.assert_allclose(time_weights(arr, ref, 0.01), [1.0, math.exp(-0.3)])
    np.testing.assert_allclose(time_weights([ref, ref - timedelta(days=100)], ref, 0.01), [1.0, math.exp(-1.0)])
    with pytest.raises(ValueError):
        time_weights(dates, ref, xi=-0.1)


# =============================================================================
# objective + analytic gradient
# =============================================================================
def test_pack_unpack_roundtrip():
    alpha, beta, gamma, rho = np.arange(4.0), -np.arange(4.0), np.array([0.3, 0.2]), -0.05
    theta = pack_theta(alpha, beta, gamma, rho)
    assert theta.shape == (11,)
    a, b, g, r = unpack_theta(theta, 4, 2)
    np.testing.assert_array_equal(a, alpha)
    np.testing.assert_array_equal(b, beta)
    np.testing.assert_array_equal(g, gamma)
    assert r == rho
    with pytest.raises(ValueError):
        unpack_theta(theta, 5, 2)


def test_gradient_matches_central_differences():
    df, _ = _small_problem(seed=11, n_matches=30)
    data, index = prepare_likelihood_data(df, xi=0.01, l2_penalty=0.1)
    assert isinstance(data, LikelihoodData) and isinstance(index, TeamIndex)
    assert data.n_teams == 6 and data.n_gammas == 2 and len(data.x) == 30
    rng = np.random.default_rng(1)
    theta = rng.normal(0.0, 0.35, 2 * data.n_teams + data.n_gammas + 1)
    theta[-1] = -0.05  # rho

    analytic = negative_log_likelihood_grad(theta, *data)
    eps = 1e-5
    eye = np.eye(theta.size)
    central = np.array(
        [
            (negative_log_likelihood(theta + eps * eye[k], *data) - negative_log_likelihood(theta - eps * eye[k], *data))
            / (2.0 * eps)
            for k in range(theta.size)
        ]
    )
    rel_err = np.linalg.norm(analytic - central) / np.linalg.norm(central)
    assert rel_err < 1e-5, rel_err
    # the rho component in particular (tau partials)
    assert analytic[-1] == pytest.approx(central[-1], rel=1e-6, abs=1e-8)
    # scipy's forward-difference check on the same point
    assert check_grad(negative_log_likelihood, negative_log_likelihood_grad, theta, *data) < 1e-3
    # the fused function agrees with the separate ones and accepts the NamedTuple directly
    f, g = negative_log_likelihood_and_grad(theta, data)
    assert f == pytest.approx(negative_log_likelihood(theta, data))
    np.testing.assert_allclose(g, analytic)
    # penalty is really in the objective (betas are centred within their girone)
    plain, _ = prepare_likelihood_data(df, xi=0.01, l2_penalty=0.0)
    alpha, beta, _, _ = unpack_theta(theta, data.n_teams, data.n_gammas)
    gidx = data.team_girone_idx
    beta_dev = beta - (np.bincount(gidx, weights=beta) / np.bincount(gidx))[gidx]
    assert negative_log_likelihood(theta, data) - negative_log_likelihood(theta, plain) == pytest.approx(
        0.1 * (alpha @ alpha + beta_dev @ beta_dev)
    )


def test_gradient_with_shared_home_advantage():
    df, _ = _small_problem(seed=5, n_matches=40)
    data, _ = prepare_likelihood_data(df, xi=0.0, l2_penalty=0.05, girone_specific_home=False)
    assert data.n_gammas == 1
    theta = np.random.default_rng(2).normal(0.0, 0.3, 2 * data.n_teams + 2)
    theta[-1] = 0.08
    eps = 1e-5
    eye = np.eye(theta.size)
    central = np.array(
        [
            (negative_log_likelihood(theta + eps * eye[k], *data) - negative_log_likelihood(theta - eps * eye[k], *data))
            / (2.0 * eps)
            for k in range(theta.size)
        ]
    )
    analytic = negative_log_likelihood_grad(theta, *data)
    assert np.linalg.norm(analytic - central) / np.linalg.norm(central) < 1e-5


# =============================================================================
# parameter recovery on the full 2025/26 composition
# =============================================================================
def test_parameter_recovery_full_season(full_league, full_fit):
    """Recovery of the generator's true parameters from ~1110 stationary matches.

    Alpha is compared per girone against the spec threshold (0.85). For beta the
    0.80 threshold is asserted on the pooled 60-team vector: with 20 teams and
    ~37 matches each, the per-girone beta correlation of the exact MLE swings
    between ~0.6 and ~0.95 across seeds purely from sampling noise, and for
    seed=3 girone C the *true optimum* sits at 0.77 (verified independently with
    a from-scratch MLE that agrees with the engine to |delta| < 3e-4, and
    confirmed below: the fitted log-likelihood exceeds the one at the true
    parameters and the gradient is flat). The spec's per-girone beta threshold
    is therefore unattainable on this single season by any correct estimator; a
    per-girone floor of 0.70 guards against gross errors here, and the spec's
    per-girone thresholds are asserted on a larger sample of the same
    ``TrueParams`` in :func:`test_parameter_recovery_multi_season_meets_spec_thresholds`.
    """
    engine, params = full_fit
    tp = full_league.true_params
    print(f"\nfull-season fit: {params.fit_seconds:.2f}s, {params.n_iter} iterations, LL={params.log_likelihood:.2f}")

    assert params.n_matches == len(full_league.matches) and params.n_matches > 1000
    assert params.converged, params.message
    assert params.fit_seconds < 60
    assert params.effective_n == pytest.approx(params.n_matches)  # xi = 0
    assert params.penalty == 0.0

    for g in Girone:
        a_hat, a_true, b_hat, b_true = _true_vectors(params, tp, g)
        assert len(a_hat) == 20
        assert _corr(a_hat, a_true) > 0.85, (g, _corr(a_hat, a_true))
        assert _corr(b_hat, b_true) > 0.70, (g, _corr(b_hat, b_true))
        assert abs(params.gamma[g] - tp.gamma[g]) < 0.12, (g, params.gamma[g], tp.gamma[g])
        assert abs(a_hat.sum()) < 1e-6
    a_hat, a_true, b_hat, b_true = _true_vectors(params, tp)
    assert _corr(a_hat, a_true) > 0.85
    assert _corr(b_hat, b_true) > 0.80, _corr(b_hat, b_true)
    assert abs(params.rho - tp.rho) < 0.10

    # The optimum is a true maximum-likelihood point: better than the truth, flat gradient.
    data, index = prepare_likelihood_data(full_league.matches, xi=0.0)
    theta_true = pack_theta(
        [tp.alpha[t] for t in index.teams], [tp.beta[t] for t in index.teams], [tp.gamma[g] for g in index.gironi], tp.rho
    )
    theta_hat = pack_theta(
        [params.alpha[t] for t in index.teams],
        [params.beta[t] for t in index.teams],
        [params.gamma[g] for g in index.gironi],
        params.rho,
    )
    assert -negative_log_likelihood(theta_hat, data) >= -negative_log_likelihood(theta_true, data)
    assert -negative_log_likelihood(theta_hat, data) == pytest.approx(params.log_likelihood)
    assert np.max(np.abs(negative_log_likelihood_grad(theta_hat, data)[2 * index.n_teams :])) < 1e-2


def test_parameter_recovery_multi_season_meets_spec_thresholds(full_league, full_fit, multi_season):
    """On ~3330 matches from the seed-3 ``TrueParams`` the spec's per-girone
    thresholds (corr(alpha) > 0.85, corr(beta) > 0.80) hold with a wide margin
    (observed >= 0.93 for every girone and seed triplet tried), and every error
    measure shrinks relative to the single-season fit (consistency)."""
    _, single = full_fit
    tp = full_league.true_params
    params = DixonColesEngine(DixonColesConfig(xi=0.0)).fit(multi_season)
    print(f"\nmulti-season fit: {params.fit_seconds:.2f}s, {params.n_iter} iterations, LL={params.log_likelihood:.2f}")

    assert params.n_matches == len(multi_season) == 3 * single.n_matches
    assert params.converged, params.message
    assert params.fit_seconds < 60
    assert params.teams_by_girone == single.teams_by_girone
    for g in Girone:
        a_hat, a_true, b_hat, b_true = _true_vectors(params, tp, g)
        assert _corr(a_hat, a_true) > 0.85, (g, _corr(a_hat, a_true))
        assert _corr(b_hat, b_true) > 0.80, (g, _corr(b_hat, b_true))
        assert abs(params.gamma[g] - tp.gamma[g]) < 0.12, (g, params.gamma[g], tp.gamma[g])
        assert abs(a_hat.sum()) < 1e-6
    assert abs(params.rho - tp.rho) < 0.10

    def _rmse(fit: FittedParams) -> tuple[float, float]:
        a_hat, a_true, b_hat, b_true = _true_vectors(fit, tp)
        return float(np.sqrt(np.mean((a_hat - a_true) ** 2))), float(np.sqrt(np.mean((b_hat - b_true) ** 2)))

    rmse_a3, rmse_b3 = _rmse(params)
    rmse_a1, rmse_b1 = _rmse(single)
    assert rmse_a3 < rmse_a1 and rmse_b3 < rmse_b1, (rmse_a3, rmse_a1, rmse_b3, rmse_b1)
    assert rmse_a3 < 0.12 and rmse_b3 < 0.12


def test_slsqp_and_lbfgsb_agree(fit12_slsqp, fit12_lbfgsb):
    _, p1 = fit12_slsqp
    _, p2 = fit12_lbfgsb
    assert p1.converged and p2.converged
    assert p1.method == "SLSQP" and p2.method == "L-BFGS-B"
    assert "re-solved with SLSQP" not in p2.message  # no bound violation -> no fallback on clean data
    rel = abs(p1.log_likelihood - p2.log_likelihood) / abs(p1.log_likelihood)
    assert rel < 1e-3, rel
    a1 = np.array([p1.alpha[t] for t in p1.teams])
    a2 = np.array([p2.alpha[t] for t in p1.teams])
    np.testing.assert_allclose(a1, a2, atol=0.02)
    for p in (p1, p2):
        for g, teams in p.teams_by_girone.items():
            assert abs(sum(p.alpha[t] for t in teams)) < 1e-6, (p.method, g)
    assert set(p1.alpha) == set(p2.alpha) == set(p1.teams)
    assert p1.teams_by_girone == p2.teams_by_girone


# =============================================================================
# mini dataset, predictions
# =============================================================================
@pytest.mark.parametrize("method", ["SLSQP", "L-BFGS-B"])
def test_mini_dataset_fit_and_predict(mini, method):
    engine = DixonColesEngine(DixonColesConfig(l2_penalty=0.1, method=method))
    params = engine.fit(mini.matches)
    assert params.n_matches == 20 and len(params.teams) == 14
    assert params.penalty > 0.0
    assert np.isfinite(params.log_likelihood)
    preds = engine.predict_fixtures(mini.fixtures)
    assert len(preds) == len(mini.fixtures) > 0
    for pred, fx in zip(preds, mini.fixtures):
        assert isinstance(pred, MatchPrediction)
        assert pred.fixture is fx and pred.match_id == fx.match_id
        assert pred.matrix.shape == (engine.config.max_goals + 1, engine.config.max_goals + 1)
        assert pred.matrix.sum() == pytest.approx(1.0)
        assert pred.rho == params.rho
        lam, mu = engine.rates(fx.home_team, fx.away_team, fx.girone)
        assert pred.home_rate == lam and pred.away_rate == mu and lam > 0 and mu > 0
        for market in Market:
            assert tuple(pred.probabilities[market]) == SELECTIONS[market]
            assert sum(pred.probabilities[market].values()) == pytest.approx(1.0, abs=1e-9)
            for sel, p in pred.probabilities[market].items():
                assert pred.fair_odds[market][sel] == (1.0 / p if p > 0 else float("inf"))
        np.testing.assert_allclose(pred.matrix, engine.score_matrix(fx.home_team, fx.away_team, fx.girone))
        assert pred.probabilities == engine.market_probabilities(fx.home_team, fx.away_team, fx.girone)
        assert pred.fair_odds == engine.fair_odds(fx.home_team, fx.away_team, fx.girone)


def test_fair_odds_inf_when_probability_zero(fit12_slsqp):
    engine, _ = fit12_slsqp
    probs = {Market.BTTS: {"GG": 1.0, "NG": 0.0}}
    odds = engine._fair_odds_from(probs)
    assert odds[Market.BTTS]["GG"] == 1.0 and odds[Market.BTTS]["NG"] == float("inf")


def test_girone_accepts_str_or_enum(fit12_slsqp):
    engine, params = fit12_slsqp
    home, away = params.teams_by_girone[Girone.B][:2]
    assert engine.rates(home, away, "B") == engine.rates(home, away, Girone.B) == engine.rates(home, away, "Girone B")


# =============================================================================
# home advantage options, warm start, time decay
# =============================================================================
def test_shared_home_advantage_is_identical_across_gironi(league12):
    engine = DixonColesEngine(DixonColesConfig(xi=0.0, girone_specific_home=False))
    params = engine.fit(league12.matches)
    assert params.converged
    assert set(params.gamma) == set(Girone)
    values = list(params.gamma.values())
    assert all(v == values[0] for v in values)
    assert -1.0 <= values[0] <= 1.5
    # the shared gamma lies between the girone-specific ones of a specific fit
    specific = DixonColesEngine(DixonColesConfig(xi=0.0)).fit(league12.matches)
    assert min(specific.gamma.values()) - 0.05 <= values[0] <= max(specific.gamma.values()) + 0.05


def test_warm_start_converges_and_mismatch_is_ignored(league12, fit12_slsqp, mini, caplog):
    engine_cold, cold = fit12_slsqp
    engine = DixonColesEngine(engine_cold.config)
    warm = engine.fit(league12.matches, warm_start=cold)
    assert warm.converged
    assert warm.log_likelihood == pytest.approx(cold.log_likelihood, abs=1e-4)
    assert warm.n_iter <= cold.n_iter
    np.testing.assert_allclose([warm.alpha[t] for t in cold.teams], [cold.alpha[t] for t in cold.teams], atol=1e-3)

    other = DixonColesEngine(DixonColesConfig(l2_penalty=0.1)).fit(mini.matches)  # different team set
    with caplog.at_level(logging.WARNING, logger="src.engine.model"):
        again = DixonColesEngine(engine_cold.config).fit(league12.matches, warm_start=other)
    assert again.converged
    assert any("warm_start ignored" in rec.getMessage() for rec in caplog.records)
    assert again.log_likelihood == pytest.approx(cold.log_likelihood, abs=1e-4)


def test_time_decay_reduces_effective_sample_size(league12):
    engine = DixonColesEngine(DixonColesConfig(xi=0.0018))
    params = engine.fit(league12.matches)
    assert params.xi == 0.0018
    assert params.n_matches == len(league12.matches)
    assert params.effective_n < params.n_matches
    assert params.reference_date == league12.matches["date"].max().date()
    # an explicit reference date far in the future shrinks the weights further
    later = DixonColesEngine(DixonColesConfig(xi=0.0018)).fit(league12.matches, reference_date=date(2026, 12, 31))
    assert later.reference_date == date(2026, 12, 31)
    assert later.effective_n < params.effective_n
    # the stationary fit has effective_n == n_matches
    flat = DixonColesEngine(DixonColesConfig(xi=0.0)).fit(league12.matches)
    assert flat.effective_n == pytest.approx(flat.n_matches)


# =============================================================================
# persistence, ranking, residuals, errors
# =============================================================================
def test_save_load_roundtrip(tmp_path, fit12_slsqp, league12):
    engine, params = fit12_slsqp
    path = engine.save(tmp_path / "nested" / "engine.json")
    assert path.exists()
    loaded = DixonColesEngine.load(path)
    assert loaded.params_ == params
    assert loaded.config == engine.config
    assert isinstance(loaded.params_.reference_date, date)
    assert all(isinstance(k, Girone) for k in loaded.params_.gamma)
    assert all(isinstance(v, tuple) for v in loaded.params_.teams_by_girone.values())
    fx = league12.fixtures[0]
    a, b = engine.predict(fx), loaded.predict(fx)
    np.testing.assert_array_equal(a.matrix, b.matrix)
    assert a.probabilities == b.probabilities and a.fair_odds == b.fair_odds

    # FittedParams alone round-trips too and an engine can be rebuilt from it
    p_path = params.to_json(tmp_path / "params.json")
    assert FittedParams.from_json(p_path) == params
    assert FittedParams.from_json(path) == params  # accepts the engine payload as well
    rebuilt = DixonColesEngine.load(p_path)
    assert rebuilt.params_ == params and rebuilt.config.method == params.method


def test_to_frame(fit12_slsqp):
    _, params = fit12_slsqp
    frame = params.to_frame()
    assert list(frame.columns) == ["team", "girone", "alpha", "beta", "gamma"]
    assert len(frame) == len(params.teams)
    assert str(frame["girone"].dtype) == "string" and str(frame["team"].dtype) == "string"
    row = frame.iloc[0]
    assert params.alpha[row["team"]] == row["alpha"] and params.gamma[Girone(row["girone"])] == row["gamma"]


def test_ranking(fit12_slsqp):
    engine, params = fit12_slsqp
    table = engine.ranking("A")
    assert list(table.columns) == RANKING_COLUMNS
    assert len(table) == len(params.teams_by_girone[Girone.A])
    assert list(table.index) == list(range(len(table)))
    assert set(table["girone"]) == {"A"}
    alphas = table["alpha"].to_numpy()
    assert np.all(np.diff(alphas) <= 0)
    for i in range(len(table) - 1):
        if alphas[i] == alphas[i + 1]:
            assert table["beta"].iloc[i] >= table["beta"].iloc[i + 1]
    np.testing.assert_allclose(table["attack_index"], np.exp(alphas))
    np.testing.assert_allclose(table["defence_index"], np.exp(table["beta"]))
    mean_exp_neg_beta = np.mean(np.exp(-table["beta"].to_numpy()))
    mean_exp_alpha = np.mean(np.exp(alphas))
    np.testing.assert_allclose(table["xg_for_vs_avg"], np.exp(alphas) * mean_exp_neg_beta)
    np.testing.assert_allclose(table["xg_against_vs_avg"], np.exp(-table["beta"].to_numpy()) * mean_exp_alpha)
    assert set(table["gamma"]) == {params.gamma[Girone.A]}

    everything = engine.ranking()
    assert len(everything) == len(params.teams)
    assert list(everything.columns) == RANKING_COLUMNS
    assert np.all(np.diff(everything["alpha"].to_numpy()) <= 0)
    # per-girone means are used even in the pooled table
    sub_c = everything.loc[everything["girone"] == "C"].sort_values("team").reset_index(drop=True)
    only_c = engine.ranking(Girone.C).sort_values("team").reset_index(drop=True)
    pd.testing.assert_frame_equal(sub_c, only_c)
    assert engine.ranking(Girone.B).equals(engine.ranking("girone b"))


def test_residuals(fit12_slsqp, league12):
    engine, params = fit12_slsqp
    res = engine.residuals(league12.matches)
    assert len(res) == params.n_matches
    for c in ("match_id", "lam", "mu", "home_goals", "away_goals", "home_residual", "away_residual", "weight", "log_score_prob"):
        assert c in res.columns
    assert np.all(res["lam"] > 0) and np.all(res["mu"] > 0)
    np.testing.assert_allclose(res["home_residual"], res["home_goals"] - res["lam"])
    np.testing.assert_allclose(res["away_residual"], res["away_goals"] - res["mu"])
    np.testing.assert_allclose(res["weight"], 1.0)  # xi = 0
    assert np.all(res["log_score_prob"] <= 0)
    # A maximum-likelihood fit reproduces the total goals closely (the tau term keeps
    # the Poisson score equations from balancing exactly when rho != 0).
    total_goals = float(res["actual_total"].sum())
    assert abs(res["total_residual"].sum()) < 0.01 * total_goals
    np.testing.assert_allclose(res["expected_total"], res["lam"] + res["mu"])
    # unknown teams give NaN rates and a warning, never an exception
    foreign = league12.matches.iloc[[0]].assign(home_team="Nowhere FC")
    nan_rows = engine.residuals(foreign)
    assert len(nan_rows) == 1 and np.isnan(nan_rows["lam"].iloc[0])


def test_unknown_team_and_girone_raise_key_error(fit12_slsqp, league12):
    engine, params = fit12_slsqp
    known = params.teams_by_girone[Girone.A][0]
    with pytest.raises(KeyError, match="unknown team 'Vicenz'") as exc:
        engine.rates("Vicenz", known, "A")
    assert "did you mean" in str(exc.value)
    with pytest.raises(KeyError):
        engine.score_matrix(known, "Nowhere FC", Girone.A)
    with pytest.raises(KeyError):
        params.girone_of("Nowhere FC")
    with pytest.raises(ValueError):
        engine.rates(known, known, "A")
    with pytest.raises(KeyError):
        engine.predict(Fixture("x", date(2026, 1, 1), 1, "A", "Nowhere FC", known))
    # a girone without a home-advantage estimate is reported, not silently defaulted
    girone_a_only = league12.matches.loc[league12.matches["girone"] == "A"].reset_index(drop=True)
    only_a = DixonColesEngine(DixonColesConfig(xi=0.0))
    only_a.fit(girone_a_only)
    assert set(only_a.params_.gamma) == {Girone.A}
    with pytest.raises(KeyError, match="girone B"):
        only_a.rates(*only_a.params_.teams_by_girone[Girone.A][:2], "B")
    with pytest.raises(KeyError):
        only_a.ranking("B")


def test_predict_before_fit_raises_not_fitted(league12):
    engine = DixonColesEngine()
    assert not engine.is_fitted
    with pytest.raises(NotFittedError):
        _ = engine.params_
    with pytest.raises(NotFittedError):
        engine.predict(league12.fixtures[0])
    with pytest.raises(NotFittedError):
        engine.ranking()
    assert issubclass(NotFittedError, RuntimeError)


def test_team_in_two_gironi_goes_to_majority_girone(mini, caplog):
    df = mini.matches.copy()
    extra = df.iloc[[0]].copy()
    extra["match_id"] = "CROSS-0001"
    extra["girone"] = "B"
    extra["home_team"] = "Vicenza"  # a Girone A team guesting in Girone B
    extra["away_team"] = "Arezzo"
    extra["home_goals"] = 1
    extra["away_goals"] = 1
    df = pd.concat([df, extra], ignore_index=True)
    with caplog.at_level(logging.WARNING, logger="src.engine.model"):
        params = DixonColesEngine(DixonColesConfig(l2_penalty=0.1)).fit(df)
    assert params.girone_of("Vicenza") is Girone.A
    assert "Vicenza" not in params.teams_by_girone[Girone.B]
    assert params.n_matches == 21
    assert any("more than one girone" in rec.getMessage() for rec in caplog.records)
    # the index helper reports the same assignment
    index = build_team_index(df.assign(girone=df["girone"].astype("string")))
    assert index.girone_of("Vicenza") is Girone.A


def test_min_matches_per_team_is_logged_not_fatal(mini, caplog):
    with caplog.at_level(logging.WARNING, logger="src.engine.model"):
        params = DixonColesEngine(DixonColesConfig(l2_penalty=0.1, min_matches_per_team=5)).fit(mini.matches)
    assert len(params.teams) == 14
    assert any("min_matches_per_team" in rec.getMessage() for rec in caplog.records)


def test_config_validation_and_serialisation():
    cfg = DixonColesConfig()
    assert cfg.xi == 0.0018 and cfg.method == "SLSQP" and cfg.max_goals == 10 and cfg.tol == 1e-9
    assert DixonColesConfig.from_dict(cfg.to_dict()) == cfg
    assert isinstance(cfg.to_dict()["rho_bounds"], list)
    with pytest.raises(ValueError):
        DixonColesConfig(method="BFGS")  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        DixonColesConfig(xi=-0.1)
    with pytest.raises(ValueError):
        DixonColesConfig(rho_bounds=(0.2, -0.2))
    with pytest.raises(ValueError):
        DixonColesConfig(max_goals=0)
    with pytest.raises(ValueError):
        DixonColesConfig(l2_penalty=-1.0)


def test_fit_rejects_invalid_frames():
    engine = DixonColesEngine()
    with pytest.raises(ValueError):
        engine.fit(pd.DataFrame({"home_team": ["a"], "away_team": ["b"]}))
    df, _ = _small_problem(seed=3, n_matches=10)
    with pytest.raises(ValueError):
        engine.fit(df.iloc[0:0])


def test_fit_rejects_negative_goals_and_same_team_rows(league12):
    """``validate_matches_df`` only coerces dtypes; the engine must range-check
    like ``MatchRecord`` does, otherwise harvested sentinels (-1 = postponed) or a
    name-matching collision would silently bias the fit."""
    clean = league12.matches
    engine = DixonColesEngine(DixonColesConfig(xi=0.0))

    negative = clean.copy()
    negative.loc[negative.index[:6], "home_goals"] = -1
    with pytest.raises(ValueError, match="6 matches have negative goals") as exc:
        engine.fit(negative)
    assert str(negative.loc[negative.index[0], "match_id"]) in str(exc.value)
    with pytest.raises(ValueError, match="negative goals"):
        engine.fit(clean.assign(away_goals=clean["away_goals"].where(clean.index != 3, -2)))
    with pytest.raises(ValueError, match="negative goals"):
        prepare_likelihood_data(negative)

    same = clean.copy()
    same.loc[same.index[0], "away_team"] = same.loc[same.index[0], "home_team"]
    with pytest.raises(ValueError, match="home_team == away_team") as exc:
        engine.fit(same)
    assert str(same.loc[same.index[0], "match_id"]) in str(exc.value)
    assert str(same.loc[same.index[0], "home_team"]) in str(exc.value)
    assert not engine.is_fitted  # nothing was stored by the rejected fits

    # the clean frame still fits, and the number of matches is untouched
    params = engine.fit(clean)
    assert params.n_matches == len(clean)


def test_lbfgsb_falls_back_to_slsqp_when_dependent_alpha_leaves_bounds(league12, caplog):
    """Regression: under the L-BFGS-B reparametrisation the dependent alpha of a
    girone (its alphabetically last team) is ``-sum(others)`` and not box-bounded.
    A team with no goals in the window drives it far below ``param_bounds``; the
    engine must detect this and re-solve the bounded, constrained problem."""
    df = league12.matches.copy()
    teams_a = sorted(set(df.loc[df["girone"] == "A", "home_team"]) | set(df.loc[df["girone"] == "A", "away_team"]))
    dependent = teams_a[-1]
    df.loc[df["home_team"] == dependent, "home_goals"] = 0
    df.loc[df["away_team"] == dependent, "away_goals"] = 0
    lo, hi = -3.0, 3.0

    engine_slsqp = DixonColesEngine(DixonColesConfig(xi=0.0, method="SLSQP", param_bounds=(lo, hi)))
    slsqp = engine_slsqp.fit(df)
    assert slsqp.alpha[dependent] == pytest.approx(lo, abs=1e-6)  # the bound is active at the optimum

    engine = DixonColesEngine(DixonColesConfig(xi=0.0, method="L-BFGS-B", param_bounds=(lo, hi)))
    with caplog.at_level(logging.WARNING, logger="src.engine.model"):
        params = engine.fit(df)
    assert any("outside param_bounds" in rec.getMessage() for rec in caplog.records)
    assert all(lo - 1e-9 <= a <= hi + 1e-9 for a in params.alpha.values()), min(params.alpha.values())
    assert params.alpha[dependent] == pytest.approx(lo, abs=1e-6)
    assert params.converged, params.message
    assert params.method == "SLSQP"  # the optimiser that produced the parameters
    assert "re-solved with SLSQP" in params.message and dependent in params.message
    assert params.log_likelihood == pytest.approx(slsqp.log_likelihood, abs=1e-6)
    np.testing.assert_allclose([params.alpha[t] for t in slsqp.teams], [slsqp.alpha[t] for t in slsqp.teams], atol=1e-6)
    for g, teams in params.teams_by_girone.items():
        assert abs(sum(params.alpha[t] for t in teams)) < 1e-6, g
    # downstream: the home rate of the zero-goal team is bounded away from zero, as under SLSQP
    opponent = teams_a[0]
    lam, _ = engine.rates(dependent, opponent, "A")
    lam_slsqp, _ = engine_slsqp.rates(dependent, opponent, "A")
    assert lam == pytest.approx(lam_slsqp, rel=1e-5) and lam > 1e-3


@pytest.mark.parametrize("half_width", [0.5, 0.3])
def test_bounds_are_respected(mini, half_width):
    """Every parameter honours its bounds under both methods. With ``half_width=0.3``
    the raw L-BFGS-B dependent alpha of one girone leaves the box on this dataset,
    so the alpha assertion exercises the SLSQP fallback."""
    lo, hi = -half_width, half_width
    cfg = DixonColesConfig(l2_penalty=0.0, rho_bounds=(-0.01, 0.01), home_bounds=(0.0, 0.05), param_bounds=(lo, hi))
    for method in ("SLSQP", "L-BFGS-B"):
        params = DixonColesEngine(replace(cfg, method=method)).fit(mini.matches)
        assert params.converged, (method, params.message)
        assert -0.01 - 1e-9 <= params.rho <= 0.01 + 1e-9
        assert all(0.0 - 1e-9 <= g <= 0.05 + 1e-9 for g in params.gamma.values())
        assert all(lo - 1e-9 <= b <= hi + 1e-9 for b in params.beta.values())
        # every alpha is bounded under both methods, including the dependent alpha of each
        # girone that the L-BFGS-B reparametrisation leaves free (SLSQP fallback)
        assert all(lo - 1e-9 <= a <= hi + 1e-9 for a in params.alpha.values()), (method, min(params.alpha.values()), max(params.alpha.values()))
        for g, teams in params.teams_by_girone.items():
            assert abs(sum(params.alpha[t] for t in teams)) < 1e-6
