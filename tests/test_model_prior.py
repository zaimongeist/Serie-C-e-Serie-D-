"""Gaussian (MAP) prior on home advantage and rho: small-sample regularisation."""
from __future__ import annotations

import numpy as np
import pytest
from scipy.optimize import check_grad

from src.engine.model import (
    DixonColesConfig,
    DixonColesEngine,
    LikelihoodData,
    negative_log_likelihood,
    negative_log_likelihood_grad,
    pack_theta,
    prepare_likelihood_data,
)
from src.simulation.generator import simulate_league, simulate_mini_dataset


@pytest.fixture(scope="module")
def mini():
    return simulate_mini_dataset(seed=42)


def test_likelihood_data_accepts_the_legacy_nine_fields(mini) -> None:
    data, _ = prepare_likelihood_data(mini.matches, xi=0.0, l2_penalty=0.2)
    legacy = LikelihoodData(*data[:9])
    assert legacy.l2_penalty_home == 0.0 and legacy.home_prior == 0.0 and legacy.l2_penalty_rho == 0.0
    theta = np.zeros(2 * data.n_teams + data.n_gammas + 1)
    assert negative_log_likelihood(theta, *legacy) == pytest.approx(negative_log_likelihood(theta, data))


def test_beta_ridge_is_centred_per_girone(mini) -> None:
    """Adding a constant to every beta of a girone (an intercept shift) must not be penalised."""
    data, index = prepare_likelihood_data(mini.matches, xi=0.0, l2_penalty=3.0)
    theta = np.zeros(2 * data.n_teams + data.n_gammas + 1)
    base = negative_log_likelihood(theta, data)
    shifted = theta.copy()
    shifted[data.n_teams : 2 * data.n_teams] += 0.7  # same shift for every team
    nll_shift_pen = negative_log_likelihood(shifted, data)
    plain, _ = prepare_likelihood_data(mini.matches, xi=0.0, l2_penalty=0.0)
    ll_change = negative_log_likelihood(shifted, plain) - negative_log_likelihood(theta, plain)
    assert nll_shift_pen - base == pytest.approx(ll_change)  # penalty unchanged by the shift
    # but spreading the betas is penalised
    spread = theta.copy()
    spread[data.n_teams] += 0.5
    assert negative_log_likelihood(spread, data) - negative_log_likelihood(spread, plain) > 0
    err = check_grad(negative_log_likelihood, negative_log_likelihood_grad, spread, data, epsilon=1e-6)
    assert err / max(np.linalg.norm(negative_log_likelihood_grad(spread, data)), 1.0) < 1e-5


def test_prior_terms_enter_objective_and_gradient(mini) -> None:
    plain, index = prepare_likelihood_data(mini.matches, xi=0.0, l2_penalty=0.0)
    prior, _ = prepare_likelihood_data(
        mini.matches, xi=0.0, l2_penalty=0.0, l2_penalty_home=2.0, home_prior=0.25, l2_penalty_rho=3.0
    )
    rng = np.random.default_rng(0)
    alpha = rng.normal(0, 0.3, index.n_teams)
    beta = rng.normal(0, 0.3, index.n_teams)
    gamma = rng.normal(0.4, 0.2, index.n_gammas)
    rho = -0.07
    theta = pack_theta(alpha, beta, gamma, rho)
    expected = 2.0 * np.sum((gamma - 0.25) ** 2) + 3.0 * rho**2
    assert negative_log_likelihood(theta, prior) - negative_log_likelihood(theta, plain) == pytest.approx(expected)
    err = check_grad(negative_log_likelihood, negative_log_likelihood_grad, theta, prior, epsilon=1e-6)
    scale = np.linalg.norm(negative_log_likelihood_grad(theta, prior))
    assert err / max(scale, 1.0) < 1e-6


def test_map_prior_keeps_small_sample_fit_regular(mini) -> None:
    """20 matches, 14 teams: the pure MLE sends gamma_C to its bound and rho to a bound;
    the MAP prior keeps every parameter in a plausible range and the rates sane."""
    mle = DixonColesEngine(DixonColesConfig(xi=0.0, l2_penalty=0.1)).fit(mini.matches)
    assert max(mle.gamma.values()) >= 1.4  # documents the degenerate MLE on 20 matches

    cfg = DixonColesConfig.with_gaussian_prior(xi=0.0)
    assert cfg.l2_penalty == pytest.approx(1 / (2 * 0.3**2))
    assert cfg.l2_penalty_rho == pytest.approx(50.0)
    engine = DixonColesEngine(cfg)
    params = engine.fit(mini.matches)
    assert params.converged
    assert all(-0.3 < g < 1.0 for g in params.gamma.values())
    assert abs(params.rho) < 0.15
    for g, teams in params.teams_by_girone.items():
        assert sum(params.alpha[t] for t in teams) == pytest.approx(0.0, abs=1e-6)
    rates = [r for p in engine.predict_fixtures(mini.fixtures) for r in (p.home_rate, p.away_rate)]
    assert 0.3 < min(rates) and max(rates) < 3.5
    # the prior is reported in the penalty, not in the log-likelihood
    assert params.penalty > 0 and params.log_likelihood < 0


def test_prior_is_negligible_on_a_full_season() -> None:
    league = simulate_league(seed=4, n_rounds=20)
    mle = DixonColesEngine(DixonColesConfig(xi=0.0)).fit(league.matches)
    prior = DixonColesEngine(DixonColesConfig.with_gaussian_prior(xi=0.0)).fit(league.matches)
    for g in mle.gamma:
        # shrinkage: the MAP gamma lies between the MLE and the prior mean, and moves little
        lo, hi = sorted((mle.gamma[g], 0.25))
        assert lo - 0.02 <= prior.gamma[g] <= hi + 0.02
        assert abs(prior.gamma[g] - mle.gamma[g]) < 0.2
    assert prior.rho == pytest.approx(mle.rho, abs=0.03)
    teams = list(mle.alpha)
    corr = np.corrcoef([mle.alpha[t] for t in teams], [prior.alpha[t] for t in teams])[0, 1]
    assert corr > 0.99


def test_config_validation_of_prior_fields() -> None:
    with pytest.raises(ValueError):
        DixonColesConfig(l2_penalty_home=-1.0)
    with pytest.raises(ValueError):
        DixonColesConfig(l2_penalty_rho=float("nan"))
    with pytest.raises(ValueError):
        DixonColesConfig(home_prior=5.0)  # outside home_bounds
    cfg = DixonColesConfig(l2_penalty_home=0.5, home_prior=0.3, l2_penalty_rho=0.5)
    assert DixonColesConfig.from_dict(cfg.to_dict()) == cfg
    with pytest.raises(ValueError):
        DixonColesConfig.with_gaussian_prior(sigma_rho=0.0)
    assert DixonColesConfig.with_gaussian_prior(method="L-BFGS-B").method == "L-BFGS-B"
    # an old config payload without the new keys still loads with the defaults
    old = {k: v for k, v in cfg.to_dict().items() if k not in ("l2_penalty_home", "home_prior", "l2_penalty_rho")}
    assert DixonColesConfig.from_dict(old).l2_penalty_home == 0.0
