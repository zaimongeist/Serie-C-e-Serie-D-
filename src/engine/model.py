"""Dixon-Coles engine for Italian Serie C (Gironi A / B / C).

Model
-----
For a match between ``home`` and ``away`` played in girone ``g`` the goal
rates are log-linear::

    lambda = exp(alpha[home] - beta[away] + gamma[g])     # expected home goals
    mu     = exp(alpha[away] - beta[home])                # expected away goals

``alpha`` is attack strength, ``beta`` is defensive solidity (a *higher* beta
concedes *less*), ``gamma`` is the home advantage estimated separately for each
girone (or shared, see :attr:`DixonColesConfig.girone_specific_home`) and
``rho`` is the Dixon-Coles low-score dependence parameter that corrects the
independent-Poisson probabilities of the 0-0, 1-0, 0-1 and 1-1 scores through
:func:`tau`.

The weighted log-likelihood of match *i* (factorial constants dropped) is::

    w_i * [ log tau(x_i, y_i, lam_i, mu_i, rho)
            + x_i log lam_i - lam_i + y_i log mu_i - mu_i ]

with time-decay weights ``w_i = exp(-xi * t_i)``, ``t_i`` being the age of the
match in days relative to the reference date (:func:`time_weights`).

Identifiability is imposed with ``sum(alpha) == 0`` *within each girone*: as an
explicit equality constraint (with Jacobian) for SLSQP, and through a
reparametrisation (the last alpha of each girone equals minus the sum of the
others) for L-BFGS-B. Both methods return the same :class:`FittedParams`
structure. An optional ridge penalty ``l2_penalty * (sum alpha^2 + sum beta^2)``
is added to the objective to stabilise very small datasets.

The objective exposes an analytic gradient (:func:`negative_log_likelihood_grad`)
which is passed to :func:`scipy.optimize.minimize`.
"""
from __future__ import annotations

import json
import math
import time
from dataclasses import asdict, dataclass, fields
from datetime import date, datetime
from difflib import get_close_matches
from pathlib import Path
from typing import Any, Final, Iterable, Literal, Mapping, NamedTuple

import numpy as np
import pandas as pd
from scipy.optimize import OptimizeResult, minimize
from scipy.special import gammaln, xlogy

from src.common.logging_config import get_logger
from src.common.schemas import (
    SELECTIONS,
    Fixture,
    Girone,
    Market,
    MatchPrediction,
    validate_matches_df,
)

__all__ = [
    "DixonColesConfig",
    "DixonColesEngine",
    "FittedParams",
    "LikelihoodData",
    "NotFittedError",
    "TeamIndex",
    "build_team_index",
    "dixon_coles_pmf",
    "markets_from_matrix",
    "negative_log_likelihood",
    "negative_log_likelihood_and_grad",
    "negative_log_likelihood_grad",
    "pack_theta",
    "prepare_likelihood_data",
    "tau",
    "time_weights",
    "unpack_theta",
    "TAU_FLOOR",
    "LOG_RATE_CLIP",
    "METHODS",
]

log = get_logger(__name__)

#: ``tau`` is clipped to this floor before taking its logarithm.
TAU_FLOOR: Final[float] = 1e-10
#: Log-rates are clipped to ``[-LOG_RATE_CLIP, LOG_RATE_CLIP]`` before ``exp`` (overflow guard).
LOG_RATE_CLIP: Final[float] = 50.0
#: Supported scipy optimisers.
METHODS: Final[tuple[str, ...]] = ("SLSQP", "L-BFGS-B")

MethodName = Literal["SLSQP", "L-BFGS-B"]

_OU_LINES: Final[dict[Market, float]] = {
    Market.OVER_UNDER_15: 1.5,
    Market.OVER_UNDER_25: 2.5,
    Market.OVER_UNDER_35: 3.5,
}
_SCHEMA_VERSION: Final[int] = 1


# =============================================================================
# Pure math helpers
# =============================================================================
def tau(
    x: np.ndarray | int,
    y: np.ndarray | int,
    lam: np.ndarray | float,
    mu: np.ndarray | float,
    rho: float,
) -> np.ndarray | float:
    """Dixon-Coles low-score correction factor, vectorised over numpy arrays.

    ``(0,0): 1 - lam*mu*rho``; ``(0,1): 1 + lam*rho``; ``(1,0): 1 + mu*rho``;
    ``(1,1): 1 - rho``; every other score: ``1``. Inputs broadcast against each
    other; scalar inputs return a Python ``float``.
    """
    x_arr = np.asarray(x)
    y_arr = np.asarray(y)
    lam_arr = np.asarray(lam, dtype=float)
    mu_arr = np.asarray(mu, dtype=float)
    shape = np.broadcast_shapes(x_arr.shape, y_arr.shape, lam_arr.shape, mu_arr.shape)
    out = np.ones(shape, dtype=float)
    out = np.where((x_arr == 0) & (y_arr == 0), 1.0 - lam_arr * mu_arr * rho, out)
    out = np.where((x_arr == 0) & (y_arr == 1), 1.0 + lam_arr * rho, out)
    out = np.where((x_arr == 1) & (y_arr == 0), 1.0 + mu_arr * rho, out)
    out = np.where((x_arr == 1) & (y_arr == 1), 1.0 - rho, out)
    if out.ndim == 0:
        return float(out)
    return out


def _tau_and_partials(
    x: np.ndarray, y: np.ndarray, lam: np.ndarray, mu: np.ndarray, rho: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """``tau`` together with its partial derivatives w.r.t. ``lam``, ``mu`` and ``rho``."""
    m00 = (x == 0) & (y == 0)
    m01 = (x == 0) & (y == 1)
    m10 = (x == 1) & (y == 0)
    m11 = (x == 1) & (y == 1)
    t = np.ones_like(lam)
    d_lam = np.zeros_like(lam)
    d_mu = np.zeros_like(lam)
    d_rho = np.zeros_like(lam)
    # (0, 0): 1 - lam*mu*rho
    t = np.where(m00, 1.0 - lam * mu * rho, t)
    d_lam = np.where(m00, -mu * rho, d_lam)
    d_mu = np.where(m00, -lam * rho, d_mu)
    d_rho = np.where(m00, -lam * mu, d_rho)
    # (0, 1): 1 + lam*rho
    t = np.where(m01, 1.0 + lam * rho, t)
    d_lam = np.where(m01, rho, d_lam)
    d_rho = np.where(m01, lam, d_rho)
    # (1, 0): 1 + mu*rho
    t = np.where(m10, 1.0 + mu * rho, t)
    d_mu = np.where(m10, rho, d_mu)
    d_rho = np.where(m10, mu, d_rho)
    # (1, 1): 1 - rho
    t = np.where(m11, 1.0 - rho, t)
    d_rho = np.where(m11, -1.0, d_rho)
    return t, d_lam, d_mu, d_rho


def time_weights(
    dates: pd.Series | np.ndarray | Iterable[Any],
    reference_date: date | datetime | pd.Timestamp,
    xi: float,
) -> np.ndarray:
    """Exponential time-decay weights ``w = exp(-xi * t)``.

    ``t`` is the age of each match in whole days relative to ``reference_date``.
    ``xi`` is expressed in 1/day (Dixon-Coles' 0.0065 per half-week is about
    0.0018 per day). ``xi = 0`` gives unit weights.

    Matches dated *after* the reference date are not extrapolated: their age is
    clipped at 0, so they receive weight 1 (the maximum). This keeps the fit
    well defined when the reference date is chosen inside the sample.
    """
    if xi < 0:
        raise ValueError(f"xi must be >= 0, got {xi}")
    if isinstance(dates, pd.Series):
        series = dates
    elif isinstance(dates, np.ndarray):
        series = pd.Series(dates)
    else:
        series = pd.Series(list(dates))
    series = pd.to_datetime(series)
    if series.dt.tz is not None:
        series = series.dt.tz_localize(None)
    series = series.dt.normalize()
    if series.isna().any():
        raise ValueError("dates contain missing values (NaT)")
    ref = pd.Timestamp(reference_date)
    if ref.tzinfo is not None:
        ref = ref.tz_localize(None)
    ref = ref.normalize()
    age_days = (ref - series).dt.days.to_numpy(dtype=np.float64)
    age_days = np.clip(age_days, 0.0, None)
    return np.exp(-float(xi) * age_days)


def _poisson_pmf(ks: np.ndarray, rate: float) -> np.ndarray:
    """Poisson pmf evaluated through log-space (``rate = 0`` is handled by ``xlogy``)."""
    return np.exp(xlogy(ks, rate) - rate - gammaln(ks + 1.0))


def dixon_coles_pmf(lam: float, mu: float, rho: float, max_goals: int = 10) -> np.ndarray:
    """Joint score matrix ``P(X = i, Y = j)`` on a ``(max_goals+1)^2`` grid.

    Independent Poisson pmfs multiplied by :func:`tau`, clipped to be
    non-negative and renormalised to sum to one (the probability mass beyond
    ``max_goals`` is folded back proportionally).
    """
    if max_goals < 1:
        raise ValueError(f"max_goals must be >= 1, got {max_goals}")
    if not (np.isfinite(lam) and np.isfinite(mu)) or lam < 0 or mu < 0:
        raise ValueError(f"rates must be finite and non-negative, got lam={lam}, mu={mu}")
    ks = np.arange(max_goals + 1)
    grid = np.outer(_poisson_pmf(ks, float(lam)), _poisson_pmf(ks, float(mu)))
    xs, ys = np.meshgrid(ks, ks, indexing="ij")
    grid = np.clip(grid * tau(xs, ys, float(lam), float(mu), float(rho)), 0.0, None)
    total = float(grid.sum())
    if not (total > 0.0 and np.isfinite(total)):
        raise ValueError(f"degenerate score matrix (sum={total}) for lam={lam}, mu={mu}, rho={rho}")
    return grid / total


def markets_from_matrix(matrix: np.ndarray) -> dict[Market, dict[str, float]]:
    """Derive 1X2, Over/Under 1.5-2.5-3.5 and GG/NG probabilities from a score matrix.

    Keys are exactly :data:`src.common.schemas.SELECTIONS`; every market sums to
    the total mass of the matrix (1 for a normalised matrix). A matrix that does
    not sum to one is renormalised first.
    """
    m = np.asarray(matrix, dtype=float)
    if m.ndim != 2 or m.shape[0] != m.shape[1]:
        raise ValueError(f"matrix must be square, got shape {m.shape}")
    total_mass = float(m.sum())
    if not (total_mass > 0.0 and np.isfinite(total_mass)):
        raise ValueError(f"matrix mass must be positive and finite, got {total_mass}")
    if not math.isclose(total_mass, 1.0, rel_tol=0.0, abs_tol=1e-9):
        m = m / total_mass
    n = m.shape[0]
    xs, ys = np.meshgrid(np.arange(n), np.arange(n), indexing="ij")
    goals = xs + ys
    raw: dict[Market, dict[str, float]] = {
        Market.MATCH_ODDS: {
            "1": float(m[xs > ys].sum()),
            "X": float(np.trace(m)),
            "2": float(m[xs < ys].sum()),
        },
        Market.BTTS: {
            "GG": float(m[(xs > 0) & (ys > 0)].sum()),
            "NG": float(m[(xs == 0) | (ys == 0)].sum()),
        },
    }
    for market, line in _OU_LINES.items():
        over_key, under_key = SELECTIONS[market]
        raw[market] = {
            over_key: float(m[goals > line].sum()),
            under_key: float(m[goals < line].sum()),
        }
    return {market: {sel: raw[market][sel] for sel in SELECTIONS[market]} for market in SELECTIONS}


# =============================================================================
# Parameter vector helpers and likelihood data
# =============================================================================
def pack_theta(alpha: np.ndarray, beta: np.ndarray, gamma: np.ndarray, rho: float) -> np.ndarray:
    """Pack ``theta = [alphas..., betas..., gammas..., rho]`` into one vector."""
    return np.concatenate(
        [
            np.asarray(alpha, dtype=float).ravel(),
            np.asarray(beta, dtype=float).ravel(),
            np.asarray(gamma, dtype=float).ravel(),
            np.array([float(rho)]),
        ]
    )


def unpack_theta(
    theta: np.ndarray, n_teams: int, n_gammas: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """Inverse of :func:`pack_theta`; returns ``(alpha, beta, gamma, rho)``."""
    theta = np.asarray(theta, dtype=float)
    expected = 2 * n_teams + n_gammas + 1
    if theta.shape != (expected,):
        raise ValueError(f"theta must have length {expected} (2*{n_teams} + {n_gammas} + 1), got {theta.shape}")
    alpha = theta[:n_teams]
    beta = theta[n_teams : 2 * n_teams]
    gamma = theta[2 * n_teams : 2 * n_teams + n_gammas]
    rho = float(theta[-1])
    return alpha, beta, gamma, rho


class LikelihoodData(NamedTuple):
    """Pre-indexed match data consumed by the likelihood functions.

    The tuple is unpacked positionally into ``negative_log_likelihood(theta, *data)``
    so it can be passed straight through ``scipy.optimize.minimize(args=...)``.
    """

    x: np.ndarray  #: home goals (int64)
    y: np.ndarray  #: away goals (int64)
    w: np.ndarray  #: time-decay weights (float64)
    home_idx: np.ndarray  #: position of the home team in the team vector
    away_idx: np.ndarray  #: position of the away team in the team vector
    gamma_idx: np.ndarray  #: position of the match's home-advantage parameter
    n_teams: int
    n_gammas: int
    l2_penalty: float


def _as_data(data: tuple[Any, ...]) -> LikelihoodData:
    if len(data) == 1 and isinstance(data[0], LikelihoodData):
        return data[0]
    if len(data) != len(LikelihoodData._fields):
        raise TypeError(
            f"expected a LikelihoodData instance or its {len(LikelihoodData._fields)} fields, got {len(data)} positional arguments"
        )
    return LikelihoodData(*data)


@dataclass(frozen=True)
class TeamIndex:
    """Team <-> parameter position bookkeeping for one fit.

    ``teams`` is the global team order (girone A teams first, alphabetical within
    the girone, then B, then C). ``gironi`` are the gironi that appear in the
    match frame, in letter order; they index the home-advantage parameters when
    ``girone_specific_home`` is true (otherwise there is a single shared gamma).
    """

    teams: tuple[str, ...]
    teams_by_girone: dict[Girone, tuple[str, ...]]
    gironi: tuple[Girone, ...]
    girone_specific_home: bool = True

    def __post_init__(self) -> None:
        if len(set(self.teams)) != len(self.teams):
            raise ValueError("duplicate team names in TeamIndex.teams")
        assigned = [t for g in self.teams_by_girone.values() for t in g]
        if sorted(assigned) != sorted(self.teams):
            raise ValueError("teams_by_girone does not partition TeamIndex.teams")
        for g in self.teams_by_girone:
            if g not in self.gironi:
                raise ValueError(f"girone {g.value} has teams but no matches")

    @property
    def n_teams(self) -> int:
        return len(self.teams)

    @property
    def n_gammas(self) -> int:
        return len(self.gironi) if self.girone_specific_home else 1

    def team_positions(self) -> dict[str, int]:
        """team name -> position in the alpha/beta vectors."""
        return {t: i for i, t in enumerate(self.teams)}

    def girone_of(self, team: str) -> Girone:
        for g, members in self.teams_by_girone.items():
            if team in members:
                return g
        raise KeyError(team)

    def team_girone_positions(self) -> np.ndarray:
        """For every team, the position of its girone inside ``gironi``."""
        girone_pos = {g: i for i, g in enumerate(self.gironi)}
        return np.array([girone_pos[self.girone_of(t)] for t in self.teams], dtype=np.int64)

    def gamma_position(self, girone: Girone | str) -> int:
        """Position of the home-advantage parameter used for matches of ``girone``."""
        if not self.girone_specific_home:
            return 0
        g = Girone.parse(girone)
        try:
            return self.gironi.index(g)
        except ValueError as exc:
            raise KeyError(f"girone {g.value} was not present in the fitted matches") from exc

    def alpha_groups(self) -> list[np.ndarray]:
        """Team positions of every girone that has at least one team (constraint groups)."""
        positions = self.team_positions()
        return [np.array([positions[t] for t in members], dtype=np.int64) for members in self.teams_by_girone.values() if members]


def build_team_index(
    matches: pd.DataFrame,
    *,
    girone_specific_home: bool = True,
    min_matches_per_team: int = 1,
) -> TeamIndex:
    """Build the :class:`TeamIndex` of a *validated* match frame.

    A team appearing in more than one girone is assigned to the girone where it
    played most matches (ties broken by letter order) and a warning is logged.
    Teams with fewer than ``min_matches_per_team`` matches are still estimated
    but reported in the log.
    """
    long = pd.concat(
        [
            matches[["home_team", "girone"]].rename(columns={"home_team": "team"}),
            matches[["away_team", "girone"]].rename(columns={"away_team": "team"}),
        ],
        ignore_index=True,
    )
    counts = long.groupby(["team", "girone"], observed=True).size().reset_index(name="n")
    counts = counts.sort_values(["team", "n", "girone"], ascending=[True, False, True], kind="stable")
    best = counts.drop_duplicates(subset="team", keep="first")
    assignment: dict[str, Girone] = {str(t): Girone.parse(g) for t, g in zip(best["team"], best["girone"])}

    per_team_gironi = counts.groupby("team", observed=True).size()
    for team in per_team_gironi[per_team_gironi > 1].index:
        detail = {str(g): int(n) for g, n in zip(counts.loc[counts["team"] == team, "girone"], counts.loc[counts["team"] == team, "n"])}
        log.warning(
            "team appears in more than one girone; assigned to the majority girone",
            extra={"team": str(team), "matches_by_girone": detail, "assigned": assignment[str(team)].value},
        )

    total_per_team = long.groupby("team", observed=True).size()
    thin = total_per_team[total_per_team < min_matches_per_team]
    if len(thin):
        log.warning(
            "teams with fewer matches than min_matches_per_team are still estimated",
            extra={"min_matches_per_team": min_matches_per_team, "teams": {str(t): int(n) for t, n in thin.items()}},
        )

    teams_by_girone: dict[Girone, tuple[str, ...]] = {}
    for g in Girone:
        members = tuple(sorted(t for t, tg in assignment.items() if tg is g))
        if members:
            teams_by_girone[g] = members
    teams = tuple(t for g in Girone for t in teams_by_girone.get(g, ()))
    gironi = tuple(g for g in Girone if g.value in set(matches["girone"].astype(str)))
    return TeamIndex(teams=teams, teams_by_girone=teams_by_girone, gironi=gironi, girone_specific_home=girone_specific_home)


def prepare_likelihood_data(
    matches: pd.DataFrame,
    *,
    reference_date: date | datetime | pd.Timestamp | None = None,
    xi: float = 0.0,
    l2_penalty: float = 0.0,
    girone_specific_home: bool = True,
    team_index: TeamIndex | None = None,
    min_matches_per_team: int = 1,
) -> tuple[LikelihoodData, TeamIndex]:
    """Validate ``matches`` and convert them into :class:`LikelihoodData`.

    Returns the data together with the :class:`TeamIndex` used to position the
    parameters (built from the frame unless ``team_index`` is given). The
    reference date defaults to the most recent match date.
    """
    df = validate_matches_df(matches)
    if df.empty:
        raise ValueError("no valid matches to fit")
    index = team_index or build_team_index(
        df, girone_specific_home=girone_specific_home, min_matches_per_team=min_matches_per_team
    )
    ref = df["date"].max() if reference_date is None else pd.Timestamp(reference_date)
    positions = index.team_positions()
    home_pos = df["home_team"].map(positions)
    away_pos = df["away_team"].map(positions)
    if home_pos.isna().any() or away_pos.isna().any():
        unknown = sorted(set(df.loc[home_pos.isna(), "home_team"]) | set(df.loc[away_pos.isna(), "away_team"]))
        raise KeyError(f"teams not present in the team index: {unknown}")
    if index.girone_specific_home:
        girone_pos = {g.value: i for i, g in enumerate(index.gironi)}
        gamma_idx_series = df["girone"].map(girone_pos)
        if gamma_idx_series.isna().any():
            unknown_g = sorted(set(df.loc[gamma_idx_series.isna(), "girone"].astype(str)))
            raise KeyError(f"gironi not present in the team index: {unknown_g}")
        gamma_idx = gamma_idx_series.to_numpy(dtype=np.int64)
    else:
        gamma_idx = np.zeros(len(df), dtype=np.int64)
    data = LikelihoodData(
        x=df["home_goals"].to_numpy(dtype=np.int64),
        y=df["away_goals"].to_numpy(dtype=np.int64),
        w=time_weights(df["date"], ref, xi),
        home_idx=home_pos.to_numpy(dtype=np.int64),
        away_idx=away_pos.to_numpy(dtype=np.int64),
        gamma_idx=gamma_idx,
        n_teams=index.n_teams,
        n_gammas=index.n_gammas,
        l2_penalty=float(l2_penalty),
    )
    return data, index


# =============================================================================
# Objective and analytic gradient
# =============================================================================
def _evaluate(theta: np.ndarray, d: LikelihoodData, *, with_grad: bool) -> tuple[float, float, np.ndarray | None]:
    """Return ``(weighted log-likelihood, penalty, gradient of the penalised NLL or None)``."""
    theta = np.asarray(theta, dtype=float)
    alpha, beta, gamma, rho = unpack_theta(theta, d.n_teams, d.n_gammas)
    log_lam = np.clip(alpha[d.home_idx] - beta[d.away_idx] + gamma[d.gamma_idx], -LOG_RATE_CLIP, LOG_RATE_CLIP)
    log_mu = np.clip(alpha[d.away_idx] - beta[d.home_idx], -LOG_RATE_CLIP, LOG_RATE_CLIP)
    lam = np.exp(log_lam)
    mu = np.exp(log_mu)
    t, dt_dlam, dt_dmu, dt_drho = _tau_and_partials(d.x, d.y, lam, mu, rho)
    clipped = t < TAU_FLOOR
    t_safe = np.where(clipped, TAU_FLOOR, t)
    ll_terms = np.log(t_safe) + d.x * log_lam - lam + d.y * log_mu - mu
    ll = float(np.dot(d.w, ll_terms))
    penalty = float(d.l2_penalty * (np.dot(alpha, alpha) + np.dot(beta, beta)))
    if not with_grad:
        return ll, penalty, None

    # Where tau is clipped the objective is locally flat in tau: derivative 0.
    inv_t = np.where(clipped, 0.0, 1.0 / t_safe)
    g_loglam = d.w * (inv_t * dt_dlam * lam + d.x - lam)  # d LL_i / d log(lam_i)
    g_logmu = d.w * (inv_t * dt_dmu * mu + d.y - mu)  # d LL_i / d log(mu_i)
    g_rho = float(np.sum(d.w * inv_t * dt_drho))
    n_t = d.n_teams
    g_alpha = np.bincount(d.home_idx, weights=g_loglam, minlength=n_t) + np.bincount(
        d.away_idx, weights=g_logmu, minlength=n_t
    )
    g_beta = -np.bincount(d.away_idx, weights=g_loglam, minlength=n_t) - np.bincount(
        d.home_idx, weights=g_logmu, minlength=n_t
    )
    g_gamma = np.bincount(d.gamma_idx, weights=g_loglam, minlength=d.n_gammas)
    grad = -pack_theta(g_alpha, g_beta, g_gamma, g_rho)
    grad[:n_t] += 2.0 * d.l2_penalty * alpha
    grad[n_t : 2 * n_t] += 2.0 * d.l2_penalty * beta
    return ll, penalty, grad


def negative_log_likelihood(theta: np.ndarray, *data: Any) -> float:
    """Penalised negative weighted log-likelihood ``-LL(theta) + l2 * (|alpha|^2 + |beta|^2)``.

    ``data`` is a :class:`LikelihoodData` instance or its unpacked fields.
    """
    ll, penalty, _ = _evaluate(theta, _as_data(data), with_grad=False)
    return -ll + penalty


def negative_log_likelihood_grad(theta: np.ndarray, *data: Any) -> np.ndarray:
    """Analytic gradient of :func:`negative_log_likelihood` w.r.t. ``theta``."""
    _, _, grad = _evaluate(theta, _as_data(data), with_grad=True)
    assert grad is not None  # for the type checker; with_grad=True always yields a gradient
    return grad


def negative_log_likelihood_and_grad(theta: np.ndarray, *data: Any) -> tuple[float, np.ndarray]:
    """Objective value and gradient in one pass (what the optimiser calls with ``jac=True``)."""
    ll, penalty, grad = _evaluate(theta, _as_data(data), with_grad=True)
    assert grad is not None
    return -ll + penalty, grad


# =============================================================================
# Configuration and fitted parameters
# =============================================================================
@dataclass(frozen=True)
class DixonColesConfig:
    """Hyper-parameters of :class:`DixonColesEngine`.

    ``xi`` is the time-decay rate in 1/day; ``girone_specific_home=False``
    estimates a single home advantage that is reported for every girone;
    teams with fewer than ``min_matches_per_team`` matches are still estimated
    but logged.
    """

    xi: float = 0.0018
    max_goals: int = 10
    method: MethodName = "SLSQP"
    rho_bounds: tuple[float, float] = (-0.2, 0.2)
    home_bounds: tuple[float, float] = (-1.0, 1.5)
    param_bounds: tuple[float, float] = (-3.0, 3.0)
    l2_penalty: float = 0.0
    max_iter: int = 1000
    tol: float = 1e-9
    girone_specific_home: bool = True
    min_matches_per_team: int = 1

    def __post_init__(self) -> None:
        if self.xi < 0:
            raise ValueError(f"xi must be >= 0, got {self.xi}")
        if self.max_goals < 1:
            raise ValueError(f"max_goals must be >= 1, got {self.max_goals}")
        if self.method not in METHODS:
            raise ValueError(f"method must be one of {METHODS}, got {self.method!r}")
        for name in ("rho_bounds", "home_bounds", "param_bounds"):
            lo, hi = getattr(self, name)
            if not float(lo) < float(hi):
                raise ValueError(f"{name} must be (low, high) with low < high, got {(lo, hi)}")
            object.__setattr__(self, name, (float(lo), float(hi)))
        if self.l2_penalty < 0:
            raise ValueError(f"l2_penalty must be >= 0, got {self.l2_penalty}")
        if self.max_iter < 1:
            raise ValueError(f"max_iter must be >= 1, got {self.max_iter}")
        if not self.tol > 0:
            raise ValueError(f"tol must be > 0, got {self.tol}")
        if self.min_matches_per_team < 0:
            raise ValueError(f"min_matches_per_team must be >= 0, got {self.min_matches_per_team}")

    def to_dict(self) -> dict[str, Any]:
        """JSON-serialisable representation (bounds as lists)."""
        payload = asdict(self)
        for name in ("rho_bounds", "home_bounds", "param_bounds"):
            payload[name] = list(payload[name])
        return payload

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "DixonColesConfig":
        known = {f.name for f in fields(cls)}
        kwargs: dict[str, Any] = {k: v for k, v in payload.items() if k in known}
        for name in ("rho_bounds", "home_bounds", "param_bounds"):
            if name in kwargs:
                lo, hi = kwargs[name]
                kwargs[name] = (float(lo), float(hi))
        return cls(**kwargs)


class NotFittedError(RuntimeError):
    """Raised when a prediction is requested before :meth:`DixonColesEngine.fit`."""


def _coerce_date(value: date | datetime | pd.Timestamp | str) -> date:
    if isinstance(value, pd.Timestamp):
        return value.date()
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    return date.fromisoformat(str(value)[:10])


@dataclass
class FittedParams:
    """Result of one Dixon-Coles fit.

    ``log_likelihood`` is the weighted log-likelihood at the optimum *without*
    the ridge penalty (``penalty`` is reported separately); ``effective_n`` is
    the sum of the time-decay weights.
    """

    alpha: dict[str, float]
    beta: dict[str, float]
    gamma: dict[Girone, float]
    rho: float
    teams_by_girone: dict[Girone, tuple[str, ...]]
    reference_date: date
    xi: float
    log_likelihood: float
    penalty: float
    n_matches: int
    effective_n: float
    converged: bool
    message: str
    n_iter: int
    fit_seconds: float
    method: str
    max_goals: int

    def __post_init__(self) -> None:
        self.alpha = {str(k): float(v) for k, v in self.alpha.items()}
        self.beta = {str(k): float(v) for k, v in self.beta.items()}
        self.gamma = {Girone.parse(k): float(v) for k, v in self.gamma.items()}
        self.teams_by_girone = {Girone.parse(k): tuple(str(t) for t in v) for k, v in self.teams_by_girone.items()}
        self.reference_date = _coerce_date(self.reference_date)
        self.rho = float(self.rho)

    # ----------------------------------------------------------------- views
    @property
    def teams(self) -> tuple[str, ...]:
        """All teams in girone order (A, B, C), alphabetical within a girone."""
        return tuple(t for g in Girone for t in self.teams_by_girone.get(g, ()))

    def girone_of(self, team: str) -> Girone:
        """Girone a team was assigned to during the fit (``KeyError`` when unknown)."""
        for g, members in self.teams_by_girone.items():
            if team in members:
                return g
        raise KeyError(_unknown_team_message(team, self.teams))

    def to_frame(self) -> pd.DataFrame:
        """One row per team: ``[team, girone, alpha, beta, gamma]`` (girone as letter)."""
        rows = [
            {
                "team": t,
                "girone": g.value,
                "alpha": self.alpha[t],
                "beta": self.beta[t],
                "gamma": self.gamma.get(g, float("nan")),
            }
            for g in Girone
            for t in self.teams_by_girone.get(g, ())
        ]
        frame = pd.DataFrame(rows, columns=["team", "girone", "alpha", "beta", "gamma"])
        frame["team"] = frame["team"].astype("string")
        frame["girone"] = frame["girone"].astype("string")
        for c in ("alpha", "beta", "gamma"):
            frame[c] = frame[c].astype("float64")
        return frame

    # ------------------------------------------------------------ serialisation
    def to_dict(self) -> dict[str, Any]:
        """JSON-serialisable representation (dates ISO, gironi as letters)."""
        return {
            "schema_version": _SCHEMA_VERSION,
            "alpha": dict(self.alpha),
            "beta": dict(self.beta),
            "gamma": {g.value: v for g, v in self.gamma.items()},
            "rho": self.rho,
            "teams_by_girone": {g.value: list(ts) for g, ts in self.teams_by_girone.items()},
            "reference_date": self.reference_date.isoformat(),
            "xi": self.xi,
            "log_likelihood": self.log_likelihood,
            "penalty": self.penalty,
            "n_matches": self.n_matches,
            "effective_n": self.effective_n,
            "converged": self.converged,
            "message": self.message,
            "n_iter": self.n_iter,
            "fit_seconds": self.fit_seconds,
            "method": self.method,
            "max_goals": self.max_goals,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "FittedParams":
        known = {f.name for f in fields(cls)}
        missing = known - set(payload)
        if missing:
            raise ValueError(f"fitted-parameter payload is missing fields: {sorted(missing)}")
        return cls(**{k: v for k, v in payload.items() if k in known})

    def to_json(self, path: str | Path) -> Path:
        """Write the parameters to ``path`` as JSON and return the path."""
        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(self.to_dict(), indent=2, ensure_ascii=False), encoding="utf-8")
        return out

    @classmethod
    def from_json(cls, path: str | Path) -> "FittedParams":
        """Read parameters written by :meth:`to_json` (or by :meth:`DixonColesEngine.save`)."""
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        if "params" in payload and isinstance(payload["params"], Mapping):
            payload = payload["params"]
        return cls.from_dict(payload)


def _unknown_team_message(team: str, known: Iterable[str]) -> str:
    known_list = sorted(known)
    hint = get_close_matches(str(team), known_list, n=3, cutoff=0.6)
    msg = f"unknown team {team!r}"
    if hint:
        msg += f"; did you mean {hint}?"
    msg += f" Known teams ({len(known_list)}): {known_list}"
    return msg


# =============================================================================
# Alpha reparametrisation (L-BFGS-B)
# =============================================================================
class _AlphaReparam:
    """Map between the full ``theta`` and the reduced vector where, inside every
    girone, the last team's alpha is minus the sum of the others.

    The dependent team of each girone is its last member (alphabetically); its
    alpha is only implicitly bounded through the bounds of the free alphas.
    """

    def __init__(self, index: TeamIndex) -> None:
        n_teams = index.n_teams
        groups = index.alpha_groups()
        dependent = np.array([grp[-1] for grp in groups], dtype=np.int64)
        free_mask = np.ones(n_teams, dtype=bool)
        free_mask[dependent] = False
        self.free: np.ndarray = np.flatnonzero(free_mask)
        dep_for_team = np.empty(n_teams, dtype=np.int64)
        for grp, dep in zip(groups, dependent):
            dep_for_team[grp] = dep
        self.dep_for_free: np.ndarray = dep_for_team[self.free]
        self.n_teams = n_teams
        self.n_free = int(self.free.size)

    def full_to_free(self, theta: np.ndarray) -> np.ndarray:
        theta = np.asarray(theta, dtype=float)
        return np.concatenate([theta[: self.n_teams][self.free], theta[self.n_teams :]])

    def free_to_full(self, theta_free: np.ndarray) -> np.ndarray:
        theta_free = np.asarray(theta_free, dtype=float)
        alpha_free = theta_free[: self.n_free]
        alpha = np.zeros(self.n_teams, dtype=float)
        alpha[self.free] = alpha_free
        np.add.at(alpha, self.dep_for_free, -alpha_free)
        return np.concatenate([alpha, theta_free[self.n_free :]])

    def grad_full_to_free(self, grad: np.ndarray) -> np.ndarray:
        grad = np.asarray(grad, dtype=float)
        g_alpha = grad[: self.n_teams]
        return np.concatenate([g_alpha[self.free] - g_alpha[self.dep_for_free], grad[self.n_teams :]])


# =============================================================================
# Engine
# =============================================================================
class DixonColesEngine:
    """Fit the Dixon-Coles model and turn it into match probabilities.

    Typical use::

        engine = DixonColesEngine(DixonColesConfig(xi=0.0018))
        params = engine.fit(matches)
        prediction = engine.predict(fixture)
    """

    def __init__(self, config: DixonColesConfig | None = None) -> None:
        self.config: DixonColesConfig = config or DixonColesConfig()
        self._params: FittedParams | None = None
        self._index: TeamIndex | None = None

    # ------------------------------------------------------------------ state
    @property
    def params_(self) -> FittedParams:
        """Parameters of the last fit (``NotFittedError`` before :meth:`fit`)."""
        if self._params is None:
            raise NotFittedError("DixonColesEngine is not fitted yet: call fit(matches) or load(path) first")
        return self._params

    @property
    def is_fitted(self) -> bool:
        return self._params is not None

    # -------------------------------------------------------------------- fit
    def fit(
        self,
        matches: pd.DataFrame,
        *,
        reference_date: date | None = None,
        warm_start: FittedParams | None = None,
    ) -> FittedParams:
        """Estimate ``alpha, beta, gamma, rho`` by weighted maximum likelihood.

        ``matches`` is validated with :func:`validate_matches_df`. The reference
        date for the time-decay defaults to the most recent match date.
        ``warm_start`` initialises the optimiser from a previous fit when its team
        set matches the current one (it is ignored, with a warning, otherwise).
        """
        started = time.perf_counter()
        cfg = self.config
        validated = validate_matches_df(matches)
        if validated.empty:
            raise ValueError("no valid matches to fit")
        ref_date = _coerce_date(validated["date"].max()) if reference_date is None else _coerce_date(reference_date)
        data, index = prepare_likelihood_data(
            validated,
            reference_date=ref_date,
            xi=cfg.xi,
            l2_penalty=cfg.l2_penalty,
            girone_specific_home=cfg.girone_specific_home,
            min_matches_per_team=cfg.min_matches_per_team,
        )
        effective_n = float(data.w.sum())
        log.info(
            "fitting Dixon-Coles",
            extra={
                "n_matches": int(len(data.x)),
                "n_teams": index.n_teams,
                "gironi": [g.value for g in index.gironi],
                "method": cfg.method,
                "xi": cfg.xi,
                "l2_penalty": cfg.l2_penalty,
                "reference_date": ref_date.isoformat(),
                "effective_n": round(effective_n, 3),
            },
        )

        theta0 = self._initial_theta(data, index, warm_start)
        scale = 1.0 / max(effective_n, 1e-12)  # optimise the mean weighted NLL per unit weight
        if cfg.method == "SLSQP":
            result, theta = self._fit_slsqp(theta0, data, index, scale)
        else:
            result, theta = self._fit_lbfgsb(theta0, data, index, scale)

        theta = self._recentre_alpha(theta, index)
        ll, penalty, grad = _evaluate(theta, data, with_grad=True)
        assert grad is not None
        alpha, beta, gamma, rho = unpack_theta(theta, data.n_teams, data.n_gammas)
        gamma_map: dict[Girone, float]
        if cfg.girone_specific_home:
            gamma_map = {g: float(gamma[i]) for i, g in enumerate(index.gironi)}
        else:
            gamma_map = {g: float(gamma[0]) for g in Girone}
        converged = bool(result.success)
        message = str(result.message)
        n_iter = int(getattr(result, "nit", 0) or 0)
        seconds = time.perf_counter() - started
        params = FittedParams(
            alpha={t: float(alpha[i]) for i, t in enumerate(index.teams)},
            beta={t: float(beta[i]) for i, t in enumerate(index.teams)},
            gamma=gamma_map,
            rho=float(rho),
            teams_by_girone=dict(index.teams_by_girone),
            reference_date=ref_date,
            xi=cfg.xi,
            log_likelihood=float(ll),
            penalty=float(penalty),
            n_matches=int(len(data.x)),
            effective_n=effective_n,
            converged=converged,
            message=message,
            n_iter=n_iter,
            fit_seconds=float(seconds),
            method=cfg.method,
            max_goals=cfg.max_goals,
        )
        self._params = params
        self._index = index
        log_fn = log.info if converged else log.warning
        log_fn(
            "Dixon-Coles fit finished",
            extra={
                "converged": converged,
                "optimizer_message": message,
                "n_iter": n_iter,
                "log_likelihood": round(float(ll), 4),
                "penalty": round(float(penalty), 6),
                "rho": round(float(rho), 5),
                "gamma": {g.value: round(v, 4) for g, v in gamma_map.items()},
                "grad_max_abs": float(np.max(np.abs(grad))),
                "fit_seconds": round(seconds, 3),
            },
        )
        return params

    # ----------------------------------------------------------- fit helpers
    def _bounds(self, n_teams: int, n_gammas: int) -> list[tuple[float, float]]:
        cfg = self.config
        return (
            [cfg.param_bounds] * n_teams
            + [cfg.param_bounds] * n_teams
            + [cfg.home_bounds] * n_gammas
            + [cfg.rho_bounds]
        )

    @staticmethod
    def _clip_to_bounds(theta: np.ndarray, bounds: list[tuple[float, float]]) -> np.ndarray:
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        return np.clip(theta, lo, hi)

    @staticmethod
    def _recentre_alpha(theta: np.ndarray, index: TeamIndex) -> np.ndarray:
        """Remove numerical drift so that ``sum(alpha) == 0`` holds exactly per girone."""
        out = np.array(theta, dtype=float, copy=True)
        for grp in index.alpha_groups():
            out[grp] -= out[grp].mean()
        return out

    def _initial_theta(self, data: LikelihoodData, index: TeamIndex, warm_start: FittedParams | None) -> np.ndarray:
        cfg = self.config
        n_t, n_g = data.n_teams, data.n_gammas
        theta0: np.ndarray | None = None
        if warm_start is not None:
            if set(warm_start.teams) == set(index.teams):
                alpha0 = np.array([warm_start.alpha[t] for t in index.teams])
                beta0 = np.array([warm_start.beta[t] for t in index.teams])
                if cfg.girone_specific_home:
                    gamma0 = np.array([warm_start.gamma.get(g, float(np.mean(list(warm_start.gamma.values())))) for g in index.gironi])
                else:
                    gamma0 = np.array([float(np.mean(list(warm_start.gamma.values())))])
                theta0 = pack_theta(alpha0, beta0, gamma0, warm_start.rho)
                log.info("warm start from previous fit", extra={"n_teams": n_t})
            else:
                log.warning(
                    "warm_start ignored: team set differs from the current matches",
                    extra={
                        "missing_from_warm_start": sorted(set(index.teams) - set(warm_start.teams)),
                        "extra_in_warm_start": sorted(set(warm_start.teams) - set(index.teams)),
                    },
                )
        if theta0 is None:
            theta0 = self._moment_start(data, index)
        theta0 = self._clip_to_bounds(theta0, self._bounds(n_t, n_g))
        return self._recentre_alpha(theta0, index)

    @staticmethod
    def _moment_start(data: LikelihoodData, index: TeamIndex) -> np.ndarray:
        """Shrunken log-rate moments: a cheap, well-scaled starting point."""
        n_t, n_g = data.n_teams, data.n_gammas
        w, x, y = data.w, data.x.astype(float), data.y.astype(float)
        total_w = max(float(w.sum()), 1e-12)
        avg_goals = max((float(np.dot(w, x)) + float(np.dot(w, y))) / (2.0 * total_w), 1e-3)
        scored = np.bincount(data.home_idx, weights=w * x, minlength=n_t) + np.bincount(data.away_idx, weights=w * y, minlength=n_t)
        conceded = np.bincount(data.home_idx, weights=w * y, minlength=n_t) + np.bincount(data.away_idx, weights=w * x, minlength=n_t)
        games = np.bincount(data.home_idx, weights=w, minlength=n_t) + np.bincount(data.away_idx, weights=w, minlength=n_t)
        shrink = 2.0  # pseudo-matches at the league average
        alpha0 = np.log((scored + shrink * avg_goals) / (games + shrink)) - math.log(avg_goals)
        beta0 = -(np.log((conceded + shrink * avg_goals) / (games + shrink)) - math.log(avg_goals))
        home_total = max(float(np.dot(w, x)), 1e-6)
        away_total = max(float(np.dot(w, y)), 1e-6)
        gamma0 = np.full(n_g, math.log(home_total / away_total))
        return pack_theta(alpha0, beta0, gamma0, 0.0)

    def _fit_slsqp(
        self, theta0: np.ndarray, data: LikelihoodData, index: TeamIndex, scale: float
    ) -> tuple[OptimizeResult, np.ndarray]:
        cfg = self.config
        bounds = self._bounds(data.n_teams, data.n_gammas)
        constraints: list[dict[str, Any]] = []
        for grp in index.alpha_groups():
            jac_row = np.zeros(theta0.size)
            jac_row[grp] = 1.0
            constraints.append(
                {
                    "type": "eq",
                    "fun": (lambda th, g=grp: float(np.sum(th[g]))),
                    "jac": (lambda th, j=jac_row: j),
                }
            )

        def objective(th: np.ndarray) -> tuple[float, np.ndarray]:
            f, g = negative_log_likelihood_and_grad(th, data)
            return f * scale, g * scale

        result = minimize(
            objective,
            theta0,
            jac=True,
            method="SLSQP",
            bounds=bounds,
            constraints=constraints,
            options={"maxiter": cfg.max_iter, "ftol": cfg.tol},
        )
        theta = self._clip_to_bounds(np.asarray(result.x, dtype=float), bounds)
        return result, theta

    def _fit_lbfgsb(
        self, theta0: np.ndarray, data: LikelihoodData, index: TeamIndex, scale: float
    ) -> tuple[OptimizeResult, np.ndarray]:
        cfg = self.config
        reparam = _AlphaReparam(index)
        full_bounds = self._bounds(data.n_teams, data.n_gammas)
        free_bounds = [full_bounds[i] for i in reparam.free] + full_bounds[data.n_teams :]

        def objective(th_free: np.ndarray) -> tuple[float, np.ndarray]:
            f, g = negative_log_likelihood_and_grad(reparam.free_to_full(th_free), data)
            return f * scale, reparam.grad_full_to_free(g) * scale

        result = minimize(
            objective,
            reparam.full_to_free(theta0),
            jac=True,
            method="L-BFGS-B",
            bounds=free_bounds,
            options={
                "maxiter": cfg.max_iter,
                "maxfun": 20 * cfg.max_iter,
                "ftol": cfg.tol,
                "gtol": math.sqrt(cfg.tol),
                "maxcor": 20,
            },
        )
        theta = reparam.free_to_full(np.asarray(result.x, dtype=float))
        return result, theta

    # ------------------------------------------------------------- inference
    def _lookup(self, mapping: Mapping[str, float], team: str) -> float:
        try:
            return mapping[team]
        except KeyError as exc:
            raise KeyError(_unknown_team_message(team, mapping.keys())) from exc

    def rates(self, home: str, away: str, girone: Girone | str) -> tuple[float, float]:
        """``(lambda, mu)`` for ``home`` vs ``away`` played in ``girone``.

        ``KeyError`` with a helpful message for an unknown team or a girone with
        no home-advantage estimate.
        """
        params = self.params_
        if home == away:
            raise ValueError(f"home and away team must differ, got {home!r} twice")
        g = Girone.parse(girone)
        if g not in params.gamma:
            raise KeyError(f"no home-advantage estimate for girone {g.value}; fitted gironi: {[k.value for k in params.gamma]}")
        alpha_h = self._lookup(params.alpha, home)
        alpha_a = self._lookup(params.alpha, away)
        beta_h = self._lookup(params.beta, home)
        beta_a = self._lookup(params.beta, away)
        lam = math.exp(min(max(alpha_h - beta_a + params.gamma[g], -LOG_RATE_CLIP), LOG_RATE_CLIP))
        mu = math.exp(min(max(alpha_a - beta_h, -LOG_RATE_CLIP), LOG_RATE_CLIP))
        return lam, mu

    def score_matrix(self, home: str, away: str, girone: Girone | str) -> np.ndarray:
        """Tau-corrected joint score matrix of shape ``(max_goals+1, max_goals+1)``."""
        params = self.params_
        lam, mu = self.rates(home, away, girone)
        return dixon_coles_pmf(lam, mu, params.rho, params.max_goals)

    def market_probabilities(self, home: str, away: str, girone: Girone | str) -> dict[Market, dict[str, float]]:
        """Probabilities for every market in :data:`SELECTIONS` (each sums to 1)."""
        return markets_from_matrix(self.score_matrix(home, away, girone))

    @staticmethod
    def _fair_odds_from(probabilities: Mapping[Market, Mapping[str, float]]) -> dict[Market, dict[str, float]]:
        return {
            market: {sel: (1.0 / p if p > 0.0 else float("inf")) for sel, p in sels.items()}
            for market, sels in probabilities.items()
        }

    def fair_odds(self, home: str, away: str, girone: Girone | str) -> dict[Market, dict[str, float]]:
        """Fair (no-margin) decimal odds ``1 / p`` (``inf`` when ``p == 0``)."""
        return self._fair_odds_from(self.market_probabilities(home, away, girone))

    def predict(self, fixture: Fixture) -> MatchPrediction:
        """Full :class:`MatchPrediction` for one fixture."""
        params = self.params_
        lam, mu = self.rates(fixture.home_team, fixture.away_team, fixture.girone)
        matrix = dixon_coles_pmf(lam, mu, params.rho, params.max_goals)
        probabilities = markets_from_matrix(matrix)
        return MatchPrediction(
            fixture=fixture,
            home_rate=lam,
            away_rate=mu,
            rho=params.rho,
            matrix=matrix,
            probabilities=probabilities,
            fair_odds=self._fair_odds_from(probabilities),
        )

    def predict_fixtures(self, fixtures: Iterable[Fixture]) -> list[MatchPrediction]:
        """Predict every fixture, preserving order."""
        return [self.predict(fx) for fx in fixtures]

    # ------------------------------------------------------------- reporting
    def ranking(self, girone: Girone | str | None = None) -> pd.DataFrame:
        """Strength table sorted by ``alpha`` (desc) then ``beta`` (desc).

        Columns: ``team, girone, alpha, beta, gamma, attack_index, defence_index,
        xg_for_vs_avg, xg_against_vs_avg`` where ``attack_index = exp(alpha)``,
        ``defence_index = exp(beta)``, ``xg_for_vs_avg = exp(alpha) * mean_girone(exp(-beta))``
        and ``xg_against_vs_avg = exp(-beta) * mean_girone(exp(alpha))``.
        """
        params = self.params_
        frame = params.to_frame()
        if girone is not None:
            g = Girone.parse(girone)
            if g not in params.teams_by_girone:
                raise KeyError(f"no teams fitted for girone {g.value}; fitted gironi: {[k.value for k in params.teams_by_girone]}")
            frame = frame.loc[frame["girone"] == g.value]
        attack = np.exp(frame["alpha"].to_numpy())
        defence = np.exp(frame["beta"].to_numpy())
        frame = frame.assign(
            attack_index=attack,
            defence_index=defence,
            _exp_neg_beta=np.exp(-frame["beta"].to_numpy()),
        )
        mean_exp_neg_beta = frame.groupby("girone", observed=True)["_exp_neg_beta"].transform("mean").to_numpy()
        mean_exp_alpha = frame.groupby("girone", observed=True)["attack_index"].transform("mean").to_numpy()
        frame = frame.assign(
            xg_for_vs_avg=attack * mean_exp_neg_beta,
            xg_against_vs_avg=frame["_exp_neg_beta"].to_numpy() * mean_exp_alpha,
        ).drop(columns=["_exp_neg_beta"])
        frame = frame.sort_values(["alpha", "beta"], ascending=[False, False], kind="stable").reset_index(drop=True)
        return frame[
            [
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
        ]

    def residuals(self, matches: pd.DataFrame) -> pd.DataFrame:
        """Per-match diagnostics: fitted rates, expected vs actual goals, weights.

        Teams unknown to the fit yield ``NaN`` rates (a warning is logged with the
        count). ``log_score_prob`` is the log of the (untruncated) Dixon-Coles
        probability of the observed score.
        """
        params = self.params_
        df = validate_matches_df(matches)
        alpha_h = pd.to_numeric(df["home_team"].map(params.alpha), errors="coerce").to_numpy(dtype=float)
        alpha_a = pd.to_numeric(df["away_team"].map(params.alpha), errors="coerce").to_numpy(dtype=float)
        beta_h = pd.to_numeric(df["home_team"].map(params.beta), errors="coerce").to_numpy(dtype=float)
        beta_a = pd.to_numeric(df["away_team"].map(params.beta), errors="coerce").to_numpy(dtype=float)
        gamma = pd.to_numeric(df["girone"].map({g.value: v for g, v in params.gamma.items()}), errors="coerce").to_numpy(dtype=float)
        unknown = np.isnan(alpha_h) | np.isnan(alpha_a) | np.isnan(gamma)
        if unknown.any():
            log.warning("residuals: matches with teams or gironi unknown to the fit", extra={"n_unknown": int(unknown.sum())})
        with np.errstate(invalid="ignore", over="ignore"):
            lam = np.exp(np.clip(alpha_h - beta_a + gamma, -LOG_RATE_CLIP, LOG_RATE_CLIP))
            mu = np.exp(np.clip(alpha_a - beta_h, -LOG_RATE_CLIP, LOG_RATE_CLIP))
            x = df["home_goals"].to_numpy(dtype=float)
            y = df["away_goals"].to_numpy(dtype=float)
            t = np.clip(np.asarray(tau(x, y, lam, mu, params.rho), dtype=float), TAU_FLOOR, None)
            log_prob = np.log(t) + xlogy(x, lam) - lam - gammaln(x + 1.0) + xlogy(y, mu) - mu - gammaln(y + 1.0)
        weights = time_weights(df["date"], params.reference_date, params.xi)
        out = pd.DataFrame(
            {
                "match_id": df["match_id"].to_numpy(),
                "date": df["date"].to_numpy(),
                "girone": df["girone"].to_numpy(),
                "home_team": df["home_team"].to_numpy(),
                "away_team": df["away_team"].to_numpy(),
                "home_goals": df["home_goals"].to_numpy(),
                "away_goals": df["away_goals"].to_numpy(),
                "lam": lam,
                "mu": mu,
                "expected_total": lam + mu,
                "actual_total": x + y,
                "home_residual": x - lam,
                "away_residual": y - mu,
                "total_residual": (x + y) - (lam + mu),
                "weight": weights,
                "log_score_prob": log_prob,
            }
        )
        for c in ("match_id", "girone", "home_team", "away_team"):
            out[c] = out[c].astype("string")
        return out

    # ----------------------------------------------------------- persistence
    def save(self, path: str | Path) -> Path:
        """Persist the configuration and the fitted parameters as JSON."""
        params = self.params_
        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        payload = {"schema_version": _SCHEMA_VERSION, "config": self.config.to_dict(), "params": params.to_dict()}
        out.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
        log.info("engine saved", extra={"path": str(out)})
        return out

    @classmethod
    def load(cls, path: str | Path) -> "DixonColesEngine":
        """Restore an engine (config + parameters) written by :meth:`save`.

        A bare :meth:`FittedParams.to_json` file is accepted too; the config is
        then reconstructed from the parameters' ``xi``, ``method`` and ``max_goals``.
        """
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        if "params" in payload and isinstance(payload["params"], Mapping):
            params = FittedParams.from_dict(payload["params"])
            config = DixonColesConfig.from_dict(payload.get("config", {}))
        else:
            params = FittedParams.from_dict(payload)
            config = DixonColesConfig(xi=params.xi, method=params.method, max_goals=params.max_goals)  # type: ignore[arg-type]
        engine = cls(config)
        engine._params = params
        log.info("engine loaded", extra={"path": str(path), "n_teams": len(params.teams)})
        return engine
