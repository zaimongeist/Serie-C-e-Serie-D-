"""Dixon-Coles engine for the configured competitions (Serie C Gironi A / B / C and Serie B).

Model
-----
For a match between ``home`` and ``away`` played in competition ``g`` (a
*girone* in the code: a Serie C girone or Serie B, whatever
``config/competitions.toml`` lists) the goal rates are log-linear::

    lambda = exp(alpha[home] - beta[away] + gamma[g])     # expected home goals
    mu     = exp(alpha[away] - beta[home])                # expected away goals

``alpha`` is attack strength, ``beta`` is defensive solidity (a *higher* beta
concedes *less*), ``gamma`` is the home advantage estimated separately for each
competition (or shared, see :attr:`DixonColesConfig.girone_specific_home`) and
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
others) for L-BFGS-B. The reparametrised problem cannot box-bound the dependent
alpha of a girone, so when an L-BFGS-B solution leaves one outside
``param_bounds`` the engine re-solves the bounded, constrained problem with
SLSQP and reports the fallback (``FittedParams.method == 'SLSQP'`` plus a note in
``message``). Both methods return the same :class:`FittedParams` structure. An
optional ridge penalty ``l2_penalty * (sum alpha^2 + sum (beta - mean beta)^2)``
(betas centred within their girone, because the mean beta is the model's
intercept) is added to the objective to stabilise very small datasets; ``l2_penalty_home`` and
``l2_penalty_rho`` extend it to a full Gaussian (MAP) prior that shrinks the
home-advantage parameters towards ``home_prior`` and rho towards 0, which keeps a
20-match sample from driving gamma or rho onto their bounds.

Time-decay weights can *collapse*: one mistyped far-future date (which becomes
the default reference date) or an explicit reference date far after the sample
makes every other match weigh ``exp(-xi * thousands of days)``, so the
optimiser "converges" on a handful of matches or on pure rounding noise. After
computing the weights :meth:`DixonColesEngine.fit` therefore runs
:func:`time_weight_diagnostics`: when ``effective_n`` (the sum of the weights)
falls below ``min_effective_n_ratio * n_matches`` the fit is flagged as
degenerate (``FittedParams.degenerate_weights``, ``converged=False`` and an
explanatory ``message``), a warning is logged, or ``ValueError`` is raised with
``on_weight_collapse='raise'``. A reference date before the first match (which
silently turns the decay off) or far after the last one is logged and recorded
in ``FittedParams.fit_warnings`` as well.

Input frames are validated with :func:`src.common.schemas.validate_matches_df`
and then range-checked: negative goals and rows with ``home_team == away_team``
raise ``ValueError`` (the same rules :class:`src.common.schemas.MatchRecord`
enforces), so sentinel values from harvested data can never corrupt a fit.

Perimeter: current and historical teams
---------------------------------------
A fit may use several seasons of history (the time decay already down-weights
old matches). When a :class:`src.common.teams.TeamRegistry` is passed to
:meth:`DixonColesEngine.fit` (or :func:`build_team_index`) the registry defines
the *perimeter*: every registry team is a **current team** and is assigned to
its registry girone even if its past matches were played in another girone
(Inter U23 moved from A to C, Juventus Next Gen from B to A, ...); every other
team in the frame is a **historical team** (relegated, promoted, excluded): it
stays in the likelihood as an opponent, so the strengths of the current teams
are estimated against the right opposition, but it is excluded from
:meth:`DixonColesEngine.ranking` by default and reported in
``FittedParams.historical_teams``. The ``sum(alpha) == 0`` constraint is
applied per *constraint group*: the registry girone for current teams, while a
historical team joins the group of the girone it played most matches in (ties
broken by letter order), exactly as a team without registry would. The home
advantage ``gamma`` of a match is always the one of the girone the match was
played in (the frame's ``girone`` column), never the team's current girone.
Without a registry every fitted team is current (the behaviour of earlier
releases is unchanged).

Several divisions in one fit
----------------------------
Nothing in the engine is specific to three gironi: competitions are whatever
the dynamic :class:`src.common.schemas.Girone` enum lists (``A``, ``B``, ``C``
and ``SB`` for Serie B by default), each with its own home advantage and its
own ``sum(alpha) == 0`` constraint group, in configuration order (also the
tie-break order when a historical team played equally often in two
competitions). A club promoted or relegated between Serie B and Serie C keeps
**one canonical name** across seasons (``'Pescara'`` in Serie B 2025/26 and in
girone B 2026/27): the registry of the current season decides its group, and
its matches in the other division stay in the likelihood. Within one season
the divisions never meet, so each block is identified by its own constraint
and its own goal level; it is through these *movers* that the joint fit across
divisions is identified: a relegated club carries the strength it showed
against Serie B opposition into its Serie C group, and the Serie B strengths
are estimated against the right opposition when a promoted club's Serie C
history is in the frame. The centring is still per competition, so an alpha is
read relative to the average of *its* division (a Serie B average and a Serie C
average both sit at 0): compare teams across divisions through the movers'
matches, not through the raw alphas.

A registry competition may have current teams but no match at all in the frame
(Serie B clubs that appear only in Serie C history because Serie B has not
been harvested yet). Those teams stay current and keep their registry
competition in every report, but no home advantage exists for it and they
join, for the constraint only, the group of the competition they played most
matches in (:attr:`TeamIndex.constraint_girone`): centring the promoted
clubs on their own would force the strongest teams of the lower division to
average zero. The fit logs a warning; a fixture of that competition is not
predictable until its matches are harvested (``KeyError`` from
:meth:`DixonColesEngine.rates`, skipped with a note by the callers).

The objective exposes an analytic gradient (:func:`negative_log_likelihood_grad`)
which is passed to :func:`scipy.optimize.minimize`.
"""
from __future__ import annotations

import json
import math
import time
from dataclasses import MISSING, asdict, dataclass, field, fields
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
from src.common.teams import TeamRegistry

__all__ = [
    "DixonColesConfig",
    "DixonColesEngine",
    "FittedParams",
    "LikelihoodData",
    "NotFittedError",
    "TeamIndex",
    "WeightDiagnostics",
    "build_team_index",
    "dixon_coles_pmf",
    "markets_from_matrix",
    "negative_log_likelihood",
    "negative_log_likelihood_and_grad",
    "negative_log_likelihood_grad",
    "pack_theta",
    "prepare_likelihood_data",
    "tau",
    "time_weight_diagnostics",
    "time_weights",
    "unpack_theta",
    "TAU_FLOOR",
    "LOG_RATE_CLIP",
    "METHODS",
    "WEIGHT_COLLAPSE_POLICIES",
]

log = get_logger(__name__)

#: ``tau`` is clipped to this floor before taking its logarithm.
TAU_FLOOR: Final[float] = 1e-10
#: Log-rates are clipped to ``[-LOG_RATE_CLIP, LOG_RATE_CLIP]`` before ``exp`` (overflow guard).
LOG_RATE_CLIP: Final[float] = 50.0
#: Supported scipy optimisers.
METHODS: Final[tuple[str, ...]] = ("SLSQP", "L-BFGS-B")
#: What :meth:`DixonColesEngine.fit` does when the time-decay weights collapse.
WEIGHT_COLLAPSE_POLICIES: Final[tuple[str, ...]] = ("warn", "raise")

MethodName = Literal["SLSQP", "L-BFGS-B"]
WeightCollapsePolicy = Literal["warn", "raise"]

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


def _normalised_dates(dates: pd.Series | np.ndarray | Iterable[Any]) -> pd.Series:
    """``dates`` as a naive, midnight-normalised datetime Series (``ValueError`` on NaT)."""
    if isinstance(dates, pd.Series):
        series = dates
    elif isinstance(dates, np.ndarray):
        series = pd.Series(dates)
    else:
        series = pd.Series(list(dates))
    series = pd.to_datetime(series)
    if series.dt.tz is not None:
        series = series.dt.tz_localize(None)
    series = series.dt.normalize().reset_index(drop=True)
    if series.isna().any():
        raise ValueError("dates contain missing values (NaT)")
    return series


@dataclass(frozen=True)
class WeightDiagnostics:
    """Health summary of the time-decay weights of one fit (see :func:`time_weight_diagnostics`).

    ``effective_n`` is the sum of the weights and ``effective_n_ratio`` its share
    of ``n_matches`` (1.0 when ``xi == 0``). ``lead_days`` is how many days the
    reference date lies *after* the latest match (negative when it falls inside
    or before the sample) and ``newest_weight`` the weight a match played on the
    latest date receives. ``latest_gap_days`` is the distance between the latest
    match date and the second most recent distinct date, and
    ``second_latest_weight_ratio = exp(-xi * latest_gap_days)`` how much lighter
    that second date is: a huge gap with a tiny ratio is the signature of a
    mistyped far-future date. ``n_for_95pct`` is the smallest number of matches
    whose weights add up to 95% of the total (1 when a single match carries
    everything); ``heaviest_match_id``/``heaviest_date`` identify the match with
    the largest weight.
    """

    n_matches: int
    effective_n: float
    effective_n_ratio: float
    reference_date: date
    earliest_date: date
    latest_date: date
    lead_days: int
    newest_weight: float
    latest_gap_days: int
    second_latest_weight_ratio: float
    n_for_95pct: int
    heaviest_match_id: str | None
    heaviest_date: date

    def is_collapsed(self, min_effective_n_ratio: float) -> bool:
        """True when ``effective_n < min_effective_n_ratio * n_matches`` (never when the ratio is 0)."""
        if not min_effective_n_ratio > 0.0:
            return False
        return bool(self.effective_n < float(min_effective_n_ratio) * self.n_matches)

    def describe(self, min_effective_n_ratio: float) -> str:
        """Human-readable diagnosis of a collapse, naming the most likely cause."""
        parts = [
            (
                f"time-decay weights collapsed: effective_n = {self.effective_n:.3g} of {self.n_matches} matches "
                f"(ratio {self.effective_n_ratio:.2e} < min_effective_n_ratio {float(min_effective_n_ratio):g})"
            )
        ]
        if self.lead_days > 0 and self.newest_weight < 0.5:
            parts.append(
                f"reference_date {self.reference_date.isoformat()} is {self.lead_days} days after the latest match "
                f"({self.latest_date.isoformat()}), so even the newest match weighs only {self.newest_weight:.2e}"
            )
        if self.n_for_95pct < 0.5 * self.n_matches:
            who = f"match_id {self.heaviest_match_id!r} " if self.heaviest_match_id is not None else ""
            parts.append(
                f"{self.n_for_95pct} match(es) carry 95% of the total weight "
                f"(heaviest: {who}dated {self.heaviest_date.isoformat()})"
            )
            if self.latest_gap_days > 0 and self.second_latest_weight_ratio < 0.05:
                parts.append(
                    f"the latest match date is {self.latest_gap_days} days after the second most recent one, "
                    f"which weighs {self.second_latest_weight_ratio:.2e} relative to it: probable mistyped date"
                )
        parts.append(
            "the estimates are not reliable: fix mistyped dates, pass a reference_date inside or just after the sample, or lower xi"
        )
        return "; ".join(parts)

    def to_dict(self) -> dict[str, Any]:
        """JSON-friendly representation (dates ISO) for structured logging."""
        payload = asdict(self)
        for key in ("reference_date", "earliest_date", "latest_date", "heaviest_date"):
            payload[key] = payload[key].isoformat()
        return payload


def time_weight_diagnostics(
    dates: pd.Series | np.ndarray | Iterable[Any],
    weights: np.ndarray | Iterable[float],
    reference_date: date | datetime | pd.Timestamp,
    xi: float,
    *,
    match_ids: pd.Series | np.ndarray | Iterable[Any] | None = None,
) -> WeightDiagnostics:
    """Summarise how the time-decay ``weights`` of ``dates`` are distributed.

    ``weights`` are normally :func:`time_weights` of the same dates (passing
    them avoids recomputing); ``match_ids`` (same length) lets the heaviest
    match be named. ``ValueError`` for an empty sample or mismatched lengths.
    """
    series = _normalised_dates(dates)
    w = np.asarray(list(weights) if not isinstance(weights, np.ndarray) else weights, dtype=float).ravel()
    n = int(series.size)
    if n == 0:
        raise ValueError("no matches: cannot diagnose time-decay weights")
    if w.shape != (n,):
        raise ValueError(f"weights must have one entry per date ({n}), got shape {w.shape}")
    if not np.all(np.isfinite(w)) or (w < 0).any():
        raise ValueError("weights must be finite and non-negative")
    ids: list[str] | None = None
    if match_ids is not None:
        ids = [str(m) for m in (match_ids.tolist() if hasattr(match_ids, "tolist") else list(match_ids))]
        if len(ids) != n:
            raise ValueError(f"match_ids must have one entry per date ({n}), got {len(ids)}")
    ref = pd.Timestamp(reference_date)
    if ref.tzinfo is not None:
        ref = ref.tz_localize(None)
    ref = ref.normalize()
    latest = series.max()
    earliest = series.min()
    total = float(w.sum())
    lead_days = int((ref - latest).days)
    newest_weight = float(np.exp(-float(xi) * max(lead_days, 0)))
    distinct = np.sort(series.unique())
    latest_gap_days = int((distinct[-1] - distinct[-2]) / np.timedelta64(1, "D")) if distinct.size >= 2 else 0
    second_ratio = float(np.exp(-float(xi) * latest_gap_days))
    order = np.argsort(-w, kind="stable")
    if total > 0.0:
        cumulative = np.cumsum(w[order])
        n_for_95 = int(min(np.searchsorted(cumulative, 0.95 * total, side="left") + 1, n))
    else:
        n_for_95 = n
    heaviest = int(order[0])
    return WeightDiagnostics(
        n_matches=n,
        effective_n=total,
        effective_n_ratio=total / n,
        reference_date=ref.date(),
        earliest_date=earliest.date(),
        latest_date=latest.date(),
        lead_days=lead_days,
        newest_weight=newest_weight,
        latest_gap_days=latest_gap_days,
        second_latest_weight_ratio=second_ratio,
        n_for_95pct=n_for_95,
        heaviest_match_id=ids[heaviest] if ids is not None else None,
        heaviest_date=series.iloc[heaviest].date(),
    )


def _reference_date_note(diag: WeightDiagnostics, xi: float) -> str:
    """Warning text for a reference date outside the sample (empty when there is nothing to say)."""
    if diag.reference_date < diag.earliest_date:
        return (
            f"reference_date {diag.reference_date.isoformat()} precedes the earliest match ({diag.earliest_date.isoformat()}): "
            f"every match gets weight 1 and xi={xi:g} has no effect (equivalent to xi=0)"
        )
    if diag.lead_days > 0 and diag.newest_weight < 0.5:
        return (
            f"reference_date {diag.reference_date.isoformat()} is {diag.lead_days} days after the latest match "
            f"({diag.latest_date.isoformat()}): every match is discounted by at least {1.0 - diag.newest_weight:.0%} "
            f"(newest weight {diag.newest_weight:.3g}), which makes an active Gaussian prior {1.0 / diag.newest_weight:.1f}x "
            "stronger relative to the data; pass a reference_date closer to the sample unless this is intended"
        )
    return ""


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
    l2_penalty: float  #: ridge strength on alpha and beta (shrinks towards 0)
    l2_penalty_home: float = 0.0  #: ridge strength on gamma (shrinks towards ``home_prior``)
    home_prior: float = 0.0  #: prior mean of the home-advantage parameter(s)
    l2_penalty_rho: float = 0.0  #: ridge strength on rho (shrinks towards 0)
    team_girone_idx: np.ndarray | None = None  #: girone position of each team; centres the beta ridge per girone


_N_REQUIRED_FIELDS = len(LikelihoodData._fields) - len(LikelihoodData._field_defaults)


def _as_data(data: tuple[Any, ...]) -> LikelihoodData:
    if len(data) == 1 and isinstance(data[0], LikelihoodData):
        return data[0]
    if not _N_REQUIRED_FIELDS <= len(data) <= len(LikelihoodData._fields):
        raise TypeError(
            f"expected a LikelihoodData instance or {_N_REQUIRED_FIELDS}-{len(LikelihoodData._fields)} of its fields, "
            f"got {len(data)} positional arguments"
        )
    return LikelihoodData(*data)


@dataclass(frozen=True)
class TeamIndex:
    """Team <-> parameter position bookkeeping for one fit.

    ``teams`` is the global team order: the competitions in the order of
    ``config/competitions.toml``, alphabetical within a competition. ``gironi``
    are the competitions that appear in the match frame, in that same order;
    they index the home-advantage parameters when ``girone_specific_home`` is
    true (otherwise there is a single shared gamma). ``teams_by_girone`` is the
    *reported* competition of every team (the registry competition for current
    teams) and, normally, also its ``sum(alpha) == 0`` constraint group.
    ``historical`` lists the fitted teams that are not part of the current
    perimeter (see :func:`build_team_index` with a registry); it is empty when no
    registry was used.

    A registry competition can have current teams but no match at all in the
    frame (Serie B clubs that appear only in Serie C history because Serie B has
    not been harvested yet). Such a group has no home advantage to estimate and
    must not be centred on its own (its members are typically the promoted,
    i.e. strongest, teams of the lower division): ``constraint_girone`` maps
    each of those teams to the competition it played most matches in, whose
    constraint group it joins. :meth:`constraint_groups` gives the effective
    groups, :meth:`constraint_girone_of` the group of one team, while
    :meth:`girone_of` keeps answering the reported competition. Building an
    index with such a group and no override raises ``ValueError``.
    """

    teams: tuple[str, ...]
    teams_by_girone: dict[Girone, tuple[str, ...]]
    gironi: tuple[Girone, ...]
    girone_specific_home: bool = True
    historical: tuple[str, ...] = ()
    constraint_girone: dict[str, Girone] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if len(set(self.teams)) != len(self.teams):
            raise ValueError("duplicate team names in TeamIndex.teams")
        assigned = [t for g in self.teams_by_girone.values() for t in g]
        if sorted(assigned) != sorted(self.teams):
            raise ValueError("teams_by_girone does not partition TeamIndex.teams")
        overrides = {str(t): Girone.parse(g) for t, g in self.constraint_girone.items()}
        unknown_override = sorted(set(overrides) - set(self.teams))
        if unknown_override:
            raise ValueError(f"constraint_girone names teams not in TeamIndex.teams: {unknown_override}")
        bad_target = sorted(t for t, g in overrides.items() if g not in self.gironi)
        if bad_target:
            raise ValueError(f"constraint_girone must point to a girone with matches; offending teams: {bad_target}")
        for g, members in self.teams_by_girone.items():
            if g not in self.gironi:
                orphans = [t for t in members if t not in overrides]
                if orphans:
                    raise ValueError(
                        f"girone {g.value} has teams but no matches ({orphans[:5]}{', ...' if len(orphans) > 5 else ''}): "
                        "no home advantage can be estimated; pass constraint_girone to attach them to a girone with matches"
                    )
        unknown = sorted(set(self.historical) - set(self.teams))
        if unknown:
            raise ValueError(f"historical teams not in TeamIndex.teams: {unknown}")
        object.__setattr__(self, "historical", tuple(str(t) for t in self.historical))
        object.__setattr__(self, "constraint_girone", overrides)

    @property
    def n_teams(self) -> int:
        return len(self.teams)

    @property
    def current_teams(self) -> tuple[str, ...]:
        """Fitted teams that belong to the perimeter (every team without a registry)."""
        gone = set(self.historical)
        return tuple(t for t in self.teams if t not in gone)

    def is_current(self, team: str) -> bool:
        """True for a fitted team of the perimeter; False for historical or unknown teams."""
        return team in self.teams and team not in set(self.historical)

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

    def constraint_girone_of(self, team: str) -> Girone:
        """Competition whose ``sum(alpha) == 0`` group ``team`` belongs to (its reported one unless overridden)."""
        override = self.constraint_girone.get(team)
        return override if override is not None else self.girone_of(team)

    def constraint_groups(self) -> dict[Girone, tuple[str, ...]]:
        """Effective ``sum(alpha) == 0`` groups, in ``teams_by_girone`` order.

        The reported groups of the competitions that have matches, each extended
        with the teams :attr:`constraint_girone` attaches to it.
        """
        groups: dict[Girone, list[str]] = {g: list(members) for g, members in self.teams_by_girone.items() if g in self.gironi}
        for g, members in self.teams_by_girone.items():
            if g in self.gironi:
                continue
            for team in members:
                groups.setdefault(self.constraint_girone[team], []).append(team)
        return {g: tuple(members) for g, members in groups.items() if members}

    def team_girone_positions(self) -> np.ndarray:
        """For every team, the position inside ``gironi`` of its constraint competition (centres the beta ridge)."""
        girone_pos = {g: i for i, g in enumerate(self.gironi)}
        return np.array([girone_pos[self.constraint_girone_of(t)] for t in self.teams], dtype=np.int64)

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
        """Team positions of every constraint group (``sum(alpha) == 0`` is imposed on each; see :meth:`constraint_groups`)."""
        positions = self.team_positions()
        return [np.array([positions[t] for t in members], dtype=np.int64) for members in self.constraint_groups().values() if members]


def build_team_index(
    matches: pd.DataFrame,
    *,
    girone_specific_home: bool = True,
    min_matches_per_team: int = 1,
    registry: TeamRegistry | None = None,
) -> TeamIndex:
    """Build the :class:`TeamIndex` of a *validated* match frame.

    Without ``registry`` a team appearing in more than one competition is
    assigned to the one where it played most matches (ties broken by the order
    of ``config/competitions.toml``) and a warning is logged. With a ``registry``
    every team of the registry (matched by its canonical name) is a *current*
    team and is assigned to its **registry** competition, whatever competition
    its matches were played in (a club that moved girone, or division, between
    seasons is logged at INFO level, not warned about); every other team is
    *historical*: it is still estimated, listed in ``TeamIndex.historical`` and
    placed in the constraint group of the competition it played most matches in.
    When a registry competition has current teams but no match in the frame
    (Serie B not harvested yet), those teams keep their registry competition but
    join, for the ``sum(alpha) == 0`` constraint only, the group of the
    competition they played in (``TeamIndex.constraint_girone``, logged as a
    warning); no home advantage is estimated for that competition. The
    opposite case -- a competition with matches whose registry entry has no
    current team yet (a Serie D girone before its auto-init) -- is handled
    without error: its teams are historical, fitted in that competition's own
    group with its own home advantage, and a warning is logged. A configured
    competition with neither teams nor matches is simply absent from the index
    (no gamma, no group). Teams with fewer than ``min_matches_per_team`` matches
    are still estimated but reported in the log. Only teams that appear in
    ``matches`` are indexed: a registry team without any match has no
    identifiable strength.
    """
    long = pd.concat(
        [
            matches[["home_team", "girone"]].rename(columns={"home_team": "team"}),
            matches[["away_team", "girone"]].rename(columns={"away_team": "team"}),
        ],
        ignore_index=True,
    )
    order = {g.value: i for i, g in enumerate(Girone)}
    counts = long.groupby(["team", "girone"], observed=True).size().reset_index(name="n")
    counts = counts.assign(_pos=counts["girone"].astype(str).map(order))
    counts = counts.sort_values(["team", "n", "_pos"], ascending=[True, False, True], kind="stable")
    best = counts.drop_duplicates(subset="team", keep="first")
    played_in: dict[str, Girone] = {str(t): Girone.parse(g) for t, g in zip(best["team"], best["girone"])}
    assignment: dict[str, Girone] = dict(played_in)

    current_set = set(registry.canonical_names) if registry is not None else set(assignment)
    historical = tuple(sorted(t for t in assignment if t not in current_set))
    if registry is not None:
        moved: dict[str, dict[str, str]] = {}
        for team in list(assignment):
            if team in current_set:
                registry_girone = registry.girone_of(team)
                if assignment[team] is not registry_girone:
                    moved[team] = {"played_in": assignment[team].value, "registry": registry_girone.value}
                assignment[team] = registry_girone
        if moved:
            log.info(
                "current teams assigned to their registry girone (matches played in another girone)",
                extra={"season": registry.season_label, "teams": moved},
            )
        if historical:
            log.info(
                "historical teams (not in the registry) kept as opponents, excluded from the perimeter",
                extra={"season": registry.season_label, "n_historical": len(historical), "teams": list(historical)},
            )

    per_team_gironi = counts.groupby("team", observed=True).size()
    for team in per_team_gironi[per_team_gironi > 1].index:
        if registry is not None and str(team) in current_set:
            continue  # a current team is placed by the registry, not by where it played
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
    if registry is not None:
        # A competition with matches but no current team in the registry (a Serie D girone whose
        # auto-init has not run yet, or raced with the harvest): nothing crashes, its teams are
        # simply all historical -- fitted, grouped and given a home advantage, but outside the
        # perimeter -- until the dictionary is filled. Loud, because rankings will skip them.
        uninitialised = [g for g in gironi if not registry.teams(g)]
        if uninitialised:
            log.warning(
                "competitions with matches but no current teams in the registry: their teams are fitted as historical",
                extra={
                    "season": registry.season_label,
                    "gironi": [g.value for g in uninitialised],
                    "teams": {g.value: len(teams_by_girone.get(g, ())) for g in uninitialised},
                },
            )
    constraint_girone: dict[str, Girone] = {}
    missing = [g for g in teams_by_girone if g not in gironi]
    if missing:
        # Only possible with a registry: current teams of a registry competition whose
        # matches were all played elsewhere (promoted clubs before their new division is
        # harvested). The gamma index follows the match's competition, so no parameter
        # is created for theirs; for the sum(alpha) == 0 constraint they join the group
        # of the competition they played most matches in (centring them on their own
        # would force the strongest teams of the lower division to average zero).
        for g in missing:
            for team in teams_by_girone[g]:
                constraint_girone[team] = played_in[team]
        log.warning(
            "registry gironi with current teams but no matches: no home advantage estimated, "
            "teams constrained with the girone they played in",
            extra={
                "season": registry.season_label if registry is not None else None,
                "gironi": [g.value for g in missing],
                "teams": {t: g.value for t, g in constraint_girone.items()},
            },
        )
    return TeamIndex(
        teams=teams,
        teams_by_girone=teams_by_girone,
        gironi=gironi,
        girone_specific_home=girone_specific_home,
        historical=historical,
        constraint_girone=constraint_girone,
    )


def _reject_invalid_rows(df: pd.DataFrame) -> None:
    """Range-check a frame already coerced by :func:`validate_matches_df`.

    ``validate_matches_df`` fixes dtypes but does not check values; the
    :class:`src.common.schemas.MatchRecord` constructor does. This applies the
    same two rules to the DataFrame path so that sentinel scores (``-1`` for a
    postponed match) or a name-matching collision (``home_team == away_team``)
    raise ``ValueError`` instead of silently biasing the estimates.
    """
    goals = df[["home_goals", "away_goals"]].to_numpy(dtype=np.int64)
    negative = (goals < 0).any(axis=1)
    if negative.any():
        ids = df.loc[negative, "match_id"].astype(str).tolist()
        raise ValueError(
            f"{int(negative.sum())} matches have negative goals (match_id: {ids[:10]}{', ...' if len(ids) > 10 else ''}); "
            "remove postponed/void matches before fitting"
        )
    same_team = (df["home_team"] == df["away_team"]).fillna(False).to_numpy(dtype=bool)
    if same_team.any():
        ids = df.loc[same_team, "match_id"].astype(str).tolist()
        teams = sorted(set(df.loc[same_team, "home_team"].astype(str)))
        raise ValueError(
            f"{int(same_team.sum())} matches have home_team == away_team (match_id: {ids[:10]}{', ...' if len(ids) > 10 else ''}; "
            f"teams: {teams[:10]}); check the team-name matching upstream"
        )


def prepare_likelihood_data(
    matches: pd.DataFrame,
    *,
    reference_date: date | datetime | pd.Timestamp | None = None,
    xi: float = 0.0,
    l2_penalty: float = 0.0,
    girone_specific_home: bool = True,
    team_index: TeamIndex | None = None,
    min_matches_per_team: int = 1,
    l2_penalty_home: float = 0.0,
    home_prior: float = 0.0,
    l2_penalty_rho: float = 0.0,
    registry: TeamRegistry | None = None,
) -> tuple[LikelihoodData, TeamIndex]:
    """Validate ``matches`` and convert them into :class:`LikelihoodData`.

    Returns the data together with the :class:`TeamIndex` used to position the
    parameters (built from the frame unless ``team_index`` is given; ``registry``
    is forwarded to :func:`build_team_index` to tell current teams from
    historical ones). The reference date defaults to the most recent match date.
    ``ValueError`` is raised for an empty frame, negative goals or
    ``home_team == away_team`` rows (see :func:`_reject_invalid_rows`).
    """
    df = validate_matches_df(matches)
    if df.empty:
        raise ValueError("no valid matches to fit")
    _reject_invalid_rows(df)
    index = team_index or build_team_index(
        df, girone_specific_home=girone_specific_home, min_matches_per_team=min_matches_per_team, registry=registry
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
        l2_penalty_home=float(l2_penalty_home),
        home_prior=float(home_prior),
        l2_penalty_rho=float(l2_penalty_rho),
        team_girone_idx=index.team_girone_positions(),
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
    # The mean beta of a girone is the model's only intercept (there is no global
    # rate parameter), so the beta ridge shrinks towards the girone mean, not 0:
    # it regularises team differences without biasing the overall goal level.
    if d.team_girone_idx is not None:
        counts = np.bincount(d.team_girone_idx).astype(float)
        beta_mean = np.bincount(d.team_girone_idx, weights=beta) / np.maximum(counts, 1.0)
        beta_dev = beta - beta_mean[d.team_girone_idx]
    else:
        beta_dev = beta
    gamma_dev = gamma - d.home_prior
    penalty = float(
        d.l2_penalty * (np.dot(alpha, alpha) + np.dot(beta_dev, beta_dev))
        + d.l2_penalty_home * np.dot(gamma_dev, gamma_dev)
        + d.l2_penalty_rho * rho * rho
    )
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
    grad[n_t : 2 * n_t] += 2.0 * d.l2_penalty * beta_dev  # d/d beta of |beta - mean|^2 is 2 (beta - mean)
    grad[2 * n_t : 2 * n_t + d.n_gammas] += 2.0 * d.l2_penalty_home * gamma_dev
    grad[-1] += 2.0 * d.l2_penalty_rho * rho
    return ll, penalty, grad


def negative_log_likelihood(theta: np.ndarray, *data: Any) -> float:
    """Penalised negative weighted log-likelihood.

    ``-LL(theta) + l2 * (|alpha|^2 + |beta - mean_girone(beta)|^2) + l2_home * |gamma - home_prior|^2 + l2_rho * rho^2``
    (the last two terms vanish with the default zero strengths; without
    ``team_girone_idx`` the beta term is uncentred, i.e. ``|beta|^2``).

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

    ``l2_penalty_home`` shrinks every gamma towards ``home_prior`` (which must lie
    inside ``home_bounds``) and ``l2_penalty_rho`` shrinks rho towards 0; both are
    off (0.0) by default so the estimator stays the pure weighted MLE. Use
    :meth:`with_gaussian_prior` to express the three strengths as prior standard
    deviations (``l2 = 1 / (2 sigma^2)``).

    ``min_effective_n_ratio`` guards against collapsed time-decay weights: when
    the sum of the weights is smaller than this fraction of the number of
    matches (one far-future date typo, or a reference date decades after the
    sample) the fit is degenerate. With ``on_weight_collapse='warn'`` (default)
    it is still computed but reported with ``converged=False``,
    ``degenerate_weights=True`` and a diagnosis in ``message``; with ``'raise'``
    :meth:`DixonColesEngine.fit` raises ``ValueError`` before optimising. ``0.0``
    disables the guard. The default 0.1 is far below what a legitimate long
    window produces (five seasons at the default xi still keep about 30% of
    the nominal sample).
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
    l2_penalty_home: float = 0.0
    home_prior: float = 0.0
    l2_penalty_rho: float = 0.0
    min_effective_n_ratio: float = 0.1
    on_weight_collapse: WeightCollapsePolicy = "warn"

    def __post_init__(self) -> None:
        if self.xi < 0:
            raise ValueError(f"xi must be >= 0, got {self.xi}")
        if self.max_goals < 1:
            raise ValueError(f"max_goals must be >= 1, got {self.max_goals}")
        if self.method not in METHODS:
            raise ValueError(f"method must be one of {METHODS}, got {self.method!r}")
        ratio = self.min_effective_n_ratio
        if not (isinstance(ratio, (int, float)) and np.isfinite(ratio) and 0.0 <= ratio <= 1.0):
            raise ValueError(f"min_effective_n_ratio must be a number in [0, 1], got {ratio!r}")
        object.__setattr__(self, "min_effective_n_ratio", float(ratio))
        if self.on_weight_collapse not in WEIGHT_COLLAPSE_POLICIES:
            raise ValueError(f"on_weight_collapse must be one of {WEIGHT_COLLAPSE_POLICIES}, got {self.on_weight_collapse!r}")
        for name in ("rho_bounds", "home_bounds", "param_bounds"):
            lo, hi = getattr(self, name)
            if not float(lo) < float(hi):
                raise ValueError(f"{name} must be (low, high) with low < high, got {(lo, hi)}")
            object.__setattr__(self, name, (float(lo), float(hi)))
        for name in ("l2_penalty", "l2_penalty_home", "l2_penalty_rho"):
            value = getattr(self, name)
            if not (np.isfinite(value) and value >= 0):
                raise ValueError(f"{name} must be a finite number >= 0, got {value}")
        if not np.isfinite(self.home_prior):
            raise ValueError(f"home_prior must be finite, got {self.home_prior}")
        lo, hi = self.home_bounds
        if not lo <= self.home_prior <= hi:
            raise ValueError(f"home_prior {self.home_prior} must lie within home_bounds {(lo, hi)}")
        if self.max_iter < 1:
            raise ValueError(f"max_iter must be >= 1, got {self.max_iter}")
        if not self.tol > 0:
            raise ValueError(f"tol must be > 0, got {self.tol}")
        if self.min_matches_per_team < 0:
            raise ValueError(f"min_matches_per_team must be >= 0, got {self.min_matches_per_team}")

    @classmethod
    def with_gaussian_prior(
        cls,
        *,
        sigma_strength: float = 0.30,
        sigma_home: float = 0.15,
        home_prior: float = 0.25,
        sigma_rho: float = 0.10,
        **overrides: Any,
    ) -> "DixonColesConfig":
        """Config whose ridge strengths encode independent Gaussian priors.

        A Normal(0, sigma^2) prior on a parameter contributes ``theta^2 / (2 sigma^2)``
        to the negative log-posterior, so ``l2 = 1 / (2 sigma^2)``. The defaults
        (alpha/beta sd 0.30, home advantage 0.25 +- 0.15, rho sd 0.10) are typical
        of European football and make a 20-match sample fit regular; on a full
        season the data dominate and the MAP estimate is close to the MLE.
        Other :class:`DixonColesConfig` fields can be passed as keyword overrides.
        """
        for name, value in (("sigma_strength", sigma_strength), ("sigma_home", sigma_home), ("sigma_rho", sigma_rho)):
            if not (np.isfinite(value) and value > 0):
                raise ValueError(f"{name} must be a finite number > 0, got {value}")
        to_l2 = lambda sigma: 1.0 / (2.0 * float(sigma) ** 2)  # noqa: E731
        return cls(
            l2_penalty=to_l2(sigma_strength),
            l2_penalty_home=to_l2(sigma_home),
            home_prior=float(home_prior),
            l2_penalty_rho=to_l2(sigma_rho),
            **overrides,
        )

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
    the sum of the time-decay weights and :attr:`effective_n_ratio` its share of
    ``n_matches``. ``method`` is the optimiser that actually produced the
    parameters: it differs from ``DixonColesConfig.method`` only when an
    L-BFGS-B solution violated ``param_bounds`` and the engine re-solved with
    SLSQP, in which case ``message`` documents the fallback.

    ``converged`` is True only when the optimiser reported success *and* the
    time-decay weights did not collapse; ``degenerate_weights`` is True when
    they did (``effective_n`` below ``DixonColesConfig.min_effective_n_ratio``
    times ``n_matches``), in which case ``message`` starts with the diagnosis.
    ``fit_warnings`` collects every data-quality note raised during the fit
    (collapsed weights, a reference date outside the sample); it is empty for a
    clean fit.

    ``historical_teams`` are the fitted teams outside the perimeter of the
    registry passed to :meth:`DixonColesEngine.fit` (empty without a registry);
    :attr:`current_teams` are the others and :meth:`is_current` tells them apart.
    ``teams_by_girone`` holds every fitted team (current and historical) in its
    reported competition (the registry one for current teams), which is also its
    constraint group except for the teams of a registry competition without
    matches (see :class:`TeamIndex`); ``gamma`` has one entry per competition
    that had matches, so such a competition is absent from it.
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
    degenerate_weights: bool = False
    fit_warnings: tuple[str, ...] = ()
    historical_teams: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        self.alpha = {str(k): float(v) for k, v in self.alpha.items()}
        self.beta = {str(k): float(v) for k, v in self.beta.items()}
        self.gamma = {Girone.parse(k): float(v) for k, v in self.gamma.items()}
        self.teams_by_girone = {Girone.parse(k): tuple(str(t) for t in v) for k, v in self.teams_by_girone.items()}
        self.reference_date = _coerce_date(self.reference_date)
        self.rho = float(self.rho)
        self.degenerate_weights = bool(self.degenerate_weights)
        self.fit_warnings = tuple(str(w) for w in self.fit_warnings)
        self.historical_teams = tuple(str(t) for t in self.historical_teams)
        unknown = sorted(set(self.historical_teams) - set(self.teams))
        if unknown:
            raise ValueError(f"historical_teams must be fitted teams; unknown: {unknown}")

    # ----------------------------------------------------------------- views
    @property
    def effective_n_ratio(self) -> float:
        """``effective_n / n_matches``: 1.0 for a stationary fit, near 0 when the weights collapsed."""
        return self.effective_n / self.n_matches if self.n_matches > 0 else 0.0

    @property
    def teams(self) -> tuple[str, ...]:
        """All fitted teams (current and historical) in competition order (``config/competitions.toml``), alphabetical within one."""
        return tuple(t for g in Girone for t in self.teams_by_girone.get(g, ()))

    @property
    def current_teams(self) -> tuple[str, ...]:
        """Fitted teams of the perimeter, in :attr:`teams` order (every team when no registry was used)."""
        gone = set(self.historical_teams)
        return tuple(t for t in self.teams if t not in gone)

    def is_current(self, team: str) -> bool:
        """True when ``team`` was fitted and belongs to the perimeter; False for historical *and* unknown teams."""
        return team in self.alpha and team not in set(self.historical_teams)

    def current_teams_by_girone(self) -> dict[Girone, tuple[str, ...]]:
        """``teams_by_girone`` restricted to current teams (gironi left without teams are dropped)."""
        gone = set(self.historical_teams)
        out: dict[Girone, tuple[str, ...]] = {}
        for g, members in self.teams_by_girone.items():
            kept = tuple(t for t in members if t not in gone)
            if kept:
                out[g] = kept
        return out

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
            "effective_n_ratio": self.effective_n_ratio,
            "degenerate_weights": self.degenerate_weights,
            "fit_warnings": list(self.fit_warnings),
            "historical_teams": list(self.historical_teams),
            "current_teams": list(self.current_teams),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "FittedParams":
        """Inverse of :meth:`to_dict`; fields with defaults (``degenerate_weights``,
        ``fit_warnings``, ``historical_teams``) may be absent, so payloads written
        before they existed load with every team current (``current_teams`` is
        derived, never read back)."""
        known = {f.name for f in fields(cls)}
        required = {f.name for f in fields(cls) if f.default is MISSING and f.default_factory is MISSING}
        missing = required - set(payload)
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

    The dependent team of each girone is its last member (alphabetically). Its
    alpha is *not* box-bounded by the free-variable bounds (``-sum(others)`` can
    exceed ``param_bounds`` when, e.g., the team scored no goals in the window), so
    :meth:`DixonColesEngine.fit` checks the mapped-back alphas and falls back to
    SLSQP when any of them is out of range.
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
        registry: TeamRegistry | None = None,
    ) -> FittedParams:
        """Estimate ``alpha, beta, gamma, rho`` by weighted maximum likelihood.

        ``matches`` is validated with :func:`validate_matches_df`. The reference
        date for the time-decay defaults to the most recent match date.
        ``warm_start`` initialises the optimiser from a previous fit when its team
        set matches the current one (it is ignored, with a warning, otherwise).
        ``registry`` defines the perimeter (see the module docstring): registry
        teams are current and sit in their registry girone, the other teams of the
        frame are fitted as historical opponents and listed in
        ``FittedParams.historical_teams``. Without it every team is current.

        The time-decay weights are checked with :func:`time_weight_diagnostics`:
        when they collapse (``effective_n < min_effective_n_ratio * n_matches``)
        the result carries ``degenerate_weights=True`` and ``converged=False``
        with the diagnosis in ``message`` and ``fit_warnings``, or ``ValueError``
        is raised when ``DixonColesConfig.on_weight_collapse == 'raise'``. A
        reference date before the first match (decay switched off) or far after
        the last one is logged and recorded in ``fit_warnings``.
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
            l2_penalty_home=cfg.l2_penalty_home,
            home_prior=cfg.home_prior,
            l2_penalty_rho=cfg.l2_penalty_rho,
            registry=registry,
        )
        effective_n = float(data.w.sum())
        n_matches = len(data.x)
        weights = data.w if len(data.w) == len(validated) else time_weights(validated["date"], ref_date, cfg.xi)
        diag = time_weight_diagnostics(validated["date"], weights, ref_date, cfg.xi, match_ids=validated["match_id"])
        collapsed = diag.is_collapsed(cfg.min_effective_n_ratio)
        fit_warnings: list[str] = []
        collapse_note = ""
        if collapsed:
            collapse_note = diag.describe(cfg.min_effective_n_ratio)
            if cfg.on_weight_collapse == "raise":
                raise ValueError(collapse_note)
            log.warning(
                "time-decay weights collapsed: the fit is degenerate",
                extra={**diag.to_dict(), "min_effective_n_ratio": cfg.min_effective_n_ratio, "xi": cfg.xi},
            )
            fit_warnings.append(collapse_note)
        elif cfg.xi > 0:
            note = _reference_date_note(diag, cfg.xi)
            if note:
                log.warning("reference_date lies outside the match sample", extra={**diag.to_dict(), "xi": cfg.xi})
                fit_warnings.append(note)
        log.info(
            "fitting Dixon-Coles",
            extra={
                "n_matches": n_matches,
                "n_teams": index.n_teams,
                "n_current_teams": len(index.current_teams),
                "n_historical_teams": len(index.historical),
                "perimeter": registry.season_label if registry is not None else None,
                "gironi": [g.value for g in index.gironi],
                "constraint_overrides": {t: g.value for t, g in index.constraint_girone.items()},
                "method": cfg.method,
                "xi": cfg.xi,
                "l2_penalty": cfg.l2_penalty,
                "l2_penalty_home": cfg.l2_penalty_home,
                "l2_penalty_rho": cfg.l2_penalty_rho,
                "reference_date": ref_date.isoformat(),
                "effective_n": round(effective_n, 3),
                "effective_n_ratio": round(diag.effective_n_ratio, 4),
                "degenerate_weights": collapsed,
            },
        )

        theta0 = self._initial_theta(data, index, warm_start)
        scale = 1.0 / max(effective_n, 1e-12)  # optimise the mean weighted NLL per unit weight
        method_used: str = cfg.method
        fallback_note = ""
        if cfg.method == "SLSQP":
            result, theta = self._fit_slsqp(theta0, data, index, scale)
        else:
            result, theta = self._fit_lbfgsb(theta0, data, index, scale)
            offenders = self._alpha_bound_violations(theta, index)
            if offenders:
                # The reparametrised problem cannot bound the dependent alpha of a
                # girone; the bounded + equality-constrained SLSQP problem can.
                lbfgsb_message = str(result.message)
                lbfgsb_nit = int(getattr(result, "nit", 0) or 0)
                log.warning(
                    "L-BFGS-B left dependent alpha(s) outside param_bounds; re-solving with SLSQP",
                    extra={
                        "param_bounds": list(cfg.param_bounds),
                        "offending_alpha": {t: round(v, 4) for t, v in offenders.items()},
                        "lbfgsb_message": lbfgsb_message,
                        "lbfgsb_n_iter": lbfgsb_nit,
                    },
                )
                result, theta = self._fit_slsqp(theta0, data, index, scale)
                method_used = "SLSQP"
                fallback_note = (
                    f"L-BFGS-B solution violated param_bounds {cfg.param_bounds} for the dependent alpha of "
                    f"{sorted(offenders)} ({lbfgsb_message}; {lbfgsb_nit} iterations); re-solved with SLSQP: "
                )

        theta = self._recentre_alpha(theta, index)
        ll, penalty, grad = _evaluate(theta, data, with_grad=True)
        assert grad is not None
        alpha, beta, gamma, rho = unpack_theta(theta, data.n_teams, data.n_gammas)
        gamma_map: dict[Girone, float]
        if cfg.girone_specific_home:
            gamma_map = {g: float(gamma[i]) for i, g in enumerate(index.gironi)}
        else:
            gamma_map = {g: float(gamma[0]) for g in Girone}
        optimizer_message = fallback_note + str(result.message)
        # A collapsed weight vector makes the optimiser's own success flag meaningless.
        converged = bool(result.success) and not collapsed
        message = f"{collapse_note} [optimiser: {optimizer_message}]" if collapsed else optimizer_message
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
            n_matches=n_matches,
            effective_n=effective_n,
            converged=converged,
            message=message,
            n_iter=n_iter,
            fit_seconds=float(seconds),
            method=method_used,
            max_goals=cfg.max_goals,
            degenerate_weights=collapsed,
            fit_warnings=tuple(fit_warnings),
            historical_teams=index.historical,
        )
        self._params = params
        self._index = index
        log_fn = log.info if converged else log.warning
        log_fn(
            "Dixon-Coles fit finished",
            extra={
                "method": method_used,
                "converged": converged,
                "degenerate_weights": collapsed,
                "effective_n_ratio": round(diag.effective_n_ratio, 4),
                "optimizer_message": optimizer_message,
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

    def _alpha_bound_violations(self, theta: np.ndarray, index: TeamIndex, *, tol: float = 1e-8) -> dict[str, float]:
        """Teams whose alpha lies outside ``param_bounds`` (beyond ``tol``), with the offending value.

        Only the dependent alpha of each girone can violate the bounds under the
        L-BFGS-B reparametrisation; every alpha is checked for robustness.
        """
        lo, hi = self.config.param_bounds
        alpha = np.asarray(theta[: index.n_teams], dtype=float)
        bad = np.flatnonzero((alpha < lo - tol) | (alpha > hi + tol))
        return {index.teams[int(i)]: float(alpha[i]) for i in bad}

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
        """L-BFGS-B on the reduced (sum-alpha-free) problem; returns the mapped-back full ``theta``.

        The dependent alpha of each girone is unbounded here; :meth:`fit` checks
        it against ``param_bounds`` and falls back to SLSQP when needed.
        """
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
    def ranking(self, girone: Girone | str | None = None, *, include_historical: bool = False) -> pd.DataFrame:
        """Strength table sorted by ``alpha`` (desc) then ``beta`` (desc).

        Columns: ``team, girone, alpha, beta, gamma, attack_index, defence_index,
        xg_for_vs_avg, xg_against_vs_avg`` where ``attack_index = exp(alpha)``,
        ``defence_index = exp(beta)``, ``xg_for_vs_avg = exp(alpha) * mean_girone(exp(-beta))``
        and ``xg_against_vs_avg = exp(-beta) * mean_girone(exp(alpha))``.

        Only the current teams of the perimeter are listed (``girone`` is their
        registry girone when the fit used a registry); ``include_historical=True``
        adds the historical opponents too. The per-girone means of the ``vs_avg``
        columns are taken over the listed teams.
        """
        params = self.params_
        frame = params.to_frame()
        if not include_historical and params.historical_teams:
            frame = frame.loc[~frame["team"].isin(params.historical_teams)]
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
