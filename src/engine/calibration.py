"""Learning from the differential: validated recalibration of the Dixon-Coles probabilities.

The walk-forward backtest (:func:`src.engine.backtest.walk_forward`) gives, for
every past match, the probability the model published **before** kick-off and
what happened. When the two disagree systematically -- the draws the model
calls 25% happen 29% of the time, its GG are a little overconfident -- a small
correction learnt on those out-of-sample pairs improves the next
probabilities. This module learns that correction and, above all, decides
honestly whether to use it.

Correction
----------
* two-way markets (GG/NG, Over/Under 1.5-2.5-3.5): Platt scaling of the first
  selection, ``logit p' = a + b * logit p`` (identity ``a = 0``, ``b = 1``);
* 1X2: a temperature and two biases on the log-probabilities,
  ``p'_k ∝ exp(s * ln p_k + c_k)`` with ``c_1 = 0`` (identity ``s = 1``,
  ``c = 0``), renormalised;
* both with an L2 penalty (``l2``, on the summed log loss) that shrinks the
  parameters towards the identity: with few matches the correction stays
  close to "do nothing". The parameters are pooled over every family (Serie
  B, Serie C, Serie D); a family with at least ``min_n`` matches also gets its
  own parameters, shrunk towards the pooled ones;
* after the correction the markets are kept coherent: ``P(Over 3.5) <=
  P(Over 2.5) <= P(Over 1.5)`` and ``P(GG) <= P(Over 1.5)``.

Validation
----------
:func:`fit_recalibration` splits the backtest **in time**: the correction is
fitted on the earlier matchdays and scored on the later ones (``test_fraction``
of the matches). A market is corrected only when the out-of-sample log loss
improves by at least ``min_gain`` (0.5% by default) on at least
``min_eval`` matches; otherwise its raw probabilities are kept
(:attr:`MarketCalibration.enabled` is ``False``, :meth:`RecalibrationModel.apply`
leaves it untouched). The decision is recorded with its numbers
(:meth:`RecalibrationModel.describe`: *Calibrazione applicata: sì/no,
miglioramento log loss X%*), the raw probabilities are never overwritten (the
correction is a pure function applied on top, switchable and reversible) and
the model serialises to JSON (:meth:`RecalibrationModel.to_dict`) for caching.
:func:`rolling_recalibration` replays the correction matchday by matchday
(each matchday corrected only with what was learnt before it), so a track
record "with calibration" is as out-of-sample as the raw one.

:func:`tune_hyperparameters` goes one level up: it reruns the walk-forward
for a small grid of time-decay rates ``xi`` and prior strengths and ranks
them by out-of-sample log loss. It is optional and slow-ish (one full
backtest per grid point, a few seconds each for the current season of the 13
competitions): run it rarely and cache :meth:`TuningResult.to_dict`.
"""
from __future__ import annotations

import math
import time
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from datetime import date, datetime, timezone
from functools import lru_cache
from typing import Any, Final, Literal

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import expit, logit

from src.common.logging_config import get_logger
from src.common.schemas import SELECTIONS, Market, MatchPrediction
from src.common.teams import TeamRegistry
from src.engine.backtest import (
    DEFAULT_BLOCK,
    DEFAULT_MARKETS,
    DEFAULT_MIN_TRAIN_MATCHES,
    MARKET_SPECS,
    PRODUCTION_XI,
    BlockName,
    MarketSpec,
    ProgressCallback,
    market_of,
    market_spec,
    production_config,
    walk_forward,
)
from src.engine.model import DixonColesConfig

__all__ = [
    "DEFAULT_L2",
    "DEFAULT_MIN_EVAL",
    "DEFAULT_MIN_GAIN",
    "DEFAULT_MIN_HISTORY",
    "DEFAULT_MIN_N",
    "DEFAULT_TEST_FRACTION",
    "DEFAULT_TUNING_GRID",
    "DEFAULT_TUNING_MARKETS",
    "RAKE_MAX_ITER",
    "RAKE_TOL",
    "BinaryParams",
    "MarketCalibration",
    "MultinomialParams",
    "RecalibrationModel",
    "RollingRecalibration",
    "TuningResult",
    "evaluate_recalibration",
    "fit_recalibration",
    "rake_score_matrix",
    "rolling_recalibration",
    "time_split",
    "tune_hyperparameters",
]

log = get_logger(__name__)

#: L2 strength (on the **summed** log loss) pulling the parameters towards the identity
#: (a Gaussian prior of standard deviation ``1 / sqrt(2 * l2)``, about 0.32, on each parameter).
DEFAULT_L2: Final[float] = 5.0
#: A family gets its own parameters only with at least this many matches (else the pooled ones).
DEFAULT_MIN_N: Final[int] = 200
#: Minimum relative improvement of the out-of-sample log loss for a market to be corrected (0.5%).
DEFAULT_MIN_GAIN: Final[float] = 0.005
#: Share of the matches (the latest ones) kept out of the fit to judge the correction.
DEFAULT_TEST_FRACTION: Final[float] = 0.3
#: Below this many evaluation matches a market is never corrected (the verdict would be noise).
DEFAULT_MIN_EVAL: Final[int] = 100
#: Probabilities are clipped to ``[_EPS, 1 - _EPS]`` before logit / log.
_EPS: Final[float] = 1e-6
_SCHEMA_VERSION: Final[int] = 1
_FAMILY_COLUMN: Final[str] = "family"

#: Default grid of :func:`tune_hyperparameters`: time decay per day and prior (``'auto'`` = production
#: rule, ``None`` = no prior, a number = :meth:`DixonColesConfig.with_gaussian_prior` ``sigma_strength``).
DEFAULT_TUNING_GRID: Final[dict[str, tuple[Any, ...]]] = {"xi": (0.0010, 0.0018, 0.0030), "prior": ("auto", 0.20, 0.40)}
#: Markets whose mean out-of-sample log loss is the tuning objective.
DEFAULT_TUNING_MARKETS: Final[tuple[Market, ...]] = (Market.MATCH_ODDS, Market.BTTS, Market.OVER_UNDER_25)
#: Cycles and tolerance (largest gap between a market and its cells) of :func:`rake_score_matrix`;
#: the usual corrections converge in about 30 cycles.
RAKE_MAX_ITER: Final[int] = 500
RAKE_TOL: Final[float] = 1e-10
_OU_LINES: Final[dict[Market, float]] = {Market.OVER_UNDER_15: 1.5, Market.OVER_UNDER_25: 2.5, Market.OVER_UNDER_35: 3.5}


# =============================================================================
# Parameters
# =============================================================================
@dataclass(frozen=True)
class BinaryParams:
    """Platt scaling of a two-way market: ``logit p' = intercept + slope * logit p`` (identity: 0, 1)."""

    intercept: float = 0.0
    slope: float = 1.0

    @property
    def is_identity(self) -> bool:
        return self.intercept == 0.0 and self.slope == 1.0

    def vector(self) -> np.ndarray:
        return np.array([self.intercept, self.slope], dtype=float)

    def adjust(self, p_first: np.ndarray) -> np.ndarray:
        """Corrected probability of the market's first selection (GG, Over)."""
        p = np.clip(np.asarray(p_first, dtype=float), _EPS, 1.0 - _EPS)
        return expit(self.intercept + self.slope * logit(p))

    def to_dict(self) -> dict[str, float]:
        return {"intercept": float(self.intercept), "slope": float(self.slope)}

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "BinaryParams":
        return cls(intercept=float(payload.get("intercept", 0.0)), slope=float(payload.get("slope", 1.0)))


@dataclass(frozen=True)
class MultinomialParams:
    """Temperature and biases of 1X2: ``p'_k ∝ exp(inv_temperature * ln p_k + bias_k)``, ``bias_1 = 0`` (identity: 1, 0, 0)."""

    inv_temperature: float = 1.0
    bias_x: float = 0.0
    bias_2: float = 0.0

    @property
    def is_identity(self) -> bool:
        return self.inv_temperature == 1.0 and self.bias_x == 0.0 and self.bias_2 == 0.0

    def vector(self) -> np.ndarray:
        return np.array([self.inv_temperature, self.bias_x, self.bias_2], dtype=float)

    def adjust(self, probs: np.ndarray) -> np.ndarray:
        """Corrected ``(n, 3)`` probabilities of 1, X, 2 (rows sum to 1)."""
        logp = np.log(np.clip(np.asarray(probs, dtype=float), _EPS, 1.0))
        z = self.inv_temperature * logp + np.array([0.0, self.bias_x, self.bias_2])
        z -= z.max(axis=1, keepdims=True)
        e = np.exp(z)
        return e / e.sum(axis=1, keepdims=True)

    def to_dict(self) -> dict[str, float]:
        return {"inv_temperature": float(self.inv_temperature), "bias_x": float(self.bias_x), "bias_2": float(self.bias_2)}

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "MultinomialParams":
        return cls(
            inv_temperature=float(payload.get("inv_temperature", 1.0)),
            bias_x=float(payload.get("bias_x", 0.0)),
            bias_2=float(payload.get("bias_2", 0.0)),
        )


Params = BinaryParams | MultinomialParams


def _identity(spec: MarketSpec) -> Params:
    return BinaryParams() if spec.is_binary else MultinomialParams()


def _params_from_vector(spec: MarketSpec, theta: np.ndarray) -> Params:
    if spec.is_binary:
        return BinaryParams(intercept=float(theta[0]), slope=float(theta[1]))
    return MultinomialParams(inv_temperature=float(theta[0]), bias_x=float(theta[1]), bias_2=float(theta[2]))


def _params_from_dict(spec: MarketSpec, payload: Mapping[str, Any]) -> Params:
    return BinaryParams.from_dict(payload) if spec.is_binary else MultinomialParams.from_dict(payload)


# =============================================================================
# Fitting
# =============================================================================
def _design(frame: pd.DataFrame, spec: MarketSpec, prefix: str) -> tuple[np.ndarray, np.ndarray]:
    """``(features, outcome index)`` of the valid rows: logit of the first selection (binary) or log-probabilities (1X2)."""
    probs = spec.probabilities(frame, prefix)
    y = spec.outcome_index(frame)
    keep = (y >= 0) & np.isfinite(probs).all(axis=1)
    probs, y = probs[keep], y[keep]
    if spec.is_binary:
        return logit(np.clip(probs[:, 0], _EPS, 1.0 - _EPS))[:, None], y
    return np.log(np.clip(probs, _EPS, 1.0)), y


def _objective(theta: np.ndarray, x: np.ndarray, y: np.ndarray, binary: bool, l2: float, centre: np.ndarray) -> tuple[float, np.ndarray]:
    """Summed log loss plus ``l2 * |theta - centre|^2``, with its gradient (both divided by ``n`` for scaling)."""
    n = max(len(y), 1)
    if binary:
        z = theta[0] + theta[1] * x[:, 0]
        target = (y == 0).astype(float)
        # log(1 + e^z) - target * z, stable
        loss = float(np.sum(np.logaddexp(0.0, z) - target * z))
        r = expit(z) - target
        grad = np.array([r.sum(), (r * x[:, 0]).sum()])
    else:
        z = theta[0] * x + np.array([0.0, theta[1], theta[2]])
        z_max = z.max(axis=1, keepdims=True)
        log_norm = (z_max[:, 0] + np.log(np.exp(z - z_max).sum(axis=1)))
        loss = float(np.sum(log_norm - z[np.arange(len(y)), y]))
        p = np.exp(z - log_norm[:, None])
        onehot = np.zeros_like(p)
        onehot[np.arange(len(y)), y] = 1.0
        r = p - onehot
        grad = np.array([(r * x).sum(), r[:, 1].sum(), r[:, 2].sum()])
    dev = theta - centre
    loss += l2 * float(dev @ dev)
    grad = grad + 2.0 * l2 * dev
    return loss / n, grad / n


def _fit_params(x: np.ndarray, y: np.ndarray, spec: MarketSpec, l2: float, centre: Params) -> Params:
    """Penalised maximum likelihood of the correction (L-BFGS-B from the centre); the centre when there is no data."""
    if len(y) == 0:
        return centre
    start = centre.vector()
    bounds = [(-5.0, 5.0), (0.05, 5.0)] if spec.is_binary else [(0.05, 5.0), (-5.0, 5.0), (-5.0, 5.0)]
    result = minimize(
        _objective,
        start,
        args=(x, y, spec.is_binary, float(l2), start),
        jac=True,
        method="L-BFGS-B",
        bounds=bounds,
        options={"maxiter": 500, "ftol": 1e-12, "gtol": 1e-9},
    )
    if not result.success:
        log.warning("recalibration fit did not converge", extra={"market": spec.key, "n": int(len(y)), "optimizer_message": str(result.message)})
    return _params_from_vector(spec, np.asarray(result.x, dtype=float))


def _fit_market(
    frame: pd.DataFrame, spec: MarketSpec, *, prefix: str, l2: float, min_n: int, family_column: str
) -> tuple[Params, dict[str, Params], int]:
    """Pooled parameters (shrunk to the identity) and per-family ones (shrunk to the pooled) for families with ``>= min_n`` rows."""
    x, y = _design(frame, spec, prefix)
    pooled = _fit_params(x, y, spec, l2, _identity(spec))
    by_family: dict[str, Params] = {}
    if family_column in frame.columns and len(frame):
        for family, group in frame.groupby(family_column, sort=True, observed=True):
            xf, yf = _design(group, spec, prefix)
            if len(yf) >= int(min_n) and frame[family_column].nunique() > 1:
                by_family[str(family)] = _fit_params(xf, yf, spec, l2, pooled)
    return pooled, by_family, int(len(y))


def _adjust_market(spec: MarketSpec, probs: np.ndarray, families: np.ndarray | None, pooled: Params, by_family: Mapping[str, Params]) -> np.ndarray:
    """Corrected ``(n, k)`` probabilities of one market (family parameters where available)."""
    out = np.empty_like(probs, dtype=float)
    groups: dict[str, np.ndarray] = {}
    if families is not None and by_family:
        for family in by_family:
            mask = families == family
            if mask.any():
                groups[family] = mask
    rest = np.ones(len(probs), dtype=bool)
    for mask in groups.values():
        rest &= ~mask
    for key, mask in [*groups.items(), ("", rest)]:
        if not mask.any():
            continue
        params = by_family[key] if key else pooled
        if spec.is_binary:
            first = params.adjust(probs[mask, 0])  # type: ignore[union-attr]
            out[mask, 0] = first
            out[mask, 1] = 1.0 - first
        else:
            out[mask] = params.adjust(probs[mask])  # type: ignore[union-attr]
    return out


def _log_loss(probs: np.ndarray, y: np.ndarray) -> float:
    if len(y) == 0:
        return float("nan")
    picked = probs[np.arange(len(y)), y]
    return float(np.mean(-np.log(np.clip(picked, 1e-12, 1.0))))


def _brier(probs: np.ndarray, y: np.ndarray, binary: bool) -> float:
    if len(y) == 0:
        return float("nan")
    onehot = np.zeros_like(probs)
    onehot[np.arange(len(y)), y] = 1.0
    if binary:
        return float(np.mean((probs[:, 0] - onehot[:, 0]) ** 2))
    return float(np.mean(((probs - onehot) ** 2).sum(axis=1)))


# =============================================================================
# Model
# =============================================================================
@dataclass(frozen=True)
class MarketCalibration:
    """The correction of one market and the out-of-sample verdict behind it.

    ``pooled``/``by_family`` are the parameters used by
    :meth:`RecalibrationModel.apply` (refitted on every match when the model
    was fitted with ``refit_full``); ``n_fit`` the matches they were fitted
    on. ``log_loss_before``/``log_loss_after`` are measured on the ``n_eval``
    **later** matches with parameters fitted on the earlier ones only;
    ``enabled`` is ``True`` when the relative improvement :attr:`gain` reached
    the model's ``min_gain`` on enough matches.
    """

    market: Market
    pooled: Params
    by_family: dict[str, Params] = field(default_factory=dict)
    enabled: bool = False
    n_fit: int = 0
    n_eval: int = 0
    log_loss_before: float = float("nan")
    log_loss_after: float = float("nan")
    brier_before: float = float("nan")
    brier_after: float = float("nan")
    reason: str = ""

    @property
    def gain(self) -> float:
        """Relative out-of-sample improvement of the log loss (``0.012`` = 1.2% better; negative = worse)."""
        if not (np.isfinite(self.log_loss_before) and np.isfinite(self.log_loss_after)) or self.log_loss_before <= 0:
            return float("nan")
        return (self.log_loss_before - self.log_loss_after) / self.log_loss_before

    def to_dict(self) -> dict[str, Any]:
        return {
            "market": self.market.value,
            "pooled": self.pooled.to_dict(),
            "by_family": {k: v.to_dict() for k, v in self.by_family.items()},
            "enabled": bool(self.enabled),
            "n_fit": int(self.n_fit),
            "n_eval": int(self.n_eval),
            "log_loss_before": _json_float(self.log_loss_before),
            "log_loss_after": _json_float(self.log_loss_after),
            "brier_before": _json_float(self.brier_before),
            "brier_after": _json_float(self.brier_after),
            "reason": self.reason,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "MarketCalibration":
        spec = market_spec(str(payload["market"]))
        return cls(
            market=spec.market,
            pooled=_params_from_dict(spec, payload.get("pooled", {})),
            by_family={str(k): _params_from_dict(spec, v) for k, v in (payload.get("by_family") or {}).items()},
            enabled=bool(payload.get("enabled", False)),
            n_fit=int(payload.get("n_fit", 0)),
            n_eval=int(payload.get("n_eval", 0)),
            log_loss_before=_float_or_nan(payload.get("log_loss_before")),
            log_loss_after=_float_or_nan(payload.get("log_loss_after")),
            brier_before=_float_or_nan(payload.get("brier_before")),
            brier_after=_float_or_nan(payload.get("brier_after")),
            reason=str(payload.get("reason", "")),
        )


def _json_float(value: float) -> float | None:
    return float(value) if value is not None and np.isfinite(value) else None


def _float_or_nan(value: Any) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return float("nan")
    return out


def _pct(value: float) -> str:
    return f"{value * 100:+.1f}%".replace(".", ",") if np.isfinite(value) else "n.d."


@dataclass(frozen=True)
class RecalibrationModel:
    """Per-market corrections with their out-of-sample verdicts (see the module docstring).

    ``markets`` maps every market considered to its :class:`MarketCalibration`;
    :attr:`enabled` is ``True`` when at least one market is corrected.
    :meth:`apply` / :meth:`apply_frame` / :meth:`apply_prediction` return
    corrected **copies** (the raw probabilities are never modified) and touch
    only the enabled markets plus the coherence of the Over/Under and GG
    probabilities (:meth:`apply_prediction` also rakes the score matrix onto
    the corrected markets, so its cells keep adding up to them). ``split_date`` is the first day of the evaluation part of the
    time split, ``min_gain`` the threshold used, ``fitted_at`` an ISO instant.
    """

    markets: dict[Market, MarketCalibration] = field(default_factory=dict)
    min_gain: float = DEFAULT_MIN_GAIN
    split_date: date | None = None
    n_matches: int = 0
    fitted_at: str = ""

    # ------------------------------------------------------------------ views
    @property
    def enabled(self) -> bool:
        """``True`` when the correction is applied to at least one market."""
        return any(m.enabled for m in self.markets.values())

    @property
    def enabled_markets(self) -> tuple[Market, ...]:
        return tuple(m for m, c in self.markets.items() if c.enabled)

    @classmethod
    def identity(cls) -> "RecalibrationModel":
        """A model that changes nothing (no market enabled)."""
        return cls(markets={}, min_gain=DEFAULT_MIN_GAIN)

    def disabled(self) -> "RecalibrationModel":
        """The same model with every market switched off (the verdicts are kept for display)."""
        return replace(self, markets={m: replace(c, enabled=False) for m, c in self.markets.items()})

    def summary_frame(self) -> pd.DataFrame:
        """One row per market: ``market, enabled, n_fit, n_eval, log_loss_before, log_loss_after, gain, families, parameters, reason``."""
        rows = [
            {
                "market": m.value,
                "enabled": c.enabled,
                "n_fit": c.n_fit,
                "n_eval": c.n_eval,
                "log_loss_before": c.log_loss_before,
                "log_loss_after": c.log_loss_after,
                "gain": c.gain,
                "brier_before": c.brier_before,
                "brier_after": c.brier_after,
                "families": ", ".join(sorted(c.by_family)),
                "parameters": ", ".join(f"{k}={v:.3f}" for k, v in c.pooled.to_dict().items()),
                "reason": c.reason,
            }
            for m, c in self.markets.items()
        ]
        return pd.DataFrame(
            rows,
            columns=["market", "enabled", "n_fit", "n_eval", "log_loss_before", "log_loss_after", "gain", "brier_before", "brier_after", "families", "parameters", "reason"],
        )

    def describe(self) -> str:
        """Italian summary for the dashboard: ``'Calibrazione applicata: sì (...)'`` or ``'Calibrazione applicata: no (...)'``."""
        if not self.markets:
            return "Calibrazione applicata: no (nessuna verifica fuori campione disponibile: servono partite già valutate dal backtest)."
        threshold = f"{self.min_gain * 100:.1f}%".replace(".", ",")
        on = [(m, c) for m, c in self.markets.items() if c.enabled]
        off = [(m, c) for m, c in self.markets.items() if not c.enabled]
        split = f" dal {self.split_date:%d/%m/%Y}" if self.split_date is not None else ""
        if on:
            parts = [
                f"{m.value}: log loss {_pct(-c.gain)} su {c.n_eval} partite di verifica{split}"
                + (f", parametri propri per {', '.join(sorted(c.by_family))}" if c.by_family else "")
                for m, c in on
            ]
            text = "Calibrazione applicata: sì (" + "; ".join(parts) + ")."
            if off:
                text += " Probabilità originali per " + ", ".join(f"{m.value} ({c.reason or 'miglioramento ' + _pct(c.gain)})" for m, c in off) + "."
            return text
        details = ", ".join(f"{m.value} {_pct(c.gain)}" for m, c in off)
        return (
            f"Calibrazione applicata: no (nessun mercato migliora la log loss fuori campione di almeno {threshold}: {details}); "
            "si usano le probabilità originali del modello."
        )

    # ------------------------------------------------------------------ apply
    def _active(self, market: Market) -> MarketCalibration | None:
        calibration = self.markets.get(market)
        return calibration if calibration is not None and calibration.enabled else None

    def apply(self, probabilities_by_market: Mapping[Any, Mapping[str, float]], *, family: str | None = None) -> dict[Any, dict[str, float]]:
        """Corrected copy of one match's probabilities (``{market: {selection: p}}``, keys as given).

        Markets may be keyed by :class:`Market` or by their value (``'1X2'``);
        the output keeps the input keys and selection order. Only the enabled
        markets change (with the parameters of ``family`` when it has its own),
        then the Over/Under and GG probabilities are made coherent; every market
        sums to 1. Markets the model does not know are copied unchanged.
        """
        resolved: dict[Market, Any] = {}
        for key in probabilities_by_market:
            try:
                resolved[market_of(key)] = key
            except ValueError:
                continue
        out: dict[Any, dict[str, float]] = {key: {s: float(p) for s, p in sels.items()} for key, sels in probabilities_by_market.items()}
        changed = False
        families = np.array([family]) if family is not None else None
        for market, key in resolved.items():
            calibration = self._active(market)
            if calibration is None:
                continue
            spec = MARKET_SPECS[market]
            if any(s not in out[key] for s in spec.selections):
                continue
            probs = np.array([[out[key][s] for s in spec.selections]], dtype=float)
            adjusted = _adjust_market(spec, probs, families, calibration.pooled, calibration.by_family)[0]
            for s, p in zip(spec.selections, adjusted):
                out[key][s] = float(p)
            changed = True
        if changed:
            _coherent_mapping(out, resolved)
        return out

    def apply_frame(self, frame: pd.DataFrame, *, prefix: str = "p_", family_column: str = _FAMILY_COLUMN) -> pd.DataFrame:
        """Corrected copy of a frame of probabilities (backtest frame or any frame with ``<prefix><suffix>`` columns).

        Families come from ``family_column`` when present (else the pooled
        parameters are used for every row). The input frame is not modified.
        """
        out = frame.copy()
        if out.empty or not self.enabled:
            return out
        families = out[family_column].astype(str).to_numpy() if family_column in out.columns else None
        for market in self.enabled_markets:
            spec = MARKET_SPECS[market]
            columns = list(spec.columns(prefix))
            if not all(c in out.columns for c in columns):
                continue
            probs = out[columns].to_numpy(dtype=float)
            valid = np.isfinite(probs).all(axis=1)
            if not valid.any():
                continue
            calibration = self.markets[market]
            adjusted = probs.copy()
            adjusted[valid] = _adjust_market(spec, probs[valid], families[valid] if families is not None else None, calibration.pooled, calibration.by_family)
            out[columns] = adjusted
        _coherent_frame(out, prefix)
        return out

    def apply_prediction(self, prediction: MatchPrediction, *, family: str | None = None) -> MatchPrediction:
        """A :class:`MatchPrediction` with corrected probabilities, fair odds and a score matrix that agrees with them.

        The score matrix is raked onto the corrected markets
        (:func:`rake_score_matrix`: the closest matrix whose cell sums give them),
        so the heatmap, the most likely score and every market shown next to
        them tell the same story -- without it the cells for Over 2.5 would
        still add up to the original model's probability. The goal rates
        (``home_rate``, ``away_rate``, ``rho``) stay the model's; the raw
        prediction is not modified. ``family`` defaults to the family of the
        fixture's competition.
        """
        if not self.enabled:
            return prediction
        fam = family if family is not None else prediction.fixture.girone.family
        probabilities = self.apply(prediction.probabilities, family=fam)
        fair = {m: {s: (1.0 / p if p > 0 else float("inf")) for s, p in sels.items()} for m, sels in probabilities.items()}
        matrix = rake_score_matrix(prediction.matrix, probabilities)
        return replace(prediction, matrix=matrix, probabilities=probabilities, fair_odds=fair)

    # ------------------------------------------------------------ persistence
    def to_dict(self) -> dict[str, Any]:
        """JSON-serialisable representation (for caching)."""
        return {
            "schema_version": _SCHEMA_VERSION,
            "min_gain": float(self.min_gain),
            "split_date": self.split_date.isoformat() if self.split_date is not None else None,
            "n_matches": int(self.n_matches),
            "fitted_at": self.fitted_at,
            "markets": [c.to_dict() for c in self.markets.values()],
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "RecalibrationModel":
        """Inverse of :meth:`to_dict` (``ValueError`` for an unknown schema version)."""
        version = int(payload.get("schema_version", _SCHEMA_VERSION))
        if version != _SCHEMA_VERSION:
            raise ValueError(f"unsupported recalibration schema version {version}")
        markets = [MarketCalibration.from_dict(item) for item in payload.get("markets", [])]
        split = payload.get("split_date")
        return cls(
            markets={c.market: c for c in markets},
            min_gain=float(payload.get("min_gain", DEFAULT_MIN_GAIN)),
            split_date=date.fromisoformat(str(split)) if split else None,
            n_matches=int(payload.get("n_matches", 0)),
            fitted_at=str(payload.get("fitted_at", "")),
        )


@lru_cache(maxsize=8)
def _score_partitions(size: int) -> tuple[tuple[Market, np.ndarray], ...]:
    """For a ``size x size`` score matrix, the selection index of every cell in each market (as :func:`markets_from_matrix`)."""
    home, away = np.meshgrid(np.arange(size), np.arange(size), indexing="ij")
    goals = home + away
    parts: list[tuple[Market, np.ndarray]] = [
        (Market.MATCH_ODDS, np.where(home > away, 0, np.where(home == away, 1, 2)).ravel()),  # 1, X, 2
        (Market.BTTS, np.where((home > 0) & (away > 0), 0, 1).ravel()),  # GG, NG
    ]
    parts += [(market, np.where(goals > line, 0, 1).ravel()) for market, line in _OU_LINES.items()]  # Over, Under
    for _market, labels in parts:
        labels.setflags(write=False)
    return tuple(parts)


def rake_score_matrix(
    matrix: np.ndarray,
    probabilities: Mapping[Any, Mapping[str, float]],
    *,
    max_iter: int = RAKE_MAX_ITER,
    tol: float = RAKE_TOL,
) -> np.ndarray:
    """The score matrix closest to ``matrix`` whose cell sums give the market ``probabilities``.

    Every market is a partition of the cells (1X2: below, on and above the
    diagonal; GG/NG; Over/Under by total goals -- exactly as
    :func:`src.engine.model.markets_from_matrix` sums them). Iterative
    proportional fitting (raking) rescales the cells of each part in turn until
    every market matches: the result is the matrix of minimum Kullback-Leibler
    divergence from ``matrix`` with those markets, so the shape inside each part
    (which scores are more likely than others) is the model's. Markets missing
    from ``probabilities`` (keyed by :class:`Market` or its value) or
    incomplete are left free. The result sums to 1; when the targets cannot be
    met exactly (an incoherent set) the last iterate, ``max_iter`` cycles
    later, is returned. A new array: ``matrix`` is not modified.
    """
    source = np.asarray(matrix, dtype=float)
    if source.ndim != 2 or source.shape[0] != source.shape[1]:
        raise ValueError(f"matrix must be square, got shape {source.shape}")
    targets: dict[Market, np.ndarray] = {}
    for key, sels in probabilities.items():
        try:
            market = market_of(key)
        except ValueError:
            continue
        values = np.array([sels.get(s, np.nan) for s in SELECTIONS[market]], dtype=float)
        if np.isfinite(values).all() and (values >= 0.0).all() and values.sum() > 0.0:
            targets[market] = values / values.sum()
    cells = source.ravel().copy()
    parts = [(labels, targets[market]) for market, labels in _score_partitions(source.shape[0]) if market in targets]
    for _cycle in range(max(int(max_iter), 0)):
        worst = 0.0
        for labels, target in parts:
            sums = np.bincount(labels, weights=cells, minlength=target.size)
            worst = max(worst, float(np.abs(sums - target).max()))
            factor = np.divide(target, sums, out=np.ones_like(target), where=sums > 0.0)  # an empty part cannot take mass
            cells *= factor[labels]
        if worst <= tol:
            break
    total = float(cells.sum())
    if not (total > 0.0 and np.isfinite(total)):
        return source.copy()
    return (cells / total).reshape(source.shape)


def _coherent_mapping(out: dict[Any, dict[str, float]], resolved: Mapping[Market, Any]) -> None:
    """In place: ``P(O3.5) <= P(O2.5) <= P(O1.5)`` and ``P(GG) <= P(O1.5)`` (complements updated)."""

    def get(market: Market, selection: str) -> float | None:
        key = resolved.get(market)
        return None if key is None else out[key].get(selection)

    def put_first(market: Market, value: float) -> None:
        """Set the first selection (Over, GG) of a two-way market and its complement."""
        key = resolved[market]
        first, second = SELECTIONS[market]
        out[key][first] = float(value)
        out[key][second] = 1.0 - float(value)

    o15 = get(Market.OVER_UNDER_15, "Over 1.5")
    o25 = get(Market.OVER_UNDER_25, "Over 2.5")
    o35 = get(Market.OVER_UNDER_35, "Over 3.5")
    gg = get(Market.BTTS, "GG")
    if o15 is not None and o25 is not None and o25 > o15:
        put_first(Market.OVER_UNDER_25, o15)
        o25 = o15
    if o25 is not None and o35 is not None and o35 > o25:
        put_first(Market.OVER_UNDER_35, o25)
    if o15 is not None and gg is not None and gg > o15:
        put_first(Market.BTTS, o15)


def _coherent_frame(frame: pd.DataFrame, prefix: str) -> None:
    """In place on a frame: the same coherence rules as :func:`_coherent_mapping`."""
    cols = {name: f"{prefix}{name}" for name in ("O15", "U15", "O25", "U25", "O35", "U35", "GG", "NG")}
    present = {name for name, column in cols.items() if column in frame.columns}
    for low, low_complement, high in (("O25", "U25", "O15"), ("O35", "U35", "O25"), ("GG", "NG", "O15")):
        if not {low, low_complement, high} <= present:
            continue
        values = frame[cols[low]].to_numpy(dtype=float)
        cap = frame[cols[high]].to_numpy(dtype=float)
        violated = values > cap
        if violated.any():  # only the incoherent rows are touched: a coherent frame stays bit-for-bit identical
            frame.loc[violated, cols[low]] = cap[violated]
            frame.loc[violated, cols[low_complement]] = 1.0 - cap[violated]


# =============================================================================
# Fit and evaluation
# =============================================================================
def time_split(frame: pd.DataFrame, *, test_fraction: float = DEFAULT_TEST_FRACTION, split: Literal["time"] | date | str = "time") -> tuple[pd.DataFrame, pd.DataFrame, date | None]:
    """``(earlier, later, first day of later)``: the time split of a backtest frame on a date boundary.

    ``split='time'`` puts the latest ``test_fraction`` of the matches (whole
    days, never half a day) in ``later``; a date (or ISO string) cuts there
    instead. ``later`` is empty (and the date ``None``) when there is nothing to
    split.
    """
    if frame.empty:
        return frame, frame, None
    dates = pd.to_datetime(frame["date"]).dt.normalize()
    if isinstance(split, str) and split != "time":
        cut = pd.Timestamp(date.fromisoformat(split[:10]))
    elif isinstance(split, (date, datetime)):
        cut = pd.Timestamp(split).normalize()
    elif split == "time":
        if not 0.0 < float(test_fraction) < 1.0:
            raise ValueError(f"test_fraction must be in (0, 1), got {test_fraction}")
        ordered = np.sort(dates.to_numpy())
        position = int(math.floor(len(ordered) * (1.0 - float(test_fraction))))
        position = min(max(position, 0), len(ordered) - 1)
        cut = pd.Timestamp(ordered[position])
        if cut == pd.Timestamp(ordered[0]):  # everything on one day: no time split possible
            return frame, frame.iloc[0:0], None
    else:
        raise ValueError(f"split must be 'time' or a date, got {split!r}")
    earlier = frame.loc[(dates < cut).to_numpy(dtype=bool)]
    later = frame.loc[(dates >= cut).to_numpy(dtype=bool)]
    return earlier, later, cut.date() if len(later) else None


def _verdict(spec: MarketSpec, fit_part: pd.DataFrame, eval_part: pd.DataFrame, *, prefix: str, l2: float, min_n: int, family_column: str) -> tuple[float, float, float, float, int]:
    """Out-of-sample ``(log_loss_before, log_loss_after, brier_before, brier_after, n_eval)``: params from ``fit_part`` scored on ``eval_part``."""
    pooled, by_family, _n = _fit_market(fit_part, spec, prefix=prefix, l2=l2, min_n=min_n, family_column=family_column)
    probs = spec.probabilities(eval_part, prefix)
    y = spec.outcome_index(eval_part)
    keep = (y >= 0) & np.isfinite(probs).all(axis=1)
    probs, y = probs[keep], y[keep]
    families = eval_part[family_column].astype(str).to_numpy()[keep] if family_column in eval_part.columns else None
    adjusted = _adjust_market(spec, probs, families, pooled, by_family) if len(y) else probs
    return (
        _log_loss(probs, y),
        _log_loss(adjusted, y),
        _brier(probs, y, spec.is_binary),
        _brier(adjusted, y, spec.is_binary),
        int(len(y)),
    )


def fit_recalibration(
    frame: pd.DataFrame,
    *,
    markets: Iterable[Market | str] = DEFAULT_MARKETS,
    min_n: int = DEFAULT_MIN_N,
    l2: float = DEFAULT_L2,
    split: Literal["time"] | date | str = "time",
    test_fraction: float = DEFAULT_TEST_FRACTION,
    min_gain: float = DEFAULT_MIN_GAIN,
    min_eval: int = DEFAULT_MIN_EVAL,
    refit_full: bool = True,
    prefix: str = "p_",
    family_column: str = _FAMILY_COLUMN,
) -> RecalibrationModel:
    """Learn the correction of every market from a walk-forward frame and decide where to use it.

    ``frame`` must hold **out-of-sample** probabilities
    (:attr:`src.engine.backtest.BacktestResult.frame`): in-sample probabilities
    would teach the correction the model's own overfit. For each market the
    parameters are fitted on the earlier part of the :func:`time_split` and
    the log loss of the later part is compared before and after; the market
    is enabled only with a relative gain of at least ``min_gain`` on at least
    ``min_eval`` matches. With ``refit_full`` the parameters of an enabled
    market are then refitted on every match (the verdict stays the
    out-of-sample one); a disabled market keeps the identity. ``min_n`` is the
    minimum number of matches for a family to get its own parameters, ``l2``
    the shrinkage towards the identity.
    """
    if float(l2) < 0:
        raise ValueError(f"l2 must be >= 0, got {l2}")
    started = time.perf_counter()
    earlier, later, split_date = time_split(frame, test_fraction=test_fraction, split=split)
    calibrations: dict[Market, MarketCalibration] = {}
    for market in markets:
        spec = market_spec(market)
        if frame.empty:
            continue
        if later.empty or earlier.empty:
            calibrations[spec.market] = MarketCalibration(spec.market, _identity(spec), reason="dati insufficienti per una verifica fuori campione")
            continue
        ll_before, ll_after, br_before, br_after, n_eval = _verdict(spec, earlier, later, prefix=prefix, l2=l2, min_n=min_n, family_column=family_column)
        gain = (ll_before - ll_after) / ll_before if np.isfinite(ll_before) and ll_before > 0 else float("nan")
        enabled = bool(n_eval >= int(min_eval) and np.isfinite(gain) and gain >= float(min_gain))
        if n_eval < int(min_eval):
            reason = f"solo {n_eval} partite di verifica (minimo {int(min_eval)})"
        elif not enabled:
            reason = f"miglioramento {_pct(gain)} sotto la soglia {_pct(float(min_gain))}"
        else:
            reason = ""
        if enabled:
            source = frame if refit_full else earlier
            pooled, by_family, n_fit = _fit_market(source, spec, prefix=prefix, l2=l2, min_n=min_n, family_column=family_column)
        else:
            pooled, by_family, n_fit = _identity(spec), {}, int(len(earlier))
        calibrations[spec.market] = MarketCalibration(
            market=spec.market,
            pooled=pooled,
            by_family=by_family,
            enabled=enabled,
            n_fit=n_fit,
            n_eval=n_eval,
            log_loss_before=ll_before,
            log_loss_after=ll_after,
            brier_before=br_before,
            brier_after=br_after,
            reason=reason,
        )
    model = RecalibrationModel(
        markets=calibrations,
        min_gain=float(min_gain),
        split_date=split_date,
        n_matches=int(len(frame)),
        fitted_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
    )
    log.info(
        "recalibration fitted",
        extra={
            "n_matches": int(len(frame)),
            "split_date": split_date.isoformat() if split_date else None,
            "enabled": [m.value for m in model.enabled_markets],
            "gains": {m.value: (round(c.gain, 4) if np.isfinite(c.gain) else None) for m, c in calibrations.items()},
            "seconds": round(time.perf_counter() - started, 3),
        },
    )
    return model


def evaluate_recalibration(
    frame: pd.DataFrame,
    model: RecalibrationModel,
    *,
    split: Literal["time"] | date | str = "time",
    test_fraction: float = DEFAULT_TEST_FRACTION,
    prefix: str = "p_",
    family_column: str = _FAMILY_COLUMN,
) -> pd.DataFrame:
    """Log loss and Brier before/after ``model`` on the **later** part of the time split of ``frame``.

    Columns ``market, n_eval, log_loss_before, log_loss_after, gain,
    brier_before, brier_after, enabled``. Out-of-sample only when ``model`` was
    fitted on matches before the split (e.g. ``fit_recalibration(..., refit_full=False)``
    on the same frame, or a model fitted on an older backtest); a model
    refitted on the whole frame is scored in sample here, which flatters it.
    Every market of the model is scored, enabled or not (a disabled market
    reads ``gain == 0``).
    """
    _earlier, later, _split_date = time_split(frame, test_fraction=test_fraction, split=split)
    adjusted = model.apply_frame(later, prefix=prefix, family_column=family_column)
    rows: list[dict[str, Any]] = []
    for market, calibration in model.markets.items():
        spec = MARKET_SPECS[market]
        if later.empty:
            continue
        raw = spec.probabilities(later, prefix)
        new = spec.probabilities(adjusted, prefix)
        y = spec.outcome_index(later)
        keep = (y >= 0) & np.isfinite(raw).all(axis=1)
        before = _log_loss(raw[keep], y[keep])
        after = _log_loss(new[keep], y[keep])
        rows.append(
            {
                "market": market.value,
                "n_eval": int(keep.sum()),
                "log_loss_before": before,
                "log_loss_after": after,
                "gain": (before - after) / before if np.isfinite(before) and before > 0 else float("nan"),
                "brier_before": _brier(raw[keep], y[keep], spec.is_binary),
                "brier_after": _brier(new[keep], y[keep], spec.is_binary),
                "enabled": calibration.enabled,
            }
        )
    return pd.DataFrame(rows, columns=["market", "n_eval", "log_loss_before", "log_loss_after", "gain", "brier_before", "brier_after", "enabled"])


#: Matches of earlier matchdays needed before :func:`rolling_recalibration` corrects a matchday.
DEFAULT_MIN_HISTORY: Final[int] = 500


@dataclass(frozen=True)
class RollingRecalibration:
    """Outcome of :func:`rolling_recalibration`.

    ``frame`` is a copy of the backtest frame whose probabilities were, matchday
    by matchday, corrected by a model fitted on the **earlier** matchdays only
    (raw where no correction was validated yet); ``steps`` has one row per
    matchday: ``matchday``, ``n_history`` (matches available before it),
    ``enabled`` (the markets corrected, comma-separated, ``''`` for none).
    """

    frame: pd.DataFrame
    steps: pd.DataFrame

    @property
    def n_corrected_matchdays(self) -> int:
        return int((self.steps["enabled"] != "").sum()) if len(self.steps) else 0


def rolling_recalibration(
    frame: pd.DataFrame,
    *,
    min_history: int = DEFAULT_MIN_HISTORY,
    prefix: str = "p_",
    family_column: str = _FAMILY_COLUMN,
    **fit_kwargs: Any,
) -> RollingRecalibration:
    """Out-of-sample recalibrated probabilities of a backtest frame, for an honest "with calibration" track record.

    Correcting a backtest frame with a model fitted on that same frame
    (:meth:`RecalibrationModel.apply_frame` of :func:`fit_recalibration`) would
    judge the correction on the matches it learnt from. Here every matchday
    (column ``matchday``, else the date) is corrected by
    :func:`fit_recalibration` (``fit_kwargs`` forwarded, validation rule
    included) fitted only on the matches dated before it; with fewer than
    ``min_history`` such matches, or when no market passes the validation, the
    matchday keeps its raw probabilities. Deterministic; the input is not
    modified.
    """
    out = frame.copy()
    rows: list[dict[str, Any]] = []
    if frame.empty:
        return RollingRecalibration(out, pd.DataFrame(columns=["matchday", "n_history", "enabled"]))
    keys = frame["matchday"].astype(str) if "matchday" in frame.columns else pd.to_datetime(frame["date"]).dt.strftime("%Y-%m-%d")
    dates = pd.to_datetime(frame["date"])
    for key in sorted(keys.unique()):
        mask = (keys == key).to_numpy(dtype=bool)
        history = frame.loc[(dates < dates[mask].min()).to_numpy(dtype=bool)]
        enabled = ""
        if len(history) >= int(min_history):
            model = fit_recalibration(history, prefix=prefix, family_column=family_column, **fit_kwargs)
            if model.enabled:
                corrected = model.apply_frame(frame.loc[mask], prefix=prefix, family_column=family_column)
                columns = [c for c in corrected.columns if c.startswith(prefix)]
                out.loc[mask, columns] = corrected[columns].to_numpy()
                enabled = ", ".join(m.value for m in model.enabled_markets)
        rows.append({"matchday": key, "n_history": int(len(history)), "enabled": enabled})
    steps = pd.DataFrame(rows, columns=["matchday", "n_history", "enabled"])
    log.info("rolling recalibration", extra={"matchdays": int(len(steps)), "corrected": int((steps["enabled"] != "").sum())})
    return RollingRecalibration(out, steps)


# =============================================================================
# Hyper-parameter tuning
# =============================================================================
def _prior_label(prior: Any) -> str:
    if prior is None or (isinstance(prior, str) and prior.lower() in ("off", "none", "no")) or (isinstance(prior, (int, float)) and float(prior) == 0.0):
        return "off"
    if isinstance(prior, str) and prior.lower() == "auto":
        return "auto"
    return f"{float(prior):g}"


def _config_rule(xi: float, prior: str, registry: TeamRegistry | None, method: Literal["SLSQP", "L-BFGS-B"] = "L-BFGS-B"):  # noqa: ANN202 - returns a ConfigRule
    def rule(train: pd.DataFrame, season: str) -> DixonColesConfig:
        if prior == "auto":
            reg = registry if registry is not None and registry.season_label.replace("_", "/") == season else None
            return production_config(train, registry=reg, season=season, xi=xi, method=method)[0]
        if prior == "off":
            return DixonColesConfig(xi=xi, method=method)
        return DixonColesConfig.with_gaussian_prior(sigma_strength=float(prior), xi=xi, method=method)

    return rule


@dataclass(frozen=True)
class TuningResult:
    """Outcome of :func:`tune_hyperparameters`.

    ``table`` has one row per grid point (``xi``, ``prior``, ``n``, the mean
    out-of-sample log loss of every objective market, ``objective`` = their
    mean, ``seconds``), best first; ``best`` is the winning ``{'xi', 'prior'}``
    and ``production`` the objective of the production setting (xi 0.0018,
    prior ``'auto'``) when it was in the grid. ``prior`` is ``'auto'``
    (production rule), ``'off'`` or the ``sigma_strength`` of the prior.
    """

    table: pd.DataFrame
    best: dict[str, Any]
    production: float | None = None
    seconds: float = 0.0

    @property
    def best_objective(self) -> float:
        return float(self.table["objective"].iloc[0]) if len(self.table) else float("nan")

    @property
    def improvement(self) -> float:
        """Relative improvement of the best point over the production setting (``NaN`` without it)."""
        if self.production is None or not np.isfinite(self.production) or self.production <= 0:
            return float("nan")
        return (self.production - self.best_objective) / self.production

    def config_for(
        self,
        train: pd.DataFrame,
        *,
        registry: TeamRegistry | None = None,
        season: str | None = None,
        method: Literal["SLSQP", "L-BFGS-B"] = "SLSQP",
    ) -> DixonColesConfig:
        """The :class:`DixonColesConfig` of the best point for a training frame (``'auto'`` resolved with the production rule)."""
        prior = str(self.best.get("prior", "auto"))
        xi = float(self.best.get("xi", PRODUCTION_XI))
        if prior == "auto":
            return production_config(train, registry=registry, season=season, xi=xi, method=method)[0]
        return _config_rule(xi, prior, registry, method)(train, season or "")

    def describe(self) -> str:
        """Italian one-liner: the best point and its gain over production."""
        if self.table.empty:
            return "Ottimizzazione di xi e prior: nessun risultato (backtest vuoto)."
        best = f"xi={float(self.best['xi']):g}, prior {self.best['prior']}"
        text = f"Ottimizzazione walk-forward: migliore {best} (log loss media {self.best_objective:.4f})"
        if np.isfinite(self.improvement):
            text += f", {_pct(self.improvement)} rispetto alla configurazione di produzione"
        return text + f"; {len(self.table)} combinazioni in {self.seconds:.0f} s."

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": _SCHEMA_VERSION,
            "table": self.table.to_dict(orient="records"),
            "best": dict(self.best),
            "production": self.production,
            "seconds": float(self.seconds),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "TuningResult":
        return cls(
            table=pd.DataFrame(list(payload.get("table", []))),
            best=dict(payload.get("best", {})),
            production=None if payload.get("production") is None else float(payload["production"]),
            seconds=float(payload.get("seconds", 0.0)),
        )


def tune_hyperparameters(
    matches: pd.DataFrame,
    registry: TeamRegistry | None = None,
    grid: Mapping[str, Sequence[Any]] | None = None,
    *,
    seasons: Sequence[str] | None = None,
    block: BlockName = DEFAULT_BLOCK,
    min_train_matches: int = DEFAULT_MIN_TRAIN_MATCHES,
    markets: Iterable[Market | str] = DEFAULT_TUNING_MARKETS,
    progress: ProgressCallback | None = None,
) -> TuningResult:
    """Rank a small grid of time decays ``xi`` and prior strengths by walk-forward log loss (slow-ish: cache it).

    ``grid`` has ``'xi'`` (per day) and ``'prior'`` (``'auto'`` = production
    rule, ``None``/``'off'`` = no prior, a number = ``sigma_strength`` of
    :meth:`DixonColesConfig.with_gaussian_prior`) lists, default
    :data:`DEFAULT_TUNING_GRID` (9 points). Every point runs
    :func:`src.engine.backtest.walk_forward` on the same seasons and blocks, so
    every point is scored on the same matches; the objective is the mean of
    the per-market mean log loss over ``markets``. ``progress(fraction,
    message)`` is called after every point.
    """
    chosen = dict(DEFAULT_TUNING_GRID if grid is None else grid)
    xis = [float(x) for x in chosen.get("xi", (PRODUCTION_XI,))]
    priors = [_prior_label(p) for p in chosen.get("prior", ("auto",))]
    if any(x < 0 for x in xis):
        raise ValueError(f"xi values must be >= 0, got {xis}")
    specs = [market_spec(m) for m in markets]
    report = progress if progress is not None else (lambda _fraction, _message: None)
    started = time.perf_counter()
    points = [(xi, prior) for xi in dict.fromkeys(xis) for prior in dict.fromkeys(priors)]
    rows: list[dict[str, Any]] = []
    for position, (xi, prior) in enumerate(points, start=1):
        t0 = time.perf_counter()
        result = walk_forward(
            matches,
            registry=registry,
            seasons=seasons,
            config=_config_rule(xi, prior, registry),
            block=block,
            min_train_matches=min_train_matches,
        )
        row: dict[str, Any] = {"xi": xi, "prior": prior, "n": result.n_matches}
        losses: list[float] = []
        for spec in specs:
            probs = spec.probabilities(result.frame, "p_") if len(result.frame) else np.empty((0, len(spec.selections)))
            y = spec.outcome_index(result.frame) if len(result.frame) else np.empty(0, dtype=np.int64)
            keep = (y >= 0) & np.isfinite(probs).all(axis=1) if len(y) else np.empty(0, dtype=bool)
            value = _log_loss(probs[keep], y[keep]) if len(y) else float("nan")
            row[f"log_loss_{spec.key}"] = value
            losses.append(value)
        row["objective"] = float(np.mean(losses)) if losses and all(np.isfinite(losses)) else float("nan")
        row["seconds"] = time.perf_counter() - t0
        rows.append(row)
        report(position / len(points), f"Ottimizzazione: {position}/{len(points)} (xi={xi:g}, prior {prior})")
    table = pd.DataFrame(rows).sort_values(["objective", "xi", "prior"], kind="stable", na_position="last").reset_index(drop=True)
    best = {"xi": float(table["xi"].iloc[0]), "prior": str(table["prior"].iloc[0])} if len(table) else {"xi": PRODUCTION_XI, "prior": "auto"}
    production_rows = table.loc[(np.isclose(table["xi"], PRODUCTION_XI)) & (table["prior"] == "auto")] if len(table) else table
    production = float(production_rows["objective"].iloc[0]) if len(production_rows) else None
    seconds = time.perf_counter() - started
    result_out = TuningResult(table=table, best=best, production=production, seconds=float(seconds))
    log.info("hyper-parameters tuned", extra={"best": best, "production": production, "points": len(points), "seconds": round(seconds, 1)})
    return result_out
