"""Walk-forward backtest of the Dixon-Coles engine: honest probabilities, scores and track record.

Why walk-forward
----------------
The production fit (:meth:`src.engine.model.DixonColesEngine.fit` on every
played match) has *seen* the results it would be judged on: its in-sample
probabilities flatter the model and must never be used to measure accuracy.
:func:`walk_forward` replays the season instead. The evaluated matches are cut
into blocks (``block='week'``: Tuesday to Monday, so a weekend round and its
Monday posticipo stay together; ``'block'``: the blocks of the football week of
:func:`src.engine.calendar.week_block`, weekend Friday-Monday and midweek
Tuesday-Thursday; ``'day'``: one block per day) and every block is predicted by
a model fitted **only on the matches dated strictly before the block's first
match** -- history included, time decay as in production (reference date =
latest training match), the production prior rule (:func:`production_config`)
and L-BFGS-B warm-started from the previous block's parameters. As in
production, one model is fitted per *fit group* (the competitions of one
country, :func:`fit_group_of_girone`; a single group with the Italian
perimeter), each on the training matches of its own group. Each evaluated
match carries the pre-match probabilities of every market, the outcome, the
size and last date of its training set (``train_end`` < ``date`` always) and
whether one of its teams had never been seen (``cold_start``: predicted with
the prior strengths, ``alpha = 0`` and the mean defence of its competition). The
result is deterministic for a given input.

A naive baseline is stored next to the model: the **base rates** of the
competition (1/X/2, GG, Over 1.5/2.5/3.5) over the 365 days before the block,
shrunk towards the rates of every competition (``base_*`` columns). It is what
a "no-model" bettor knows and what the model must beat.

Scores
------
:func:`metrics` gives, per section / competition and market (1X2, GG/NG,
Over/Under 1.5-2.5-3.5): the number of matches, the Brier score (multi-class,
0-2, for 1X2; binary, 0-1, for the two-way markets), the log loss, the accuracy
of the top pick, the mean predicted probability of the reference selection
against its observed frequency, and the skill against the naive baseline
``1 - logloss_model / logloss_naive`` (positive = better than the base rates).
:func:`reliability` bins the predicted probabilities of one selection against
the observed frequencies with Wilson intervals; :func:`market_comparison` scores
the de-vigged pre-match bookmaker probabilities (the odds history of the
completed matches, :func:`odds_table`) on the same matches, with the skill
against the market ``1 - logloss_model / logloss_market``.

Track record
------------
:func:`picks_track_record` rebuilds, for every past matchday (the block of the
football week of the match, column ``matchday``), the lists the dashboard
would have shown -- the top-N draws and the top-N GG, with the ranking of
:func:`src.engine.picks.rank_picks`, across sections and within each section --
and records hits, predicted probability, realised frequency, the top-k
accumulators and the ROI at the real pre-match odds when available.
:func:`slips_track_record` does the same for the matchday slips produced by an
injected builder (:data:`SlipBuilder`; :func:`default_slip_builder` is a minimal
internal one), :func:`value_track_record` for the value singles (EV above a
threshold at the best price). Rates come with Wilson intervals
(:func:`wilson_interval`), ROI with a skew-aware interval (:func:`roi_interval`:
percentile bootstrap of the mean profit, or the exact bounds of the win rate
times the mean price when wins are too few -- never a zero-width interval,
never below -100%): one anecdotal win is not evidence, the sample size is.

This module is pure (no ``streamlit``, no network) and imports nothing from the
dashboard or the ingestion layers.
"""
from __future__ import annotations

import logging
import threading
import time
from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, field, replace
from datetime import date, timedelta
from typing import Any, Final, Literal

import numpy as np
import pandas as pd
from scipy.special import betaincinv, gammaln, ndtr, xlogy

from src.common.logging_config import get_logger
from src.common.schemas import SEASON_START_MONTH, SELECTIONS, Fixture, Girone, Market, MatchPrediction, season_bounds, validate_matches_df
from src.common.teams import TeamRegistry
from src.engine.calendar import week_block
from src.engine.model import DixonColesConfig, DixonColesEngine, FittedParams, dixon_coles_pmf
from src.engine.picks import DEFAULT_N_BTTS, DEFAULT_N_DRAWS, rank_picks
from src.engine.value import devig

__all__ = [
    "ALL_SECTIONS",
    "BACKTEST_COLUMNS",
    "BASE_COLUMNS",
    "BLOCKS",
    "DEFAULT_ACCUMULATOR_SIZES",
    "DEFAULT_BLOCK",
    "DEFAULT_MARKETS",
    "DEFAULT_MIN_TRAIN_MATCHES",
    "DEFAULT_MIXED_SELECTIONS",
    "DEFAULT_N_MIXED",
    "DEFAULT_N_SLIP_BTTS",
    "DEFAULT_N_SLIP_DRAWS",
    "DEFAULT_VALUE_MIN_EV",
    "MARKET_SPECS",
    "OUTCOME_COLUMNS",
    "PRIOR_MIN_MATCHES_PER_TEAM",
    "PROB_COLUMNS",
    "PRODUCTION_XI",
    "ROI_BOOTSTRAP_SAMPLES",
    "ROI_BOOTSTRAP_SEED",
    "ROI_MIN_EVENTS",
    "SELECTION_SUFFIX",
    "BacktestLeg",
    "BacktestResult",
    "BacktestSlip",
    "MarketSpec",
    "PicksTrackRecord",
    "ProgressCallback",
    "SlipBuilder",
    "SlipsTrackRecord",
    "ValueTrackRecord",
    "default_slip_builder",
    "exact_binomial_interval",
    "fit_group_of_girone",
    "market_comparison",
    "market_of",
    "market_spec",
    "metrics",
    "odds_table",
    "picks_track_record",
    "prediction_columns",
    "production_config",
    "production_prior_on",
    "reliability",
    "roi_interval",
    "section_of_girone",
    "selection_calibration",
    "slips_track_record",
    "value_track_record",
    "walk_forward",
    "wilson_interval",
]

log = get_logger(__name__)
_model_log = logging.getLogger("src.engine.model")

# =============================================================================
# Constants and market specifications
# =============================================================================
#: Supported block sizes of :func:`walk_forward`.
BLOCKS: Final[tuple[str, ...]] = ("week", "block", "day")
DEFAULT_BLOCK: Final[str] = "week"
BlockName = Literal["week", "block", "day"]
#: A block is evaluated only when at least this many matches precede it.
DEFAULT_MIN_TRAIN_MATCHES: Final[int] = 150
#: Time decay of the production fit (1/day), as in :func:`src.dashboard.state.build_state`.
PRODUCTION_XI: Final[float] = 0.0018
#: Mirror of :data:`src.dashboard.state.PRIOR_MIN_MATCHES_PER_TEAM`: the Gaussian prior is on while the
#: least-covered competition has fewer current-season matches per current team than this.
PRIOR_MIN_MATCHES_PER_TEAM: Final[float] = 15.0
#: Window (days before the block) and pseudo-count of the naive base rates.
BASE_RATE_WINDOW_DAYS: Final[int] = 365
BASE_RATE_PSEUDO_COUNT: Final[float] = 20.0
#: Probabilities are clipped to ``[eps, 1 - eps]`` inside the log loss.
LOG_LOSS_EPS: Final[float] = 1e-12
#: Name of the scope that ranks every section together in the track record.
ALL_SECTIONS: Final[str] = "Tutte le sezioni"
#: Top-k accumulators tracked for every pick list.
DEFAULT_ACCUMULATOR_SIZES: Final[tuple[int, ...]] = (2, 3, 4, 5)
#: Default slips of :func:`default_slip_builder`: 5 draws, 10 GG, a mixed slip of 6 legs among these signs.
DEFAULT_N_SLIP_DRAWS: Final[int] = 5
DEFAULT_N_SLIP_BTTS: Final[int] = 10
DEFAULT_N_MIXED: Final[int] = 6
DEFAULT_MIXED_SELECTIONS: Final[tuple[str, ...]] = ("1", "2", "GG", "Over 2.5")
#: Minimum expected value of a value single (as the dashboard's default ``min_ev``).
DEFAULT_VALUE_MIN_EV: Final[float] = 0.035
_Z95: Final[float] = 1.959963984540054
#: :func:`roi_interval` bootstraps the mean profit only when at least this many bets won **and** this many lost;
#: with rarer wins (accumulators, long odds) or rarer losses the resamples cannot show what was never seen, so the
#: bounds come from the exact (Clopper-Pearson) interval of the win rate times the mean price instead.
ROI_MIN_EVENTS: Final[int] = 5
#: Resamples of the percentile bootstrap of :func:`roi_interval` (fixed seed: the interval is deterministic).
ROI_BOOTSTRAP_SAMPLES: Final[int] = 2000
ROI_BOOTSTRAP_SEED: Final[int] = 20_261_108

#: Column suffix of every selection (``p_O25`` is the model probability of ``Over 2.5``).
SELECTION_SUFFIX: Final[dict[str, str]] = {
    "1": "1",
    "X": "X",
    "2": "2",
    "GG": "GG",
    "NG": "NG",
    "Over 1.5": "O15",
    "Under 1.5": "U15",
    "Over 2.5": "O25",
    "Under 2.5": "U25",
    "Over 3.5": "O35",
    "Under 3.5": "U35",
}
_MARKET_SUFFIX: Final[dict[Market, str]] = {
    Market.MATCH_ODDS: "1X2",
    Market.BTTS: "GGNG",
    Market.OVER_UNDER_15: "OU15",
    Market.OVER_UNDER_25: "OU25",
    Market.OVER_UNDER_35: "OU35",
}


@dataclass(frozen=True)
class MarketSpec:
    """How one market lives in a backtest frame.

    ``selections`` follow :data:`~src.common.schemas.SELECTIONS`; ``suffixes``
    are their column suffixes (``p_<suffix>``, ``base_<suffix>``,
    ``odds_<suffix>``, ``book_<suffix>``); ``outcome`` is the outcome column
    (``res_1X2`` holds the winning selection, the binary markets hold ``1``
    when their **first** selection -- GG, Over -- happened). The reference
    selection of :func:`metrics` is the first one.
    """

    market: Market
    selections: tuple[str, ...]
    suffixes: tuple[str, ...]
    outcome: str

    @property
    def key(self) -> str:
        """The market value used in every output frame (``'1X2'``, ``'GG/NG'``, ``'OU2.5'``)."""
        return self.market.value

    @property
    def is_binary(self) -> bool:
        return len(self.selections) == 2

    def columns(self, prefix: str = "p_") -> tuple[str, ...]:
        """Probability columns of the market with ``prefix`` (``('p_1', 'p_X', 'p_2')``)."""
        return tuple(f"{prefix}{s}" for s in self.suffixes)

    def outcome_index(self, frame: pd.DataFrame) -> np.ndarray:
        """Index (in :attr:`selections`) of the selection that happened, one per row."""
        if self.is_binary:
            happened = pd.to_numeric(frame[self.outcome], errors="coerce").fillna(0).to_numpy(dtype=float)
            return np.where(happened > 0.5, 0, 1).astype(np.int64)
        values = frame[self.outcome].astype(str).to_numpy()
        lookup = {s: i for i, s in enumerate(self.selections)}
        return np.array([lookup.get(v, -1) for v in values], dtype=np.int64)

    def probabilities(self, frame: pd.DataFrame, prefix: str = "p_") -> np.ndarray:
        """``(n, k)`` matrix of the market probabilities with ``prefix``."""
        return frame[list(self.columns(prefix))].to_numpy(dtype=float)


MARKET_SPECS: Final[dict[Market, MarketSpec]] = {
    Market.MATCH_ODDS: MarketSpec(Market.MATCH_ODDS, ("1", "X", "2"), ("1", "X", "2"), "res_1X2"),
    Market.BTTS: MarketSpec(Market.BTTS, ("GG", "NG"), ("GG", "NG"), "gg"),
    Market.OVER_UNDER_15: MarketSpec(Market.OVER_UNDER_15, ("Over 1.5", "Under 1.5"), ("O15", "U15"), "o15"),
    Market.OVER_UNDER_25: MarketSpec(Market.OVER_UNDER_25, ("Over 2.5", "Under 2.5"), ("O25", "U25"), "o25"),
    Market.OVER_UNDER_35: MarketSpec(Market.OVER_UNDER_35, ("Over 3.5", "Under 3.5"), ("O35", "U35"), "o35"),
}
#: Markets scored by default, in display order.
DEFAULT_MARKETS: Final[tuple[Market, ...]] = (
    Market.MATCH_ODDS,
    Market.BTTS,
    Market.OVER_UNDER_15,
    Market.OVER_UNDER_25,
    Market.OVER_UNDER_35,
)

#: Model probability columns of a backtest frame.
PROB_COLUMNS: Final[tuple[str, ...]] = tuple(c for spec in MARKET_SPECS.values() for c in spec.columns("p_"))
#: Naive base-rate columns (same suffixes as :data:`PROB_COLUMNS`).
BASE_COLUMNS: Final[tuple[str, ...]] = tuple(c for spec in MARKET_SPECS.values() for c in spec.columns("base_"))
#: Outcome columns: ``res_1X2`` (``'1'``/``'X'``/``'2'``) and ``gg``/``o15``/``o25``/``o35`` (``0``/``1``).
OUTCOME_COLUMNS: Final[tuple[str, ...]] = ("res_1X2", "gg", "o15", "o25", "o35")
#: Column order of :attr:`BacktestResult.frame`.
BACKTEST_COLUMNS: Final[tuple[str, ...]] = (
    "match_id",
    "date",
    "season",
    "matchday",
    "block",
    "girone",
    "family",
    "section",
    "round",
    "home_team",
    "away_team",
    "home_goals",
    "away_goals",
    "lam",
    "mu",
    "rho",
    *PROB_COLUMNS,
    *OUTCOME_COLUMNS,
    *BASE_COLUMNS,
    "train_size",
    "train_end",
    "fit_converged",
    "fit_method",
    "prior",
    "cold_start",
)
#: Columns a slip builder receives: everything known **before** the matches (no goals, no outcomes).
_PRE_MATCH_COLUMNS: Final[tuple[str, ...]] = (
    "match_id",
    "date",
    "season",
    "matchday",
    "girone",
    "family",
    "section",
    "round",
    "home_team",
    "away_team",
    "lam",
    "mu",
    "rho",
    *PROB_COLUMNS,
)

#: ``progress(fraction in [0, 1], Italian message)``.
ProgressCallback = Callable[[float, str], None]


def market_of(value: Market | str) -> Market:
    """The :class:`Market` of any accepted spelling: the member, its value (``'GG/NG'``) or its name (``'BTTS'``)."""
    if isinstance(value, Market):
        return value
    text = str(value).strip()
    try:
        return Market(text)
    except ValueError:
        pass
    try:
        return Market[text.upper()]
    except KeyError:
        raise ValueError(f"unknown market {value!r} (expected one of {[m.value for m in Market]})") from None


def market_spec(market: Market | str) -> MarketSpec:
    """The :class:`MarketSpec` of a market (any spelling accepted by :func:`market_of`)."""
    return MARKET_SPECS[market_of(market)]


def _market_of_selection(selection: str) -> Market:
    for market, selections in SELECTIONS.items():
        if selection in selections:
            return market
    raise ValueError(f"unknown selection {selection!r}")


def prediction_columns(prefix: str = "p_") -> tuple[str, ...]:
    """The probability columns of every market with ``prefix`` (``p_`` model, ``base_`` naive, ``book_`` market)."""
    return tuple(c for spec in MARKET_SPECS.values() for c in spec.columns(prefix))


def _girone_order(code: str) -> int:
    try:
        return list(Girone).index(Girone.parse(code))
    except ValueError:
        return len(Girone)


# =============================================================================
# Sections
# =============================================================================
def section_of_girone(girone: Girone | str) -> str:
    """The dashboard section of a competition.

    Uses ``src.common.competitions.section_of`` when that helper exists (the
    sections are configuration-driven there), else falls back to the
    competition's family (``'Serie C'``, ``'Serie B'``, ``'Serie D'``).
    """
    g = Girone.parse(girone)
    try:
        from src.common import competitions as _competitions  # noqa: PLC0415 - optional helper, resolved at call time

        resolver = getattr(_competitions, "section_of", None)
    except ImportError:  # pragma: no cover - the module always exists in this repository
        resolver = None
    if callable(resolver):
        for argument in (g.value, g.competition, g):
            try:
                value = resolver(argument)
            except (TypeError, ValueError, KeyError, LookupError, AttributeError):
                continue
            if value is None:
                continue
            label = getattr(value, "label", None) or getattr(value, "name", None) or value
            text = str(label).strip()
            if text:
                return text
    return g.family


def fit_group_of_girone(girone: Girone | str) -> str:
    """The fit group of a competition: one Dixon-Coles model per group, as in production.

    Uses ``src.common.competitions.fit_group_of`` when that helper exists (the
    competitions of one country are fitted jointly, countries separately, see
    :mod:`src.engine.fitgroups`), else the competition's country.
    """
    g = Girone.parse(girone)
    try:
        from src.common import competitions as _competitions  # noqa: PLC0415 - optional helper, resolved at call time

        resolver = getattr(_competitions, "fit_group_of", None)
    except ImportError:  # pragma: no cover - the module always exists in this repository
        resolver = None
    if callable(resolver):
        try:
            value = resolver(g.value)
        except (TypeError, ValueError, KeyError, LookupError, AttributeError):
            value = None
        if value:
            return str(value)
    return g.country


def _section_map(codes: Iterable[str]) -> dict[str, str]:
    return {code: section_of_girone(code) for code in codes}


# =============================================================================
# Statistics helpers
# =============================================================================
def wilson_interval(successes: Any, n: Any, *, z: float = _Z95) -> tuple[Any, Any]:
    """Wilson score interval of a binomial proportion (vectorised; ``NaN`` where ``n == 0``).

    Accepts scalars or arrays; returns ``(low, high)`` of the same shape
    (floats for scalar input). Better behaved than the normal approximation
    for small samples and rates near 0 or 1.
    """
    k = np.asarray(successes, dtype=float)
    m = np.asarray(n, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        p = np.where(m > 0, k / np.where(m > 0, m, 1.0), np.nan)
        denom = 1.0 + z * z / m
        centre = (p + z * z / (2.0 * m)) / denom
        half = z * np.sqrt(p * (1.0 - p) / m + z * z / (4.0 * m * m)) / denom
        low = np.where(m > 0, np.where(k <= 0, 0.0, np.clip(centre - half, 0.0, 1.0)), np.nan)
        high = np.where(m > 0, np.where(k >= m, 1.0, np.clip(centre + half, 0.0, 1.0)), np.nan)
    if low.ndim == 0:
        return float(low), float(high)
    return low, high


def exact_binomial_interval(successes: int, n: int, *, z: float = _Z95) -> tuple[float, float]:
    """Clopper-Pearson ("exact") interval of a binomial proportion at the two-sided level of ``z``.

    Never narrower than the coverage requires (conservative), unlike the Wilson
    interval whose lower bound is too high with one or two successes. ``NaN``
    for ``n == 0``; ``0`` lower bound with no success, ``1`` upper bound when
    every trial succeeded.
    """
    k, m = int(successes), int(n)
    if m <= 0:
        return float("nan"), float("nan")
    if not 0 <= k <= m:
        raise ValueError(f"successes must be in [0, n], got {k} of {m}")
    alpha = 2.0 * (1.0 - float(ndtr(z)))
    low = 0.0 if k == 0 else float(betaincinv(k, m - k + 1, alpha / 2.0))
    high = 1.0 if k == m else float(betaincinv(k + 1, m - k, 1.0 - alpha / 2.0))
    return low, high


def _bootstrap_means(values: np.ndarray, n_boot: int, rng: np.random.Generator) -> np.ndarray:
    """``n_boot`` nonparametric bootstrap means of ``values`` (resampling with replacement), computed cheaply.

    Exact in distribution: of the ``n`` draws of a resample, the number that
    land on the smallest value (the lost bets, all ``-1``) is binomial and the
    others are uniform over the remaining values, so only those are drawn.
    """
    n = values.size
    floor = float(values.min())
    rest = values[values != floor]
    if rest.size == 0:
        return np.full(n_boot, floor)
    means: list[np.ndarray] = []
    chunk = max(1, 2_000_000 // rest.size)  # about rest.size draws per resample: bounded memory
    for done in range(0, n_boot, chunk):
        size = min(chunk, n_boot - done)
        hits = rng.binomial(n, rest.size / n, size=size)
        draws = rest[rng.integers(0, rest.size, size=int(hits.sum()))]
        cumulative = np.concatenate(([0.0], np.cumsum(draws)))
        ends = np.cumsum(hits)
        means.append((cumulative[ends] - cumulative[ends - hits] + (n - hits) * floor) / n)
    return np.concatenate(means)


def roi_interval(
    profits: Iterable[float] | np.ndarray,
    *,
    odds: Iterable[float] | np.ndarray | None = None,
    z: float = _Z95,
    min_events: int = ROI_MIN_EVENTS,
    n_boot: int = ROI_BOOTSTRAP_SAMPLES,
    seed: int = ROI_BOOTSTRAP_SEED,
) -> tuple[float, float, float]:
    """``(roi, low, high)`` of a series of profits per unit staked, with a skew-aware interval at level ``z``.

    ``roi`` is the mean profit per unit. A bet at stake 1 either loses the stake
    (profit ``-1``) or wins ``odds - 1``: the payoff is bounded below and, for
    accumulators and long prices, extremely skewed, so a normal approximation
    (``roi +- z * sd / sqrt(n)``) lies twice -- zero width when nothing won
    ("certainly -100%") and a lower bound under the possible minimum of -100%
    after one big win. Here:

    * with at least ``min_events`` winning **and** ``min_events`` losing bets:
      percentile bootstrap of the mean profit (``n_boot`` resamples, fixed
      ``seed``: deterministic);
    * otherwise (no win, or a handful): the bounds of the exact
      Clopper-Pearson interval of the win rate (:func:`exact_binomial_interval`)
      times the mean price of the bets, minus 1 -- exact when every bet has the
      same price, an approximation otherwise. Clopper-Pearson rather than
      Wilson because with one or two wins the Wilson lower bound is too high
      (Brown, Cai & DasGupta 2001) and two lucky wins out of twenty long-odds
      accumulators would read as a "significant" profit. With no win the lower
      bound is the floor -100% and the upper bound says how much ROI the sample
      cannot exclude. ``odds`` (aligned with ``profits``) gives the prices of
      the losing bets too; without it the mean price of the winning bets is
      used, and with no win and no ``odds`` the bounds are ``NaN`` (not
      determinable).

    The bounds never go below -100% and always contain ``roi``. ``NaN`` bounds
    for fewer than two profits, all ``NaN`` for none. ``NaN`` profits (no odds)
    are ignored.
    """
    values = np.asarray(list(profits) if not isinstance(profits, np.ndarray) else profits, dtype=float)
    finite = np.isfinite(values)
    prices: np.ndarray | None = None
    if odds is not None:
        prices = np.asarray(list(odds) if not isinstance(odds, np.ndarray) else odds, dtype=float)
        if prices.shape != values.shape:
            raise ValueError(f"odds must be aligned with profits ({prices.shape} != {values.shape})")
        prices = prices[finite]
    values = values[finite]
    if values.size == 0:
        return float("nan"), float("nan"), float("nan")
    mean = float(values.mean())
    if values.size < 2:
        return mean, float("nan"), float("nan")
    won = values > -1.0 + 1e-12  # a lost bet at stake 1 loses exactly the stake
    wins = int(won.sum())
    losses = int(values.size - wins)
    if wins >= min_events and losses >= min_events:
        alpha = 2.0 * (1.0 - float(ndtr(z)))
        means = _bootstrap_means(values, max(int(n_boot), 1), np.random.default_rng(seed))
        low, high = (float(q) for q in np.quantile(means, [alpha / 2.0, 1.0 - alpha / 2.0]))
    else:
        quoted = prices[np.isfinite(prices) & (prices > 1.0)] if prices is not None else np.empty(0)
        if quoted.size == 0:
            quoted = values[won] + 1.0  # the prices the profits reveal: those of the winning bets
        if quoted.size == 0:
            return mean, float("nan"), float("nan")
        price = float(quoted.mean())
        rate_low, rate_high = exact_binomial_interval(wins, values.size, z=z)
        low, high = price * rate_low - 1.0, price * rate_high - 1.0
    low = max(-1.0, min(low, mean))
    high = max(high, mean)
    return mean, low, high


def _log_loss_rows(probs: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Per-row ``-log p(outcome)`` with clipping (rows with an unknown outcome are ``NaN``)."""
    out = np.full(len(y), np.nan)
    ok = y >= 0
    if ok.any():
        picked = probs[np.flatnonzero(ok), y[ok]]
        out[ok] = -np.log(np.clip(picked, LOG_LOSS_EPS, 1.0))
    return out


def _brier_rows(probs: np.ndarray, y: np.ndarray, binary: bool) -> np.ndarray:
    """Per-row Brier score: binary ``(p_first - y_first)^2`` or multi-class ``sum_k (p_k - y_k)^2``."""
    onehot = np.zeros_like(probs)
    ok = y >= 0
    onehot[np.flatnonzero(ok), y[ok]] = 1.0
    if binary:
        out = (probs[:, 0] - onehot[:, 0]) ** 2
    else:
        out = ((probs - onehot) ** 2).sum(axis=1)
    return np.where(ok, out, np.nan)


def _accuracy_rows(probs: np.ndarray, y: np.ndarray) -> np.ndarray:
    top = np.argmax(probs, axis=1)
    return np.where(y >= 0, (top == y).astype(float), np.nan)


# =============================================================================
# Production configuration rule
# =============================================================================
def _season_labels(dates: pd.Series) -> pd.Series:
    """Season label of every date (``'2026/27'`` from July 2026 to June 2027), vectorised."""
    stamps = pd.to_datetime(dates)
    start = np.where(stamps.dt.month.to_numpy() >= SEASON_START_MONTH, stamps.dt.year.to_numpy(), stamps.dt.year.to_numpy() - 1)
    labels = {int(y): f"{int(y)}/{(int(y) + 1) % 100:02d}" for y in np.unique(start)}
    return pd.Series([labels[int(y)] for y in start], index=dates.index, dtype="string")


def _normalise_season(label: str) -> str:
    first, _ = season_bounds(str(label))
    return f"{first.year}/{(first.year + 1) % 100:02d}"


def production_prior_on(
    train: pd.DataFrame,
    *,
    registry: TeamRegistry | None = None,
    season: str | None = None,
    min_matches_per_team: float = PRIOR_MIN_MATCHES_PER_TEAM,
) -> bool:
    """The production prior rule (:func:`src.dashboard.state.prior_recommended`) on a training frame.

    Coverage is measured on the **current-season** matches of ``train``
    (``season``; all rows when ``None``): with ``registry`` only the rows
    between two current teams count and each registry competition present in
    those rows gets ``2 * matches / teams``; without it each competition of the
    rows counts its own teams. The prior is on while the least-covered
    competition is below ``min_matches_per_team`` (always on before the season
    has started). History never counts as coverage.
    """
    frame = train
    if season is not None and len(frame):
        frame = frame.loc[(_season_labels(frame["date"]) == _normalise_season(season)).to_numpy(dtype=bool)]
    present = set(frame["girone"].astype(str)) if len(frame) else set()
    per: list[float] = []
    if registry is not None:
        current = set(registry.canonical_names)
        if len(frame):
            both = frame["home_team"].isin(current) & frame["away_team"].isin(current)
            frame = frame.loc[both.to_numpy(dtype=bool)]
        for g, teams in registry.teams_by_girone.items():
            if not teams or Girone.parse(g).value not in present:
                continue
            members = set(teams)
            n = int((frame["home_team"].isin(members) | frame["away_team"].isin(members)).sum()) if len(frame) else 0
            per.append(2.0 * n / len(members))
    elif len(frame):
        for _code, group in frame.groupby("girone", observed=True):
            teams = set(group["home_team"].astype(str)) | set(group["away_team"].astype(str))
            per.append(2.0 * len(group) / len(teams))
    thinnest = min(per) if per else 0.0
    return bool(thinnest < float(min_matches_per_team))


def production_config(
    train: pd.DataFrame,
    *,
    registry: TeamRegistry | None = None,
    season: str | None = None,
    xi: float = PRODUCTION_XI,
    method: Literal["SLSQP", "L-BFGS-B"] = "L-BFGS-B",
) -> tuple[DixonColesConfig, bool]:
    """``(config, prior_on)``: the production model for a training frame, solved with ``method``.

    Same rule as the dashboard (:func:`production_prior_on`): the Gaussian
    prior of :meth:`DixonColesConfig.with_gaussian_prior` while coverage is
    thin, the plain weighted MLE afterwards; ``xi`` per day. L-BFGS-B reaches
    the optimum of SLSQP about 40 times faster, which is what makes a
    walk-forward over many blocks affordable.
    """
    prior_on = production_prior_on(train, registry=registry, season=season)
    if prior_on:
        return DixonColesConfig.with_gaussian_prior(xi=xi, method=method), True
    return DixonColesConfig(xi=xi, method=method), False


def _has_prior(config: DixonColesConfig) -> bool:
    return bool(config.l2_penalty > 0 or config.l2_penalty_home > 0 or config.l2_penalty_rho > 0)


#: ``config`` of :func:`walk_forward` may be a fixed config or ``rule(train, season_label) -> config``.
ConfigRule = Callable[[pd.DataFrame, str], DixonColesConfig]


# =============================================================================
# Vectorised Dixon-Coles market probabilities
# =============================================================================
def _market_arrays(lam: np.ndarray, mu: np.ndarray, rho: float, max_goals: int) -> dict[str, np.ndarray]:
    """Every market probability for vectors of rates (identical to :func:`markets_from_matrix` of :func:`dixon_coles_pmf`)."""
    lam = np.asarray(lam, dtype=float)
    mu = np.asarray(mu, dtype=float)
    ks = np.arange(max_goals + 1, dtype=float)
    log_fact = gammaln(ks + 1.0)
    ph = np.exp(xlogy(ks[None, :], lam[:, None]) - lam[:, None] - log_fact[None, :])
    pa = np.exp(xlogy(ks[None, :], mu[:, None]) - mu[:, None] - log_fact[None, :])
    grid = ph[:, :, None] * pa[:, None, :]
    grid[:, 0, 0] *= 1.0 - lam * mu * rho
    grid[:, 0, 1] *= 1.0 + lam * rho
    grid[:, 1, 0] *= 1.0 + mu * rho
    grid[:, 1, 1] *= 1.0 - rho
    grid = np.clip(grid, 0.0, None)
    total = grid.sum(axis=(1, 2))
    grid /= np.where(total > 0, total, 1.0)[:, None, None]
    xs, ys = np.meshgrid(np.arange(max_goals + 1), np.arange(max_goals + 1), indexing="ij")
    goals = xs + ys

    def mass(mask: np.ndarray) -> np.ndarray:
        return (grid * mask[None, :, :]).sum(axis=(1, 2))

    out = {
        "p_1": mass(xs > ys),
        "p_X": mass(xs == ys),
        "p_2": mass(xs < ys),
        "p_GG": mass((xs > 0) & (ys > 0)),
        "p_NG": mass((xs == 0) | (ys == 0)),
    }
    for line, over, under in ((1.5, "p_O15", "p_U15"), (2.5, "p_O25", "p_U25"), (3.5, "p_O35", "p_U35")):
        out[over] = mass(goals > line)
        out[under] = mass(goals < line)
    return out


# =============================================================================
# Walk-forward
# =============================================================================
@dataclass(frozen=True)
class BacktestResult:
    """Outcome of :func:`walk_forward`.

    ``frame`` has one row per evaluated match, columns :data:`BACKTEST_COLUMNS`
    (pre-match probabilities, outcomes, naive base rates, training-set size and
    end, fit diagnostics). ``seasons`` are the evaluated seasons, ``block`` the
    block size, ``n_blocks`` the blocks fitted and predicted (``n_fits`` the
    model fits, one per fit group and block), ``n_skipped_blocks`` the blocks
    left out entirely, with ``skipped_matches`` counting the
    matches of every reason (``'storico insufficiente'`` when fewer than
    ``min_train_matches`` matches preceded the block, ``'stima non riuscita'``
    when the fit raised). ``seconds`` is the wall time, ``fit_seconds`` the
    time spent in the optimiser.
    """

    frame: pd.DataFrame
    seasons: tuple[str, ...]
    block: str
    n_blocks: int = 0
    n_skipped_blocks: int = 0
    skipped_matches: dict[str, int] = field(default_factory=dict)
    seconds: float = 0.0
    fit_seconds: float = 0.0
    n_not_converged: int = 0
    min_train_matches: int = DEFAULT_MIN_TRAIN_MATCHES
    n_fits: int = 0

    @property
    def empty(self) -> bool:
        return self.frame.empty

    @property
    def n_matches(self) -> int:
        return int(len(self.frame))

    @property
    def n_cold_start(self) -> int:
        return int(self.frame["cold_start"].sum()) if len(self.frame) else 0

    @property
    def matchdays(self) -> tuple[str, ...]:
        """Matchday keys (block of the football week, ISO date of its first day) in chronological order."""
        if self.frame.empty:
            return ()
        return tuple(sorted(self.frame["matchday"].astype(str).unique()))

    def summary(self) -> str:
        """One Italian line for the dashboard and the console."""
        if self.frame.empty:
            reasons = ", ".join(f"{n} partite: {r}" for r, n in self.skipped_matches.items())
            return "Backtest walk-forward: nessuna partita valutabile" + (f" ({reasons})." if reasons else ".")
        span = f"{self.frame['date'].min():%d/%m/%Y}-{self.frame['date'].max():%d/%m/%Y}"
        text = (
            f"Backtest walk-forward ({', '.join(self.seasons)}): {self.n_matches} partite valutate fuori campione "
            f"in {self.n_blocks} blocchi ({span}), ciascun blocco previsto da una stima con le sole partite precedenti; "
            f"{self.n_cold_start} partite con una squadra mai vista prima (forza a priori)"
        )
        if self.skipped_matches:
            text += "; non valutate: " + ", ".join(f"{n} ({r})" for r, n in self.skipped_matches.items())
        if self.n_not_converged:
            text += f"; {self.n_not_converged} stime senza convergenza"
        return text + f". Tempo: {self.seconds:.1f} s."


@contextmanager
def _quiet_model_logs(enabled: bool) -> Iterator[None]:
    """Silence the per-fit logs of :mod:`src.engine.model` emitted by **this thread** (a walk-forward fits dozens of models)."""
    if not enabled:
        yield
        return
    ident = threading.get_ident()

    def keep(record: logging.LogRecord) -> bool:
        return record.thread != ident or record.levelno >= logging.ERROR

    _model_log.addFilter(keep)
    try:
        yield
    finally:
        _model_log.removeFilter(keep)


def _block_starts(dates: pd.Series, block: str) -> pd.Series:
    """Start date of the block of every date (``week``: Tuesday-Monday; ``block``: football-week block; ``day``)."""
    days = pd.to_datetime(dates).dt.date
    starts: dict[date, date] = {}
    for d in days.unique():
        if block == "week":
            starts[d] = d - timedelta(days=(d.weekday() - 1) % 7)
        elif block == "block":
            starts[d] = week_block(d)[0]
        else:
            starts[d] = d
    return days.map(starts)


def _warm_start_for(previous: FittedParams, train: pd.DataFrame) -> FittedParams:
    """``previous`` reshaped to the team set of ``train`` (what :meth:`DixonColesEngine.fit` needs to accept it).

    Teams new to the training set start at ``alpha = 0`` and the mean defence
    of the previous fit; teams that left it are dropped. Only a starting point:
    the optimiser converges to the same optimum, just faster.
    """
    teams = sorted(set(train["home_team"].astype(str)) | set(train["away_team"].astype(str)))
    if set(previous.alpha) == set(teams):
        return previous
    mean_beta = float(np.mean(list(previous.beta.values()))) if previous.beta else 0.0
    alpha = {t: previous.alpha.get(t, 0.0) for t in teams}
    beta = {t: previous.beta.get(t, mean_beta) for t in teams}
    anchor = Girone.parse(str(train["girone"].iloc[-1]))
    return replace(previous, alpha=alpha, beta=beta, teams_by_girone={anchor: tuple(teams)}, historical_teams=())


def _base_rates(train: pd.DataFrame, first: pd.Timestamp, codes: Sequence[str]) -> dict[str, np.ndarray]:
    """Naive base rates ``[1, X, 2, GG, O15, O25, O35]`` per competition from the year before ``first``.

    Each competition's rates are shrunk towards the rates of every competition
    with :data:`BASE_RATE_PSEUDO_COUNT` pseudo-matches; a competition without
    recent matches gets the pooled rates.
    """
    recent = train.loc[train["date"] >= first - pd.Timedelta(days=BASE_RATE_WINDOW_DAYS)]
    if recent.empty:
        recent = train
    hg = recent["home_goals"].to_numpy(dtype=float)
    ag = recent["away_goals"].to_numpy(dtype=float)
    indicators = np.column_stack([hg > ag, hg == ag, hg < ag, (hg > 0) & (ag > 0), hg + ag > 1.5, hg + ag > 2.5, hg + ag > 3.5]).astype(float)
    pooled = indicators.mean(axis=0)
    girone = recent["girone"].astype(str).to_numpy()
    out: dict[str, np.ndarray] = {}
    for code in codes:
        mask = girone == code
        n = int(mask.sum())
        out[code] = (indicators[mask].sum(axis=0) + BASE_RATE_PSEUDO_COUNT * pooled) / (n + BASE_RATE_PSEUDO_COUNT) if n else pooled
    return out


def _predict_block(rows: pd.DataFrame, params: FittedParams, max_goals: int) -> tuple[dict[str, np.ndarray], np.ndarray, np.ndarray, np.ndarray]:
    """``(probabilities, lam, mu, cold_start)`` of the rows of one block.

    A team unknown to the fit gets the prior strengths (``alpha = 0``, the mean
    defence of its competition's fitted teams); a competition without a home
    advantage estimate gets the mean of the fitted ones. Both are flagged.
    """
    group_beta: dict[str, float] = {}
    for g, members in params.teams_by_girone.items():
        values = [params.beta[t] for t in members if t in params.beta]
        if values:
            group_beta[g.value] = float(np.mean(values))
    all_beta = float(np.mean(list(params.beta.values()))) if params.beta else 0.0
    gamma = {g.value: v for g, v in params.gamma.items()}
    mean_gamma = float(np.mean(list(gamma.values()))) if gamma else 0.0
    home = rows["home_team"].astype(str).to_numpy()
    away = rows["away_team"].astype(str).to_numpy()
    codes = rows["girone"].astype(str).to_numpy()
    known_h = np.array([t in params.alpha for t in home])
    known_a = np.array([t in params.alpha for t in away])
    known_g = np.array([c in gamma for c in codes])
    alpha_h = np.array([params.alpha.get(t, 0.0) for t in home])
    alpha_a = np.array([params.alpha.get(t, 0.0) for t in away])
    beta_h = np.array([params.beta.get(t, group_beta.get(c, all_beta)) for t, c in zip(home, codes)])
    beta_a = np.array([params.beta.get(t, group_beta.get(c, all_beta)) for t, c in zip(away, codes)])
    gam = np.array([gamma.get(c, mean_gamma) for c in codes])
    lam = np.exp(np.clip(alpha_h - beta_a + gam, -50.0, 50.0))
    mu = np.exp(np.clip(alpha_a - beta_h, -50.0, 50.0))
    cold = ~(known_h & known_a & known_g)
    return _market_arrays(lam, mu, params.rho, max_goals), lam, mu, cold


def _outcomes(rows: pd.DataFrame) -> dict[str, np.ndarray]:
    hg = rows["home_goals"].to_numpy(dtype=np.int64)
    ag = rows["away_goals"].to_numpy(dtype=np.int64)
    total = hg + ag
    return {
        "res_1X2": np.where(hg > ag, "1", np.where(hg == ag, "X", "2")),
        "gg": ((hg > 0) & (ag > 0)).astype(np.int64),
        "o15": (total > 1.5).astype(np.int64),
        "o25": (total > 2.5).astype(np.int64),
        "o35": (total > 3.5).astype(np.int64),
    }


def _empty_backtest_frame() -> pd.DataFrame:
    dtypes = _backtest_dtypes()
    return pd.DataFrame({c: pd.Series(dtype=dtypes[c]) for c in BACKTEST_COLUMNS})


def _backtest_dtypes() -> dict[str, str]:
    dtypes: dict[str, str] = {c: "float64" for c in (*PROB_COLUMNS, *BASE_COLUMNS, "lam", "mu", "rho")}
    for c in ("match_id", "season", "matchday", "block", "girone", "family", "section", "home_team", "away_team", "res_1X2", "fit_method"):
        dtypes[c] = "string"
    for c in ("round", "home_goals", "away_goals", "gg", "o15", "o25", "o35", "train_size"):
        dtypes[c] = "int64"
    for c in ("fit_converged", "prior", "cold_start"):
        dtypes[c] = "bool"
    dtypes["date"] = "datetime64[ns]"
    dtypes["train_end"] = "datetime64[ns]"
    return dtypes


def _block_part(
    rows: pd.DataFrame,
    train: pd.DataFrame,
    params: FittedParams,
    config: DixonColesConfig,
    prior_on: bool,
    sections: Mapping[str, str],
) -> dict[str, Any]:
    """The backtest columns of the ``rows`` of one block and fit group, predicted by ``params`` (fitted on ``train``)."""
    probs, lam, mu, cold = _predict_block(rows, params, config.max_goals)
    codes = rows["girone"].astype(str)
    base = _base_rates(train, rows["date"].min(), sorted(set(codes)))
    base_matrix = np.vstack([base[c] for c in codes])
    n = len(rows)
    return {
        "match_id": rows["match_id"].to_numpy(),
        "date": rows["date"].to_numpy(),
        "season": rows["season"].to_numpy(),
        "matchday": rows["matchday"].to_numpy(),
        "block": rows["block"].to_numpy(),
        "girone": rows["girone"].to_numpy(),
        "family": [Girone.parse(c).family for c in codes],
        "section": [sections[c] for c in codes],
        "round": rows["round"].to_numpy(),
        "home_team": rows["home_team"].to_numpy(),
        "away_team": rows["away_team"].to_numpy(),
        "home_goals": rows["home_goals"].to_numpy(),
        "away_goals": rows["away_goals"].to_numpy(),
        "lam": lam,
        "mu": mu,
        "rho": np.full(n, params.rho),
        **probs,
        **_outcomes(rows),
        "base_1": base_matrix[:, 0],
        "base_X": base_matrix[:, 1],
        "base_2": base_matrix[:, 2],
        "base_GG": base_matrix[:, 3],
        "base_NG": 1.0 - base_matrix[:, 3],
        "base_O15": base_matrix[:, 4],
        "base_U15": 1.0 - base_matrix[:, 4],
        "base_O25": base_matrix[:, 5],
        "base_U25": 1.0 - base_matrix[:, 5],
        "base_O35": base_matrix[:, 6],
        "base_U35": 1.0 - base_matrix[:, 6],
        "train_size": np.full(n, len(train)),
        "train_end": np.full(n, train["date"].max().to_datetime64()),
        "fit_converged": np.full(n, bool(params.converged)),
        "fit_method": np.full(n, params.method, dtype=object),
        "prior": np.full(n, bool(prior_on)),
        "cold_start": cold,
    }


def _drop_unfittable(df: pd.DataFrame) -> pd.DataFrame:
    """Rows the engine would refuse (negative goals, a team against itself) are dropped with a warning."""
    bad = (df["home_goals"] < 0) | (df["away_goals"] < 0) | (df["home_team"] == df["away_team"]).fillna(False)
    if bad.any():
        log.warning("backtest: invalid matches dropped", extra={"n_dropped": int(bad.sum()), "match_ids": df.loc[bad, "match_id"].astype(str).tolist()[:10]})
        df = df.loc[~bad.to_numpy(dtype=bool)].reset_index(drop=True)
    return df


def walk_forward(
    matches: pd.DataFrame,
    *,
    registry: TeamRegistry | None = None,
    seasons: Sequence[str] | None = None,
    config: DixonColesConfig | ConfigRule | None = None,
    block: BlockName = DEFAULT_BLOCK,
    min_train_matches: int = DEFAULT_MIN_TRAIN_MATCHES,
    progress: ProgressCallback | None = None,
    quiet: bool = True,
) -> BacktestResult:
    """Out-of-sample probabilities of every match of ``seasons``, block by block (see the module docstring).

    ``matches`` is the canonical frame (history included: it is training data
    for the first blocks). ``seasons`` defaults to the registry's season, else
    to the season of the latest match. Each block of ``block`` size is
    predicted by a model fitted on the matches dated strictly before its first
    match, one model per fit group (:func:`fit_group_of_girone`, warm-started
    from the same group's previous fit); the matches of a group with fewer than
    ``min_train_matches`` preceding matches are skipped (counted in
    :attr:`BacktestResult.skipped_matches`). ``config``:
    ``None`` applies the production rule (:func:`production_config`: xi 0.0018,
    Gaussian prior while the current season is thin, L-BFGS-B), a
    :class:`DixonColesConfig` is used as is for every block, a callable
    ``rule(train, season_label)`` is asked for each block. ``registry`` defines
    the perimeter of the blocks of its own season (current vs historical teams,
    constraint groups, coverage of the prior rule); blocks of other seasons are
    fitted without it. ``progress(fraction, message)`` is called after every
    block. ``quiet`` silences the per-fit logs of the engine for this thread
    (one summary line is logged instead). ``ValueError`` for an unknown block,
    a negative ``min_train_matches`` or an invalid season label.
    """
    if block not in BLOCKS:
        raise ValueError(f"block must be one of {BLOCKS}, got {block!r}")
    if int(min_train_matches) < 0:
        raise ValueError(f"min_train_matches must be >= 0, got {min_train_matches}")
    started = time.perf_counter()
    report = progress if progress is not None else (lambda _fraction, _message: None)
    df = validate_matches_df(matches)
    df = _drop_unfittable(df)
    if seasons is None:
        if registry is not None:
            wanted = (_normalise_season(registry.season_label),)
        elif len(df):
            wanted = (str(_season_labels(df["date"].iloc[[-1]]).iloc[0]),)
        else:
            wanted = ()
    else:
        wanted = tuple(dict.fromkeys(_normalise_season(s) for s in seasons))
    if df.empty or not wanted:
        log.warning("walk-forward: no matches to evaluate", extra={"n_matches": int(len(df)), "seasons": list(wanted)})
        return BacktestResult(frame=_empty_backtest_frame(), seasons=wanted, block=block, min_train_matches=int(min_train_matches))
    df = df.assign(season=_season_labels(df["date"]), _group=df["girone"].astype(str).map(fit_group_of_girone))
    evaluate = df.loc[df["season"].isin(wanted).to_numpy(dtype=bool)].copy()
    evaluate["block"] = _block_starts(evaluate["date"], block).map(lambda d: d.isoformat())
    evaluate["matchday"] = _block_starts(evaluate["date"], "block").map(lambda d: d.isoformat())
    block_keys = sorted(evaluate["block"].unique())
    sections = _section_map(sorted(set(df["girone"].astype(str))))
    dates = df["date"].to_numpy()
    registry_season = _normalise_season(registry.season_label) if registry is not None else None

    parts: list[dict[str, Any]] = []
    skipped: dict[str, int] = {}
    n_blocks = 0
    n_skipped_blocks = 0
    n_fits = 0
    fit_seconds = 0.0
    n_not_converged = 0
    previous: dict[str, FittedParams] = {}
    groups_present = list(dict.fromkeys(df["_group"].astype(str)))
    with _quiet_model_logs(quiet):
        for position, key in enumerate(block_keys, start=1):
            block_rows = evaluate.loc[evaluate["block"] == key]
            first = block_rows["date"].min()
            cut = int(np.searchsorted(dates, pd.Timestamp(first).to_datetime64(), side="left"))
            train_all = df.iloc[:cut]
            fitted_any = False
            for group in groups_present:
                rows = block_rows.loc[block_rows["_group"] == group]
                if rows.empty:
                    continue
                train = train_all.loc[train_all["_group"] == group]
                if train.empty or len(train) < int(min_train_matches):
                    reason = "storico insufficiente"
                    skipped[reason] = skipped.get(reason, 0) + int(len(rows))
                    continue
                block_season = str(rows["season"].iloc[0])
                fit_registry = registry if registry is not None and registry_season == block_season else None
                if fit_registry is not None and not any(fit_registry.teams_by_girone.get(Girone.parse(c)) for c in set(train["girone"].astype(str))):
                    fit_registry = None  # the registry has no team in this group's competitions: every team would be historical
                if config is None:
                    cfg, prior_on = production_config(train, registry=fit_registry, season=block_season)
                elif isinstance(config, DixonColesConfig):
                    cfg, prior_on = config, _has_prior(config)
                else:
                    cfg = config(train, block_season)
                    prior_on = _has_prior(cfg)
                engine = DixonColesEngine(cfg)
                warm = _warm_start_for(previous[group], train) if group in previous else None
                try:
                    params = engine.fit(train.drop(columns=["season", "_group"]), registry=fit_registry, warm_start=warm)
                except (ValueError, KeyError, FloatingPointError, np.linalg.LinAlgError) as exc:
                    reason = "stima non riuscita"
                    skipped[reason] = skipped.get(reason, 0) + int(len(rows))
                    log.warning("walk-forward: fit failed, block skipped", extra={"block": key, "fit_group": group, "n_train": int(len(train)), "error": str(exc)})
                    continue
                previous[group] = params
                fitted_any = True
                fit_seconds += params.fit_seconds
                n_fits += 1
                n_not_converged += int(not params.converged)
                parts.append(_block_part(rows, train, params, cfg, prior_on, sections))
            if fitted_any:
                n_blocks += 1
            else:
                n_skipped_blocks += 1
            report(position / len(block_keys), f"Backtest: blocco {position}/{len(block_keys)} ({key}), {len(block_rows)} partite")

    if parts:
        frame = pd.concat([pd.DataFrame(p) for p in parts], ignore_index=True)
        dtypes = _backtest_dtypes()
        for column in BACKTEST_COLUMNS:
            frame[column] = frame[column].astype(dtypes[column])
        frame = frame[list(BACKTEST_COLUMNS)]
        order = frame["girone"].astype(str).map(_girone_order)
        frame = frame.assign(_order=order).sort_values(["date", "_order", "match_id"], kind="stable").drop(columns="_order").reset_index(drop=True)
    else:
        frame = _empty_backtest_frame()
    seconds = time.perf_counter() - started
    result = BacktestResult(
        frame=frame,
        seasons=wanted,
        block=block,
        n_blocks=n_blocks,
        n_skipped_blocks=n_skipped_blocks,
        n_fits=n_fits,
        skipped_matches=skipped,
        seconds=float(seconds),
        fit_seconds=float(fit_seconds),
        n_not_converged=n_not_converged,
        min_train_matches=int(min_train_matches),
    )
    log.info(
        "walk-forward backtest complete",
        extra={
            "seasons": list(wanted),
            "block": block,
            "n_matches": result.n_matches,
            "n_blocks": result.n_blocks,
            "n_skipped_blocks": n_skipped_blocks,
            "skipped_matches": skipped,
            "n_cold_start": result.n_cold_start,
            "n_not_converged": n_not_converged,
            "seconds": round(seconds, 2),
            "fit_seconds": round(fit_seconds, 2),
        },
    )
    return result


# =============================================================================
# Scores
# =============================================================================
def _by_columns(by: str | Sequence[str] | None) -> list[str]:
    if by is None:
        return []
    if isinstance(by, str):
        return [by]
    return list(by)


def _order_frame(frame: pd.DataFrame, columns: Sequence[str]) -> pd.DataFrame:
    """Sort a result frame by section/family/competition in configuration order, then market order."""
    if frame.empty:
        return frame.reset_index(drop=True)
    keys: dict[str, pd.Series] = {}
    section_rank = {s: i for i, s in enumerate(dict.fromkeys(section_of_girone(g) for g in Girone))}
    family_rank = {f: i for i, f in enumerate(dict.fromkeys(g.family for g in Girone))}
    market_rank = {m.value: i for i, m in enumerate(MARKET_SPECS)}
    sort_cols: list[str] = []
    for column in columns:
        if column not in frame.columns:
            continue
        name = f"_k_{column}"
        values = frame[column].astype(str)
        if column == "girone":
            keys[name] = values.map(_girone_order)
        elif column in ("section", "scope"):
            keys[name] = values.map(lambda v: -1 if v == ALL_SECTIONS else section_rank.get(v, len(section_rank)))
        elif column == "family":
            keys[name] = values.map(lambda v: family_rank.get(v, len(family_rank)))
        elif column == "market":
            keys[name] = values.map(lambda v: market_rank.get(v, len(market_rank)))
        elif column == "list":
            list_rank = {name: i for i, (name, _m, _s) in enumerate(_PICK_LISTS)}
            keys[name] = values.map(lambda v: list_rank.get(v, len(list_rank)))
        else:
            keys[name] = values
        sort_cols.extend([name, column] if column not in ("market", "list") else [name])
    ordered = frame.assign(**keys).sort_values(sort_cols, kind="stable")
    return ordered.drop(columns=list(keys)).reset_index(drop=True)


def _groups(frame: pd.DataFrame, by: Sequence[str]) -> Iterator[tuple[dict[str, Any], pd.DataFrame]]:
    if not by:
        yield {}, frame
        return
    for key, group in frame.groupby(list(by), sort=False, observed=True, dropna=False):
        values = key if isinstance(key, tuple) else (key,)
        yield dict(zip(by, values)), group


def metrics(
    frame: pd.DataFrame,
    *,
    by: str | Sequence[str] | None = ("section", "girone"),
    markets: Iterable[Market | str] = DEFAULT_MARKETS,
    prefix: str = "p_",
) -> pd.DataFrame:
    """Probabilistic scores of a backtest frame per group and market.

    One row per group of ``by`` (``()``/``None``: the whole frame) and market
    with: ``n``; ``brier`` (multi-class sum over 1/X/2, range 0-2, for 1X2;
    binary on the first selection, range 0-1, for GG/NG and Over/Under);
    ``log_loss`` (mean ``-ln p(outcome)``); ``accuracy`` (share of matches whose
    most probable selection happened); ``selection`` (the reference selection,
    the first of the market: ``1``, ``GG``, ``Over x``) with ``mean_pred`` (its
    mean predicted probability) and ``observed`` (its frequency);
    ``log_loss_naive`` and ``skill_naive = 1 - log_loss / log_loss_naive``
    against the base rates (``NaN`` without ``base_*`` columns). ``prefix``
    selects the probability columns scored (``'p_'`` model, ``'base_'`` naive,
    or the columns of a recalibrated copy).
    """
    group_by = _by_columns(by)
    rows: list[dict[str, Any]] = []
    has_base = all(c in frame.columns for c in BASE_COLUMNS)
    for market in markets:
        spec = market_spec(market)
        if frame.empty:
            continue
        probs = spec.probabilities(frame, prefix)
        y = spec.outcome_index(frame)
        valid = (y >= 0) & np.isfinite(probs).all(axis=1)
        per_row = pd.DataFrame(
            {
                "_brier": _brier_rows(probs, y, spec.is_binary),
                "_ll": _log_loss_rows(probs, y),
                "_acc": _accuracy_rows(probs, y),
                "_pred": probs[:, 0],
                "_obs": (y == 0).astype(float),
                "_ll_naive": _log_loss_rows(spec.probabilities(frame, "base_"), y) if has_base else np.nan,
            },
            index=frame.index,
        )
        scored = pd.concat([frame[group_by], per_row], axis=1).loc[valid]
        for key, group in _groups(scored, group_by):
            ll = float(group["_ll"].mean())
            ll_naive = float(group["_ll_naive"].mean()) if has_base else float("nan")
            rows.append(
                {
                    **key,
                    "market": spec.key,
                    "selection": spec.selections[0],
                    "n": int(len(group)),
                    "brier": float(group["_brier"].mean()),
                    "log_loss": ll,
                    "accuracy": float(group["_acc"].mean()),
                    "mean_pred": float(group["_pred"].mean()),
                    "observed": float(group["_obs"].mean()),
                    "log_loss_naive": ll_naive,
                    "skill_naive": 1.0 - ll / ll_naive if ll_naive and np.isfinite(ll_naive) and ll_naive > 0 else float("nan"),
                }
            )
    columns = [*group_by, "market", "selection", "n", "brier", "log_loss", "accuracy", "mean_pred", "observed", "log_loss_naive", "skill_naive"]
    out = pd.DataFrame(rows, columns=columns)
    return _order_frame(out, [*group_by, "market"])


def selection_calibration(
    frame: pd.DataFrame,
    *,
    by: str | Sequence[str] | None = ("section",),
    markets: Iterable[Market | str] = DEFAULT_MARKETS,
    prefix: str = "p_",
) -> pd.DataFrame:
    """Mean predicted probability vs observed frequency of **every** selection, per group.

    Columns ``[*by, market, selection, n, hits, mean_pred, observed,
    wilson_low, wilson_high, gap]`` with ``gap = observed - mean_pred``
    (positive: the selection happened more often than the model said).
    """
    group_by = _by_columns(by)
    rows: list[dict[str, Any]] = []
    for market in markets:
        spec = market_spec(market)
        if frame.empty:
            continue
        probs = spec.probabilities(frame, prefix)
        y = spec.outcome_index(frame)
        valid = (y >= 0) & np.isfinite(probs).all(axis=1)
        for index, selection in enumerate(spec.selections):
            per_row = pd.DataFrame({"_p": probs[:, index], "_hit": (y == index).astype(float)}, index=frame.index)
            scored = pd.concat([frame[group_by], per_row], axis=1).loc[valid]
            for key, group in _groups(scored, group_by):
                n = int(len(group))
                hits = int(group["_hit"].sum())
                low, high = wilson_interval(hits, n)
                mean_pred = float(group["_p"].mean())
                rows.append(
                    {
                        **key,
                        "market": spec.key,
                        "selection": selection,
                        "n": n,
                        "hits": hits,
                        "mean_pred": mean_pred,
                        "observed": hits / n if n else float("nan"),
                        "wilson_low": low,
                        "wilson_high": high,
                        "gap": (hits / n - mean_pred) if n else float("nan"),
                    }
                )
    columns = [*group_by, "market", "selection", "n", "hits", "mean_pred", "observed", "wilson_low", "wilson_high", "gap"]
    out = pd.DataFrame(rows, columns=columns)
    return _order_frame(out, [*group_by, "market"])


def reliability(
    frame: pd.DataFrame,
    market: Market | str,
    selection: str,
    bins: int = 10,
    *,
    strategy: Literal["uniform", "quantile"] = "uniform",
    prefix: str = "p_",
) -> pd.DataFrame:
    """Reliability table of one selection: predicted-probability bins against observed frequencies.

    Columns ``bin`` (1-based), ``bin_low``/``bin_high`` (edges), ``mean_pred``,
    ``observed``, ``n``, ``hits``, ``wilson_low``/``wilson_high`` (95% Wilson
    interval of ``observed``). ``strategy='uniform'`` uses ``bins`` equal-width
    bins on ``[0, 1]``, ``'quantile'`` equal-count bins; empty bins are left
    out. A well-calibrated model has ``mean_pred`` inside most intervals.
    """
    spec = market_spec(market)
    if selection not in spec.selections:
        raise ValueError(f"{selection!r} is not a selection of {spec.key} (expected one of {spec.selections})")
    if int(bins) < 1:
        raise ValueError(f"bins must be >= 1, got {bins}")
    if strategy not in ("uniform", "quantile"):
        raise ValueError(f"strategy must be 'uniform' or 'quantile', got {strategy!r}")
    columns = ["bin", "bin_low", "bin_high", "mean_pred", "observed", "n", "hits", "wilson_low", "wilson_high"]
    if frame.empty:
        return pd.DataFrame(columns=columns)
    index = spec.selections.index(selection)
    p = spec.probabilities(frame, prefix)[:, index]
    y = spec.outcome_index(frame)
    keep = (y >= 0) & np.isfinite(p)
    p, hit = p[keep], (y[keep] == index).astype(float)
    if p.size == 0:
        return pd.DataFrame(columns=columns)
    if strategy == "uniform":
        edges = np.linspace(0.0, 1.0, int(bins) + 1)
    else:
        edges = np.unique(np.quantile(p, np.linspace(0.0, 1.0, int(bins) + 1)))
        if edges.size < 2:
            edges = np.array([p.min(), p.max() + 1e-12])
    which = np.clip(np.searchsorted(edges, p, side="right") - 1, 0, edges.size - 2)
    rows: list[dict[str, Any]] = []
    for b in range(edges.size - 1):
        mask = which == b
        n = int(mask.sum())
        if not n:
            continue
        hits = int(hit[mask].sum())
        low, high = wilson_interval(hits, n)
        rows.append(
            {
                "bin": b + 1,
                "bin_low": float(edges[b]),
                "bin_high": float(edges[b + 1]),
                "mean_pred": float(p[mask].mean()),
                "observed": hits / n,
                "n": n,
                "hits": hits,
                "wilson_low": low,
                "wilson_high": high,
            }
        )
    return pd.DataFrame(rows, columns=columns)


# =============================================================================
# Bookmaker odds
# =============================================================================
_ODDS_KEYS: Final[tuple[str, ...]] = ("match_id", "bookmaker", "market", "selection", "odds")


def _odds_long(odds_history: pd.DataFrame | Iterable[Any] | None) -> pd.DataFrame:
    """Validated long frame ``match_id, bookmaker, market, selection, odds`` of complete books only."""
    empty = pd.DataFrame({c: pd.Series(dtype="float64" if c == "odds" else "string") for c in _ODDS_KEYS})
    if odds_history is None:
        return empty
    if isinstance(odds_history, pd.DataFrame):
        raw = odds_history.copy()
        if "bookmaker" not in raw.columns:
            raw["bookmaker"] = "book"
        missing = [c for c in _ODDS_KEYS if c not in raw.columns]
        if missing:
            raise ValueError(f"odds history is missing columns {missing}")
        raw = raw[list(_ODDS_KEYS)]
    else:
        records = [
            {"match_id": q.match_id, "bookmaker": q.bookmaker, "market": market_of(q.market).value, "selection": s, "odds": float(o)}
            for q in odds_history
            for s, o in q.odds.items()
        ]
        if not records:
            return empty
        raw = pd.DataFrame(records, columns=list(_ODDS_KEYS))
    if raw.empty:
        return empty
    for column in ("match_id", "bookmaker", "market", "selection"):
        raw[column] = raw[column].astype("string").str.strip()
    raw["odds"] = pd.to_numeric(raw["odds"], errors="coerce").astype("float64")
    valid_pairs = {(m.value, s) for m, sels in SELECTIONS.items() for s in sels}
    pairs = pd.Series(list(zip(raw["market"].astype(str), raw["selection"].astype(str))), index=raw.index)
    raw = raw.loc[(pairs.isin(valid_pairs) & raw["odds"].gt(1.0).fillna(False)).to_numpy(dtype=bool)]
    raw = raw.drop_duplicates(subset=["match_id", "bookmaker", "market", "selection"], keep="last")
    sizes = raw.groupby(["match_id", "bookmaker", "market"], sort=False)["selection"].transform("size")
    expected = raw["market"].astype(str).map({m.value: len(s) for m, s in SELECTIONS.items()})
    return raw.loc[(sizes == expected).to_numpy(dtype=bool)].reset_index(drop=True)


def odds_table(odds_history: pd.DataFrame | Iterable[Any] | None, *, method: str = "proportional") -> pd.DataFrame:
    """One row per ``match_id`` with the best price and the de-vigged probability of every selection.

    ``odds_history`` is the long frame of :func:`src.ingestion.harvester.load_odds_history`
    (or any frame with ``match_id, market, selection, odds`` and optionally
    ``bookmaker``) or an iterable of :class:`~src.common.schemas.OddsQuote`.
    Columns: ``odds_<suffix>`` -- the best (highest) price of the selection
    across bookmakers, what a bettor would take; ``book_<suffix>`` -- the
    bookmaker probability with the margin removed (``method`` of
    :func:`src.engine.value.devig`) from the complete book with the **lowest**
    over-round; ``overround_<market>`` (``1X2``, ``GGNG``, ``OU15``, ``OU25``,
    ``OU35``) of that book. Missing markets are ``NaN``; incomplete books and
    odds ``<= 1`` are ignored.
    """
    long = _odds_long(odds_history)
    columns = ["match_id", *(f"odds_{s}" for s in SELECTION_SUFFIX.values()), *(f"book_{s}" for s in SELECTION_SUFFIX.values()), *(f"overround_{m}" for m in _MARKET_SUFFIX.values())]
    if long.empty:
        return pd.DataFrame({c: pd.Series(dtype="string" if c == "match_id" else "float64") for c in columns})
    long = long.assign(suffix=long["selection"].astype(str).map(SELECTION_SUFFIX), inv=1.0 / long["odds"])
    best = long.groupby(["match_id", "suffix"], sort=False)["odds"].max().unstack("suffix")
    best.columns = [f"odds_{c}" for c in best.columns]
    book_keys = ["match_id", "bookmaker", "market"]
    long = long.assign(_over=long.groupby(book_keys, sort=False)["inv"].transform("sum") - 1.0)
    chosen = long.sort_values(["match_id", "market", "_over", "bookmaker"], kind="stable").drop_duplicates(["match_id", "market"], keep="first")[book_keys]
    sharp = long.merge(chosen, on=book_keys, how="inner")
    if method == "proportional":
        sharp = sharp.assign(p=sharp["inv"] / sharp.groupby(book_keys, sort=False)["inv"].transform("sum"))
    else:
        parts = []
        for _key, group in sharp.groupby(book_keys, sort=False):
            fair = devig(dict(zip(group["selection"].astype(str), group["odds"].astype(float))), method)  # type: ignore[arg-type]
            parts.append(group.assign(p=group["selection"].astype(str).map(fair).astype(float)))
        sharp = pd.concat(parts, ignore_index=True)
    probs = sharp.pivot_table(index="match_id", columns="suffix", values="p", aggfunc="first")
    probs.columns = [f"book_{c}" for c in probs.columns]
    over = sharp.drop_duplicates(["match_id", "market"]).assign(m=lambda d: d["market"].astype(str).map(lambda v: _MARKET_SUFFIX[Market(v)]))
    over = over.pivot_table(index="match_id", columns="m", values="_over", aggfunc="first")
    over.columns = [f"overround_{c}" for c in over.columns]
    table = best.join(probs, how="outer").join(over, how="outer").reset_index().rename(columns={"index": "match_id"})
    for column in columns:
        if column not in table.columns:
            table[column] = np.nan
    table["match_id"] = table["match_id"].astype("string")
    return table[columns].reset_index(drop=True)


def market_comparison(
    frame: pd.DataFrame,
    odds_history: pd.DataFrame | Iterable[Any] | None,
    *,
    by: str | Sequence[str] | None = ("section", "girone"),
    markets: Iterable[Market | str] = DEFAULT_MARKETS,
    method: str = "proportional",
    prefix: str = "p_",
) -> pd.DataFrame:
    """Model vs bookmaker on the matches that have pre-match odds, per group and market.

    For every market only the matches with a complete book are scored; both
    the model probabilities (``prefix``) and the de-vigged bookmaker
    probabilities (:func:`odds_table`) are scored on exactly those matches.
    Columns ``[*by, market, n, log_loss_model, log_loss_market, log_loss_naive,
    brier_model, brier_market, accuracy_model, accuracy_market, skill,
    skill_naive, mean_overround]`` with ``skill = 1 - log_loss_model /
    log_loss_market`` (positive: the model beat the market's own
    probabilities, rare and fragile on small samples) and ``skill_naive``
    against the base rates. An empty frame when no match has odds.
    """
    group_by = _by_columns(by)
    columns = [*group_by, "market", "n", "log_loss_model", "log_loss_market", "log_loss_naive", "brier_model", "brier_market", "accuracy_model", "accuracy_market", "skill", "skill_naive", "mean_overround"]
    table = odds_table(odds_history, method=method)
    if frame.empty or table.empty:
        return pd.DataFrame(columns=columns)
    merged = frame.merge(table, on="match_id", how="inner")
    has_base = all(c in merged.columns for c in BASE_COLUMNS)
    rows: list[dict[str, Any]] = []
    for market in markets:
        spec = market_spec(market)
        book_cols = list(spec.columns("book_"))
        quoted = merged.loc[merged[book_cols].notna().all(axis=1).to_numpy(dtype=bool)]
        if quoted.empty:
            continue
        y = spec.outcome_index(quoted)
        model_p = spec.probabilities(quoted, prefix)
        book_p = spec.probabilities(quoted, "book_")
        per_row = pd.DataFrame(
            {
                "_ll_model": _log_loss_rows(model_p, y),
                "_ll_market": _log_loss_rows(book_p, y),
                "_ll_naive": _log_loss_rows(spec.probabilities(quoted, "base_"), y) if has_base else np.nan,
                "_br_model": _brier_rows(model_p, y, spec.is_binary),
                "_br_market": _brier_rows(book_p, y, spec.is_binary),
                "_acc_model": _accuracy_rows(model_p, y),
                "_acc_market": _accuracy_rows(book_p, y),
                "_over": quoted[f"overround_{_MARKET_SUFFIX[spec.market]}"].to_numpy(dtype=float),
            },
            index=quoted.index,
        )
        scored = pd.concat([quoted[group_by], per_row], axis=1).loc[(y >= 0) & np.isfinite(model_p).all(axis=1)]
        for key, group in _groups(scored, group_by):
            ll_model = float(group["_ll_model"].mean())
            ll_market = float(group["_ll_market"].mean())
            ll_naive = float(group["_ll_naive"].mean()) if has_base else float("nan")
            rows.append(
                {
                    **key,
                    "market": spec.key,
                    "n": int(len(group)),
                    "log_loss_model": ll_model,
                    "log_loss_market": ll_market,
                    "log_loss_naive": ll_naive,
                    "brier_model": float(group["_br_model"].mean()),
                    "brier_market": float(group["_br_market"].mean()),
                    "accuracy_model": float(group["_acc_model"].mean()),
                    "accuracy_market": float(group["_acc_market"].mean()),
                    "skill": 1.0 - ll_model / ll_market if ll_market > 0 else float("nan"),
                    "skill_naive": 1.0 - ll_model / ll_naive if np.isfinite(ll_naive) and ll_naive > 0 else float("nan"),
                    "mean_overround": float(group["_over"].mean()),
                }
            )
    return _order_frame(pd.DataFrame(rows, columns=columns), [*group_by, "market"])


# =============================================================================
# Track record: picks
# =============================================================================
def _scoped(frame: pd.DataFrame, by_section: bool) -> Iterator[tuple[str, pd.DataFrame]]:
    yield ALL_SECTIONS, frame
    if by_section:
        for section in dict.fromkeys(frame["section"].astype(str)):
            yield section, frame.loc[frame["section"].astype(str) == section]


def _predictions(rows: pd.DataFrame, prefix: str) -> list[MatchPrediction]:
    """:class:`MatchPrediction` objects of backtest rows (probabilities from ``prefix`` columns) for :func:`rank_picks`."""
    out: list[MatchPrediction] = []
    for row in rows.itertuples(index=False):
        record = row._asdict()
        fixture = Fixture(
            match_id=str(record["match_id"]),
            date=pd.Timestamp(record["date"]).date(),
            round=int(record["round"]),
            girone=str(record["girone"]),
            home_team=str(record["home_team"]),
            away_team=str(record["away_team"]),
        )
        rho = float(record.get("rho", 0.0) or 0.0)
        matrix = dixon_coles_pmf(float(record["lam"]), float(record["mu"]), rho if np.isfinite(rho) else 0.0)
        probabilities = {
            spec.market: {sel: float(record[f"{prefix}{suffix}"]) for sel, suffix in zip(spec.selections, spec.suffixes)}
            for spec in MARKET_SPECS.values()
        }
        fair = {m: {s: (1.0 / p if p > 0 else float("inf")) for s, p in sels.items()} for m, sels in probabilities.items()}
        out.append(MatchPrediction(fixture=fixture, home_rate=float(record["lam"]), away_rate=float(record["mu"]), rho=rho, matrix=matrix, probabilities=probabilities, fair_odds=fair))
    return out


def _won(row: Mapping[str, Any], market: Market, selection: str) -> bool:
    spec = MARKET_SPECS[market]
    if spec.is_binary:
        first = bool(int(row[spec.outcome]))
        return first if selection == spec.selections[0] else not first
    return str(row[spec.outcome]) == selection


def _best_odds_lookup(odds_history: pd.DataFrame | Iterable[Any] | None) -> dict[tuple[str, str], float]:
    table = odds_table(odds_history)
    lookup: dict[tuple[str, str], float] = {}
    if table.empty:
        return lookup
    for selection, suffix in SELECTION_SUFFIX.items():
        column = table[f"odds_{suffix}"]
        for match_id, value in zip(table["match_id"].astype(str), column.to_numpy(dtype=float)):
            if np.isfinite(value):
                lookup[(match_id, selection)] = float(value)
    return lookup


@dataclass(frozen=True)
class PicksTrackRecord:
    """What :func:`picks_track_record` found.

    ``picks``: one row per pick the model would have listed (``matchday``,
    ``scope`` -- :data:`ALL_SECTIONS` or a section --, ``list`` -- ``'X'`` or
    ``'GG'`` --, ``rank``, match, ``selection``, ``p_model``, ``won``, ``odds``
    -- best pre-match price, ``NaN`` when unknown -- and ``profit`` at stake 1).
    ``by_matchday``: per scope, list and matchday the picks, hits, expected
    hits (sum of the probabilities) and whether every pick won.
    ``summary``: per scope and list the hit rate with its Wilson interval
    against the mean predicted probability, and the ROI at stake 1 per pick
    with its interval (:func:`roi_interval`). ``accumulators``: per scope, list
    and size ``k`` how often the top-``k`` picks of a matchday **all** won,
    against the mean predicted probability of that accumulator (product of the
    ``k`` probabilities, independence assumed) and the ROI of the accumulator
    at the product of the best prices when every leg was quoted.
    """

    picks: pd.DataFrame
    by_matchday: pd.DataFrame
    summary: pd.DataFrame
    accumulators: pd.DataFrame


_PICK_LISTS: Final[tuple[tuple[str, Market, str], ...]] = (("X", Market.MATCH_ODDS, "X"), ("GG", Market.BTTS, "GG"))


def _profit(won: bool, odds: float) -> float:
    if not np.isfinite(odds):
        return float("nan")
    return float(odds - 1.0) if won else -1.0


def _rate_row(n: int, hits: int, mean_p: float, profits: np.ndarray, odds: np.ndarray) -> dict[str, Any]:
    low, high = wilson_interval(hits, n)
    roi, roi_low, roi_high = roi_interval(profits, odds=odds)
    quoted = int(np.isfinite(profits).sum())
    return {
        "hit_rate": hits / n if n else float("nan"),
        "wilson_low": low,
        "wilson_high": high,
        "mean_p": mean_p,
        "n_with_odds": quoted,
        "profit": float(np.nansum(profits)) if quoted else float("nan"),
        "roi": roi,
        "roi_low": roi_low,
        "roi_high": roi_high,
    }


def picks_track_record(
    frame: pd.DataFrame,
    *,
    n_draws: int = DEFAULT_N_DRAWS,
    n_btts: int = DEFAULT_N_BTTS,
    odds_history: pd.DataFrame | Iterable[Any] | None = None,
    by_section: bool = True,
    accumulator_sizes: Sequence[int] = DEFAULT_ACCUMULATOR_SIZES,
    prefix: str = "p_",
) -> PicksTrackRecord:
    """Rebuild the draw and GG lists of every past matchday and score them (see :class:`PicksTrackRecord`).

    The lists are ranked with :func:`src.engine.picks.rank_picks` on the
    out-of-sample probabilities of the backtest (``prefix`` columns, e.g. a
    recalibrated copy), exactly like the dashboard ranks the matches of a
    matchday: across every section (:data:`ALL_SECTIONS`) and, with
    ``by_section``, within each section. ``odds_history`` adds the ROI at the
    real pre-match prices.
    """
    pick_columns = ["matchday", "scope", "list", "rank", "match_id", "date", "section", "girone", "home_team", "away_team", "market", "selection", "p_model", "won", "odds", "profit"]
    lookup = _best_odds_lookup(odds_history)
    records: list[dict[str, Any]] = []
    sizes = {"X": int(n_draws), "GG": int(n_btts)}
    if not frame.empty:
        for matchday, rows in frame.groupby("matchday", sort=True):
            by_id = {str(r["match_id"]): r for r in rows.to_dict(orient="records")}
            all_predictions = _predictions(rows, prefix)  # built once per matchday, filtered per scope
            for scope, scoped in _scoped(rows, by_section):
                ids = set(scoped["match_id"].astype(str))
                predictions = [p for p in all_predictions if p.match_id in ids]
                for name, market, selection in _PICK_LISTS:
                    for pick in rank_picks(predictions, market=market, selection=selection, n=sizes[name]):
                        row = by_id[pick.match_id]
                        won = _won(row, market, selection)
                        odds = lookup.get((pick.match_id, selection), float("nan"))
                        records.append(
                            {
                                "matchday": str(matchday),
                                "scope": scope,
                                "list": name,
                                "rank": pick.rank,
                                "match_id": pick.match_id,
                                "date": pd.Timestamp(pick.date),
                                "section": str(row["section"]),
                                "girone": pick.girone.value,
                                "home_team": pick.home_team,
                                "away_team": pick.away_team,
                                "market": market.value,
                                "selection": selection,
                                "p_model": pick.p_model,
                                "won": won,
                                "odds": odds,
                                "profit": _profit(won, odds),
                            }
                        )
    picks = pd.DataFrame(records, columns=pick_columns)
    if picks.empty:
        empty_md = pd.DataFrame(columns=["scope", "list", "matchday", "n_picks", "hits", "expected_hits", "all_won"])
        empty_summary = pd.DataFrame(columns=["scope", "list", "n_matchdays", "n_picks", "hits", "expected_hits", "hit_rate", "wilson_low", "wilson_high", "mean_p", "n_with_odds", "profit", "roi", "roi_low", "roi_high"])
        empty_acc = pd.DataFrame(columns=["scope", "list", "k", "n_matchdays", "wins", "win_rate", "wilson_low", "wilson_high", "mean_p", "expected_wins", "n_with_odds", "profit", "roi", "roi_low", "roi_high"])
        return PicksTrackRecord(picks, empty_md, empty_summary, empty_acc)
    picks["won"] = picks["won"].astype(bool)
    by_matchday = (
        picks.groupby(["scope", "list", "matchday"], sort=False)
        .agg(n_picks=("won", "size"), hits=("won", "sum"), expected_hits=("p_model", "sum"), all_won=("won", "all"))
        .reset_index()
    )
    by_matchday["hits"] = by_matchday["hits"].astype(int)
    summary_rows: list[dict[str, Any]] = []
    for (scope, name), group in picks.groupby(["scope", "list"], sort=False):
        n = int(len(group))
        hits = int(group["won"].sum())
        summary_rows.append(
            {
                "scope": scope,
                "list": name,
                "n_matchdays": int(group["matchday"].nunique()),
                "n_picks": n,
                "hits": hits,
                "expected_hits": float(group["p_model"].sum()),
                **_rate_row(n, hits, float(group["p_model"].mean()), group["profit"].to_numpy(dtype=float), group["odds"].to_numpy(dtype=float)),
            }
        )
    summary = _order_frame(pd.DataFrame(summary_rows), ["scope", "list"])
    acc_rows: list[dict[str, Any]] = []
    for (scope, name), group in picks.groupby(["scope", "list"], sort=False):
        for k in sorted({int(k) for k in accumulator_sizes if int(k) >= 1}):
            products: list[float] = []
            wins = 0
            profits: list[float] = []
            prices: list[float] = []
            for _md, day in group.groupby("matchday", sort=True):
                top = day.sort_values("rank").head(k)
                if len(top) < k:
                    continue
                won = bool(top["won"].all())
                wins += int(won)
                products.append(float(np.prod(top["p_model"].to_numpy(dtype=float))))
                odds = top["odds"].to_numpy(dtype=float)
                price = float(np.prod(odds)) if np.isfinite(odds).all() else float("nan")
                prices.append(price)
                profits.append(_profit(won, price))
            n = len(products)
            if not n:
                continue
            acc_rows.append(
                {
                    "scope": scope,
                    "list": name,
                    "k": k,
                    "n_matchdays": n,
                    "wins": wins,
                    "win_rate": wins / n,
                    "expected_wins": float(np.sum(products)),
                    **{
                        key: value
                        for key, value in _rate_row(n, wins, float(np.mean(products)), np.asarray(profits, dtype=float), np.asarray(prices, dtype=float)).items()
                        if key != "hit_rate"
                    },
                }
            )
    acc_columns = ["scope", "list", "k", "n_matchdays", "wins", "win_rate", "wilson_low", "wilson_high", "mean_p", "expected_wins", "n_with_odds", "profit", "roi", "roi_low", "roi_high"]
    accumulators = _order_frame(pd.DataFrame(acc_rows, columns=acc_columns), ["scope", "list"])
    log.info(
        "picks track record",
        extra={"n_picks": int(len(picks)), "n_matchdays": int(picks["matchday"].nunique()), "n_with_odds": int(picks["odds"].notna().sum())},
    )
    return PicksTrackRecord(picks=picks, by_matchday=by_matchday, summary=summary, accumulators=accumulators)


# =============================================================================
# Track record: slips
# =============================================================================
@dataclass(frozen=True)
class BacktestLeg:
    """One leg of a slip: a selection of one match with its model probability."""

    match_id: str
    market: Market
    selection: str
    p: float

    def __post_init__(self) -> None:
        object.__setattr__(self, "market", market_of(self.market))
        if self.selection not in SELECTIONS[self.market]:
            raise ValueError(f"{self.selection!r} is not a selection of {self.market.value}")
        if not 0.0 <= float(self.p) <= 1.0:
            raise ValueError(f"p must be in [0, 1], got {self.p}")


@dataclass(frozen=True)
class BacktestSlip:
    """A slip (schedina): a name and its legs, one per match.

    ``p`` is the probability that every leg wins assuming independence (the
    product of the legs' probabilities).
    """

    name: str
    legs: tuple[BacktestLeg, ...]

    @property
    def p(self) -> float:
        return float(np.prod([leg.p for leg in self.legs])) if self.legs else float("nan")


#: ``builder(pre-match predictions of one matchday) -> slips``. The frame has the columns of the backtest
#: known before kick-off (match, competition, section, ``lam``/``mu``/``rho`` and the ``p_*`` probabilities;
#: never goals or outcomes); each slip exposes ``legs`` (attribute or mapping key) whose items expose
#: ``match_id``, ``market``, ``selection`` and ``p`` (attributes or keys). ``name`` (or ``kind`` / ``label``) names it.
SlipBuilder = Callable[[pd.DataFrame], Sequence[Any]]


def _rank_frame(frame: pd.DataFrame, column: str) -> pd.DataFrame:
    order = frame["girone"].astype(str).map(_girone_order)
    return frame.assign(_o=order, _p=-frame[column].astype(float)).sort_values(["_p", "date", "_o", "match_id"], kind="stable").drop(columns=["_o", "_p"])


def default_slip_builder(
    predictions: pd.DataFrame,
    *,
    n_draws: int = DEFAULT_N_SLIP_DRAWS,
    n_btts: int = DEFAULT_N_SLIP_BTTS,
    n_mixed: int = DEFAULT_N_MIXED,
    mixed_selections: Sequence[str] = DEFAULT_MIXED_SELECTIONS,
    min_probability: float = 0.0,
) -> list[BacktestSlip]:
    """Minimal slip builder (the dashboard injects its own): three slips over the given matches.

    ``'X'``: the ``n_draws`` most probable draws; ``'GG'``: the ``n_btts`` most
    probable GG; ``'Mista'``: for every match its best sign among
    ``mixed_selections`` (default 1, 2, GG, Over 2.5) by model probability,
    then the ``n_mixed`` matches whose best sign is most probable. One leg per
    match per slip; legs below ``min_probability`` are left out; ties are
    broken by date, competition order and ``match_id`` (deterministic).
    """
    if predictions.empty:
        return []
    pool = predictions
    slips: list[BacktestSlip] = []
    for name, selection, n in (("X", "X", n_draws), ("GG", "GG", n_btts)):
        column = f"p_{SELECTION_SUFFIX[selection]}"
        ranked = _rank_frame(pool.loc[pool[column] >= min_probability], column).head(max(int(n), 0))
        market = _market_of_selection(selection)
        slips.append(BacktestSlip(name, tuple(BacktestLeg(str(r["match_id"]), market, selection, float(r[column])) for r in ranked.to_dict(orient="records"))))
    signs = [s for s in mixed_selections if s in SELECTION_SUFFIX]
    if signs:
        values = pool[[f"p_{SELECTION_SUFFIX[s]}" for s in signs]].to_numpy(dtype=float)
        best = np.argmax(values, axis=1)
        mixed = pool.assign(_best=values[np.arange(len(pool)), best], _sign=[signs[i] for i in best])
        ranked = _rank_frame(mixed.loc[mixed["_best"] >= min_probability], "_best").head(max(int(n_mixed), 0))
        slips.append(
            BacktestSlip(
                "Mista",
                tuple(BacktestLeg(str(r["match_id"]), _market_of_selection(str(r["_sign"])), str(r["_sign"]), float(r["_best"])) for r in ranked.to_dict(orient="records")),
            )
        )
    return slips


def _field(obj: Any, *names: str) -> Any:
    for name in names:
        if isinstance(obj, Mapping) and name in obj:
            return obj[name]
        if hasattr(obj, name):
            return getattr(obj, name)
    return None


@dataclass(frozen=True)
class SlipsTrackRecord:
    """What :func:`slips_track_record` found.

    ``slips``: one row per matchday and slip (``n_legs``, ``legs_won``,
    ``expected_legs`` = sum of the leg probabilities, ``p_slip`` = their
    product, ``won``, ``fair_odds`` = ``1 / p_slip``, ``book_odds`` = product of
    the best prices when every leg was quoted, ``profit`` at stake 1).
    ``legs``: one row per leg with its outcome and price. ``summary``: per slip
    name the win rate (Wilson) against the mean ``p_slip``, the leg hit rate
    (Wilson) against the mean leg probability, the ROI with its interval.
    """

    slips: pd.DataFrame
    legs: pd.DataFrame
    summary: pd.DataFrame


def slips_track_record(
    frame: pd.DataFrame,
    *,
    odds_history: pd.DataFrame | Iterable[Any] | None = None,
    slip_builder: SlipBuilder | None = None,
    prefix: str = "p_",
) -> SlipsTrackRecord:
    """Ask ``slip_builder`` (default :func:`default_slip_builder`) for the slips of every past matchday and score them.

    The builder receives only the pre-match columns of the matchday's matches
    (probabilities from ``prefix``, renamed to ``p_*``): it cannot peek at the
    results. A leg whose match is not in the matchday is an error of the
    builder: the slip is skipped with a warning. Stake 1 per slip; ROI at the
    product of the best prices only when every leg was quoted.
    """
    builder = slip_builder if slip_builder is not None else default_slip_builder
    lookup = _best_odds_lookup(odds_history)
    slip_rows: list[dict[str, Any]] = []
    leg_rows: list[dict[str, Any]] = []
    if not frame.empty:
        rename = {f"{prefix}{s}": f"p_{s}" for s in SELECTION_SUFFIX.values()} if prefix != "p_" else {}
        for matchday, rows in frame.groupby("matchday", sort=True):
            source = rows.drop(columns=list(PROB_COLUMNS), errors="ignore").rename(columns=rename) if rename else rows
            pre = source[[c for c in _PRE_MATCH_COLUMNS if c in source.columns]].reset_index(drop=True)
            by_id = {str(r["match_id"]): r for r in rows.to_dict(orient="records")}
            for index, slip in enumerate(builder(pre), start=1):
                name = str(_field(slip, "name", "kind", "label") or f"Schedina {index}")
                legs = list(_field(slip, "legs") or [])
                parsed: list[tuple[str, Market, str, float]] = []
                for leg in legs:
                    match_id = str(_field(leg, "match_id"))
                    selection = str(_field(leg, "selection"))
                    raw_market = _field(leg, "market")
                    market = market_of(raw_market) if raw_market is not None else _market_of_selection(selection)
                    p = float(_field(leg, "p", "p_model", "probability"))
                    parsed.append((match_id, market, selection, p))
                unknown = [m for m, *_ in parsed if m not in by_id]
                if unknown:
                    log.warning("slip skipped: legs outside the matchday", extra={"matchday": str(matchday), "slip": name, "match_ids": unknown[:5]})
                    continue
                if not parsed:
                    continue
                outcomes = [_won(by_id[m], market, selection) for m, market, selection, _p in parsed]
                prices = np.array([lookup.get((m, selection), float("nan")) for m, _market, selection, _p in parsed], dtype=float)
                probabilities = np.array([p for *_rest, p in parsed], dtype=float)
                won = bool(all(outcomes))
                book = float(np.prod(prices)) if np.isfinite(prices).all() else float("nan")
                p_slip = float(np.prod(probabilities))
                slip_rows.append(
                    {
                        "matchday": str(matchday),
                        "slip": name,
                        "n_legs": len(parsed),
                        "legs_won": int(sum(outcomes)),
                        "expected_legs": float(probabilities.sum()),
                        "p_slip": p_slip,
                        "won": won,
                        "fair_odds": 1.0 / p_slip if p_slip > 0 else float("inf"),
                        "book_odds": book,
                        "profit": _profit(won, book),
                    }
                )
                for position, ((m, market, selection, p), ok, price) in enumerate(zip(parsed, outcomes, prices), start=1):
                    row = by_id[m]
                    leg_rows.append(
                        {
                            "matchday": str(matchday),
                            "slip": name,
                            "leg": position,
                            "match_id": m,
                            "section": str(row["section"]),
                            "girone": str(row["girone"]),
                            "home_team": str(row["home_team"]),
                            "away_team": str(row["away_team"]),
                            "market": market.value,
                            "selection": selection,
                            "p": p,
                            "won": bool(ok),
                            "odds": float(price),
                        }
                    )
    slips = pd.DataFrame(slip_rows, columns=["matchday", "slip", "n_legs", "legs_won", "expected_legs", "p_slip", "won", "fair_odds", "book_odds", "profit"])
    legs = pd.DataFrame(leg_rows, columns=["matchday", "slip", "leg", "match_id", "section", "girone", "home_team", "away_team", "market", "selection", "p", "won", "odds"])
    summary_columns = [
        "slip",
        "n_slips",
        "wins",
        "win_rate",
        "wilson_low",
        "wilson_high",
        "mean_p",
        "expected_wins",
        "n_legs",
        "legs_won",
        "leg_hit_rate",
        "leg_wilson_low",
        "leg_wilson_high",
        "mean_leg_p",
        "n_with_odds",
        "profit",
        "roi",
        "roi_low",
        "roi_high",
    ]
    rows_out: list[dict[str, Any]] = []
    for name, group in slips.groupby("slip", sort=False):
        n = int(len(group))
        wins = int(group["won"].astype(bool).sum())
        mine = legs.loc[legs["slip"] == name]
        n_legs = int(len(mine))
        legs_won = int(mine["won"].astype(bool).sum())
        leg_low, leg_high = wilson_interval(legs_won, n_legs)
        rate = _rate_row(n, wins, float(group["p_slip"].mean()), group["profit"].to_numpy(dtype=float), group["book_odds"].to_numpy(dtype=float))
        rows_out.append(
            {
                "slip": name,
                "n_slips": n,
                "wins": wins,
                "win_rate": rate["hit_rate"],
                "wilson_low": rate["wilson_low"],
                "wilson_high": rate["wilson_high"],
                "mean_p": rate["mean_p"],
                "expected_wins": float(group["p_slip"].sum()),
                "n_legs": n_legs,
                "legs_won": legs_won,
                "leg_hit_rate": legs_won / n_legs if n_legs else float("nan"),
                "leg_wilson_low": leg_low,
                "leg_wilson_high": leg_high,
                "mean_leg_p": float(mine["p"].mean()) if n_legs else float("nan"),
                "n_with_odds": rate["n_with_odds"],
                "profit": rate["profit"],
                "roi": rate["roi"],
                "roi_low": rate["roi_low"],
                "roi_high": rate["roi_high"],
            }
        )
    summary = pd.DataFrame(rows_out, columns=summary_columns)
    log.info("slips track record", extra={"n_slips": int(len(slips)), "n_legs": int(len(legs)), "n_with_odds": int(slips["book_odds"].notna().sum()) if len(slips) else 0})
    return SlipsTrackRecord(slips=slips, legs=legs, summary=summary)


# =============================================================================
# Track record: value singles
# =============================================================================
@dataclass(frozen=True)
class ValueTrackRecord:
    """What :func:`value_track_record` found: ``bets`` (one row per value single) and ``summary`` (per scope and market)."""

    bets: pd.DataFrame
    summary: pd.DataFrame


def value_track_record(
    frame: pd.DataFrame,
    odds_history: pd.DataFrame | Iterable[Any] | None,
    *,
    min_ev: float = DEFAULT_VALUE_MIN_EV,
    markets: Iterable[Market | str] = DEFAULT_MARKETS,
    by_section: bool = True,
    prefix: str = "p_",
) -> ValueTrackRecord:
    """The value singles the model would have flagged, with their outcome at the real pre-match price.

    A bet is every selection of a match with a price whose expected value at
    the **best** price, ``p_model * odds - 1``, is at least ``min_ev`` (stake 1,
    no flags, no Kelly sizing: the raw signal). ``summary`` gives, per scope
    (:data:`ALL_SECTIONS` and, with ``by_section``, each section) and market
    plus an ``'Tutti'`` market row, the hit rate (Wilson) against the mean
    model probability and the ROI with its interval (:func:`roi_interval`).
    """
    bet_columns = ["matchday", "match_id", "date", "section", "girone", "home_team", "away_team", "market", "selection", "p_model", "odds", "ev", "won", "profit"]
    summary_columns = ["scope", "market", "n_bets", "hits", "hit_rate", "wilson_low", "wilson_high", "mean_p", "mean_odds", "n_with_odds", "profit", "roi", "roi_low", "roi_high"]
    table = odds_table(odds_history)
    if frame.empty or table.empty:
        return ValueTrackRecord(pd.DataFrame(columns=bet_columns), pd.DataFrame(columns=summary_columns))
    merged = frame.merge(table, on="match_id", how="inner")
    bets: list[pd.DataFrame] = []
    for market in markets:
        spec = market_spec(market)
        y = spec.outcome_index(merged)
        for index, (selection, suffix) in enumerate(zip(spec.selections, spec.suffixes)):
            p = merged[f"{prefix}{suffix}"].to_numpy(dtype=float)
            odds = merged[f"odds_{suffix}"].to_numpy(dtype=float)
            ev = p * odds - 1.0
            take = np.isfinite(ev) & (ev >= float(min_ev)) & (y >= 0)
            if not take.any():
                continue
            chosen = merged.loc[take]
            won = y[take] == index
            bets.append(
                pd.DataFrame(
                    {
                        "matchday": chosen["matchday"].astype(str).to_numpy(),
                        "match_id": chosen["match_id"].astype(str).to_numpy(),
                        "date": chosen["date"].to_numpy(),
                        "section": chosen["section"].astype(str).to_numpy(),
                        "girone": chosen["girone"].astype(str).to_numpy(),
                        "home_team": chosen["home_team"].astype(str).to_numpy(),
                        "away_team": chosen["away_team"].astype(str).to_numpy(),
                        "market": spec.key,
                        "selection": selection,
                        "p_model": p[take],
                        "odds": odds[take],
                        "ev": ev[take],
                        "won": won,
                        "profit": np.where(won, odds[take] - 1.0, -1.0),
                    }
                )
            )
    frame_bets = pd.concat(bets, ignore_index=True)[bet_columns] if bets else pd.DataFrame(columns=bet_columns)
    rows: list[dict[str, Any]] = []
    if len(frame_bets):
        frame_bets = frame_bets.sort_values(["date", "match_id", "market", "selection"], kind="stable").reset_index(drop=True)
        scopes: list[tuple[str, pd.DataFrame]] = [(ALL_SECTIONS, frame_bets)]
        if by_section:
            scopes += [(s, frame_bets.loc[frame_bets["section"] == s]) for s in dict.fromkeys(frame_bets["section"])]
        for scope, scoped in scopes:
            groups = [("Tutti", scoped)] + [(m, g) for m, g in scoped.groupby("market", sort=False)]
            for market_name, group in groups:
                n = int(len(group))
                hits = int(group["won"].astype(bool).sum())
                rate = _rate_row(n, hits, float(group["p_model"].mean()), group["profit"].to_numpy(dtype=float), group["odds"].to_numpy(dtype=float))
                rows.append({"scope": scope, "market": market_name, "n_bets": n, "hits": hits, "mean_odds": float(group["odds"].mean()), **rate})
    summary = pd.DataFrame(rows, columns=summary_columns)
    return ValueTrackRecord(bets=frame_bets, summary=summary)
