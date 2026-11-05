"""Synthetic Serie C data with *known* Dixon-Coles parameters.

Used by the test-suite (parameter-recovery tests), by ``main.py`` (console demo
on a 20-match mini dataset) and by the dashboard as an offline fallback. The
sampler implements the Dixon-Coles bivariate pmf independently from the engine
so the engine can be validated against it.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, timedelta
from typing import Mapping, Sequence

import numpy as np
import pandas as pd

from src.common.schemas import (
    SELECTIONS,
    Fixture,
    Girone,
    Market,
    MatchRecord,
    OddsQuote,
    matches_to_frame,
)
from src.common.teams import TeamRegistry, load_registry

__all__ = [
    "TrueParams",
    "SimulatedLeague",
    "dixon_coles_matrix",
    "market_probabilities",
    "round_robin",
    "simulate_league",
    "simulate_mini_dataset",
    "simulate_bookmaker_odds",
    "inject_regression_trap",
]

MAX_GOALS = 10


# --------------------------------------------------------------------------- math
def _tau(x: np.ndarray, y: np.ndarray, lam: float, mu: float, rho: float) -> np.ndarray:
    out = np.ones_like(x, dtype=float)
    out = np.where((x == 0) & (y == 0), 1.0 - lam * mu * rho, out)
    out = np.where((x == 0) & (y == 1), 1.0 + lam * rho, out)
    out = np.where((x == 1) & (y == 0), 1.0 + mu * rho, out)
    out = np.where((x == 1) & (y == 1), 1.0 - rho, out)
    return out


def dixon_coles_matrix(lam: float, mu: float, rho: float, max_goals: int = MAX_GOALS) -> np.ndarray:
    """Joint pmf P(X=x, Y=y) on a (max_goals+1)x(max_goals+1) grid, renormalised."""
    from scipy.stats import poisson

    xs = np.arange(max_goals + 1)
    px = poisson.pmf(xs, lam)
    py = poisson.pmf(xs, mu)
    grid = np.outer(px, py)
    X, Y = np.meshgrid(xs, xs, indexing="ij")
    grid = grid * np.clip(_tau(X, Y, lam, mu, rho), 1e-12, None)
    return grid / grid.sum()


def market_probabilities(matrix: np.ndarray) -> dict[Market, dict[str, float]]:
    """Derive 1X2 / O-U / GG-NG probabilities from a joint score matrix."""
    n = matrix.shape[0]
    X, Y = np.meshgrid(np.arange(n), np.arange(n), indexing="ij")
    total = X + Y
    out: dict[Market, dict[str, float]] = {
        Market.MATCH_ODDS: {
            "1": float(matrix[X > Y].sum()),
            "X": float(np.trace(matrix)),
            "2": float(matrix[X < Y].sum()),
        },
        Market.BTTS: {
            "GG": float(matrix[(X > 0) & (Y > 0)].sum()),
            "NG": float(matrix[(X == 0) | (Y == 0)].sum()),
        },
    }
    for market, line in ((Market.OVER_UNDER_15, 1.5), (Market.OVER_UNDER_25, 2.5), (Market.OVER_UNDER_35, 3.5)):
        over = float(matrix[total > line].sum())
        out[market] = {f"Over {line}": over, f"Under {line}": 1.0 - over}
    return out


# ------------------------------------------------------------------- parameters
@dataclass(frozen=True)
class TrueParams:
    """Ground-truth parameters used to generate a league."""

    alpha: Mapping[str, float]  # attack, sum == 0 within each girone
    beta: Mapping[str, float]  # defence (higher = concedes less)
    gamma: Mapping[Girone, float]  # home advantage per girone
    rho: float
    xi: float = 0.0  # time-decay used for generation (0 = stationary)

    def rates(self, home: str, away: str, girone: Girone) -> tuple[float, float]:
        lam = float(np.exp(self.alpha[home] - self.beta[away] + self.gamma[girone]))
        mu = float(np.exp(self.alpha[away] - self.beta[home]))
        return lam, mu


def sample_true_params(
    registry: TeamRegistry,
    rng: np.random.Generator,
    *,
    sigma_alpha: float = 0.28,
    sigma_beta: float = 0.22,
    gamma: Mapping[Girone, float] | None = None,
    rho: float = -0.06,
) -> TrueParams:
    alpha: dict[str, float] = {}
    beta: dict[str, float] = {}
    for g in Girone:
        teams = registry.teams_by_girone.get(g, ())
        if not teams:
            continue
        a = rng.normal(0.0, sigma_alpha, len(teams))
        a -= a.mean()  # identifiability: sum alpha = 0 per girone
        b = rng.normal(0.0, sigma_beta, len(teams))
        for t, ai, bi in zip(teams, a, b):
            alpha[t] = float(ai)
            beta[t] = float(bi)
    if gamma is None:
        gamma = {Girone.A: 0.30, Girone.B: 0.22, Girone.C: 0.38}
    return TrueParams(alpha=alpha, beta=beta, gamma=dict(gamma), rho=rho)


# -------------------------------------------------------------------- schedule
def round_robin(teams: Sequence[str], *, double: bool = True) -> list[list[tuple[str, str]]]:
    """Circle-method round robin. Returns rounds of (home, away) pairs."""
    teams = list(teams)
    if len(teams) % 2:
        teams.append("__BYE__")
    n = len(teams)
    rounds: list[list[tuple[str, str]]] = []
    order = teams[:]
    for r in range(n - 1):
        pairs: list[tuple[str, str]] = []
        for i in range(n // 2):
            a, b = order[i], order[n - 1 - i]
            if "__BYE__" in (a, b):
                continue
            pairs.append((a, b) if (r + i) % 2 == 0 else (b, a))
        rounds.append(pairs)
        order = [order[0]] + [order[-1]] + order[1:-1]
    if double:
        rounds += [[(b, a) for a, b in rnd] for rnd in rounds]
    return rounds


# ------------------------------------------------------------------- generator
@dataclass
class SimulatedLeague:
    registry: TeamRegistry
    true_params: TrueParams
    matches: pd.DataFrame
    fixtures: list[Fixture]
    odds: list[OddsQuote] = field(default_factory=list)

    @property
    def n_matches(self) -> int:
        return int(len(self.matches))


def _simulate_stats(rng: np.random.Generator, lam: float, goals: int) -> tuple[float, float, float, float]:
    """xG, shots, shots on target, corners roughly consistent with the rate."""
    xg = float(lam * rng.lognormal(0.0, 0.35))
    xg = max(xg, 0.05 + 0.4 * goals * rng.uniform(0.3, 0.9))
    shots = int(max(goals, rng.poisson(3.0 + 7.5 * xg)))
    sot = int(np.clip(rng.binomial(shots, 0.36), goals, shots))
    corners = int(rng.poisson(4.5 + 1.2 * xg))
    return round(xg, 2), float(shots), float(sot), float(corners)


def simulate_league(
    *,
    seed: int = 7,
    registry: TeamRegistry | None = None,
    teams_by_girone: Mapping[str, Sequence[str]] | None = None,
    n_rounds: int | None = None,
    start_date: date = date(2025, 8, 24),
    days_between_rounds: int = 7,
    true_params: TrueParams | None = None,
    with_odds: bool = True,
    bookmaker_margin: float = 0.06,
    bookmaker_noise: float = 0.04,
) -> SimulatedLeague:
    """Generate a full (or truncated) season with known parameters.

    ``n_rounds`` limits the number of *played* rounds; the following round becomes
    the fixture list. ``None`` plays every round but the last one, so there is
    always a fixture list. By default the real 2025/26 Serie C composition is used.
    """
    rng = np.random.default_rng(seed)
    if registry is None:
        registry = (
            TeamRegistry.from_teams("sim", teams_by_girone)
            if teams_by_girone is not None
            else load_registry()
        )
    params = true_params or sample_true_params(registry, rng)

    schedules = {g: round_robin(registry.teams_by_girone[g]) for g in Girone if g in registry.teams_by_girone}
    total_rounds = max(len(s) for s in schedules.values())
    played_rounds = total_rounds - 1 if n_rounds is None else max(0, min(n_rounds, total_rounds))

    records: list[MatchRecord] = []
    fixtures: list[Fixture] = []
    counter = 0
    for rnd in range(total_rounds):
        when = start_date + timedelta(days=rnd * days_between_rounds)
        for g, sched in schedules.items():
            if rnd >= len(sched):
                continue
            for home, away in sched[rnd]:
                counter += 1
                mid = f"SIM-{g.value}-{rnd + 1:02d}-{counter:04d}"
                if rnd >= played_rounds:
                    if rnd == played_rounds:
                        fixtures.append(Fixture(mid, when, rnd + 1, g, home, away))
                    continue
                lam, mu = params.rates(home, away, g)
                grid = dixon_coles_matrix(lam, mu, params.rho)
                idx = rng.choice(grid.size, p=grid.ravel())
                hg, ag = divmod(int(idx), grid.shape[1])
                hxg, hs, hst, hc = _simulate_stats(rng, lam, hg)
                axg, as_, ast, ac = _simulate_stats(rng, mu, ag)
                records.append(
                    MatchRecord(
                        match_id=mid,
                        date=when,
                        round=rnd + 1,
                        girone=g,
                        home_team=home,
                        away_team=away,
                        home_goals=hg,
                        away_goals=ag,
                        home_shots=hs,
                        away_shots=as_,
                        home_shots_target=hst,
                        away_shots_target=ast,
                        home_corners=hc,
                        away_corners=ac,
                        home_xg=hxg,
                        away_xg=axg,
                    )
                )
    league = SimulatedLeague(registry=registry, true_params=params, matches=matches_to_frame(records), fixtures=fixtures)
    if with_odds and fixtures:
        league.odds = simulate_bookmaker_odds(league, rng, margin=bookmaker_margin, noise=bookmaker_noise)
    return league


def simulate_bookmaker_odds(
    league: SimulatedLeague,
    rng: np.random.Generator | None = None,
    *,
    margin: float = 0.06,
    noise: float = 0.04,
    bookmaker: str = "SimBook",
    markets: Sequence[Market] = tuple(Market),
) -> list[OddsQuote]:
    """Bookmaker prices = true probability * multiplicative noise, then an
    over-round of ``margin`` is applied proportionally (sum implied = 1+margin).
    A noisy book is exactly what creates +EV spots for the value engine."""
    rng = rng or np.random.default_rng(0)
    quotes: list[OddsQuote] = []
    for fx in league.fixtures:
        lam, mu = league.true_params.rates(fx.home_team, fx.away_team, fx.girone)
        probs = market_probabilities(dixon_coles_matrix(lam, mu, league.true_params.rho))
        for market in markets:
            p = np.array([probs[market][s] for s in SELECTIONS[market]])
            p = p * rng.lognormal(0.0, noise, len(p))
            p = p / p.sum() * (1.0 + margin)
            quotes.append(
                OddsQuote(
                    match_id=fx.match_id,
                    bookmaker=bookmaker,
                    market=market,
                    odds={s: max(1.01, round(float(1.0 / pi), 2)) for s, pi in zip(SELECTIONS[market], p)},
                )
            )
    return quotes


def inject_regression_trap(
    matches: pd.DataFrame,
    team: str,
    *,
    window: int = 3,
    goals_against_total: int = 1,
    xga_total: float = 4.2,
) -> pd.DataFrame:
    """Rewrite ``team``'s last ``window`` matches so that it conceded at most
    ``goals_against_total`` goals in total while facing ``xga_total`` xG against.
    Returns a modified copy (used to showcase the REGRESSION_TRAP filter)."""
    df = matches.copy()
    mask = (df["home_team"] == team) | (df["away_team"] == team)
    idx = df.loc[mask].sort_values("date").index[-window:]
    if len(idx) < window:
        raise ValueError(f"{team} has fewer than {window} matches")
    ga_split = [goals_against_total] + [0] * (window - 1)
    xga_split = np.full(window, xga_total / window)
    for i, ga, xga in zip(idx, ga_split, xga_split):
        is_home = df.at[i, "home_team"] == team
        if is_home:
            df.at[i, "away_goals"] = int(ga)
            df.at[i, "away_xg"] = float(round(xga, 2))
            df.at[i, "away_shots"] = max(float(df.at[i, "away_shots"]) if pd.notna(df.at[i, "away_shots"]) else 0.0, 12.0)
            df.at[i, "away_shots_target"] = max(float(df.at[i, "away_shots_target"]) if pd.notna(df.at[i, "away_shots_target"]) else 0.0, 5.0)
        else:
            df.at[i, "home_goals"] = int(ga)
            df.at[i, "home_xg"] = float(round(xga, 2))
            df.at[i, "home_shots"] = max(float(df.at[i, "home_shots"]) if pd.notna(df.at[i, "home_shots"]) else 0.0, 12.0)
            df.at[i, "home_shots_target"] = max(float(df.at[i, "home_shots_target"]) if pd.notna(df.at[i, "home_shots_target"]) else 0.0, 5.0)
    return df


MINI_TEAMS: dict[str, tuple[str, ...]] = {
    "A": ("Vicenza", "Cittadella", "Lecco", "Novara", "Trento", "Renate"),
    "B": ("Arezzo", "Ascoli", "Ternana", "Perugia"),
    "C": ("Benevento", "Salernitana", "Catania", "Crotone"),
}


def simulate_mini_dataset(seed: int = 42, n_matches: int = 20) -> SimulatedLeague:
    """Exactly ``n_matches`` played games across the three gironi (14 teams),
    with the following round as fixtures and simulated bookmaker odds."""
    league = simulate_league(seed=seed, teams_by_girone=MINI_TEAMS, n_rounds=10, with_odds=False)
    df = league.matches.sort_values(["date", "match_id"]).head(n_matches).reset_index(drop=True)
    last_round = int(df["round"].max())
    nxt = league.matches[league.matches["round"] == last_round + 1]
    fixtures = [
        Fixture(r.match_id, r.date.date(), int(r.round), Girone.parse(r.girone), r.home_team, r.away_team)
        for r in nxt.itertuples(index=False)
    ]
    mini = SimulatedLeague(registry=league.registry, true_params=league.true_params, matches=df, fixtures=fixtures)
    mini.odds = simulate_bookmaker_odds(mini, np.random.default_rng(seed))
    return mini
