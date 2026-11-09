"""One Dixon-Coles model per *fit group*: the competitions of one country are fitted jointly, countries separately.

Why groups
----------
The Italian competitions (Serie A, B, C and D) are one connected system: the
clubs promoted and relegated between divisions carry their strength across
seasons, so one joint fit (:class:`~src.engine.model.DixonColesEngine` with a
home advantage and a ``sum(alpha) == 0`` constraint group per competition)
estimates every Italian club against the right opposition. Two countries share
no match at all: a joint fit of Italy and England would just be two
independent blocks inside one bigger optimisation (slower, one shared ``rho``
and one shared prior for leagues with different scoring patterns). The fit
group of a competition is therefore its ``country``
(:attr:`src.common.competitions.Competition.fit_group`,
:func:`src.common.competitions.fit_groups`), and :func:`fit_by_group` runs one
engine per group present in the data.

Same interface as one engine
----------------------------
:class:`GroupedFit` answers the calls the dashboard, the console and the
tables make on a single engine -- :meth:`~GroupedFit.predict`,
:meth:`~GroupedFit.predict_fixtures`, :meth:`~GroupedFit.rates`,
:meth:`~GroupedFit.score_matrix`, :meth:`~GroupedFit.market_probabilities`,
:meth:`~GroupedFit.fair_odds`, :meth:`~GroupedFit.ranking`,
:meth:`~GroupedFit.residuals`, :attr:`~GroupedFit.config` and
:attr:`~GroupedFit.params_` -- by routing each competition to the engine of
its group. With the shipped configuration there is one group (``'Italy'``) and
every answer is the one the single engine gives (``params_`` is the very same
:class:`~src.engine.model.FittedParams` object). With several groups
:attr:`~GroupedFit.params_` is a combined *summary* (union of the team
parameters, sums of the fit statistics, ``rho`` averaged with the effective
sample sizes as weights): predictions never use it, they use each group's own
parameters, which :attr:`~GroupedFit.params_by_group` and
:meth:`~GroupedFit.diagnostics` expose per group.

Optimiser
---------
:func:`fit_by_group` defaults to ``method='L-BFGS-B'``: on the full 242-team /
10k-match perimeter it converges in about 0.2 s against about 9 s for SLSQP,
and both reach the same optimum within tolerance (``tests/test_model.py::
test_slsqp_and_lbfgsb_agree``; L-BFGS-B falls back to SLSQP by itself when its
reparametrised solution leaves a dependent alpha outside the parameter bounds).
Pass ``method=None`` to keep the method of the given config.
"""
from __future__ import annotations

import time
from dataclasses import replace
from datetime import date
from typing import Any, Final, Iterable, Mapping

import numpy as np
import pandas as pd

from src.common.competitions import COMPETITIONS, Competition
from src.common.logging_config import get_logger
from src.common.schemas import Fixture, Girone, Market, MatchPrediction, validate_matches_df
from src.common.teams import TeamRegistry
from src.engine.model import (
    DixonColesConfig,
    DixonColesEngine,
    FittedParams,
    MethodName,
    NotFittedError,
)

__all__ = [
    "DEFAULT_FIT_METHOD",
    "DIAGNOSTICS_COLUMNS",
    "RESIDUAL_COLUMNS",
    "GroupedFit",
    "fit_by_group",
    "group_of_codes",
    "split_by_group",
]

log = get_logger(__name__)

#: Optimiser used by :func:`fit_by_group` unless told otherwise (same optimum as SLSQP, ~40x faster).
DEFAULT_FIT_METHOD: Final[MethodName] = "L-BFGS-B"

#: Columns of :meth:`DixonColesEngine.residuals`, in order; :meth:`GroupedFit.residuals` appends ``fit_group`` and ``rho``.
RESIDUAL_COLUMNS: Final[tuple[str, ...]] = (
    "match_id",
    "date",
    "girone",
    "home_team",
    "away_team",
    "home_goals",
    "away_goals",
    "lam",
    "mu",
    "expected_total",
    "actual_total",
    "home_residual",
    "away_residual",
    "total_residual",
    "weight",
    "log_score_prob",
)

#: Columns of :meth:`GroupedFit.diagnostics` (one row per fitted group).
DIAGNOSTICS_COLUMNS: Final[tuple[str, ...]] = (
    "group",
    "competitions",
    "n_matches",
    "n_teams",
    "n_current_teams",
    "n_historical_teams",
    "method",
    "converged",
    "n_iter",
    "log_likelihood",
    "penalty",
    "effective_n",
    "rho",
    "degenerate_weights",
    "fit_seconds",
    "message",
)

_STRING_RESIDUAL_COLUMNS: Final[tuple[str, ...]] = ("match_id", "girone", "home_team", "away_team")


def group_of_codes(competitions: Iterable[Competition] = COMPETITIONS) -> dict[str, str]:
    """Competition code -> fit group (``{'A': 'Italy', ..., 'DI': 'Italy'}`` with the shipped file), configuration order."""
    return {comp.code: comp.fit_group for comp in competitions}


def split_by_group(matches: pd.DataFrame, competitions: Iterable[Competition] = COMPETITIONS) -> dict[str, pd.DataFrame]:
    """Validate ``matches`` and split them by fit group (groups in configuration order, absent groups left out).

    Each frame keeps the validated row order and a fresh index. ``ValueError``
    when a row belongs to a competition that ``competitions`` does not list
    (its group would be unknown).
    """
    comps = tuple(competitions)
    mapping = group_of_codes(comps)
    df = validate_matches_df(matches)
    groups = df["girone"].astype(str).map(mapping)
    unknown = groups.isna()
    if unknown.any():
        codes = sorted(set(df.loc[unknown, "girone"].astype(str)))
        raise ValueError(f"matches of competitions without a fit group {codes}: they are not among the given competitions {list(mapping)}")
    out: dict[str, pd.DataFrame] = {}
    for group in dict.fromkeys(mapping.values()):
        mask = (groups == group).to_numpy(dtype=bool)
        if mask.any():
            out[group] = df.loc[mask].reset_index(drop=True)
    return out


class GroupedFit:
    """Fitted Dixon-Coles engines, one per fit group, behind the interface of a single engine.

    Built by :func:`fit_by_group`. ``engines`` maps each fitted group (in
    configuration order) to its fitted :class:`DixonColesEngine`; ``config`` is
    the configuration every engine was fitted with; ``group_codes`` maps every
    known group to its competition codes (fitted or not); ``missing_groups``
    lists the groups that fixtures or the registry refer to but that had no
    match to fit (their fixtures raise ``KeyError`` until matches are
    harvested); ``fitted_codes`` maps each fitted group to the competitions
    that had matches in its fit; ``team_collisions`` names the teams fitted in more than one
    group (a name used by two clubs of two countries: each group predicts its
    own, but the combined :attr:`params_` refuses to merge them); ``notes``
    are Italian lines for the console and the dashboard.
    """

    def __init__(
        self,
        config: DixonColesConfig,
        engines: Mapping[str, DixonColesEngine],
        *,
        group_codes: Mapping[str, Iterable[str]],
        missing_groups: Iterable[str] = (),
        fitted_codes: Mapping[str, Iterable[str]] | None = None,
        team_collisions: Mapping[str, Iterable[str]] | None = None,
        notes: Iterable[str] = (),
    ) -> None:
        if not engines:
            raise ValueError("GroupedFit needs at least one fitted group")
        not_fitted = [g for g, e in engines.items() if not e.is_fitted]
        if not_fitted:
            raise NotFittedError(f"engines of groups {not_fitted} are not fitted")
        codes = {str(g): tuple(str(c) for c in members) for g, members in group_codes.items()}
        unknown = [g for g in engines if g not in codes]
        if unknown:
            raise ValueError(f"group_codes has no entry for the fitted groups {unknown}")
        owner: dict[str, str] = {}
        for group, members in codes.items():
            for code in members:
                if code in owner and owner[code] != group:
                    raise ValueError(f"competition {code!r} belongs to both fit groups {owner[code]!r} and {group!r}")
                owner[code] = group
        self.config: DixonColesConfig = config
        self.engines: dict[str, DixonColesEngine] = dict(engines)
        self.group_codes: dict[str, tuple[str, ...]] = codes
        self.missing_groups: tuple[str, ...] = tuple(missing_groups)
        self.team_collisions: dict[str, tuple[str, ...]] = {str(t): tuple(gs) for t, gs in (team_collisions or {}).items()}
        self.notes: tuple[str, ...] = tuple(notes)
        self._group_of_code: dict[str, str] = owner
        if fitted_codes is None:
            fitted_codes = {
                group: [code for code in codes[group] if Girone.parse(code) in engine.params_.gamma] for group, engine in self.engines.items()
            }
        self.fitted_codes: dict[str, tuple[str, ...]] = {str(g): tuple(str(c) for c in cs) for g, cs in fitted_codes.items()}
        self._combined: FittedParams | None = None

    # ------------------------------------------------------------------ groups
    @property
    def groups(self) -> tuple[str, ...]:
        """The fitted groups, in configuration order (``('Italy',)`` with the shipped file)."""
        return tuple(self.engines)

    @property
    def is_fitted(self) -> bool:
        """Always ``True`` for a constructed instance (kept for interface parity with :class:`DixonColesEngine`)."""
        return all(engine.is_fitted for engine in self.engines.values())

    @property
    def params_by_group(self) -> dict[str, FittedParams]:
        """The fitted parameters of every group (each with its own ``rho`` and home advantages)."""
        return {group: engine.params_ for group, engine in self.engines.items()}

    def group_of(self, girone: Girone | str) -> str:
        """Fit group of a competition (any spelling ``Girone.parse`` accepts); ``KeyError`` when the competition has none."""
        code = Girone.parse(girone).value
        try:
            return self._group_of_code[code]
        except KeyError as exc:
            raise KeyError(f"competition {code} has no fit group; known: {self._group_of_code}") from exc

    def engine_for(self, girone: Girone | str) -> DixonColesEngine:
        """The fitted engine of the competition's group; ``KeyError`` when that group had no match to fit."""
        group = self.group_of(girone)
        engine = self.engines.get(group)
        if engine is None:
            raise KeyError(
                f"no fit for group {group!r} (competition {Girone.parse(girone).value}): no matches of that group in the data; "
                f"fitted groups: {list(self.engines)}"
            )
        return engine

    def params_for(self, girone: Girone | str) -> FittedParams:
        """The fitted parameters used for a competition (those of its group)."""
        return self.engine_for(girone).params_

    # ------------------------------------------------------------------ params
    @property
    def params_(self) -> FittedParams:
        """Parameters of the fit: the group's own with one group, a combined summary with several.

        One group: the very :class:`FittedParams` object of its engine (no
        behaviour change against a single engine). Several groups: a new
        ``FittedParams`` with the union of ``alpha``/``beta``/``gamma``/
        ``teams_by_girone``/``historical_teams`` (each group's home advantages
        restricted to its own competitions), summed ``log_likelihood``,
        ``penalty``, ``n_matches``, ``effective_n``, ``n_iter`` and
        ``fit_seconds``, ``converged`` only when every group converged, the
        latest ``reference_date``, the methods joined with ``'/'`` and ``rho``
        averaged with the effective sample sizes as weights -- a summary for
        banners and team lookups, never used to predict. ``ValueError`` when a
        team name was fitted in two groups (see :attr:`team_collisions`).
        """
        if len(self.engines) == 1:
            return next(iter(self.engines.values())).params_
        if self._combined is None:
            self._combined = self._combine()
        return self._combined

    def _combine(self) -> FittedParams:
        if self.team_collisions:
            raise ValueError(
                f"teams fitted in more than one group cannot be merged into one parameter set: {self.team_collisions}; "
                "use params_by_group / params_for(girone)"
            )
        alpha: dict[str, float] = {}
        beta: dict[str, float] = {}
        gamma: dict[Girone, float] = {}
        teams_by_girone: dict[Girone, tuple[str, ...]] = {}
        historical: list[str] = []
        warnings: list[str] = []
        messages: list[str] = []
        by_group = self.params_by_group
        for group, params in by_group.items():
            own = set(self.group_codes.get(group, ()))
            alpha.update(params.alpha)
            beta.update(params.beta)
            gamma.update({g: v for g, v in params.gamma.items() if g.value in own})
            for g, members in params.teams_by_girone.items():
                teams_by_girone[g] = teams_by_girone.get(g, ()) + tuple(members)
            historical.extend(params.historical_teams)
            warnings.extend(f"[{group}] {w}" for w in params.fit_warnings)
            messages.append(f"{group}: {params.message}")
        total_n = sum(p.effective_n for p in by_group.values())
        if total_n > 0:
            rho = sum(p.rho * p.effective_n for p in by_group.values()) / total_n
        else:
            rho = float(np.mean([p.rho for p in by_group.values()]))
        return FittedParams(
            alpha=alpha,
            beta=beta,
            gamma=gamma,
            rho=rho,
            teams_by_girone=teams_by_girone,
            reference_date=max(p.reference_date for p in by_group.values()),
            xi=self.config.xi,
            log_likelihood=sum(p.log_likelihood for p in by_group.values()),
            penalty=sum(p.penalty for p in by_group.values()),
            n_matches=sum(p.n_matches for p in by_group.values()),
            effective_n=total_n,
            converged=all(p.converged for p in by_group.values()),
            message="; ".join(messages),
            n_iter=sum(p.n_iter for p in by_group.values()),
            fit_seconds=sum(p.fit_seconds for p in by_group.values()),
            method="/".join(dict.fromkeys(p.method for p in by_group.values())),
            max_goals=self.config.max_goals,
            degenerate_weights=any(p.degenerate_weights for p in by_group.values()),
            fit_warnings=tuple(warnings),
            historical_teams=tuple(dict.fromkeys(historical)),
        )

    # --------------------------------------------------------------- inference
    def rates(self, home: str, away: str, girone: Girone | str) -> tuple[float, float]:
        """``(lambda, mu)`` from the engine of the competition's group (see :meth:`DixonColesEngine.rates`)."""
        return self.engine_for(girone).rates(home, away, girone)

    def score_matrix(self, home: str, away: str, girone: Girone | str) -> np.ndarray:
        """Tau-corrected joint score matrix from the engine of the competition's group."""
        return self.engine_for(girone).score_matrix(home, away, girone)

    def market_probabilities(self, home: str, away: str, girone: Girone | str) -> dict[Market, dict[str, float]]:
        """Probabilities of every market from the engine of the competition's group."""
        return self.engine_for(girone).market_probabilities(home, away, girone)

    def fair_odds(self, home: str, away: str, girone: Girone | str) -> dict[Market, dict[str, float]]:
        """Fair (no-margin) decimal odds from the engine of the competition's group."""
        return self.engine_for(girone).fair_odds(home, away, girone)

    def predict(self, fixture: Fixture) -> MatchPrediction:
        """Full :class:`MatchPrediction` of one fixture by the engine of its group.

        ``KeyError`` for an unknown team or a group without matches, ``ValueError``
        for ``home == away`` (exactly what a single engine raises).
        """
        return self.engine_for(fixture.girone).predict(fixture)

    def predict_fixtures(self, fixtures: Iterable[Fixture]) -> list[MatchPrediction]:
        """Predict every fixture, preserving order (raises on the first unpredictable one, like the engine)."""
        return [self.predict(fx) for fx in fixtures]

    # --------------------------------------------------------------- reporting
    def ranking(self, girone: Girone | str | None = None, *, include_historical: bool = False) -> pd.DataFrame:
        """Strength table (columns of :meth:`DixonColesEngine.ranking`).

        With ``girone`` the ranking of that competition from its group's engine;
        without, the rankings of every group concatenated and sorted by
        ``alpha`` then ``beta`` (descending). With one group the engine's own
        table is returned unchanged. Alphas are centred per competition, so
        compare teams across competitions with care (see :mod:`src.engine.model`).
        """
        if girone is not None:
            return self.engine_for(girone).ranking(girone, include_historical=include_historical)
        if len(self.engines) == 1:
            return next(iter(self.engines.values())).ranking(None, include_historical=include_historical)
        frames = [engine.ranking(None, include_historical=include_historical) for engine in self.engines.values()]
        frame = pd.concat(frames, ignore_index=True)
        return frame.sort_values(["alpha", "beta"], ascending=[False, False], kind="stable").reset_index(drop=True)

    def residuals(self, matches: pd.DataFrame) -> pd.DataFrame:
        """Per-match diagnostics of :meth:`DixonColesEngine.residuals`, each row from its group's engine.

        Columns :data:`RESIDUAL_COLUMNS` plus ``fit_group`` and ``rho`` (the
        dependence parameter of the row's group, needed to rebuild its score
        matrix from ``lam``/``mu``). Rows are in validated order (date,
        competition, ``match_id``), as the engine returns them; rows of a group
        without a fit (or of a competition without group) carry ``NaN`` rates.
        """
        df = validate_matches_df(matches)
        groups = df["girone"].astype(str).map(self._group_of_code)
        if len(self.engines) == 1:
            group, engine = next(iter(self.engines.items()))
            if bool((groups == group).all()):
                out = engine.residuals(df)
                return out.assign(fit_group=pd.Series([group] * len(out), dtype="string"), rho=float(engine.params_.rho))
        parts: list[pd.DataFrame] = []
        covered = np.zeros(len(df), dtype=bool)
        for group, engine in self.engines.items():
            mask = (groups == group).fillna(False).to_numpy(dtype=bool)
            if not mask.any():
                continue
            covered |= mask
            part = engine.residuals(df.loc[mask])
            parts.append(part.assign(fit_group=group, rho=float(engine.params_.rho)))
        if (~covered).any():
            rest = df.loc[~covered]
            log.warning(
                "residuals: matches of groups without a fit",
                extra={"n_matches": int((~covered).sum()), "gironi": sorted(set(rest["girone"].astype(str)))},
            )
            parts.append(_unfitted_residuals(rest, groups.loc[~covered]))
        out = pd.concat(parts, ignore_index=True) if parts else _unfitted_residuals(df, groups)
        position = pd.Series(np.arange(len(df)), index=df["match_id"].astype(str).to_numpy())
        order = out["match_id"].astype(str).map(position).to_numpy(dtype=np.int64)
        out = out.iloc[np.argsort(order, kind="stable")].reset_index(drop=True)
        for column in _STRING_RESIDUAL_COLUMNS:
            out[column] = out[column].astype("string")
        out["fit_group"] = out["fit_group"].astype("string")
        out["rho"] = out["rho"].astype("float64")
        return out[[*RESIDUAL_COLUMNS, "fit_group", "rho"]]

    def diagnostics(self) -> pd.DataFrame:
        """One row per fitted group with the fit statistics (:data:`DIAGNOSTICS_COLUMNS`)."""
        rows: list[dict[str, Any]] = []
        for group, params in self.params_by_group.items():
            rows.append(
                {
                    "group": group,
                    "competitions": ", ".join(self.fitted_codes.get(group, ())),
                    "n_matches": int(params.n_matches),
                    "n_teams": len(params.teams),
                    "n_current_teams": len(params.current_teams),
                    "n_historical_teams": len(params.historical_teams),
                    "method": params.method,
                    "converged": bool(params.converged),
                    "n_iter": int(params.n_iter),
                    "log_likelihood": float(params.log_likelihood),
                    "penalty": float(params.penalty),
                    "effective_n": float(params.effective_n),
                    "rho": float(params.rho),
                    "degenerate_weights": bool(params.degenerate_weights),
                    "fit_seconds": float(params.fit_seconds),
                    "message": params.message,
                }
            )
        frame = pd.DataFrame(rows, columns=list(DIAGNOSTICS_COLUMNS))
        for column in ("group", "competitions", "method", "message"):
            frame[column] = frame[column].astype("string")
        return frame

    def describe(self) -> list[str]:
        """Italian lines for notes: one per fitted group, then :attr:`notes` (missing groups, name collisions)."""
        lines: list[str] = []
        for row in self.diagnostics().itertuples(index=False):
            lines.append(
                f"Gruppo di stima {row.group} ({row.competitions}): {row.n_matches} partite, {row.n_teams} squadre "
                f"({row.n_historical_teams} storiche), metodo {row.method}, convergenza {'sì' if row.converged else 'no'}, "
                f"{row.n_iter} iterazioni in {row.fit_seconds:.2f} s, rho={row.rho:+.4f}."
            )
        lines.extend(self.notes)
        return lines

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"GroupedFit(groups={list(self.engines)}, method={self.config.method!r}, missing_groups={list(self.missing_groups)})"


def _unfitted_residuals(df: pd.DataFrame, groups: pd.Series) -> pd.DataFrame:
    """Residual rows with ``NaN`` rates for matches no engine covers (same columns as the fitted rows)."""
    n = len(df)
    nan = np.full(n, np.nan)
    x = df["home_goals"].to_numpy(dtype=float)
    y = df["away_goals"].to_numpy(dtype=float)
    return pd.DataFrame(
        {
            "match_id": df["match_id"].astype("string").to_numpy(),
            "date": df["date"].to_numpy(),
            "girone": df["girone"].astype("string").to_numpy(),
            "home_team": df["home_team"].astype("string").to_numpy(),
            "away_team": df["away_team"].astype("string").to_numpy(),
            "home_goals": df["home_goals"].to_numpy(),
            "away_goals": df["away_goals"].to_numpy(),
            "lam": nan,
            "mu": nan,
            "expected_total": nan,
            "actual_total": x + y,
            "home_residual": nan,
            "away_residual": nan,
            "total_residual": nan,
            "weight": nan,
            "log_score_prob": nan,
            "fit_group": groups.astype("string").to_numpy(),
            "rho": nan,
        }
    )


def _resolve_config(config: DixonColesConfig | None, method: MethodName | None) -> DixonColesConfig:
    cfg = config or DixonColesConfig()
    if method is not None and cfg.method != method:
        cfg = replace(cfg, method=method)
    return cfg


def _warm_start_for(warm_start: "GroupedFit | Mapping[str, FittedParams] | None", group: str) -> FittedParams | None:
    if warm_start is None:
        return None
    if isinstance(warm_start, GroupedFit):
        engine = warm_start.engines.get(group)
        return engine.params_ if engine is not None else None
    return warm_start.get(group)


def fit_by_group(
    matches: pd.DataFrame,
    *,
    registry: TeamRegistry | None = None,
    config: DixonColesConfig | None = None,
    fixtures: Iterable[Fixture] | None = None,
    method: MethodName | None = DEFAULT_FIT_METHOD,
    reference_date: date | None = None,
    warm_start: "GroupedFit | Mapping[str, FittedParams] | None" = None,
    competitions: Iterable[Competition] = COMPETITIONS,
) -> GroupedFit:
    """Fit one :class:`DixonColesEngine` per fit group present in ``matches`` and wrap them in a :class:`GroupedFit`.

    ``matches`` (canonical match frame, history included) is validated and split
    by the fit group of its ``girone`` (:func:`split_by_group`; the groups of
    ``competitions``, default the enabled ones). Every group is fitted with the
    same ``config`` (default :class:`DixonColesConfig`) and the same
    ``registry`` (the perimeter: a registry team only matters to the group it
    plays in). ``method`` overrides ``config.method`` (default ``'L-BFGS-B'``,
    see the module docstring; ``None`` keeps the config's).
    ``reference_date`` is passed to every engine; ``None`` lets each group decay
    from its own latest match. ``warm_start`` (a previous :class:`GroupedFit`
    or ``{group: FittedParams}``) seeds each group's optimiser, as in the
    walk-forward backtest.

    ``fixtures`` and ``registry`` are only *checked*: groups they refer to (by
    fixture competition, or by current registry teams) that have no match to
    fit are listed in :attr:`GroupedFit.missing_groups`, logged and described
    in :attr:`GroupedFit.notes`. A team name fitted in two groups is logged and
    recorded in :attr:`GroupedFit.team_collisions`.

    ``ValueError`` for an empty frame or a competition without a fit group.
    """
    started = time.perf_counter()
    comps = tuple(competitions)
    codes_by_group: dict[str, list[str]] = {}
    for comp in comps:
        codes_by_group.setdefault(comp.fit_group, []).append(comp.code)
    group_codes = {group: tuple(codes) for group, codes in codes_by_group.items()}
    cfg = _resolve_config(config, method)
    split = split_by_group(matches, comps)
    if not split:
        raise ValueError("no valid matches to fit")

    engines: dict[str, DixonColesEngine] = {}
    for group, frame in split.items():
        engine = DixonColesEngine(cfg)
        engine.fit(frame, reference_date=reference_date, warm_start=_warm_start_for(warm_start, group), registry=registry)
        engines[group] = engine
        params = engine.params_
        log.info(
            "fit group fitted",
            extra={
                "group": group,
                "competitions": sorted(set(frame["girone"].astype(str))),
                "n_matches": params.n_matches,
                "n_teams": len(params.teams),
                "method": params.method,
                "converged": params.converged,
                "fit_seconds": round(params.fit_seconds, 3),
            },
        )

    mapping = group_of_codes(comps)
    referenced: dict[str, str] = {}
    for fx in fixtures or ():
        group = mapping.get(fx.girone.value)
        if group is not None and group not in engines:
            referenced.setdefault(group, "partite in programma")
    if registry is not None:
        for g in registry.gironi:
            group = mapping.get(g.value)
            if group is not None and group not in engines:
                referenced.setdefault(group, "squadre nel dizionario")
    missing = tuple(g for g in group_codes if g in referenced)
    notes: list[str] = []
    if missing:
        log.warning(
            "fit groups without matches: their fixtures cannot be predicted until their matches are harvested",
            extra={"groups": list(missing), "fitted_groups": list(engines)},
        )
        for group in missing:
            notes.append(
                f"Gruppo di stima {group} ({', '.join(group_codes[group])}): {referenced[group]} ma nessuna partita giocata nei dati, "
                "nessuna stima finché le sue partite non vengono scaricate."
            )

    owners: dict[str, list[str]] = {}
    for group, engine in engines.items():
        for team in engine.params_.alpha:
            owners.setdefault(team, []).append(group)
    collisions = {team: tuple(groups) for team, groups in owners.items() if len(groups) > 1}
    if collisions:
        log.warning("team names fitted in more than one group", extra={"teams": {t: list(g) for t, g in collisions.items()}})
        notes.append(
            f"{len(collisions)} nomi di squadra stimati in più gruppi ({', '.join(sorted(collisions)[:5])}"
            f"{', ...' if len(collisions) > 5 else ''}): ogni gruppo usa la propria stima."
        )

    fitted_codes = {group: tuple(c for c in group_codes[group] if c in set(split[group]["girone"].astype(str))) for group in engines}
    result = GroupedFit(
        cfg,
        engines,
        group_codes=group_codes,
        missing_groups=missing,
        fitted_codes=fitted_codes,
        team_collisions=collisions,
        notes=notes,
    )
    log.info(
        "grouped Dixon-Coles fit finished",
        extra={
            "groups": list(engines),
            "missing_groups": list(missing),
            "method": cfg.method,
            "n_matches": {g: e.params_.n_matches for g, e in engines.items()},
            "seconds": round(time.perf_counter() - started, 3),
        },
    )
    return result
