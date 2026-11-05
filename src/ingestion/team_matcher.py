"""Fuzzy team-name canonicalisation on top of :class:`~src.common.teams.TeamRegistry`.

External feeds (SofaScore, FotMob, bookmakers) spell the same club in many
ways: ``"L.R. Vicenza"``, ``"LR Vicenza"``, ``"Vicenza"``. The engine needs one
canonical spelling per club, so every ingested name goes through
:class:`TeamMatcher`, which resolves it in three steps:

1. **exact** -- the normalised spelling equals a normalised canonical name;
2. **alias** -- the normalised spelling is in the registry alias table;
3. **fuzzy** -- similarity search (``thefuzz``) over the normalised alias-table
   keys, optionally restricted to one girone, accepted only above a threshold
   and only when the best candidate is not ambiguous.

When ``thefuzz`` is not importable the module falls back to a
:mod:`difflib`-based scorer (``DIFFLIB_FALLBACK`` is then ``True``); the flag
is consulted on every lookup so tests can exercise both code paths.
"""
from __future__ import annotations

import difflib
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, replace
from typing import Literal

import pandas as pd

from src.common.logging_config import get_logger
from src.common.schemas import Girone
from src.common.teams import TeamRegistry, normalize_name

try:  # guarded import: the matcher must keep working without thefuzz
    from thefuzz import fuzz as _thefuzz_fuzz
    from thefuzz import process as _thefuzz_process

    DIFFLIB_FALLBACK: bool = False
except ImportError:  # pragma: no cover - depends on the environment
    _thefuzz_fuzz = None
    _thefuzz_process = None
    DIFFLIB_FALLBACK = True

__all__ = [
    "DIFFLIB_FALLBACK",
    "MatchMethod",
    "MatchResult",
    "TeamMatcher",
    "difflib_extract",
    "difflib_score",
]

log = get_logger(__name__)

MatchMethod = Literal["exact", "alias", "fuzzy", "none"]
Scorer = Callable[[str, str], int]

#: How many ranked candidates are kept on a :class:`MatchResult`.
MAX_CANDIDATES = 5
#: How many raw alias keys are pulled from the similarity search before
#: collapsing them to distinct canonical names.
_EXTRACT_LIMIT = 12


@dataclass(frozen=True, slots=True)
class MatchResult:
    """Outcome of one canonicalisation lookup.

    ``candidates`` lists ``(canonical, score)`` pairs, best first, one entry per
    distinct canonical name; it is populated for fuzzy and ambiguous outcomes so
    callers can log or display what the matcher considered.
    """

    query: str
    canonical: str | None
    score: int
    method: MatchMethod
    girone: Girone | None
    candidates: tuple[tuple[str, int], ...] = ()

    @property
    def matched(self) -> bool:
        """``True`` when a canonical name was found."""
        return self.canonical is not None


def difflib_score(query: str, choice: str) -> int:
    """Similarity in ``0..100`` computed with :class:`difflib.SequenceMatcher`."""
    return int(round(100.0 * difflib.SequenceMatcher(None, query, choice).ratio()))


def difflib_extract(
    query: str,
    choices: Iterable[str],
    *,
    limit: int = _EXTRACT_LIMIT,
    scorer: Scorer = difflib_score,
) -> list[tuple[str, int]]:
    """Rank ``choices`` against ``query`` (best first). Pure stdlib fallback."""
    scored = sorted(((choice, scorer(query, choice)) for choice in choices), key=lambda item: (-item[1], item[0]))
    return scored[: max(limit, 0)]


def _resolve_scorer(name: str) -> Scorer | None:
    """Look ``name`` up in :mod:`thefuzz.fuzz`; ``None`` when thefuzz is absent."""
    if _thefuzz_fuzz is None:
        return None
    scorer = getattr(_thefuzz_fuzz, name, None)
    if scorer is None or not callable(scorer):
        raise ValueError(f"unknown thefuzz scorer {name!r}")
    return scorer


class TeamMatcher:
    """Map raw team spellings to the canonical names of a :class:`TeamRegistry`.

    Parameters
    ----------
    registry:
        Season dictionary with canonical names and aliases.
    threshold:
        Minimum similarity (0-100) for a fuzzy hit.
    ambiguity_margin:
        When the two best candidates map to *different* canonical names and
        their scores differ by less than this margin the lookup is declared
        ambiguous and returns ``None``.
    scorer:
        Name of a scorer in :mod:`thefuzz.fuzz` (``token_set_ratio`` by
        default). Ignored on the difflib fallback path.

    Results are memoised per ``(normalised name, girone)``.
    """

    def __init__(
        self,
        registry: TeamRegistry,
        *,
        threshold: int = 86,
        ambiguity_margin: int = 3,
        scorer: str = "token_set_ratio",
    ) -> None:
        if not 0 <= int(threshold) <= 100:
            raise ValueError("threshold must be within 0..100")
        if int(ambiguity_margin) < 0:
            raise ValueError("ambiguity_margin must be >= 0")
        self.registry = registry
        self.threshold = int(threshold)
        self.ambiguity_margin = int(ambiguity_margin)
        self.scorer_name = scorer
        self._scorer_fn: Scorer | None = _resolve_scorer(scorer)
        self._alias_table: dict[str, str] = dict(registry.alias_table())
        self._canonical_by_norm: dict[str, str] = {normalize_name(c): c for c in registry.canonical_names}
        self._girone_of: dict[str, Girone] = {
            canonical: girone for girone, teams in registry.teams_by_girone.items() for canonical in teams
        }
        keys_by_girone: dict[Girone | None, tuple[str, ...]] = {None: tuple(self._alias_table)}
        for girone in Girone:
            members = set(registry.teams_by_girone.get(girone, ()))
            keys_by_girone[girone] = tuple(k for k, c in self._alias_table.items() if c in members)
        self._keys_by_girone = keys_by_girone
        self._cache: dict[tuple[str, Girone | None], MatchResult] = {}

    # ------------------------------------------------------------------ helpers
    def _in_girone(self, canonical: str, girone: Girone | None) -> bool:
        return girone is None or self._girone_of.get(canonical) == girone

    def _extract(self, query: str, keys: Sequence[str], limit: int) -> list[tuple[str, int]]:
        """Rank alias keys for ``query`` with thefuzz or, when unavailable, difflib."""
        if DIFFLIB_FALLBACK or _thefuzz_process is None or self._scorer_fn is None:
            return difflib_extract(query, keys, limit=limit)
        ranked = _thefuzz_process.extract(query, list(keys), scorer=self._scorer_fn, limit=limit)
        return [(str(choice), int(score)) for choice, score, *_ in ranked]

    def _collapse(self, ranked: Iterable[tuple[str, int]]) -> tuple[tuple[str, int], ...]:
        """Best score per distinct canonical, sorted best first."""
        best: dict[str, int] = {}
        for key, score in ranked:
            canonical = self._alias_table.get(key)
            if canonical is None:
                continue
            if score > best.get(canonical, -1):
                best[canonical] = score
        ordered = sorted(best.items(), key=lambda item: (-item[1], item[0]))
        return tuple(ordered[:MAX_CANDIDATES])

    def _resolve(self, query: str, norm: str, girone: Girone | None) -> MatchResult:
        if not norm:
            return MatchResult(query, None, 0, "none", girone)

        canonical = self._canonical_by_norm.get(norm)
        if canonical is not None and self._in_girone(canonical, girone):
            return MatchResult(query, canonical, 100, "exact", girone, ((canonical, 100),))

        canonical = self._alias_table.get(norm)
        if canonical is not None and self._in_girone(canonical, girone):
            return MatchResult(query, canonical, 100, "alias", girone, ((canonical, 100),))

        keys = self._keys_by_girone.get(girone, ())
        candidates = self._collapse(self._extract(norm, keys, _EXTRACT_LIMIT)) if keys else ()
        if not candidates:
            return MatchResult(query, None, 0, "none", girone, ())

        best_name, best_score = candidates[0]
        if best_score < self.threshold:
            return MatchResult(query, None, best_score, "none", girone, candidates)
        if len(candidates) > 1 and (best_score - candidates[1][1]) < self.ambiguity_margin:
            log.warning(
                "ambiguous team name",
                extra={"query": query, "girone": girone.value if girone else None, "candidates": list(candidates[:2])},
            )
            return MatchResult(query, None, best_score, "none", girone, candidates)
        return MatchResult(query, best_name, best_score, "fuzzy", girone, candidates)

    # ------------------------------------------------------------------- public
    def match(self, name: str, girone: Girone | str | None = None) -> MatchResult:
        """Resolve ``name`` to a canonical team of ``girone`` (or of any girone)."""
        g = Girone.parse(girone) if girone is not None else None
        query = "" if name is None else str(name)
        norm = normalize_name(query) if query.strip() else ""
        key = (norm, g)
        cached = self._cache.get(key)
        if cached is None:
            cached = self._resolve(query, norm, g)
            self._cache[key] = cached
        return cached if cached.query == query else replace(cached, query=query)

    def map_names(self, names: Iterable[str], girone: Girone | str | None = None) -> dict[str, str | None]:
        """Canonical name (or ``None``) for every distinct raw spelling in ``names``."""
        return {name: self.match(name, girone).canonical for name in dict.fromkeys(names)}

    def map_frame(
        self,
        df: pd.DataFrame,
        columns: Sequence[str] = ("home_team", "away_team"),
        girone_column: str = "girone",
    ) -> tuple[pd.DataFrame, dict[str, int]]:
        """Canonicalise team columns on a copy of ``df``.

        The lookup is restricted to the girone found in ``girone_column`` when
        that column exists. Unmapped spellings keep the raw text and are
        reported with their occurrence counts in the returned dict. Missing
        values are left untouched and not counted.
        """
        missing = [c for c in columns if c not in df.columns]
        if missing:
            raise ValueError(f"map_frame: columns not in frame: {missing}")
        out = df.copy()
        unmapped: dict[str, int] = {}
        gironi: list[Girone | None]
        if girone_column in out.columns:
            gironi = [None if pd.isna(g) else Girone.parse(g) for g in out[girone_column].tolist()]
        else:
            gironi = [None] * len(out)
        for column in columns:
            values: list[str | None] = []
            for raw, g in zip(out[column].tolist(), gironi, strict=True):
                if raw is None or (not isinstance(raw, str) and pd.isna(raw)):
                    values.append(None)
                    continue
                result = self.match(str(raw), g)
                if result.canonical is None:
                    unmapped[str(raw)] = unmapped.get(str(raw), 0) + 1
                    values.append(str(raw))
                else:
                    values.append(result.canonical)
            out[column] = pd.Series(values, index=out.index, dtype="string")
        if unmapped:
            log.warning("unmapped team names", extra={"count": len(unmapped), "names": sorted(unmapped)})
        return out, unmapped

    def clear_cache(self) -> None:
        """Drop memoised lookups (e.g. after swapping the fallback flag in tests)."""
        self._cache.clear()

    @property
    def cache_size(self) -> int:
        """Number of memoised ``(name, girone)`` lookups."""
        return len(self._cache)
