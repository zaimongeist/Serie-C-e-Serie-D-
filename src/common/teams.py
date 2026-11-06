"""Canonical team dictionary (per season, per girone) + alias lookup.

The dictionary lives in ``config/teams_<season>.json``. Code never hard-codes a
team name: it asks :class:`TeamRegistry` for the canonical spelling.

Seasons and perimeter
---------------------
The registry of :data:`DEFAULT_SEASON` (``config/teams_2026_27.json``, the
official Lega Pro composition of the three gironi) is the *perimeter*: its 60
teams are the only ones that are ranked, predicted and priced. Older season
files (``teams_2025_26.json``, ...) are kept as **historical alias sources**:
:meth:`TeamRegistry.with_historical_aliases` merges their spellings into the
current registry so that harvested history keeps consistent names, while
:meth:`TeamRegistry.is_current_team` tells the current teams apart from the
historical ones (relegated, promoted, excluded clubs) that only appear in past
seasons. :func:`list_seasons` enumerates the season files available.
"""
from __future__ import annotations

import json
import re
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Mapping

from .paths import CONFIG_DIR
from .schemas import Girone

__all__ = [
    "DEFAULT_SEASON",
    "TeamRegistry",
    "list_seasons",
    "load_registry",
    "normalize_name",
    "season_file_stem",
]

#: Season of the default registry (file stem suffix of ``config/teams_<season>.json``).
DEFAULT_SEASON = "2026_27"

_SEASON_FILE_RE = re.compile(r"^teams_(\d{4}_\d{2})\.json$")


def normalize_name(name: str) -> str:
    """Lower-case, strip accents, drop punctuation and legal suffixes.

    ``"U.S. Pergolettese 1932"`` -> ``"pergolettese"``.
    """
    text = unicodedata.normalize("NFKD", str(name))
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    text = text.lower().replace(".", "")  # "U.S." -> "us", "L.R." -> "lr"
    for ch in ",'’-_/()&":
        text = text.replace(ch, " ")
    tokens = [t for t in text.split() if t]
    drop = {"fc", "ac", "as", "us", "ss", "ssd", "asd", "sef", "uc", "acf", "calcio", "club", "srl", "spa", "lr", "bc"}
    kept = [t for t in tokens if t not in drop and len(t) > 1 and not (t.isdigit() and len(t) == 4)]
    return " ".join(kept) if kept else " ".join(tokens)


def season_file_stem(season: str) -> str:
    """``'2026/27'`` or ``'2026_27'`` -> ``'2026_27'`` (the suffix of the registry file name)."""
    return str(season).strip().replace("/", "_").replace("-", "_")


@dataclass(frozen=True)
class TeamRegistry:
    """Canonical teams of one season, grouped by girone, plus an alias table.

    ``aliases`` maps a *normalised* spelling to a canonical name. The canonical
    names of ``teams_by_girone`` are the **current** teams; ``historical_names``
    lists further canonical names reachable through the alias table that are
    *not* current (teams of previous seasons merged in with
    :meth:`with_historical_aliases`). An alias of a historical team resolves to
    the historical canonical spelling, never to a current team, so history keeps
    consistent names and :meth:`is_current_team` stays unambiguous.
    """

    season: str
    teams_by_girone: Mapping[Girone, tuple[str, ...]]
    aliases: Mapping[str, str] = field(default_factory=dict)  # normalized alias -> canonical
    historical_names: tuple[str, ...] = ()

    # ----------------------------------------------------------------- views
    @property
    def canonical_names(self) -> tuple[str, ...]:
        """Current teams in girone order (A, B, C), file order within a girone."""
        return tuple(t for g in Girone for t in self.teams_by_girone.get(g, ()))

    @property
    def n_teams(self) -> int:
        """Number of current teams."""
        return len(self.canonical_names)

    @property
    def season_label(self) -> str:
        """Season as a public label, ``'2026/27'`` (whatever the file spelled)."""
        return str(self.season).strip().replace("_", "/").replace("-", "/")

    def girone_of(self, canonical: str) -> Girone:
        for g, teams in self.teams_by_girone.items():
            if canonical in teams:
                return g
        raise KeyError(canonical)

    def teams(self, girone: "Girone | str") -> tuple[str, ...]:
        return self.teams_by_girone[Girone.parse(girone)]

    def alias_table(self) -> dict[str, str]:
        """normalized spelling -> canonical name (canonicals included)."""
        table = {normalize_name(c): c for c in self.canonical_names}
        table.update(self.aliases)
        return table

    def exact(self, name: str) -> str | None:
        """Exact (normalised) lookup; ``None`` when the spelling is unknown."""
        return self.alias_table().get(normalize_name(name))

    def is_current_team(self, name: str) -> bool:
        """True when ``name`` (any known spelling) resolves to a team of this season's perimeter.

        Historical teams merged in through :meth:`with_historical_aliases` and
        unknown spellings both yield ``False``.
        """
        canonical = self.exact(name)
        return canonical is not None and canonical in set(self.canonical_names)

    # -------------------------------------------------------------- builders
    @classmethod
    def from_mapping(cls, season: str, gironi: Mapping[str, Mapping[str, Iterable[str]]]) -> "TeamRegistry":
        teams_by_girone: dict[Girone, tuple[str, ...]] = {}
        aliases: dict[str, str] = {}
        for g_key, teams in gironi.items():
            g = Girone.parse(g_key)
            teams_by_girone[g] = tuple(teams.keys())
            for canonical, alias_list in teams.items():
                aliases[normalize_name(canonical)] = canonical
                for alias in alias_list:
                    aliases[normalize_name(alias)] = canonical
        return cls(season=season, teams_by_girone=teams_by_girone, aliases=aliases)

    @classmethod
    def from_json(cls, path: Path) -> "TeamRegistry":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls.from_mapping(payload.get("season", path.stem), payload["gironi"])

    @classmethod
    def from_teams(cls, season: str, teams_by_girone: Mapping[str, Iterable[str]]) -> "TeamRegistry":
        """Build a registry with no aliases (e.g. for simulated leagues)."""
        return cls.from_mapping(season, {g: {t: [] for t in ts} for g, ts in teams_by_girone.items()})

    def with_historical_aliases(self, *older: "TeamRegistry") -> "TeamRegistry":
        """Same current teams, alias table extended with the spellings of older seasons.

        For every canonical team of an older registry:

        * if it still exists (its canonical spelling, or one of its aliases,
          resolves to a current team) its aliases are added for that current team,
          so ``'Juventus U23'`` from an old file still maps to ``'Juventus Next Gen'``;
        * otherwise the team is **historical**: its old canonical name and its
          aliases map to the *old canonical spelling itself* (``'Ternana Calcio'``
          -> ``'Ternana'``), which is also recorded in ``historical_names``.

        Spellings already known to the current registry are never overridden, and
        among several older registries the first argument wins, so pass them from
        the most recent to the oldest. The result is a new registry; ``self`` is
        unchanged.
        """
        current = set(self.canonical_names)
        table = self.alias_table()
        historical: list[str] = list(self.historical_names)
        for old in older:
            old_aliases_by_canonical: dict[str, list[str]] = {c: [] for c in old.canonical_names}
            for norm, canonical in old.aliases.items():
                old_aliases_by_canonical.setdefault(canonical, []).append(norm)
            for old_canonical, norms in old_aliases_by_canonical.items():
                spellings = [normalize_name(old_canonical), *norms]
                target: str | None = None
                for spelling in spellings:
                    resolved = table.get(spelling)
                    if resolved is not None and resolved in current:
                        target = resolved
                        break
                if target is None:
                    target = old_canonical
                    if old_canonical not in current and old_canonical not in historical:
                        historical.append(old_canonical)
                for spelling in spellings:
                    table.setdefault(spelling, target)
        return TeamRegistry(
            season=self.season,
            teams_by_girone=dict(self.teams_by_girone),
            aliases=dict(table),
            historical_names=tuple(historical),
        )


def list_seasons(config_dir: Path = CONFIG_DIR) -> list[str]:
    """Season stems of the registry files in ``config_dir`` (``teams_<season>.json``), sorted ascending.

    ``['2025_26', '2026_27']`` with the shipped configuration.
    """
    seasons: list[str] = []
    for path in Path(config_dir).glob("teams_*.json"):
        match = _SEASON_FILE_RE.match(path.name)
        if match is not None:
            seasons.append(match.group(1))
    return sorted(seasons)


def load_registry(season: str = DEFAULT_SEASON, config_dir: Path = CONFIG_DIR, *, with_history: bool = False) -> TeamRegistry:
    """Load ``config_dir/teams_<season>.json`` (``'2026/27'`` and ``'2026_27'`` both work).

    With ``with_history=True`` the aliases of every *older* season file in
    ``config_dir`` are merged in through :meth:`TeamRegistry.with_historical_aliases`
    (most recent first), so harvested history can be normalised with one registry
    whose current teams are still exactly the perimeter of ``season``.
    """
    stem = season_file_stem(season)
    path = Path(config_dir) / f"teams_{stem}.json"
    if not path.exists():
        raise FileNotFoundError(f"no team dictionary for season {season!r} at {path}")
    registry = TeamRegistry.from_json(path)
    if not with_history:
        return registry
    older = [s for s in list_seasons(config_dir) if s < stem]
    return registry.with_historical_aliases(*(TeamRegistry.from_json(Path(config_dir) / f"teams_{s}.json") for s in reversed(older)))
