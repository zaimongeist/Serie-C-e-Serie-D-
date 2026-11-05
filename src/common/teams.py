"""Canonical team dictionary (per season, per girone) + alias lookup.

The dictionary lives in ``config/teams_<season>.json``. Code never hard-codes a
team name: it asks :class:`TeamRegistry` for the canonical spelling.
"""
from __future__ import annotations

import json
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Mapping

from .paths import CONFIG_DIR
from .schemas import Girone

__all__ = ["TeamRegistry", "normalize_name", "load_registry", "DEFAULT_SEASON"]

DEFAULT_SEASON = "2025_26"


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


@dataclass(frozen=True)
class TeamRegistry:
    season: str
    teams_by_girone: Mapping[Girone, tuple[str, ...]]
    aliases: Mapping[str, str] = field(default_factory=dict)  # normalized alias -> canonical

    # ----------------------------------------------------------------- views
    @property
    def canonical_names(self) -> tuple[str, ...]:
        return tuple(t for g in Girone for t in self.teams_by_girone.get(g, ()))

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


def load_registry(season: str = DEFAULT_SEASON, config_dir: Path = CONFIG_DIR) -> TeamRegistry:
    path = Path(config_dir) / f"teams_{season.replace('/', '_')}.json"
    if not path.exists():
        raise FileNotFoundError(f"no team dictionary for season {season!r} at {path}")
    return TeamRegistry.from_json(path)
