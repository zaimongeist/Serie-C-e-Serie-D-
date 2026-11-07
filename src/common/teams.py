"""Canonical team dictionary (per season, per competition) + alias lookup.

The dictionary lives in ``config/teams_<season>.json``: under ``"gironi"`` one
object per competition code of ``config/competitions.toml`` (``A``, ``B``, ``C``
for the Serie C gironi, ``SB`` for Serie B, ``DA`` .. ``DI`` for the Serie D
gironi, more as leagues are added), each mapping a canonical team name to its
aliases. Code never hard-codes a team name: it asks :class:`TeamRegistry` for
the canonical spelling. The registry works with any number of competitions: it
iterates the dynamic :class:`src.common.schemas.Girone` enum and never assumes
A/B/C.

A configured competition may be **absent** from the file or hold an **empty**
object (the Serie D gironi ship empty: the FootyStats harvester fills them from
``/league-teams`` on its first run). Both cases are the same registry state:
:meth:`TeamRegistry.teams` answers ``()``, :attr:`TeamRegistry.gironi` lists only
the competitions that have teams, :meth:`TeamRegistry.missing_competitions`
names the empty ones and :attr:`TeamRegistry.perimeter_label` counts only the
teams present.

Seasons and perimeter
---------------------
The registry of :data:`DEFAULT_SEASON` (``config/teams_2026_27.json``: the
official Lega Pro composition of the three Serie C gironi plus the Serie B
organico) is the *perimeter*: its 80 teams are the only ones that are ranked,
predicted and priced (:attr:`TeamRegistry.perimeter_label` spells it out:
``'Serie C 2026/27 Gironi A/B/C + Serie B (80 squadre)'``). Older season files
(``teams_2025_26.json``, ...) are kept as **historical alias sources**:
:meth:`TeamRegistry.with_historical_aliases` merges their spellings into the
current registry so that harvested history keeps consistent names, while
:meth:`TeamRegistry.is_current_team` tells the current teams apart from the
historical ones (relegated, promoted, excluded clubs) that only appear in past
seasons. A club keeps **one canonical name across divisions**: Pescara is
``'Pescara'`` whether it plays Serie B (2025/26) or Serie C girone B (2026/27),
and the registry of the current season decides its competition.
:func:`list_seasons` enumerates the season files available.
"""
from __future__ import annotations

import json
import re
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Mapping

from .competitions import describe_competitions, family_label
from .paths import CONFIG_DIR
from .schemas import Girone

__all__ = [
    "DEFAULT_SEASON",
    "PerimeterLabel",
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


class PerimeterLabel(str):
    """The perimeter description of a registry: a plain ``str`` that can also be *called* with a family.

    :attr:`TeamRegistry.perimeter_label` has always been a string property
    (``'Serie C 2026/27 Gironi A/B/C + Serie B (80 squadre)'``) and every banner
    uses it as such. The user-facing surfaces are split by family, so the same
    attribute can now be narrowed: ``registry.perimeter_label('Serie D')`` (or
    ``registry.perimeter_label(family='Serie D')``) returns the description of
    that family alone, exactly like :meth:`TeamRegistry.perimeter_label_for`.
    Equality, hashing, formatting and every other ``str`` behaviour are unchanged.
    """

    __slots__ = ("_registry",)

    def __new__(cls, text: str, registry: "TeamRegistry") -> "PerimeterLabel":
        label = super().__new__(cls, text)
        label._registry = registry
        return label

    def __call__(self, family: str | None = None) -> str:
        """Perimeter description restricted to ``family`` (``None``: the whole registry, i.e. this very text)."""
        return self._registry.perimeter_label_for(family)

    def __reduce__(self) -> tuple[type[str], tuple[str]]:
        """Pickle as a plain string (the registry reference is a convenience, not state)."""
        return (str, (str(self),))


@dataclass(frozen=True)
class TeamRegistry:
    """Canonical teams of one season, grouped by competition, plus an alias table.

    ``teams_by_girone`` maps a :class:`Girone` (competition) to its canonical
    names; a registry may hold any subset of the configured competitions (the
    mini demo has three, the default perimeter four with teams plus the nine
    Serie D gironi awaiting their auto-init). ``aliases`` maps a *normalised*
    spelling to a canonical name. The canonical names of ``teams_by_girone`` are
    the **current** teams; ``historical_names`` lists further canonical names
    reachable through the alias table that are *not* current (teams of previous
    seasons merged in with :meth:`with_historical_aliases`). An alias of a
    historical team resolves to the historical canonical spelling, never to a
    current team, so history keeps consistent names and :meth:`is_current_team`
    stays unambiguous.
    """

    season: str
    teams_by_girone: Mapping[Girone, tuple[str, ...]]
    aliases: Mapping[str, str] = field(default_factory=dict)  # normalized alias -> canonical
    historical_names: tuple[str, ...] = ()

    # ----------------------------------------------------------------- views
    @property
    def gironi(self) -> tuple[Girone, ...]:
        """Competitions that have at least one team, in configuration order (empty or absent ones are left out)."""
        return tuple(g for g in Girone if self.teams_by_girone.get(g))

    def missing_competitions(self) -> tuple[Girone, ...]:
        """Configured competitions with no team in this registry, in configuration order.

        ``(Girone.DA, ..., Girone.DI)`` for the shipped 2026/27 dictionary until the
        FootyStats harvester initialises the Serie D gironi; ``()`` when every
        competition of ``config/competitions.toml`` has teams.
        """
        present = set(self.gironi)
        return tuple(g for g in Girone if g not in present)

    def families(self) -> tuple[str, ...]:
        """Families that have at least one team, in configuration order (``('Serie C', 'Serie B')`` before the Serie D auto-init)."""
        return tuple(dict.fromkeys(g.family for g in self.gironi))

    @property
    def canonical_names(self) -> tuple[str, ...]:
        """Current teams in competition order (``config/competitions.toml``), file order within a competition."""
        return tuple(t for g in Girone for t in self.teams_by_girone.get(g, ()))

    @property
    def n_teams(self) -> int:
        """Number of current teams."""
        return len(self.canonical_names)

    @property
    def season_label(self) -> str:
        """Season as a public label, ``'2026/27'`` (whatever the file spelled)."""
        return str(self.season).strip().replace("_", "/").replace("-", "/")

    @property
    def perimeter_label(self) -> PerimeterLabel:
        """Italian description of the perimeter derived from the configuration, never hard-coded.

        ``'Serie C 2026/27 Gironi A/B/C + Serie B (80 squadre)'`` for the default
        registry (the empty Serie D gironi are not named and not counted),
        ``'Serie C 2026/27 Gironi A/B/C (14 squadre)'`` for the mini demo; only
        the competitions that have teams are named (see
        :func:`src.common.competitions.describe_competitions`). The value is a
        :class:`PerimeterLabel`: a ``str`` that, called with a family
        (``registry.perimeter_label('Serie D')``), gives the description of that
        family alone (see :meth:`perimeter_label_for`).
        """
        return PerimeterLabel(self.perimeter_label_for(None), self)

    def perimeter_label_for(self, family: str | None = None) -> str:
        """Perimeter description of one family, or of the whole registry with ``family=None``.

        Only the competitions of ``family`` that have teams are named and only
        their teams are counted: ``'Serie D 2026/27 Gironi A-I (162 squadre)'``
        once the Serie D gironi are initialised, ``'Serie D 2026/27 (0 squadre)'``
        before (so a banner can still say which family is empty). Any spelling of
        the family that :func:`src.common.competitions.family_label` accepts works;
        ``ValueError`` for an unknown family.
        """
        if family is None:
            gironi = self.gironi
            text = describe_competitions((g.competition for g in gironi), season=self.season_label)
        else:
            name = family_label(family)
            gironi = tuple(g for g in self.gironi if g.family == name)
            text = describe_competitions((g.competition for g in gironi), season=self.season_label) if gironi else f"{name} {self.season_label}"
        n_teams = sum(len(self.teams_by_girone.get(g, ())) for g in gironi)
        return f"{text} ({n_teams} squadre)"

    def girone_of(self, canonical: str) -> Girone:
        for g, teams in self.teams_by_girone.items():
            if canonical in teams:
                return g
        raise KeyError(canonical)

    def teams(self, girone: "Girone | str") -> tuple[str, ...]:
        """Canonical teams of one competition; ``()`` when the registry has none for it (absent or empty).

        ``ValueError`` only for a spelling that is not a configured competition.
        """
        return tuple(self.teams_by_girone.get(Girone.parse(girone), ()))

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
        """Build from ``{competition code: {canonical: [aliases]}}`` (the JSON layout).

        Keys are parsed with :meth:`Girone.parse`, so ``'SB'`` and ``'Serie B'``
        both work. A competition mapped to an **empty** object (the Serie D
        gironi before their auto-init) is accepted and simply has no teams: it is
        left out of ``teams_by_girone`` and reported by
        :meth:`missing_competitions`. ``ValueError`` when a canonical name is
        listed under two competitions (a club has one canonical name and plays
        one league per season) or when a *normalised* spelling would resolve to
        two different clubs, whatever competitions they play in: with 80 teams in
        four competitions ``'Verona'`` as an alias of both ``'Virtus Verona'``
        (girone A) and ``'Hellas Verona'`` (Serie B) would otherwise silently keep
        only the club listed last. Repeating a spelling for the *same* club is fine.
        """
        teams_by_girone: dict[Girone, tuple[str, ...]] = {}
        aliases: dict[str, str] = {}
        owner: dict[str, Girone] = {}
        seen: set[Girone] = set()

        def register(spelling: str, canonical: str, g: Girone) -> None:
            norm = normalize_name(spelling)
            other = aliases.get(norm)
            if other is not None and other != canonical:
                raise ValueError(
                    f"season {season}: spelling {spelling!r} (normalised {norm!r}) of {canonical!r} [{g.value}] "
                    f"collides with {other!r} [{owner[other].value}]: one spelling must resolve to one club"
                )
            aliases[norm] = canonical

        for g_key, teams in gironi.items():
            g = Girone.parse(g_key)
            if g in seen:
                raise ValueError(f"season {season}: competition {g.value} listed twice")
            seen.add(g)
            if not teams:
                continue  # configured but not yet populated (see missing_competitions)
            teams_by_girone[g] = tuple(teams.keys())
            for canonical, alias_list in teams.items():
                if canonical in owner:
                    raise ValueError(f"season {season}: {canonical!r} is listed in both {owner[canonical].value} and {g.value}")
                owner[canonical] = g
                register(canonical, canonical, g)
                for alias in alias_list:
                    register(alias, canonical, g)
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
          so ``'Juventus U23'`` from an old file still maps to ``'Juventus Next Gen'``
          and ``'Delfino Pescara'`` (Serie B 2025/26) to ``'Pescara'`` (girone B
          2026/27), whatever competition the club plays now;
        * otherwise the team is **historical**: its old canonical name and its
          aliases map to the *old canonical spelling itself* (``'Ternana Calcio'``
          -> ``'Ternana'``, ``'Frosinone Calcio'`` -> ``'Frosinone'``), which is
          also recorded in ``historical_names``.

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
