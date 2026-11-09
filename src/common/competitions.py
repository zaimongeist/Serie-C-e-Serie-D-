"""Competitions of the perimeter, read from ``config/competitions.toml``.

A *competition* is one league table the model fits separately: a Serie C
girone, Serie B, a Serie D girone, later other leagues. Each has a short
upper-case ``code`` (``'A'``, ``'B'``, ``'C'``, ``'SB'``, ``'DA'`` .. ``'DI'``)
that is the value of the dynamic :class:`src.common.schemas.Girone` enum, the key
of the season registries (``config/teams_<season>.json``) and the ``girone``
column of the canonical match frame. The configuration file is an array of
``[[competition]]`` tables in display order; :data:`ALL_COMPETITIONS` holds every
one of them as frozen :class:`Competition` records and :data:`COMPETITIONS` the
**enabled** ones. When the file is missing (or unreadable, or enables nothing)
the built-in :data:`DEFAULT_COMPETITIONS` are used so importing the package never
fails; the problem is logged.

Enabled and disabled competitions
---------------------------------
A table with ``enabled = false`` is configured but switched off (Serie A and the
European top flights of the shipped file, ready to be enabled once the league
is chosen in the FootyStats account): it is in :data:`ALL_COMPETITIONS` only.
:data:`COMPETITIONS` -- and therefore the ``Girone`` enum, the harvester's
default competitions and every consumer -- sees enabled competitions only. The
lookup helpers default to :data:`COMPETITIONS`; pass ``ALL_COMPETITIONS`` to
reach a disabled one (``competition('SA', ALL_COMPETITIONS)``).

Families, sections and fit groups
---------------------------------
Competitions are grouped three ways:

* **family** -- the league a competition belongs to (``'Serie C'`` for the
  three gironi, ``'Serie B'``, ``'Serie D'`` for the nine gironi): used by the
  descriptions (:func:`describe_competitions`) and the family helpers
  (:func:`families`, :func:`family_label`, :func:`competitions_of_family`);
* **section** -- the dashboard page that shows it (``section`` in the TOML,
  default: the family): :func:`sections`, :func:`section_label`,
  :func:`competitions_of_section`, :func:`section_of`. With the shipped file the
  sections are ``'Serie C'``, ``'Serie B'`` and ``'Serie D'``;
* **fit group** -- the set of competitions one Dixon-Coles model is fitted on:
  the ``country`` (:attr:`Competition.fit_group`, :func:`fit_group_of`,
  :func:`fit_groups`). Competitions of one country are connected by the clubs
  promoted and relegated between them and are fitted jointly; different
  countries share no match and are fitted separately
  (:mod:`src.engine.fitgroups`).

This module deliberately imports nothing from :mod:`src.common.schemas` (which
builds the enum from it) so the dependency graph stays acyclic.
"""
from __future__ import annotations

import re
import tomllib
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any, Final, Iterable, Mapping, Sequence

from .logging_config import get_logger
from .paths import CONFIG_DIR

__all__ = [
    "ALL_COMPETITIONS",
    "COMPETITIONS",
    "COMPETITIONS_PATH",
    "DEFAULT_COMPETITIONS",
    "Competition",
    "competition",
    "competitions_by_family",
    "competitions_by_section",
    "competitions_of_family",
    "competitions_of_section",
    "describe_competitions",
    "disabled_competitions",
    "enabled_competitions",
    "families",
    "family_label",
    "find_competition",
    "fit_group_of",
    "fit_groups",
    "load_competitions",
    "normalize_competition_name",
    "parse_competitions",
    "section_label",
    "section_of",
    "sections",
]

log = get_logger(__name__)

#: Default location of the competitions file.
COMPETITIONS_PATH: Final[Path] = CONFIG_DIR / "competitions.toml"

_CODE_RE: Final[re.Pattern[str]] = re.compile(r"^[A-Z][A-Z0-9]{0,7}$")
_SPACES_RE: Final[re.Pattern[str]] = re.compile(r"\s+")
_SEASON_LABEL_RE: Final[re.Pattern[str]] = re.compile(r"^(\d{4})[/_\-](\d{2}|\d{4})$")
#: Words that spellings of a competition may carry without changing its identity
#: (``'Girone A'``, ``'Group A'``, ``'Italy Serie C Group A'`` all mean code ``A``).
_FILLER_WORDS: Final[tuple[str, ...]] = ("GIRONE", "GIRONI", "GROUP", "GRUPPO")
#: Providers that resolve a competition at runtime by name and therefore serve every configured one.
_NAME_RESOLVING_PROVIDERS: Final[frozenset[str]] = frozenset({"footystats", "fotmob"})
#: Minimum number of consecutive single-letter gironi described as a range (``'A-I'`` instead of ``'A/B/C'``).
_RANGE_MIN_MEMBERS: Final[int] = 4


def normalize_competition_name(value: object) -> str:
    """Upper-case spelling with ``_``/``-`` as spaces and collapsed whitespace (``'serie_b'`` -> ``'SERIE B'``)."""
    text = str(value).replace("_", " ").replace("-", " ").strip().upper()
    return _SPACES_RE.sub(" ", text)


def _normalize_season_label(label: object) -> str:
    """``'2026_27'`` / ``'2026-27'`` / ``'2026/2027'`` -> ``'2026/27'``; ``ValueError`` for anything else."""
    text = str(label).strip()
    found = _SEASON_LABEL_RE.match(text)
    if found is None:
        raise ValueError(f"season label must look like '2026/27', got {label!r}")
    start = int(found.group(1))
    second = found.group(2)
    end = int(second) if len(second) == 4 else (start // 100) * 100 + int(second)
    if len(second) == 2 and end < start:  # '1999/00'
        end += 100
    if end != start + 1:
        raise ValueError(f"season label must span two consecutive years, got {label!r}")
    return f"{start}/{end % 100:02d}"


@dataclass(frozen=True, slots=True)
class Competition:
    """One league table of the perimeter (a row of ``config/competitions.toml``).

    ``code`` is the stable identifier used everywhere in the code base; ``label``
    is the full public name (``'Serie C Girone A'``, ``'Serie D Girone A'``),
    ``short`` the compact one for tables and selectors (``'Girone A'``,
    ``'Serie B'``, ``'D-A'``); ``footystats_name`` is the ``name`` field of the
    FootyStats ``/league-list`` entry; ``sofascore_tournament_id`` the legacy
    SofaScore unique-tournament id (``0`` when unknown: the competition is then
    not served by that provider); ``tier`` the level in the national pyramid and
    ``family`` the league the competition belongs to (the three Serie C gironi
    share ``'Serie C'``, the nine Serie D gironi ``'Serie D'``). ``season_ids``
    maps a season label (``'2026/27'``) to the FootyStats ``season_id`` and is the
    fallback used only when ``/league-list`` itself cannot be reached: it cannot
    stand in for a league the account has not chosen, whose data the API refuses
    (the harvester skips such a competition with a warning).

    ``enabled`` (default ``True``) switches the competition on: a disabled one is
    configured but kept out of :data:`COMPETITIONS` (and so out of the ``Girone``
    enum and of every consumer). ``section`` is the dashboard section that shows
    it; empty (the default) means the family, so ``Competition(...).section`` is
    never blank. The fit group is the country (:attr:`fit_group`).
    """

    code: str
    label: str
    short: str
    country: str
    footystats_name: str
    sofascore_tournament_id: int
    tier: int
    family: str
    season_ids: dict[str, int] = field(default_factory=dict, hash=False)
    enabled: bool = True
    section: str = ""

    def __post_init__(self) -> None:
        code = str(self.code).strip().upper()
        if not _CODE_RE.match(code):
            raise ValueError(f"competition code must be 1-8 upper-case letters/digits starting with a letter, got {self.code!r}")
        object.__setattr__(self, "code", code)
        for name in ("label", "short", "country", "footystats_name", "family"):
            text = str(getattr(self, name)).strip()
            if not text:
                raise ValueError(f"competition {code}: {name} must not be empty")
            object.__setattr__(self, name, text)
        for name in ("sofascore_tournament_id", "tier"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int):
                raise ValueError(f"competition {code}: {name} must be an integer, got {value!r}")
        if self.sofascore_tournament_id < 0:
            raise ValueError(f"competition {code}: sofascore_tournament_id must be >= 0, got {self.sofascore_tournament_id}")
        if self.tier < 1:
            raise ValueError(f"competition {code}: tier must be >= 1, got {self.tier}")
        if not isinstance(self.season_ids, Mapping):
            raise ValueError(f"competition {code}: season_ids must be a mapping of season label -> FootyStats season id")
        ids: dict[str, int] = {}
        for raw_label, raw_id in self.season_ids.items():
            try:
                label = _normalize_season_label(raw_label)
            except ValueError as exc:
                raise ValueError(f"competition {code}: season_ids: {exc}") from exc
            if isinstance(raw_id, bool) or not isinstance(raw_id, int) or raw_id < 1:
                raise ValueError(f"competition {code}: season_ids[{label!r}] must be a positive integer, got {raw_id!r}")
            if label in ids:
                raise ValueError(f"competition {code}: season_ids lists season {label!r} twice")
            ids[label] = raw_id
        object.__setattr__(self, "season_ids", ids)
        if not isinstance(self.enabled, bool):
            raise ValueError(f"competition {code}: enabled must be true or false, got {self.enabled!r}")
        if not isinstance(self.section, str):
            raise ValueError(f"competition {code}: section must be a string, got {self.section!r}")
        object.__setattr__(self, "section", self.section.strip() or self.family)

    @property
    def fit_group(self) -> str:
        """The fit group of the competition: its ``country``.

        Competitions of one country are fitted jointly (promotions and relegations
        connect them), competitions of different countries separately (see
        :mod:`src.engine.fitgroups`).
        """
        return self.country

    @property
    def spellings(self) -> frozenset[str]:
        """Normalised spellings that identify this competition unambiguously.

        The code, the label, the short label and the FootyStats name (``family``
        is *not* included because several competitions may share it).
        """
        return frozenset(normalize_competition_name(v) for v in (self.code, self.label, self.short, self.footystats_name))

    @property
    def girone_letter(self) -> str:
        """The girone designation inside the family: ``'A'`` for ``'Serie C Girone A'`` *and* ``'Serie D Girone A'``, ``''`` for ``'Serie B'``.

        Derived from the label with the family name and the filler words removed, so
        descriptions can say ``Gironi A-I`` for Serie D instead of ``DA/DB/...``.
        """
        words = _without_phrase(_code_words(normalize_competition_name(self.label)), _code_words(normalize_competition_name(self.family)))
        return " ".join(words)

    def supports_provider(self, name: str) -> bool:
        """Whether data provider ``name`` can serve this competition.

        ``'sofascore'`` needs a known tournament id (``False`` when it is ``0``);
        ``'footystats'`` and ``'fotmob'`` resolve competitions by name at runtime
        and serve every configured one. ``ValueError`` for an empty name; an
        unknown provider name is assumed to resolve by name as well.
        """
        provider = str(name).strip().lower()
        if not provider:
            raise ValueError("provider name must not be empty")
        if provider == "sofascore":
            return self.sofascore_tournament_id > 0
        if provider in _NAME_RESOLVING_PROVIDERS:
            return bool(self.footystats_name)
        return True

    def to_dict(self) -> dict[str, Any]:
        """Plain mapping with the field names of the TOML table (``season_ids`` as a copy)."""
        out = {f.name: getattr(self, f.name) for f in fields(self)}
        out["season_ids"] = dict(self.season_ids)
        return out


def _serie_d(letter: str, season_ids: Mapping[str, int]) -> Competition:
    return Competition(
        f"D{letter}", f"Serie D Girone {letter}", f"D-{letter}", "Italy", f"Italy Serie D Group {letter}", 0, 4, "Serie D", dict(season_ids)
    )


def _top_flight(code: str, label: str, country: str, footystats_name: str, ids: tuple[int, int, int]) -> Competition:
    """A disabled single-table top flight whose family and section are its label (Serie A, the European leagues)."""
    season_ids = dict(zip(("2024/25", "2025/26", "2026/27"), ids))
    return Competition(code, label, label, country, footystats_name, 0, 1, label, season_ids, enabled=False, section=label)


#: Built-in definitions, used when ``config/competitions.toml`` is missing or invalid: identical to the
#: shipped file, disabled tables included (Serie B/C/D FootyStats season ids verified live on 2026-11-06).
DEFAULT_COMPETITIONS: Final[tuple[Competition, ...]] = (
    _top_flight("SA", "Serie A", "Italy", "Italy Serie A", (12530, 15068, 17084)),
    Competition("A", "Serie C Girone A", "Girone A", "Italy", "Italy Serie C Group A", 141, 3, "Serie C", {"2024/25": 12659, "2025/26": 15535, "2026/27": 17487}),
    Competition("B", "Serie C Girone B", "Girone B", "Italy", "Italy Serie C Group B", 142, 3, "Serie C", {"2024/25": 12647, "2025/26": 15532, "2026/27": 17492}),
    Competition("C", "Serie C Girone C", "Girone C", "Italy", "Italy Serie C Group C", 143, 3, "Serie C", {"2024/25": 12670, "2025/26": 15519, "2026/27": 17473}),
    Competition("SB", "Serie B", "Serie B", "Italy", "Italy Serie B", 53, 2, "Serie B", {"2024/25": 12621, "2025/26": 15632, "2026/27": 17404}),
    _serie_d("A", {"2024/25": 13475, "2025/26": 15710, "2026/27": 17664}),
    _serie_d("B", {"2024/25": 13232, "2025/26": 15706, "2026/27": 17667}),
    _serie_d("C", {"2024/25": 13479, "2025/26": 15718, "2026/27": 17666}),
    _serie_d("D", {"2024/25": 13481, "2025/26": 15714, "2026/27": 17676}),
    _serie_d("E", {"2024/25": 13473, "2025/26": 15685, "2026/27": 17669}),
    _serie_d("F", {"2024/25": 13470, "2025/26": 15707, "2026/27": 17671}),
    _serie_d("G", {"2024/25": 13469, "2025/26": 15712, "2026/27": 17672}),
    _serie_d("H", {"2024/25": 13477, "2025/26": 15693, "2026/27": 17674}),
    _serie_d("I", {"2024/25": 13472, "2025/26": 15705, "2026/27": 17675}),
    _top_flight("EPL", "Premier League", "England", "England Premier League", (12325, 15050, 17146)),
    _top_flight("LL", "La Liga", "Spain", "Spain La Liga", (12316, 14956, 17199)),
    _top_flight("BL", "Bundesliga", "Germany", "Germany Bundesliga", (12529, 14968, 17210)),
    _top_flight("L1", "Ligue 1", "France", "France Ligue 1", (12337, 14932, 17102)),
    _top_flight("ERE", "Eredivisie", "Netherlands", "Netherlands Eredivisie", (12322, 14936, 17097)),
    _top_flight("PL1", "Primeira Liga", "Portugal", "Portugal Liga NOS", (12931, 15115, 17217)),
)

#: Keys a ``[[competition]]`` table may omit: ``season_ids`` (no fallback ids), ``enabled`` (``true``) and
#: ``section`` (the family); every other field of :class:`Competition` is required.
_OPTIONAL_KEYS: Final[frozenset[str]] = frozenset({"season_ids", "enabled", "section"})
_REQUIRED_KEYS: Final[tuple[str, ...]] = tuple(f.name for f in fields(Competition) if f.name not in _OPTIONAL_KEYS)
_ALL_KEYS: Final[frozenset[str]] = frozenset(f.name for f in fields(Competition))


def parse_competitions(payload: Mapping[str, Any], *, source: str = "<mapping>") -> tuple[Competition, ...]:
    """Turn the decoded TOML document into competitions, in file order.

    ``payload`` must hold a non-empty ``competition`` array of tables with every
    :class:`Competition` field except the optional ``season_ids`` sub-table,
    ``enabled`` (default ``true``) and ``section`` (default: the family; an
    explicitly blank ``section`` is rejected, omit the key instead); unknown keys
    are rejected (typos must not pass silently), codes must be unique and so
    must the spellings each code answers to -- across enabled *and* disabled
    tables, so enabling one never makes a spelling ambiguous. Disabled tables
    are returned too (filter with :func:`enabled_competitions`). ``ValueError``
    names ``source`` in its message.
    """
    tables = payload.get("competition")
    if not isinstance(tables, list) or not tables:
        raise ValueError(f"{source}: expected a non-empty [[competition]] array")
    out: list[Competition] = []
    seen_codes: dict[str, int] = {}
    seen_spellings: dict[str, str] = {}
    for position, table in enumerate(tables, start=1):
        if not isinstance(table, Mapping):
            raise ValueError(f"{source}: [[competition]] #{position} is not a table")
        missing = [k for k in _REQUIRED_KEYS if k not in table]
        unknown = sorted(set(table) - _ALL_KEYS)
        if missing or unknown:
            raise ValueError(f"{source}: [[competition]] #{position} has missing keys {missing} / unknown keys {unknown}")
        section = table.get("section")
        if isinstance(section, str) and not section.strip():
            raise ValueError(f"{source}: [[competition]] #{position}: section must not be blank (omit the key to use the family)")
        try:
            comp = Competition(**{k: table[k] for k in _ALL_KEYS if k in table})
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{source}: [[competition]] #{position}: {exc}") from exc
        if comp.code in seen_codes:
            raise ValueError(f"{source}: duplicate competition code {comp.code!r} (tables #{seen_codes[comp.code]} and #{position})")
        for spelling in comp.spellings:
            owner = seen_spellings.get(spelling)
            if owner is not None and owner != comp.code:
                raise ValueError(f"{source}: spelling {spelling!r} belongs to both {owner!r} and {comp.code!r}")
            seen_spellings[spelling] = comp.code
        seen_codes[comp.code] = position
        out.append(comp)
    return tuple(out)


def load_competitions(path: Path | str = COMPETITIONS_PATH) -> tuple[Competition, ...]:
    """Read ``path`` (TOML) and return **all** its competitions (disabled included) in file order.

    ``FileNotFoundError`` when the file does not exist, ``ValueError`` when it is
    not valid TOML or does not describe competitions (see :func:`parse_competitions`).
    """
    file = Path(path)
    if not file.is_file():
        raise FileNotFoundError(f"competitions file not found: {file}")
    try:
        payload = tomllib.loads(file.read_text(encoding="utf-8"))
    except tomllib.TOMLDecodeError as exc:
        raise ValueError(f"{file}: invalid TOML ({exc})") from exc
    return parse_competitions(payload, source=str(file))


def enabled_competitions(competitions: Iterable[Competition] | None = None) -> tuple[Competition, ...]:
    """The enabled competitions of ``competitions`` (default :data:`ALL_COMPETITIONS`), in their order."""
    return tuple(c for c in (ALL_COMPETITIONS if competitions is None else competitions) if c.enabled)


def disabled_competitions(competitions: Iterable[Competition] | None = None) -> tuple[Competition, ...]:
    """The configured but switched-off competitions of ``competitions`` (default :data:`ALL_COMPETITIONS`), in their order."""
    return tuple(c for c in (ALL_COMPETITIONS if competitions is None else competitions) if not c.enabled)


def _load_or_default(path: Path = COMPETITIONS_PATH) -> tuple[Competition, ...]:
    """All configured competitions, or :data:`DEFAULT_COMPETITIONS` when the file is missing, broken or enables none (logged)."""
    try:
        comps = load_competitions(path)
    except FileNotFoundError:
        log.info("competitions file not found, using the built-in definitions", extra={"path": str(path), "codes": [c.code for c in DEFAULT_COMPETITIONS]})
        return DEFAULT_COMPETITIONS
    except ValueError as exc:
        log.error("competitions file invalid, using the built-in definitions", extra={"path": str(path), "error": str(exc)})
        return DEFAULT_COMPETITIONS
    enabled = [c.code for c in comps if c.enabled]
    if not enabled:
        log.error(
            "competitions file enables no competition, using the built-in definitions",
            extra={"path": str(path), "disabled": [c.code for c in comps]},
        )
        return DEFAULT_COMPETITIONS
    # DEBUG on purpose: this runs at import time, before the CLIs configure logging, so an INFO
    # line would open every ``python main.py --log-level ERROR`` run; the fallbacks above stay loud.
    log.debug("competitions loaded", extra={"path": str(path), "codes": enabled, "disabled": [c.code for c in comps if not c.enabled]})
    return comps


#: Every competition of ``config/competitions.toml`` (enabled and disabled), in display order.
ALL_COMPETITIONS: Final[tuple[Competition, ...]] = _load_or_default()

#: The **enabled** competitions of this installation, in display order: the ``Girone`` enum, the harvester
#: and every consumer see these only (13 with the shipped file: Serie C A/B/C, Serie B, Serie D DA..DI).
COMPETITIONS: Final[tuple[Competition, ...]] = enabled_competitions(ALL_COMPETITIONS)


def competition(code: str, competitions: Iterable[Competition] = COMPETITIONS) -> Competition:
    """The competition with ``code`` (case-insensitive exact match); ``KeyError`` when unknown.

    Only enabled competitions are searched by default: a disabled one is
    reached with ``competition('SA', ALL_COMPETITIONS)``.
    """
    comps = tuple(competitions)
    wanted = str(code).strip().upper()
    for comp in comps:
        if comp.code == wanted:
            return comp
    raise KeyError(f"unknown competition code {code!r}; known: {[c.code for c in comps]}")


def _code_words(text: str) -> list[str]:
    """Words of a normalised spelling without the filler words.

    A filler glued to a code (``'GIRONEA'``) is split so that the pre-2026/27
    spellings keep working: ``'GIRONE A'`` -> ``['A']``, ``'GIRONEA'`` -> ``['A']``,
    ``'SERIE C GROUP A'`` -> ``['SERIE', 'C', 'A']``.
    """
    words: list[str] = []
    for word in text.split():
        if word in _FILLER_WORDS:
            continue
        for filler in _FILLER_WORDS:
            if word.startswith(filler) and len(word) > len(filler):
                word = word[len(filler):]
                break
        words.append(word)
    return words


def _without_phrase(words: Sequence[str], phrase: Sequence[str]) -> list[str]:
    """``words`` with the first contiguous occurrence of ``phrase`` removed (unchanged when absent or empty)."""
    size = len(phrase)
    if size == 0:
        return list(words)
    for start in range(len(words) - size + 1):
        if list(words[start : start + size]) == list(phrase):
            return [*words[:start], *words[start + size :]]
    return list(words)


def find_competition(value: object, competitions: Iterable[Competition] = COMPETITIONS) -> Competition:
    """Resolve any accepted spelling of a competition (``ValueError`` when none matches).

    Accepted, case-insensitively and with ``_``/``-`` as spaces: the code
    (``'A'``, ``'SB'``, ``'DA'``), the label, the short label and the FootyStats
    name (``'Girone A'``, ``'Serie B'``, ``'serie_b'``, ``'Italy Serie B'``,
    ``'Serie D Girone A'``, ``'D-A'``, ``'Italy Serie D Group A'``), the family
    alone when it has a single competition, and a code wrapped in filler words
    (``GIRONE``/``GROUP``/``GRUPPO``) and/or in the name of **its own** family
    (``'Serie C Girone A'``, ``'Group A'``, ``'GIRONE_A'``, ``'GIRONEA'``,
    ``'serie c girone a'``). ``'A'``, ``'B'`` and ``'C'`` always mean the Serie C
    gironi (the competitions with those codes), never Serie B or a Serie D girone;
    ``'D'`` alone is ambiguous (nine Serie D gironi) and raises.

    Contradictory spellings are rejected rather than guessed: ``'Serie B Girone
    A'`` (the family of another competition), ``'Serie C SB'``, ``'Italy A'`` (a
    country is not a filler word) and ``'Serie B B'`` all raise, so a mislabelled
    ``girone`` cell never passes validation as a different competition.
    """
    comps = tuple(competitions)
    text = normalize_competition_name(value)
    if not text:
        raise ValueError(f"Unknown girone: {value!r}")
    for comp in comps:  # exact code first: 'B' is Girone B, not Serie B
        if comp.code == text:
            return comp
    for comp in comps:
        if text in comp.spellings:
            return comp
    for family, members in competitions_by_family(comps).items():
        if len(members) == 1 and normalize_competition_name(family) == text:
            return members[0]
    # a code wrapped in filler words and, at most, the name of its own family
    words = _code_words(text)
    for comp in comps:
        family_words = _code_words(normalize_competition_name(comp.family))
        if _without_phrase(words, family_words) == [comp.code]:
            return comp
    raise ValueError(f"Unknown girone: {value!r}")


def competitions_by_family(competitions: Iterable[Competition] = COMPETITIONS) -> dict[str, tuple[Competition, ...]]:
    """Competitions grouped by ``family`` in first-appearance order (``{'Serie C': (A, B, C), 'Serie B': (SB,), 'Serie D': (DA, ..., DI)}``)."""
    groups: dict[str, list[Competition]] = {}
    for comp in competitions:
        groups.setdefault(comp.family, []).append(comp)
    return {family: tuple(members) for family, members in groups.items()}


def families(competitions: Iterable[Competition] = COMPETITIONS) -> tuple[str, ...]:
    """The family names in configuration order: ``('Serie C', 'Serie B', 'Serie D')`` with the shipped file."""
    return tuple(competitions_by_family(competitions))


def family_label(family: object, competitions: Iterable[Competition] = COMPETITIONS) -> str:
    """Canonical spelling of a family given any case/underscore variant (``'serie_d'`` -> ``'Serie D'``).

    ``ValueError`` when no configured competition belongs to that family.
    """
    wanted = normalize_competition_name(family)
    for name in families(competitions):
        if normalize_competition_name(name) == wanted:
            return name
    raise ValueError(f"Unknown family: {family!r} (known: {list(families(competitions))})")


def competitions_of_family(family: object, competitions: Iterable[Competition] = COMPETITIONS) -> tuple[Competition, ...]:
    """The competitions of one family in configuration order (``ValueError`` for an unknown family)."""
    comps = tuple(competitions)
    name = family_label(family, comps)
    return competitions_by_family(comps)[name]


def competitions_by_section(competitions: Iterable[Competition] = COMPETITIONS) -> dict[str, tuple[Competition, ...]]:
    """Competitions grouped by ``section`` in first-appearance order (``{'Serie C': (A, B, C), 'Serie B': (SB,), 'Serie D': (DA, ..., DI)}``)."""
    groups: dict[str, list[Competition]] = {}
    for comp in competitions:
        groups.setdefault(comp.section, []).append(comp)
    return {section: tuple(members) for section, members in groups.items()}


def sections(competitions: Iterable[Competition] = COMPETITIONS) -> tuple[str, ...]:
    """The dashboard sections in configuration order: ``('Serie C', 'Serie B', 'Serie D')`` with the shipped file.

    A section is the ``section`` value of its competitions (the family unless the
    TOML says otherwise); only the sections of ``competitions`` are listed, so
    a disabled league has none until it is enabled.
    """
    return tuple(competitions_by_section(competitions))


def section_label(section: object, competitions: Iterable[Competition] = COMPETITIONS) -> str:
    """Canonical spelling of a section given any case/underscore variant (``'serie_b'`` -> ``'Serie B'``).

    ``ValueError`` when no competition of ``competitions`` belongs to that section.
    """
    comps = tuple(competitions)
    wanted = normalize_competition_name(section)
    for name in sections(comps):
        if normalize_competition_name(name) == wanted:
            return name
    raise ValueError(f"Unknown section: {section!r} (known: {list(sections(comps))})")


def competitions_of_section(section: object, competitions: Iterable[Competition] = COMPETITIONS) -> tuple[Competition, ...]:
    """The competitions of one section in configuration order (``ValueError`` for an unknown section)."""
    comps = tuple(competitions)
    return competitions_by_section(comps)[section_label(section, comps)]


def section_of(code: object, competitions: Iterable[Competition] = COMPETITIONS) -> str:
    """Section of the competition with ``code`` (``'A'`` -> ``'Serie C'``, ``'SB'`` -> ``'Serie B'``); ``KeyError`` when unknown.

    A ``Girone`` member is accepted as well (its value is the code).
    """
    return competition(_code_text(code), competitions).section


def fit_group_of(code: object, competitions: Iterable[Competition] = COMPETITIONS) -> str:
    """Fit group (the country) of the competition with ``code`` (``'DA'`` -> ``'Italy'``); ``KeyError`` when unknown."""
    return competition(_code_text(code), competitions).fit_group


def fit_groups(competitions: Iterable[Competition] = COMPETITIONS) -> dict[str, tuple[str, ...]]:
    """Competition codes per fit group, both in configuration order.

    ``{'Italy': ('A', 'B', 'C', 'SB', 'DA', ..., 'DI')}`` with the shipped file;
    enabling the Premier League adds ``'England': ('EPL',)``. One Dixon-Coles
    model is fitted per group (:func:`src.engine.fitgroups.fit_by_group`).
    """
    groups: dict[str, list[str]] = {}
    for comp in competitions:
        groups.setdefault(comp.fit_group, []).append(comp.code)
    return {group: tuple(codes) for group, codes in groups.items()}


def _code_text(code: object) -> str:
    """The code of a competition given as a string or as an enum member (``Girone.A`` -> ``'A'``)."""
    value = getattr(code, "value", code)
    return str(value)


def _gironi_text(members: Sequence[Competition]) -> str:
    """``'A/B/C'`` for a few gironi, ``'A-I'`` for a run of at least four consecutive letters, codes when letters are missing."""
    letters = [c.girone_letter for c in members]
    if any(not letter for letter in letters):
        return "/".join(c.code for c in members)
    consecutive = all(len(letter) == 1 and letter.isalpha() for letter in letters) and all(
        ord(b) == ord(a) + 1 for a, b in zip(letters, letters[1:])
    )
    if consecutive and len(letters) >= _RANGE_MIN_MEMBERS:
        return f"{letters[0]}-{letters[-1]}"
    return "/".join(letters)


def describe_competitions(
    competitions: Iterable[Competition] = COMPETITIONS,
    *,
    season: str | None = None,
    family: str | None = None,
    section: str | None = None,
) -> str:
    """Italian one-liner naming the competitions: ``'Serie C Gironi A/B/C + Serie B + Serie D Gironi A-I'``.

    Families are joined with ``' + '``; a family with several competitions lists
    their girone letters after ``Gironi`` (``A/B/C``, or a range ``A-I`` for at
    least four consecutive letters), a single-competition family is named alone.
    With ``season`` the label is inserted after the first family
    (``'Serie C 2026/27 Gironi A/B/C + Serie B'``), which is how banners read.
    ``family`` restricts the description to that family
    (``describe_competitions(family='Serie D', season='2026/27')`` ->
    ``'Serie D 2026/27 Gironi A-I'``; ``ValueError`` for an unknown family when
    the iterable holds competitions); ``section`` likewise restricts it to one
    dashboard section (``describe_competitions(section='Serie C')`` ->
    ``'Serie C Gironi A/B/C'``; ``ValueError`` for an unknown section). An empty
    selection yields ``''`` (or the bare season).
    """
    comps = tuple(competitions)
    if section is not None and comps:
        comps = competitions_of_section(section, comps)
    if family is not None and comps:
        comps = competitions_of_family(family, comps)
    parts: list[str] = []
    for position, (name, members) in enumerate(competitions_by_family(comps).items()):
        text = name
        if season is not None and position == 0:
            text += f" {season}"
        if len(members) > 1:
            text += " Gironi " + _gironi_text(members)
        parts.append(text)
    if not parts:
        return str(season) if season is not None else ""
    return " + ".join(parts)
