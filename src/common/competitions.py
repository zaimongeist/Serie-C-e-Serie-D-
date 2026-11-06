"""Competitions of the perimeter, read from ``config/competitions.toml``.

A *competition* is one league table the model fits separately: a Serie C
girone, Serie B, later other leagues. Each has a short upper-case ``code``
(``'A'``, ``'B'``, ``'C'``, ``'SB'``) that is the value of the dynamic
:class:`src.common.schemas.Girone` enum, the key of the season registries
(``config/teams_<season>.json``) and the ``girone`` column of the canonical
match frame. The configuration file is an array of ``[[competition]]`` tables in
display order; :data:`COMPETITIONS` holds them as frozen :class:`Competition`
records. When the file is missing (or unreadable) the built-in
:data:`DEFAULT_COMPETITIONS` are used so importing the package never fails; the
problem is logged.

This module deliberately imports nothing from :mod:`src.common.schemas` (which
builds the enum from it) so the dependency graph stays acyclic.
"""
from __future__ import annotations

import re
import tomllib
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any, Final, Iterable, Mapping, Sequence

from .logging_config import get_logger
from .paths import CONFIG_DIR

__all__ = [
    "COMPETITIONS",
    "COMPETITIONS_PATH",
    "DEFAULT_COMPETITIONS",
    "Competition",
    "competition",
    "competitions_by_family",
    "describe_competitions",
    "find_competition",
    "load_competitions",
    "normalize_competition_name",
    "parse_competitions",
]

log = get_logger(__name__)

#: Default location of the competitions file.
COMPETITIONS_PATH: Final[Path] = CONFIG_DIR / "competitions.toml"

_CODE_RE: Final[re.Pattern[str]] = re.compile(r"^[A-Z][A-Z0-9]{0,7}$")
_SPACES_RE: Final[re.Pattern[str]] = re.compile(r"\s+")
#: Words that spellings of a competition may carry without changing its identity
#: (``'Girone A'``, ``'Group A'``, ``'Italy Serie C Group A'`` all mean code ``A``).
_FILLER_WORDS: Final[tuple[str, ...]] = ("GIRONE", "GIRONI", "GROUP", "GRUPPO")


def normalize_competition_name(value: object) -> str:
    """Upper-case spelling with ``_``/``-`` as spaces and collapsed whitespace (``'serie_b'`` -> ``'SERIE B'``)."""
    text = str(value).replace("_", " ").replace("-", " ").strip().upper()
    return _SPACES_RE.sub(" ", text)


@dataclass(frozen=True, slots=True)
class Competition:
    """One league table of the perimeter (a row of ``config/competitions.toml``).

    ``code`` is the stable identifier used everywhere in the code base; ``label``
    is the full public name (``'Serie C Girone A'``), ``short`` the compact one
    for tables and selectors (``'Girone A'``, ``'Serie B'``); ``footystats_name``
    is the ``name`` field of the FootyStats ``/league-list`` entry;
    ``sofascore_tournament_id`` the legacy SofaScore unique-tournament id (``0``
    when unknown); ``tier`` the level in the national pyramid and ``family`` the
    league the competition belongs to (the three Serie C gironi share
    ``'Serie C'``).
    """

    code: str
    label: str
    short: str
    country: str
    footystats_name: str
    sofascore_tournament_id: int
    tier: int
    family: str

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

    @property
    def spellings(self) -> frozenset[str]:
        """Normalised spellings that identify this competition unambiguously.

        The code, the label, the short label and the FootyStats name (``family``
        is *not* included because several competitions may share it).
        """
        return frozenset(normalize_competition_name(v) for v in (self.code, self.label, self.short, self.footystats_name))

    def to_dict(self) -> dict[str, Any]:
        """Plain mapping with the field names of the TOML table."""
        return {f.name: getattr(self, f.name) for f in fields(self)}


#: Built-in definitions, used when ``config/competitions.toml`` is missing.
DEFAULT_COMPETITIONS: Final[tuple[Competition, ...]] = (
    Competition("A", "Serie C Girone A", "Girone A", "Italy", "Italy Serie C Group A", 141, 3, "Serie C"),
    Competition("B", "Serie C Girone B", "Girone B", "Italy", "Italy Serie C Group B", 142, 3, "Serie C"),
    Competition("C", "Serie C Girone C", "Girone C", "Italy", "Italy Serie C Group C", 143, 3, "Serie C"),
    Competition("SB", "Serie B", "Serie B", "Italy", "Italy Serie B", 53, 2, "Serie B"),
)

_REQUIRED_KEYS: Final[tuple[str, ...]] = tuple(f.name for f in fields(Competition))


def parse_competitions(payload: Mapping[str, Any], *, source: str = "<mapping>") -> tuple[Competition, ...]:
    """Turn the decoded TOML document into competitions, in file order.

    ``payload`` must hold a non-empty ``competition`` array of tables with every
    :class:`Competition` field; unknown keys are rejected (typos must not pass
    silently), codes must be unique and so must the spellings each code answers
    to. ``ValueError`` names ``source`` in its message.
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
        unknown = sorted(set(table) - set(_REQUIRED_KEYS))
        if missing or unknown:
            raise ValueError(f"{source}: [[competition]] #{position} has missing keys {missing} / unknown keys {unknown}")
        try:
            comp = Competition(**{k: table[k] for k in _REQUIRED_KEYS})
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
    """Read ``path`` (TOML) and return its competitions in file order.

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


def _load_or_default(path: Path = COMPETITIONS_PATH) -> tuple[Competition, ...]:
    """The configured competitions, or :data:`DEFAULT_COMPETITIONS` when the file is missing or broken (logged)."""
    try:
        comps = load_competitions(path)
    except FileNotFoundError:
        log.info("competitions file not found, using the built-in definitions", extra={"path": str(path), "codes": [c.code for c in DEFAULT_COMPETITIONS]})
        return DEFAULT_COMPETITIONS
    except ValueError as exc:
        log.error("competitions file invalid, using the built-in definitions", extra={"path": str(path), "error": str(exc)})
        return DEFAULT_COMPETITIONS
    # DEBUG on purpose: this runs at import time, before the CLIs configure logging, so an INFO
    # line would open every ``python main.py --log-level ERROR`` run; the fallbacks above stay loud.
    log.debug("competitions loaded", extra={"path": str(path), "codes": [c.code for c in comps]})
    return comps


#: The competitions of this installation, in display order (``config/competitions.toml``).
COMPETITIONS: Final[tuple[Competition, ...]] = _load_or_default()


def competition(code: str, competitions: Iterable[Competition] = COMPETITIONS) -> Competition:
    """The competition with ``code`` (case-insensitive exact match); ``KeyError`` when unknown."""
    wanted = str(code).strip().upper()
    for comp in competitions:
        if comp.code == wanted:
            return comp
    raise KeyError(f"unknown competition code {code!r}; known: {[c.code for c in competitions]}")


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
    (``'A'``, ``'SB'``), the label, the short label and the FootyStats name
    (``'Girone A'``, ``'Serie B'``, ``'serie_b'``, ``'Italy Serie B'``), the
    family alone when it has a single competition, and a code wrapped in filler
    words (``GIRONE``/``GROUP``/``GRUPPO``) and/or in the name of **its own**
    family (``'Serie C Girone A'``, ``'Group A'``, ``'GIRONE_A'``, ``'GIRONEA'``,
    ``'serie c girone a'``). ``'B'`` always means the competition whose code is
    ``'B'`` (Serie C Girone B), never Serie B.

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
    """Competitions grouped by ``family`` in first-appearance order (``{'Serie C': (A, B, C), 'Serie B': (SB,)}``)."""
    groups: dict[str, list[Competition]] = {}
    for comp in competitions:
        groups.setdefault(comp.family, []).append(comp)
    return {family: tuple(members) for family, members in groups.items()}


def describe_competitions(competitions: Iterable[Competition] = COMPETITIONS, *, season: str | None = None) -> str:
    """Italian one-liner naming the competitions: ``'Serie C Gironi A/B/C + Serie B'``.

    Families are joined with ``' + '``; a family with several competitions lists
    their codes after ``Gironi``, a single-competition family is named alone.
    With ``season`` the label is inserted after the first family
    (``'Serie C 2026/27 Gironi A/B/C + Serie B'``), which is how banners read.
    An empty iterable yields ``''`` (or the bare season).
    """
    parts: list[str] = []
    for position, (family, members) in enumerate(competitions_by_family(competitions).items()):
        text = family
        if season is not None and position == 0:
            text += f" {season}"
        if len(members) > 1:
            text += " Gironi " + "/".join(c.code for c in members)
        parts.append(text)
    if not parts:
        return str(season) if season is not None else ""
    return " + ".join(parts)
