"""FootyStats provider: the official ``api.football-data-api.com`` API.

Unlike SofaScore, FootyStats is a documented, key-authenticated API that answers
from datacenter IPs, so it is the provider the scheduled GitHub Actions job runs
with. It also quotes pre-match odds (1X2, Over/Under, both teams to score), which
feed the value engine through ``data/odds/footystats.csv``.

The module is layered like :mod:`src.ingestion.harvester`:

* key handling -- :func:`resolve_api_key` (``--key``, ``FOOTYSTATS_API_KEY``,
  ``.env``), :func:`mask_key`, :class:`KeyInjectingTransport`: the key is added
  to the request at the transport level, below the client that caches, logs and
  raises, so it never reaches a cache file, a log line or an exception;
* :class:`FootyStatsClient` -- the JSON envelope (``success``, ``pager``,
  ``data``), pagination, the three endpoints (``/league-list``,
  ``/league-teams``, ``/league-matches``) and season resolution by competition
  name;
* pure parsers -- :func:`parse_match` and :func:`parse_odds` from the 215-field
  match objects to :class:`MatchRecord` / :class:`Fixture` / :class:`OddsQuote`;
* :class:`FootyStatsTeamMapper` -- FootyStats team ids -> canonical names of the
  :class:`TeamRegistry`, persisted in ``config/footystats_teams.json`` so ids map
  deterministically on the next run;
* :class:`FootyStatsProvider` -- harvests the newest seasons of every requested
  competition (Serie C gironi ``A``/``B``/``C``, Serie B ``SB`` and the nine
  Serie D gironi ``DA`` .. ``DI``) into one canonical frame, with the current
  season's fixtures and their odds, seeds a team dictionary from
  ``/league-teams`` (:meth:`init_registry`) and **auto-initialises** the
  competitions whose dictionary entry is still empty (the Serie D gironi ship
  without a team list) before harvesting them
  (``FootyStatsProvider.harvest(auto_init_registry=True)``);
* a CLI (``python -m src.ingestion.footystats``).

Serie D
-------
FootyStats covers the nine Serie D gironi but with thinner statistics: shots
and xG are often not collected (reported as ``0``/``-1`` and read as missing by
:func:`parse_stats`) and pre-match odds exist for fewer matches
(:func:`parse_odds` simply skips the markets without a complete book). The
season ids of every configured competition (``[competition.season_ids]`` in
``config/competitions.toml``) are the fallback when ``/league-list`` cannot be
reached or does not list the league among the account's chosen ones
(:func:`competition_season_ids`, :meth:`FootyStatsClient.resolve_seasons`).

Competition codes
-----------------
A match frame carries the competition in its ``girone`` column, so a competition
can only be harvested when :class:`~src.common.schemas.Girone` accepts its code.
Codes the schema does not know yet are *skipped* with an explicit reason
(:attr:`FootyStatsHarvest.skipped`), never silently dropped and never a crash:
as soon as the schema learns a code, the same run harvests it. The FootyStats
league name of a code comes from :mod:`src.common.competitions` when that module
is available, else from ``[footystats.league_names]`` in ``endpoints.toml``.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import random
import re
import sys
import tempfile
import time
from collections import Counter
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any, Final

import httpx
import pandas as pd

from src.common.logging_config import configure_logging, get_logger
from src.common.paths import CONFIG_DIR, ODDS_DIR, PROCESSED_DIR, RAW_DIR, ROOT
from src.common.schemas import (
    Fixture,
    Girone,
    Market,
    MatchRecord,
    OddsQuote,
    empty_matches_frame,
    matches_to_frame,
    season_bounds,
    season_of,
)
from src.common.teams import DEFAULT_SEASON, TeamRegistry, list_seasons, load_registry, normalize_name, season_file_stem
from src.ingestion.harvester import (
    AsyncHarvesterClient,
    ClientStats,
    FootyStatsEndpoints,
    HarvestError,
    HttpConfig,
    RawCache,
    Sleeper,
    _write_decision,
    load_endpoints,
    save_fixtures,
    save_odds,
    save_processed,
)
from src.ingestion.team_matcher import TeamMatcher

try:  # the shared competitions module is optional: the configuration carries the names too
    from src.common.competitions import competition as _shared_competition
except ImportError:  # pragma: no cover - depends on the repository state
    _shared_competition = None

__all__ = [
    "DEFAULT_COMPETITIONS",
    "DEFAULT_ENV_FILE",
    "DEFAULT_MAPPING_PATH",
    "FOOTYSTATS_BOOKMAKER",
    "FOOTYSTATS_USER_AGENT",
    "FootyStatsClient",
    "FootyStatsError",
    "FootyStatsHarvest",
    "FootyStatsProvider",
    "FootyStatsTeamMapper",
    "KeyInjectingTransport",
    "LeagueInfo",
    "MappedTeam",
    "MissingApiKeyError",
    "ParseReport",
    "RegistryDiff",
    "SeasonRef",
    "TeamInfo",
    "competition_label",
    "competition_season_ids",
    "configured_league_names",
    "find_api_key",
    "footystats_league_name",
    "init_registry",
    "main",
    "mask_key",
    "normalize_code",
    "parse_match",
    "parse_odds",
    "parse_stats",
    "read_env_file",
    "resolve_api_key",
    "schema_supports",
    "season_completed",
    "season_label_from_footystats_year",
    "write_registry_update",
]

log = get_logger(__name__)

def _footystats_members() -> tuple[str, ...]:
    """Codes of the configured competitions FootyStats can serve (every one with a ``footystats_name``)."""
    codes: list[str] = []
    for member in Girone:
        try:
            supported = member.competition.supports_provider("footystats")
        except LookupError:  # pragma: no cover - only a hand-built enum lacks the competition table
            supported = True
        if supported:
            codes.append(str(member.value))
    return tuple(codes)


#: Competitions harvested when none is given: every competition of the perimeter the provider
#: supports, in the order of ``config/competitions.toml`` (the :class:`~src.common.schemas.Girone`
#: members: ``A``, ``B``, ``C``, ``SB`` and the Serie D gironi ``DA`` .. ``DI``, 13 by default), so a
#: league added to the configuration is harvested, documented and listed in the key instructions
#: without a code change.
DEFAULT_COMPETITIONS: Final[tuple[str, ...]] = _footystats_members()
#: ``.env`` file read by :func:`find_api_key` when the environment has no key.
DEFAULT_ENV_FILE: Final[Path] = ROOT / ".env"
#: Persisted FootyStats team-id mapping (see :class:`FootyStatsTeamMapper`).
DEFAULT_MAPPING_PATH: Final[Path] = CONFIG_DIR / "footystats_teams.json"
#: ``bookmaker`` column of the odds FootyStats quotes (an aggregate pre-match price).
FOOTYSTATS_BOOKMAKER: Final[str] = "FootyStats"
#: Honest signature for an official API (no browser impersonation needed).
FOOTYSTATS_USER_AGENT: Final[str] = "SerieC-Quant-Lab/1.0 (python-httpx; +https://github.com)"
#: Query parameter that carries the key.
KEY_PARAM: Final[str] = "key"
#: Hard stop for runaway pagination (the API serves up to 600 matches per page).
MAX_PAGES: Final[int] = 200
#: Mask that replaces the key wherever it could surface.
MASK: Final[str] = "***"

#: Last-resort league names (``name`` field of ``/league-list``), verified 2026-11-06.
_FALLBACK_LEAGUE_NAMES: Final[dict[str, str]] = {
    "A": "Italy Serie C Group A",
    "B": "Italy Serie C Group B",
    "C": "Italy Serie C Group C",
    "SB": "Italy Serie B",
}

_KEY_IN_TEXT_RE: Final[re.Pattern[str]] = re.compile(r"(?i)(\bkey\b['\"]?\s*[:=]\s*['\"]?)([^&\s'\"]+)")
_PLAIN_YEAR_RE: Final[re.Pattern[str]] = re.compile(r"^\d{4}$")
_DOUBLE_YEAR_RE: Final[re.Pattern[str]] = re.compile(r"^(\d{4})(\d{4})$")
_ENV_LINE_RE: Final[re.Pattern[str]] = re.compile(r"^\s*(?:export\s+)?([A-Za-z_][A-Za-z0-9_]*)\s*=\s*(.*?)\s*$")
#: Start of a trailing comment on an unquoted ``.env`` value (``value # comment``).
_ENV_COMMENT_RE: Final[re.Pattern[str]] = re.compile(r"\s+#")

#: FootyStats field -> :class:`MatchRecord` statistic field (``team_a`` = home, ``team_b`` = away).
_STAT_FIELDS: Final[dict[str, str]] = {
    "home_shots": "team_a_shots",
    "away_shots": "team_b_shots",
    "home_shots_target": "team_a_shotsOnTarget",
    "away_shots_target": "team_b_shotsOnTarget",
    "home_corners": "team_a_corners",
    "away_corners": "team_b_corners",
    "home_xg": "team_a_xg",
    "away_xg": "team_b_xg",
}

#: Market -> selection -> FootyStats odds field.
_ODDS_FIELDS: Final[dict[Market, dict[str, str]]] = {
    Market.MATCH_ODDS: {"1": "odds_ft_1", "X": "odds_ft_x", "2": "odds_ft_2"},
    Market.OVER_UNDER_15: {"Over 1.5": "odds_ft_over15", "Under 1.5": "odds_ft_under15"},
    Market.OVER_UNDER_25: {"Over 2.5": "odds_ft_over25", "Under 2.5": "odds_ft_under25"},
    Market.OVER_UNDER_35: {"Over 3.5": "odds_ft_over35", "Under 3.5": "odds_ft_under35"},
    Market.BTTS: {"GG": "odds_btts_yes", "NG": "odds_btts_no"},
}

Clock = Callable[[], datetime]


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


# =============================================================================
# Errors
# =============================================================================
class FootyStatsError(HarvestError):
    """The API answered but refused the request or returned an unusable envelope
    (``success: false``, no ``data`` list, a competition missing from the account)."""


class MissingApiKeyError(HarvestError):
    """No FootyStats API key could be resolved (``--key``, environment, ``.env``)."""

    def __init__(
        self,
        message: str,
        *,
        key_env: str = "FOOTYSTATS_API_KEY",
        env_file: Path | None = DEFAULT_ENV_FILE,
        league_names: Sequence[str] | None = None,
    ) -> None:
        super().__init__(message, url="", status=None, attempts=0)
        self.key_env = key_env
        self.env_file = env_file
        #: FootyStats names of the leagues the account must hold: the configured competitions
        #: (``config/competitions.toml`` through :func:`configured_league_names`) unless given.
        self.league_names: tuple[str, ...] = tuple(league_names) if league_names is not None else tuple(configured_league_names())

    def __str__(self) -> str:
        return Exception.__str__(self)

    @property
    def instructions(self) -> str:
        """How to provide the key (printed by the CLI), naming the leagues of the configured competitions."""
        where = f" or in {self.env_file} as {self.key_env}=<key>" if self.env_file is not None else ""
        return (
            f"{self}. Get the key from footystats.org (your account -> API), then pass it with --key, "
            f"set the environment variable {self.key_env}{where}. On footystats.org also add the competitions "
            "to your account (Choose leagues): " + ", ".join(self.league_names) + "."
        )


# =============================================================================
# API key: resolution and masking
# =============================================================================
def read_env_file(path: Path) -> dict[str, str]:
    """Parse ``KEY=VALUE`` lines of a ``.env`` file (no dependency).

    Blank lines and ``#`` comments are ignored and an optional ``export`` prefix
    is accepted. A value wrapped in single or double quotes is unwrapped and
    whatever follows the closing quote (typically a ``# comment``) is dropped; an
    unquoted value stops at the first ``#`` preceded by whitespace. The file may
    be UTF-8 with a BOM (what Windows Notepad writes as "UTF-8 with BOM") and
    may use CRLF line endings. A missing or unreadable file yields ``{}``.
    """
    try:
        text = Path(path).read_text(encoding="utf-8-sig")
    except (OSError, UnicodeDecodeError):
        return {}
    out: dict[str, str] = {}
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        found = _ENV_LINE_RE.match(line)
        if found is None:
            continue
        out[found.group(1)] = _env_value(found.group(2))
    return out


def _env_value(raw: str) -> str:
    """Value of a ``.env`` assignment: a quoted value is unwrapped first, then any trailing ``# comment`` goes.

    ``'"abc" # key'`` -> ``'abc'``, ``"'a # b' # c"`` -> ``'a # b'``, ``'value # note'``
    -> ``'value'``, ``'# only a comment'`` -> ``''``, ``'a#b'`` -> ``'a#b'``. An
    unterminated quote is kept literally (minus a trailing comment).
    """
    if raw[:1] in ("'", '"'):
        closing = raw.find(raw[0], 1)
        if closing != -1:
            return raw[1:closing]
    if raw.startswith("#"):
        return ""
    return _ENV_COMMENT_RE.split(raw, maxsplit=1)[0].rstrip()


def find_api_key(
    explicit: str | None = None,
    *,
    key_env: str = "FOOTYSTATS_API_KEY",
    environ: Mapping[str, str] | None = None,
    env_file: Path | None = DEFAULT_ENV_FILE,
) -> str | None:
    """The API key, or ``None`` when none is configured (never raises).

    Precedence: ``explicit`` (the ``--key`` argument), the environment variable
    ``key_env``, the ``key_env`` entry of ``env_file`` (``None`` disables the
    file lookup). Blank values count as missing.
    """
    if explicit is not None and str(explicit).strip():
        return str(explicit).strip()
    source = environ if environ is not None else os.environ
    from_env = source.get(key_env, "")
    if from_env and from_env.strip():
        return from_env.strip()
    if env_file is not None:
        from_file = read_env_file(env_file).get(key_env, "")
        if from_file and from_file.strip():
            return from_file.strip()
    return None


def resolve_api_key(
    explicit: str | None = None,
    *,
    key_env: str = "FOOTYSTATS_API_KEY",
    environ: Mapping[str, str] | None = None,
    env_file: Path | None = DEFAULT_ENV_FILE,
    endpoints: FootyStatsEndpoints | None = None,
) -> str:
    """Like :func:`find_api_key` but raises :class:`MissingApiKeyError` with instructions when no key is found.

    ``endpoints`` (the ``[footystats]`` section) lets the instructions name the
    leagues of competitions configured only in ``[footystats.league_names]``.
    """
    key = find_api_key(explicit, key_env=key_env, environ=environ, env_file=env_file)
    if key is None:
        raise MissingApiKeyError(
            "FootyStats API key not configured",
            key_env=key_env,
            env_file=env_file,
            league_names=configured_league_names(endpoints),
        )
    return key


def mask_key(text: Any, key: str | None = None) -> str:
    """``text`` with any API key replaced by ``***``.

    Masks ``key=<value>`` query parameters and ``'key': '<value>'`` mappings
    whatever the value, plus every literal occurrence of ``key`` when given.
    """
    out = _KEY_IN_TEXT_RE.sub(lambda m: f"{m.group(1)}{MASK}", str(text))
    if key:
        out = out.replace(key, MASK)
    return out


def _proxy_from_environment(environ: Mapping[str, str] | None = None) -> str | None:
    """``HTTPS_PROXY`` / ``ALL_PROXY`` (either case), the way ``httpx`` would honour it."""
    source = environ if environ is not None else os.environ
    for name in ("HTTPS_PROXY", "https_proxy", "ALL_PROXY", "all_proxy"):
        value = source.get(name)
        if value:
            return value
    return None


class KeyInjectingTransport(httpx.AsyncBaseTransport):
    """Transport wrapper that adds ``key=<api key>`` to every request URL.

    The key is appended *below* :class:`AsyncHarvesterClient`: the URL that the
    client hashes for its cache, writes into cache records, puts in log lines
    and in :class:`HarvestError` messages, and the URL ``httpx`` itself logs
    (``HTTP Request: GET ...``), is the original key-free request. Only the inner
    transport (the network, or a test ``MockTransport``) sees the key. A URL that
    already carries the parameter is sent unchanged.
    """

    def __init__(self, key: str, inner: httpx.AsyncBaseTransport | None = None, *, param: str = KEY_PARAM) -> None:
        if not str(key).strip():
            raise ValueError("the API key must not be empty")
        self._key = str(key).strip()
        self._inner = inner if inner is not None else httpx.AsyncHTTPTransport(proxy=_proxy_from_environment())
        self.param = param

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        url = request.url
        if self.param not in url.params:
            url = url.copy_add_param(self.param, self._key)
        keyed = httpx.Request(request.method, url, headers=request.headers, stream=request.stream, extensions=request.extensions)
        return await self._inner.handle_async_request(keyed)

    async def aclose(self) -> None:
        await self._inner.aclose()

    def __repr__(self) -> str:  # pragma: no cover - trivial, keeps the key out of reprs
        return f"KeyInjectingTransport(param={self.param!r}, key={MASK!r})"


# =============================================================================
# Competition codes and season labels (pure)
# =============================================================================
def normalize_code(code: "Girone | str") -> str:
    """Competition code as the upper-case string the frame stores (``Girone.A`` -> ``'A'``, ``'sb'`` -> ``'SB'``)."""
    if isinstance(code, Girone):
        return str(code.value)
    text = str(code).strip().upper()
    if not text:
        raise ValueError("empty competition code")
    return text


def schema_supports(code: "Girone | str") -> bool:
    """Whether :class:`~src.common.schemas.Girone` accepts ``code`` (so matches of that competition can be stored)."""
    try:
        Girone.parse(normalize_code(code))
    except ValueError:
        return False
    return True


def footystats_league_name(code: "Girone | str", endpoints: FootyStatsEndpoints | None = None) -> str:
    """``name`` field of ``/league-list`` for a competition code.

    Sources, in order: the ``footystats_name`` of the
    :class:`~src.common.competitions.Competition` record (``config/competitions.toml``)
    when that module exists, ``endpoints.league_names`` (``[footystats.league_names]``),
    the built-in table. Raises ``ValueError`` for an unknown code.
    """
    normalized = normalize_code(code)
    if _shared_competition is not None:
        try:
            shared = getattr(_shared_competition(normalized), "footystats_name", None)
        except (KeyError, ValueError, LookupError):
            shared = None
        if isinstance(shared, str) and shared.strip():
            return shared.strip()
    if endpoints is not None:
        configured = endpoints.league_names.get(normalized)
        if configured:
            return str(configured)
    try:
        return _FALLBACK_LEAGUE_NAMES[normalized]
    except KeyError as exc:
        known = sorted({*DEFAULT_COMPETITIONS, *(endpoints.league_names if endpoints is not None else ()), *_FALLBACK_LEAGUE_NAMES})
        raise ValueError(
            f"unknown competition code {code!r}: expected one of {known} "
            "(a code of config/competitions.toml or one configured in [footystats.league_names])"
        ) from exc


def configured_league_names(
    endpoints: FootyStatsEndpoints | None = None, codes: Iterable["Girone | str"] | None = None
) -> list[str]:
    """FootyStats league names of the configured competitions, unique and in configuration order.

    ``codes`` defaults to :data:`DEFAULT_COMPETITIONS` (every :class:`Girone`
    member, i.e. ``config/competitions.toml``); a code whose name cannot be
    resolved is skipped and the built-in table is the last resort when nothing
    resolves. Used by :class:`MissingApiKeyError` for its instructions.
    """
    names: list[str] = []
    for code in DEFAULT_COMPETITIONS if codes is None else codes:
        try:
            name = footystats_league_name(code, endpoints)
        except ValueError:
            continue
        if name not in names:
            names.append(name)
    return names or list(_FALLBACK_LEAGUE_NAMES.values())


def competition_season_ids(code: "Girone | str", endpoints: FootyStatsEndpoints | None = None) -> dict[str, int]:
    """Known FootyStats season ids of a competition (season label -> id), ``{}`` when none is configured.

    The ``season_ids`` of the :class:`~src.common.competitions.Competition` record
    (``[competition.season_ids]`` in ``config/competitions.toml``) come first;
    ``[footystats.season_ids]`` of ``endpoints.toml`` adds the labels the record
    does not list. These ids are the fallback of
    :meth:`FootyStatsClient.resolve_seasons`; the live ``/league-list`` always wins.
    """
    normalized = normalize_code(code)
    ids: dict[str, int] = {}
    if _shared_competition is not None:
        try:
            shared = getattr(_shared_competition(normalized), "season_ids", None)
        except (KeyError, ValueError, LookupError):
            shared = None
        if isinstance(shared, Mapping):
            ids.update({str(label): int(season_id) for label, season_id in shared.items()})
    if endpoints is not None:
        for label, season_id in endpoints.known_season_ids(normalized).items():
            ids.setdefault(str(label), int(season_id))
    return ids


def competition_label(codes: Iterable["Girone | str"]) -> str:
    """Human label of the leagues behind ``codes``: their ``family`` names in configuration order, ``' + '``-joined.

    ``('A', 'B', 'C')`` -> ``'Serie C'``, ``('SB',)`` -> ``'Serie B'``, ``('A', 'SB')``
    -> ``'Serie C + Serie B'``. A code the schema does not know contributes
    itself (after the known ones, sorted); an empty iterable gives ``''``.
    """
    wanted = {normalize_code(code) for code in codes}
    labels: list[str] = []
    for member in Girone:
        if str(member.value) not in wanted:
            continue
        try:
            family = str(member.family)
        except LookupError:  # pragma: no cover - only a hand-built enum lacks the competition table
            family = str(member.value)
        if family not in labels:
            labels.append(family)
    labels.extend(sorted(wanted - {str(member.value) for member in Girone}))
    return " + ".join(labels)


def season_label_from_footystats_year(year: Any) -> str | None:
    """FootyStats ``season[].year`` -> public label: ``20262027`` -> ``'2026/27'``, ``2026`` -> ``'2026'``.

    Both ints and digit strings are accepted; anything else (or a pair of years
    that are not consecutive) yields ``None``.
    """
    if year is None or isinstance(year, bool):
        return None
    text = str(year).strip()
    if _PLAIN_YEAR_RE.match(text):
        return text
    found = _DOUBLE_YEAR_RE.match(text)
    if found is None:
        return None
    start, end = int(found.group(1)), int(found.group(2))
    if end != start + 1:
        return None
    return f"{start}/{end % 100:02d}"


def _season_start_year(label: str) -> int:
    """First year of a season label (``'2026/27'`` -> 2026, ``'2026'`` -> 2026); -1 when not a season."""
    found = re.match(r"^\s*(\d{4})", str(label))
    return int(found.group(1)) if found else -1


def season_completed(label: str, today: date | None = None) -> bool:
    """Whether the season ``label`` has ended (its payloads are immutable, cached as *stable*).

    ``'2025/26'`` ends on 30 June 2026; a calendar-year label ``'2025'`` ends on
    31 December 2025. Labels that are not seasons count as still running.
    """
    today = today if today is not None else _utc_now().date()
    text = str(label).strip()
    if _PLAIN_YEAR_RE.match(text):
        return today.year > int(text)
    try:
        _, last = season_bounds(text)
    except ValueError:
        return False
    return last < today


# =============================================================================
# Payload records
# =============================================================================
@dataclass(frozen=True, slots=True)
class SeasonRef:
    """One season of a competition: public label and FootyStats ``season_id``."""

    label: str
    season_id: int

    @property
    def start_year(self) -> int:
        return _season_start_year(self.label)


@dataclass(frozen=True)
class LeagueInfo:
    """One entry of ``/league-list``: ``seasons`` maps the season label to its id."""

    name: str
    league_name: str
    country: str
    seasons: Mapping[str, int] = field(default_factory=dict)

    def newest_seasons(self, n: int) -> list[SeasonRef]:
        """The ``n`` most recent seasons, newest first."""
        refs = [SeasonRef(label, int(season_id)) for label, season_id in self.seasons.items()]
        refs.sort(key=lambda ref: (-ref.start_year, -ref.season_id))
        return refs[: max(int(n), 0)]

    @classmethod
    def from_payload(cls, entry: Mapping[str, Any]) -> "LeagueInfo | None":
        name = str(entry.get("name") or "").strip()
        if not name:
            return None
        seasons: dict[str, int] = {}
        for season in entry.get("season") or ():
            if not isinstance(season, Mapping):
                continue
            label = season_label_from_footystats_year(season.get("year"))
            season_id = _as_int_or_none(season.get("id"))
            if label is None or season_id is None:
                log.debug("league season entry skipped", extra={"league": name, "entry": dict(season)})
                continue
            seasons[label] = season_id
        return cls(
            name=name,
            league_name=str(entry.get("league_name") or name).strip(),
            country=str(entry.get("country") or "").strip(),
            seasons=seasons,
        )


@dataclass(frozen=True, slots=True)
class TeamInfo:
    """One entry of ``/league-teams`` (the spellings FootyStats knows for a club)."""

    id: int
    name: str
    clean_name: str
    english_name: str = ""
    alt_names: tuple[str, ...] = ()

    @staticmethod
    def _distinct(ordered: Iterable[str]) -> tuple[str, ...]:
        seen: set[str] = set()
        out: list[str] = []
        for spelling in ordered:
            text = str(spelling or "").strip()
            key = text.casefold()
            if text and key not in seen:
                seen.add(key)
                out.append(text)
        return tuple(out)

    @property
    def spellings(self) -> tuple[str, ...]:
        """Distinct non-empty spellings in matching order: clean name, name, english name, alternatives."""
        return self._distinct([self.clean_name, self.name, self.english_name, *self.alt_names])

    @property
    def identity_spellings(self) -> tuple[str, ...]:
        """The spellings that *identify* the club: clean name, name and english name, without ``alt_names``.

        Nicknames in ``alt_names`` are shared between clubs (``'The Blues'`` is
        Everton's and Chelsea's), so :func:`write_registry_update` never uses them
        to decide which dictionary entry a FootyStats team is; they are only
        written as aliases, and only when no other club already answers to them.
        """
        return self._distinct([self.clean_name, self.name, self.english_name])

    @classmethod
    def from_payload(cls, entry: Mapping[str, Any]) -> "TeamInfo | None":
        team_id = _as_int_or_none(entry.get("id"))
        name = str(entry.get("name") or "").strip()
        clean = str(entry.get("cleanName") or entry.get("clean_name") or name).strip()
        if team_id is None or not (name or clean):
            return None
        alt: list[str] = []
        raw_alt = entry.get("alt_names") or ()
        if isinstance(raw_alt, str):
            raw_alt = [raw_alt]
        for item in raw_alt:
            alt.extend(part.strip() for part in str(item).split("/") if part.strip())
        return cls(id=team_id, name=name or clean, clean_name=clean or name, english_name=str(entry.get("english_name") or "").strip(), alt_names=tuple(alt))


def _as_int_or_none(value: Any) -> int | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _as_float_or_none(value: Any) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return None if out != out else out  # NaN -> None


def _utc_datetime_from_epoch(value: Any) -> datetime | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        seconds = float(value)
    except (TypeError, ValueError):
        return None
    if seconds <= 0:
        return None
    try:
        return datetime.fromtimestamp(seconds, tz=timezone.utc)
    except (OverflowError, OSError, ValueError):
        return None


# =============================================================================
# Client
# =============================================================================
class FootyStatsClient:
    """FootyStats API client on top of :class:`AsyncHarvesterClient`.

    Adds the key through :class:`KeyInjectingTransport`, unwraps the envelope,
    paginates, and keeps the hourly budget in view (``metadata.request_remaining``).
    ``transport`` is the *inner* transport (tests inject ``httpx.MockTransport``);
    production uses ``httpx.AsyncHTTPTransport`` honouring ``HTTPS_PROXY``.
    """

    def __init__(
        self,
        *,
        api_key: str,
        endpoints: FootyStatsEndpoints | None = None,
        http: HttpConfig | None = None,
        cache: RawCache | None = None,
        transport: httpx.AsyncBaseTransport | None = None,
        sleeper: Sleeper = asyncio.sleep,
        rng: random.Random | None = None,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        key = str(api_key or "").strip()
        if not key:
            raise MissingApiKeyError("FootyStats API key is empty")
        self._key = key
        self.endpoints = endpoints if endpoints is not None else FootyStatsEndpoints()
        self.http = http if http is not None else HttpConfig()
        self._client = AsyncHarvesterClient(
            base_url=self.endpoints.base_url,
            http=self.http,
            cache=cache,
            transport=KeyInjectingTransport(key, transport),
            user_agents=(FOOTYSTATS_USER_AGENT,),
            sleeper=sleeper,
            rng=rng,
            clock=clock,
        )
        self.pages = 0
        self.api_errors = 0
        self.request_limit: int | None = None
        self.request_remaining: int | None = None
        self._budget_warned = False

    # ------------------------------------------------------------ lifecycle
    async def __aenter__(self) -> "FootyStatsClient":
        await self._client.__aenter__()
        return self

    async def __aexit__(self, *exc_info: object) -> None:
        await self.close()

    async def close(self) -> None:
        await self._client.close()

    # -------------------------------------------------------------- helpers
    @property
    def cache(self) -> RawCache | None:
        return self._client.cache

    def client_stats(self) -> ClientStats:
        """Copy of the HTTP counters (requests, retries, cache hits, errors)."""
        inner = self._client.stats
        return ClientStats(requests=inner.requests, retries=inner.retries, cache_hits=inner.cache_hits, errors=inner.errors)

    @property
    def stats(self) -> dict[str, int]:
        """HTTP counters plus ``pages`` fetched, ``api_errors`` (``success: false``) and the hourly budget seen."""
        out = self.client_stats().as_dict()
        out.update(pages=self.pages, api_errors=self.api_errors)
        if self.request_remaining is not None:
            out["request_remaining"] = self.request_remaining
        return out

    def mask(self, text: Any) -> str:
        """``text`` with this client's key (and any ``key=`` parameter) masked."""
        return mask_key(text, self._key)

    def _masked_error(self, exc: HarvestError) -> HarvestError:
        """``exc`` itself when it is clean, otherwise a copy of the same type with the key masked."""
        if self._key not in str(exc):
            return exc
        message = self.mask(exc.args[0] if exc.args else "")
        try:
            clone = type(exc)(message, url=self.mask(exc.url), status=exc.status, attempts=exc.attempts)
        except TypeError:  # a subclass with its own constructor (MissingApiKeyError)
            clone = HarvestError(message, url=self.mask(exc.url), status=exc.status, attempts=exc.attempts)
        return clone

    def _evict(self, url: str, params: Mapping[str, Any] | None) -> None:
        """Forget a cached envelope that turned out to be unusable (never cache a refusal)."""
        cache = self._client.cache
        if cache is None:
            return
        try:
            cache.path_for(url, params).unlink(missing_ok=True)
        except OSError as exc:  # pragma: no cover - defensive
            log.warning("could not evict cache entry", extra={"url": url, "error": str(exc)})

    def _note_metadata(self, metadata: Any) -> None:
        if not isinstance(metadata, Mapping):
            return
        limit = _as_int_or_none(metadata.get("request_limit"))
        remaining = _as_int_or_none(metadata.get("request_remaining"))
        if limit is not None:
            self.request_limit = limit
        if remaining is not None:
            self.request_remaining = remaining
            budget = self.request_limit or int(self.endpoints.max_requests_per_hour)
            if remaining <= max(10, budget // 20) and not self._budget_warned:
                self._budget_warned = True
                log.warning(
                    "FootyStats hourly request budget nearly exhausted",
                    extra={"request_remaining": remaining, "request_limit": budget},
                )

    # ----------------------------------------------------------------- core
    async def get(self, path: str, params: Mapping[str, Any] | None = None, *, stable: bool = False) -> dict[str, Any]:
        """GET one page and return the whole envelope (``success``, ``pager``, ``metadata``, ``data``).

        Raises :class:`FootyStatsError` on ``success: false`` or when ``data`` is
        not a list (the refused envelope is evicted from the cache); transport and
        HTTP failures surface as the :class:`HarvestError` of the underlying
        client, with the key masked should it ever appear in a message.
        """
        url = self._client.build_url(path)
        try:
            payload = await self._client.get_json(path, params, stable=stable)
        except HarvestError as exc:
            self.api_errors += 1
            raise self._masked_error(exc) from None
        if not isinstance(payload, Mapping):
            self._evict(url, params)
            self.api_errors += 1
            raise FootyStatsError(f"unexpected response body ({type(payload).__name__}, not a JSON object)", url=url)
        self._note_metadata(payload.get("metadata"))
        if not payload.get("success", False):
            self._evict(url, params)
            self.api_errors += 1
            message = self.mask(payload.get("message") or "success=false")
            raise FootyStatsError(f"API refused the request: {message}", url=url)
        data = payload.get("data")
        if not isinstance(data, list):
            self._evict(url, params)
            self.api_errors += 1
            raise FootyStatsError(f"envelope without a data list ({type(data).__name__})", url=url)
        return dict(payload)

    async def get_all(self, path: str, params: Mapping[str, Any] | None = None, *, stable: bool = False) -> list[Any]:
        """Every ``data`` row of ``path``: pages ``1..pager.max_page`` concatenated."""
        base = dict(params or {})
        rows: list[Any] = []
        page = 1
        while True:
            payload = await self.get(path, {**base, "page": page}, stable=stable)
            data = payload["data"]
            rows.extend(data)
            self.pages += 1
            pager = payload.get("pager") if isinstance(payload.get("pager"), Mapping) else {}
            max_page = _as_int_or_none(pager.get("max_page")) or 1
            current = _as_int_or_none(pager.get("current_page")) or page
            if current >= max_page or not data:
                break
            if page >= MAX_PAGES:
                raise FootyStatsError(f"pagination did not terminate after {page} pages", url=self._client.build_url(path))
            page = current + 1
        return rows

    # ------------------------------------------------------------ endpoints
    async def league_list(self, chosen_only: bool = True) -> list[LeagueInfo]:
        """``/league-list``: the leagues of the account (``chosen_only``) or every league FootyStats covers."""
        params = {"chosen_leagues_only": "true"} if chosen_only else {}
        entries = await self.get_all(self.endpoints.league_list, params)
        leagues = [LeagueInfo.from_payload(entry) for entry in entries if isinstance(entry, Mapping)]
        return [league for league in leagues if league is not None]

    async def resolve_seasons(self, competition_codes: Iterable["Girone | str"], seasons: int = 3) -> dict[str, list[SeasonRef]]:
        """Newest ``seasons`` seasons (label + id) of every competition code, newest first.

        Competitions are looked up by :func:`footystats_league_name` among the
        account's *chosen* leagues. A competition that is **not chosen** falls
        back to its configured season ids (:func:`competition_season_ids`:
        ``[competition.season_ids]`` of ``config/competitions.toml``, then
        ``[footystats.season_ids]`` of ``endpoints.toml``) with a warning naming
        the league the user should add on footystats.org; only a competition with
        no configured ids raises :class:`FootyStatsError` with the same advice.
        When ``/league-list`` itself cannot be reached the configured ids are
        used for every requested competition (logged as a warning) and the error
        propagates when one of them has none.
        """
        if int(seasons) < 1:
            raise ValueError("seasons must be >= 1")
        codes = list(dict.fromkeys(normalize_code(code) for code in competition_codes))
        names = {code: footystats_league_name(code, self.endpoints) for code in codes}
        list_url = self.endpoints.url_league_list()
        fallback = {code: competition_season_ids(code, self.endpoints) for code in codes}

        def from_configured(code: str) -> list[SeasonRef]:
            return LeagueInfo(names[code], names[code], "", fallback[code]).newest_seasons(seasons)

        try:
            leagues = await self.league_list(chosen_only=True)
        except HarvestError as exc:
            if codes and all(fallback.values()):
                log.warning(
                    "league-list unavailable, using configured season ids",
                    extra={"error": self.mask(exc), "competitions": codes},
                )
                return {code: from_configured(code) for code in codes}
            raise
        by_name = {league.name.casefold(): league for league in leagues}
        out: dict[str, list[SeasonRef]] = {}
        missing: list[str] = []
        for code in codes:
            league = by_name.get(names[code].casefold())
            if league is None:
                if fallback[code]:
                    out[code] = from_configured(code)
                    log.warning(
                        "league not among the account's chosen leagues, using configured season ids",
                        extra={
                            "competition": code,
                            "league": names[code],
                            "seasons": [ref.label for ref in out[code]],
                            "advice": f"add {names[code]!r} on footystats.org (your account -> Choose leagues)",
                        },
                    )
                    continue
                missing.append(names[code])
                continue
            refs = league.newest_seasons(seasons)
            if not refs:
                raise FootyStatsError(f"no season listed for {names[code]!r}", url=list_url)
            out[code] = refs
        if missing:
            raise FootyStatsError(
                "competitions not among the account's chosen leagues and without configured season ids: "
                + ", ".join(repr(name) for name in missing)
                + " -- add them on footystats.org (your account -> Choose leagues) and retry; chosen now: "
                + ", ".join(sorted(league.name for league in leagues)),
                url=list_url,
            )
        return {code: out[code] for code in codes if code in out}

    async def league_teams(self, season_id: int, *, stable: bool = False) -> list[TeamInfo]:
        """``/league-teams`` of one season (``stable`` for a completed season: long cache TTL)."""
        entries = await self.get_all(self.endpoints.league_teams, {"season_id": int(season_id)}, stable=stable)
        teams = [TeamInfo.from_payload(entry) for entry in entries if isinstance(entry, Mapping)]
        return [team for team in teams if team is not None]

    async def league_matches(self, season_id: int, *, stable: bool = False) -> list[dict[str, Any]]:
        """Raw ``/league-matches`` objects of one season, every page."""
        rows = await self.get_all(self.endpoints.league_matches, {"season_id": int(season_id)}, stable=stable)
        return [dict(row) for row in rows if isinstance(row, Mapping)]


# =============================================================================
# Parsers (pure)
# =============================================================================
@dataclass
class ParseReport:
    """Counters :func:`parse_match` accumulates over one season, logged once per season."""

    complete: int = 0
    fixtures: int = 0
    incomplete_past: int = 0
    skipped_status: Counter = field(default_factory=Counter)
    skipped_invalid: int = 0
    missing_shots: int = 0
    missing_xg: int = 0

    def as_dict(self) -> dict[str, Any]:
        return {
            "complete": self.complete,
            "fixtures": self.fixtures,
            "incomplete_past": self.incomplete_past,
            "skipped_status": dict(self.skipped_status),
            "skipped_invalid": self.skipped_invalid,
            "missing_shots": self.missing_shots,
            "missing_xg": self.missing_xg,
        }


def parse_stats(m: Mapping[str, Any]) -> tuple[dict[str, float | None], bool, bool]:
    """Shots / shots on target / corners / xG of a match, with FootyStats' missing-value rules.

    Negative values (``-1``) are missing. Both teams with ``0`` shots means the
    statistics were never collected: shots, shots on target and (when also
    ``0``) corners are set to ``None``. Both teams with ``0`` xG means xG is not
    covered for this competition: both xG fields are ``None``. Returns the
    statistics plus two flags, ``shots_missing`` and ``xg_missing``.
    """
    values: dict[str, float | None] = {}
    for target, source in _STAT_FIELDS.items():
        number = _as_float_or_none(m.get(source))
        values[target] = None if number is None or number < 0 else number
    shots_missing = not values["home_shots"] and not values["away_shots"]
    if shots_missing:
        for name in ("home_shots", "away_shots", "home_shots_target", "away_shots_target"):
            values[name] = None
        if not values["home_corners"] and not values["away_corners"]:
            values["home_corners"] = values["away_corners"] = None
    xg_missing = not values["home_xg"] and not values["away_xg"]
    if xg_missing:
        values["home_xg"] = values["away_xg"] = None
    return values, shots_missing, xg_missing


def _round_of(m: Mapping[str, Any]) -> int:
    for key in ("game_week", "revised_game_week"):
        value = _as_int_or_none(m.get(key))
        if value is not None and value > 0:
            return value
    return 0


def _team_name_of(names: Mapping[int, str], team_id: int | None, raw: Any) -> str:
    if team_id is not None and team_id in names and str(names[team_id]).strip():
        return str(names[team_id]).strip()
    return str(raw or "").strip()


def parse_match(
    m: Mapping[str, Any],
    code: "Girone | str",
    names: Mapping[int, str],
    *,
    now: datetime | None = None,
    report: ParseReport | None = None,
) -> MatchRecord | Fixture | None:
    """One ``/league-matches`` object -> :class:`MatchRecord`, :class:`Fixture` or ``None``.

    * ``status == 'complete'`` -> :class:`MatchRecord` ``FS-<id>`` dated with the
      UTC day of ``date_unix``, round ``game_week`` (``revised_game_week`` when the
      former is missing, else 0), team names from ``names[homeID]`` /
      ``names[awayID]`` (falling back to ``home_name`` / ``away_name``), goals and
      the statistics of :func:`parse_stats`;
    * ``status == 'incomplete'`` with a kick-off after ``now`` (UTC, default: the
      current time) or without goals -> :class:`Fixture`; an incomplete match in
      the past (in play, or postponed without a new date) is dropped;
    * ``suspended`` / ``canceled`` / anything else -> ``None``.

    ``code`` is the competition stored in ``girone``; it must be a code the
    schema accepts. ``report`` accumulates counters for one log line per season.
    """
    report = report if report is not None else ParseReport()
    match_id_raw = _as_int_or_none(m.get("id"))
    if match_id_raw is None:
        report.skipped_invalid += 1
        log.warning("match without id skipped", extra={"competition": normalize_code(code)})
        return None
    match_id = f"FS-{match_id_raw}"
    status = str(m.get("status") or "").strip().lower()
    kickoff = _utc_datetime_from_epoch(m.get("date_unix"))
    if kickoff is None:
        report.skipped_invalid += 1
        log.warning("match without kick-off time skipped", extra={"match_id": match_id})
        return None
    home = _team_name_of(names, _as_int_or_none(m.get("homeID")), m.get("home_name"))
    away = _team_name_of(names, _as_int_or_none(m.get("awayID")), m.get("away_name"))
    if not home or not away:
        report.skipped_invalid += 1
        log.warning("match without team names skipped", extra={"match_id": match_id})
        return None
    if home == away:
        report.skipped_invalid += 1
        log.warning("match with identical teams skipped", extra={"match_id": match_id, "team": home})
        return None
    girone = normalize_code(code)
    round_no = _round_of(m)
    if status == "complete":
        home_goals = _as_int_or_none(m.get("homeGoalCount"))
        away_goals = _as_int_or_none(m.get("awayGoalCount"))
        if home_goals is None or away_goals is None or home_goals < 0 or away_goals < 0:
            report.skipped_invalid += 1
            log.warning("complete match without a score skipped", extra={"match_id": match_id})
            return None
        stats, shots_missing, xg_missing = parse_stats(m)
        report.complete += 1
        report.missing_shots += int(shots_missing)
        report.missing_xg += int(xg_missing)
        return MatchRecord(
            match_id=match_id,
            date=kickoff.date(),
            round=round_no,
            girone=girone,
            home_team=home,
            away_team=away,
            home_goals=home_goals,
            away_goals=away_goals,
            **stats,
        )
    if status == "incomplete":
        reference = now if now is not None else _utc_now()
        if reference.tzinfo is None:
            reference = reference.replace(tzinfo=timezone.utc)
        goals_absent = m.get("homeGoalCount") is None or m.get("awayGoalCount") is None
        if kickoff > reference or goals_absent:
            report.fixtures += 1
            return Fixture(match_id=match_id, date=kickoff.date(), round=round_no, girone=girone, home_team=home, away_team=away)
        report.incomplete_past += 1
        log.debug("incomplete match in the past dropped", extra={"match_id": match_id, "kickoff": kickoff.isoformat()})
        return None
    report.skipped_status[status or "unknown"] += 1
    return None


def _odds_value(value: Any) -> float | None:
    number = _as_float_or_none(value)
    return number if number is not None and number > 1.0 else None


def parse_odds(m: Mapping[str, Any], match_id: str, bookmaker: str = FOOTYSTATS_BOOKMAKER) -> list[OddsQuote]:
    """Pre-match quotes of a match: 1X2, Over/Under 1.5-2.5-3.5 and both teams to score.

    A market is skipped as soon as one of its selections is unavailable
    (``0``, ``None`` or odds ``<= 1.0``), so every :class:`OddsQuote` is complete.
    """
    quotes: list[OddsQuote] = []
    for market, fields in _ODDS_FIELDS.items():
        odds = {selection: _odds_value(m.get(source)) for selection, source in fields.items()}
        if any(value is None for value in odds.values()):
            continue
        quotes.append(OddsQuote(match_id=match_id, bookmaker=bookmaker, market=market, odds={k: float(v) for k, v in odds.items() if v is not None}))
    return quotes


# =============================================================================
# Team mapping
# =============================================================================
@dataclass(frozen=True, slots=True)
class MappedTeam:
    """How one FootyStats team resolved to a canonical name."""

    team_id: int
    canonical: str
    method: str  # 'exact' | 'alias' | 'persisted' | 'fuzzy' | 'none'
    current: bool
    historical: bool


class FootyStatsTeamMapper:
    """FootyStats team ids -> canonical names of the team dictionary.

    A team's spellings (``cleanName``, ``name``, ``english_name``, ``alt_names``,
    in that order) are looked up through ``matcher`` (built on ``registry``, which
    should carry the aliases of the older dictionaries). Precedence:

    1. an **exact** or **alias** hit of any spelling (dictionary knowledge wins);
    2. the **persisted** canonical of the same id from ``mapping_path``;
    3. the best **fuzzy** hit;
    4. nothing: the team keeps ``cleanName`` and is reported as *unmapped*.

    For the current season of a Serie C girone the lookup is restricted to that
    girone's composition first; past seasons and competitions outside the
    dictionary (Serie B) use unrestricted lookups, which may return historical
    names (clubs that left the perimeter). Every resolution is merged into the
    persisted mapping ``{team_id: {canonical, clean_name, competition, seasons}}``
    (:meth:`save`), so ids map deterministically on the next run.
    """

    def __init__(
        self,
        registry: TeamRegistry,
        matcher: TeamMatcher | None = None,
        mapping_path: Path = DEFAULT_MAPPING_PATH,
    ) -> None:
        self.registry = registry
        self.matcher = matcher if matcher is not None else TeamMatcher(registry)
        self.mapping_path = Path(mapping_path)
        self._persisted: dict[int, dict[str, Any]] = self._load(self.mapping_path)
        self.resolved: dict[int, MappedTeam] = {}
        self.unmapped: Counter[str] = Counter()

    # -------------------------------------------------------------- storage
    @staticmethod
    def _load(path: Path) -> dict[int, dict[str, Any]]:
        if not path.exists():
            return {}
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            log.warning("team mapping unreadable, starting empty", extra={"path": str(path), "error": str(exc)})
            return {}
        table = raw.get("teams", raw) if isinstance(raw, Mapping) else {}
        out: dict[int, dict[str, Any]] = {}
        if isinstance(table, Mapping):
            for key, entry in table.items():
                team_id = _as_int_or_none(key)
                if team_id is None or not isinstance(entry, Mapping) or not str(entry.get("canonical") or "").strip():
                    continue
                out[team_id] = dict(entry)
        return out

    @property
    def persisted(self) -> dict[int, dict[str, Any]]:
        """The merged mapping as it would be written (id -> entry)."""
        return dict(sorted(self._persisted.items()))

    def save(self, path: Path | None = None) -> Path:
        """Write the merged mapping atomically (sorted by id); returns the path."""
        target = Path(path) if path is not None else self.mapping_path
        target.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "note": "FootyStats team id -> canonical name of config/teams_<season>.json. Maintained by "
            "src/ingestion/footystats.py (persisted ids beat fuzzy matches; exact/alias dictionary hits beat both). "
            "Edit 'canonical' to correct a mapping.",
            "teams": {str(team_id): entry for team_id, entry in sorted(self._persisted.items())},
        }
        handle = tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=target.parent, suffix=".tmp", delete=False)
        try:
            with handle:
                json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=False)
                handle.write("\n")
            os.replace(handle.name, target)
        except BaseException:
            Path(handle.name).unlink(missing_ok=True)
            raise
        log.info("team mapping saved", extra={"path": str(target), "teams": len(self._persisted)})
        return target

    # ----------------------------------------------------------- resolution
    def _lookup_girone(self, competition: str) -> Girone | None:
        """Girone to restrict the current-season lookup to (``None`` when the dictionary has no such group)."""
        try:
            girone = Girone.parse(competition)
        except ValueError:
            return None
        return girone if self.registry.teams_by_girone.get(girone) else None

    def _best(self, team: TeamInfo, girone: Girone | None) -> tuple[str, str] | None:
        """``(canonical, method)`` of the first exact/alias hit, else the best fuzzy hit, else ``None``."""
        fuzzy: tuple[int, str] | None = None
        for spelling in team.spellings:
            result = self.matcher.match(spelling, girone)
            if result.canonical is None:
                continue
            if result.method in {"exact", "alias"}:
                return result.canonical, result.method
            if fuzzy is None or result.score > fuzzy[0]:
                fuzzy = (result.score, result.canonical)
        return (fuzzy[1], "fuzzy") if fuzzy is not None else None

    def resolve(self, team: TeamInfo, competition: "Girone | str", season: str, *, current: bool = True) -> MappedTeam:
        """Canonical name of ``team`` in ``competition`` / ``season`` (see the class docstring for the precedence).

        A team already resolved during this run keeps its result (the mapping only
        records the season); so the same id maps to the same name across the
        seasons and competitions of one harvest.
        """
        code = normalize_code(competition)
        already = self.resolved.get(team.id)
        if already is not None:
            entry = self._persisted.get(team.id)
            if entry is not None and season not in entry.get("seasons", []):
                entry["seasons"] = sorted({*entry.get("seasons", []), season}, reverse=True)
            return already
        girone = self._lookup_girone(code) if current else None
        hit = self._best(team, girone)
        if hit is None and girone is not None:
            hit = self._best(team, None)
            if hit is not None and self.matcher.is_current(hit[0]):
                log.warning(
                    "team listed by FootyStats in another girone than the dictionary",
                    extra={"team": team.clean_name, "canonical": hit[0], "footystats_competition": code, "season": season},
                )
        persisted = self._persisted.get(team.id)
        if hit is not None and hit[1] in {"exact", "alias"}:
            canonical, method = hit
        elif persisted is not None:
            canonical, method = str(persisted["canonical"]).strip(), "persisted"
        elif hit is not None:
            canonical, method = hit
        else:
            canonical, method = team.clean_name or team.name, "none"
            self.unmapped[canonical] += 1
            log.warning(
                "unmapped FootyStats team kept with its clean name",
                extra={"team": team.clean_name, "team_id": team.id, "competition": code, "season": season, "spellings": list(team.spellings)},
            )
        if persisted is not None and str(persisted.get("canonical", "")).strip() != canonical:
            log.info(
                "persisted team mapping updated",
                extra={"team_id": team.id, "old": persisted.get("canonical"), "new": canonical, "method": method},
            )
        mapped = MappedTeam(
            team_id=team.id,
            canonical=canonical,
            method=method,
            current=self.matcher.is_current(canonical),
            historical=self.matcher.is_historical(canonical),
        )
        self.resolved[team.id] = mapped
        seasons = sorted(set(persisted.get("seasons", [])) | {season}, reverse=True) if persisted else [season]
        self._persisted[team.id] = {
            "canonical": canonical,
            "clean_name": team.clean_name,
            "competition": code,
            "seasons": seasons,
            # the file keeps how the name was originally determined, so a persisted hit stays auditable
            "method": (str(persisted.get("method") or "persisted") if method == "persisted" and persisted is not None else method),
        }
        return mapped

    def names_for(self, teams: Iterable[TeamInfo], competition: "Girone | str", season: str, *, current: bool = True) -> dict[int, str]:
        """``team id -> canonical name`` for a whole ``/league-teams`` listing."""
        return {team.id: self.resolve(team, competition, season, current=current).canonical for team in teams}

    # -------------------------------------------------------------- reports
    @property
    def historical_teams(self) -> tuple[str, ...]:
        """Canonical names resolved to historical teams (clubs that left the perimeter), sorted."""
        return tuple(sorted({m.canonical for m in self.resolved.values() if m.historical}))

    @property
    def unmapped_teams(self) -> dict[str, int]:
        """Clean names no dictionary knows, with how many times they were resolved."""
        return dict(sorted(self.unmapped.items()))


# =============================================================================
# Provider
# =============================================================================
@dataclass
class FootyStatsHarvest:
    """Outcome of :meth:`FootyStatsProvider.harvest`.

    ``matches`` is the validated 16-column frame of every competition and
    season harvested (no season column: :func:`season_of` of the date);
    ``records`` the same rows as :class:`MatchRecord` objects for callers that
    split them per competition. ``fixtures`` come from the current season of
    each competition only, deduplicated; ``odds`` quote those fixtures.
    ``seasons`` lists the season labels per competition code (newest first),
    ``n_matches_by_competition`` the completed matches per code and season and
    ``n_matches_by_season`` the same totals per season. ``failures`` records a
    season that could not be fetched (``"<code>:<season>"``), ``skipped`` a
    competition the schema cannot store yet (code -> reason).
    ``initialised_competitions`` lists the competitions whose team dictionary was
    empty and was filled from ``/league-teams`` during this harvest (the Serie D
    gironi on the first run; see ``FootyStatsProvider.harvest(auto_init_registry=...)``),
    ``registry_path`` the dictionary file that was written (``None`` when none was).
    """

    matches: pd.DataFrame
    fixtures: list[Fixture]
    odds: list[OddsQuote]
    seasons: dict[str, list[str]]
    n_matches_by_season: dict[str, int]
    unmapped_teams: dict[str, int]
    historical_teams: tuple[str, ...]
    stats: dict[str, int]
    records: tuple[MatchRecord, ...] = ()
    n_matches_by_competition: dict[str, dict[str, int]] = field(default_factory=dict)
    failures: dict[str, str] = field(default_factory=dict)
    skipped: dict[str, str] = field(default_factory=dict)
    provider: str = "footystats"
    initialised_competitions: tuple[str, ...] = ()
    registry_path: Path | None = None

    def problem_for(self, code: "Girone | str") -> str | None:
        """Why competition ``code`` delivered nothing (``None`` when it delivered)."""
        normalized = normalize_code(code)
        if normalized in self.skipped:
            return self.skipped[normalized]
        delivered = any(m.girone.value == normalized for m in self.records) or any(f.girone.value == normalized for f in self.fixtures)
        if delivered:
            return None
        for key, message in self.failures.items():
            if key.split(":", 1)[0] == normalized:
                return message
        if normalized not in self.seasons:
            return f"competition {normalized} was not harvested"
        return None

    @property
    def competitions(self) -> tuple[str, ...]:
        return tuple(self.seasons)

    def season_lines(self) -> list[str]:
        """One human line per competition and season (used by the CLI)."""
        lines: list[str] = []
        for code, labels in self.seasons.items():
            counts = self.n_matches_by_competition.get(code, {})
            lines.append(f"  {code}: " + ", ".join(f"{label}: {counts.get(label, 0)} matches" for label in labels))
        for code, reason in self.skipped.items():
            lines.append(f"  {code}: skipped ({reason})")
        return lines

    def summary(self) -> str:
        """One-line human summary (used by the CLI and the logs)."""
        return (
            f"provider={self.provider} competitions={','.join(self.seasons) or '-'} matches={len(self.matches)} "
            f"fixtures={len(self.fixtures)} odds={len(self.odds)} "
            f"seasons={','.join(sorted(self.n_matches_by_season, reverse=True)) or '-'} "
            f"historical_teams={len(self.historical_teams)} unmapped_teams={len(self.unmapped_teams)} "
            f"requests={self.stats.get('requests', 0)} cache_hits={self.stats.get('cache_hits', 0)} "
            f"retries={self.stats.get('retries', 0)} errors={self.stats.get('errors', 0)} "
            f"failures={len(self.failures)} skipped={len(self.skipped)} "
            f"initialised={','.join(self.initialised_competitions) or '-'}"
        )


class FootyStatsProvider:
    """Harvest the configured competitions (Serie C gironi, Serie B, Serie D gironi) from FootyStats into the canonical schema.

    ``mapper`` carries the perimeter registry and its matcher. ``config_dir`` is
    the directory of the ``teams_<season>.json`` dictionaries and
    ``registry_path`` the file of the perimeter season (by default
    ``config_dir / teams_<registry season>.json``): :meth:`harvest` writes the
    team lists of the competitions whose dictionary entry is empty into that file
    (auto-init) and reloads the registry from it, so tests point both at a
    temporary copy.
    """

    name: Final[str] = "footystats"

    def __init__(
        self,
        client: FootyStatsClient,
        mapper: FootyStatsTeamMapper,
        endpoints: FootyStatsEndpoints | None = None,
        *,
        now: Clock | None = None,
        config_dir: Path = CONFIG_DIR,
        registry_path: Path | None = None,
    ) -> None:
        self.client = client
        self.mapper = mapper
        self.endpoints = endpoints if endpoints is not None else client.endpoints
        self._now: Clock = now if now is not None else _utc_now
        self.config_dir = Path(config_dir)
        self._registry_path = Path(registry_path) if registry_path is not None else None

    @property
    def registry(self) -> TeamRegistry:
        """The perimeter registry currently used (replaced after an auto-init)."""
        return self.mapper.registry

    @property
    def registry_path(self) -> Path:
        """``teams_<season>.json`` of the perimeter season (``registry_path`` given to the constructor, else under ``config_dir``)."""
        if self._registry_path is not None:
            return self._registry_path
        return self.config_dir / f"teams_{season_file_stem(str(self.registry.season))}.json"

    # ----------------------------------------------------------- auto-init
    def _reload_registry(self) -> None:
        """Rebuild the mapper on the dictionary file just written (same mapping path, older seasons' aliases merged)."""
        from src.ingestion.harvester import Harvester  # local import keeps the module import graph acyclic

        registry = TeamRegistry.from_json(self.registry_path)
        matcher = TeamMatcher(Harvester.registry_with_history(registry, self.config_dir))
        self.mapper = FootyStatsTeamMapper(registry, matcher, mapping_path=self.mapper.mapping_path)

    async def _auto_init_registry(
        self, codes: Sequence[str], season_refs: Mapping[str, Sequence[SeasonRef]], *, now: datetime
    ) -> tuple[tuple[str, ...], dict[str, list[TeamInfo]]]:
        """Fill the dictionary of every competition in ``codes`` that has no current team, from ``/league-teams``.

        The season fetched is the one labelled like the registry season when the
        competition lists it, else its newest. Returns the codes initialised (in
        request order) and the team listings fetched, so the harvest of the current
        season reuses them instead of calling ``/league-teams`` twice. Nothing is
        written when every competition already has teams; a registry whose season is
        not a season label (synthetic leagues) is never touched.
        """
        registry = self.registry
        empty = [code for code in codes if not registry.teams(code) and season_refs.get(code)]
        if not empty:
            return (), {}
        try:
            season_bounds(registry.season_label)
        except ValueError:
            log.info(
                "registry auto-init skipped: the registry season is not a season label",
                extra={"season": str(registry.season), "competitions": empty},
            )
            return (), {}
        listings: dict[str, list[TeamInfo]] = {}
        for code in empty:
            refs = season_refs[code]
            chosen = next((ref for ref in refs if ref.label == registry.season_label), refs[0])
            if chosen.label != registry.season_label:
                log.warning(
                    "registry auto-init: the registry season is not listed for the competition, using its newest season",
                    extra={"competition": code, "registry_season": registry.season_label, "season": chosen.label},
                )
            listings[code] = await self.client.league_teams(chosen.season_id, stable=season_completed(chosen.label, now.date()))
        diff = write_registry_update(self.registry_path, registry.season_label, listings, matcher=self.mapper.matcher)
        self._reload_registry()
        initialised = tuple(code for code in empty if self.registry.teams(code))
        log.info(
            "registry initialised from FootyStats league-teams",
            extra={
                "path": str(self.registry_path),
                "season": registry.season_label,
                "competitions": list(initialised),
                "teams": {code: len(self.registry.teams(code)) for code in initialised},
                "added": {code: len(teams) for code, teams in diff.added_teams.items()},
                "conflicts": len(diff.conflicts),
            },
        )
        not_filled = [code for code in empty if code not in initialised]
        if not_filled:
            log.warning(
                "registry auto-init left competitions without teams",
                extra={"competitions": not_filled, "listed_by_footystats": {code: len(listings[code]) for code in not_filled}},
            )
        return initialised, listings

    # --------------------------------------------------------------- harvest
    async def _harvest_season(
        self,
        code: str,
        ref: SeasonRef,
        *,
        current: bool,
        with_odds: bool,
        now: datetime,
        teams: Sequence[TeamInfo] | None = None,
    ) -> tuple[list[MatchRecord], list[Fixture], list[OddsQuote], ParseReport]:
        completed = season_completed(ref.label, now.date())
        if teams is None:
            teams = await self.client.league_teams(ref.season_id, stable=completed)
        names = self.mapper.names_for(teams, code, ref.label, current=current)
        raw = await self.client.league_matches(ref.season_id, stable=completed)
        report = ParseReport()
        records: list[MatchRecord] = []
        fixtures: list[Fixture] = []
        odds: list[OddsQuote] = []
        out_of_season = 0
        for m in raw:
            parsed = parse_match(m, code, names, now=now, report=report)
            if isinstance(parsed, MatchRecord):
                records.append(parsed)
                if season_of(parsed.date) != ref.label and not _PLAIN_YEAR_RE.match(ref.label):
                    out_of_season += 1
            elif isinstance(parsed, Fixture) and current:
                fixtures.append(parsed)
                if with_odds:
                    odds.extend(parse_odds(m, parsed.match_id))
        log.info(
            "footystats season harvested",
            extra={
                "competition": code,
                "season": ref.label,
                "season_id": ref.season_id,
                "teams": len(teams),
                "raw_matches": len(raw),
                "current": current,
                "completed": completed,
                **report.as_dict(),
            },
        )
        if report.missing_shots:
            log.warning(
                "matches without shot statistics (FootyStats reports 0 for both teams)",
                extra={"competition": code, "season": ref.label, "matches": report.missing_shots, "of": report.complete},
            )
        if report.missing_xg and report.missing_xg == report.complete:
            log.info("xG not covered for this season", extra={"competition": code, "season": ref.label})
        elif report.missing_xg:
            log.warning("matches without xG", extra={"competition": code, "season": ref.label, "matches": report.missing_xg, "of": report.complete})
        if out_of_season:
            log.warning(
                "matches dated outside their season label",
                extra={"competition": code, "season": ref.label, "matches": out_of_season},
            )
        return records, fixtures, odds, report

    async def harvest(
        self,
        codes: Iterable["Girone | str"] = DEFAULT_COMPETITIONS,
        *,
        seasons: int = 3,
        with_odds: bool = True,
        auto_init_registry: bool = True,
    ) -> FootyStatsHarvest:
        """Harvest the newest ``seasons`` seasons of every competition in ``codes``.

        Fixtures (and, with ``with_odds``, their pre-match quotes) come from the
        newest season of each competition only. A season that fails is recorded
        in :attr:`FootyStatsHarvest.failures` and the rest is kept; a competition
        the schema cannot store yet is recorded in :attr:`FootyStatsHarvest.skipped`.
        With ``auto_init_registry`` (the default) every requested competition that
        has **no current team** in the registry (the Serie D gironi on the first
        run, or a new season's file) is initialised first: its ``/league-teams``
        of the registry season are merged into :attr:`registry_path` through
        :func:`write_registry_update` (canonical = ``cleanName`` unless the
        dictionaries know the club, aliases = ``name`` / ``english_name`` /
        ``alt_names``, existing aliases preserved), the registry and the mapper are
        reloaded and the codes are reported in
        :attr:`FootyStatsHarvest.initialised_competitions`. Raises
        :class:`HarvestError` when the seasons cannot be resolved at all, or when
        every season of every competition failed.
        """
        if int(seasons) < 1:
            raise ValueError("seasons must be >= 1")
        requested = list(dict.fromkeys(normalize_code(code) for code in codes))
        skipped: dict[str, str] = {}
        supported: list[str] = []
        for code in requested:
            if schema_supports(code):
                supported.append(code)
            else:
                skipped[code] = f"the match schema (Girone) does not accept competition code {code!r} yet"
                log.warning("competition skipped: not supported by the match schema", extra={"competition": code})
        now = self._now()
        if now.tzinfo is None:
            now = now.replace(tzinfo=timezone.utc)
        labels: dict[str, list[str]] = {}
        counts: dict[str, dict[str, int]] = {}
        failures: dict[str, str] = {}
        records: list[MatchRecord] = []
        fixtures: list[Fixture] = []
        odds: list[OddsQuote] = []
        last_error: HarvestError | None = None
        initialised: tuple[str, ...] = ()
        prefetched: dict[str, list[TeamInfo]] = {}
        if supported:
            season_refs = await self.client.resolve_seasons(supported, seasons=seasons)
            if auto_init_registry:
                initialised, prefetched = await self._auto_init_registry(supported, season_refs, now=now)
            for code in supported:
                refs = season_refs.get(code, [])
                labels[code] = [ref.label for ref in refs]
                counts[code] = {}
                for index, ref in enumerate(refs):
                    listing = prefetched.get(code) if index == 0 and ref.label == self.registry.season_label else None
                    try:
                        season_records, season_fixtures, season_odds, report = await self._harvest_season(
                            code, ref, current=index == 0, with_odds=with_odds, now=now, teams=listing
                        )
                    except HarvestError as exc:
                        last_error = exc
                        failures[f"{code}:{ref.label}"] = self.client.mask(exc)
                        log.error(
                            "footystats season failed",
                            extra={"competition": code, "season": ref.label, "season_id": ref.season_id, "status": exc.status, "url": self.client.mask(exc.url)},
                        )
                        continue
                    counts[code][ref.label] = report.complete
                    records.extend(season_records)
                    fixtures.extend(season_fixtures)
                    odds.extend(season_odds)
            try:
                self.mapper.save()
            except OSError as exc:
                log.warning("team mapping not saved", extra={"path": str(self.mapper.mapping_path), "error": str(exc)})
        if supported and not records and not fixtures and last_error is not None:
            raise last_error
        seen: set[str] = set()
        fixtures_unique = [f for f in fixtures if not (f.match_id in seen or seen.add(f.match_id))]
        fixture_ids = {f.match_id for f in fixtures_unique}
        seen_quotes: set[tuple[str, Market]] = set()
        odds_unique = [
            q for q in odds if q.match_id in fixture_ids and not ((q.match_id, q.market) in seen_quotes or seen_quotes.add((q.match_id, q.market)))
        ]
        frame = matches_to_frame(records) if records else empty_matches_frame()
        totals: Counter[str] = Counter()
        for per_season in counts.values():
            totals.update(per_season)
        result = FootyStatsHarvest(
            matches=frame,
            fixtures=fixtures_unique,
            odds=odds_unique,
            seasons=labels,
            n_matches_by_season={label: int(n) for label, n in sorted(totals.items(), key=lambda item: -_season_start_year(item[0]))},
            unmapped_teams=self.mapper.unmapped_teams,
            historical_teams=self.mapper.historical_teams,
            stats=self.client.stats,
            records=tuple(records),
            n_matches_by_competition=counts,
            failures=failures,
            skipped=skipped,
            initialised_competitions=initialised,
            registry_path=self.registry_path if initialised else None,
        )
        log.info("footystats harvest complete", extra={"summary": result.summary()})
        return result

    # --------------------------------------------------------- registry init
    async def init_registry(self, codes: Iterable["Girone | str"], season_label: str | None = None) -> dict[str, list[TeamInfo]]:
        """``/league-teams`` of ``season_label`` (``'2026/27'`` or ``'2026_27'``; the newest season when ``None``) per competition.

        Raises :class:`FootyStatsError` when a competition does not list that season.
        """
        requested = list(dict.fromkeys(normalize_code(code) for code in codes))
        wanted = season_label.strip().replace("_", "/") if season_label else None
        season_refs = await self.client.resolve_seasons(requested, seasons=MAX_PAGES)
        out: dict[str, list[TeamInfo]] = {}
        for code in requested:
            refs = season_refs[code]
            if wanted is None:
                chosen = refs[0]
            else:
                matching = [ref for ref in refs if ref.label == wanted]
                if not matching:
                    raise FootyStatsError(
                        f"season {wanted!r} is not listed for {footystats_league_name(code, self.endpoints)!r} "
                        f"(available: {', '.join(ref.label for ref in refs)})",
                        url=self.endpoints.url_league_list(),
                    )
                chosen = matching[0]
            out[code] = await self.client.league_teams(chosen.season_id, stable=season_completed(chosen.label, self._now().date()))
            log.info("league teams fetched", extra={"competition": code, "season": chosen.label, "teams": len(out[code])})
        return out


#: Module-level alias so ``from src.ingestion.footystats import init_registry`` reads naturally.
init_registry = FootyStatsProvider.init_registry


@dataclass(frozen=True)
class RegistryDiff:
    """What :func:`write_registry_update` changed in a ``teams_<season>.json`` file.

    ``unlisted_teams`` (code -> canonical names) are teams the file holds under a
    competition that FootyStats did not list for it: the function never removes
    them, so they are reported for a manual check (a club that folded, left the
    perimeter or sits in the wrong girone). ``removed_aliases`` is filled only by
    ``keep_existing_aliases=False``. ``skipped_aliases`` (canonical -> spellings)
    are FootyStats spellings that were **not** written because another club of
    the file already answers to them (one normalised spelling must resolve to
    one club, or the dictionary would not load).
    """

    path: Path
    season: str
    created: bool
    added_teams: dict[str, list[str]]
    new_aliases: dict[str, list[str]]
    unchanged_teams: int
    conflicts: dict[str, str]
    skipped: dict[str, str]
    unlisted_teams: dict[str, list[str]] = field(default_factory=dict)
    removed_aliases: dict[str, list[str]] = field(default_factory=dict)
    skipped_aliases: dict[str, list[str]] = field(default_factory=dict)

    def describe(self) -> str:
        lines = [f"{'created' if self.created else 'updated'} {self.path} (season {self.season})"]
        for code, teams in self.added_teams.items():
            lines.append(f"  {code}: added {len(teams)} teams: {', '.join(teams)}")
        if self.new_aliases:
            lines.append(f"  new aliases for {len(self.new_aliases)} teams: " + "; ".join(f"{c} <- {', '.join(a)}" for c, a in self.new_aliases.items()))
        if self.removed_aliases:
            lines.append(f"  aliases removed from {len(self.removed_aliases)} teams: " + "; ".join(f"{c} -> {', '.join(a)}" for c, a in self.removed_aliases.items()))
        if self.skipped_aliases:
            lines.append(
                f"  aliases not written for {len(self.skipped_aliases)} teams (spelling already used by another club): "
                + "; ".join(f"{c}: {', '.join(a)}" for c, a in self.skipped_aliases.items())
            )
        lines.append(f"  unchanged teams: {self.unchanged_teams}")
        for code, teams in self.unlisted_teams.items():
            lines.append(
                f"  {code}: {len(teams)} teams in the file no longer listed by FootyStats (kept; remove them by hand if they left): "
                + ", ".join(teams)
            )
        for team, reason in self.conflicts.items():
            lines.append(f"  conflict {team}: {reason}")
        for code, reason in self.skipped.items():
            lines.append(f"  {code}: skipped ({reason})")
        return "\n".join(lines)


def write_registry_update(
    path: Path,
    season: str,
    teams_by_code: Mapping[str, Sequence[TeamInfo]],
    *,
    matcher: TeamMatcher | None = None,
    keep_existing_aliases: bool = True,
) -> RegistryDiff:
    """Create or update ``config/teams_<season>.json`` from ``/league-teams`` listings.

    For every team of every competition the schema supports: the canonical name
    is the dictionary's when ``matcher`` (ideally built on the previous season
    with historical aliases) recognises one of the FootyStats spellings, else
    ``cleanName``; the aliases are the existing ones plus ``name``,
    ``english_name`` and ``alt_names``. With ``keep_existing_aliases=False`` the
    existing aliases are *replaced* by the FootyStats spellings and what was
    dropped is reported in ``removed_aliases``. Teams already in the file are
    never removed, even when FootyStats no longer lists them: they are reported
    in ``unlisted_teams`` for a manual check. A team that would land in two
    gironi is reported in ``conflicts`` and left where it was. A new file gets a
    ``competition`` label derived from the families of the codes written
    (``'Serie C'``, ``'Serie B'``, ``'Serie C + Serie B'``); an existing file keeps
    its own metadata. Returns a :class:`RegistryDiff` (``describe()`` for the console).
    """
    path = Path(path)
    label = str(season).strip().replace("_", "/")
    created = not path.exists()
    if created:
        payload: dict[str, Any] = {
            "season": label,
            "competition": competition_label(code for code in teams_by_code if schema_supports(code)) or "FootyStats leagues",
            "source": "FootyStats /league-teams (python -m src.ingestion.footystats --init-registry)",
            "note": (
                "Canonical team names per competition (keys = competition codes of config/competitions.toml). "
                "Aliases are the spellings used by FootyStats, SofaScore and bookmakers."
            ),
            "gironi": {},
        }
    else:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(payload, Mapping) or not isinstance(payload.get("gironi"), Mapping):
            raise ValueError(f"{path} is not a team dictionary (missing 'gironi' table)")
        payload = dict(payload)
        payload["gironi"] = {code: dict(teams) for code, teams in payload["gironi"].items()}
    gironi: dict[str, dict[str, list[str]]] = payload["gironi"]
    added: dict[str, list[str]] = {}
    new_aliases: dict[str, list[str]] = {}
    removed_aliases: dict[str, list[str]] = {}
    skipped_aliases: dict[str, list[str]] = {}
    conflicts: dict[str, str] = {}
    skipped: dict[str, str] = {}
    listed_by_code: dict[str, set[str]] = {}
    unchanged = 0

    def spelling_owners() -> dict[str, str]:
        """Normalised spelling (canonical names and aliases of every competition of the file) -> canonical."""
        owners: dict[str, str] = {}
        for members in gironi.values():
            for canonical_name, alias_list in members.items():
                owners.setdefault(normalize_name(canonical_name), canonical_name)
                for alias in alias_list:
                    owners.setdefault(normalize_name(alias), canonical_name)
        return owners

    for raw_code, teams in teams_by_code.items():
        code = normalize_code(raw_code)
        if not schema_supports(code):
            skipped[code] = f"the match schema (Girone) does not accept competition code {code!r} yet"
            continue
        group = gironi.setdefault(code, {})
        listed = listed_by_code.setdefault(code, set())
        owners = spelling_owners()
        for team in teams:
            canonical: str | None = None
            fuzzy: str | None = None
            if matcher is not None:
                for spelling in team.identity_spellings:
                    result = matcher.match(spelling, None)
                    if result.canonical is None:
                        continue
                    if result.method in {"exact", "alias"}:
                        canonical = result.canonical
                        break
                    if fuzzy is None:
                        fuzzy = result.canonical
            if canonical is None:
                for spelling in team.identity_spellings:
                    canonical = owners.get(normalize_name(spelling))
                    if canonical is not None:
                        break
            if canonical is None and fuzzy is not None:
                # a fuzzy guess is trusted only when it does not clash with a club the file already
                # places elsewhere: 'Pro Sesto' must not land on 'Pro Patria' and vanish as a conflict
                if not any(g != code and fuzzy in members for g, members in gironi.items()):
                    canonical = fuzzy
            if canonical is None:
                canonical = team.clean_name or team.name
            elsewhere = [g for g, members in gironi.items() if g != code and canonical in members]
            if elsewhere:
                conflicts[canonical] = f"FootyStats lists it in {code}, the file in {elsewhere[0]}; left unchanged"
                continue
            listed.add(canonical)
            existing_aliases = list(group.get(canonical, []))
            current_aliases = existing_aliases if keep_existing_aliases else []
            known = {normalize_name(canonical), *(normalize_name(a) for a in current_aliases)}
            fresh: list[str] = []
            refused: list[str] = []
            for spelling in team.spellings:
                norm = normalize_name(spelling)
                if spelling == canonical or norm in known:
                    continue
                owner = owners.get(norm)
                if owner is not None and owner != canonical:
                    refused.append(spelling)  # another club already answers to this spelling
                    continue
                known.add(norm)
                fresh.append(spelling)
            merged = [*current_aliases, *fresh]
            truly_new = [alias for alias in fresh if alias not in existing_aliases]
            dropped = [alias for alias in existing_aliases if alias not in merged]
            if canonical not in group:
                added.setdefault(code, []).append(canonical)
            elif not truly_new and not dropped:
                unchanged += 1
            if truly_new:
                new_aliases[canonical] = truly_new
            if dropped:
                removed_aliases[canonical] = dropped
            if refused:
                skipped_aliases[canonical] = refused
                log.warning(
                    "FootyStats spellings not written as aliases: already used by another club",
                    extra={"path": str(path), "competition": code, "team": canonical, "spellings": refused},
                )
            group[canonical] = merged
            owners[normalize_name(canonical)] = canonical
            for alias in fresh:
                owners[normalize_name(alias)] = canonical
    unlisted: dict[str, list[str]] = {}
    for code, listed in listed_by_code.items():
        missing = [name for name in gironi.get(code, {}) if name not in listed and name not in conflicts]
        if missing:
            unlisted[code] = missing
            log.warning(
                "teams in the dictionary no longer listed by FootyStats (kept)",
                extra={"path": str(path), "competition": code, "teams": missing},
            )
    payload["season"] = label
    # never write a dictionary the registry would refuse to load (the whole app reads it at start-up)
    try:
        TeamRegistry.from_mapping(label, gironi)
    except ValueError as exc:
        raise ValueError(f"{path}: the updated dictionary would not load, nothing written: {exc}") from exc
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    diff = RegistryDiff(
        path,
        label,
        created,
        added,
        new_aliases,
        unchanged,
        conflicts,
        skipped,
        unlisted_teams=unlisted,
        removed_aliases=removed_aliases,
        skipped_aliases=skipped_aliases,
    )
    log.info(
        "team dictionary written",
        extra={
            "path": str(path),
            "season": label,
            "added": {c: len(t) for c, t in added.items()},
            "new_aliases": len(new_aliases),
            "removed_aliases": len(removed_aliases),
            "skipped_aliases": len(skipped_aliases),
            "unlisted": {c: len(t) for c, t in unlisted.items()},
        },
    )
    return diff


# =============================================================================
# CLI
# =============================================================================
def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m src.ingestion.footystats",
        description=(
            "Harvest the configured competitions (Serie C gironi A/B/C, Serie B, Serie D gironi A-I) from the official "
            "FootyStats API into the canonical CSVs, with pre-match odds. Competitions whose team dictionary is still "
            "empty (Serie D) are initialised from /league-teams first (auto-init)."
        ),
    )
    parser.add_argument(
        "--competitions",
        nargs="+",
        default=list(DEFAULT_COMPETITIONS),
        metavar="CODE",
        help="competition codes of config/competitions.toml: "
        + ", ".join(f"{member.value} = {member.label}" for member in Girone)
        + "; default: every configured competition (%(default)s)",
    )
    parser.add_argument(
        "--no-auto-init",
        action="store_true",
        help="do not fill the empty competitions of config/teams_<season>.json from /league-teams before harvesting "
        "(their teams are then harvested as historical, outside the perimeter)",
    )
    parser.add_argument("--seasons", type=int, default=3, help="seasons per competition, newest first (default: %(default)s)")
    parser.add_argument(
        "--season",
        default=DEFAULT_SEASON,
        help="team dictionary config/teams_<season>.json defining the perimeter of current teams (default: %(default)s)",
    )
    parser.add_argument("--out", type=Path, default=PROCESSED_DIR / "matches.csv", help="matches CSV output")
    parser.add_argument("--fixtures-out", type=Path, default=PROCESSED_DIR / "fixtures.csv", help="fixtures CSV output")
    parser.add_argument("--odds-out", type=Path, default=ODDS_DIR / "footystats.csv", help="odds CSV output (match_id,bookmaker,market,selection,odds)")
    parser.add_argument("--no-odds", action="store_true", help="do not collect pre-match odds")
    parser.add_argument("--key", default=None, help="FootyStats API key (default: env FOOTYSTATS_API_KEY, then .env in the repository root); never printed")
    parser.add_argument("--list-leagues", action="store_true", help="list the leagues chosen on the account with their season ids, then exit")
    parser.add_argument(
        "--init-registry",
        action="store_true",
        help="create/update config/teams_<season>.json from /league-teams of --season: adds missing teams and aliases and prints "
        "the diff (existing teams and aliases are never removed; teams FootyStats no longer lists are reported), then exit",
    )
    parser.add_argument("--allow-partial", action="store_true", help="write the CSVs even when some season failed (exit code stays non-zero)")
    parser.add_argument("--log-json", action="store_true", help="emit JSON-lines logs")
    parser.add_argument("--cache-dir", type=Path, default=RAW_DIR, help="raw JSON cache directory")
    parser.add_argument("--mapping", type=Path, default=DEFAULT_MAPPING_PATH, help="persisted team-id mapping JSON")
    parser.add_argument("--endpoints", type=Path, default=CONFIG_DIR / "endpoints.toml", help="endpoints TOML file")
    parser.add_argument("--config-dir", type=Path, default=CONFIG_DIR, help="directory of the team dictionaries")
    return parser


def _print_leagues(leagues: Sequence[LeagueInfo]) -> None:
    if not leagues:
        print("no leagues chosen on the account: add them on footystats.org (your account -> Choose leagues)")
        return
    print(f"{len(leagues)} leagues chosen on the account:")
    for league in sorted(leagues, key=lambda item: (item.country, item.name)):
        seasons = ", ".join(f"{ref.label}={ref.season_id}" for ref in league.newest_seasons(len(league.seasons)))
        print(f"  {league.name} [{league.country}] seasons: {seasons or '-'}")


def _load_perimeter(season: str, config_dir: Path, *, for_init: bool) -> tuple[TeamRegistry, TeamMatcher]:
    """The perimeter registry (with the aliases of older dictionaries) and its matcher.

    For ``--init-registry`` a missing dictionary of ``season`` is allowed: the newest
    available dictionary then supplies canonical names and aliases.
    """
    from src.ingestion.harvester import Harvester  # local import keeps the module import graph acyclic

    stem = season_file_stem(season)
    try:
        registry = load_registry(stem, config_dir)
    except FileNotFoundError:
        if not for_init:
            raise
        older = [s for s in list_seasons(config_dir) if s < stem]
        if not older:
            registry = TeamRegistry.from_teams(stem.replace("_", "/"), {})
            return registry, TeamMatcher(registry)
        registry = load_registry(older[-1], config_dir)
    matcher = TeamMatcher(Harvester.registry_with_history(registry, config_dir))
    return registry, matcher


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entry point; exit codes: 0 ok, 1 harvest failure, 2 missing API key (or bad arguments)."""
    parser = _build_parser()
    args = parser.parse_args(argv)
    if args.seasons < 1:
        parser.error("--seasons must be >= 1")
    configure_logging(json_output=args.log_json, force=True)
    try:
        endpoints = load_endpoints(args.endpoints)
    except FileNotFoundError as exc:
        print(f"footystats failed: endpoints file not found: {args.endpoints}", file=sys.stderr)
        log.error("endpoints file not found", extra={"path": str(args.endpoints), "error": str(exc)})
        return 1
    except ValueError as exc:
        print(f"footystats failed: invalid endpoints file {args.endpoints}: {exc}", file=sys.stderr)
        return 1
    try:
        codes = [normalize_code(code) for code in args.competitions]
        for code in codes:
            footystats_league_name(code, endpoints.footystats)
    except ValueError as exc:
        parser.error(str(exc))
    try:
        key = resolve_api_key(args.key, key_env=endpoints.footystats.key_env, env_file=DEFAULT_ENV_FILE, endpoints=endpoints.footystats)
    except MissingApiKeyError as exc:
        print(f"footystats: {exc.instructions}", file=sys.stderr)
        log.error("FootyStats API key missing", extra={"key_env": exc.key_env})
        return 2
    try:
        registry, matcher = _load_perimeter(args.season, args.config_dir, for_init=args.init_registry)
    except FileNotFoundError as exc:
        print(f"footystats failed: {exc}", file=sys.stderr)
        log.error("team dictionary not found", extra={"season": args.season, "error": str(exc)})
        return 1
    cache = RawCache(Path(args.cache_dir), ttl=endpoints.http.cache_ttl, stable_ttl=endpoints.http.history_cache_ttl)
    client = FootyStatsClient(api_key=key, endpoints=endpoints.footystats, http=endpoints.http, cache=cache)
    mapper = FootyStatsTeamMapper(registry, matcher, mapping_path=args.mapping)
    provider = FootyStatsProvider(client, mapper, endpoints.footystats)

    async def run_list() -> int:
        async with client:
            _print_leagues(await client.league_list(chosen_only=True))
        return 0

    async def run_init() -> int:
        async with client:
            teams = await provider.init_registry(codes, args.season)
        target = Path(args.config_dir) / f"teams_{season_file_stem(args.season)}.json"
        diff = write_registry_update(target, args.season, teams, matcher=matcher)
        print(diff.describe())
        return 0

    async def run_harvest() -> int:
        async with client:
            result = await provider.harvest(
                codes, seasons=args.seasons, with_odds=not args.no_odds, auto_init_registry=not args.no_auto_init
            )
        complete = not result.failures and len(result.matches) > 0
        write, reason = _write_decision(result, args.out, allow_partial=args.allow_partial)
        print(result.summary())
        perimeter = provider.registry  # reloaded when the auto-init filled competitions
        print(f"perimeter: season {perimeter.season_label} ({len(perimeter.canonical_names)} current teams)")
        if result.initialised_competitions:
            print(
                "registry initialised: "
                + ", ".join(f"{code} ({len(perimeter.teams(code))} teams)" for code in result.initialised_competitions)
                + f" -> {result.registry_path}"
            )
        print("competitions harvested:")
        for line in result.season_lines() or ["  (none)"]:
            print(line)
        if result.historical_teams:
            print("historical teams (history only): " + ", ".join(result.historical_teams) + f" ({len(result.historical_teams)})")
        if write:
            save_processed(result.matches, args.out)
            save_fixtures(result.fixtures, args.fixtures_out)
            print(f"matches -> {args.out}")
            print(f"fixtures -> {args.fixtures_out}")
            if not args.no_odds:
                save_odds(result.odds, args.odds_out)
                print(f"odds -> {args.odds_out}")
        else:
            log.error("harvest output not written", extra={"reason": reason, "out": str(args.out)})
            print(reason, file=sys.stderr)
        if result.unmapped_teams:
            print("unmapped teams: " + ", ".join(f"{team} x{count}" for team, count in result.unmapped_teams.items()))
        for name, message in result.failures.items():
            print(f"failure {name}: {mask_key(message, key)}", file=sys.stderr)
        for code, reason in result.skipped.items():
            print(f"skipped {code}: {reason}", file=sys.stderr)
        if not complete:
            print("harvest incomplete: " + ("no matches harvested" if result.matches.empty else f"{len(result.failures)} failures"), file=sys.stderr)
        return 0 if complete else 1

    try:
        if args.list_leagues:
            return asyncio.run(run_list())
        if args.init_registry:
            return asyncio.run(run_init())
        return asyncio.run(run_harvest())
    except HarvestError as exc:
        message = mask_key(str(exc), key)
        log.error("footystats failed", extra={"error": message, "status": exc.status})
        print(f"footystats failed: {message}", file=sys.stderr)
        return 1
    except ValueError as exc:  # a malformed dictionary file during --init-registry
        print(f"footystats failed: {mask_key(exc, key)}", file=sys.stderr)
        return 1


if __name__ == "__main__":  # pragma: no cover - manual entry point
    raise SystemExit(main())
