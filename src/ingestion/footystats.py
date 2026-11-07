"""FootyStats provider: the official ``api.football-data-api.com`` API.

FootyStats is the **single source of truth** of the project for teams and
calendars: a documented, key-authenticated API that answers from datacenter IPs
too. It also quotes pre-match odds (1X2, Over/Under, both teams to score),
which feed the value engine through ``data/odds/footystats.csv``. Its data are
licensed to the key holder: they are fetched at runtime with the user's key and
stored only locally (``data/`` is git-ignored), never committed to the public
repository.

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
  match objects to :class:`MatchRecord` / :class:`Fixture` / :class:`OddsQuote`
  (a match not played at its kick-off is kept as a *postponed* fixture);
* the **registry sync** -- :func:`sync_registry` / :func:`apply_registry_sync`
  make the current season's dictionary (``config/teams_<season>.json``) equal to
  ``/league-teams`` for every requested competition (identity by FootyStats team
  id, then by exact/alias spelling, never fuzzy; clubs added, moved, renamed and
  removed) and report it in a :class:`RegistrySyncReport`;
* :class:`FootyStatsTeamMapper` -- FootyStats team ids -> canonical names of the
  :class:`TeamRegistry`, persisted in ``data/footystats_teams.json``
  (:data:`DEFAULT_MAPPING_PATH`, git-ignored, created on demand) so ids map
  deterministically on the next run;
* :class:`FootyStatsProvider` -- harvests the newest seasons of every requested
  competition (Serie C gironi ``A``/``B``/``C``, Serie B ``SB`` and the nine
  Serie D gironi ``DA`` .. ``DI``) into one canonical frame, with the current
  season's fixtures (kick-off time and status) and their odds, after
  synchronising the dictionary of the current season
  (``FootyStatsProvider.harvest(sync_registry=True)``);
* a CLI (``python -m src.ingestion.footystats``).

Serie D
-------
FootyStats covers the nine Serie D gironi but with thinner statistics: shots
and xG are often not collected (reported as ``0``/``-1`` and read as missing by
:func:`parse_stats`) and pre-match odds exist for fewer matches
(:func:`parse_odds` simply skips the markets without a complete book). The
season ids of every configured competition (``[competition.season_ids]`` in
``config/competitions.toml``) are the fallback **only** when ``/league-list``
itself cannot be reached (:func:`competition_season_ids`,
:meth:`FootyStatsClient.season_resolution`).

Chosen leagues
--------------
Every competition to harvest must be among the account's *chosen* leagues on
footystats.org: the API refuses ``/league-teams`` and ``/league-matches`` of a
league the account has not chosen (``success: false``, *"League is not chosen by
the user"*, verified live on 2026-11-07), so no configured season id can stand
in for it. A requested competition whose league is not chosen is therefore
**skipped** with a warning naming the league to add
(:attr:`FootyStatsHarvest.not_chosen`, mirrored in
:attr:`FootyStatsHarvest.skipped` and printed by the CLI as ``skipped DA: ...``)
while the other competitions are harvested and written as usual; it is never a
failure that blocks the CSVs or the exit code. FootyStats plans cap the number
of chosen leagues, so an account that cannot hold all 13 narrows
``--competitions`` instead.

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
import dataclasses
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
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Final

import httpx
import pandas as pd

from src.common.logging_config import configure_logging, get_logger
from src.common.paths import CONFIG_DIR, DATA_DIR, ODDS_DIR, PROCESSED_DIR, RAW_DIR, ROOT
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
from src.common.teams import (
    DEFAULT_SEASON,
    TeamRegistry,
    dumps_registry_payload,
    list_seasons,
    load_registry,
    normalize_name,
    season_file_stem,
)
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
    "IN_PLAY_WINDOW",
    "LEGACY_MAPPING_PATH",
    "SYNC_METHOD",
    "CompetitionSync",
    "FootyStatsClient",
    "FootyStatsError",
    "FootyStatsHarvest",
    "FootyStatsProvider",
    "FootyStatsTeamMapper",
    "KeyInjectingTransport",
    "LeagueInfo",
    "MappedTeam",
    "MatchesSnapshot",
    "MissingApiKeyError",
    "ParseReport",
    "RegistryDiff",
    "RegistrySyncReport",
    "SeasonRef",
    "SeasonResolution",
    "TeamInfo",
    "apply_registry_sync",
    "competition_label",
    "competition_season_ids",
    "configured_league_names",
    "find_api_key",
    "footystats_league_name",
    "init_registry",
    "is_not_chosen_refusal",
    "load_team_mapping",
    "main",
    "mask_key",
    "normalize_code",
    "not_chosen_reason",
    "parse_match",
    "parse_odds",
    "parse_stats",
    "prune_stale_odds",
    "read_env_file",
    "resolve_api_key",
    "schema_supports",
    "season_completed",
    "season_label_from_footystats_year",
    "stale_quote_ids",
    "sync_registry",
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
#: Persisted FootyStats team-id mapping (see :class:`FootyStatsTeamMapper` and :func:`sync_registry`):
#: under ``data/``, git-ignored and created on demand (it is derived from licensed FootyStats data).
DEFAULT_MAPPING_PATH: Final[Path] = DATA_DIR / "footystats_teams.json"
#: Where the mapping lived before it moved to ``data/``: read once, when :data:`DEFAULT_MAPPING_PATH`
#: does not exist yet, so an existing installation keeps its ids (see :func:`load_team_mapping`).
LEGACY_MAPPING_PATH: Final[Path] = CONFIG_DIR / "footystats_teams.json"
#: ``method`` of the mapping entries written by the registry sync: an id the sync assigned is the
#: club's identity and beats every spelling lookup (see :class:`FootyStatsTeamMapper`).
SYNC_METHOD: Final[str] = "sync"
#: An ``incomplete`` match whose kick-off passed less than this long ago is in play (kept as a
#: *scheduled* fixture with its past kick-off, no odds); beyond it, it was not played at its date and
#: is kept as a *postponed* fixture. Equal to ``src.engine.calendar.KICKOFF_GRACE``.
IN_PLAY_WINDOW: Final[timedelta] = timedelta(hours=3)
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


#: ``message`` FootyStats puts in the ``success: false`` envelope of ``/league-teams`` and ``/league-matches``
#: for a league the account has not chosen (verified live on 2026-11-07 with the public example key:
#: "League is not chosen by the user ... League may not exist, or is not available to this user.").
_NOT_CHOSEN_RE: Final[re.Pattern[str]] = re.compile(r"(?i)\bnot chosen by the user\b|\bleague is not chosen\b")


def not_chosen_reason(league_name: str) -> str:
    """Why a competition is skipped when ``league_name`` is not among the account's chosen leagues (the advice the CLI prints)."""
    return (
        f"{league_name!r} is not among the account's chosen leagues: add it on footystats.org "
        "(your account -> Choose leagues) or leave it out of --competitions"
    )


def is_not_chosen_refusal(exc: BaseException) -> bool:
    """Whether ``exc`` is FootyStats refusing a league the account has not chosen.

    True for a :class:`FootyStatsError` raised on a ``success: false`` envelope whose
    message says *League is not chosen by the user* (what ``/league-teams`` and
    ``/league-matches`` answer for such a league, whatever season id is passed).
    """
    return isinstance(exc, FootyStatsError) and _NOT_CHOSEN_RE.search(str(exc)) is not None


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
    :meth:`FootyStatsClient.season_resolution` **only when** ``/league-list`` itself
    cannot be reached; they never stand in for a league the account has not chosen
    (the API refuses its data, so such a competition is skipped instead).
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


@dataclass(frozen=True)
class SeasonResolution:
    """Outcome of :meth:`FootyStatsClient.season_resolution` for a set of competition codes.

    ``seasons`` maps every *resolvable* code to its newest seasons (newest first);
    ``not_chosen`` the codes whose league ``/league-list`` did not list among the
    account's chosen leagues (code -> FootyStats league name): the API refuses
    their data, so they carry no seasons and the caller skips them.
    ``chosen_leagues`` are the league names the account holds (empty when the list
    could not be read) and ``from_configured_ids`` says that ``/league-list`` was
    unreachable and every season came from the configured ids
    (``[competition.season_ids]`` of ``config/competitions.toml``).
    """

    seasons: dict[str, list[SeasonRef]]
    not_chosen: dict[str, str] = field(default_factory=dict)
    chosen_leagues: tuple[str, ...] = ()
    from_configured_ids: bool = False

    @property
    def codes(self) -> tuple[str, ...]:
        """The competition codes that resolved, in request order."""
        return tuple(self.seasons)

    def reason(self, code: "Girone | str") -> str | None:
        """The skip reason of a not-chosen competition (``None`` when it resolved)."""
        name = self.not_chosen.get(normalize_code(code))
        return not_chosen_reason(name) if name is not None else None


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
@dataclass(frozen=True)
class MatchesSnapshot:
    """The ``/league-matches`` rows of one season and when FootyStats served them.

    ``fetched_at`` is the aware UTC instant the oldest page was downloaded when
    it came from the raw cache, ``None`` when every page was fetched by this
    call. :meth:`reference_time` is the instant the rows describe: an
    ``incomplete`` match is judged (to play, in play, postponed) against it,
    never against a clock that is later than the download.
    """

    rows: list[dict[str, Any]]
    fetched_at: datetime | None = None

    def reference_time(self, now: datetime) -> datetime:
        """``min(now, fetched_at)`` (aware UTC): the instant the payload's statuses were true."""
        reference = now if now.tzinfo is not None else now.replace(tzinfo=timezone.utc)
        if self.fetched_at is None:
            return reference
        fetched = self.fetched_at if self.fetched_at.tzinfo is not None else self.fetched_at.replace(tzinfo=timezone.utc)
        return min(reference, fetched)


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
    async def get(
        self, path: str, params: Mapping[str, Any] | None = None, *, stable: bool = False, max_age: timedelta | None = None
    ) -> dict[str, Any]:
        """GET one page and return the whole envelope (``success``, ``pager``, ``metadata``, ``data``).

        Raises :class:`FootyStatsError` on ``success: false`` or when ``data`` is
        not a list (the refused envelope is evicted from the cache); transport and
        HTTP failures surface as the :class:`HarvestError` of the underlying
        client, with the key masked should it ever appear in a message.
        ``max_age`` caps the age of a cached envelope for this read.
        """
        envelope, _ = await self._get_envelope(path, params, stable=stable, max_age=max_age)
        return envelope

    async def _get_envelope(
        self, path: str, params: Mapping[str, Any] | None, *, stable: bool, max_age: timedelta | None
    ) -> tuple[dict[str, Any], datetime | None]:
        """:meth:`get` plus the download instant of a cached envelope (``None`` when fetched by this call)."""
        url = self._client.build_url(path)
        try:
            payload, fetched_at = await self._client.get_json_entry(path, params, stable=stable, max_age=max_age)
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
        return dict(payload), fetched_at

    async def get_all(self, path: str, params: Mapping[str, Any] | None = None, *, stable: bool = False) -> list[Any]:
        """Every ``data`` row of ``path``: pages ``1..pager.max_page`` concatenated."""
        rows, _ = await self._get_all_with_time(path, params, stable=stable, max_age=None)
        return rows

    async def _get_all_with_time(
        self, path: str, params: Mapping[str, Any] | None, *, stable: bool, max_age: timedelta | None
    ) -> tuple[list[Any], datetime | None]:
        """:meth:`get_all` plus the download instant of the **oldest** cached page (``None`` when every page was fetched now)."""
        base = dict(params or {})
        rows: list[Any] = []
        oldest: datetime | None = None
        page = 1
        while True:
            payload, fetched_at = await self._get_envelope(path, {**base, "page": page}, stable=stable, max_age=max_age)
            if fetched_at is not None and (oldest is None or fetched_at < oldest):
                oldest = fetched_at
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
        return rows, oldest

    # ------------------------------------------------------------ endpoints
    async def league_list(self, chosen_only: bool = True) -> list[LeagueInfo]:
        """``/league-list``: the leagues of the account (``chosen_only``) or every league FootyStats covers."""
        params = {"chosen_leagues_only": "true"} if chosen_only else {}
        entries = await self.get_all(self.endpoints.league_list, params)
        leagues = [LeagueInfo.from_payload(entry) for entry in entries if isinstance(entry, Mapping)]
        return [league for league in leagues if league is not None]

    async def season_resolution(self, competition_codes: Iterable["Girone | str"], seasons: int = 3) -> SeasonResolution:
        """Newest ``seasons`` seasons (label + id) of every competition code, plus the leagues the account lacks.

        Competitions are looked up by :func:`footystats_league_name` among the
        account's *chosen* leagues (``/league-list?chosen_leagues_only=true``). A
        competition whose league is **not chosen** is returned in
        :attr:`SeasonResolution.not_chosen` instead of being resolved: FootyStats
        refuses ``/league-teams`` and ``/league-matches`` of such a league (*League
        is not chosen by the user*), so no configured season id can stand in for
        it; a warning names the league to add on footystats.org. Only when
        ``/league-list`` itself cannot be reached are the configured season ids
        (:func:`competition_season_ids`: ``[competition.season_ids]`` of
        ``config/competitions.toml``, then ``[footystats.season_ids]`` of
        ``endpoints.toml``) used for every requested competition, with a warning
        (``from_configured_ids``); the error propagates when one of them has none.
        A chosen league that lists no season raises :class:`FootyStatsError`.
        """
        if int(seasons) < 1:
            raise ValueError("seasons must be >= 1")
        codes = list(dict.fromkeys(normalize_code(code) for code in competition_codes))
        names = {code: footystats_league_name(code, self.endpoints) for code in codes}
        list_url = self.endpoints.url_league_list()
        try:
            leagues = await self.league_list(chosen_only=True)
        except HarvestError as exc:
            fallback = {code: competition_season_ids(code, self.endpoints) for code in codes}
            if codes and all(fallback.values()):
                log.warning(
                    "league-list unavailable, using configured season ids",
                    extra={"error": self.mask(exc), "competitions": codes},
                )
                return SeasonResolution(
                    seasons={code: LeagueInfo(names[code], names[code], "", fallback[code]).newest_seasons(seasons) for code in codes},
                    from_configured_ids=True,
                )
            raise
        by_name = {league.name.casefold(): league for league in leagues}
        chosen = tuple(sorted(league.name for league in leagues))
        out: dict[str, list[SeasonRef]] = {}
        not_chosen: dict[str, str] = {}
        for code in codes:
            league = by_name.get(names[code].casefold())
            if league is None:
                not_chosen[code] = names[code]
                log.warning(
                    "league not among the account's chosen leagues, competition skipped",
                    extra={
                        "competition": code,
                        "league": names[code],
                        "advice": not_chosen_reason(names[code]),
                        "chosen_leagues": list(chosen),
                    },
                )
                continue
            refs = league.newest_seasons(seasons)
            if not refs:
                raise FootyStatsError(f"no season listed for {names[code]!r}", url=list_url)
            out[code] = refs
        return SeasonResolution(seasons=out, not_chosen=not_chosen, chosen_leagues=chosen)

    async def resolve_seasons(self, competition_codes: Iterable["Girone | str"], seasons: int = 3) -> dict[str, list[SeasonRef]]:
        """Newest ``seasons`` seasons (label + id) of every competition code, newest first.

        The strict form of :meth:`season_resolution`: a competition whose league is
        not among the account's chosen leagues raises :class:`FootyStatsError`
        naming the leagues to add on footystats.org (and the ones chosen now), so
        callers that need *every* requested competition (``--init-registry``) stop
        with a clear message. :meth:`FootyStatsProvider.harvest` uses
        :meth:`season_resolution` and skips such competitions instead. When
        ``/league-list`` is unreachable the configured season ids are used, as in
        :meth:`season_resolution`.
        """
        resolution = await self.season_resolution(competition_codes, seasons=seasons)
        if resolution.not_chosen:
            raise FootyStatsError(
                "competitions not among the account's chosen leagues: "
                + ", ".join(repr(name) for name in resolution.not_chosen.values())
                + " -- add them on footystats.org (your account -> Choose leagues) and retry, or leave them out; chosen now: "
                + (", ".join(resolution.chosen_leagues) or "none"),
                url=self.endpoints.url_league_list(),
            )
        return resolution.seasons

    async def league_teams(self, season_id: int, *, stable: bool = False) -> list[TeamInfo]:
        """``/league-teams`` of one season (``stable`` for a completed season: long cache TTL)."""
        entries = await self.get_all(self.endpoints.league_teams, {"season_id": int(season_id)}, stable=stable)
        teams = [TeamInfo.from_payload(entry) for entry in entries if isinstance(entry, Mapping)]
        return [team for team in teams if team is not None]

    async def league_matches(self, season_id: int, *, stable: bool = False, max_age: timedelta | None = None) -> list[dict[str, Any]]:
        """Raw ``/league-matches`` objects of one season, every page."""
        return (await self.league_matches_snapshot(season_id, stable=stable, max_age=max_age)).rows

    async def league_matches_snapshot(self, season_id: int, *, stable: bool = False, max_age: timedelta | None = None) -> MatchesSnapshot:
        """Raw ``/league-matches`` objects of one season **and** when they were downloaded (:class:`MatchesSnapshot`).

        ``max_age`` caps the age of the cached pages (the provider passes
        :attr:`FootyStatsEndpoints.live_matches_cache_ttl` for a season still
        being played). The snapshot's ``fetched_at`` is the oldest cached page's
        download instant, ``None`` when every page came from the network now.
        """
        rows, fetched_at = await self._get_all_with_time(
            self.endpoints.league_matches, {"season_id": int(season_id)}, stable=stable, max_age=max_age
        )
        return MatchesSnapshot(rows=[dict(row) for row in rows if isinstance(row, Mapping)], fetched_at=fetched_at)


# =============================================================================
# Parsers (pure)
# =============================================================================
@dataclass
class ParseReport:
    """Counters :func:`parse_match` accumulates over one season, logged once per season.

    ``fixtures`` counts every :class:`Fixture` returned (scheduled, in play and
    postponed); ``postponed`` the postponed ones among them (kick-off more than
    :data:`IN_PLAY_WINDOW` in the past, not played); ``in_play`` the incomplete
    matches that kicked off less than :data:`IN_PLAY_WINDOW` ago, kept as
    *scheduled* fixtures with their (past) kick-off so the round still lists
    them; ``incomplete_past`` = ``postponed + in_play`` (every incomplete match
    whose kick-off is in the past).
    """

    complete: int = 0
    fixtures: int = 0
    incomplete_past: int = 0
    skipped_status: Counter = field(default_factory=Counter)
    skipped_invalid: int = 0
    missing_shots: int = 0
    missing_xg: int = 0
    postponed: int = 0
    in_play: int = 0

    def as_dict(self) -> dict[str, Any]:
        return {
            "complete": self.complete,
            "fixtures": self.fixtures,
            "postponed": self.postponed,
            "in_play": self.in_play,
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
    * ``status == 'incomplete'`` (``now`` is UTC, default: the current time):

      - kick-off after ``now`` -> :class:`Fixture` ``status='scheduled'`` with the
        kick-off instant (``kickoff``, aware UTC);
      - kick-off within the last :data:`IN_PLAY_WINDOW` (3 hours) -> in play:
        :class:`Fixture` ``status='scheduled'`` with its (past) kick-off, so the
        round keeps listing it (:data:`src.engine.calendar.KICKOFF_GRACE`, the
        same three hours, keeps such a round current) until the result comes
        with a later harvest (:attr:`ParseReport.in_play`). It is never dropped:
        a match that is neither played nor in the fixture list would vanish
        from its round. Pre-match odds are not collected for it (see
        :meth:`FootyStatsProvider.harvest`);
      - kick-off more than 3 hours before ``now`` -> :class:`Fixture`
        ``status='postponed'`` keeping the original date and kick-off:
        FootyStats leaves a postponed match ``incomplete`` at its old date until
        the recovery is scheduled, and a round must not look shorter than it is
        (counted in :attr:`ParseReport.postponed`);

    * ``suspended`` / ``canceled`` / anything else -> ``None``.

    ``now`` must be the instant the payload describes: for a payload served
    from the raw cache, the instant it was downloaded
    (:meth:`MatchesSnapshot.reference_time`), never a later clock -- a match
    still ``incomplete`` in a five-hour-old payload was simply not played yet.
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
        if kickoff > reference:
            report.fixtures += 1
            return Fixture(match_id, kickoff.date(), round_no, girone, home, away, kickoff=kickoff, status="scheduled")
        report.incomplete_past += 1
        if reference - kickoff <= IN_PLAY_WINDOW:
            report.in_play += 1
            report.fixtures += 1
            log.debug(
                "incomplete match kicked off less than 3 hours ago: in play, kept as a scheduled fixture",
                extra={"match_id": match_id, "kickoff": kickoff.isoformat(), "round": round_no},
            )
            return Fixture(match_id, kickoff.date(), round_no, girone, home, away, kickoff=kickoff, status="scheduled")
        report.fixtures += 1
        report.postponed += 1
        log.debug("incomplete match with a past kick-off kept as postponed", extra={"match_id": match_id, "kickoff": kickoff.isoformat(), "round": round_no})
        return Fixture(match_id, kickoff.date(), round_no, girone, home, away, kickoff=kickoff, status="postponed")
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


def stale_quote_ids(fixtures: Iterable[Fixture], *, now: datetime) -> set[str]:
    """``match_id`` of the fixtures whose pre-match quotes are stale at ``now``.

    A postponed fixture (no valid date: the recovery is still to be fixed) and a
    scheduled one that has already kicked off (in play, or played since the
    download) have no pre-match price any more; a quote kept for them would be
    evaluated by the value engine at a stale price.
    """
    reference = now if now.tzinfo is not None else now.replace(tzinfo=timezone.utc)
    return {
        f.match_id
        for f in fixtures
        if f.status != "scheduled" or (f.kickoff is not None and f.kickoff <= reference)
    }


def prune_stale_odds(path: Path, fixtures: Iterable[Fixture], *, now: datetime) -> int:
    """Remove from the odds CSV ``path`` the rows of fixtures whose quotes are stale (:func:`stale_quote_ids`).

    Used when a harvest does not rewrite the odds file (``--no-odds``): the
    quotes an earlier run wrote for a match that has since been postponed or
    has kicked off must not reach the value engine. Every other row is kept as
    it is. A missing or unreadable file is left alone. The file is rewritten
    atomically, only when something was removed; returns how many rows (one per
    quoted selection) were.
    """
    target = Path(path)
    if not target.exists():
        return 0
    stale = stale_quote_ids(fixtures, now=now)
    if not stale:
        return 0
    try:
        frame = pd.read_csv(target, dtype="string", keep_default_na=False)
    except (OSError, ValueError) as exc:
        log.warning("odds file unreadable, stale quotes not pruned", extra={"path": str(target), "error": str(exc)})
        return 0
    if "match_id" not in frame.columns:
        return 0
    drop = frame["match_id"].astype(str).isin(stale).to_numpy(dtype=bool)
    removed = int(drop.sum())
    if removed:
        _atomic_write_text(target, frame.loc[~drop].to_csv(index=False))
        log.info(
            "stale quotes removed from the odds file (postponed or already kicked-off matches)",
            extra={"path": str(target), "rows": removed, "matches": len(set(frame.loc[drop, "match_id"].astype(str)))},
        )
    return removed


# =============================================================================
# Team mapping
# =============================================================================
def _atomic_write_text(path: Path, text: str) -> Path:
    """Write ``text`` to ``path`` through a temporary file in the same directory and :func:`os.replace`.

    A crash or a full disk never leaves a half-written dictionary or mapping
    behind: readers see either the old file or the new one. Returns ``path``.
    """
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    handle = tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=target.parent, prefix=f".{target.name}.", suffix=".tmp", delete=False)
    try:
        with handle:
            handle.write(text)
        os.replace(handle.name, target)
    except BaseException:
        Path(handle.name).unlink(missing_ok=True)
        raise
    return target


def _read_mapping_file(path: Path) -> dict[int, dict[str, Any]]:
    try:
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
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


def load_team_mapping(path: Path = DEFAULT_MAPPING_PATH) -> dict[int, dict[str, Any]]:
    """The persisted FootyStats team-id mapping (``{team id: {canonical, clean_name, competition, seasons, method}}``).

    A missing file is an empty mapping, an unreadable one too (with a warning).
    When ``path`` is :data:`DEFAULT_MAPPING_PATH` and does not exist yet, the
    mapping of :data:`LEGACY_MAPPING_PATH` (``config/footystats_teams.json``,
    where it lived before moving under the git-ignored ``data/``) is read
    instead, once: the next save writes it to the new place.
    """
    target = Path(path)
    if target.exists():
        return _read_mapping_file(target)
    if target.resolve() == DEFAULT_MAPPING_PATH.resolve() and LEGACY_MAPPING_PATH.exists():
        migrated = _read_mapping_file(LEGACY_MAPPING_PATH)
        log.info(
            "team mapping migrated from its old location",
            extra={"from": str(LEGACY_MAPPING_PATH), "to": str(target), "teams": len(migrated)},
        )
        return migrated
    return {}


def _mapping_payload(entries: Mapping[int, Mapping[str, Any]]) -> str:
    payload = {
        "note": "FootyStats team id -> canonical name of config/teams_<season>.json. Maintained by "
        "src/ingestion/footystats.py: entries with method 'sync' are written by the registry sync and identify the club "
        "(they beat every spelling lookup); for the others exact/alias dictionary hits beat persisted ids, which beat fuzzy "
        "matches. Local file derived from FootyStats data: git-ignored, never committed. Edit 'canonical' to correct a mapping.",
        "teams": {str(team_id): dict(entry) for team_id, entry in sorted(entries.items())},
    }
    return json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=False) + "\n"


@dataclass(frozen=True, slots=True)
class MappedTeam:
    """How one FootyStats team resolved to a canonical name."""

    team_id: int
    canonical: str
    method: str  # 'sync' | 'exact' | 'alias' | 'persisted' | 'fuzzy' | 'none'
    current: bool
    historical: bool


class FootyStatsTeamMapper:
    """FootyStats team ids -> canonical names of the team dictionary.

    A team's spellings (``cleanName``, ``name``, ``english_name``, ``alt_names``,
    in that order) are looked up through ``matcher`` (built on ``registry``, which
    should carry the aliases of the older dictionaries). Precedence:

    0. a persisted entry written by the **registry sync** (method
       :data:`SYNC_METHOD`): the sync identified the club by its id (or by a
       collision-free spelling) and FootyStats ids are stable across seasons, so
       it beats every spelling lookup -- ``'AC Palermo'`` (Serie D) normalises
       like ``'Palermo'`` (Serie B) and only the id tells them apart;
    1. an **exact** or **alias** hit of any spelling (dictionary knowledge wins);
    2. the **persisted** canonical of the same id from ``mapping_path``;
    3. the best **fuzzy** hit;
    4. nothing: the team keeps ``cleanName`` and is reported as *unmapped*.

    For the current season the lookup is restricted to the composition of the
    competition being parsed first; past seasons and competitions outside the
    dictionary use unrestricted lookups, which may return historical names
    (clubs that left the perimeter).

    **One club, one competition.** In the current season a club plays in
    exactly one competition, so a hit (spelling, persisted or fuzzy) on a
    *current club of another competition* is never accepted: without the
    sync's id evidence it cannot be told from a different club sharing a
    normalised spelling (``'AC Palermo'`` in Serie D / ``'Palermo'`` in Serie B),
    and merging them would make one team play in two leagues. Such a club takes
    the name the registry sync gives it (:func:`_derived_names`:
    ``'AC Palermo Serie D'``, the dictionary's own name when it holds it), else
    the first free collision name, reported as *unmapped* with the advice to
    run the sync (a genuine move between competitions is applied by the sync,
    which identifies clubs by id). In every season a hit whose canonical a
    **different** FootyStats id owns through a sync entry is refused the same
    way. Every resolution is merged into the persisted mapping
    ``{team_id: {canonical, clean_name, competition, seasons}}`` (:meth:`save`),
    so ids map deterministically on the next run.
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
        return load_team_mapping(path)

    @property
    def persisted(self) -> dict[int, dict[str, Any]]:
        """The merged mapping as it would be written (id -> entry)."""
        return dict(sorted(self._persisted.items()))

    def save(self, path: Path | None = None) -> Path:
        """Write the merged mapping atomically (sorted by id); returns the path."""
        target = Path(path) if path is not None else self.mapping_path
        _atomic_write_text(target, _mapping_payload(self._persisted))
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

    def _foreign_current(self, canonical: str, code: str) -> bool:
        """Whether ``canonical`` is a club of the current composition of **another** competition than ``code``."""
        if not self.matcher.is_current(canonical):
            return False
        try:
            return self.registry.girone_of(canonical).value != code
        except KeyError:  # current for the matcher (with history) but not in this registry: not a current club here
            return False

    def _owned_by_other_id(self, canonical: str, team_id: int) -> bool:
        """Whether a **different** FootyStats id owns ``canonical`` through a registry-sync entry (ids are stable)."""
        return any(
            other != team_id and str(entry.get("method") or "") == SYNC_METHOD and str(entry.get("canonical") or "").strip() == canonical
            for other, entry in self._persisted.items()
        )

    def _refused(self, canonical: str, team: TeamInfo, code: str, *, current: bool) -> str | None:
        """Why ``canonical`` cannot be ``team`` in ``code`` (``None`` when it can): see "One club, one competition"."""
        if current and self._foreign_current(canonical, code):
            return f"current club of {self.registry.girone_of(canonical).value}"
        if self._owned_by_other_id(canonical, team.id):
            return "owned by another FootyStats id (registry sync)"
        return None

    def _own_name(self, team: TeamInfo, code: str, *, current: bool) -> tuple[str, str]:
        """``(canonical, method)`` of a club whose spellings belong to another club (the sync's collision naming).

        First a collision name (:func:`_collision_names`: ``cleanName``, ``name``,
        ``cleanName + family`` ...) the dictionary knows exactly for this very
        competition (``'AC Palermo Serie D'``); then the first collision name no
        other club answers to, with method ``'none'`` (reported as unmapped).
        """
        girone = self._lookup_girone(code) if current else None
        candidates = _collision_names(team, code)
        if girone is not None:
            for candidate in candidates:
                result = self.matcher.match(candidate, girone)
                if result.canonical is not None and result.method in {"exact", "alias"} and self._refused(result.canonical, team, code, current=current) is None:
                    return result.canonical, result.method
        for candidate in candidates:
            result = self.matcher.match(candidate, None)
            if result.canonical is None or result.method not in {"exact", "alias"}:
                return candidate, "none"
            if self._refused(result.canonical, team, code, current=current) is None:
                return result.canonical, result.method
        return f"{(team.clean_name or team.name).strip()} ({code}, id {team.id})", "none"

    def resolve(self, team: TeamInfo, competition: "Girone | str", season: str, *, current: bool = True) -> MappedTeam:
        """Canonical name of ``team`` in ``competition`` / ``season`` (see the class docstring for the precedence).

        A team already resolved during this run keeps its result (the mapping only
        records the season); so the same id maps to the same name across the
        seasons and competitions of one harvest. ``current`` says ``season`` is the
        season of the dictionary (its composition is the truth of that season).
        """
        code = normalize_code(competition)
        already = self.resolved.get(team.id)
        if already is not None:
            entry = self._persisted.get(team.id)
            if entry is not None and season not in entry.get("seasons", []):
                entry["seasons"] = sorted({*entry.get("seasons", []), season}, reverse=True)
            return already
        stored = self._persisted.get(team.id)
        persisted = stored  # the entry that may decide the name (dropped below when it points at another club)
        synced = persisted is not None and str(persisted.get("method") or "") == SYNC_METHOD
        hit: tuple[str, str] | None = None
        refused: dict[str, str] = {}
        if not synced:
            girone = self._lookup_girone(code) if current else None
            hit = self._best(team, girone)
            if hit is None and girone is not None:
                hit = self._best(team, None)
            if hit is not None:
                reason = self._refused(hit[0], team, code, current=current)
                if reason is not None:
                    refused[hit[0]] = reason
                    hit = None
            if persisted is not None:
                reason = self._refused(str(persisted.get("canonical") or "").strip(), team, code, current=current)
                if reason is not None:
                    refused[str(persisted.get("canonical") or "").strip()] = reason
                    persisted = None
        if synced and persisted is not None:
            canonical, method = str(persisted["canonical"]).strip(), SYNC_METHOD
        elif hit is not None and hit[1] in {"exact", "alias"}:
            canonical, method = hit
        elif persisted is not None:
            canonical, method = str(persisted["canonical"]).strip(), "persisted"
        elif hit is not None:
            canonical, method = hit
        elif refused:
            canonical, method = self._own_name(team, code, current=current)
            if method == "none":
                self.unmapped[canonical] += 1
            log.warning(
                "FootyStats team answers to a club of another competition: not merged with it",
                extra={
                    "team": team.clean_name,
                    "team_id": team.id,
                    "competition": code,
                    "season": season,
                    "refused": refused,
                    "canonical": canonical,
                    "advice": "run the registry sync (python -m src.ingestion.footystats --sync-registry): it identifies clubs by FootyStats id",
                },
            )
        else:
            canonical, method = team.clean_name or team.name, "none"
            self.unmapped[canonical] += 1
            log.warning(
                "unmapped FootyStats team kept with its clean name",
                extra={"team": team.clean_name, "team_id": team.id, "competition": code, "season": season, "spellings": list(team.spellings)},
            )
        if stored is not None and str(stored.get("canonical", "")).strip() != canonical:
            log.info(
                "persisted team mapping updated",
                extra={"team_id": team.id, "old": stored.get("canonical"), "new": canonical, "method": method},
            )
        mapped = MappedTeam(
            team_id=team.id,
            canonical=canonical,
            method=method,
            current=self.matcher.is_current(canonical),
            historical=self.matcher.is_historical(canonical),
        )
        self.resolved[team.id] = mapped
        seasons = sorted(set(stored.get("seasons", [])) | {season}, reverse=True) if stored else [season]
        entry: dict[str, Any] = {
            "canonical": canonical,
            "clean_name": team.clean_name,
            "competition": code,
            "seasons": seasons,
            # the file keeps how the name was originally determined, so a persisted hit stays auditable
            "method": (str(persisted.get("method") or "persisted") if method == "persisted" and persisted is not None else method),
        }
        if method == SYNC_METHOD and persisted is not None:
            # the sync owns its entries: keep its competition (the current one) and the club's spelling it recorded
            entry["competition"] = str(persisted.get("competition") or code)
            entry["clean_name"] = str(persisted.get("clean_name") or team.clean_name)
        self._persisted[team.id] = entry
        return mapped

    def adopt(self, assignments: Mapping[int, str], teams: Iterable[TeamInfo], competition: "Girone | str", season: str) -> dict[int, str]:
        """Record the canonical names the registry sync assigned to the ids of ``teams`` (method :data:`SYNC_METHOD`).

        Used by :meth:`FootyStatsProvider.harvest` for the current season: the
        names of the season the sync just aligned come from the sync itself, so
        the parse of its matches can never disagree with the dictionary. Teams
        without an assignment are resolved as usual. Returns ``team id ->
        canonical`` for the whole listing (like :meth:`names_for`).
        """
        code = normalize_code(competition)
        out: dict[int, str] = {}
        for team in teams:
            canonical = assignments.get(team.id)
            if canonical is None:
                out[team.id] = self.resolve(team, code, season).canonical
                continue
            previous = self._persisted.get(team.id) or {}
            seasons = sorted(set(previous.get("seasons", [])) | {season}, reverse=True)
            self._persisted[team.id] = {
                "canonical": canonical,
                "clean_name": team.clean_name,
                "competition": code,
                "seasons": seasons,
                "method": SYNC_METHOD,
            }
            self.resolved[team.id] = MappedTeam(
                team_id=team.id,
                canonical=canonical,
                method=SYNC_METHOD,
                current=self.matcher.is_current(canonical),
                historical=self.matcher.is_historical(canonical),
            )
            out[team.id] = canonical
        return out

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
    competition that was not harvested at all (code -> reason): one the schema
    cannot store yet, or one whose league the FootyStats account has **not
    chosen** (the API refuses its data; such codes are also listed in
    ``not_chosen`` as code -> league name, so the caller can print what to add on
    footystats.org). Skipped competitions are not failures: the CSVs of the other
    competitions are written and the CLI exit code stays 0.
    ``registry_sync`` is the :class:`RegistrySyncReport` of the dictionary sync
    that ran before the parse (``None`` when it was disabled, could not run --
    ``registry_sync_error`` then says why -- or the registry was not loaded from
    the provider's dictionary file); ``initialised_competitions`` lists the
    competitions that had no team before the sync and have some after it, and
    ``registry_path`` the dictionary file the sync wrote (``None`` when nothing
    was written). ``fixtures`` hold scheduled (matches in play included, with
    their past kick-off) and postponed matches (:attr:`Fixture.status`); odds are
    collected for the scheduled ones that have not kicked off yet.
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
    not_chosen: dict[str, str] = field(default_factory=dict)
    registry_sync: RegistrySyncReport | None = None
    registry_sync_error: str | None = None

    @property
    def n_postponed(self) -> int:
        """Postponed fixtures (original kick-off passed, not played) among :attr:`fixtures`."""
        return sum(1 for f in self.fixtures if f.status == "postponed")

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
            f"initialised={','.join(self.initialised_competitions) or '-'} "
            f"not_chosen={','.join(self.not_chosen) or '-'} "
            f"postponed={self.n_postponed} registry_sync={self._sync_state()}"
        )

    def _sync_state(self) -> str:
        if self.registry_sync is None:
            return "error" if self.registry_sync_error else "-"
        return f"changed:{self.registry_sync.n_changes}" if self.registry_sync.changed else "unchanged"


class FootyStatsProvider:
    """Harvest the configured competitions (Serie C gironi, Serie B, Serie D gironi) from FootyStats into the canonical schema.

    ``mapper`` carries the perimeter registry and its matcher. ``config_dir`` is
    the directory of the ``teams_<season>.json`` dictionaries and
    ``registry_path`` the file of the perimeter season (by default
    ``config_dir / teams_<registry season>.json``): :meth:`harvest` synchronises
    that file with ``/league-teams`` of the registry season (:func:`apply_registry_sync`)
    and reloads the registry from it. The sync only runs when the mapper's
    registry *is* that file (same composition): a registry built in memory is
    never written over an unrelated dictionary. Tests point both at a temporary
    copy.
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
        """The perimeter registry currently used (reloaded after the registry sync)."""
        return self.mapper.registry

    @property
    def registry_path(self) -> Path:
        """``teams_<season>.json`` of the perimeter season (``registry_path`` given to the constructor, else under ``config_dir``)."""
        if self._registry_path is not None:
            return self._registry_path
        return self.config_dir / f"teams_{season_file_stem(str(self.registry.season))}.json"

    # ---------------------------------------------------------- registry sync
    def _reload_registry(self) -> None:
        """Rebuild the mapper on the dictionary file just written (same mapping path, older seasons' aliases merged)."""
        from src.ingestion.harvester import Harvester  # local import keeps the module import graph acyclic

        registry = TeamRegistry.from_json(self.registry_path)
        matcher = TeamMatcher(Harvester.registry_with_history(registry, self.config_dir))
        self.mapper = FootyStatsTeamMapper(registry, matcher, mapping_path=self.mapper.mapping_path)

    def _registry_is_the_file(self) -> bool:
        """Whether the mapper's registry was loaded from :attr:`registry_path` (same competitions and clubs)."""
        path = self.registry_path
        if not path.exists():
            return False
        try:
            on_disk = TeamRegistry.from_json(path)
        except (OSError, ValueError, KeyError) as exc:
            log.warning("team dictionary unreadable, sync skipped", extra={"path": str(path), "error": str(exc)})
            return False
        mine = {g.value: tuple(teams) for g, teams in self.registry.teams_by_girone.items() if teams}
        theirs = {g.value: tuple(teams) for g, teams in on_disk.teams_by_girone.items() if teams}
        return mine == theirs and on_disk.season_label == self.registry.season_label

    async def _sync_current_registry(
        self, codes: Sequence[str], season_refs: Mapping[str, Sequence[SeasonRef]], *, now: datetime
    ) -> tuple[RegistrySyncReport | None, dict[str, list[TeamInfo]], str | None, tuple[str, ...]]:
        """Synchronise the dictionary of the registry season with ``/league-teams`` before the parse.

        Returns the report (``None`` when the sync could not run), the listings
        fetched (reused by the harvest of the current season, so ``/league-teams``
        is called once), the Italian reason when the sync did not run and the
        competitions that had no team before and have some after. A
        competition that does not list the registry season, or whose listing fails
        (a league not chosen, a transient error), is reported as skipped and left
        untouched; the others are synchronised. A registry whose season is not a
        season label (synthetic leagues) or that was not loaded from
        :attr:`registry_path` is never synchronised.
        """
        registry = self.registry
        try:
            season_bounds(registry.season_label)
        except ValueError:
            log.info("registry sync skipped: the registry season is not a season label", extra={"season": str(registry.season)})
            return None, {}, None, ()
        if not self._registry_is_the_file():
            reason = f"il dizionario in uso non è stato letto da {self.registry_path}: sincronizzazione saltata"
            log.warning(
                "registry sync skipped: the provider's registry was not loaded from its dictionary file",
                extra={"path": str(self.registry_path), "season": registry.season_label},
            )
            return None, {}, reason, ()
        listings: dict[str, list[TeamInfo]] = {}
        unavailable: dict[str, str] = {}
        for code in codes:
            ref = next((r for r in season_refs.get(code, ()) if r.label == registry.season_label), None)
            if ref is None:
                unavailable[code] = f"FootyStats non elenca la stagione {registry.season_label}"
                continue
            try:
                listings[code] = await self.client.league_teams(ref.season_id, stable=season_completed(ref.label, now.date()))
            except HarvestError as exc:
                masked = self.client.mask(exc)
                refused = is_not_chosen_refusal(exc)
                unavailable[code] = (
                    f"lega {footystats_league_name(code, self.endpoints)!r} non scelta nell'account FootyStats"
                    if refused
                    else "lista squadre non disponibile: " + masked
                )
                log.warning(
                    "registry sync: league-teams unavailable, competition left unchanged",
                    extra={"competition": code, "season": ref.label, "season_id": ref.season_id, "status": exc.status, "error": masked},
                )
        before = {code: bool(registry.teams(code)) for code in codes}
        try:
            report = apply_registry_sync(
                listings,
                season=registry.season_label,
                registry_path=self.registry_path,
                mapping_path=self.mapper.mapping_path,
                competitions=codes,
            )
        except (OSError, ValueError) as exc:
            reason = f"sincronizzazione del dizionario non riuscita: {self.client.mask(exc)}"
            log.error("registry sync failed, the harvest goes on with the dictionary as it was", extra={"path": str(self.registry_path), "error": self.client.mask(exc)})
            return None, listings, reason, ()
        if unavailable:
            report = dataclasses.replace(report, skipped={**report.skipped, **unavailable})
        self._reload_registry()
        initialised = tuple(code for code in codes if not before[code] and self.registry.teams(code))
        if initialised:
            log.info(
                "registry initialised from FootyStats league-teams",
                extra={"path": str(self.registry_path), "competitions": list(initialised), "teams": {c: len(self.registry.teams(c)) for c in initialised}},
            )
        return report, listings, None, initialised

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
        assignments: Mapping[int, str] | None = None,
    ) -> tuple[list[MatchRecord], list[Fixture], list[OddsQuote], ParseReport]:
        completed = season_completed(ref.label, now.date())
        if teams is None:
            teams = await self.client.league_teams(ref.season_id, stable=completed)
        if assignments:
            names = self.mapper.adopt(assignments, teams, code, ref.label)
        else:
            names = self.mapper.names_for(teams, code, ref.label, current=current)
        snapshot = await self.client.league_matches_snapshot(
            ref.season_id, stable=completed, max_age=None if completed else self.endpoints.live_matches_cache_ttl
        )
        raw = snapshot.rows
        # an ``incomplete`` match is judged as of the download, not as of now: a cached payload is older than the clock
        reference = snapshot.reference_time(now)
        if reference < now:
            log.info(
                "league-matches served from the raw cache: incomplete matches judged as of the download",
                extra={
                    "competition": code,
                    "season": ref.label,
                    "fetched_at": reference.isoformat(),
                    "age_minutes": round((now - reference).total_seconds() / 60.0, 1),
                },
            )
        report = ParseReport()
        records: list[MatchRecord] = []
        fixtures: list[Fixture] = []
        odds: list[OddsQuote] = []
        out_of_season = 0
        for m in raw:
            parsed = parse_match(m, code, names, now=reference, report=report)
            if isinstance(parsed, MatchRecord):
                records.append(parsed)
                if season_of(parsed.date) != ref.label and not _PLAIN_YEAR_RE.match(ref.label):
                    out_of_season += 1
            elif isinstance(parsed, Fixture) and current:
                fixtures.append(parsed)
                # pre-match quotes only for a match that has not kicked off yet: a postponed match has no date and
                # a match in play (or kicked off since the download) is no longer pre-match, so its quotes are stale
                if with_odds and parsed.status == "scheduled" and (parsed.kickoff is None or parsed.kickoff > now):
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
        if report.postponed:
            log.info(
                "postponed matches kept in the calendar (incomplete with a past kick-off)",
                extra={"competition": code, "season": ref.label, "matches": report.postponed},
            )
        if report.in_play:
            log.info(
                "matches in play kept in the calendar as scheduled (kicked off less than 3 hours ago)",
                extra={"competition": code, "season": ref.label, "matches": report.in_play},
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
        sync_registry: bool = True,
        auto_init_registry: bool | None = None,
    ) -> FootyStatsHarvest:
        """Harvest the newest ``seasons`` seasons of every competition in ``codes``.

        Fixtures (scheduled -- matches in play included -- and postponed, with
        their kick-off) and, with ``with_odds``, the pre-match quotes of the
        scheduled ones that have not kicked off come from the newest season of
        each competition only. The ``/league-matches`` of a season still being
        played is re-downloaded once its cached copy is older than
        :attr:`FootyStatsEndpoints.live_matches_cache_ttl`, and its
        ``incomplete`` matches are judged as of the instant the payload was
        downloaded (:meth:`MatchesSnapshot.reference_time`), never as of a later
        clock. A season that fails is recorded
        in :attr:`FootyStatsHarvest.failures` and the rest is kept; a competition
        the schema cannot store yet is recorded in :attr:`FootyStatsHarvest.skipped`.
        A competition whose league is **not among the account's chosen leagues**
        is skipped too (:attr:`FootyStatsHarvest.not_chosen` and ``skipped``, with
        the advice to add it on footystats.org) and nothing of it is fetched: the
        API would refuse it, and a skipped competition is not a failure, so the
        other competitions are still written. The same happens when
        ``/league-list`` was unreachable, the configured season ids stood in and
        the API then refuses a league as not chosen (:func:`is_not_chosen_refusal`).

        With ``sync_registry`` (the default) the dictionary of the registry season
        (:attr:`registry_path`) is first made equal to ``/league-teams`` of that
        season for every resolved competition (:func:`apply_registry_sync`: clubs
        added, moved, renamed and removed, identity by FootyStats id then by
        exact/alias spelling, never fuzzy); the registry and the mapper are
        reloaded, the names of the current season come from the sync's own
        assignments and the report is :attr:`FootyStatsHarvest.registry_sync`. A
        sync that cannot run (I/O error, a dictionary that would not load) is
        logged and recorded in ``registry_sync_error``: the harvest goes on with
        the dictionary as it was. ``auto_init_registry`` is the former name of
        the switch and, when given, wins over ``sync_registry``. Raises
        :class:`HarvestError` when the seasons cannot be resolved at all, or when
        every season of every competition failed.
        """
        if auto_init_registry is not None:
            sync_registry = bool(auto_init_registry)
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
        not_chosen: dict[str, str] = {}
        resolved: list[str] = []
        sync_report: RegistrySyncReport | None = None
        sync_error: str | None = None

        def skip_not_chosen(code: str, league: str) -> None:
            not_chosen[code] = league
            skipped[code] = not_chosen_reason(league)

        if supported:
            resolution = await self.client.season_resolution(supported, seasons=seasons)
            season_refs = resolution.seasons
            for code, league in resolution.not_chosen.items():
                skip_not_chosen(code, league)
            resolved = [code for code in supported if code in season_refs]
            if sync_registry and resolved:
                sync_report, prefetched, sync_error, initialised = await self._sync_current_registry(resolved, season_refs, now=now)
            for code in resolved:
                refs = season_refs.get(code, [])
                labels[code] = [ref.label for ref in refs]
                counts[code] = {}
                for index, ref in enumerate(refs):
                    is_registry_season = ref.label == self.registry.season_label
                    listing = prefetched.get(code) if index == 0 and is_registry_season else None
                    assignments = sync_report.assignments.get(code) if sync_report is not None and is_registry_season else None
                    try:
                        season_records, season_fixtures, season_odds, report = await self._harvest_season(
                            code, ref, current=index == 0, with_odds=with_odds, now=now, teams=listing, assignments=assignments
                        )
                    except HarvestError as exc:
                        if is_not_chosen_refusal(exc):
                            # /league-list was unreachable and the configured ids stood in: the account has not
                            # chosen this league, so every season of it is refused -> a skip, not a failure
                            league = footystats_league_name(code, self.endpoints)
                            skip_not_chosen(code, league)
                            labels.pop(code, None)
                            counts.pop(code, None)
                            for key in [k for k in failures if k.split(":", 1)[0] == code]:
                                failures.pop(key)
                            log.warning(
                                "competition skipped: FootyStats refused the league as not chosen by the account",
                                extra={"competition": code, "league": league, "season": ref.label, "season_id": ref.season_id, "advice": skipped[code]},
                            )
                            break
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
        if not_chosen:
            log.warning(
                "competitions skipped: leagues not among the account's chosen leagues",
                extra={"competitions": list(not_chosen), "leagues": list(not_chosen.values()), "advice": "add them on footystats.org (Choose leagues) or narrow --competitions"},
            )
        if resolved and not records and not fixtures and last_error is not None:
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
            registry_path=self.registry_path if sync_report is not None and sync_report.written else None,
            not_chosen=not_chosen,
            registry_sync=sync_report,
            registry_sync_error=sync_error,
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
    one club, or the dictionary would not load). ``suggestions`` (written clean
    name -> dictionary name it resembles) are teams written with their FootyStats
    ``cleanName`` although a *fuzzy* match pointed at a name the dictionaries
    know: the resemblance is never adopted (``'Trestina'``, Serie D, is not
    ``'Triestina'``, a former Serie C club), only reported for a manual check.
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
    suggestions: dict[str, str] = field(default_factory=dict)

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
        if self.suggestions:
            lines.append(
                f"  {len(self.suggestions)} teams written with their FootyStats clean name although they resemble a dictionary name "
                "(not adopted; check by hand): " + "; ".join(f"{written} ~ {known}" for written, known in self.suggestions.items())
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
    with historical aliases) recognises one of the FootyStats *identity*
    spellings (``cleanName``, ``name``, ``english_name``) through an **exact or
    alias** hit, else ``cleanName``. A fuzzy hit is never adopted: a Serie D club
    whose name merely resembles a former Serie C club (``'Trestina'`` vs
    ``'Triestina'``) keeps its own name and the resemblance is reported in
    ``suggestions``; cross-division continuity works through the aliases of the
    previous season's file. The aliases are the existing ones plus ``name``,
    ``english_name`` and ``alt_names``. With ``keep_existing_aliases=False`` the
    existing aliases are *replaced* by the FootyStats spellings and what was
    dropped is reported in ``removed_aliases``. Teams already in the file are
    never removed, even when FootyStats no longer lists them: they are reported
    in ``unlisted_teams`` for a manual check. A team that would land in two
    gironi is reported in ``conflicts`` and left where it was. **Two FootyStats
    clubs answering to one dictionary club** (different team ids whose
    spellings resolve to the same canonical, ``'AC Palermo'`` in Serie D and
    ``'Palermo'`` in Serie B) are never merged: the canonical stays with the club
    listed in the competition that holds it (in the file, else in ``matcher``'s
    registry), else with the one spelled exactly like it, else with the first
    listed; the other gets the registry sync's collision name
    (:func:`_derived_names`: ``'AC Palermo Serie D'``, the name the dictionaries
    already know when they do) and the split is reported in ``conflicts``. A new file gets a
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
    suggestions: dict[str, str] = {}
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

    def identity(team: TeamInfo, owners: Mapping[str, str]) -> tuple[str, str | None]:
        """``(canonical, resembles)``: an exact/alias hit of an identity spelling, else ``cleanName`` (never fuzzy)."""
        resembles: str | None = None
        if matcher is not None:
            for spelling in team.identity_spellings:
                result = matcher.match(spelling, None)
                if result.canonical is None:
                    continue
                if result.method in {"exact", "alias"}:
                    return result.canonical, None
                if resembles is None:
                    resembles = result.canonical
        for spelling in team.identity_spellings:
            owner = owners.get(normalize_name(spelling))
            if owner is not None:
                return owner, None
        # never adopt a fuzzy hit: the pool of historical names grows every season and a Serie D
        # club whose name resembles a former Serie C club ('Trestina' / 'Triestina', score 94)
        # would silently become that club in the dictionary and in the joint fit
        return (team.clean_name or team.name), resembles

    def home_of(canonical: str) -> str | None:
        """Competition that holds ``canonical``: the file's, else the one of ``matcher``'s registry."""
        for g, members in gironi.items():
            if canonical in members:
                return g
        if matcher is not None:
            try:
                return matcher.registry.girone_of(canonical).value
            except KeyError:
                return None
        return None

    # 1. identity of every listed club (a running table, so clubs of different competitions sharing a
    #    normalised spelling are seen even when neither is in the dictionaries yet)
    work: list[tuple[str, TeamInfo, str, str | None]] = []
    listed_codes: list[str] = []
    running = spelling_owners()
    for raw_code, teams in teams_by_code.items():
        code = normalize_code(raw_code)
        if not schema_supports(code):
            skipped[code] = f"the match schema (Girone) does not accept competition code {code!r} yet"
            continue
        listed_codes.append(code)
        for team in teams:
            canonical, resembles = identity(team, running)
            running.setdefault(normalize_name(canonical), canonical)
            work.append((code, team, canonical, resembles))
    # 2. two FootyStats clubs (different ids) on one canonical: the canonical stays with one of them
    claimants: dict[str, list[int]] = {}
    for index, (_code, team, canonical, _r) in enumerate(work):
        claimants.setdefault(canonical, []).append(index)
    taken = {normalize_name(c) for c in claimants} | set(spelling_owners())
    if matcher is not None:
        taken |= set(matcher.registry.alias_table())
    for canonical, indexes in claimants.items():
        if len({work[i][1].id for i in indexes}) < 2:
            continue
        home = home_of(canonical)
        exact = [i for i in indexes if any(s.casefold() == canonical.casefold() for s in work[i][1].identity_spellings)]
        by_home = [i for i in indexes if work[i][0] == home]
        keeper = (by_home or exact or indexes)[0]
        for i in indexes:
            code, team, _c, _r = work[i]
            if i == keeper or team.id == work[keeper][1].id:
                continue
            renamed: str | None = None
            for candidate in _derived_names(team, code):
                known = matcher.match(candidate, None) if matcher is not None else None
                if known is not None and known.canonical is not None and known.method in {"exact", "alias"} and known.canonical not in claimants:
                    renamed = known.canonical  # the name the dictionaries already give this club
                    break
                if normalize_name(candidate) not in taken:
                    renamed = candidate
                    break
            if renamed is None:
                renamed = f"{(team.clean_name or team.name).strip()} ({code}, id {team.id})"
            taken.add(normalize_name(renamed))
            work[i] = (code, team, renamed, None)
            conflicts[team.clean_name or team.name] = (
                f"answers to {canonical!r} like another FootyStats club ({work[keeper][1].clean_name}, {work[keeper][0]}): "
                f"written in {code} as {renamed!r}, not merged"
            )
            log.warning(
                "two FootyStats clubs answer to one dictionary club: not merged",
                extra={"path": str(path), "competition": code, "team": team.clean_name, "team_id": team.id, "canonical": canonical, "written_as": renamed, "kept_by": work[keeper][1].clean_name},
            )
    # 3. write the listing
    for code in listed_codes:
        gironi.setdefault(code, {})
        listed_by_code.setdefault(code, set())
    owners = spelling_owners()
    for code, team, canonical, resembles in work:
        group = gironi[code]
        listed = listed_by_code[code]
        if resembles is not None and normalize_name(resembles) != normalize_name(canonical):
            suggestions[canonical] = resembles
            log.warning(
                "FootyStats team written with its clean name although it resembles a dictionary name (check by hand)",
                extra={"path": str(path), "competition": code, "team": canonical, "resembles": resembles, "spellings": list(team.identity_spellings)},
            )
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
        suggestions=suggestions,
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
            "suggestions": suggestions,
            "unlisted": {c: len(t) for c, t in unlisted.items()},
        },
    )
    return diff


# =============================================================================
# Registry sync: the current season's dictionary = FootyStats /league-teams
# =============================================================================
@dataclass(frozen=True)
class CompetitionSync:
    """What :func:`apply_registry_sync` changed in one competition of the dictionary.

    ``added`` are clubs new to the dictionary, ``removed`` clubs FootyStats no
    longer lists for the competition (dropped from the current season; older
    season files keep them as history), ``moved`` the clubs that were in another
    competition of the dictionary and FootyStats lists here, as ``(canonical,
    previous code)``, ``renamed`` the canonical names that followed a FootyStats
    rename, as ``(old, new)`` (the old spelling stays as an alias). ``total`` is
    the number of clubs of the competition after the sync.
    """

    code: str
    label: str
    total: int
    added: tuple[str, ...] = ()
    removed: tuple[str, ...] = ()
    moved: tuple[tuple[str, str], ...] = ()
    renamed: tuple[tuple[str, str], ...] = ()

    @property
    def changed(self) -> bool:
        """Whether the composition or a canonical name of the competition changed."""
        return bool(self.added or self.removed or self.moved or self.renamed)

    @property
    def n_changes(self) -> int:
        return len(self.added) + len(self.removed) + len(self.moved) + len(self.renamed)

    def describe(self) -> str:
        """One Italian line: ``'Serie C Girone B (20 squadre): aggiunte Nuova Monterosi; rimosse Guidonia Montecelio'``."""
        parts: list[str] = []
        if self.added:
            parts.append("aggiunte " + ", ".join(self.added))
        if self.removed:
            parts.append("rimosse " + ", ".join(self.removed))
        if self.moved:
            parts.append("spostate qui " + ", ".join(f"{name} (da {_competition_text(origin)})" for name, origin in self.moved))
        if self.renamed:
            parts.append("rinominate " + ", ".join(f"{old} -> {new}" for old, new in self.renamed))
        return f"{self.label} ({self.total} squadre): " + ("; ".join(parts) if parts else "invariata")

    def to_dict(self) -> dict[str, Any]:
        return {
            "code": self.code,
            "label": self.label,
            "total": self.total,
            "added": list(self.added),
            "removed": list(self.removed),
            "moved": [list(item) for item in self.moved],
            "renamed": [list(item) for item in self.renamed],
        }


@dataclass(frozen=True)
class RegistrySyncReport:
    """Outcome of :func:`apply_registry_sync` / :func:`sync_registry`.

    ``competitions`` (code -> :class:`CompetitionSync`, configuration order) are
    the competitions synchronised; ``skipped`` (code -> Italian reason) the
    requested ones left untouched (no listing, an empty listing, a code the
    schema does not know). ``changed`` says the dictionary content changed
    (clubs, moves, renames or aliases) and ``written`` that the file was
    (re)written; ``created`` that it did not exist. ``new_aliases`` /
    ``skipped_aliases`` (canonical -> spellings) are the FootyStats spellings
    added as aliases and those not written because another club already answers
    to them; ``conflicts`` (FootyStats club -> Italian note) are identity
    decisions worth a look (a move refused because the other competition was not
    synchronised in this run). ``assignments`` (code -> FootyStats team id ->
    canonical) is the identity of every listed club, persisted in the id mapping
    with method :data:`SYNC_METHOD`; ``total`` the clubs of the whole dictionary
    after the sync.
    """

    path: Path
    mapping_path: Path
    season: str
    competitions: dict[str, CompetitionSync] = field(default_factory=dict)
    skipped: dict[str, str] = field(default_factory=dict)
    created: bool = False
    changed: bool = False
    written: bool = False
    mapping_changed: bool = False
    total: int = 0
    new_aliases: dict[str, tuple[str, ...]] = field(default_factory=dict)
    skipped_aliases: dict[str, tuple[str, ...]] = field(default_factory=dict)
    conflicts: dict[str, str] = field(default_factory=dict)
    assignments: dict[str, dict[int, str]] = field(default_factory=dict)

    # ------------------------------------------------------------- aggregates
    @property
    def added(self) -> dict[str, tuple[str, ...]]:
        """Code -> clubs added (only competitions with additions)."""
        return {code: c.added for code, c in self.competitions.items() if c.added}

    @property
    def removed(self) -> dict[str, tuple[str, ...]]:
        """Code -> clubs removed (only competitions with removals)."""
        return {code: c.removed for code, c in self.competitions.items() if c.removed}

    @property
    def moved(self) -> dict[str, tuple[str, str]]:
        """Club -> ``(from code, to code)`` for every club moved between competitions."""
        return {name: (origin, code) for code, c in self.competitions.items() for name, origin in c.moved}

    @property
    def renamed(self) -> dict[str, str]:
        """Old canonical -> new canonical for every rename."""
        return {old: new for c in self.competitions.values() for old, new in c.renamed}

    @property
    def n_changes(self) -> int:
        """Clubs added, removed, moved or renamed (aliases not counted)."""
        return sum(c.n_changes for c in self.competitions.values())

    def describe(self) -> str:
        """Italian, multi-line: the headline, one line per competition synchronised, then aliases, conflicts and skips."""
        season = self.season
        where = str(self.path)
        n_comp = len(self.competitions)
        if self.n_changes:
            head = (
                f"Dizionario squadre {season} sincronizzato con FootyStats ({where}): {self.n_changes} variazioni, "
                f"{self.total} squadre in totale."
            )
        elif self.changed:
            head = f"Dizionario squadre {season} allineato a FootyStats ({where}): composizione invariata, alias aggiornati; {self.total} squadre in totale."
        else:
            head = f"Dizionario squadre {season} già allineato a FootyStats ({where}): nessuna variazione, {self.total} squadre in totale."
        if self.created:
            head += " File creato."
        lines = [head]
        if n_comp:
            lines.append(f"  Competizioni sincronizzate: {n_comp}")
        lines.extend(f"  {c.describe()}" for c in self.competitions.values())
        if self.new_aliases:
            lines.append(f"  Nuovi alias per {len(self.new_aliases)} squadre: " + "; ".join(f"{c} <- {', '.join(a)}" for c, a in self.new_aliases.items()))
        if self.skipped_aliases:
            lines.append(
                f"  Alias non scritti (grafia già usata da un altro club) per {len(self.skipped_aliases)} squadre: "
                + "; ".join(f"{c}: {', '.join(a)}" for c, a in self.skipped_aliases.items())
            )
        for team, note in self.conflicts.items():
            lines.append(f"  Da controllare: {team}: {note}")
        for code, reason in self.skipped.items():
            lines.append(f"  {_competition_text(code)}: non sincronizzata ({reason})")
        return "\n".join(lines)

    def to_dict(self) -> dict[str, Any]:
        return {
            "path": str(self.path),
            "mapping_path": str(self.mapping_path),
            "season": self.season,
            "created": self.created,
            "changed": self.changed,
            "written": self.written,
            "mapping_changed": self.mapping_changed,
            "total": self.total,
            "n_changes": self.n_changes,
            "competitions": {code: c.to_dict() for code, c in self.competitions.items()},
            "skipped": dict(self.skipped),
            "new_aliases": {c: list(a) for c, a in self.new_aliases.items()},
            "skipped_aliases": {c: list(a) for c, a in self.skipped_aliases.items()},
            "conflicts": dict(self.conflicts),
        }

    def log_extra(self) -> dict[str, Any]:
        """The fields of the structured log line ``team dictionary synchronised with FootyStats``."""
        return {
            "path": str(self.path),
            "season": self.season,
            "changed": self.changed,
            "written": self.written,
            "file_created": self.created,
            "total": self.total,
            "competitions": list(self.competitions),
            "added": {code: list(names) for code, names in self.added.items()},
            "removed": {code: list(names) for code, names in self.removed.items()},
            "moved": {name: list(codes) for name, codes in self.moved.items()},
            "renamed": dict(self.renamed),
            "new_aliases": len(self.new_aliases),
            "skipped_aliases": len(self.skipped_aliases),
            "conflicts": len(self.conflicts),
            "skipped": dict(self.skipped),
        }


def _competition_text(code: str) -> str:
    """``'Serie C Girone B'`` for ``'B'`` (the code itself when the schema does not know it)."""
    try:
        return Girone.parse(code).label
    except (ValueError, LookupError):
        return str(code)


def _as_team_info(entry: "TeamInfo | Mapping[str, Any]") -> TeamInfo | None:
    if isinstance(entry, TeamInfo):
        return entry
    if isinstance(entry, Mapping):
        return TeamInfo.from_payload(entry)
    raise TypeError(f"expected a TeamInfo or a /league-teams entry, got {type(entry).__name__}")


def _derived_names(team: TeamInfo, code: str) -> list[str]:
    """The suffixed canonical candidates of a club whose own spellings collide: ``cleanName + family``, ``+ label``, ``+ (code)``.

    The family step comes before the label step so that a Serie D club whose
    name normalises like a Serie B club (``'AC Palermo'`` / ``'Palermo'``) reads
    ``'AC Palermo Serie D'``; two Serie D clubs colliding with each other fall
    through to the girone label (``'... Serie D Girone A'``); the code is the
    last resort for a competition the configuration does not describe.
    """
    clean = (team.clean_name or team.name).strip()
    out: list[str] = []
    try:
        member = Girone.parse(code)
        out.extend([f"{clean} {member.family}", f"{clean} {member.label}"])
    except (ValueError, LookupError):
        pass
    out.append(f"{clean} ({code})")
    return list(dict.fromkeys(out))


def _collision_names(team: TeamInfo, code: str) -> list[str]:
    """Canonical candidates of a new club, in order: ``cleanName``, ``name``, then :func:`_derived_names`."""
    out = [team.clean_name, team.name, *_derived_names(team, code)]
    return list(dict.fromkeys(name.strip() for name in out if name and name.strip()))


def _new_registry_payload(label: str, codes: Iterable[str]) -> dict[str, Any]:
    wanted = set(codes)
    return {
        "season": label,
        "competition": competition_label(wanted) or "FootyStats leagues",
        "source": "FootyStats /league-teams (src.ingestion.footystats.sync_registry)",
        "note": (
            "Canonical team names per competition (keys = competition codes of config/competitions.toml), "
            "synchronised with FootyStats /league-teams by every harvest. Aliases are the spellings used by FootyStats, "
            "SofaScore and bookmakers."
        ),
        "gironi": {member.value: {} for member in Girone if member.value in wanted},
    }


def apply_registry_sync(
    teams_by_code: Mapping["Girone | str", Sequence["TeamInfo | Mapping[str, Any]"]],
    *,
    season: str,
    registry_path: Path,
    mapping_path: Path = DEFAULT_MAPPING_PATH,
    competitions: Iterable["Girone | str"] | None = None,
    write: bool = True,
) -> RegistrySyncReport:
    """Make ``registry_path`` equal to the FootyStats ``/league-teams`` listings of ``season`` (pure, no network).

    ``teams_by_code`` maps a competition code to its listing (:class:`TeamInfo`
    objects or raw ``/league-teams`` entries: data already downloaded works as
    is). ``competitions`` (default: the keys of ``teams_by_code``) are the codes
    to synchronise; a requested code without a listing, with an empty listing
    (never read as "every club left") or unknown to the schema is reported in
    :attr:`RegistrySyncReport.skipped` and left untouched. Every other
    competition of the file is left untouched as well.

    Identity of a listed club, in order, **never fuzzy**:

    1. its FootyStats team id in the persisted mapping (``mapping_path``,
       entries with method :data:`SYNC_METHOD`, i.e. written by an earlier sync);
    2. an exact/alias hit of its identity spellings (``cleanName``, ``name``,
       ``english_name``) on a club of the **same** competition;
    3. a club of the same competition named like the collision rule would name
       it (``'AC Palermo Serie D'``, see below);
    4. an exact/alias hit on a club of **another** competition of the dictionary:
       the club is *moved* -- when that competition is synchronised in the same
       run; otherwise the move is refused (``conflicts``) and the club is added as
       new, because only a full view tells a move from two clubs sharing a
       spelling (``'AC Palermo'`` / ``'Palermo'``).

    A club no step identifies is **added** with canonical ``cleanName`` -- or,
    when that spelling already belongs to another club, ``name``, then
    ``cleanName + ' ' + family`` (``'AC Palermo Serie D'``), then ``cleanName + ' '
    + label``; a club id the mapping knows from an earlier sync gets its old
    canonical back. Clubs of a synchronised competition that no listed club
    claims are **removed** (older season files keep them as history). A club
    identified by id whose canonical equals the ``cleanName`` the mapping
    recorded follows a FootyStats rename (``renamed``, the old spelling becomes
    an alias); canonical names chosen by hand (``'Juventus Next Gen'``) never
    change. Every FootyStats spelling (``cleanName``, ``name``, ``english_name``,
    ``alt_names``) is added as an alias unless another club already answers to
    it (``skipped_aliases``); existing aliases are preserved.

    The file is validated with :meth:`TeamRegistry.from_mapping` (``ValueError``
    and nothing written when it would not load) and, when its content changed,
    written atomically in the compact layout of
    :func:`~src.common.teams.dumps_registry_payload`; the id mapping gets one
    :data:`SYNC_METHOD` entry per listed club (atomic write too). ``write=False``
    computes the report without touching any file (a dry run). ``ValueError``
    when the file holds another season than ``season``.
    """
    path = Path(registry_path)
    mapping_file = Path(mapping_path)
    label = str(season).strip().replace("_", "/")
    listings: dict[str, list[TeamInfo]] = {}
    for raw_code, entries in teams_by_code.items():
        code = normalize_code(raw_code)
        listings[code] = [team for team in (_as_team_info(entry) for entry in entries or ()) if team is not None]
    requested = list(dict.fromkeys(normalize_code(code) for code in (competitions if competitions is not None else teams_by_code)))
    skipped: dict[str, str] = {}
    sync_codes: list[str] = []
    for code in requested:
        if not schema_supports(code):
            skipped[code] = f"codice {code!r} sconosciuto allo schema delle competizioni"
        elif code not in listings:
            skipped[code] = "nessuna lista squadre FootyStats disponibile in questa esecuzione"
        elif not listings[code]:
            skipped[code] = "FootyStats non elenca squadre per la stagione: competizione lasciata invariata"
        else:
            sync_codes.append(code)
    order = {member.value: index for index, member in enumerate(Girone)}
    sync_codes.sort(key=lambda code: order.get(code, len(order)))

    created = not path.exists()
    if created:
        payload: dict[str, Any] = _new_registry_payload(label, sync_codes)
        original: dict[str, Any] | None = None
    else:
        loaded = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(loaded, Mapping) or not isinstance(loaded.get("gironi"), Mapping):
            raise ValueError(f"{path} is not a team dictionary (missing 'gironi' table)")
        file_season = str(loaded.get("season") or label).strip().replace("_", "/")
        if file_season != label:
            raise ValueError(f"{path} holds season {file_season!r}, not {label!r}: refusing to synchronise it")
        original = json.loads(json.dumps(loaded))
        payload = dict(loaded)
    gironi: dict[str, dict[str, list[str]]] = {
        normalize_code(code): {str(name): [str(a) for a in (aliases or ())] for name, aliases in dict(members or {}).items()}
        for code, members in dict(payload["gironi"]).items()
    }
    for code in sync_codes:
        gironi.setdefault(code, {})

    location: dict[str, str] = {name: code for code, members in gironi.items() for name in members}
    owners: dict[str, str] = {}
    for code, members in gironi.items():
        for name, aliases in members.items():
            owners.setdefault(normalize_name(name), name)
            for alias in aliases:
                owners.setdefault(normalize_name(alias), name)
    persisted = load_team_mapping(mapping_file)
    synced_ids = {team_id: str(entry["canonical"]).strip() for team_id, entry in persisted.items() if str(entry.get("method") or "") == SYNC_METHOD}
    owner_id: dict[str, int] = {}
    for team_id, canonical in synced_ids.items():
        owner_id.setdefault(canonical, team_id)

    items = [(code, team) for code in sync_codes for team in listings[code]]
    claims: dict[str, tuple[int, str]] = {}  # canonical -> (team id, listing code)
    assigned: dict[tuple[str, int], str] = {}  # (listing code, team id) -> canonical
    via_id: set[tuple[str, int]] = set()
    conflicts: dict[str, str] = {}

    def foreign(canonical: str, team_id: int) -> bool:
        """The canonical belongs (by a synced id) to another FootyStats club."""
        other = owner_id.get(canonical)
        return other is not None and other != team_id

    def claim(code: str, team: TeamInfo, canonical: str) -> None:
        claims[canonical] = (team.id, code)
        assigned[(code, team.id)] = canonical

    # 1. FootyStats id (an earlier sync)
    for code, team in items:
        canonical = synced_ids.get(team.id)
        if canonical and canonical in location and canonical not in claims:
            claim(code, team, canonical)
            via_id.add((code, team.id))
    # 2. exact/alias spelling on a club of the same competition
    for code, team in items:
        if (code, team.id) in assigned:
            continue
        for spelling in team.identity_spellings:
            owner = owners.get(normalize_name(spelling))
            if owner is not None and location.get(owner) == code and owner not in claims and not foreign(owner, team.id):
                claim(code, team, owner)
                break
    # 3. the name the collision rule gives the club in this competition
    for code, team in items:
        if (code, team.id) in assigned:
            continue
        by_norm = {normalize_name(name): name for name in gironi.get(code, {})}
        for candidate in _derived_names(team, code):
            owner = by_norm.get(normalize_name(candidate))
            if owner is not None and owner not in claims and not foreign(owner, team.id):
                claim(code, team, owner)
                break
    # 4. exact/alias spelling on a club of another competition: a move
    for code, team in items:
        if (code, team.id) in assigned:
            continue
        for spelling in team.identity_spellings:
            owner = owners.get(normalize_name(spelling))
            if owner is None or owner in claims or foreign(owner, team.id) or location.get(owner) == code:
                continue
            origin = location[owner]
            if origin in sync_codes:
                claim(code, team, owner)
            else:
                conflicts[team.clean_name] = (
                    f"la grafia {spelling!r} è di {owner} ({_competition_text(origin)}, non sincronizzata in questa esecuzione): "
                    f"non spostata, {team.clean_name} aggiunta come squadra nuova di {_competition_text(code)}"
                )
                log.warning(
                    "registry sync: move refused, the other competition is not synchronised in this run",
                    extra={"team": team.clean_name, "team_id": team.id, "competition": code, "owner": owner, "owner_competition": origin},
                )
            break

    # renames: an id-identified club whose canonical is the cleanName FootyStats used before
    renames: dict[str, str] = {}
    for code, team in items:
        if (code, team.id) not in via_id:
            continue
        canonical = assigned[(code, team.id)]
        previous_clean = str(persisted.get(team.id, {}).get("clean_name") or "").strip()
        new_clean = (team.clean_name or team.name).strip()
        if previous_clean and canonical == previous_clean and new_clean and normalize_name(new_clean) != normalize_name(canonical):
            holder = owners.get(normalize_name(new_clean))
            if holder is None or holder == canonical:
                renames[canonical] = new_clean

    # the new composition of the synchronised competitions
    claimed_by_code: dict[str, list[str]] = {code: [] for code in sync_codes}
    for (code, _team_id), canonical in assigned.items():
        claimed_by_code[code].append(canonical)
    removed: dict[str, list[str]] = {}
    moved: dict[str, list[tuple[str, str]]] = {}
    new_gironi: dict[str, dict[str, list[str]]] = {code: dict(members) for code, members in gironi.items()}
    for code in sync_codes:
        keep = set(claimed_by_code[code])
        survivors = {name: aliases for name, aliases in gironi[code].items() if name in keep}
        gone = [name for name in gironi[code] if name not in keep and claims.get(name, (None, code))[1] == code]
        removed[code] = gone
        new_gironi[code] = survivors
    for code, team in items:
        canonical = assigned.get((code, team.id))
        if canonical is None or location.get(canonical) == code:
            continue
        origin = location[canonical]
        moved.setdefault(code, []).append((canonical, origin))
        new_gironi[code][canonical] = list(gironi[origin][canonical])
        new_gironi[origin].pop(canonical, None)

    # spellings of the clubs that remain (removed clubs free theirs)
    def spelling_owners(groups: Mapping[str, Mapping[str, Sequence[str]]]) -> dict[str, str]:
        table: dict[str, str] = {}
        for members in groups.values():
            for name, aliases in members.items():
                table.setdefault(normalize_name(name), name)
                for alias in aliases:
                    table.setdefault(normalize_name(alias), name)
        return table

    current_owners = spelling_owners(new_gironi)

    # renames applied (the old canonical stays as an alias)
    renamed: dict[str, list[tuple[str, str]]] = {}
    for old, new in renames.items():
        code = next(c for c, members in new_gironi.items() if old in members)
        new_gironi[code] = {
            (new if name == old else name): ([old, *(a for a in aliases if normalize_name(a) != normalize_name(new))] if name == old else aliases)
            for name, aliases in new_gironi[code].items()
        }
        renamed.setdefault(code, []).append((old, new))
        for key in [k for k, v in assigned.items() if v == old]:
            assigned[key] = new
        current_owners = spelling_owners(new_gironi)

    # new clubs
    added: dict[str, list[str]] = {}
    for code, team in items:
        if (code, team.id) in assigned:
            continue
        preferred = synced_ids.get(team.id)
        candidates = ([preferred] if preferred else []) + _collision_names(team, code)
        chosen: str | None = None
        for candidate in candidates:
            if not candidate:
                continue
            norm = normalize_name(candidate)
            if norm in current_owners or any(candidate in members for members in new_gironi.values()):
                continue
            chosen = candidate
            break
        if chosen is None:
            conflicts[team.clean_name] = "nessun nome canonico libero (ogni candidato è già usato da un altro club): non aggiunta"
            log.error("registry sync: no free canonical name, club not added", extra={"team": team.clean_name, "team_id": team.id, "competition": code})
            continue
        new_gironi[code][chosen] = []
        current_owners[normalize_name(chosen)] = chosen
        assigned[(code, team.id)] = chosen
        added.setdefault(code, []).append(chosen)

    # FootyStats spellings as aliases (existing aliases preserved, collisions refused)
    new_aliases: dict[str, list[str]] = {}
    skipped_aliases: dict[str, list[str]] = {}
    for code, team in items:
        canonical = assigned.get((code, team.id))
        if canonical is None or canonical not in new_gironi[code]:
            continue
        aliases = new_gironi[code][canonical]
        known = {normalize_name(canonical), *(normalize_name(a) for a in aliases)}
        for spelling in team.spellings:
            norm = normalize_name(spelling)
            if norm in known or spelling == canonical:
                continue
            holder = current_owners.get(norm)
            if holder is not None and holder != canonical:
                skipped_aliases.setdefault(canonical, []).append(spelling)
                continue
            aliases.append(spelling)
            known.add(norm)
            current_owners[norm] = canonical
            new_aliases.setdefault(canonical, []).append(spelling)

    # never write a dictionary the registry would refuse to load (the whole app reads it at start-up)
    try:
        registry = TeamRegistry.from_mapping(label, new_gironi)
    except ValueError as exc:
        raise ValueError(f"{path}: the synchronised dictionary would not load, nothing written: {exc}") from exc
    ordered_codes = [m.value for m in Girone if m.value in new_gironi] + [c for c in new_gironi if c not in order]
    payload["season"] = label
    payload["gironi"] = {code: new_gironi[code] for code in ordered_codes}
    changed = original is None or json.loads(json.dumps(payload)) != original

    per_competition = {
        code: CompetitionSync(
            code=code,
            label=_competition_text(code),
            total=len(new_gironi[code]),
            added=tuple(added.get(code, ())),
            removed=tuple(removed.get(code, ())),
            moved=tuple(moved.get(code, ())),
            renamed=tuple(renamed.get(code, ())),
        )
        for code in sync_codes
    }
    assignments: dict[str, dict[int, str]] = {code: {} for code in sync_codes}
    for (code, team_id), canonical in assigned.items():
        assignments[code][team_id] = canonical

    # the id mapping: one sync entry per listed club, every other entry preserved
    mapping = {team_id: dict(entry) for team_id, entry in persisted.items()}
    for code, team in items:
        canonical = assigned.get((code, team.id))
        if canonical is None:
            continue
        previous = mapping.get(team.id, {})
        mapping[team.id] = {
            "canonical": canonical,
            "clean_name": team.clean_name,
            "competition": code,
            "seasons": sorted(set(previous.get("seasons", [])) | {label}, reverse=True),
            "method": SYNC_METHOD,
        }
    mapping_changed = mapping != persisted or not mapping_file.exists()

    written = False
    if write:
        if changed:
            _atomic_write_text(path, dumps_registry_payload(payload))
            written = True
        if mapping_changed and mapping:
            _atomic_write_text(mapping_file, _mapping_payload(mapping))
    report = RegistrySyncReport(
        path=path,
        mapping_path=mapping_file,
        season=label,
        competitions=per_competition,
        skipped=skipped,
        created=created,
        changed=changed,
        written=written,
        mapping_changed=mapping_changed,
        total=registry.n_teams,
        new_aliases={c: tuple(a) for c, a in new_aliases.items()},
        skipped_aliases={c: tuple(a) for c, a in skipped_aliases.items()},
        conflicts=conflicts,
        assignments=assignments,
    )
    log.info("team dictionary synchronised with FootyStats" if write else "team dictionary sync computed (dry run)", extra=report.log_extra())
    return report


async def _fetch_current_listings(
    client: "FootyStatsClient",
    codes: Sequence[str],
    season_label: str,
    *,
    today: date | None = None,
) -> tuple[dict[str, list[TeamInfo]], dict[str, str]]:
    """``/league-teams`` of ``season_label`` for every code, plus the Italian reason of each code left without a listing."""
    resolution = await client.season_resolution(codes, seasons=MAX_PAGES)
    listings: dict[str, list[TeamInfo]] = {}
    missing: dict[str, str] = {}
    for code in codes:
        if code in resolution.not_chosen:
            missing[code] = f"lega {resolution.not_chosen[code]!r} non scelta nell'account FootyStats"
            continue
        ref = next((r for r in resolution.seasons.get(code, ()) if r.label == season_label), None)
        if ref is None:
            missing[code] = f"FootyStats non elenca la stagione {season_label}"
            continue
        try:
            listings[code] = await client.league_teams(ref.season_id, stable=season_completed(ref.label, today))
        except HarvestError as exc:
            missing[code] = "lista squadre non disponibile: " + client.mask(exc)
    return listings, missing


async def sync_registry(
    source: "FootyStatsClient | Mapping[Girone | str, Sequence[TeamInfo | Mapping[str, Any]]]",
    *,
    season: str,
    registry_path: Path,
    mapping_path: Path = DEFAULT_MAPPING_PATH,
    competitions: Iterable["Girone | str"] | None = None,
    write: bool = True,
) -> RegistrySyncReport:
    """Synchronise ``registry_path`` with FootyStats ``/league-teams`` of ``season`` (see :func:`apply_registry_sync`).

    ``source`` is either a :class:`FootyStatsClient` -- the listing of every
    competition in ``competitions`` (default: :data:`DEFAULT_COMPETITIONS`) is
    fetched for ``season``; a competition the account has not chosen, that does
    not list the season or whose listing fails is reported in ``skipped`` and
    left untouched -- or the listings themselves (code -> :class:`TeamInfo` or
    raw ``/league-teams`` entries), e.g. data already downloaded, in which case
    nothing is fetched. Returns the :class:`RegistrySyncReport`.
    """
    label = str(season).strip().replace("_", "/")
    if isinstance(source, Mapping):
        return apply_registry_sync(source, season=label, registry_path=registry_path, mapping_path=mapping_path, competitions=competitions, write=write)
    codes = list(dict.fromkeys(normalize_code(code) for code in (competitions if competitions is not None else DEFAULT_COMPETITIONS)))
    supported = [code for code in codes if schema_supports(code)]
    listings, missing = await _fetch_current_listings(source, supported, label)
    report = apply_registry_sync(listings, season=label, registry_path=registry_path, mapping_path=mapping_path, competitions=codes, write=write)
    if missing:
        skipped = {**report.skipped, **missing}
        report = dataclasses.replace(report, skipped={code: skipped[code] for code in codes if code in skipped})
    return report


# =============================================================================
# CLI
# =============================================================================
def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m src.ingestion.footystats",
        description=(
            "Harvest the configured competitions (Serie C gironi A/B/C, Serie B, Serie D gironi A-I) from the official "
            "FootyStats API into the canonical CSVs (local files under data/, git-ignored), with kick-off times, postponed "
            "matches and pre-match odds. The team dictionary of the current season is synchronised with /league-teams first "
            "(clubs added, moved, renamed or removed so that it equals FootyStats)."
        ),
    )
    parser.add_argument(
        "--competitions",
        nargs="+",
        default=list(DEFAULT_COMPETITIONS),
        metavar="CODE",
        help="competition codes of config/competitions.toml: "
        + ", ".join(f"{member.value} = {member.label}" for member in Girone)
        + "; default: every configured competition (%(default)s). Every league must be chosen on the FootyStats account: "
        "a competition whose league is not chosen is skipped with a warning (the others are still written)",
    )
    parser.add_argument(
        "--no-sync",
        action="store_true",
        help="do not synchronise config/teams_<season>.json with /league-teams before harvesting (the dictionary is used as it is)",
    )
    parser.add_argument("--no-auto-init", action="store_true", help="former name of --no-sync")
    parser.add_argument(
        "--sync-registry",
        action="store_true",
        help="only synchronise config/teams_<season>.json with /league-teams of --season (no matches, no CSVs), print the report, then exit",
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
    parser.add_argument(
        "--no-odds",
        action="store_true",
        help="do not collect pre-match odds: --odds-out is not rewritten, only its quotes of postponed or already "
        "kicked-off matches are removed (stale prices)",
    )
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
    parser.add_argument("--mapping", type=Path, default=DEFAULT_MAPPING_PATH, help="persisted team-id mapping JSON (default: %(default)s, git-ignored)")
    parser.add_argument("--endpoints", type=Path, default=CONFIG_DIR / "endpoints.toml", help="endpoints TOML file")
    parser.add_argument(
        "--config-dir",
        type=Path,
        default=CONFIG_DIR,
        help="directory of the team dictionaries: the perimeter is read from, and the registry sync writes to, "
        "<config-dir>/teams_<season>.json (default: %(default)s)",
    )
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
    # the registry sync writes the dictionary it was read from: --config-dir, never another one
    registry_path = Path(args.config_dir) / f"teams_{season_file_stem(args.season)}.json"
    provider = FootyStatsProvider(client, mapper, endpoints.footystats, config_dir=Path(args.config_dir), registry_path=registry_path)
    sync_enabled = not (args.no_sync or args.no_auto_init)

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

    async def run_sync() -> int:
        async with client:
            report = await sync_registry(
                client,
                season=registry.season_label,
                registry_path=registry_path,
                mapping_path=args.mapping,
                competitions=codes,
            )
        print(report.describe())
        return 0 if report.competitions else 1

    async def run_harvest() -> int:
        async with client:
            result = await provider.harvest(codes, seasons=args.seasons, with_odds=not args.no_odds, sync_registry=sync_enabled)
        complete = not result.failures and len(result.matches) > 0
        write, reason = _write_decision(result, args.out, allow_partial=args.allow_partial)
        print(result.summary())
        if result.registry_sync is not None:
            print(result.registry_sync.describe())
        elif result.registry_sync_error:
            print(f"dizionario squadre non sincronizzato: {result.registry_sync_error}", file=sys.stderr)
        perimeter = provider.registry  # reloaded after the registry sync
        print(f"perimeter: season {perimeter.season_label} ({len(perimeter.canonical_names)} current teams)")
        if result.initialised_competitions:
            print(
                "registry initialised: "
                + ", ".join(f"{code} ({len(perimeter.teams(code))} teams)" for code in result.initialised_competitions)
                + f" -> {registry_path}"
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
            print(f"fixtures -> {args.fixtures_out} ({len(result.fixtures)} fixtures, {result.n_postponed} postponed)")
            if not args.no_odds:
                save_odds(result.odds, args.odds_out)
                print(f"odds -> {args.odds_out}")
            else:
                pruned = prune_stale_odds(args.odds_out, result.fixtures, now=_utc_now())
                if pruned:
                    print(f"odds: {pruned} stale quote rows removed from {args.odds_out} (postponed or already kicked-off matches)")
        else:
            log.error("harvest output not written", extra={"reason": reason, "out": str(args.out)})
            print(reason, file=sys.stderr)
        if result.unmapped_teams:
            print("unmapped teams: " + ", ".join(f"{team} x{count}" for team, count in result.unmapped_teams.items()))
        for name, message in result.failures.items():
            print(f"failure {name}: {mask_key(message, key)}", file=sys.stderr)
        for code, reason in result.skipped.items():
            print(f"skipped {code}: {reason}", file=sys.stderr)
        if result.not_chosen:
            print(
                "leagues not chosen on the FootyStats account (competitions skipped, not failures): "
                + ", ".join(f"{code} = {league}" for code, league in result.not_chosen.items())
                + " -- add them on footystats.org (Choose leagues) or narrow --competitions",
                file=sys.stderr,
            )
        if not complete:
            print("harvest incomplete: " + ("no matches harvested" if result.matches.empty else f"{len(result.failures)} failures"), file=sys.stderr)
        return 0 if complete else 1

    try:
        if args.list_leagues:
            return asyncio.run(run_list())
        if args.init_registry:
            return asyncio.run(run_init())
        if args.sync_registry:
            return asyncio.run(run_sync())
        return asyncio.run(run_harvest())
    except HarvestError as exc:
        message = mask_key(str(exc), key)
        log.error("footystats failed", extra={"error": message, "status": exc.status})
        print(f"footystats failed: {message}", file=sys.stderr)
        return 1
    except (ValueError, OSError) as exc:  # a malformed dictionary file (--init-registry / --sync-registry) or an unwritable one
        print(f"footystats failed: {mask_key(exc, key)}", file=sys.stderr)
        return 1


if __name__ == "__main__":  # pragma: no cover - manual entry point
    raise SystemExit(main())
