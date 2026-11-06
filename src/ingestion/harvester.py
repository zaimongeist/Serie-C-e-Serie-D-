"""Async ingestion layer: SofaScore (primary) and FotMob (secondary) harvesters.

The module is organised in layers that can be used independently:

* :func:`load_endpoints` -- parse ``config/endpoints.toml`` into frozen
  dataclasses (:class:`EndpointsConfig`).
* :class:`RawCache` -- JSON file cache of raw responses keyed by the full URL.
* :class:`AsyncHarvesterClient` -- polite ``httpx`` client: user-agent rotation,
  realistic headers, bounded concurrency, minimum inter-request delay, retries
  with exponential full-jitter backoff and ``Retry-After`` support.
* :class:`SofaScoreProvider` / :class:`FotMobProvider` -- endpoint navigation and
  parsers from the undocumented JSON payloads to :class:`MatchRecord` /
  :class:`Fixture`.
* :class:`Harvester` -- facade that harvests every girone through the configured
  providers and returns a validated canonical ``DataFrame``; also offers CSV
  persistence helpers and a CLI (``python -m src.ingestion.harvester``).

Seasons and perimeter
---------------------
The team dictionary of one season (``config/teams_<season>.json``,
:data:`src.common.teams.DEFAULT_SEASON` by default) is the *perimeter*: its
teams are the current teams. SofaScore exposes the previous seasons of every
girone, and :meth:`SofaScoreProvider.harvest_seasons` /
:meth:`Harvester.harvest_all` harvest the newest ``seasons`` of them (three by
default) into **one** frame with the fixed 16-column schema -- no season column,
the season of a match is :func:`src.common.schemas.season_of` of its date.
Fixtures (not-started events) come from the current season only. Team names of
past seasons are resolved with the current dictionary extended by the aliases
of the older dictionaries (:meth:`TeamRegistry.with_historical_aliases`): a
club that still plays keeps its current canonical name whatever girone it was
in, a club that left keeps its old canonical spelling and is reported in
:attr:`HarvestResult.historical_teams`. :attr:`HarvestResult.seasons` and
:attr:`HarvestResult.n_matches_by_season` say what the frame holds. Payloads
of completed seasons are immutable and are cached with the long
``[http] history_cache_ttl_hours`` (365 days) instead of ``cache_ttl_hours``
(6 h), so a multi-season run repeated later re-downloads the current season only.

Nothing here performs network I/O at import time, and every network path can
be driven by an injected ``httpx`` transport, a fake sleeper and a seeded RNG,
so the whole module is testable offline.
"""
from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import math
import os
import random
import re
import sys
import tempfile
import time
import tomllib
from collections import Counter
from collections.abc import Awaitable, Callable, Coroutine, Iterable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from datetime import date, datetime, timedelta, timezone
from email.utils import parsedate_to_datetime
from pathlib import Path
from typing import Any, Final, TypeVar
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

import httpx
import pandas as pd

from src.common.logging_config import configure_logging, get_logger
from src.common.paths import CONFIG_DIR, ODDS_DIR, PROCESSED_DIR, RAW_DIR
from src.common.schemas import (
    MATCH_COLUMNS,
    Fixture,
    Girone,
    MatchRecord,
    OddsQuote,
    empty_matches_frame,
    matches_to_frame,
    season_of,
    validate_matches_df,
)
from src.common.teams import DEFAULT_SEASON, TeamRegistry, list_seasons, load_registry, season_file_stem
from src.ingestion.team_matcher import TeamMatcher

__all__ = [
    "DEFAULT_USER_AGENTS",
    "FIXTURE_COLUMNS",
    "FOTMOB_STAT_KEYS",
    "ODDS_COLUMNS",
    "PROVIDER_NAMES",
    "STAT_FIELDS",
    "AsyncHarvesterClient",
    "ClientStats",
    "EndpointsConfig",
    "FootyStatsEndpoints",
    "FotMobEndpoints",
    "FotMobProvider",
    "HarvestError",
    "HarvestResult",
    "Harvester",
    "HttpConfig",
    "NotFoundError",
    "RateLimitedError",
    "RawCache",
    "SeasonHarvest",
    "SeasonInfo",
    "SofaScoreEndpoints",
    "SofaScoreProvider",
    "default_providers",
    "load_endpoints",
    "load_fixtures",
    "load_processed",
    "main",
    "parse_retry_after",
    "parse_stat_value",
    "run_concurrently",
    "save_fixtures",
    "save_odds",
    "save_processed",
    "season_label_from_year",
    "site_origin",
]

log = get_logger(__name__)

#: Realistic, current desktop and mobile browser signatures (rotated per request).
DEFAULT_USER_AGENTS: Final[tuple[str, ...]] = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/136.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/135.0.0.0 Safari/537.36",
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/136.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:138.0) Gecko/20100101 Firefox/138.0",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 14.7; rv:137.0) Gecko/20100101 Firefox/137.0",
    "Mozilla/5.0 (X11; Ubuntu; Linux x86_64; rv:136.0) Gecko/20100101 Firefox/136.0",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/18.4 Safari/605.1.15",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/136.0.0.0 Safari/537.36 Edg/136.0.0.0",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/135.0.0.0 Safari/537.36 OPR/119.0.0.0",
    "Mozilla/5.0 (Linux; Android 15; Pixel 9) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/136.0.0.0 Mobile Safari/537.36",
    "Mozilla/5.0 (Linux; Android 14; SM-S928B) AppleWebKit/537.36 (KHTML, like Gecko) SamsungBrowser/27.0 Chrome/125.0.0.0 Mobile Safari/537.36",
    "Mozilla/5.0 (iPhone; CPU iPhone OS 18_4 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/18.4 Mobile/15E148 Safari/604.1",
    "Mozilla/5.0 (iPad; CPU OS 18_3 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/18.3 Mobile/15E148 Safari/604.1",
    "Mozilla/5.0 (Android 15; Mobile; rv:138.0) Gecko/138.0 Firefox/138.0",
)

#: Optional statistic fields of :class:`MatchRecord`, in canonical column order.
STAT_FIELDS: Final[tuple[str, ...]] = (
    "home_shots",
    "away_shots",
    "home_shots_target",
    "away_shots_target",
    "home_corners",
    "away_corners",
    "home_xg",
    "away_xg",
)

#: Metric name (as used in ``stat_keys``) -> (home field, away field).
_METRIC_FIELDS: Final[dict[str, tuple[str, str]]] = {
    "shots": ("home_shots", "away_shots"),
    "shots_target": ("home_shots_target", "away_shots_target"),
    "corners": ("home_corners", "away_corners"),
    "xg": ("home_xg", "away_xg"),
}

#: FotMob ``matchDetails`` statistic keys/titles per metric.
FOTMOB_STAT_KEYS: Final[dict[str, tuple[str, ...]]] = {
    "shots": ("total_shots", "Total shots"),
    "shots_target": ("shots_on_target", "ShotsOnTarget", "Shots on target"),
    "corners": ("corners", "Corners"),
    "xg": ("expected_goals", "Expected goals (xG)", "Expected goals"),
}

#: Columns of the fixtures CSV written by :meth:`Harvester.save_fixtures`.
FIXTURE_COLUMNS: Final[tuple[str, ...]] = ("match_id", "date", "round", "girone", "home_team", "away_team")

#: Columns of the odds CSV written by :func:`save_odds` (one row per quoted selection),
#: the format ``main.load_odds_dir`` and the dashboard read from ``data/odds/*.csv``.
ODDS_COLUMNS: Final[tuple[str, ...]] = ("match_id", "bookmaker", "market", "selection", "odds")

_NUMBER_RE = re.compile(r"-?\d+(?:[.,]\d+)?")
_SCORE_RE = re.compile(r"(\d+)\s*[-:–]\s*(\d+)")
_GIRONE_RE = re.compile(r"\b(?:girone|gruppo|group|grp)\s*[-:]?\s*([abc])\b", re.IGNORECASE)
_SERIE_C_RE = re.compile(r"\bserie\s*c\b", re.IGNORECASE)


# =============================================================================
# Configuration
# =============================================================================
def _format_template(template: str, **variables: Any) -> str:
    """``str.format`` with a clear error when a template variable is unknown."""
    try:
        return template.format(**variables)
    except (KeyError, IndexError) as exc:
        raise ValueError(f"endpoint template {template!r} references unknown variable {exc}") from exc


def _join_url(base_url: str, path: str) -> str:
    if path.startswith(("http://", "https://")):
        return path
    return f"{base_url.rstrip('/')}/{path.lstrip('/')}"


def site_origin(base_url: str) -> str:
    """Public site origin for ``base_url`` (``api.sofascore.com`` -> ``https://www.sofascore.com``)."""
    parts = urlsplit(base_url)
    host = (parts.hostname or "").lower()
    labels = [label for label in host.split(".") if label]
    if len(labels) < 2:
        return f"{parts.scheme or 'https'}://{host}" if host else base_url
    if len(labels) > 2 and labels[0] in {"api", "www", "m", "mobile"}:
        labels = labels[1:]
    return f"https://www.{'.'.join(labels)}"


@dataclass(frozen=True, slots=True)
class HttpConfig:
    """``[http]`` section of ``endpoints.toml``."""

    timeout_seconds: float = 20.0
    max_retries: int = 5
    backoff_base_seconds: float = 1.0
    backoff_max_seconds: float = 60.0
    max_concurrency: int = 4
    min_delay_seconds: float = 0.4
    cache_ttl_hours: float = 6.0
    history_cache_ttl_hours: float = 8760.0

    def __post_init__(self) -> None:
        if self.timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be > 0")
        if self.max_retries < 0:
            raise ValueError("max_retries must be >= 0")
        if self.backoff_base_seconds < 0 or self.backoff_max_seconds < 0:
            raise ValueError("backoff seconds must be >= 0")
        if self.max_concurrency < 1:
            raise ValueError("max_concurrency must be >= 1")
        if self.min_delay_seconds < 0:
            raise ValueError("min_delay_seconds must be >= 0")
        if self.cache_ttl_hours < 0:
            raise ValueError("cache_ttl_hours must be >= 0")
        if self.history_cache_ttl_hours < 0:
            raise ValueError("history_cache_ttl_hours must be >= 0")

    @property
    def cache_ttl(self) -> timedelta:
        """Time-to-live of volatile payloads (current season) as a :class:`timedelta`."""
        return timedelta(hours=float(self.cache_ttl_hours))

    @property
    def history_cache_ttl(self) -> timedelta:
        """Time-to-live of stable payloads (completed seasons) as a :class:`timedelta`.

        Rounds, events and statistics of a season that has ended never change, so
        they are kept far longer (365 days by default) than the current season's
        payloads: repeated multi-season runs re-download the current season only.
        """
        return timedelta(hours=float(self.history_cache_ttl_hours))


@dataclass(frozen=True)
class SofaScoreEndpoints:
    """``[sofascore]`` section: base URL, tournament ids and path templates."""

    base_url: str
    tournament_ids: Mapping[Girone, int]
    seasons: str
    rounds: str
    events_by_round: str
    events_last: str
    events_next: str
    event_statistics: str
    stat_keys: Mapping[str, tuple[str, ...]] = field(default_factory=dict)

    def tournament_id(self, girone: Girone | str) -> int:
        """Unique-tournament id of ``girone`` (``KeyError`` when not configured)."""
        g = Girone.parse(girone)
        try:
            return int(self.tournament_ids[g])
        except KeyError as exc:
            raise KeyError(f"no SofaScore tournament id configured for girone {g.value}") from exc

    def url_seasons(self, girone: Girone | str) -> str:
        return _join_url(self.base_url, _format_template(self.seasons, tournament_id=self.tournament_id(girone)))

    def url_rounds(self, girone: Girone | str, season_id: int) -> str:
        path = _format_template(self.rounds, tournament_id=self.tournament_id(girone), season_id=int(season_id))
        return _join_url(self.base_url, path)

    def url_events_by_round(self, girone: Girone | str, season_id: int, round: int) -> str:
        path = _format_template(
            self.events_by_round, tournament_id=self.tournament_id(girone), season_id=int(season_id), round=int(round)
        )
        return _join_url(self.base_url, path)

    def url_events_last(self, girone: Girone | str, season_id: int, page: int = 0) -> str:
        path = _format_template(
            self.events_last, tournament_id=self.tournament_id(girone), season_id=int(season_id), page=int(page)
        )
        return _join_url(self.base_url, path)

    def url_events_next(self, girone: Girone | str, season_id: int, page: int = 0) -> str:
        path = _format_template(
            self.events_next, tournament_id=self.tournament_id(girone), season_id=int(season_id), page=int(page)
        )
        return _join_url(self.base_url, path)

    def url_event_statistics(self, event_id: int | str) -> str:
        return _join_url(self.base_url, _format_template(self.event_statistics, event_id=event_id))


@dataclass(frozen=True)
class FotMobEndpoints:
    """``[fotmob]`` section. League ids equal to 0 are resolved at runtime."""

    base_url: str
    league_ids: Mapping[Girone, int]
    all_leagues: str
    league: str
    match_details: str

    def configured_league_ids(self) -> dict[Girone, int]:
        """Only the non-zero (explicitly configured) league ids."""
        return {g: int(v) for g, v in self.league_ids.items() if int(v) > 0}

    def url_all_leagues(self) -> str:
        return _join_url(self.base_url, self.all_leagues)

    def url_league(self, league_id: int) -> str:
        return _join_url(self.base_url, _format_template(self.league, league_id=int(league_id)))

    def url_match_details(self, match_id: int | str) -> str:
        return _join_url(self.base_url, _format_template(self.match_details, match_id=match_id))


@dataclass(frozen=True)
class FootyStatsEndpoints:
    """``[footystats]`` section: the official FootyStats API (``api.football-data-api.com``).

    The API key never lives in the configuration: ``key_env`` names the environment
    variable (also looked up in a ``.env`` file in the repository root) that holds
    it. ``league_names`` maps a competition code (``'A'``, ``'B'``, ``'C'``, ``'SB'``)
    to the ``name`` field of ``/league-list`` and is the fallback when the shared
    competitions module is not available; ``season_ids`` (code -> season label ->
    FootyStats ``season_id``) is used only when ``/league-list`` cannot be reached.
    The defaults are the public API as documented in November 2026, so a
    configuration file without the section still works.
    """

    base_url: str = "https://api.football-data-api.com"
    league_list: str = "/league-list"
    league_matches: str = "/league-matches"
    league_teams: str = "/league-teams"
    key_env: str = "FOOTYSTATS_API_KEY"
    max_requests_per_hour: int = 1800
    league_names: Mapping[str, str] = field(default_factory=dict)
    season_ids: Mapping[str, Mapping[str, int]] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if int(self.max_requests_per_hour) < 1:
            raise ValueError("max_requests_per_hour must be >= 1")

    def url_league_list(self) -> str:
        return _join_url(self.base_url, self.league_list)

    def url_league_matches(self) -> str:
        return _join_url(self.base_url, self.league_matches)

    def url_league_teams(self) -> str:
        return _join_url(self.base_url, self.league_teams)

    def known_season_ids(self, code: str) -> dict[str, int]:
        """Configured fallback season ids of ``code`` (label ``'2026/27'`` -> id), ``{}`` when none."""
        return {str(label): int(season_id) for label, season_id in (self.season_ids.get(str(code).upper()) or {}).items()}


@dataclass(frozen=True)
class EndpointsConfig:
    """Whole ``endpoints.toml``."""

    sofascore: SofaScoreEndpoints
    fotmob: FotMobEndpoints
    http: HttpConfig
    footystats: FootyStatsEndpoints = field(default_factory=FootyStatsEndpoints)


def _require(section: Mapping[str, Any], key: str, section_name: str, path: Path) -> Any:
    try:
        return section[key]
    except KeyError as exc:
        raise ValueError(f"{path}: [{section_name}] is missing required key {key!r}") from exc


def _parse_stat_keys(raw: Any, path: Path) -> dict[str, tuple[str, ...]]:
    """Validate ``[sofascore.stat_keys]``: every metric must map to a non-empty list of
    non-empty strings.

    A scalar such as ``shots = "totalShotsOnGoal"`` is rejected instead of being
    silently iterated character by character (which would make every statistic
    ``None`` while the run reports success). Metrics the parser does not know
    are kept but logged, since they can never be located.
    """
    if raw is None:
        return {}
    if not isinstance(raw, Mapping):
        raise ValueError(f"{path}: [sofascore.stat_keys] must be a table of metric -> list of keys, got {type(raw).__name__}")
    out: dict[str, tuple[str, ...]] = {}
    for metric, value in raw.items():
        valid = (
            isinstance(value, (list, tuple))
            and len(value) > 0
            and all(isinstance(item, str) and item.strip() for item in value)
        )
        if not valid:
            raise ValueError(
                f"{path}: [sofascore.stat_keys] {metric!r} must be a non-empty list of non-empty strings, got {value!r}"
            )
        if str(metric) not in _METRIC_FIELDS:
            log.warning(
                "unknown stat_keys metric is never used",
                extra={"path": str(path), "metric": str(metric), "known": sorted(_METRIC_FIELDS)},
            )
        out[str(metric)] = tuple(str(item).strip() for item in value)
    return out


def load_endpoints(path: Path = CONFIG_DIR / "endpoints.toml") -> EndpointsConfig:
    """Parse ``endpoints.toml`` (stdlib :mod:`tomllib`) into :class:`EndpointsConfig`.

    Raises ``ValueError`` on missing keys and on malformed ``stat_keys`` entries.
    """
    path = Path(path)
    with path.open("rb") as handle:
        raw = tomllib.load(handle)
    sofa_raw = _require(raw, "sofascore", "root", path)
    fot_raw = _require(raw, "fotmob", "root", path)
    http_raw: Mapping[str, Any] = raw.get("http", {})

    sofascore = SofaScoreEndpoints(
        base_url=str(_require(sofa_raw, "base_url", "sofascore", path)).rstrip("/"),
        tournament_ids={
            Girone.parse(k): int(v) for k, v in _require(sofa_raw, "tournament_ids", "sofascore", path).items()
        },
        seasons=str(_require(sofa_raw, "seasons", "sofascore", path)),
        rounds=str(_require(sofa_raw, "rounds", "sofascore", path)),
        events_by_round=str(_require(sofa_raw, "events_by_round", "sofascore", path)),
        events_last=str(_require(sofa_raw, "events_last", "sofascore", path)),
        events_next=str(_require(sofa_raw, "events_next", "sofascore", path)),
        event_statistics=str(_require(sofa_raw, "event_statistics", "sofascore", path)),
        stat_keys=_parse_stat_keys(sofa_raw.get("stat_keys"), path),
    )
    fotmob = FotMobEndpoints(
        base_url=str(_require(fot_raw, "base_url", "fotmob", path)).rstrip("/"),
        league_ids={Girone.parse(k): int(v) for k, v in fot_raw.get("league_ids", {}).items()},
        all_leagues=str(_require(fot_raw, "all_leagues", "fotmob", path)),
        league=str(_require(fot_raw, "league", "fotmob", path)),
        match_details=str(_require(fot_raw, "match_details", "fotmob", path)),
    )
    defaults = HttpConfig()
    http = HttpConfig(
        timeout_seconds=float(http_raw.get("timeout_seconds", defaults.timeout_seconds)),
        max_retries=int(http_raw.get("max_retries", defaults.max_retries)),
        backoff_base_seconds=float(http_raw.get("backoff_base_seconds", defaults.backoff_base_seconds)),
        backoff_max_seconds=float(http_raw.get("backoff_max_seconds", defaults.backoff_max_seconds)),
        max_concurrency=int(http_raw.get("max_concurrency", defaults.max_concurrency)),
        min_delay_seconds=float(http_raw.get("min_delay_seconds", defaults.min_delay_seconds)),
        cache_ttl_hours=float(http_raw.get("cache_ttl_hours", defaults.cache_ttl_hours)),
        history_cache_ttl_hours=float(http_raw.get("history_cache_ttl_hours", defaults.history_cache_ttl_hours)),
    )
    footystats = _parse_footystats_section(raw.get("footystats"), path)
    log.debug("endpoints loaded", extra={"path": str(path)})
    return EndpointsConfig(sofascore=sofascore, fotmob=fotmob, http=http, footystats=footystats)


def _parse_footystats_section(raw: Any, path: Path) -> FootyStatsEndpoints:
    """``[footystats]`` -> :class:`FootyStatsEndpoints`; a missing section yields the defaults."""
    if raw is None:
        return FootyStatsEndpoints()
    if not isinstance(raw, Mapping):
        raise ValueError(f"{path}: [footystats] must be a table, got {type(raw).__name__}")
    defaults = FootyStatsEndpoints()
    names_raw = raw.get("league_names") or {}
    if not isinstance(names_raw, Mapping) or not all(isinstance(v, str) and v.strip() for v in names_raw.values()):
        raise ValueError(f"{path}: [footystats.league_names] must map competition codes to non-empty strings")
    ids_raw = raw.get("season_ids") or {}
    if not isinstance(ids_raw, Mapping):
        raise ValueError(f"{path}: [footystats.season_ids] must be a table of code -> {{season label = id}}")
    season_ids: dict[str, dict[str, int]] = {}
    for code, labels in ids_raw.items():
        if not isinstance(labels, Mapping):
            raise ValueError(f"{path}: [footystats.season_ids] {code!r} must be a table of season label -> id")
        season_ids[str(code).upper()] = {str(label): int(season_id) for label, season_id in labels.items()}
    return FootyStatsEndpoints(
        base_url=str(raw.get("base_url", defaults.base_url)).rstrip("/"),
        league_list=str(raw.get("league_list", defaults.league_list)),
        league_matches=str(raw.get("league_matches", defaults.league_matches)),
        league_teams=str(raw.get("league_teams", defaults.league_teams)),
        key_env=str(raw.get("key_env", defaults.key_env)),
        max_requests_per_hour=int(raw.get("max_requests_per_hour", defaults.max_requests_per_hour)),
        league_names={str(code).upper(): str(name).strip() for code, name in names_raw.items()},
        season_ids=season_ids,
    )


# =============================================================================
# Raw cache
# =============================================================================
class RawCache:
    """JSON file cache of raw responses.

    One file per URL (``<sha1>.json``) holding ``{"url", "fetched_at", "payload"}``.
    Writes are atomic (temp file + :func:`os.replace`). ``ttl=None`` never
    expires; ``ttl=timedelta(0)`` expires immediately (cache effectively
    write-only, useful for debugging).

    Two time-to-lives apply: ``ttl`` to *volatile* payloads (the default, e.g.
    the running season, whose results and statistics still change) and
    ``stable_ttl`` to payloads the caller declares *stable* with
    ``get(..., stable=True)`` (e.g. a completed season, whose rounds, events and
    statistics are immutable). Both use the same ``None`` / ``timedelta(0)``
    semantics; ``stable_ttl`` defaults to never expiring. The file format is the
    same for both kinds, so a payload fetched as volatile is served to a stable
    read and vice versa; only the expiry rule differs.
    """

    def __init__(self, directory: Path, ttl: timedelta | None = None, *, stable_ttl: timedelta | None = None) -> None:
        self.directory = Path(directory)
        self.ttl = ttl
        self.stable_ttl = stable_ttl
        self.hits = 0
        self.misses = 0
        self.directory.mkdir(parents=True, exist_ok=True)

    # ---------------------------------------------------------------- keys
    @staticmethod
    def canonical_url(url: str, params: Mapping[str, Any] | None = None) -> str:
        """Full URL with every query parameter (URL + ``params``) sorted."""
        parts = urlsplit(url)
        pairs: list[tuple[str, str]] = list(parse_qsl(parts.query, keep_blank_values=True))
        for key, value in (params or {}).items():
            if isinstance(value, (list, tuple, set, frozenset)):
                pairs.extend((str(key), str(v)) for v in value)
            elif value is not None:
                pairs.append((str(key), str(value)))
        pairs.sort()
        return urlunsplit((parts.scheme, parts.netloc, parts.path, urlencode(pairs), ""))

    @classmethod
    def key_for(cls, url: str, params: Mapping[str, Any] | None = None) -> str:
        """SHA-1 of the canonical URL."""
        return hashlib.sha1(cls.canonical_url(url, params).encode("utf-8")).hexdigest()

    def path_for(self, url: str, params: Mapping[str, Any] | None = None) -> Path:
        """File that holds (or would hold) the cached response for ``url``."""
        return self.directory / f"{self.key_for(url, params)}.json"

    # ------------------------------------------------------------- access
    def ttl_for(self, *, stable: bool = False) -> timedelta | None:
        """Time-to-live that applies to a read: ``stable_ttl`` for stable payloads, else ``ttl``."""
        return self.stable_ttl if stable else self.ttl

    def _is_expired(self, fetched_at: str, *, stable: bool = False) -> bool:
        ttl = self.ttl_for(stable=stable)
        if ttl is None:
            return False
        stamp = datetime.fromisoformat(str(fetched_at))
        if stamp.tzinfo is None:
            stamp = stamp.replace(tzinfo=timezone.utc)
        return datetime.now(timezone.utc) >= stamp + ttl

    def get(self, url: str, params: Mapping[str, Any] | None = None, *, stable: bool = False) -> Any | None:
        """Cached payload or ``None`` on miss, expiry or unreadable file.

        ``stable`` says the payload cannot change upstream any more (a completed
        season): the entry then expires after ``stable_ttl`` instead of ``ttl``.
        """
        path = self.path_for(url, params)
        if not path.exists():
            self.misses += 1
            return None
        try:
            record = json.loads(path.read_text(encoding="utf-8"))
            if not isinstance(record, dict) or "payload" not in record:
                raise ValueError("cache record without payload")
            expired = self._is_expired(record.get("fetched_at", ""), stable=stable)
        except (OSError, ValueError, TypeError) as exc:
            log.warning("corrupt cache file treated as miss", extra={"path": str(path), "error": str(exc)})
            self.misses += 1
            return None
        if expired:
            log.debug("cache entry expired", extra={"path": str(path), "url": url, "stable": stable})
            self.misses += 1
            return None
        self.hits += 1
        return record["payload"]

    def put(self, url: str, payload: Any, params: Mapping[str, Any] | None = None) -> Path:
        """Atomically store ``payload`` for ``url``; returns the file path."""
        path = self.path_for(url, params)
        record = {
            "url": self.canonical_url(url, params),
            "fetched_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "payload": payload,
        }
        self.directory.mkdir(parents=True, exist_ok=True)
        handle = tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=self.directory, suffix=".tmp", delete=False)
        try:
            with handle:
                json.dump(record, handle, ensure_ascii=False, default=str)
            os.replace(handle.name, path)
        except BaseException:
            Path(handle.name).unlink(missing_ok=True)
            raise
        return path

    def clear(self) -> int:
        """Delete every cached file; returns how many were removed."""
        removed = 0
        for file in self.directory.glob("*.json"):
            file.unlink(missing_ok=True)
            removed += 1
        for leftover in self.directory.glob("*.tmp"):
            leftover.unlink(missing_ok=True)
        return removed

    @property
    def stats(self) -> dict[str, int]:
        return {"hits": self.hits, "misses": self.misses}

    def __len__(self) -> int:
        return sum(1 for _ in self.directory.glob("*.json"))


# =============================================================================
# Errors
# =============================================================================
class HarvestError(Exception):
    """A request failed definitively (after retries when retryable)."""

    def __init__(self, message: str, *, url: str, status: int | None = None, attempts: int = 0) -> None:
        super().__init__(message)
        self.url = url
        self.status = status
        self.attempts = attempts

    def __str__(self) -> str:
        return f"{super().__str__()} [url={self.url} status={self.status} attempts={self.attempts}]"


class NotFoundError(HarvestError):
    """HTTP 404: the resource does not exist (never retried)."""


class RateLimitedError(HarvestError):
    """HTTP 429 persisted through every retry."""


def parse_retry_after(value: str | None, *, now: datetime | None = None) -> float | None:
    """Seconds to wait from a ``Retry-After`` header (delta-seconds or HTTP date)."""
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    if re.fullmatch(r"\d+(?:\.\d+)?", text):
        return max(0.0, float(text))
    try:
        when = parsedate_to_datetime(text)
    except (TypeError, ValueError, IndexError):
        return None
    if when is None:
        return None
    if when.tzinfo is None:
        when = when.replace(tzinfo=timezone.utc)
    reference = now or datetime.now(timezone.utc)
    return max(0.0, (when - reference).total_seconds())


# =============================================================================
# HTTP client
# =============================================================================
@dataclass
class ClientStats:
    """Counters kept by :class:`AsyncHarvesterClient`."""

    requests: int = 0
    retries: int = 0
    cache_hits: int = 0
    errors: int = 0

    def as_dict(self) -> dict[str, int]:
        return {"requests": self.requests, "retries": self.retries, "cache_hits": self.cache_hits, "errors": self.errors}

    def add(self, other: "ClientStats") -> None:
        self.requests += other.requests
        self.retries += other.retries
        self.cache_hits += other.cache_hits
        self.errors += other.errors


Sleeper = Callable[[float], Awaitable[None]]


class AsyncHarvesterClient:
    """Polite async JSON client with cache, UA rotation, pacing and retries.

    Parameters
    ----------
    base_url:
        Origin of the API (``https://api.sofascore.com``). ``get_json`` accepts
        paths relative to it or absolute URLs.
    http:
        Timeouts, retry and pacing policy.
    cache:
        Optional :class:`RawCache`; hits short-circuit the network.
    transport:
        Optional ``httpx`` transport (tests inject :class:`httpx.MockTransport`).
    user_agents:
        Pool to rotate; defaults to :data:`DEFAULT_USER_AGENTS`.
    sleeper:
        Awaitable used for every wait (``asyncio.sleep`` in production).
    rng:
        Source of randomness for UA choice and backoff jitter.
    extra_headers:
        Merged last into every request (e.g. FotMob's ``x-mas`` signature).
    clock:
        Monotonic clock used for request pacing (``time.monotonic`` in
        production; tests inject a fixed clock).
    """

    def __init__(
        self,
        *,
        base_url: str,
        http: HttpConfig,
        cache: RawCache | None = None,
        transport: httpx.AsyncBaseTransport | None = None,
        user_agents: Sequence[str] | None = None,
        sleeper: Sleeper = asyncio.sleep,
        rng: random.Random | None = None,
        extra_headers: Mapping[str, str] | None = None,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        pool = tuple(user_agents) if user_agents is not None else DEFAULT_USER_AGENTS
        if not pool:
            raise ValueError("user_agents must not be empty")
        self.base_url = base_url.rstrip("/")
        self.http = http
        self.cache = cache
        self.site = site_origin(self.base_url)
        self.stats = ClientStats()
        self.last_user_agent: str | None = None
        self._user_agents = pool
        self._transport = transport
        self._sleeper = sleeper
        self._rng = rng if rng is not None else random.Random()
        self._extra_headers = dict(extra_headers or {})
        self._clock = clock
        self._client: httpx.AsyncClient | None = None
        self._semaphore = asyncio.Semaphore(http.max_concurrency)
        self._pace_lock = asyncio.Lock()
        self._last_started: float | None = None

    # ------------------------------------------------------------ lifecycle
    async def __aenter__(self) -> "AsyncHarvesterClient":
        self._ensure_client()
        return self

    async def __aexit__(self, *exc_info: object) -> None:
        await self.close()

    async def close(self) -> None:
        """Close the underlying ``httpx.AsyncClient`` (idempotent)."""
        if self._client is not None:
            client, self._client = self._client, None
            await client.aclose()

    def _ensure_client(self) -> httpx.AsyncClient:
        if self._client is None:
            self._client = httpx.AsyncClient(
                transport=self._transport,
                timeout=httpx.Timeout(self.http.timeout_seconds),
                follow_redirects=True,
            )
        return self._client

    # -------------------------------------------------------------- helpers
    def build_url(self, path: str) -> str:
        """Absolute URL for ``path`` (returned unchanged when already absolute)."""
        return _join_url(self.base_url, path)

    def _pick_user_agent(self) -> str:
        choices = [ua for ua in self._user_agents if ua != self.last_user_agent] or list(self._user_agents)
        agent = self._rng.choice(choices)
        self.last_user_agent = agent
        return agent

    def headers(self) -> dict[str, str]:
        """Fresh request headers (new user agent, realistic browser-like set)."""
        base = {
            "User-Agent": self._pick_user_agent(),
            "Accept": "application/json",
            "Accept-Language": "it-IT,it;q=0.9,en;q=0.8",
            "Referer": f"{self.site}/",
            "Origin": self.site,
            "Cache-Control": "no-cache",
            "Pragma": "no-cache",
            "Connection": "keep-alive",
        }
        base.update(self._extra_headers)
        return base

    def backoff_delay(self, attempt: int, retry_after: float | None = None) -> float:
        """Exponential backoff with full jitter, floored by ``Retry-After``."""
        cap = min(self.http.backoff_max_seconds, self.http.backoff_base_seconds * (2**attempt))
        delay = cap * self._rng.uniform(0.5, 1.0)
        if retry_after is not None:
            delay = max(delay, retry_after)
        return delay

    async def _respect_min_delay(self) -> None:
        """Keep at least ``min_delay_seconds`` between the start of two requests."""
        min_delay = self.http.min_delay_seconds
        async with self._pace_lock:
            now = self._clock()
            if self._last_started is not None and min_delay > 0:
                remaining = self._last_started + min_delay - now
                if remaining > 0:
                    await self._sleeper(remaining)
                    now = max(self._clock(), self._last_started + min_delay)
            self._last_started = now

    # ----------------------------------------------------------------- core
    async def get_json(
        self, path: str, params: Mapping[str, Any] | None = None, *, use_cache: bool = True, stable: bool = False
    ) -> Any:
        """GET ``path`` and return the decoded JSON body.

        Cache hits return immediately; ``stable`` marks a payload that cannot
        change upstream any more (a completed season), so the cache applies its
        long ``stable_ttl`` to it (see :class:`RawCache`). Retryable failures (HTTP 429, 5xx,
        transport errors, timeouts) are retried up to ``http.max_retries``
        times; other 4xx raise at once (:class:`NotFoundError` for 404), as do
        undecodable bodies and other non-transient ``httpx`` errors. Every
        failure surfaces as a :class:`HarvestError` subclass. (``Accept-Encoding``
        is left to ``httpx`` so only decodable encodings are advertised.)
        """
        url = self.build_url(path)
        if use_cache and self.cache is not None:
            try:
                cached = self.cache.get(url, params, stable=stable)
            except OSError as exc:  # unreadable cache directory: fetch instead
                cached = None
                log.warning("cache read failed, fetching", extra={"url": url, "error": str(exc)})
            if cached is not None:
                self.stats.cache_hits += 1
                log.debug("cache hit", extra={"url": url, "stable": stable})
                return cached
        client = self._ensure_client()
        async with self._semaphore:
            payload = await self._fetch_with_retry(client, url, params)
        if use_cache and self.cache is not None:
            # A full disk, a read-only volume or a cache directory replaced by a
            # file must not throw away a payload that was downloaded successfully
            # (and must not escape as a non-HarvestError that aborts the harvest).
            try:
                self.cache.put(url, payload, params)
            except OSError as exc:
                log.warning("cache write failed, payload kept", extra={"url": url, "error": str(exc)})
        return payload

    async def _fetch_with_retry(self, client: httpx.AsyncClient, url: str, params: Mapping[str, Any] | None) -> Any:
        max_attempts = self.http.max_retries + 1
        last_status: int | None = None
        last_error = "unknown"
        attempts = 0
        for attempt in range(max_attempts):
            attempts = attempt + 1
            retry_after: float | None = None
            await self._respect_min_delay()
            self.stats.requests += 1
            try:
                response = await client.get(url, params=dict(params) if params else None, headers=self.headers())
            except (httpx.TimeoutException, httpx.TransportError) as exc:
                last_status = None
                last_error = f"{type(exc).__name__}: {exc}"
            except httpx.HTTPError as exc:  # decoding / redirect problems: not transient
                self.stats.errors += 1
                raise HarvestError(f"{type(exc).__name__}: {exc}", url=url, status=None, attempts=attempts) from exc
            else:
                status = response.status_code
                last_status = status
                if 200 <= status < 300:
                    try:
                        return response.json()
                    except ValueError as exc:
                        self.stats.errors += 1
                        raise HarvestError("invalid JSON body", url=url, status=status, attempts=attempts) from exc
                if status == 404:
                    self.stats.errors += 1
                    raise NotFoundError("resource not found", url=url, status=status, attempts=attempts)
                if status != 429 and status < 500:
                    self.stats.errors += 1
                    raise HarvestError(f"HTTP {status}", url=url, status=status, attempts=attempts)
                last_error = f"HTTP {status}"
                retry_after = parse_retry_after(response.headers.get("Retry-After"))
            if attempt >= self.http.max_retries:
                break
            delay = self.backoff_delay(attempt, retry_after)
            self.stats.retries += 1
            log.warning(
                "retrying request",
                extra={
                    "url": url,
                    "status": last_status,
                    "attempt": attempts,
                    "max_attempts": max_attempts,
                    "delay": round(delay, 3),
                    "error": last_error,
                },
            )
            await self._sleeper(delay)
        self.stats.errors += 1
        message = f"giving up after {attempts} attempts: {last_error}"
        if last_status == 429:
            raise RateLimitedError(message, url=url, status=last_status, attempts=attempts)
        raise HarvestError(message, url=url, status=last_status, attempts=attempts)


# =============================================================================
# Concurrency helper
# =============================================================================
def _first_leaf(group: BaseExceptionGroup) -> BaseException:
    """First non-group exception inside ``group`` (depth first), preferring a
    :class:`HarvestError` so callers keep their ``except HarvestError`` contract."""
    leaves: list[BaseException] = []
    stack: list[BaseException] = [group]
    while stack:
        current = stack.pop(0)
        if isinstance(current, BaseExceptionGroup):
            stack[:0] = list(current.exceptions)
        else:
            leaves.append(current)
    for leaf in leaves:
        if isinstance(leaf, HarvestError):
            return leaf
    return leaves[0] if leaves else group


#: Result type of the coroutines awaited by :func:`run_concurrently`. Declared
#: with :class:`typing.TypeVar` rather than PEP 695 ``def f[T]`` syntax so the
#: module keeps parsing on Python 3.11, the floor advertised in ``pyproject.toml``.
T = TypeVar("T")


async def run_concurrently(coroutines: Iterable[Coroutine[Any, Any, T]]) -> list[T]:
    """Await ``coroutines`` concurrently and return their results in order.

    Unlike a bare :func:`asyncio.gather`, the first failure cancels every
    sibling task before the exception propagates, so a round that keeps
    answering 5xx does not leave orphan requests hitting the API (or a closed
    client) after the caller has already given up. The exception raised is the
    first child failure itself (a :class:`HarvestError` stays a ``HarvestError``),
    not an ``ExceptionGroup``.
    """
    pending = list(coroutines)
    if not pending:
        return []
    tasks: list[asyncio.Task[T]] = []
    try:
        async with asyncio.TaskGroup() as group:
            tasks = [group.create_task(coro) for coro in pending]
    except BaseExceptionGroup as group_exc:
        raise _first_leaf(group_exc) from group_exc
    return [task.result() for task in tasks]


# =============================================================================
# Value parsing helpers (pure)
# =============================================================================
def parse_stat_value(value: Any) -> float | None:
    """Robust number extraction: ``12`` / ``"1.23"`` / ``"55%"`` / ``"7 (3)"`` -> ``7.0``."""
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        out = float(value)
        return None if math.isnan(out) or math.isinf(out) else out
    text = str(value).strip()
    if not text:
        return None
    found = _NUMBER_RE.search(text)
    if found is None:
        return None
    try:
        return float(found.group(0).replace(",", "."))
    except ValueError:  # pragma: no cover - regex guarantees a number
        return None


def _as_int(value: Any, default: int = 0) -> int:
    if value is None or isinstance(value, bool):
        return default
    if isinstance(value, (int, float)):
        return default if isinstance(value, float) and math.isnan(value) else int(value)
    found = re.search(r"\d+", str(value))
    return int(found.group(0)) if found else default


def _utc_date_from_epoch(value: Any) -> date | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        return datetime.fromtimestamp(float(value), tz=timezone.utc).date()
    except (TypeError, ValueError, OverflowError, OSError):
        return None


def _utc_date_from_iso(value: Any) -> date | None:
    if not value:
        return None
    text = str(value).strip()
    try:
        when = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError:
        return None
    if when.tzinfo is None:
        when = when.replace(tzinfo=timezone.utc)
    return when.astimezone(timezone.utc).date()


# =============================================================================
# Season helpers (pure)
# =============================================================================
_SEASON_YEAR_RE = re.compile(r"^\s*(\d{2}|\d{4})\s*[/_\-]\s*(\d{2}|\d{4})\s*$")


def season_label_from_year(year: str | None) -> str | None:
    """Normalise a season spelling to the public label ``'2026/27'``.

    Accepts SofaScore's ``year`` field (``'26/27'``), registry stems
    (``'2026_27'``), ``'2026-27'`` and ``'2026/2027'``. Returns ``None`` when the
    text is not a season or its second year does not follow the first, so callers
    can fall back to :func:`src.common.schemas.season_of` on match dates.
    """
    if year is None:
        return None
    found = _SEASON_YEAR_RE.match(str(year))
    if found is None:
        return None
    first, second = found.group(1), found.group(2)
    start = int(first) if len(first) == 4 else (1900 if int(first) >= 70 else 2000) + int(first)
    end = int(second) if len(second) == 4 else (start // 100) * 100 + int(second)
    if len(second) == 2 and end < start:  # '1999/00'
        end += 100
    if end != start + 1:
        return None
    return f"{start}/{end % 100:02d}"


def _season_start_year(label: str) -> int | None:
    normalised = season_label_from_year(label)
    return int(normalised[:4]) if normalised else None


def _sort_season_labels(labels: Iterable[str]) -> list[str]:
    """Distinct labels, newest first; labels that are not seasons go last, in first-seen order."""
    distinct = list(dict.fromkeys(str(label) for label in labels if label))
    seasons = [label for label in distinct if _season_start_year(label) is not None]
    others = [label for label in distinct if _season_start_year(label) is None]
    seasons.sort(key=lambda label: -(_season_start_year(label) or 0))
    return [*seasons, *others]


def _season_counts(records: Iterable[MatchRecord]) -> Counter[str]:
    """Matches per season label (:func:`season_of` of the match date)."""
    return Counter(season_of(record.date) for record in records)


def _events_season_label(batches: Iterable[Iterable[dict]]) -> str | None:
    """Season label most of the raw events belong to (by ``startTimestamp``), ``None`` without dates."""
    counts: Counter[str] = Counter()
    for events in batches:
        for event in events:
            when = _utc_date_from_epoch(event.get("startTimestamp")) if isinstance(event, dict) else None
            if when is not None:
                counts[season_of(when)] += 1
    return counts.most_common(1)[0][0] if counts else None


def _season_is_completed(label: str | None, *, current: bool, today: date | None = None) -> bool:
    """Whether a listed season has ended, so its payloads are immutable (cached as *stable*).

    The SofaScore listing is newest first: a season that is not the first entry
    (``current=False``) is older than the newest one. It counts as completed when
    its ``label`` precedes the running season (:func:`season_of` of ``today``,
    July-June), or when no label is known at all. The first listed season, and a
    season pinned by id (always ``current``), are never completed: they may still
    be in progress, so their rounds, events and statistics keep the short TTL.
    """
    if current:
        return False
    start = _season_start_year(label) if label else None
    if start is None:
        return True
    running = _season_start_year(season_of(today if today is not None else date.today()))
    return running is not None and start < running


def _canonical_team(matcher: TeamMatcher, raw_name: str, girone: Girone | None, source: str) -> str:
    """Canonical name via ``matcher``; the raw spelling (with a warning) when unmapped.

    ``girone=None`` performs an unrestricted lookup (historical seasons).
    """
    result = matcher.match(raw_name, girone)
    if result.canonical is None:
        log.warning(
            "unmapped team name kept raw",
            extra={
                "source": source,
                "team": raw_name,
                "girone": girone.value if girone is not None else None,
                "candidates": list(result.candidates[:3]),
            },
        )
        return raw_name.strip()
    return result.canonical


def _team_name(team: Any) -> str | None:
    if not isinstance(team, dict):
        return None
    for key in ("name", "shortName", "longName", "slug"):
        value = team.get(key)
        if value:
            return str(value)
    return None


def _empty_stats() -> dict[str, float | None]:
    return {name: None for name in STAT_FIELDS}


def _warn_missing_metrics(
    records: Sequence[MatchRecord], stat_keys: Mapping[str, Sequence[str]], source: str
) -> list[str]:
    """Metrics that none of ``records`` (matches whose statistics payload *was*
    fetched) provides; logged as a warning because it usually means the
    upstream keys drifted or ``stat_keys`` is misconfigured."""
    if not records:
        return []
    missing = [
        metric
        for metric, (home_field, away_field) in _METRIC_FIELDS.items()
        if all(getattr(r, home_field) is None and getattr(r, away_field) is None for r in records)
    ]
    if missing:
        log.warning(
            "statistics metrics never located",
            extra={
                "source": source,
                "metrics": missing,
                "payloads": len(records),
                "keys": {metric: list(stat_keys.get(metric, ())) for metric in missing},
            },
        )
    return missing


# =============================================================================
# SofaScore
# =============================================================================
@dataclass(frozen=True, slots=True)
class SeasonInfo:
    """One entry of the SofaScore ``seasons`` payload."""

    id: int
    year: str
    name: str


@dataclass(frozen=True)
class SeasonHarvest:
    """Outcome of :meth:`SofaScoreProvider.harvest_seasons` for one season of one girone.

    ``label`` is the public season label (``'2026/27'``); ``historical`` says the
    season lies outside the perimeter of the team dictionary (its names were
    resolved without girone restriction); ``current`` marks the newest season of
    the listing, the only one whose not-started events are kept as ``fixtures``.
    ``completed`` says the season has ended (see :func:`_season_is_completed`):
    its payloads were read and written through the cache as *stable*, with the
    long ``history_cache_ttl`` instead of the short ``cache_ttl``.
    A past season that could not be harvested has ``error`` set and no matches.
    """

    season: SeasonInfo
    label: str
    matches: tuple[MatchRecord, ...]
    fixtures: tuple[Fixture, ...]
    rounds: int
    current: bool
    historical: bool
    error: str | None = None
    completed: bool = False


class SofaScoreProvider:
    """Navigate the SofaScore unique-tournament API for one girone."""

    name: Final[str] = "sofascore"

    def __init__(self, client: AsyncHarvesterClient, endpoints: SofaScoreEndpoints, matcher: TeamMatcher) -> None:
        self.client = client
        self.endpoints = endpoints
        self.matcher = matcher

    # ------------------------------------------------------------ endpoints
    async def seasons(self, girone: Girone | str) -> list[SeasonInfo]:
        """Seasons of the girone, most recent first (as returned by the API)."""
        payload = await self.client.get_json(self.endpoints.url_seasons(girone))
        out: list[SeasonInfo] = []
        for item in (payload or {}).get("seasons", []) or []:
            if not isinstance(item, dict) or item.get("id") is None:
                continue
            out.append(SeasonInfo(id=int(item["id"]), year=str(item.get("year", "")), name=str(item.get("name", ""))))
        return out

    async def current_season_id(self, girone: Girone | str) -> int:
        """Id of the first (current) season."""
        seasons = await self.seasons(girone)
        if not seasons:
            raise HarvestError("no seasons returned", url=self.endpoints.url_seasons(girone))
        return seasons[0].id

    async def rounds(
        self, girone: Girone | str, season_id: int, *, regular_only: bool = True, stable: bool = False
    ) -> list[int]:
        """Sorted distinct round numbers of the season.

        With ``regular_only`` entries carrying a ``name``/``prefix`` (play-off
        and play-out rounds) are skipped. When the API lists no rounds but a
        ``currentRound`` exists, rounds ``1..current`` are assumed. ``stable``
        says the season has ended: the payload is cached with the long TTL.
        """
        payload = await self.client.get_json(self.endpoints.url_rounds(girone, season_id), stable=stable) or {}
        numbers: set[int] = set()
        for item in payload.get("rounds", []) or []:
            if not isinstance(item, dict) or item.get("round") is None:
                continue
            if regular_only and (item.get("name") or item.get("prefix")):
                continue
            numbers.add(int(item["round"]))
        if not numbers:
            current = (payload.get("currentRound") or {}).get("round")
            if current:
                numbers = set(range(1, int(current) + 1))
        return sorted(numbers)

    async def events_for_round(self, girone: Girone | str, season_id: int, round: int, *, stable: bool = False) -> list[dict]:
        """Raw events of one round; ``stable`` (completed season) selects the long cache TTL."""
        payload = await self.client.get_json(self.endpoints.url_events_by_round(girone, season_id, round), stable=stable) or {}
        return [e for e in payload.get("events", []) or [] if isinstance(e, dict)]

    async def next_events(self, girone: Girone | str, season_id: int, page: int = 0) -> list[dict]:
        payload = await self.client.get_json(self.endpoints.url_events_next(girone, season_id, page)) or {}
        return [e for e in payload.get("events", []) or [] if isinstance(e, dict)]

    async def last_events(self, girone: Girone | str, season_id: int, page: int = 0) -> list[dict]:
        payload = await self.client.get_json(self.endpoints.url_events_last(girone, season_id, page)) or {}
        return [e for e in payload.get("events", []) or [] if isinstance(e, dict)]

    async def statistics(self, event_id: int | str, *, stable: bool = False) -> dict:
        """Raw statistics payload of one event; ``stable`` (completed season) selects the long cache TTL."""
        payload = await self.client.get_json(self.endpoints.url_event_statistics(event_id), stable=stable)
        return payload if isinstance(payload, dict) else {}

    # -------------------------------------------------------------- parsers
    @staticmethod
    def parse_event(
        event: dict, girone: Girone, matcher: TeamMatcher, *, historical: bool = False
    ) -> MatchRecord | Fixture | None:
        """Convert one SofaScore event to a :class:`MatchRecord` (finished),
        a :class:`Fixture` (not started) or ``None`` (anything else).

        ``girone`` is the tournament the event was listed under and is stored on
        the record. Team names are resolved within that girone's *current*
        composition, unless ``historical`` is set (events of a past season): a
        club may since have moved girone or left Serie C, so the lookup is then
        unrestricted and may yield a historical canonical name.
        """
        girone = Girone.parse(girone)
        lookup: Girone | None = None if historical else girone
        event_id = event.get("id")
        if event_id is None:
            return None
        match_id = f"SS-{event_id}"
        status = str(((event.get("status") or {}).get("type")) or "").lower()
        when = _utc_date_from_epoch(event.get("startTimestamp"))
        if when is None:
            log.warning("event without start timestamp skipped", extra={"match_id": match_id})
            return None
        round_no = _as_int((event.get("roundInfo") or {}).get("round"), 0)
        home_raw = _team_name(event.get("homeTeam"))
        away_raw = _team_name(event.get("awayTeam"))
        if not home_raw or not away_raw:
            log.warning("event without team names skipped", extra={"match_id": match_id})
            return None
        home = _canonical_team(matcher, home_raw, lookup, "sofascore")
        away = _canonical_team(matcher, away_raw, lookup, "sofascore")
        if home == away:
            log.warning("event with identical teams skipped", extra={"match_id": match_id, "team": home})
            return None
        if status == "notstarted":
            return Fixture(match_id=match_id, date=when, round=round_no, girone=girone, home_team=home, away_team=away)
        if status != "finished":
            return None
        home_score = event.get("homeScore") or {}
        away_score = event.get("awayScore") or {}
        hg = home_score.get("current", home_score.get("display"))
        ag = away_score.get("current", away_score.get("display"))
        if hg is None or ag is None:
            log.warning("finished event without scores skipped", extra={"match_id": match_id})
            return None
        return MatchRecord(
            match_id=match_id,
            date=when,
            round=round_no,
            girone=girone,
            home_team=home,
            away_team=away,
            home_goals=int(hg),
            away_goals=int(ag),
        )

    @staticmethod
    def parse_statistics(payload: dict, stat_keys: Mapping[str, Sequence[str]]) -> dict[str, float | None]:
        """Extract shots / shots on target / corners / xG from period ``ALL``.

        Items are located by ``key`` first, then by ``name`` (case-insensitive)
        using the candidate lists in ``stat_keys``; numeric ``homeValue`` /
        ``awayValue`` are preferred over the display strings.
        """
        out = _empty_stats()
        blocks = (payload or {}).get("statistics", []) or []
        block = next(
            (b for b in blocks if isinstance(b, dict) and str(b.get("period", "")).upper() == "ALL"),
            None,
        )
        if block is None:
            return out
        items: list[dict] = [
            item
            for group in block.get("groups", []) or []
            if isinstance(group, dict)
            for item in group.get("statisticsItems", []) or []
            if isinstance(item, dict)
        ]
        by_key = {str(item.get("key")): item for item in items if item.get("key") is not None}
        by_key_ci = {str(item.get("key")).lower(): item for item in items if item.get("key") is not None}
        by_name_ci = {str(item.get("name")).lower(): item for item in items if item.get("name") is not None}

        def locate(candidates: Sequence[str]) -> dict | None:
            for candidate in candidates:
                item = by_key.get(candidate) or by_key_ci.get(candidate.lower()) or by_name_ci.get(candidate.lower())
                if item is not None:
                    return item
            return None

        def side(item: dict, numeric_key: str, text_key: str) -> float | None:
            numeric = item.get(numeric_key)
            value = parse_stat_value(numeric) if isinstance(numeric, (int, float)) and not isinstance(numeric, bool) else None
            return value if value is not None else parse_stat_value(item.get(text_key))

        for metric, (home_field, away_field) in _METRIC_FIELDS.items():
            item = locate(tuple(stat_keys.get(metric, ())))
            if item is None:
                continue
            out[home_field] = side(item, "homeValue", "home")
            out[away_field] = side(item, "awayValue", "away")
        return out

    # -------------------------------------------------------------- harvest
    async def _safe_statistics(self, match: MatchRecord, *, stable: bool = False) -> dict[str, float | None] | None:
        event_id = match.match_id.removeprefix("SS-")
        try:
            payload = await self.statistics(event_id, stable=stable)
        except HarvestError as exc:
            log.warning(
                "statistics unavailable, keeping match without stats",
                extra={"match_id": match.match_id, "status": exc.status, "attempts": exc.attempts, "url": exc.url},
            )
            return None
        return self.parse_statistics(payload, self.endpoints.stat_keys)

    async def _events_or_empty(self, girone: Girone, season_id: int, round_no: int, *, stable: bool = False) -> list[dict]:
        try:
            return await self.events_for_round(girone, season_id, round_no, stable=stable)
        except NotFoundError as exc:
            log.warning("round not found, skipped", extra={"girone": girone.value, "round": round_no, "url": exc.url})
            return []

    def _perimeter_label(self) -> str | None:
        """Season label of the matcher's dictionary (``None`` for synthetic registries)."""
        return season_label_from_year(str(getattr(self.matcher.registry, "season", "") or ""))

    async def _fetch_season_events(
        self, g: Girone, season_id: int, rounds: Iterable[int] | None, *, stable: bool = False
    ) -> tuple[list[list[dict]], int]:
        """Raw events of every (regular) round of ``season_id``; returns the batches and the round count.

        ``stable`` (completed season) is forwarded to every request so the cache
        keeps the payloads with the long TTL.
        """
        if rounds is not None:
            round_list = sorted({int(r) for r in rounds})
        else:
            round_list = await self.rounds(g, season_id, stable=stable)
            if not round_list:
                # A payload drift (``{"rounds": []}`` without ``currentRound``) must
                # surface as a failure, not as a silently empty season.
                raise HarvestError("rounds payload lists no rounds", url=self.endpoints.url_rounds(g, season_id))
        batches = await run_concurrently(self._events_or_empty(g, season_id, r, stable=stable) for r in round_list)
        return batches, len(round_list)

    async def _parse_season(
        self, g: Girone, batches: Iterable[Iterable[dict]], *, historical: bool, with_stats: bool, stable: bool = False
    ) -> tuple[list[MatchRecord], list[Fixture]]:
        """Parse raw event batches (deduplicated by id) and attach statistics on request.

        ``stable`` (completed season) makes the statistics requests use the long cache TTL.
        """
        matches: dict[str, MatchRecord] = {}
        fixtures: dict[str, Fixture] = {}
        for events in batches:
            for event in events:
                parsed = self.parse_event(event, g, self.matcher, historical=historical)
                if isinstance(parsed, MatchRecord):
                    matches[parsed.match_id] = parsed
                elif isinstance(parsed, Fixture):
                    fixtures[parsed.match_id] = parsed
        ordered = list(matches.values())
        if with_stats and ordered:
            stats = await run_concurrently(self._safe_statistics(m, stable=stable) for m in ordered)
            ordered = [replace(m, **s) if s is not None else m for m, s in zip(ordered, stats, strict=True)]
            fetched = [m for m, s in zip(ordered, stats, strict=True) if s is not None]
            _warn_missing_metrics(fetched, self.endpoints.stat_keys, self.name)
        return ordered, list(fixtures.values())

    async def _harvest_season(
        self,
        g: Girone,
        info: SeasonInfo,
        *,
        current: bool,
        perimeter: str | None,
        rounds: Iterable[int] | None,
        with_stats: bool,
    ) -> SeasonHarvest:
        # A season that has ended cannot change upstream: its rounds, events and
        # statistics are read and written through the cache as *stable* (long TTL),
        # so repeated multi-season runs re-download the current season only.
        completed = _season_is_completed(season_label_from_year(info.year), current=current)
        batches, n_rounds = await self._fetch_season_events(g, info.id, rounds, stable=completed)
        label = season_label_from_year(info.year) or _events_season_label(batches)
        historical = (label != perimeter) if (label is not None and perimeter is not None) else not current
        matches, fixtures = await self._parse_season(
            g, batches, historical=historical, with_stats=with_stats, stable=completed
        )
        if not current and fixtures:
            log.info(
                "not-started events of a past season dropped",
                extra={"girone": g.value, "season_id": info.id, "dropped": len(fixtures)},
            )
            fixtures = []
        if label is None:
            label = info.name or f"season {info.id}"
        log.info(
            "sofascore season harvest complete",
            extra={
                "girone": g.value,
                "season_id": info.id,
                "season": label,
                "historical": historical,
                "completed": completed,
                "rounds": n_rounds,
                "matches": len(matches),
                "fixtures": len(fixtures),
            },
        )
        return SeasonHarvest(
            season=info,
            label=label,
            matches=tuple(matches),
            fixtures=tuple(fixtures),
            rounds=n_rounds,
            current=current,
            historical=historical,
            completed=completed,
        )

    async def harvest_seasons(
        self,
        girone: Girone | str,
        *,
        season_id: int | None = None,
        seasons: int = 1,
        rounds: Iterable[int] | None = None,
        with_stats: bool = True,
        strict: bool = False,
    ) -> list[SeasonHarvest]:
        """Harvest the newest ``seasons`` seasons of ``girone``, one :class:`SeasonHarvest` each, newest first.

        The first entry is the *current* season of the listing: it is the only one
        whose not-started events are kept as fixtures, and a failure there always
        propagates as :class:`HarvestError`. A season whose label differs from the
        matcher's dictionary season is *historical*: its team names are resolved
        without girone restriction (clubs may have moved girone or left Serie C)
        and, unless ``strict`` is set, a past season that fails is returned with
        ``error`` set instead of aborting the whole girone.

        ``season_id`` pins one season (``seasons`` is then ignored); ``rounds`` is
        an explicit round list applied to every harvested season (default: all
        regular rounds of each season). Statistics are fetched concurrently for
        finished matches; a failing statistics call keeps the match with ``None``
        stats. The payloads of a *completed* season (every listed season after the
        first whose label precedes the running one) go through the cache as
        stable, with the long ``history_cache_ttl``; a pinned ``season_id`` is
        always treated as current (short TTL).
        """
        g = Girone.parse(girone)
        if int(seasons) < 1:
            raise ValueError("seasons must be >= 1")
        if season_id is not None:
            infos = [SeasonInfo(id=int(season_id), year="", name="")]
        else:
            listed = await self.seasons(g)
            if not listed:
                raise HarvestError("no seasons returned", url=self.endpoints.url_seasons(g))
            infos = listed[: int(seasons)]
            if len(infos) < int(seasons):
                log.warning(
                    "fewer seasons available than requested",
                    extra={"girone": g.value, "requested": int(seasons), "available": len(listed)},
                )
        perimeter = self._perimeter_label()
        newest = season_label_from_year(infos[0].year)
        if perimeter is not None and newest is not None and newest != perimeter:
            log.warning(
                "newest listed season differs from the team dictionary season",
                extra={"girone": g.value, "newest_season": newest, "dictionary_season": perimeter},
            )
        out: list[SeasonHarvest] = []
        for index, info in enumerate(infos):
            current = index == 0
            try:
                out.append(
                    await self._harvest_season(
                        g, info, current=current, perimeter=perimeter, rounds=rounds, with_stats=with_stats
                    )
                )
            except HarvestError as exc:
                if current or strict:
                    raise
                label = season_label_from_year(info.year) or info.name or f"season {info.id}"
                log.error(
                    "historical season failed, skipped",
                    extra={"girone": g.value, "season_id": info.id, "season": label, "status": exc.status, "url": exc.url},
                )
                out.append(
                    SeasonHarvest(
                        season=info,
                        label=label,
                        matches=(),
                        fixtures=(),
                        rounds=0,
                        current=False,
                        historical=True,
                        error=str(exc),
                        completed=_season_is_completed(label, current=False),
                    )
                )
        log.info(
            "sofascore harvest complete",
            extra={
                "girone": g.value,
                "seasons": [item.label for item in out],
                "matches": sum(len(item.matches) for item in out),
                "fixtures": sum(len(item.fixtures) for item in out),
                "failed": [item.label for item in out if item.error],
            },
        )
        return out

    async def harvest(
        self,
        girone: Girone | str,
        *,
        season_id: int | None = None,
        seasons: int = 1,
        rounds: Iterable[int] | None = None,
        with_stats: bool = True,
    ) -> tuple[list[MatchRecord], list[Fixture]]:
        """Played matches and upcoming fixtures of ``girone``.

        With ``seasons > 1`` (and no ``season_id``) the newest ``seasons`` seasons
        are harvested and concatenated, newest first; fixtures come from the
        current season only. Any failing season raises (see
        :meth:`harvest_seasons` for the lenient, per-season view).
        """
        harvested = await self.harvest_seasons(
            girone, season_id=season_id, seasons=seasons, rounds=rounds, with_stats=with_stats, strict=True
        )
        matches = [m for item in harvested for m in item.matches]
        fixtures = [f for item in harvested for f in item.fixtures]
        return matches, fixtures


# =============================================================================
# FotMob
# =============================================================================
class FotMobProvider:
    """Secondary source based on FotMob's web JSON API.

    FotMob protects some endpoints with a request signature sent in the
    ``x-mas`` header; when required, pass it through the client's
    ``extra_headers`` (``AsyncHarvesterClient(extra_headers={"x-mas": "..."})``
    or ``Harvester(extra_headers=...)``). League ids are not stable across
    seasons, so :meth:`resolve_league_ids` discovers them from ``allLeagues``.
    """

    name: Final[str] = "fotmob"

    def __init__(self, client: AsyncHarvesterClient, endpoints: FotMobEndpoints, matcher: TeamMatcher) -> None:
        self.client = client
        self.endpoints = endpoints
        self.matcher = matcher
        self._league_ids: dict[Girone, int] | None = None

    # ----------------------------------------------------------- discovery
    @staticmethod
    def league_ids_from_payload(payload: Any) -> dict[Girone, int]:
        """Walk ``allLeagues`` recursively and map Serie C girone names to ids."""
        found: dict[Girone, int] = {}

        def visit(node: Any) -> None:
            if isinstance(node, dict):
                name = node.get("name")
                league_id = node.get("id")
                if isinstance(name, str) and league_id is not None and _SERIE_C_RE.search(name):
                    hit = _GIRONE_RE.search(name)
                    if hit:
                        girone = Girone.parse(hit.group(1))
                        if girone not in found:
                            found[girone] = _as_int(league_id, 0)
                for value in node.values():
                    visit(value)
            elif isinstance(node, list):
                for value in node:
                    visit(value)

        visit(payload)
        return {g: i for g, i in found.items() if i > 0}

    async def resolve_league_ids(self) -> dict[Girone, int]:
        """Discovered ids, overridden by any non-zero configured id (memoised)."""
        if self._league_ids is None:
            payload = await self.client.get_json(self.endpoints.url_all_leagues())
            ids = self.league_ids_from_payload(payload)
            ids.update(self.endpoints.configured_league_ids())
            log.info("fotmob league ids resolved", extra={"ids": {g.value: i for g, i in ids.items()}})
            self._league_ids = ids
        return dict(self._league_ids)

    async def league_payload(self, league_id: int) -> dict:
        payload = await self.client.get_json(self.endpoints.url_league(league_id))
        return payload if isinstance(payload, dict) else {}

    async def match_details(self, match_id: int | str) -> dict:
        payload = await self.client.get_json(self.endpoints.url_match_details(match_id))
        return payload if isinstance(payload, dict) else {}

    # -------------------------------------------------------------- parsers
    @staticmethod
    def parse_match(match: dict, girone: Girone, matcher: TeamMatcher) -> MatchRecord | Fixture | None:
        """Convert one ``allMatches`` entry (same tolerance as SofaScore)."""
        girone = Girone.parse(girone)
        raw_id = match.get("id")
        if raw_id is None:
            return None
        match_id = f"FM-{raw_id}"
        status = match.get("status") or {}
        when = _utc_date_from_iso(status.get("utcTime") or match.get("utcTime"))
        if when is None:
            log.warning("fotmob match without utcTime skipped", extra={"match_id": match_id})
            return None
        round_no = _as_int(match.get("round", match.get("roundName")), 0)
        home_raw = _team_name(match.get("home"))
        away_raw = _team_name(match.get("away"))
        if not home_raw or not away_raw:
            log.warning("fotmob match without team names skipped", extra={"match_id": match_id})
            return None
        home = _canonical_team(matcher, home_raw, girone, "fotmob")
        away = _canonical_team(matcher, away_raw, girone, "fotmob")
        if home == away:
            return None
        reason = str(((status.get("reason") or {}).get("short")) or "").lower()
        if status.get("cancelled") or reason in {"pp", "postp", "canc", "abd", "aban"}:
            return None
        finished = bool(status.get("finished"))
        started = bool(status.get("started"))
        if not started and not finished:
            return Fixture(match_id=match_id, date=when, round=round_no, girone=girone, home_team=home, away_team=away)
        if not finished:
            return None
        score = _SCORE_RE.search(str(status.get("scoreStr") or ""))
        if score is not None:
            hg, ag = int(score.group(1)), int(score.group(2))
        else:
            hg_raw = (match.get("home") or {}).get("score")
            ag_raw = (match.get("away") or {}).get("score")
            if hg_raw is None or ag_raw is None:
                log.warning("fotmob finished match without score skipped", extra={"match_id": match_id})
                return None
            hg, ag = _as_int(hg_raw), _as_int(ag_raw)
        return MatchRecord(
            match_id=match_id, date=when, round=round_no, girone=girone, home_team=home, away_team=away, home_goals=hg, away_goals=ag
        )

    @staticmethod
    def parse_match_details(payload: dict, stat_keys: Mapping[str, Sequence[str]] = FOTMOB_STAT_KEYS) -> dict[str, float | None]:
        """Extract the canonical stats from ``content.stats.Periods.All``."""
        out = _empty_stats()
        periods = (((payload or {}).get("content") or {}).get("stats") or {}).get("Periods") or {}
        block = periods.get("All") or periods.get("ALL") or {}
        items: list[dict] = [
            item
            for group in block.get("stats", []) or []
            if isinstance(group, dict)
            for item in group.get("stats", []) or []
            if isinstance(item, dict)
        ]
        by_key_ci = {str(item.get("key")).lower(): item for item in items if item.get("key") is not None}
        by_title_ci = {str(item.get("title")).lower(): item for item in items if item.get("title") is not None}
        for metric, (home_field, away_field) in _METRIC_FIELDS.items():
            item = None
            for candidate in stat_keys.get(metric, ()):
                item = by_key_ci.get(candidate.lower()) or by_title_ci.get(candidate.lower())
                if item is not None:
                    break
            if item is None:
                continue
            values = item.get("stats")
            if not isinstance(values, (list, tuple)) or len(values) < 2:
                continue
            out[home_field] = parse_stat_value(values[0])
            out[away_field] = parse_stat_value(values[1])
        return out

    # -------------------------------------------------------------- harvest
    async def _safe_details(self, match: MatchRecord) -> dict[str, float | None] | None:
        fotmob_id = match.match_id.removeprefix("FM-")
        try:
            payload = await self.match_details(fotmob_id)
        except HarvestError as exc:
            log.warning(
                "match details unavailable, keeping match without stats",
                extra={"match_id": match.match_id, "status": exc.status, "attempts": exc.attempts, "url": exc.url},
            )
            return None
        return self.parse_match_details(payload)

    async def harvest(self, girone: Girone | str, *, with_stats: bool = True) -> tuple[list[MatchRecord], list[Fixture]]:
        """Played matches and fixtures of ``girone`` from the league ``allMatches`` payload."""
        g = Girone.parse(girone)
        ids = await self.resolve_league_ids()
        league_id = ids.get(g)
        if not league_id:
            raise HarvestError(f"could not resolve FotMob league id for girone {g.value}", url=self.endpoints.url_all_leagues())
        payload = await self.league_payload(league_id)
        all_matches = ((payload.get("matches") or {}).get("allMatches")) or []
        matches: dict[str, MatchRecord] = {}
        fixtures: dict[str, Fixture] = {}
        for raw in all_matches:
            if not isinstance(raw, dict):
                continue
            parsed = self.parse_match(raw, g, self.matcher)
            if isinstance(parsed, MatchRecord):
                matches[parsed.match_id] = parsed
            elif isinstance(parsed, Fixture):
                fixtures[parsed.match_id] = parsed
        ordered = list(matches.values())
        if with_stats and ordered:
            stats = await run_concurrently(self._safe_details(m) for m in ordered)
            ordered = [replace(m, **s) if s is not None else m for m, s in zip(ordered, stats, strict=True)]
            fetched = [m for m, s in zip(ordered, stats, strict=True) if s is not None]
            _warn_missing_metrics(fetched, FOTMOB_STAT_KEYS, self.name)
        log.info(
            "fotmob harvest complete",
            extra={"girone": g.value, "league_id": league_id, "matches": len(ordered), "fixtures": len(fixtures)},
        )
        return ordered, list(fixtures.values())


# =============================================================================
# Facade
# =============================================================================
@dataclass
class HarvestResult:
    """Outcome of :meth:`Harvester.harvest_all`.

    ``matches`` holds every season harvested in the fixed 16-column schema (no
    season column: the season of a row is :func:`season_of` of its date).
    ``seasons`` lists the season labels present, newest first, and
    ``n_matches_by_season`` the matches of each (the counts sum to
    ``len(matches)``). ``registry_season`` is the perimeter: the season of the
    team dictionary whose teams are the current ones. ``historical_teams`` are
    canonical names resolved through the aliases of older dictionaries that are
    not part of the perimeter (relegated / promoted / excluded clubs);
    ``unmapped_teams`` are raw spellings no dictionary knows, with their counts
    (the two sets are disjoint). ``odds`` holds the pre-match quotes of the
    ``fixtures`` when the provider supplies them (FootyStats; SofaScore does
    not), ready for :func:`save_odds`. ``provider`` names the provider(s) that
    delivered data (``'footystats'``, ``'sofascore'`` or ``'footystats+sofascore'``
    when the fallback kicked in for some girone).
    """

    matches: pd.DataFrame
    fixtures: list[Fixture]
    unmapped_teams: dict[str, int]
    stats: dict[str, int]
    provider: str
    failures: dict[str, str] = field(default_factory=dict)
    seasons: tuple[str, ...] = ()
    n_matches_by_season: dict[str, int] = field(default_factory=dict)
    historical_teams: tuple[str, ...] = ()
    registry_season: str = ""
    odds: list[OddsQuote] = field(default_factory=list)

    @property
    def perimeter_season(self) -> str | None:
        """Season whose teams are the perimeter: the dictionary season, else the newest harvested."""
        return self.registry_season or (self.seasons[0] if self.seasons else None)

    @property
    def n_historical_matches(self) -> int:
        """Matches that belong to a season other than :attr:`perimeter_season`."""
        perimeter = self.perimeter_season
        return int(sum(n for label, n in self.n_matches_by_season.items() if label != perimeter))

    def season_lines(self) -> list[str]:
        """One human line per harvested season (used by the CLI)."""
        perimeter = self.perimeter_season
        lines: list[str] = []
        for label in self.seasons:
            kind = "perimeter" if label == perimeter else "history"
            lines.append(f"  {label}: {self.n_matches_by_season.get(label, 0)} matches ({kind})")
        return lines

    def summary(self) -> str:
        """One-line human summary (used by the CLI)."""
        gironi = ",".join(sorted(self.matches["girone"].dropna().unique().tolist())) if len(self.matches) else "-"
        return (
            f"provider={self.provider} gironi={gironi} matches={len(self.matches)} fixtures={len(self.fixtures)} "
            f"season={self.registry_season or '-'} seasons={','.join(self.seasons) or '-'} "
            f"historical_matches={self.n_historical_matches} historical_teams={len(self.historical_teams)} "
            f"unmapped_teams={len(self.unmapped_teams)} odds={len(self.odds)} requests={self.stats.get('requests', 0)} "
            f"retries={self.stats.get('retries', 0)} cache_hits={self.stats.get('cache_hits', 0)} "
            f"errors={self.stats.get('errors', 0)} failures={len(self.failures)}"
        )


#: Name of the FootyStats provider (:class:`src.ingestion.footystats.FootyStatsProvider`);
#: that module imports this one, so the name is spelled here to avoid a circular import.
FOOTYSTATS_PROVIDER: Final[str] = "footystats"

#: Every provider the facade can drive, in the default fallback order.
PROVIDER_NAMES: Final[tuple[str, ...]] = (FOOTYSTATS_PROVIDER, SofaScoreProvider.name, FotMobProvider.name)


def default_providers(api_key: str | None = None, *, key_env: str = "FOOTYSTATS_API_KEY") -> tuple[str, ...]:
    """Provider order used when none is given: FootyStats first when a key is available.

    ``('footystats', 'sofascore')`` when ``api_key`` is given or
    :func:`src.ingestion.footystats.find_api_key` finds one in the environment
    variable ``key_env`` or in the ``.env`` file (``key_env`` is ``[footystats]
    key_env`` of ``endpoints.toml``; the facade passes its configured value, so a
    renamed variable is honoured), ``('sofascore',)`` otherwise, so an
    installation without a FootyStats account keeps the historical behaviour.
    """
    if api_key is None:
        from src.ingestion import footystats as _footystats  # local import: footystats imports this module

        api_key = _footystats.find_api_key(key_env=key_env)
    if api_key:
        return (FOOTYSTATS_PROVIDER, SofaScoreProvider.name)
    return (SofaScoreProvider.name,)


class Harvester:
    """Harvest every girone through the configured providers (ordered fallbacks).

    ``providers`` is tried in order for each girone; the first provider that
    succeeds wins, failures are logged and recorded in
    :attr:`HarvestResult.failures`. ``season_id`` and multi-season harvesting
    apply to SofaScore only (FotMob exposes the current season).
    ``extra_headers`` reach every provider client (FotMob's ``x-mas`` signature
    header belongs there).

    ``registry`` (or ``season``, the stem of ``config/teams_<season>.json``,
    :data:`DEFAULT_SEASON` by default) defines the perimeter of current teams.
    Unless a ``matcher`` is injected, the matcher is built on that registry
    extended with the aliases of every *older* dictionary found in
    ``config_dir`` (``historical_aliases``), so the names of past seasons are
    canonicalised consistently: clubs that still play keep their current name,
    clubs that left keep their old canonical spelling and are reported as
    :attr:`HarvestResult.historical_teams`.

    Raw payloads are cached in ``cache_dir``: ``cache_ttl`` (``[http]
    cache_ttl_hours``, 6 h) applies to the current season, whose results and
    statistics still change; ``history_cache_ttl`` (``[http]
    history_cache_ttl_hours``, 365 days) applies to completed seasons, whose
    payloads are immutable, so a multi-season run repeated later re-downloads
    the current season only. ``None`` for either means "take the endpoints
    configuration".

    ``providers`` defaults to :func:`default_providers`: FootyStats (the
    official API, reachable from datacenters, with pre-match odds) followed by
    SofaScore when an API key is available (``api_key``, the environment
    variable ``FOOTYSTATS_API_KEY`` or a ``.env`` file), SofaScore alone
    otherwise. FootyStats harvests every requested competition in one pass;
    the facade then falls back to the next provider for the gironi it could not
    deliver. ``mapping_path`` is the persisted FootyStats team-id mapping
    (``config/footystats_teams.json`` by default).
    """

    def __init__(
        self,
        *,
        endpoints: EndpointsConfig | None = None,
        registry: TeamRegistry | None = None,
        cache_dir: Path = RAW_DIR,
        cache_ttl: timedelta | None = None,
        providers: Sequence[str] | None = None,
        transport: httpx.AsyncBaseTransport | None = None,
        extra_headers: Mapping[str, str] | None = None,
        matcher: TeamMatcher | None = None,
        sleeper: Sleeper = asyncio.sleep,
        rng: random.Random | None = None,
        season: str | None = None,
        historical_aliases: bool = True,
        config_dir: Path = CONFIG_DIR,
        history_cache_ttl: timedelta | None = None,
        api_key: str | None = None,
        mapping_path: Path | None = None,
    ) -> None:
        self.api_key = api_key
        self.endpoints = endpoints if endpoints is not None else load_endpoints()
        if providers is None:  # the key lookup must use the variable name the configuration declares
            providers = default_providers(api_key, key_env=self.endpoints.footystats.key_env)
        names = tuple(str(p).lower() for p in providers)
        unknown = [p for p in names if p not in PROVIDER_NAMES]
        if unknown or not names:
            raise ValueError(f"providers must be a non-empty subset of {PROVIDER_NAMES}, got {providers!r}")
        self.mapping_path = Path(mapping_path) if mapping_path is not None else Path(config_dir) / "footystats_teams.json"
        self.config_dir = Path(config_dir)
        if registry is not None:
            self.registry = registry
        else:
            self.registry = load_registry(season or DEFAULT_SEASON, self.config_dir)
        if matcher is not None:
            self.matcher = matcher
        else:
            match_registry = self.registry_with_history(self.registry, self.config_dir) if historical_aliases else self.registry
            self.matcher = TeamMatcher(match_registry)
        ttl = cache_ttl if cache_ttl is not None else self.endpoints.http.cache_ttl
        stable_ttl = history_cache_ttl if history_cache_ttl is not None else self.endpoints.http.history_cache_ttl
        self.cache = RawCache(Path(cache_dir), ttl=ttl, stable_ttl=stable_ttl)
        self.providers = names
        self._transport = transport
        self._extra_headers = dict(extra_headers or {})
        self._sleeper = sleeper
        self._rng = rng

    # -------------------------------------------------------------- plumbing
    def _make_client(self, base_url: str) -> AsyncHarvesterClient:
        return AsyncHarvesterClient(
            base_url=base_url,
            http=self.endpoints.http,
            cache=self.cache,
            transport=self._transport,
            sleeper=self._sleeper,
            rng=self._rng,
            extra_headers=self._extra_headers,
        )

    def _make_provider(self, name: str, client: AsyncHarvesterClient) -> SofaScoreProvider | FotMobProvider:
        if name == SofaScoreProvider.name:
            return SofaScoreProvider(client, self.endpoints.sofascore, self.matcher)
        return FotMobProvider(client, self.endpoints.fotmob, self.matcher)

    def _make_footystats(self) -> Any:
        """:class:`src.ingestion.footystats.FootyStatsProvider` on this facade's cache, matcher and transport.

        Raises :class:`src.ingestion.footystats.MissingApiKeyError` (a
        :class:`HarvestError`) when no key can be resolved, so the facade records
        it as a provider failure and moves on to the next provider.
        """
        from src.ingestion import footystats as _footystats  # local import: footystats imports this module

        client = _footystats.FootyStatsClient(
            api_key=_footystats.resolve_api_key(
                self.api_key, key_env=self.endpoints.footystats.key_env, endpoints=self.endpoints.footystats
            ),
            endpoints=self.endpoints.footystats,
            http=self.endpoints.http,
            cache=self.cache,
            transport=self._transport,
            sleeper=self._sleeper,
            rng=self._rng,
        )
        mapper = _footystats.FootyStatsTeamMapper(self.registry, self.matcher, mapping_path=self.mapping_path)
        return _footystats.FootyStatsProvider(client, mapper, self.endpoints.footystats)

    def _base_url(self, name: str) -> str:
        return self.endpoints.sofascore.base_url if name == SofaScoreProvider.name else self.endpoints.fotmob.base_url

    def _supports(self, name: str, girone: Girone) -> str | None:
        """Why provider ``name`` cannot serve ``girone`` at all (``None`` when it can).

        SofaScore needs a configured tournament id; FotMob resolves ids at runtime
        and FootyStats resolves seasons by competition name, so both always qualify.
        """
        if name == SofaScoreProvider.name and girone not in self.endpoints.sofascore.tournament_ids:
            return f"no SofaScore tournament id configured for girone {girone.value}"
        return None

    def supported_gironi(self) -> tuple[Girone, ...]:
        """The competitions at least one configured provider can serve, in configuration order.

        This is the default of :meth:`harvest_all`: a SofaScore-only installation
        (no FootyStats key) harvests the gironi that have a SofaScore tournament
        id (the three Serie C gironi) and never records Serie B as a failure,
        while with the FootyStats provider every competition qualifies. A girone
        requested explicitly is always attempted and reported.
        """
        return tuple(girone for girone in Girone if any(self._supports(name, girone) is None for name in self.providers))

    @staticmethod
    def registry_with_history(registry: TeamRegistry, config_dir: Path = CONFIG_DIR) -> TeamRegistry:
        """``registry`` extended with the aliases of every older ``teams_<season>.json`` in ``config_dir``.

        Older dictionaries are merged most recent first through
        :meth:`TeamRegistry.with_historical_aliases`; the current composition is
        unchanged. A registry whose ``season`` is not a season label (synthetic
        leagues) is returned as is.
        """
        stem = season_file_stem(str(registry.season))
        if season_label_from_year(stem) is None:
            return registry
        older = sorted((s for s in list_seasons(Path(config_dir)) if s < stem), reverse=True)
        if not older:
            return registry
        merged = registry.with_historical_aliases(*(load_registry(s, Path(config_dir)) for s in older))
        log.info(
            "historical aliases merged",
            extra={"season": stem, "older_seasons": older, "historical_teams": len(merged.historical_names)},
        )
        return merged

    def _classify_teams(
        self, matches: Iterable[MatchRecord], fixtures: Iterable[Fixture]
    ) -> tuple[dict[str, int], tuple[str, ...]]:
        """Raw spellings no dictionary knows (with counts) and the historical canonical names seen."""
        unmapped: dict[str, int] = {}
        historical: set[str] = set()
        for record in (*matches, *fixtures):
            for name in (record.home_team, record.away_team):
                if self.matcher.is_current(name):
                    continue
                if self.matcher.is_historical(name):
                    historical.add(name)
                else:
                    unmapped[name] = unmapped.get(name, 0) + 1
        return dict(sorted(unmapped.items())), tuple(sorted(historical))

    # ---------------------------------------------------------------- public
    async def _footystats_once(
        self, targets: Sequence[Girone], *, seasons: int, with_odds: bool
    ) -> tuple[Any | None, HarvestError | None, ClientStats]:
        """Run the FootyStats provider once for every target (it harvests all competitions in one pass)."""
        try:
            provider = self._make_footystats()
        except HarvestError as exc:
            log.error("footystats unavailable", extra={"error": str(exc)})
            return None, exc, ClientStats()
        try:
            result = await provider.harvest([g.value for g in targets], seasons=seasons, with_odds=with_odds)
        except HarvestError as exc:
            log.error("footystats harvest failed", extra={"status": exc.status, "url": exc.url})
            return None, exc, provider.client.client_stats()
        finally:
            await provider.client.close()
        return result, None, provider.client.client_stats()

    async def harvest_all(
        self,
        gironi: Iterable[Girone] | None = None,
        *,
        season_id: int | None = None,
        seasons: int = 3,
        with_stats: bool = True,
        with_odds: bool = True,
    ) -> HarvestResult:
        """Harvest ``gironi`` and return a validated canonical frame plus fixtures.

        ``gironi`` defaults to :meth:`supported_gironi`, the competitions the
        configured providers can serve (every one with FootyStats, the Serie C
        gironi with SofaScore alone), so a default call on a keyless installation
        does not record Serie B as a failure.
        Providers are tried in order for every girone and the first that succeeds
        wins. FootyStats harvests the newest ``seasons`` seasons of every requested
        competition in one pass (``with_odds`` adds the pre-match quotes of the
        fixtures); a competition it could not deliver falls through to the next
        provider. On SofaScore the newest ``seasons`` seasons of every girone are
        harvested (current season plus history; ``season_id`` pins a single season
        instead); fixtures come from the current season only. A past season that
        fails is recorded in ``HarvestResult.failures`` under
        ``"<girone>:<provider>:<season>"`` and the rest is kept. Raises
        :class:`HarvestError` only when every girone failed on every provider;
        partial failures are reported in ``HarvestResult.failures``.
        """
        if int(seasons) < 1:
            raise ValueError("seasons must be >= 1")
        targets = [Girone.parse(g) for g in (self.supported_gironi() if gironi is None else gironi)]
        clients: dict[str, AsyncHarvesterClient] = {}
        providers: dict[str, SofaScoreProvider | FotMobProvider] = {}
        all_matches: list[MatchRecord] = []
        all_fixtures: list[Fixture] = []
        all_odds: list[OddsQuote] = []
        season_labels: list[str] = []
        used: list[str] = []
        failures: dict[str, str] = {}
        last_error: HarvestError | None = None
        footystats_result: Any | None = None
        footystats_error: HarvestError | None = None
        footystats_stats = ClientStats()
        footystats_done = False
        try:
            for girone in targets:
                success = False
                for name in self.providers:
                    unsupported = self._supports(name, girone)
                    if unsupported is not None:
                        failures[f"{girone.value}:{name}"] = unsupported
                        log.warning("provider cannot serve girone", extra={"provider": name, "girone": girone.value, "reason": unsupported})
                        continue
                    if name == FOOTYSTATS_PROVIDER:
                        if not footystats_done:
                            footystats_done = True
                            footystats_result, footystats_error, footystats_stats = await self._footystats_once(
                                targets, seasons=seasons, with_odds=with_odds
                            )
                        if footystats_result is None:
                            last_error = footystats_error
                            failures[f"{girone.value}:{name}"] = str(footystats_error)
                            continue
                        problem = footystats_result.problem_for(girone.value)
                        if problem is not None:
                            failures[f"{girone.value}:{name}"] = problem
                            log.error("footystats could not deliver girone", extra={"girone": girone.value, "reason": problem})
                            continue
                        matches = [m for m in footystats_result.records if m.girone == girone]
                        fixtures = [f for f in footystats_result.fixtures if f.girone == girone]
                        fixture_ids = {f.match_id for f in fixtures}
                        all_odds.extend(q for q in footystats_result.odds if q.match_id in fixture_ids)
                        for label in footystats_result.seasons.get(girone.value, ()):
                            if label not in season_labels:
                                season_labels.append(label)
                        for key, message in footystats_result.failures.items():
                            if key.split(":", 1)[0] == girone.value:
                                failures[f"{girone.value}:{name}:{key.split(':', 1)[1]}"] = message
                    else:
                        if name not in clients:
                            clients[name] = self._make_client(self._base_url(name))
                            providers[name] = self._make_provider(name, clients[name])
                        provider = providers[name]
                        try:
                            if isinstance(provider, SofaScoreProvider):
                                harvested = await provider.harvest_seasons(
                                    girone, season_id=season_id, seasons=seasons, with_stats=with_stats
                                )
                                matches = [m for item in harvested for m in item.matches]
                                fixtures = [f for item in harvested for f in item.fixtures]
                                for item in harvested:
                                    if item.label not in season_labels:
                                        season_labels.append(item.label)
                                    if item.error:
                                        failures[f"{girone.value}:{name}:{item.label}"] = item.error
                            else:
                                matches, fixtures = await provider.harvest(girone, with_stats=with_stats)
                        except HarvestError as exc:
                            last_error = exc
                            failures[f"{girone.value}:{name}"] = str(exc)
                            log.error(
                                "provider failed for girone",
                                extra={"provider": name, "girone": girone.value, "status": exc.status, "url": exc.url},
                            )
                            continue
                    all_matches.extend(matches)
                    all_fixtures.extend(fixtures)
                    if name not in used:
                        used.append(name)
                    success = True
                    break
                if not success:
                    log.error("no provider could harvest girone", extra={"girone": girone.value})
        finally:
            for client in clients.values():
                await client.close()

        if targets and not used and last_error is not None:
            raise last_error

        totals = ClientStats()
        totals.add(footystats_stats)
        for client in clients.values():
            totals.add(client.stats)
        seen: set[str] = set()
        fixtures_unique = [f for f in all_fixtures if not (f.match_id in seen or seen.add(f.match_id))]
        frame = matches_to_frame(all_matches) if all_matches else empty_matches_frame()
        counts = _season_counts(all_matches)
        labels = _sort_season_labels([*season_labels, *counts])
        unmapped, historical = self._classify_teams(all_matches, fixtures_unique)
        result = HarvestResult(
            matches=frame,
            fixtures=fixtures_unique,
            unmapped_teams=unmapped,
            stats=totals.as_dict(),
            provider="+".join(used) if used else self.providers[0],
            failures=failures,
            seasons=tuple(labels),
            n_matches_by_season={label: int(counts.get(label, 0)) for label in labels},
            historical_teams=historical,
            registry_season=season_label_from_year(str(self.registry.season)) or str(self.registry.season),
            odds=all_odds,
        )
        log.info("harvest_all complete", extra={"summary": result.summary()})
        return result

    # ----------------------------------------------------------- persistence
    @staticmethod
    def save_processed(df: pd.DataFrame, path: Path = PROCESSED_DIR / "matches.csv") -> Path:
        """Write the validated canonical frame to CSV (ISO dates); returns ``path``."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        validate_matches_df(df).to_csv(path, index=False, date_format="%Y-%m-%d")
        log.info("processed matches saved", extra={"path": str(path), "rows": int(len(df))})
        return path

    @staticmethod
    def load_processed(path: Path = PROCESSED_DIR / "matches.csv") -> pd.DataFrame:
        """Read a CSV written by :meth:`save_processed` into a validated frame."""
        path = Path(path)
        try:
            raw = pd.read_csv(path, dtype={c: "string" for c in ("match_id", "girone", "home_team", "away_team")})
        except pd.errors.EmptyDataError:
            return empty_matches_frame()
        if raw.empty and not any(c in raw.columns for c in MATCH_COLUMNS):
            return empty_matches_frame()
        return validate_matches_df(raw)

    @staticmethod
    def save_fixtures(fixtures: Iterable[Fixture], path: Path = PROCESSED_DIR / "fixtures.csv") -> Path:
        """Write fixtures to CSV with :data:`FIXTURE_COLUMNS`; returns ``path``."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        rows = [
            {
                "match_id": f.match_id,
                "date": f.date.isoformat(),
                "round": int(f.round),
                "girone": f.girone.value,
                "home_team": f.home_team,
                "away_team": f.away_team,
            }
            for f in fixtures
        ]
        frame = pd.DataFrame(rows, columns=list(FIXTURE_COLUMNS))
        frame.to_csv(path, index=False)
        log.info("fixtures saved", extra={"path": str(path), "rows": len(rows)})
        return path

    @staticmethod
    def load_fixtures(path: Path = PROCESSED_DIR / "fixtures.csv") -> list[Fixture]:
        """Read fixtures written by :meth:`save_fixtures`.

        Rows are deduplicated by ``match_id`` (first occurrence wins, file order
        otherwise preserved), mirroring :meth:`harvest_all`. A hand-edited or
        concatenated CSV with a repeated line would otherwise surface as a
        duplicated prediction in ``main.py`` and as a duplicated entry in the
        dashboard's match selector and CSV export; duplicates are logged.
        """
        path = Path(path)
        try:
            raw = pd.read_csv(path, dtype="string", keep_default_na=False)
        except pd.errors.EmptyDataError:
            return []
        missing = [c for c in FIXTURE_COLUMNS if c not in raw.columns]
        if missing:
            raise ValueError(f"fixtures file {path} is missing columns {missing}")
        out: list[Fixture] = []
        seen: set[str] = set()
        duplicates: dict[str, int] = {}
        for row in raw.itertuples(index=False):
            match_id = str(row.match_id)
            if match_id in seen:
                duplicates[match_id] = duplicates.get(match_id, 0) + 1
                continue
            seen.add(match_id)
            out.append(
                Fixture(
                    match_id=match_id,
                    date=date.fromisoformat(str(row.date)[:10]),
                    round=_as_int(row.round, 0),
                    girone=Girone.parse(str(row.girone)),
                    home_team=str(row.home_team),
                    away_team=str(row.away_team),
                )
            )
        if duplicates:
            log.warning(
                "duplicate fixtures dropped, first occurrence kept",
                extra={
                    "path": str(path),
                    "dropped": int(sum(duplicates.values())),
                    "match_ids": sorted(duplicates),
                    "kept": len(out),
                },
            )
        return out


    @staticmethod
    def save_odds(odds: Iterable[OddsQuote], path: Path = ODDS_DIR / "footystats.csv") -> Path:
        """Write quotes to CSV with :data:`ODDS_COLUMNS` (one row per selection); returns ``path``.

        The format is the one ``main.load_odds_dir`` and
        :func:`src.dashboard.state.load_odds_quotes` read from ``data/odds/*.csv``:
        ``market`` is the :class:`~src.common.schemas.Market` value, ``selection``
        a key of :data:`~src.common.schemas.SELECTIONS`. Rows follow the input
        order, selections the canonical order of their market. Odds are written
        with Python's shortest round-trip representation (``2.1``, ``101.25``,
        ``10.125``), never rounded to a number of significant digits, so the
        value engine reads exactly the price the provider quoted. An empty
        iterable writes the header only, so a stale file never survives a run
        that found no fixtures.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        rows = [
            {
                "match_id": quote.match_id,
                "bookmaker": quote.bookmaker,
                "market": quote.market.value,
                "selection": selection,
                "odds": float(quote.odds[selection]),
            }
            for quote in odds
            for selection in quote.odds
        ]
        frame = pd.DataFrame(rows, columns=list(ODDS_COLUMNS))
        frame.to_csv(path, index=False)
        log.info("odds saved", extra={"path": str(path), "rows": len(rows)})
        return path


#: Module-level aliases of the persistence helpers: ``main.py`` and other
#: integrators import ``from src.ingestion.harvester import load_processed, load_fixtures``.
save_processed = Harvester.save_processed
load_processed = Harvester.load_processed
save_fixtures = Harvester.save_fixtures
load_fixtures = Harvester.load_fixtures
save_odds = Harvester.save_odds


# =============================================================================
# CLI
# =============================================================================
def _csv_has_rows(path: Path) -> bool:
    """``True`` when ``path`` exists and holds at least one data row below the header."""
    try:
        with Path(path).open("r", encoding="utf-8") as handle:
            next(handle)
            return any(line.strip() for line in handle)
    except (OSError, StopIteration):
        return False


def _write_decision(result: HarvestResult, out: Path, *, allow_partial: bool) -> tuple[bool, str | None]:
    """Whether the CLI may write ``result`` to ``out`` (and why not).

    * an empty harvest never overwrites a file that already has rows;
    * a partial harvest (some girone/provider failed) is written only with
      ``allow_partial``, so a transient outage cannot replace a complete
      dataset with a fragment.
    """
    if result.matches.empty:
        if _csv_has_rows(out):
            return False, f"empty harvest: refusing to overwrite non-empty {out}"
        return True, None
    if result.failures and not allow_partial:
        return False, f"partial harvest ({len(result.failures)} failures): {out} left untouched (use --allow-partial to write anyway)"
    return True, None


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m src.ingestion.harvester",
        description="Harvest Serie C matches and fixtures from SofaScore / FotMob into the canonical CSVs.",
    )
    parser.add_argument(
        "--gironi",
        nargs="+",
        default=["A", "B", "C"],
        choices=[g.value for g in Girone],
        help="competitions to harvest (Serie C gironi by default; SB = Serie B is served by the footystats provider only)",
    )
    parser.add_argument(
        "--seasons",
        type=int,
        default=3,
        help="seasons to harvest, newest first: the current one plus N-1 of history, all in ONE matches CSV (default: %(default)s)",
    )
    parser.add_argument(
        "--season",
        default=DEFAULT_SEASON,
        help="team dictionary config/teams_<season>.json defining the perimeter of current teams (default: %(default)s)",
    )
    parser.add_argument("--season-id", type=int, default=None, help="SofaScore season id: harvest only that season (default: the newest --seasons)")
    parser.add_argument("--out", type=Path, default=PROCESSED_DIR / "matches.csv", help="matches CSV output")
    parser.add_argument("--fixtures-out", type=Path, default=PROCESSED_DIR / "fixtures.csv", help="fixtures CSV output")
    parser.add_argument(
        "--odds-out",
        type=Path,
        default=ODDS_DIR / "footystats.csv",
        help="pre-match odds CSV (match_id,bookmaker,market,selection,odds), written when the provider supplies odds (FootyStats)",
    )
    parser.add_argument("--no-odds", action="store_true", help="do not collect pre-match odds (FootyStats)")
    parser.add_argument("--no-stats", action="store_true", help="skip per-match statistics calls")
    parser.add_argument(
        "--allow-partial",
        action="store_true",
        help="write the CSVs even when some girone/provider failed (exit code stays non-zero)",
    )
    parser.add_argument("--log-json", action="store_true", help="emit JSON-lines logs")
    parser.add_argument(
        "--providers",
        nargs="+",
        default=None,
        choices=list(PROVIDER_NAMES),
        help="ordered provider fallbacks (default: footystats sofascore when a FootyStats key is available, else sofascore)",
    )
    parser.add_argument("--cache-dir", type=Path, default=RAW_DIR, help="raw JSON cache directory")
    parser.add_argument(
        "--cache-ttl-hours",
        type=float,
        default=None,
        help="cache TTL (hours) of the current season's payloads, which still change (default: [http] cache_ttl_hours, 6)",
    )
    parser.add_argument(
        "--history-cache-ttl-hours",
        type=float,
        default=None,
        help="cache TTL (hours) of completed seasons, whose payloads never change; 0 forces their re-download "
        "(default: [http] history_cache_ttl_hours, 8760 = 365 days)",
    )
    parser.add_argument("--endpoints", type=Path, default=CONFIG_DIR / "endpoints.toml", help="endpoints TOML file")
    parser.add_argument("--x-mas", default=os.getenv("FOTMOB_X_MAS"), help="FotMob x-mas signature header (env FOTMOB_X_MAS)")
    return parser


def _print_season_report(result: HarvestResult, *, season_stem: str, n_current_teams: int) -> None:
    """Per-season console summary: perimeter, matches per season, history used, historical teams."""
    perimeter = result.perimeter_season or "-"
    print(f"perimeter: season {perimeter} (config/teams_{season_stem}.json, {n_current_teams} current teams)")
    print("seasons harvested:")
    for line in result.season_lines() or ["  (none)"]:
        print(line)
    print(f"historical matches used: {result.n_historical_matches}")
    if result.historical_teams:
        print(
            f"historical teams (not in the {perimeter} perimeter, history only): "
            + ", ".join(result.historical_teams)
            + f" ({len(result.historical_teams)})"
        )


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entry point; returns the process exit code."""
    parser = _build_parser()
    args = parser.parse_args(argv)
    if args.seasons < 1:
        parser.error("--seasons must be >= 1")
    configure_logging(json_output=args.log_json, force=True)
    extra_headers = {"x-mas": args.x_mas} if args.x_mas else None
    try:
        endpoints = load_endpoints(args.endpoints)
    except FileNotFoundError as exc:
        log.error("endpoints file not found", extra={"path": str(args.endpoints), "error": str(exc)})
        print(f"harvest failed: endpoints file not found: {args.endpoints}", file=sys.stderr)
        return 1
    except ValueError as exc:  # malformed TOML (TOMLDecodeError is a ValueError) or a missing key
        log.error("endpoints file invalid", extra={"path": str(args.endpoints), "error": str(exc)})
        print(f"harvest failed: invalid endpoints file {args.endpoints}: {exc}", file=sys.stderr)
        return 1
    try:
        harvester = Harvester(
            endpoints=endpoints,
            cache_dir=args.cache_dir,
            cache_ttl=timedelta(hours=args.cache_ttl_hours) if args.cache_ttl_hours is not None else None,
            history_cache_ttl=timedelta(hours=args.history_cache_ttl_hours) if args.history_cache_ttl_hours is not None else None,
            providers=tuple(args.providers) if args.providers else None,
            extra_headers=extra_headers,
            season=args.season,
        )
    except FileNotFoundError as exc:
        log.error("team dictionary not found", extra={"season": args.season, "error": str(exc)})
        print(f"harvest failed: {exc}", file=sys.stderr)
        return 1
    gironi = [Girone.parse(g) for g in args.gironi]
    try:
        result = asyncio.run(
            harvester.harvest_all(
                gironi, season_id=args.season_id, seasons=args.seasons, with_stats=not args.no_stats, with_odds=not args.no_odds
            )
        )
    except HarvestError as exc:
        log.error("harvest failed", extra={"url": exc.url, "status": exc.status, "attempts": exc.attempts})
        print(f"harvest failed: {exc}", file=sys.stderr)
        return 1
    complete = not result.failures and len(result.matches) > 0
    write, reason = _write_decision(result, args.out, allow_partial=args.allow_partial)
    print(result.summary())
    _print_season_report(
        result, season_stem=season_file_stem(args.season), n_current_teams=len(harvester.registry.canonical_names)
    )
    if write:
        save_processed(result.matches, args.out)
        save_fixtures(result.fixtures, args.fixtures_out)
        print(f"matches -> {args.out}")
        print(f"fixtures -> {args.fixtures_out}")
        if result.odds:
            save_odds(result.odds, args.odds_out)
            print(f"odds -> {args.odds_out}")
    else:
        log.error("harvest output not written", extra={"reason": reason, "out": str(args.out)})
        print(reason, file=sys.stderr)
    if result.unmapped_teams:
        print("unmapped teams: " + ", ".join(f"{name} x{count}" for name, count in result.unmapped_teams.items()))
    for key, message in result.failures.items():
        print(f"failure {key}: {message}", file=sys.stderr)
    if not complete:
        print("harvest incomplete: " + ("no matches harvested" if result.matches.empty else f"{len(result.failures)} failures"), file=sys.stderr)
    return 0 if complete else 1


if __name__ == "__main__":  # pragma: no cover - manual entry point
    raise SystemExit(main())
