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
from collections.abc import Awaitable, Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from datetime import date, datetime, timedelta, timezone
from email.utils import parsedate_to_datetime
from pathlib import Path
from typing import Any, Final
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

import httpx
import pandas as pd

from src.common.logging_config import configure_logging, get_logger
from src.common.paths import CONFIG_DIR, PROCESSED_DIR, RAW_DIR
from src.common.schemas import (
    MATCH_COLUMNS,
    Fixture,
    Girone,
    MatchRecord,
    empty_matches_frame,
    matches_to_frame,
    validate_matches_df,
)
from src.common.teams import TeamRegistry, load_registry
from src.ingestion.team_matcher import TeamMatcher

__all__ = [
    "DEFAULT_USER_AGENTS",
    "FIXTURE_COLUMNS",
    "FOTMOB_STAT_KEYS",
    "STAT_FIELDS",
    "AsyncHarvesterClient",
    "ClientStats",
    "EndpointsConfig",
    "FotMobEndpoints",
    "FotMobProvider",
    "HarvestError",
    "HarvestResult",
    "Harvester",
    "HttpConfig",
    "NotFoundError",
    "RateLimitedError",
    "RawCache",
    "SeasonInfo",
    "SofaScoreEndpoints",
    "SofaScoreProvider",
    "load_endpoints",
    "main",
    "parse_retry_after",
    "parse_stat_value",
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

    @property
    def cache_ttl(self) -> timedelta:
        """Cache time-to-live as a :class:`timedelta`."""
        return timedelta(hours=float(self.cache_ttl_hours))


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
class EndpointsConfig:
    """Whole ``endpoints.toml``."""

    sofascore: SofaScoreEndpoints
    fotmob: FotMobEndpoints
    http: HttpConfig


def _require(section: Mapping[str, Any], key: str, section_name: str, path: Path) -> Any:
    try:
        return section[key]
    except KeyError as exc:
        raise ValueError(f"{path}: [{section_name}] is missing required key {key!r}") from exc


def load_endpoints(path: Path = CONFIG_DIR / "endpoints.toml") -> EndpointsConfig:
    """Parse ``endpoints.toml`` (stdlib :mod:`tomllib`) into :class:`EndpointsConfig`."""
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
        stat_keys={str(k): tuple(str(x) for x in v) for k, v in sofa_raw.get("stat_keys", {}).items()},
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
    )
    log.debug("endpoints loaded", extra={"path": str(path)})
    return EndpointsConfig(sofascore=sofascore, fotmob=fotmob, http=http)


# =============================================================================
# Raw cache
# =============================================================================
class RawCache:
    """JSON file cache of raw responses.

    One file per URL (``<sha1>.json``) holding ``{"url", "fetched_at", "payload"}``.
    Writes are atomic (temp file + :func:`os.replace`). ``ttl=None`` never
    expires; ``ttl=timedelta(0)`` expires immediately (cache effectively
    write-only, useful for debugging).
    """

    def __init__(self, directory: Path, ttl: timedelta | None = None) -> None:
        self.directory = Path(directory)
        self.ttl = ttl
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
    def _is_expired(self, fetched_at: str) -> bool:
        if self.ttl is None:
            return False
        stamp = datetime.fromisoformat(str(fetched_at))
        if stamp.tzinfo is None:
            stamp = stamp.replace(tzinfo=timezone.utc)
        return datetime.now(timezone.utc) >= stamp + self.ttl

    def get(self, url: str, params: Mapping[str, Any] | None = None) -> Any | None:
        """Cached payload or ``None`` on miss, expiry or unreadable file."""
        path = self.path_for(url, params)
        if not path.exists():
            self.misses += 1
            return None
        try:
            record = json.loads(path.read_text(encoding="utf-8"))
            if not isinstance(record, dict) or "payload" not in record:
                raise ValueError("cache record without payload")
            expired = self._is_expired(record.get("fetched_at", ""))
        except (OSError, ValueError, TypeError) as exc:
            log.warning("corrupt cache file treated as miss", extra={"path": str(path), "error": str(exc)})
            self.misses += 1
            return None
        if expired:
            log.debug("cache entry expired", extra={"path": str(path), "url": url})
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
    async def get_json(self, path: str, params: Mapping[str, Any] | None = None, *, use_cache: bool = True) -> Any:
        """GET ``path`` and return the decoded JSON body.

        Cache hits return immediately. Retryable failures (HTTP 429, 5xx,
        transport errors, timeouts) are retried up to ``http.max_retries``
        times; other 4xx raise at once (:class:`NotFoundError` for 404), as do
        undecodable bodies and other non-transient ``httpx`` errors. Every
        failure surfaces as a :class:`HarvestError` subclass. (``Accept-Encoding``
        is left to ``httpx`` so only decodable encodings are advertised.)
        """
        url = self.build_url(path)
        if use_cache and self.cache is not None:
            cached = self.cache.get(url, params)
            if cached is not None:
                self.stats.cache_hits += 1
                log.debug("cache hit", extra={"url": url})
                return cached
        client = self._ensure_client()
        async with self._semaphore:
            payload = await self._fetch_with_retry(client, url, params)
        if use_cache and self.cache is not None:
            self.cache.put(url, payload, params)
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


def _canonical_team(matcher: TeamMatcher, raw_name: str, girone: Girone, source: str) -> str:
    """Canonical name via ``matcher``; the raw spelling (with a warning) when unmapped."""
    result = matcher.match(raw_name, girone)
    if result.canonical is None:
        log.warning(
            "unmapped team name kept raw",
            extra={"source": source, "team": raw_name, "girone": girone.value, "candidates": list(result.candidates[:3])},
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


# =============================================================================
# SofaScore
# =============================================================================
@dataclass(frozen=True, slots=True)
class SeasonInfo:
    """One entry of the SofaScore ``seasons`` payload."""

    id: int
    year: str
    name: str


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

    async def rounds(self, girone: Girone | str, season_id: int, *, regular_only: bool = True) -> list[int]:
        """Sorted distinct round numbers of the season.

        With ``regular_only`` entries carrying a ``name``/``prefix`` (play-off
        and play-out rounds) are skipped. When the API lists no rounds but a
        ``currentRound`` exists, rounds ``1..current`` are assumed.
        """
        payload = await self.client.get_json(self.endpoints.url_rounds(girone, season_id)) or {}
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

    async def events_for_round(self, girone: Girone | str, season_id: int, round: int) -> list[dict]:
        payload = await self.client.get_json(self.endpoints.url_events_by_round(girone, season_id, round)) or {}
        return [e for e in payload.get("events", []) or [] if isinstance(e, dict)]

    async def next_events(self, girone: Girone | str, season_id: int, page: int = 0) -> list[dict]:
        payload = await self.client.get_json(self.endpoints.url_events_next(girone, season_id, page)) or {}
        return [e for e in payload.get("events", []) or [] if isinstance(e, dict)]

    async def last_events(self, girone: Girone | str, season_id: int, page: int = 0) -> list[dict]:
        payload = await self.client.get_json(self.endpoints.url_events_last(girone, season_id, page)) or {}
        return [e for e in payload.get("events", []) or [] if isinstance(e, dict)]

    async def statistics(self, event_id: int | str) -> dict:
        payload = await self.client.get_json(self.endpoints.url_event_statistics(event_id))
        return payload if isinstance(payload, dict) else {}

    # -------------------------------------------------------------- parsers
    @staticmethod
    def parse_event(event: dict, girone: Girone, matcher: TeamMatcher) -> MatchRecord | Fixture | None:
        """Convert one SofaScore event to a :class:`MatchRecord` (finished),
        a :class:`Fixture` (not started) or ``None`` (anything else)."""
        girone = Girone.parse(girone)
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
        home = _canonical_team(matcher, home_raw, girone, "sofascore")
        away = _canonical_team(matcher, away_raw, girone, "sofascore")
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
    async def _safe_statistics(self, match: MatchRecord) -> dict[str, float | None] | None:
        event_id = match.match_id.removeprefix("SS-")
        try:
            payload = await self.statistics(event_id)
        except HarvestError as exc:
            log.warning(
                "statistics unavailable, keeping match without stats",
                extra={"match_id": match.match_id, "status": exc.status, "attempts": exc.attempts, "url": exc.url},
            )
            return None
        return self.parse_statistics(payload, self.endpoints.stat_keys)

    async def _events_or_empty(self, girone: Girone, season_id: int, round_no: int) -> list[dict]:
        try:
            return await self.events_for_round(girone, season_id, round_no)
        except NotFoundError as exc:
            log.warning("round not found, skipped", extra={"girone": girone.value, "round": round_no, "url": exc.url})
            return []

    async def harvest(
        self,
        girone: Girone | str,
        *,
        season_id: int | None = None,
        rounds: Iterable[int] | None = None,
        with_stats: bool = True,
    ) -> tuple[list[MatchRecord], list[Fixture]]:
        """Played matches and upcoming fixtures of ``girone``.

        Statistics are fetched concurrently for finished matches; a failing
        statistics call keeps the match with ``None`` stats.
        """
        g = Girone.parse(girone)
        if season_id is None:
            season_id = await self.current_season_id(g)
        round_list = sorted({int(r) for r in rounds}) if rounds is not None else await self.rounds(g, season_id)
        batches = await asyncio.gather(*(self._events_or_empty(g, season_id, r) for r in round_list))

        matches: dict[str, MatchRecord] = {}
        fixtures: dict[str, Fixture] = {}
        for events in batches:
            for event in events:
                parsed = self.parse_event(event, g, self.matcher)
                if isinstance(parsed, MatchRecord):
                    matches[parsed.match_id] = parsed
                elif isinstance(parsed, Fixture):
                    fixtures[parsed.match_id] = parsed
        ordered = list(matches.values())
        if with_stats and ordered:
            stats = await asyncio.gather(*(self._safe_statistics(m) for m in ordered))
            ordered = [replace(m, **s) if s is not None else m for m, s in zip(ordered, stats, strict=True)]
        log.info(
            "sofascore harvest complete",
            extra={"girone": g.value, "season_id": season_id, "rounds": len(round_list), "matches": len(ordered), "fixtures": len(fixtures)},
        )
        return ordered, list(fixtures.values())


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
            stats = await asyncio.gather(*(self._safe_details(m) for m in ordered))
            ordered = [replace(m, **s) if s is not None else m for m, s in zip(ordered, stats, strict=True)]
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
    """Outcome of :meth:`Harvester.harvest_all`."""

    matches: pd.DataFrame
    fixtures: list[Fixture]
    unmapped_teams: dict[str, int]
    stats: dict[str, int]
    provider: str
    failures: dict[str, str] = field(default_factory=dict)

    def summary(self) -> str:
        """One-line human summary (used by the CLI)."""
        gironi = ",".join(sorted(self.matches["girone"].dropna().unique().tolist())) if len(self.matches) else "-"
        return (
            f"provider={self.provider} gironi={gironi} matches={len(self.matches)} fixtures={len(self.fixtures)} "
            f"unmapped_teams={len(self.unmapped_teams)} requests={self.stats.get('requests', 0)} "
            f"retries={self.stats.get('retries', 0)} cache_hits={self.stats.get('cache_hits', 0)} "
            f"errors={self.stats.get('errors', 0)} failures={len(self.failures)}"
        )


PROVIDER_NAMES: Final[tuple[str, ...]] = (SofaScoreProvider.name, FotMobProvider.name)


class Harvester:
    """Harvest every girone through the configured providers (ordered fallbacks).

    ``providers`` is tried in order for each girone; the first provider that
    succeeds wins, failures are logged and recorded in
    :attr:`HarvestResult.failures`. ``season_id`` applies to SofaScore only
    (FotMob exposes the current season). ``extra_headers`` reach every
    provider client (FotMob's ``x-mas`` signature header belongs there).
    """

    def __init__(
        self,
        *,
        endpoints: EndpointsConfig | None = None,
        registry: TeamRegistry | None = None,
        cache_dir: Path = RAW_DIR,
        cache_ttl: timedelta | None = None,
        providers: Sequence[str] = ("sofascore",),
        transport: httpx.AsyncBaseTransport | None = None,
        extra_headers: Mapping[str, str] | None = None,
        matcher: TeamMatcher | None = None,
        sleeper: Sleeper = asyncio.sleep,
        rng: random.Random | None = None,
    ) -> None:
        names = tuple(str(p).lower() for p in providers)
        unknown = [p for p in names if p not in PROVIDER_NAMES]
        if unknown or not names:
            raise ValueError(f"providers must be a non-empty subset of {PROVIDER_NAMES}, got {providers!r}")
        self.endpoints = endpoints if endpoints is not None else load_endpoints()
        self.registry = registry if registry is not None else load_registry()
        self.matcher = matcher if matcher is not None else TeamMatcher(self.registry)
        ttl = cache_ttl if cache_ttl is not None else self.endpoints.http.cache_ttl
        self.cache = RawCache(Path(cache_dir), ttl=ttl)
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

    def _base_url(self, name: str) -> str:
        return self.endpoints.sofascore.base_url if name == SofaScoreProvider.name else self.endpoints.fotmob.base_url

    def _count_unmapped(self, matches: Iterable[MatchRecord], fixtures: Iterable[Fixture]) -> dict[str, int]:
        canonical = {g: set(teams) for g, teams in self.registry.teams_by_girone.items()}
        counts: dict[str, int] = {}
        for record in (*matches, *fixtures):
            for name in (record.home_team, record.away_team):
                if name not in canonical.get(record.girone, set()):
                    counts[name] = counts.get(name, 0) + 1
        return dict(sorted(counts.items()))

    # ---------------------------------------------------------------- public
    async def harvest_all(
        self,
        gironi: Iterable[Girone] = tuple(Girone),
        *,
        season_id: int | None = None,
        with_stats: bool = True,
    ) -> HarvestResult:
        """Harvest ``gironi`` and return a validated canonical frame plus fixtures.

        Raises :class:`HarvestError` only when every girone failed on every
        provider; partial failures are reported in ``HarvestResult.failures``.
        """
        targets = [Girone.parse(g) for g in gironi]
        clients: dict[str, AsyncHarvesterClient] = {}
        providers: dict[str, SofaScoreProvider | FotMobProvider] = {}
        all_matches: list[MatchRecord] = []
        all_fixtures: list[Fixture] = []
        used: list[str] = []
        failures: dict[str, str] = {}
        last_error: HarvestError | None = None
        try:
            for girone in targets:
                success = False
                for name in self.providers:
                    if name not in clients:
                        clients[name] = self._make_client(self._base_url(name))
                        providers[name] = self._make_provider(name, clients[name])
                    provider = providers[name]
                    try:
                        if isinstance(provider, SofaScoreProvider):
                            matches, fixtures = await provider.harvest(girone, season_id=season_id, with_stats=with_stats)
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
        for client in clients.values():
            totals.add(client.stats)
        seen: set[str] = set()
        fixtures_unique = [f for f in all_fixtures if not (f.match_id in seen or seen.add(f.match_id))]
        frame = matches_to_frame(all_matches) if all_matches else empty_matches_frame()
        result = HarvestResult(
            matches=frame,
            fixtures=fixtures_unique,
            unmapped_teams=self._count_unmapped(all_matches, fixtures_unique),
            stats=totals.as_dict(),
            provider="+".join(used) if used else self.providers[0],
            failures=failures,
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
        """Read fixtures written by :meth:`save_fixtures`."""
        path = Path(path)
        try:
            raw = pd.read_csv(path, dtype="string", keep_default_na=False)
        except pd.errors.EmptyDataError:
            return []
        missing = [c for c in FIXTURE_COLUMNS if c not in raw.columns]
        if missing:
            raise ValueError(f"fixtures file {path} is missing columns {missing}")
        out: list[Fixture] = []
        for row in raw.itertuples(index=False):
            out.append(
                Fixture(
                    match_id=str(row.match_id),
                    date=date.fromisoformat(str(row.date)[:10]),
                    round=_as_int(row.round, 0),
                    girone=Girone.parse(str(row.girone)),
                    home_team=str(row.home_team),
                    away_team=str(row.away_team),
                )
            )
        return out


# =============================================================================
# CLI
# =============================================================================
def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m src.ingestion.harvester",
        description="Harvest Serie C matches and fixtures from SofaScore / FotMob into the canonical CSVs.",
    )
    parser.add_argument("--gironi", nargs="+", default=["A", "B", "C"], choices=["A", "B", "C"], help="gironi to harvest")
    parser.add_argument("--season-id", type=int, default=None, help="SofaScore season id (default: current)")
    parser.add_argument("--out", type=Path, default=PROCESSED_DIR / "matches.csv", help="matches CSV output")
    parser.add_argument("--fixtures-out", type=Path, default=PROCESSED_DIR / "fixtures.csv", help="fixtures CSV output")
    parser.add_argument("--no-stats", action="store_true", help="skip per-match statistics calls")
    parser.add_argument("--log-json", action="store_true", help="emit JSON-lines logs")
    parser.add_argument("--providers", nargs="+", default=["sofascore"], choices=list(PROVIDER_NAMES), help="ordered provider fallbacks")
    parser.add_argument("--cache-dir", type=Path, default=RAW_DIR, help="raw JSON cache directory")
    parser.add_argument("--cache-ttl-hours", type=float, default=None, help="override cache TTL (hours)")
    parser.add_argument("--endpoints", type=Path, default=CONFIG_DIR / "endpoints.toml", help="endpoints TOML file")
    parser.add_argument("--x-mas", default=os.getenv("FOTMOB_X_MAS"), help="FotMob x-mas signature header (env FOTMOB_X_MAS)")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entry point; returns the process exit code."""
    args = _build_parser().parse_args(argv)
    configure_logging(json_output=args.log_json, force=True)
    extra_headers = {"x-mas": args.x_mas} if args.x_mas else None
    harvester = Harvester(
        endpoints=load_endpoints(args.endpoints),
        cache_dir=args.cache_dir,
        cache_ttl=timedelta(hours=args.cache_ttl_hours) if args.cache_ttl_hours is not None else None,
        providers=tuple(args.providers),
        extra_headers=extra_headers,
    )
    gironi = [Girone.parse(g) for g in args.gironi]
    try:
        result = asyncio.run(harvester.harvest_all(gironi, season_id=args.season_id, with_stats=not args.no_stats))
    except HarvestError as exc:
        log.error("harvest failed", extra={"url": exc.url, "status": exc.status, "attempts": exc.attempts})
        print(f"harvest failed: {exc}", file=sys.stderr)
        return 1
    Harvester.save_processed(result.matches, args.out)
    Harvester.save_fixtures(result.fixtures, args.fixtures_out)
    print(result.summary())
    print(f"matches -> {args.out}")
    print(f"fixtures -> {args.fixtures_out}")
    if result.unmapped_teams:
        print("unmapped teams: " + ", ".join(f"{name} x{count}" for name, count in result.unmapped_teams.items()))
    for key, message in result.failures.items():
        print(f"failure {key}: {message}", file=sys.stderr)
    return 0 if not result.failures or len(result.matches) else 1


if __name__ == "__main__":  # pragma: no cover - manual entry point
    raise SystemExit(main())
