"""Offline tests for :mod:`src.ingestion.harvester` (mock transports, no sleeping).

The girone A fixtures are 2025/26 SofaScore payloads: the single-season tests use
the 2025/26 dictionary (``LEGACY_SEASON``) as their perimeter so they do not depend
on :data:`src.common.teams.DEFAULT_SEASON`. The multi-season tests (girone B, four
listed seasons) use the current 2026/27 perimeter with the 2025/26 aliases merged
in, the configuration the CLI runs with by default.
"""
from __future__ import annotations

import asyncio
import dataclasses
import importlib.util
import json
import logging
import random
import re
import shutil
from collections import Counter
from collections.abc import Callable
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import httpx
import pandas as pd
import pytest

from src.common.schemas import MATCH_COLUMNS, MATCH_DTYPES, Fixture, Girone, MatchRecord, season_of
from src.common.teams import DEFAULT_SEASON, TeamRegistry, load_registry
from src.ingestion import harvester as hv
from src.ingestion.harvester import (
    DEFAULT_USER_AGENTS,
    AsyncHarvesterClient,
    FotMobEndpoints,
    FotMobProvider,
    HarvestError,
    HarvestResult,
    Harvester,
    HttpConfig,
    NotFoundError,
    RateLimitedError,
    RawCache,
    SeasonHarvest,
    SeasonInfo,
    SofaScoreProvider,
    load_endpoints,
    parse_retry_after,
    parse_stat_value,
    season_label_from_year,
)
from src.ingestion.team_matcher import TeamMatcher

FIXTURES = Path(__file__).parent / "fixtures" / "sofascore"
SEASON_ID = 77030
TOURNAMENT_A = 141
LEGACY_SEASON = "2025_26"  # dictionary matching the girone A fixture payloads
CURRENT_SEASON = "2026_27"  # the perimeter (DEFAULT_SEASON)
BROKEN_STATS_EVENT = 13579103  # statistics endpoint answers 500 every time
MISSING_STATS_EVENT = 13579113  # statistics endpoint answers 404
UTC = timezone.utc


# ============================================================================ helpers
class RecordingSleeper:
    """Async sleeper that records delays and never waits."""

    def __init__(self) -> None:
        self.calls: list[float] = []

    async def __call__(self, delay: float) -> None:
        self.calls.append(delay)


class CountingTransport(httpx.MockTransport):
    """MockTransport that counts calls and keeps every request."""

    def __init__(self, handler: Callable[[httpx.Request], Any]) -> None:
        super().__init__(handler)
        self.calls = 0
        self.requests: list[httpx.Request] = []

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        self.calls += 1
        self.requests.append(request)
        return await super().handle_async_request(request)


def json_response(payload: Any, status: int = 200, headers: dict[str, str] | None = None) -> httpx.Response:
    return httpx.Response(status, json=payload, headers=headers)


def sequence_handler(*factories: Callable[[], httpx.Response | Exception]) -> Callable[[httpx.Request], httpx.Response]:
    """Serve responses in order; the last factory repeats forever."""
    state = {"i": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        factory = factories[min(state["i"], len(factories) - 1)]
        state["i"] += 1
        outcome = factory()
        if isinstance(outcome, Exception):
            raise outcome
        return outcome

    return handler


def load_fixture(name: str) -> Any:
    return json.loads((FIXTURES / name).read_text(encoding="utf-8"))


def sofascore_router(request: httpx.Request) -> httpx.Response:
    """Route SofaScore-shaped paths to the fixture files."""
    path = request.url.path
    if path == f"/api/v1/unique-tournament/{TOURNAMENT_A}/seasons":
        return json_response(load_fixture("seasons_girone_a.json"))
    if path == f"/api/v1/unique-tournament/{TOURNAMENT_A}/season/{SEASON_ID}/rounds":
        return json_response(load_fixture("rounds_girone_a.json"))
    found = re.fullmatch(rf"/api/v1/unique-tournament/{TOURNAMENT_A}/season/{SEASON_ID}/events/round/(\d+)", path)
    if found:
        file = FIXTURES / f"events_round_{found.group(1)}.json"
        return json_response(load_fixture(file.name)) if file.exists() else json_response({"events": []})
    found = re.fullmatch(r"/api/v1/event/(\d+)/statistics", path)
    if found:
        event_id = int(found.group(1))
        if event_id == BROKEN_STATS_EVENT:
            return json_response({"error": {"code": 500, "message": "internal"}}, status=500)
        file = FIXTURES / f"statistics_{event_id}.json"
        if file.exists():
            return json_response(load_fixture(file.name))
        return json_response({"error": {"code": 404}}, status=404)
    return json_response({"error": {"code": 404, "message": f"no route for {path}"}}, status=404)


@pytest.fixture(autouse=True)
def _restore_root_logging() -> Any:
    """``hv.main`` reconfigures the root logger onto the captured stderr of its test;
    put the previous handlers back so later tests do not log to a closed stream."""
    root = logging.getLogger()
    handlers, level = list(root.handlers), root.level
    yield
    for handler in list(root.handlers):
        root.removeHandler(handler)
    for handler in handlers:
        root.addHandler(handler)
    root.setLevel(level)


@pytest.fixture(autouse=True)
def _no_footystats_key(monkeypatch: pytest.MonkeyPatch) -> None:
    """A developer's FootyStats key (environment or ``.env``) must not change the default provider
    order these SofaScore tests rely on; the facade tests pass their key explicitly."""
    from src.ingestion import footystats  # noqa: PLC0415

    def explicit_only(explicit: str | None = None, **kwargs: Any) -> str | None:
        return explicit.strip() if explicit and explicit.strip() else None

    monkeypatch.delenv("FOOTYSTATS_API_KEY", raising=False)
    monkeypatch.setattr(footystats, "find_api_key", explicit_only)


@pytest.fixture
def http_cfg() -> HttpConfig:
    return HttpConfig(
        timeout_seconds=5.0,
        max_retries=3,
        backoff_base_seconds=1.0,
        backoff_max_seconds=60.0,
        max_concurrency=2,
        min_delay_seconds=0.0,
        cache_ttl_hours=1.0,
    )


@pytest.fixture(scope="module")
def registry() -> TeamRegistry:
    return load_registry(LEGACY_SEASON)


@pytest.fixture(scope="module")
def matcher(registry: TeamRegistry) -> TeamMatcher:
    return TeamMatcher(registry)


@pytest.fixture(scope="module")
def current_registry() -> TeamRegistry:
    return load_registry(CURRENT_SEASON)


def make_client(
    handler: Callable[[httpx.Request], Any],
    http: HttpConfig,
    *,
    base_url: str = "https://api.sofascore.com",
    cache: RawCache | None = None,
    seed: int = 7,
    **kwargs: Any,
) -> tuple[AsyncHarvesterClient, CountingTransport, RecordingSleeper]:
    transport = CountingTransport(handler)
    sleeper = RecordingSleeper()
    client = AsyncHarvesterClient(
        base_url=base_url, http=http, cache=cache, transport=transport, sleeper=sleeper, rng=random.Random(seed), **kwargs
    )
    return client, transport, sleeper


# ============================================================================ config
def test_load_endpoints_parses_toml() -> None:
    cfg = load_endpoints()
    assert cfg.sofascore.tournament_ids == {Girone.A: 141, Girone.B: 142, Girone.C: 143}
    assert cfg.sofascore.stat_keys["xg"] == ("expectedGoals", "Expected goals")
    assert cfg.http.max_retries == 5 and cfg.http.max_concurrency == 4 and cfg.http.cache_ttl == timedelta(hours=6)
    assert cfg.sofascore.url_events_by_round(Girone.A, 77030, 3) == (
        "https://api.sofascore.com/api/v1/unique-tournament/141/season/77030/events/round/3"
    )
    assert cfg.sofascore.url_seasons("B").endswith("/unique-tournament/142/seasons")
    assert cfg.sofascore.url_events_next(Girone.C, 1, page=2).endswith("/season/1/events/next/2")
    assert cfg.sofascore.url_event_statistics(42).endswith("/event/42/statistics")
    assert cfg.fotmob.url_league(10) == "https://www.fotmob.com/api/leagues?id=10&tab=matches&type=league"
    assert cfg.fotmob.url_match_details("4521234").endswith("matchId=4521234")
    assert cfg.fotmob.configured_league_ids() == {}
    assert cfg.sofascore.tournament_id("A") == 141
    unconfigured = hv.SofaScoreEndpoints(
        base_url="x", tournament_ids={}, seasons="", rounds="", events_by_round="", events_last="", events_next="", event_statistics=""
    )
    with pytest.raises(KeyError):
        unconfigured.tournament_id("A")


def test_load_endpoints_missing_key(tmp_path: Path) -> None:
    bad = tmp_path / "endpoints.toml"
    bad.write_text('[sofascore]\nbase_url = "https://x"\n[fotmob]\nbase_url = "https://y"\n', encoding="utf-8")
    with pytest.raises(ValueError, match="tournament_ids"):
        load_endpoints(bad)


def test_http_config_validation() -> None:
    with pytest.raises(ValueError):
        HttpConfig(max_concurrency=0)
    with pytest.raises(ValueError):
        HttpConfig(timeout_seconds=0)


def test_site_origin() -> None:
    assert hv.site_origin("https://api.sofascore.com") == "https://www.sofascore.com"
    assert hv.site_origin("https://www.fotmob.com") == "https://www.fotmob.com"
    assert hv.site_origin("http://testserver") == "http://testserver"


# ============================================================================ client
@pytest.mark.asyncio
async def test_user_agent_rotation(http_cfg: HttpConfig) -> None:
    client, transport, _ = make_client(lambda request: json_response({"ok": True}), http_cfg)
    async with client:
        for i in range(20):
            await client.get_json(f"/ping/{i}", use_cache=False)
    agents = [request.headers["User-Agent"] for request in transport.requests]
    assert len(agents) == 20
    assert len(set(agents)) >= 3
    assert all(a != b for a, b in zip(agents, agents[1:]))
    assert set(agents) <= set(DEFAULT_USER_AGENTS)
    assert client.last_user_agent == agents[-1]
    assert len(DEFAULT_USER_AGENTS) >= 8


@pytest.mark.asyncio
async def test_realistic_headers_and_extra_headers(http_cfg: HttpConfig) -> None:
    client, transport, _ = make_client(
        lambda request: json_response({}), http_cfg, base_url="https://www.fotmob.com", extra_headers={"x-mas": "signed"}
    )
    async with client:
        await client.get_json("/api/allLeagues", use_cache=False)
    headers = transport.requests[0].headers
    assert headers["Accept"] == "application/json"
    # Accept-Encoding is left to httpx: brotli is advertised only when it can be decoded.
    assert "br" not in headers["Accept-Encoding"] or importlib.util.find_spec("brotli") is not None
    assert headers["Accept-Language"] == "it-IT,it;q=0.9,en;q=0.8"
    assert headers["Referer"] == "https://www.fotmob.com/"
    assert headers["Origin"] == "https://www.fotmob.com"
    assert headers["Cache-Control"] == "no-cache"
    assert headers["x-mas"] == "signed"


@pytest.mark.asyncio
async def test_backoff_on_429_then_success(http_cfg: HttpConfig) -> None:
    handler = sequence_handler(
        lambda: json_response({"error": 429}, status=429),
        lambda: json_response({"error": 429}, status=429),
        lambda: json_response({"ok": 1}),
    )
    client, transport, sleeper = make_client(handler, http_cfg)
    async with client:
        payload = await client.get_json("/x", use_cache=False)
    assert payload == {"ok": 1}
    assert transport.calls == 3
    assert client.stats.requests == 3 and client.stats.retries == 2 and client.stats.errors == 0
    assert len(sleeper.calls) == 2
    base = http_cfg.backoff_base_seconds
    for k, delay in enumerate(sleeper.calls):
        assert 0.5 * base * 2**k <= delay <= base * 2**k


@pytest.mark.asyncio
async def test_retry_after_header_is_honoured(http_cfg: HttpConfig) -> None:
    handler = sequence_handler(
        lambda: json_response({}, status=429, headers={"Retry-After": "7"}),
        lambda: json_response({"ok": 1}),
    )
    client, _, sleeper = make_client(handler, http_cfg)
    async with client:
        assert await client.get_json("/x", use_cache=False) == {"ok": 1}
    assert sleeper.calls and sleeper.calls[0] >= 7


def test_parse_retry_after() -> None:
    now = datetime(2025, 10, 21, 7, 27, 30, tzinfo=UTC)
    assert parse_retry_after("3") == 3.0
    assert parse_retry_after("  12.5 ") == 12.5
    assert parse_retry_after("Tue, 21 Oct 2025 07:28:00 GMT", now=now) == 30.0
    assert parse_retry_after("Tue, 21 Oct 2025 07:00:00 GMT", now=now) == 0.0
    assert parse_retry_after("garbage") is None
    assert parse_retry_after(None) is None and parse_retry_after("") is None


@pytest.mark.asyncio
async def test_backoff_is_capped(http_cfg: HttpConfig) -> None:
    cfg = dataclasses.replace(http_cfg, backoff_max_seconds=1.5, max_retries=4)
    client, _, sleeper = make_client(sequence_handler(lambda: json_response({}, status=503)), cfg)
    async with client:
        with pytest.raises(HarvestError):
            await client.get_json("/x", use_cache=False)
    assert len(sleeper.calls) == 4
    assert all(0.5 <= d <= 1.5 for d in sleeper.calls)


@pytest.mark.asyncio
async def test_server_errors_exhaust_retries(http_cfg: HttpConfig) -> None:
    client, transport, sleeper = make_client(sequence_handler(lambda: json_response({"e": 1}, status=500)), http_cfg)
    async with client:
        with pytest.raises(HarvestError) as info:
            await client.get_json("/x", use_cache=False)
    exc = info.value
    assert not isinstance(exc, RateLimitedError)
    assert exc.attempts == http_cfg.max_retries + 1 == transport.calls
    assert exc.status == 500 and exc.url == "https://api.sofascore.com/x"
    assert client.stats.retries == http_cfg.max_retries and client.stats.errors == 1
    assert len(sleeper.calls) == http_cfg.max_retries


@pytest.mark.asyncio
async def test_persistent_429_raises_rate_limited(http_cfg: HttpConfig) -> None:
    client, transport, _ = make_client(sequence_handler(lambda: json_response({}, status=429)), http_cfg)
    async with client:
        with pytest.raises(RateLimitedError) as info:
            await client.get_json("/x", use_cache=False)
    assert info.value.status == 429 and info.value.attempts == http_cfg.max_retries + 1 == transport.calls


@pytest.mark.asyncio
async def test_404_raises_not_found_without_retry(http_cfg: HttpConfig) -> None:
    client, transport, sleeper = make_client(lambda request: json_response({"error": 404}, status=404), http_cfg)
    async with client:
        with pytest.raises(NotFoundError) as info:
            await client.get_json("/missing", use_cache=False)
    assert info.value.status == 404 and info.value.attempts == 1
    assert transport.calls == 1 and sleeper.calls == [] and client.stats.retries == 0 and client.stats.errors == 1


@pytest.mark.asyncio
async def test_403_raises_without_retry(http_cfg: HttpConfig) -> None:
    client, transport, sleeper = make_client(lambda request: json_response({"error": 403}, status=403), http_cfg)
    async with client:
        with pytest.raises(HarvestError) as info:
            await client.get_json("/forbidden", use_cache=False)
    assert type(info.value) is HarvestError and info.value.status == 403
    assert transport.calls == 1 and sleeper.calls == []


@pytest.mark.asyncio
async def test_invalid_json_body(http_cfg: HttpConfig) -> None:
    client, transport, sleeper = make_client(lambda request: httpx.Response(200, text="<html>blocked</html>"), http_cfg)
    async with client:
        with pytest.raises(HarvestError) as info:
            await client.get_json("/x", use_cache=False)
    assert info.value.status == 200 and transport.calls == 1 and sleeper.calls == []


@pytest.mark.asyncio
async def test_non_transient_httpx_errors_are_wrapped_without_retry(http_cfg: HttpConfig) -> None:
    client, transport, sleeper = make_client(sequence_handler(lambda: httpx.DecodingError("unsupported encoding")), http_cfg)
    async with client:
        with pytest.raises(HarvestError) as info:
            await client.get_json("/x", use_cache=False)
    assert type(info.value) is HarvestError and info.value.status is None and info.value.attempts == 1
    assert "DecodingError" in str(info.value) and transport.calls == 1 and sleeper.calls == []


@pytest.mark.asyncio
async def test_transport_errors_are_retried(http_cfg: HttpConfig) -> None:
    handler = sequence_handler(
        lambda: httpx.ConnectError("connection reset"),
        lambda: httpx.ReadTimeout("timed out"),
        lambda: json_response({"ok": 1}),
    )
    client, transport, sleeper = make_client(handler, http_cfg)
    async with client:
        assert await client.get_json("/x", use_cache=False) == {"ok": 1}
    assert transport.calls == 3 and client.stats.retries == 2 and len(sleeper.calls) == 2


@pytest.mark.asyncio
async def test_retry_logs_structured_fields(http_cfg: HttpConfig, caplog: pytest.LogCaptureFixture) -> None:
    handler = sequence_handler(lambda: json_response({}, status=502), lambda: json_response({"ok": 1}))
    client, _, _ = make_client(handler, http_cfg)
    with caplog.at_level(logging.WARNING, logger="src.ingestion.harvester"):
        async with client:
            await client.get_json("/x", use_cache=False)
    records = [r for r in caplog.records if r.getMessage() == "retrying request"]
    assert len(records) == 1
    record = records[0]
    assert record.url == "https://api.sofascore.com/x" and record.status == 502 and record.attempt == 1 and record.delay > 0


@pytest.mark.asyncio
async def test_min_delay_between_requests(http_cfg: HttpConfig) -> None:
    cfg = dataclasses.replace(http_cfg, min_delay_seconds=0.4)
    frozen_clock = {"now": 1000.0}
    client, transport, sleeper = make_client(lambda request: json_response({}), cfg, clock=lambda: frozen_clock["now"])
    async with client:
        await client.get_json("/a", use_cache=False)
        await client.get_json("/b", use_cache=False)  # starts "immediately": must wait the full 0.4 s
        frozen_clock["now"] += 10.0
        await client.get_json("/c", use_cache=False)  # enough time elapsed: no wait
    assert transport.calls == 3
    assert sleeper.calls == [pytest.approx(0.4)]

    unpaced, _, quiet = make_client(lambda request: json_response({}), http_cfg)  # min_delay_seconds == 0
    async with unpaced:
        await unpaced.get_json("/a", use_cache=False)
        await unpaced.get_json("/b", use_cache=False)
    assert quiet.calls == []


@pytest.mark.asyncio
async def test_semaphore_bounds_concurrency(http_cfg: HttpConfig) -> None:
    state = {"inflight": 0, "peak": 0}

    async def handler(request: httpx.Request) -> httpx.Response:
        state["inflight"] += 1
        state["peak"] = max(state["peak"], state["inflight"])
        await asyncio.sleep(0.005)
        state["inflight"] -= 1
        return json_response({"ok": 1})

    client, transport, _ = make_client(handler, http_cfg)
    async with client:
        await asyncio.gather(*(client.get_json(f"/c/{i}", use_cache=False) for i in range(6)))
    assert transport.calls == 6
    assert 1 <= state["peak"] <= http_cfg.max_concurrency


@pytest.mark.asyncio
async def test_context_manager_closes_client(http_cfg: HttpConfig) -> None:
    client, _, _ = make_client(lambda request: json_response({}), http_cfg)
    async with client as entered:
        assert entered is client and client._client is not None
    assert client._client is None
    await client.close()  # idempotent
    with pytest.raises(ValueError):
        AsyncHarvesterClient(base_url="https://x", http=http_cfg, user_agents=[])


# ============================================================================ cache
@pytest.mark.asyncio
async def test_cache_hit_skips_transport(http_cfg: HttpConfig, tmp_path: Path) -> None:
    cache = RawCache(tmp_path / "raw")
    client, transport, _ = make_client(lambda request: json_response({"n": 1}), http_cfg, cache=cache)
    async with client:
        first = await client.get_json("/x", params={"b": 2, "a": 1})
        second = await client.get_json("/x", params={"a": 1, "b": 2})
    assert first == second == {"n": 1}
    assert transport.calls == 1
    assert client.stats.cache_hits == 1 and client.stats.requests == 1
    assert cache.stats == {"hits": 1, "misses": 1}
    stored = json.loads(cache.path_for("https://api.sofascore.com/x", {"a": 1, "b": 2}).read_text(encoding="utf-8"))
    assert stored["payload"] == {"n": 1}
    assert stored["url"] == "https://api.sofascore.com/x?a=1&b=2"
    datetime.fromisoformat(stored["fetched_at"])  # ISO timestamp
    assert len(cache) == 1 and not list((tmp_path / "raw").glob("*.tmp"))


@pytest.mark.asyncio
async def test_cache_ttl_expiry_refetches(http_cfg: HttpConfig, tmp_path: Path) -> None:
    cache = RawCache(tmp_path, ttl=timedelta(0))
    client, transport, _ = make_client(lambda request: json_response({"n": 1}), http_cfg, cache=cache)
    async with client:
        await client.get_json("/x")
        await client.get_json("/x")
    assert transport.calls == 2 and client.stats.cache_hits == 0
    forever = RawCache(tmp_path / "forever", ttl=None)
    forever.put("https://a/b", {"v": 1})
    assert forever.get("https://a/b") == {"v": 1}


def test_cache_corrupt_file_is_miss(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    cache = RawCache(tmp_path)
    cache.put("https://a/b", {"v": 1})
    cache.path_for("https://a/b").write_text("{not json", encoding="utf-8")
    with caplog.at_level(logging.WARNING, logger="src.ingestion.harvester"):
        assert cache.get("https://a/b") is None
    assert cache.misses == 1 and cache.hits == 0
    assert any("corrupt" in r.getMessage() for r in caplog.records)
    cache.path_for("https://a/b").write_text(json.dumps({"no_payload": True}), encoding="utf-8")
    assert cache.get("https://a/b") is None


def test_cache_key_ignores_param_order() -> None:
    url = "https://api.sofascore.com/api/v1/x"
    assert RawCache.key_for(url, {"a": 1, "b": 2}) == RawCache.key_for(url, {"b": 2, "a": 1})
    assert RawCache.key_for(url + "?b=2&a=1") == RawCache.key_for(url, {"a": 1, "b": 2})
    assert RawCache.key_for(url, {"a": 1}) != RawCache.key_for(url, {"a": 2})
    assert RawCache.canonical_url(url, {"page": [2, 1], "x": None}) == url + "?page=1&page=2"
    assert re.fullmatch(r"[0-9a-f]{40}", RawCache.key_for(url))


def test_cache_put_and_clear(tmp_path: Path) -> None:
    cache = RawCache(tmp_path / "nested" / "dir")
    path = cache.put("https://a/1", [1, 2, 3])
    cache.put("https://a/2", {"k": "v"})
    assert path.exists() and path.parent == tmp_path / "nested" / "dir"
    assert cache.get("https://a/1") == [1, 2, 3]
    assert cache.clear() == 2
    assert len(cache) == 0 and cache.get("https://a/1") is None


# ============================================================================ parsing
def test_parse_stat_value() -> None:
    assert parse_stat_value("12") == 12.0
    assert parse_stat_value("1.23") == 1.23
    assert parse_stat_value("1,23") == 1.23
    assert parse_stat_value("55%") == 55.0
    assert parse_stat_value("7 (3)") == 7.0
    assert parse_stat_value(9) == 9.0 and parse_stat_value(0.5) == 0.5
    assert parse_stat_value(None) is None and parse_stat_value("") is None and parse_stat_value("-") is None
    assert parse_stat_value(float("nan")) is None and parse_stat_value(True) is None


def test_parse_event_finished(matcher: TeamMatcher) -> None:
    event = load_fixture("events_round_1.json")["events"][0]
    record = SofaScoreProvider.parse_event(event, Girone.A, matcher)
    assert isinstance(record, MatchRecord)
    assert record.match_id == "SS-13579101"
    assert record.date == date(2025, 8, 23) == datetime.fromtimestamp(1755973800, tz=UTC).date()
    assert record.round == 1 and record.girone is Girone.A
    assert record.home_team == "Pro Vercelli" and record.away_team == "Vicenza"  # 'L.R. Vicenza' canonicalised
    assert (record.home_goals, record.away_goals) == (0, 2)
    assert record.home_xg is None and record.home_shots is None


def test_parse_event_utc_date_boundary(matcher: TeamMatcher) -> None:
    event = load_fixture("events_round_1.json")["events"][0] | {"startTimestamp": 1756593000}  # 2025-08-30T22:30Z
    record = SofaScoreProvider.parse_event(event, Girone.A, matcher)
    assert record is not None and record.date == date(2025, 8, 30)


def test_parse_event_notstarted_postponed_inprogress(matcher: TeamMatcher) -> None:
    round_two = load_fixture("events_round_2.json")["events"]
    round_three = load_fixture("events_round_3.json")["events"]
    fixture = SofaScoreProvider.parse_event(round_three[0], Girone.A, matcher)
    assert isinstance(fixture, Fixture)
    assert fixture == Fixture("SS-13579121", date(2025, 9, 6), 3, Girone.A, "AlbinoLeffe", "Lumezzane")
    assert SofaScoreProvider.parse_event(round_two[1], Girone.A, matcher) is None  # postponed
    assert SofaScoreProvider.parse_event(round_three[2], Girone.A, matcher) is None  # inprogress
    canceled = round_two[1] | {"status": {"type": "canceled", "code": 70}}
    assert SofaScoreProvider.parse_event(canceled, Girone.A, matcher) is None


def test_parse_event_edge_cases(matcher: TeamMatcher, caplog: pytest.LogCaptureFixture) -> None:
    finished = load_fixture("events_round_2.json")["events"][2]
    with caplog.at_level(logging.WARNING, logger="src.ingestion.harvester"):
        record = SofaScoreProvider.parse_event(finished, Girone.A, matcher)
    assert isinstance(record, MatchRecord)
    assert record.home_team == "Cittadella" and record.away_team == "Sestri Levante"  # unmapped kept raw
    assert any(r.getMessage() == "unmapped team name kept raw" and r.team == "Sestri Levante" for r in caplog.records)

    no_round = {k: v for k, v in finished.items() if k != "roundInfo"}
    assert SofaScoreProvider.parse_event(no_round, Girone.A, matcher).round == 0
    no_scores = finished | {"homeScore": {}, "awayScore": {}}
    assert SofaScoreProvider.parse_event(no_scores, Girone.A, matcher) is None
    assert SofaScoreProvider.parse_event(finished | {"startTimestamp": None}, Girone.A, matcher) is None
    assert SofaScoreProvider.parse_event({"status": {"type": "finished"}}, Girone.A, matcher) is None
    same_team = finished | {"awayTeam": finished["homeTeam"]}
    assert SofaScoreProvider.parse_event(same_team, Girone.A, matcher) is None


def test_parse_statistics_numeric_fields_and_period_all() -> None:
    stat_keys = load_endpoints().sofascore.stat_keys
    stats = SofaScoreProvider.parse_statistics(load_fixture("statistics_13579101.json"), stat_keys)
    assert stats == {
        "home_shots": 9.0,
        "away_shots": 14.0,
        "home_shots_target": 2.0,
        "away_shots_target": 6.0,
        "home_corners": 3.0,
        "away_corners": 6.0,
        "home_xg": 0.87,
        "away_xg": 1.64,
    }


def test_parse_statistics_strings_and_missing_xg() -> None:
    stat_keys = load_endpoints().sofascore.stat_keys
    stats = SofaScoreProvider.parse_statistics(load_fixture("statistics_13579102.json"), stat_keys)
    assert stats["home_shots"] == 12.0 and stats["away_shots"] == 7.0  # parsed from strings (no homeValue)
    assert stats["home_shots_target"] == 4.0 and stats["away_shots_target"] == 3.0
    assert stats["home_corners"] == 5.0 and stats["away_corners"] == 2.0
    assert stats["home_xg"] is None and stats["away_xg"] is None


def test_parse_statistics_name_lookup_percentages_and_odd_strings() -> None:
    payload = {
        "statistics": [
            {"period": "1ST", "groups": [{"groupName": "Shots", "statisticsItems": [{"name": "Total shots", "home": "1", "away": "1", "key": "totalShotsOnGoal"}]}]},
            {
                "period": "ALL",
                "groups": [
                    {
                        "groupName": "Shots",
                        "statisticsItems": [
                            {"name": "Total shots", "home": "7 (3)", "away": "55%"},
                            {"name": "Shots on target", "home": "3", "away": "2", "key": "ShotsOnGoal"},
                            {"name": "Expected goals", "home": "1,10", "away": "-", "key": "expectedGoals"},
                        ],
                    }
                ],
            },
        ]
    }
    stat_keys = {"shots": ("totalShotsOnGoal", "Total shots"), "shots_target": ("shotsOnGoal",), "corners": ("cornerKicks",), "xg": ("expectedGoals",)}
    stats = SofaScoreProvider.parse_statistics(payload, stat_keys)
    assert stats["home_shots"] == 7.0 and stats["away_shots"] == 55.0  # located by name, odd strings parsed
    assert stats["home_shots_target"] == 3.0 and stats["away_shots_target"] == 2.0  # case-insensitive key
    assert stats["home_corners"] is None and stats["away_corners"] is None
    assert stats["home_xg"] == 1.1 and stats["away_xg"] is None
    assert set(stats) == set(hv.STAT_FIELDS)
    assert all(v is None for v in SofaScoreProvider.parse_statistics({"statistics": [{"period": "1ST", "groups": []}]}, stat_keys).values())
    assert all(v is None for v in SofaScoreProvider.parse_statistics({}, stat_keys).values())


# ============================================================================ SofaScore provider
@pytest.mark.asyncio
async def test_sofascore_provider_navigation(http_cfg: HttpConfig, matcher: TeamMatcher) -> None:
    endpoints = load_endpoints().sofascore
    client, transport, _ = make_client(sofascore_router, http_cfg)
    provider = SofaScoreProvider(client, endpoints, matcher)
    async with client:
        seasons = await provider.seasons(Girone.A)
        assert seasons[0] == SeasonInfo(77030, "25/26", "Serie C, Girone A 25/26") and len(seasons) == 3
        assert await provider.current_season_id("A") == SEASON_ID
        assert await provider.rounds(Girone.A, SEASON_ID) == [1, 2, 3]  # playoff round skipped
        assert await provider.rounds(Girone.A, SEASON_ID, regular_only=False) == [1, 2, 3, 39]
        events = await provider.events_for_round(Girone.A, SEASON_ID, 1)
        assert [e["id"] for e in events] == [13579101, 13579102, 13579103]
        assert await provider.events_for_round(Girone.A, SEASON_ID, 9) == []
        stats = await provider.statistics(13579101)
        assert stats["statistics"][0]["period"] == "ALL"
        with pytest.raises(NotFoundError):
            await provider.statistics(MISSING_STATS_EVENT)
    assert transport.calls == 8  # seasons x2 (no cache), rounds x2, events x2, statistics x2


@pytest.mark.asyncio
async def test_sofascore_provider_rounds_fallback_to_current_round(http_cfg: HttpConfig, matcher: TeamMatcher) -> None:
    client, _, _ = make_client(lambda request: json_response({"currentRound": {"round": 4}}), http_cfg)
    provider = SofaScoreProvider(client, load_endpoints().sofascore, matcher)
    async with client:
        assert await provider.rounds(Girone.B, 1) == [1, 2, 3, 4]


@pytest.mark.asyncio
async def test_sofascore_harvest_end_to_end_keeps_match_when_stats_fail(
    http_cfg: HttpConfig, matcher: TeamMatcher, caplog: pytest.LogCaptureFixture
) -> None:
    endpoints = load_endpoints().sofascore
    client, transport, sleeper = make_client(sofascore_router, http_cfg)
    provider = SofaScoreProvider(client, endpoints, matcher)
    with caplog.at_level(logging.WARNING, logger="src.ingestion.harvester"):
        async with client:
            matches, fixtures = await provider.harvest(Girone.A)
    by_id = {m.match_id: m for m in matches}
    assert sorted(by_id) == ["SS-13579101", "SS-13579102", "SS-13579103", "SS-13579111", "SS-13579113"]
    assert [f.match_id for f in fixtures] == ["SS-13579121", "SS-13579122"]
    assert by_id["SS-13579101"].home_xg == 0.87 and by_id["SS-13579101"].away_shots == 14.0
    assert by_id["SS-13579111"].home_shots == 18.0 and by_id["SS-13579111"].away_xg == 0.52
    broken = by_id[f"SS-{BROKEN_STATS_EVENT}"]
    assert broken.home_goals == 2 and broken.away_goals == 0
    assert all(getattr(broken, f) is None for f in hv.STAT_FIELDS)
    missing = by_id[f"SS-{MISSING_STATS_EVENT}"]
    assert missing.away_team == "Sestri Levante" and missing.home_shots is None
    warnings = [r for r in caplog.records if r.getMessage() == "statistics unavailable, keeping match without stats"]
    assert {r.match_id for r in warnings} == {f"SS-{BROKEN_STATS_EVENT}", f"SS-{MISSING_STATS_EVENT}"}
    # seasons + rounds + 3 rounds of events + 3 ok stats + 1 x 404 + (max_retries + 1) x 500
    assert transport.calls == 2 + 3 + 3 + 1 + (http_cfg.max_retries + 1)
    assert client.stats.retries == http_cfg.max_retries and len(sleeper.calls) == http_cfg.max_retries


@pytest.mark.asyncio
async def test_sofascore_harvest_explicit_rounds_without_stats(http_cfg: HttpConfig, matcher: TeamMatcher) -> None:
    client, transport, _ = make_client(sofascore_router, http_cfg)
    provider = SofaScoreProvider(client, load_endpoints().sofascore, matcher)
    async with client:
        matches, fixtures = await provider.harvest("A", season_id=SEASON_ID, rounds=[2, 1], with_stats=False)
    assert [m.match_id for m in matches] == ["SS-13579101", "SS-13579102", "SS-13579103", "SS-13579111", "SS-13579113"]
    assert fixtures == [] and transport.calls == 2
    assert all(m.home_xg is None for m in matches)


# ============================================================================ FotMob
FOTMOB_ALL_LEAGUES = {
    "favourite": [],
    "popular": [{"id": 55, "name": "Serie A", "ccode": "ITA"}],
    "international": [{"ccode": "INT", "name": "International", "leagues": [{"id": 42, "name": "Champions League"}]}],
    "countries": [
        {
            "ccode": "ITA",
            "name": "Italy",
            "leagues": [
                {"id": 55, "name": "Serie A", "pageUrl": "/leagues/55/overview/serie"},
                {"id": 86, "name": "Serie B"},
                {"id": 10110, "name": "Serie C - Girone A"},
                {"id": 10111, "name": "Serie C Group B", "localizedName": "Serie C Girone B"},
                {"id": 10112, "name": "Serie C - Girone C"},
                {"id": 10190, "name": "Serie C - Coppa Italia"},
            ],
        }
    ],
}

FOTMOB_LEAGUE_B = {
    "details": {"id": 10111, "name": "Serie C Girone B", "country": "ITA"},
    "matches": {
        "allMatches": [
            {
                "round": "3",
                "roundName": 3,
                "pageUrl": "/matches/vicenza-vs-arezzo/1x2y3z#4521234",
                "id": "4521234",
                "home": {"name": "Arezzo", "shortName": "Arezzo", "id": "8530", "score": 2},
                "away": {"name": "Juventus U23", "shortName": "Juventus NG", "id": "1012009", "score": 1},
                "status": {"utcTime": "2025-09-07T13:00:00Z", "finished": True, "started": True, "cancelled": False, "scoreStr": "2 - 1", "reason": {"short": "FT", "long": "Full-Time"}},
            },
            {
                "round": "3",
                "id": "4521235",
                "home": {"name": "Ternana", "id": "8554"},
                "away": {"name": "Ascoli Calcio", "id": "8528"},
                "status": {"utcTime": "2025-09-07T15:00:00.000Z", "finished": True, "started": True, "cancelled": False, "scoreStr": "0 - 0"},
            },
            {
                "round": "4",
                "id": "4521301",
                "home": {"name": "Rimini", "id": "8571"},
                "away": {"name": "Pontedera", "id": "8580"},
                "status": {"utcTime": "2025-09-14T13:00:00Z", "finished": False, "started": False, "cancelled": False},
            },
            {
                "round": "4",
                "id": "4521302",
                "home": {"name": "Forlì", "id": "9999"},
                "away": {"name": "Perugia", "id": "8540"},
                "status": {"utcTime": "2025-09-14T13:00:00Z", "finished": False, "started": False, "cancelled": True, "reason": {"short": "PP", "long": "Postponed"}},
            },
            {
                "round": "4",
                "id": "4521303",
                "home": {"name": "Livorno", "id": "8544"},
                "away": {"name": "Gubbio", "id": "8545"},
                "status": {"utcTime": "2025-09-14T15:00:00Z", "finished": False, "started": True, "cancelled": False, "scoreStr": "1 - 0", "liveTime": {"short": "67'"}},
            },
        ]
    },
}

FOTMOB_DETAILS = {
    "general": {"matchId": "4521234"},
    "content": {
        "stats": {
            "Periods": {
                "All": {
                    "stats": [
                        {
                            "title": "Top stats",
                            "stats": [
                                {"title": "Ball possession", "stats": [58, 42], "key": "BallPossesion"},
                                {"title": "Expected goals (xG)", "stats": ["1.23", "0.98"], "key": "expected_goals"},
                                {"title": "Total shots", "stats": [12, 7], "key": "total_shots"},
                                {"title": "Corners", "stats": [6, 4], "key": "corners"},
                            ],
                        },
                        {"title": "Shots", "stats": [{"title": "Shots on target", "stats": [5, 2], "key": "ShotsOnTarget"}]},
                    ]
                },
                "FirstHalf": {"stats": [{"title": "Top stats", "stats": [{"title": "Total shots", "stats": [5, 3], "key": "total_shots"}]}]},
            }
        }
    },
}


def fotmob_router(request: httpx.Request) -> httpx.Response:
    path, params = request.url.path, request.url.params
    if path == "/api/allLeagues":
        return json_response(FOTMOB_ALL_LEAGUES)
    if path == "/api/leagues" and params.get("id") == "10111":
        return json_response(FOTMOB_LEAGUE_B)
    if path == "/api/matchDetails":
        if params.get("matchId") == "4521234":
            return json_response(FOTMOB_DETAILS)
        return json_response({"error": "upstream"}, status=503)
    return json_response({"error": "not found"}, status=404)


def test_fotmob_league_ids_from_nested_payload() -> None:
    ids = FotMobProvider.league_ids_from_payload(FOTMOB_ALL_LEAGUES)
    assert ids == {Girone.A: 10110, Girone.B: 10111, Girone.C: 10112}
    assert FotMobProvider.league_ids_from_payload({"countries": []}) == {}


@pytest.mark.asyncio
async def test_fotmob_resolve_league_ids_merges_configured(http_cfg: HttpConfig, matcher: TeamMatcher) -> None:
    base = load_endpoints().fotmob
    endpoints = FotMobEndpoints(
        base_url=base.base_url,
        league_ids={Girone.A: 0, Girone.B: 0, Girone.C: 777},
        all_leagues=base.all_leagues,
        league=base.league,
        match_details=base.match_details,
    )
    client, transport, _ = make_client(fotmob_router, http_cfg, base_url=base.base_url)
    provider = FotMobProvider(client, endpoints, matcher)
    async with client:
        ids = await provider.resolve_league_ids()
        assert await provider.resolve_league_ids() == ids  # memoised
    assert ids == {Girone.A: 10110, Girone.B: 10111, Girone.C: 777}
    assert transport.calls == 1


def test_fotmob_parse_match(matcher: TeamMatcher) -> None:
    rows = FOTMOB_LEAGUE_B["matches"]["allMatches"]
    finished = FotMobProvider.parse_match(rows[0], Girone.B, matcher)
    assert finished == MatchRecord("FM-4521234", date(2025, 9, 7), 3, Girone.B, "Arezzo", "Juventus Next Gen", 2, 1)
    draw = FotMobProvider.parse_match(rows[1], "B", matcher)
    assert isinstance(draw, MatchRecord) and (draw.home_goals, draw.away_goals) == (0, 0) and draw.away_team == "Ascoli"
    fixture = FotMobProvider.parse_match(rows[2], Girone.B, matcher)
    assert fixture == Fixture("FM-4521301", date(2025, 9, 14), 4, Girone.B, "Rimini", "Pontedera")
    assert FotMobProvider.parse_match(rows[3], Girone.B, matcher) is None  # cancelled / postponed
    assert FotMobProvider.parse_match(rows[4], Girone.B, matcher) is None  # live
    no_score = rows[0] | {"status": {**rows[0]["status"], "scoreStr": None}, "home": {"name": "Arezzo"}, "away": {"name": "Ternana"}}
    assert FotMobProvider.parse_match(no_score, Girone.B, matcher) is None
    from_team_scores = rows[0] | {"status": {**rows[0]["status"], "scoreStr": None}}
    assert FotMobProvider.parse_match(from_team_scores, Girone.B, matcher).home_goals == 2
    assert FotMobProvider.parse_match(rows[0] | {"status": {"finished": True}}, Girone.B, matcher) is None  # no utcTime


def test_fotmob_parse_match_details() -> None:
    stats = FotMobProvider.parse_match_details(FOTMOB_DETAILS)
    assert stats == {
        "home_shots": 12.0,
        "away_shots": 7.0,
        "home_shots_target": 5.0,
        "away_shots_target": 2.0,
        "home_corners": 6.0,
        "away_corners": 4.0,
        "home_xg": 1.23,
        "away_xg": 0.98,
    }
    assert all(v is None for v in FotMobProvider.parse_match_details({}).values())


@pytest.mark.asyncio
async def test_fotmob_harvest_end_to_end(http_cfg: HttpConfig, matcher: TeamMatcher) -> None:
    endpoints = load_endpoints().fotmob
    client, transport, _ = make_client(fotmob_router, http_cfg, base_url=endpoints.base_url, extra_headers={"x-mas": "token"})
    provider = FotMobProvider(client, endpoints, matcher)
    async with client:
        matches, fixtures = await provider.harvest(Girone.B)
    assert [m.match_id for m in matches] == ["FM-4521234", "FM-4521235"]
    assert matches[0].home_xg == 1.23 and matches[0].home_shots_target == 5.0
    assert matches[1].home_xg is None  # details endpoint 503 -> stats kept None, match kept
    assert [f.match_id for f in fixtures] == ["FM-4521301"]
    assert all(r.headers["x-mas"] == "token" for r in transport.requests)
    assert transport.calls == 1 + 1 + 1 + (http_cfg.max_retries + 1)


@pytest.mark.asyncio
async def test_fotmob_harvest_unresolved_league_raises(http_cfg: HttpConfig, matcher: TeamMatcher) -> None:
    client, _, _ = make_client(lambda request: json_response({"countries": []}), http_cfg, base_url="https://www.fotmob.com")
    endpoints = load_endpoints().fotmob
    async with client:
        with pytest.raises(HarvestError, match="league id"):
            await FotMobProvider(client, endpoints, matcher).harvest(Girone.A)


# ============================================================================ facade
@pytest.mark.asyncio
async def test_harvester_harvest_all_dataframe_contract(tmp_path: Path, registry: TeamRegistry) -> None:
    cfg = load_endpoints()
    transport = CountingTransport(sofascore_router)
    harvester = Harvester(
        endpoints=cfg, registry=registry, cache_dir=tmp_path / "raw", transport=transport, sleeper=RecordingSleeper(), rng=random.Random(3)
    )
    result = await harvester.harvest_all([Girone.A], seasons=1)
    assert isinstance(result, HarvestResult)
    df = result.matches
    assert list(df.columns) == list(MATCH_COLUMNS)
    for column, dtype in MATCH_DTYPES.items():
        assert str(df[column].dtype) == dtype, column
    assert len(df) == 5 and df["match_id"].is_unique
    assert df["girone"].unique().tolist() == ["A"]
    assert df["date"].is_monotonic_increasing
    broken = df.loc[df["match_id"] == f"SS-{BROKEN_STATS_EVENT}"].iloc[0]
    assert broken["home_goals"] == 2 and broken["away_goals"] == 0
    assert all(pd.isna(broken[c]) for c in hv.STAT_FIELDS)
    good = df.loc[df["match_id"] == "SS-13579101"].iloc[0]
    assert good["home_xg"] == 0.87 and good["away_shots"] == 14.0 and good["away_team"] == "Vicenza"
    assert result.fixtures == [
        Fixture("SS-13579121", date(2025, 9, 6), 3, Girone.A, "AlbinoLeffe", "Lumezzane"),
        Fixture("SS-13579122", date(2025, 9, 7), 3, Girone.A, "Renate", "Alcione Milano"),
    ]
    assert result.unmapped_teams == {"Sestri Levante": 1}
    assert result.provider == "sofascore" and result.failures == {}
    assert set(result.stats) == {"requests", "retries", "cache_hits", "errors"}
    assert result.stats["retries"] == cfg.http.max_retries and result.stats["cache_hits"] == 0
    assert "matches=5" in result.summary()
    # Season bookkeeping of a single-season harvest on its own perimeter.
    assert result.seasons == ("2025/26",) and result.n_matches_by_season == {"2025/26": 5}
    assert result.registry_season == "2025/26" == result.perimeter_season
    assert result.n_historical_matches == 0 and result.historical_teams == ()
    assert result.season_lines() == ["  2025/26: 5 matches (perimeter)"]
    assert "season=2025/26 seasons=2025/26 historical_matches=0" in result.summary()

    # Second run on the same cache directory: only the uncacheable failures hit the transport again.
    before = transport.calls
    second = Harvester(endpoints=cfg, registry=registry, cache_dir=tmp_path / "raw", transport=transport, sleeper=RecordingSleeper())
    again = await second.harvest_all([Girone.A], seasons=1, with_stats=True)
    assert transport.calls - before == (cfg.http.max_retries + 1) + 1
    assert again.stats["cache_hits"] == 2 + 3 + 3
    pd.testing.assert_frame_equal(again.matches, df)

    empty = await Harvester(endpoints=cfg, registry=registry, cache_dir=tmp_path / "raw", transport=transport).harvest_all([], seasons=1)
    assert empty.matches.empty and empty.seasons == () and empty.n_matches_by_season == {} and empty.perimeter_season == "2025/26"
    with pytest.raises(ValueError, match="seasons"):
        await Harvester(endpoints=cfg, registry=registry, cache_dir=tmp_path / "raw", transport=transport).harvest_all([Girone.A], seasons=0)


@pytest.mark.asyncio
async def test_harvester_provider_fallback_and_failures(tmp_path: Path, registry: TeamRegistry) -> None:
    cfg = load_endpoints()

    def router(request: httpx.Request) -> httpx.Response:
        if request.url.host == "api.sofascore.com":
            return json_response({"error": "down"}, status=503)
        return fotmob_router(request)

    harvester = Harvester(
        endpoints=cfg,
        registry=registry,
        cache_dir=tmp_path,
        providers=("sofascore", "fotmob"),
        transport=CountingTransport(router),
        sleeper=RecordingSleeper(),
    )
    result = await harvester.harvest_all([Girone.B])
    assert result.provider == "fotmob"
    assert list(result.failures) == ["B:sofascore"]
    assert result.matches["match_id"].tolist() == ["FM-4521234", "FM-4521235"]
    assert result.unmapped_teams == {}

    only_sofa = Harvester(endpoints=cfg, registry=registry, cache_dir=tmp_path, providers=["sofascore"], transport=CountingTransport(router), sleeper=RecordingSleeper())
    with pytest.raises(HarvestError):
        await only_sofa.harvest_all([Girone.A])
    with pytest.raises(ValueError):
        Harvester(endpoints=cfg, registry=registry, cache_dir=tmp_path, providers=("flashscore",))


def test_harvester_defaults_and_cache_ttl(tmp_path: Path, registry: TeamRegistry) -> None:
    cfg = load_endpoints()
    default_ttl = Harvester(endpoints=cfg, registry=registry, cache_dir=tmp_path)
    assert default_ttl.cache.ttl == timedelta(hours=6) and default_ttl.providers == ("sofascore",)
    custom = Harvester(endpoints=cfg, registry=registry, cache_dir=tmp_path, cache_ttl=timedelta(minutes=5))
    assert custom.cache.ttl == timedelta(minutes=5)


def test_save_and_load_processed_round_trip(tmp_path: Path) -> None:
    records = [
        MatchRecord("SS-1", date(2025, 8, 23), 1, Girone.A, "Pro Vercelli", "Vicenza", 0, 2, home_xg=0.87, away_xg=1.64, home_shots=9, away_shots=14),
        MatchRecord("SS-2", date(2025, 8, 24), 1, Girone.B, "Arezzo", "Ternana", 1, 1),
        MatchRecord("SS-3", date(2025, 8, 24), 1, Girone.C, "Catania", "Forlì", 3, 0, home_corners=7, away_corners=2),
    ]
    df = hv.matches_to_frame(records)
    path = Harvester.save_processed(df, tmp_path / "out" / "matches.csv")
    assert path.exists()
    loaded = Harvester.load_processed(path)
    pd.testing.assert_frame_equal(loaded, df)
    assert list(loaded.columns) == list(MATCH_COLUMNS)
    assert pd.isna(loaded.loc[1, "home_xg"]) and loaded.loc[2, "home_corners"] == 7.0

    empty_path = Harvester.save_processed(hv.empty_matches_frame(), tmp_path / "empty.csv")
    assert Harvester.load_processed(empty_path).empty and list(Harvester.load_processed(empty_path).columns) == list(MATCH_COLUMNS)
    (tmp_path / "zero.csv").write_text("", encoding="utf-8")
    assert Harvester.load_processed(tmp_path / "zero.csv").empty


def test_save_and_load_fixtures_round_trip(tmp_path: Path) -> None:
    fixtures = [
        Fixture("SS-13579121", date(2025, 9, 6), 3, Girone.A, "AlbinoLeffe", "Lumezzane"),
        Fixture("FM-4521301", date(2025, 9, 14), 4, Girone.B, "Rimini", "Pontedera"),
    ]
    path = Harvester.save_fixtures(fixtures, tmp_path / "fixtures.csv")
    assert Harvester.load_fixtures(path) == fixtures
    header_only = Harvester.save_fixtures([], tmp_path / "none.csv")
    assert Harvester.load_fixtures(header_only) == []
    (tmp_path / "bad.csv").write_text("match_id,date\nx,2025-01-01\n", encoding="utf-8")
    with pytest.raises(ValueError, match="missing columns"):
        Harvester.load_fixtures(tmp_path / "bad.csv")


# ============================================================================ CLI
def test_cli_parser_defaults() -> None:
    args = hv._build_parser().parse_args([])
    assert args.gironi == ["A", "B", "C"] and args.season_id is None and args.no_stats is False
    assert args.seasons == 3 and args.season == DEFAULT_SEASON == CURRENT_SEASON
    assert args.out == hv.PROCESSED_DIR / "matches.csv" and args.fixtures_out == hv.PROCESSED_DIR / "fixtures.csv"
    custom = hv._build_parser().parse_args(["--gironi", "A", "C", "--season-id", "77030", "--no-stats", "--log-json", "--providers", "fotmob", "sofascore"])
    assert custom.gironi == ["A", "C"] and custom.season_id == 77030 and custom.no_stats and custom.log_json
    assert custom.providers == ["fotmob", "sofascore"]


def test_cli_main_writes_outputs(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    seen: dict[str, Any] = {}

    async def fake_harvest_all(self: Harvester, gironi, *, season_id=None, seasons=3, with_stats=True, with_odds=True) -> HarvestResult:
        seen.update(gironi=list(gironi), season_id=season_id, seasons=seasons, with_stats=with_stats, providers=self.providers)
        matches = hv.matches_to_frame([MatchRecord("SS-1", date(2025, 8, 23), 1, Girone.A, "Pro Vercelli", "Vicenza", 0, 2)])
        return HarvestResult(
            matches=matches,
            fixtures=[Fixture("SS-2", date(2025, 9, 6), 3, Girone.A, "AlbinoLeffe", "Lumezzane")],
            unmapped_teams={"Sestri Levante": 1},
            stats={"requests": 3, "retries": 0, "cache_hits": 0, "errors": 0},
            provider="sofascore",
        )

    monkeypatch.setattr(Harvester, "harvest_all", fake_harvest_all)
    out = tmp_path / "processed" / "matches.csv"
    fixtures_out = tmp_path / "processed" / "fixtures.csv"
    code = hv.main(
        ["--gironi", "A", "--season-id", "77030", "--out", str(out), "--fixtures-out", str(fixtures_out), "--no-stats", "--cache-dir", str(tmp_path / "raw")]
    )
    assert code == 0
    assert seen == {"gironi": [Girone.A], "season_id": 77030, "seasons": 3, "with_stats": False, "providers": ("sofascore",)}
    assert Harvester.load_processed(out)["match_id"].tolist() == ["SS-1"]
    assert Harvester.load_fixtures(fixtures_out)[0].match_id == "SS-2"
    printed = capsys.readouterr().out
    assert "matches=1" in printed and "Sestri Levante x1" in printed


# ============================================================================ review findings
def test_module_level_persistence_helpers_match_main_py_import(tmp_path: Path) -> None:
    """``main.py`` does ``from src.ingestion.harvester import load_fixtures, load_processed``."""
    from src.ingestion.harvester import load_fixtures, load_processed, save_fixtures, save_processed  # noqa: PLC0415
    import src.ingestion as package  # noqa: PLC0415

    assert load_processed is Harvester.load_processed and load_fixtures is Harvester.load_fixtures
    assert save_processed is Harvester.save_processed and save_fixtures is Harvester.save_fixtures
    for name in ("load_processed", "load_fixtures", "save_processed", "save_fixtures"):
        assert hasattr(hv, name) and name in hv.__all__ and name in package.__all__
        assert getattr(package, name) is getattr(hv, name)
    df = hv.matches_to_frame([MatchRecord("SS-1", date(2025, 8, 23), 1, Girone.A, "Pro Vercelli", "Vicenza", 0, 2)])
    fixtures = [Fixture("SS-2", date(2025, 9, 6), 3, Girone.A, "AlbinoLeffe", "Lumezzane")]
    pd.testing.assert_frame_equal(load_processed(save_processed(df, tmp_path / "m.csv")), df)
    assert load_fixtures(save_fixtures(fixtures, tmp_path / "f.csv")) == fixtures


@pytest.mark.asyncio
async def test_harvest_failure_cancels_sibling_round_fetches(http_cfg: HttpConfig, matcher: TeamMatcher) -> None:
    """A round that keeps failing must not leave orphan requests running after harvest() raised."""
    cfg = dataclasses.replace(http_cfg, max_retries=1, max_concurrency=4)
    state = {"blocked": 0, "cancelled": 0}

    async def handler(request: httpx.Request) -> httpx.Response:
        found = re.search(r"/events/round/(\d+)$", request.url.path)
        if found is None:
            return sofascore_router(request)
        if found.group(1) == "1":
            return json_response({"error": "down"}, status=503)
        state["blocked"] += 1
        try:
            await asyncio.Event().wait()  # never resolves: only a cancellation can end it
        except asyncio.CancelledError:
            state["cancelled"] += 1
            raise
        return json_response({"events": []})  # pragma: no cover - unreachable

    client, transport, _ = make_client(handler, cfg)
    provider = SofaScoreProvider(client, load_endpoints().sofascore, matcher)
    async with client:
        with pytest.raises(HarvestError) as info:
            await asyncio.wait_for(provider.harvest(Girone.A, season_id=SEASON_ID, rounds=range(1, 5)), timeout=5)
        leftovers = [t for t in asyncio.all_tasks() if t is not asyncio.current_task() and not t.done()]
    assert type(info.value) is HarvestError and info.value.status == 503 and info.value.attempts == 2  # not an ExceptionGroup
    assert state["blocked"] == 3 and state["cancelled"] == 3  # every in-flight sibling was cancelled
    assert leftovers == []
    assert transport.calls == 2 + 3  # two attempts for round 1, one (cancelled) attempt per sibling


@pytest.mark.asyncio
async def test_run_concurrently_contract() -> None:
    async def value(x: int) -> int:
        return x * 2

    async def boom() -> int:
        raise NotFoundError("gone", url="https://x/404", status=404)

    assert await hv.run_concurrently([value(1), value(2), value(3)]) == [2, 4, 6]
    assert await hv.run_concurrently([]) == []
    with pytest.raises(NotFoundError) as info:
        await hv.run_concurrently([value(1), boom()])
    assert info.value.status == 404 and isinstance(info.value.__cause__, BaseExceptionGroup)

    async def bug() -> int:
        raise RuntimeError("programming error")

    with pytest.raises(RuntimeError):
        await hv.run_concurrently([bug(), value(1)])


@pytest.mark.asyncio
async def test_cache_write_failure_keeps_payload_and_match(
    http_cfg: HttpConfig, tmp_path: Path, matcher: TeamMatcher, caplog: pytest.LogCaptureFixture
) -> None:
    cache_dir = tmp_path / "raw"
    cache = RawCache(cache_dir)
    shutil.rmtree(cache_dir)
    cache_dir.write_text("not a directory", encoding="utf-8")  # every write now fails with FileExistsError
    client, transport, _ = make_client(sofascore_router, http_cfg, cache=cache)
    provider = SofaScoreProvider(client, load_endpoints().sofascore, matcher)
    with caplog.at_level(logging.WARNING, logger="src.ingestion.harvester"):
        async with client:
            matches, fixtures = await provider.harvest(Girone.A, season_id=SEASON_ID, rounds=[1], with_stats=True)
    assert [m.match_id for m in matches] == ["SS-13579101", "SS-13579102", "SS-13579103"]
    assert matches[0].home_xg == 0.87 and fixtures == []
    assert transport.calls == 1 + 2 + (http_cfg.max_retries + 1)  # events, two good stats payloads, the broken one
    failed = [r for r in caplog.records if r.getMessage() == "cache write failed, payload kept"]
    assert len(failed) == 1 + 2 and all("File exists" in r.error for r in failed)  # events + the two good stats payloads
    assert client.stats.errors == 1  # only the genuine 500 counts as a request error


@pytest.mark.asyncio
async def test_cache_read_failure_falls_back_to_network(
    http_cfg: HttpConfig, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    cache = RawCache(tmp_path)

    def unreadable(url: str, params: Any = None, *, stable: bool = False) -> Any:
        raise PermissionError(13, "Permission denied", str(tmp_path))

    monkeypatch.setattr(cache, "get", unreadable)
    client, transport, _ = make_client(lambda request: json_response({"n": 1}), http_cfg, cache=cache)
    with caplog.at_level(logging.WARNING, logger="src.ingestion.harvester"):
        async with client:
            assert await client.get_json("/x") == {"n": 1}
    assert transport.calls == 1 and client.stats.cache_hits == 0
    assert any(r.getMessage() == "cache read failed, fetching" for r in caplog.records)


@pytest.mark.asyncio
async def test_sofascore_harvest_treats_no_rounds_as_failure(http_cfg: HttpConfig, matcher: TeamMatcher, registry: TeamRegistry, tmp_path: Path) -> None:
    def router(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("/rounds"):
            return json_response({"rounds": []})  # payload drift: no rounds, no currentRound
        return sofascore_router(request)

    client, transport, _ = make_client(router, http_cfg)
    provider = SofaScoreProvider(client, load_endpoints().sofascore, matcher)
    async with client:
        with pytest.raises(HarvestError, match="no rounds") as info:
            await provider.harvest(Girone.A, season_id=SEASON_ID)
    assert info.value.url.endswith(f"/season/{SEASON_ID}/rounds") and transport.calls == 1
    harvester = Harvester(endpoints=load_endpoints(), registry=registry, cache_dir=tmp_path, transport=CountingTransport(router), sleeper=RecordingSleeper())
    with pytest.raises(HarvestError, match="no rounds"):
        await harvester.harvest_all([Girone.A])


def _fake_harvest_all(matches: pd.DataFrame, failures: dict[str, str]) -> Callable[..., Any]:
    async def fake(self: Harvester, gironi, *, season_id=None, seasons=3, with_stats=True, with_odds=True) -> HarvestResult:
        return HarvestResult(
            matches=matches,
            fixtures=[Fixture("SS-9", date(2025, 9, 6), 3, Girone.A, "AlbinoLeffe", "Lumezzane")],
            unmapped_teams={},
            stats={"requests": 1, "retries": 0, "cache_hits": 0, "errors": 0},
            provider="sofascore",
            failures=failures,
        )

    return fake


def _write_existing_outputs(out: Path, fixtures_out: Path) -> pd.DataFrame:
    existing = hv.matches_to_frame(
        [
            MatchRecord("SS-1", date(2025, 8, 23), 1, Girone.A, "Pro Vercelli", "Vicenza", 0, 2),
            MatchRecord("SS-2", date(2025, 8, 24), 1, Girone.B, "Arezzo", "Ternana", 1, 1),
        ]
    )
    Harvester.save_processed(existing, out)
    Harvester.save_fixtures([Fixture("SS-8", date(2025, 9, 6), 3, Girone.B, "Rimini", "Pontedera")], fixtures_out)
    return existing


def test_cli_partial_harvest_exits_nonzero_and_keeps_existing_outputs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    out, fixtures_out = tmp_path / "matches.csv", tmp_path / "fixtures.csv"
    existing = _write_existing_outputs(out, fixtures_out)
    partial = hv.matches_to_frame([MatchRecord("SS-3", date(2025, 9, 1), 2, Girone.A, "Lecco", "Novara", 2, 0)])
    monkeypatch.setattr(Harvester, "harvest_all", _fake_harvest_all(partial, {"B:sofascore": "HTTP 503", "C:sofascore": "HTTP 503"}))
    args = ["--gironi", "A", "B", "C", "--out", str(out), "--fixtures-out", str(fixtures_out), "--cache-dir", str(tmp_path / "raw")]

    assert hv.main(args) == 1
    pd.testing.assert_frame_equal(Harvester.load_processed(out), existing)  # untouched
    assert [f.match_id for f in Harvester.load_fixtures(fixtures_out)] == ["SS-8"]
    captured = capsys.readouterr()
    assert "partial harvest" in captured.err and "--allow-partial" in captured.err and "failure B:sofascore" in captured.err
    assert "matches ->" not in captured.out

    assert hv.main([*args, "--allow-partial"]) == 1  # written on request, exit code still signals the failures
    assert Harvester.load_processed(out)["match_id"].tolist() == ["SS-3"]
    assert [f.match_id for f in Harvester.load_fixtures(fixtures_out)] == ["SS-9"]
    assert "matches ->" in capsys.readouterr().out


def test_cli_empty_harvest_never_overwrites_good_data(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    out, fixtures_out = tmp_path / "matches.csv", tmp_path / "fixtures.csv"
    existing = _write_existing_outputs(out, fixtures_out)
    monkeypatch.setattr(Harvester, "harvest_all", _fake_harvest_all(hv.empty_matches_frame(), {}))
    args = ["--out", str(out), "--fixtures-out", str(fixtures_out), "--cache-dir", str(tmp_path / "raw")]

    assert hv.main(args) == 1
    pd.testing.assert_frame_equal(Harvester.load_processed(out), existing)
    assert "refusing to overwrite" in capsys.readouterr().err
    assert hv.main([*args, "--allow-partial"]) == 1  # --allow-partial never licenses destroying data with an empty frame
    assert len(Harvester.load_processed(out)) == 2

    fresh_out, fresh_fixtures = tmp_path / "fresh" / "matches.csv", tmp_path / "fresh" / "fixtures.csv"
    assert hv.main(["--out", str(fresh_out), "--fixtures-out", str(fresh_fixtures), "--cache-dir", str(tmp_path / "raw")]) == 1
    assert fresh_out.exists() and Harvester.load_processed(fresh_out).empty  # harmless header-only file where nothing existed
    assert hv._csv_has_rows(fresh_out) is False and hv._csv_has_rows(out) is True and hv._csv_has_rows(tmp_path / "missing.csv") is False


def test_cli_parser_allow_partial_flag() -> None:
    assert hv._build_parser().parse_args([]).allow_partial is False
    assert hv._build_parser().parse_args(["--allow-partial"]).allow_partial is True


def _write_endpoints_toml(path: Path, stat_keys_block: str) -> Path:
    base = (Path(__file__).resolve().parents[1] / "config" / "endpoints.toml").read_text(encoding="utf-8")
    head, _, _ = base.partition("[sofascore.stat_keys]")
    _, _, tail = base.partition("[fotmob]")
    path.write_text(head + "[sofascore.stat_keys]\n" + stat_keys_block + "\n[fotmob]" + tail, encoding="utf-8")
    return path


@pytest.mark.parametrize(
    "block",
    [
        'shots = "totalShotsOnGoal"\nxg = ["expectedGoals"]',  # scalar string: would iterate characters
        'shots = []\nxg = ["expectedGoals"]',  # empty list
        'shots = [""]\nxg = ["expectedGoals"]',  # blank key
        'shots = [1, 2]\nxg = ["expectedGoals"]',  # non-string keys
        "shots = 12",
    ],
)
def test_load_endpoints_rejects_malformed_stat_keys(tmp_path: Path, block: str) -> None:
    with pytest.raises(ValueError, match="stat_keys"):
        load_endpoints(_write_endpoints_toml(tmp_path / "endpoints.toml", block))


def test_load_endpoints_accepts_lists_and_warns_on_unknown_metric(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    toml = _write_endpoints_toml(tmp_path / "endpoints.toml", 'shots = ["totalShotsOnGoal", " Total shots "]\nxG = ["expectedGoals"]')
    with caplog.at_level(logging.WARNING, logger="src.ingestion.harvester"):
        cfg = load_endpoints(toml)
    assert cfg.sofascore.stat_keys == {"shots": ("totalShotsOnGoal", "Total shots"), "xG": ("expectedGoals",)}
    assert any(r.getMessage() == "unknown stat_keys metric is never used" and r.metric == "xG" for r in caplog.records)
    assert SofaScoreProvider.parse_statistics(load_fixture("statistics_13579101.json"), cfg.sofascore.stat_keys)["home_shots"] == 9.0
    no_keys = _write_endpoints_toml(tmp_path / "none.toml", "")
    assert load_endpoints(no_keys).sofascore.stat_keys == {}


@pytest.mark.asyncio
async def test_harvest_warns_when_a_metric_is_never_located(http_cfg: HttpConfig, matcher: TeamMatcher, caplog: pytest.LogCaptureFixture) -> None:
    base = load_endpoints().sofascore
    drifted = dataclasses.replace(base, stat_keys={**base.stat_keys, "xg": ("xgTotalRenamedUpstream",)})
    client, _, _ = make_client(sofascore_router, http_cfg)
    with caplog.at_level(logging.WARNING, logger="src.ingestion.harvester"):
        async with client:
            matches, _ = await SofaScoreProvider(client, drifted, matcher).harvest(Girone.A, season_id=SEASON_ID, rounds=[1])
    assert all(m.home_xg is None for m in matches) and matches[0].home_shots == 9.0
    warnings = [r for r in caplog.records if r.getMessage() == "statistics metrics never located"]
    assert len(warnings) == 1 and warnings[0].metrics == ["xg"] and warnings[0].payloads == 2 and warnings[0].source == "sofascore"
    assert warnings[0].keys == {"xg": ["xgTotalRenamedUpstream"]}

    caplog.clear()
    client, _, _ = make_client(sofascore_router, http_cfg)
    with caplog.at_level(logging.WARNING, logger="src.ingestion.harvester"):
        async with client:
            await SofaScoreProvider(client, base, matcher).harvest(Girone.A, season_id=SEASON_ID, rounds=[1])
    assert not [r for r in caplog.records if r.getMessage() == "statistics metrics never located"]  # xG present in one payload
    assert hv._warn_missing_metrics([], base.stat_keys, "sofascore") == []  # no payload fetched: nothing to judge


# ============================================================================ review findings (python floor, fixture duplicates)
def test_ingestion_package_parses_under_python_311_grammar() -> None:
    """``pyproject.toml`` advertises ``requires-python = ">=3.11"``: no module in the
    ingestion package may use syntax newer than 3.11 (``run_concurrently`` used to be
    declared with PEP 695 ``def f[T]`` syntax, which made the harvester -- and every
    importer such as ``main.py`` and the dashboard -- fail to even compile on 3.11)."""
    import ast
    import subprocess

    package_dir = Path(hv.__file__).parent
    sources = sorted(package_dir.glob("*.py"))
    assert Path(hv.__file__) in sources and len(sources) >= 3
    for source in sources:
        ast.parse(source.read_text(encoding="utf-8"), filename=str(source), feature_version=(3, 11))
    # The generic contract is preserved through a classic TypeVar.
    assert isinstance(hv.T, __import__("typing").TypeVar)
    assert "T" in hv.run_concurrently.__annotations__["coroutines"]
    # Belt and braces: when a real 3.11 interpreter is present, let it parse the module too.
    py311 = shutil.which("python3.11")
    if py311:
        proc = subprocess.run(
            [py311, "-I", "-c", "import ast, sys; ast.parse(open(sys.argv[1], encoding='utf-8').read(), sys.argv[1])", str(hv.__file__)],
            capture_output=True,
            text=True,
            check=False,
        )
        assert proc.returncode == 0, proc.stderr


def test_load_fixtures_dedupes_by_match_id_keeps_first_and_warns(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    """A repeated line in ``fixtures.csv`` used to yield a duplicated prediction in
    ``main.py`` and a duplicated selector entry / export row in the dashboard."""
    first = Fixture("SS-1", date(2025, 9, 6), 3, Girone.A, "AlbinoLeffe", "Lumezzane")
    second = Fixture("FM-2", date(2025, 9, 14), 4, Girone.B, "Rimini", "Pontedera")
    third = Fixture("SS-3", date(2025, 9, 14), 4, Girone.C, "Catania", "Benevento")
    conflicting = dataclasses.replace(first, home_team="Virtus Verona", away_team="Trento")  # same id, different content
    path = Harvester.save_fixtures([first, second, first, conflicting, third, second], tmp_path / "fixtures.csv")
    assert sum(1 for line in path.read_text(encoding="utf-8").splitlines() if line.strip()) == 7  # header + 6 rows on disk

    with caplog.at_level(logging.WARNING, logger="src.ingestion.harvester"):
        loaded = Harvester.load_fixtures(path)

    assert loaded == [first, second, third]  # unique, first occurrence wins, file order preserved
    assert len({f.match_id for f in loaded}) == len(loaded)
    warnings = [r for r in caplog.records if r.getMessage() == "duplicate fixtures dropped, first occurrence kept"]
    assert len(warnings) == 1
    assert warnings[0].dropped == 3 and warnings[0].kept == 3 and warnings[0].match_ids == ["FM-2", "SS-1"]
    assert warnings[0].path == str(path)
    assert hv.load_fixtures(path) == loaded  # module-level alias used by main.py / dashboard shares the behaviour

    caplog.clear()
    with caplog.at_level(logging.WARNING, logger="src.ingestion.harvester"):
        clean = Harvester.load_fixtures(Harvester.save_fixtures([first, second], tmp_path / "clean.csv"))
    assert clean == [first, second]
    assert not [r for r in caplog.records if "duplicate fixtures" in r.getMessage()]


# ============================================================================ multi-season (girone B, 2026/27 perimeter)
TOURNAMENT_B = 142
SEASON_B_2627, SEASON_B_2526, SEASON_B_2425, SEASON_B_2324 = 88031, 77031, 63516, 52885
SEASON_B_NAMES = {SEASON_B_2627: "26/27", SEASON_B_2526: "25/26", SEASON_B_2425: "24/25", SEASON_B_2324: "23/24"}
MULTI_ROUNDS: dict[int, dict[str, Any]] = {
    SEASON_B_2627: {"currentRound": {"round": 2}, "rounds": [{"round": 1}, {"round": 2}, {"round": 3}]},
    SEASON_B_2526: {
        "currentRound": {"round": 38},
        "rounds": [{"round": 1}, {"round": 2}, {"round": 39, "name": "Playoff Round 1", "slug": "playoff-round-1", "prefix": "Playoff"}],
    },
    SEASON_B_2425: {"currentRound": {"round": 38}, "rounds": [{"round": 1}]},
    # 2023/24 deliberately has no rounds payload: the router answers 404 for it.
}


def _ts(iso: str) -> int:
    return int(datetime.fromisoformat(iso).replace(tzinfo=UTC).timestamp())


def make_event(
    event_id: int,
    season_id: int,
    round_no: int,
    home: str,
    away: str,
    when: str,
    *,
    status: str = "finished",
    score: tuple[int, int] = (1, 0),
) -> dict[str, Any]:
    """A SofaScore event of girone B shaped like the real payload."""
    year = SEASON_B_NAMES[season_id]
    finished = status == "finished"
    return {
        "tournament": {"name": "Serie C, Girone B", "uniqueTournament": {"name": "Serie C, Girone B", "id": TOURNAMENT_B}, "id": TOURNAMENT_B},
        "season": {"name": f"Serie C, Girone B {year}", "year": year, "id": season_id},
        "roundInfo": {"round": round_no},
        "status": {"code": 100 if finished else 0, "description": "Ended" if finished else "Not started", "type": status},
        "homeTeam": {"name": home, "shortName": home, "id": 1000 + event_id % 997},
        "awayTeam": {"name": away, "shortName": away, "id": 2000 + event_id % 991},
        "homeScore": {"current": score[0], "display": score[0]} if finished else {},
        "awayScore": {"current": score[1], "display": score[1]} if finished else {},
        "id": event_id,
        "startTimestamp": _ts(when),
    }


MULTI_EVENTS: dict[tuple[int, int], list[dict[str, Any]]] = {
    # current season: current teams only (Pescara, Spezia, Atalanta U23 and Latina joined girone B in 2026/27)
    (SEASON_B_2627, 1): [
        make_event(90001, SEASON_B_2627, 1, "Perugia", "Ravenna", "2026-08-23T18:30:00", score=(2, 1)),
        make_event(90002, SEASON_B_2627, 1, "Delfino Pescara", "Spezia Calcio", "2026-08-23T20:45:00", score=(0, 0)),
    ],
    (SEASON_B_2627, 2): [
        make_event(90011, SEASON_B_2627, 2, "Atalanta U23", "Latina Calcio 1932", "2026-08-30T15:00:00", score=(1, 3)),
        make_event(90012, SEASON_B_2627, 2, "Forlì", "Gubbio", "2026-11-18T18:30:00", status="notstarted"),  # postponed
    ],
    (SEASON_B_2627, 3): [
        make_event(90021, SEASON_B_2627, 3, "Torres", "Vis Pesaro", "2026-11-08T15:00:00", status="notstarted"),
        make_event(90022, SEASON_B_2627, 3, "Livorno", "Grosseto", "2026-11-08T17:30:00", status="notstarted"),
    ],
    # 2025/26: Ternana / Arezzo / Rimini left Serie C B, Juventus Next Gen and Carpi moved to girone A
    (SEASON_B_2526, 1): [
        make_event(80001, SEASON_B_2526, 1, "Ternana Calcio", "Juventus U23", "2025-08-24T18:30:00", score=(2, 0)),
        make_event(80002, SEASON_B_2526, 1, "Arezzo", "Carpi FC", "2025-08-24T20:45:00", score=(1, 1)),
    ],
    (SEASON_B_2526, 2): [
        make_event(80011, SEASON_B_2526, 2, "Rimini", "Perugia", "2025-08-31T15:00:00", score=(0, 1)),
        # never played (excluded club): a not-started event of a PAST season must not become a fixture
        make_event(80012, SEASON_B_2526, 2, "Pontedera", "Ascoli Calcio", "2025-08-31T15:00:00", status="notstarted"),
    ],
    # 2024/25: SPAL is in no dictionary (unmapped), Pontedera is historical, Pescara is current
    (SEASON_B_2425, 1): [
        make_event(70001, SEASON_B_2425, 1, "Ternana Calcio", "Pescara", "2024-08-25T18:30:00", score=(3, 1)),
        make_event(70002, SEASON_B_2425, 1, "SPAL", "Pontedera", "2024-08-25T20:45:00", score=(1, 0)),
    ],
}


def multi_season_router(request: httpx.Request) -> httpx.Response:
    """Girone B with four listed seasons; the oldest one has no rounds payload (404)."""
    path = request.url.path
    if path == f"/api/v1/unique-tournament/{TOURNAMENT_B}/seasons":
        return json_response(load_fixture("seasons_girone_b.json"))
    found = re.fullmatch(rf"/api/v1/unique-tournament/{TOURNAMENT_B}/season/(\d+)/rounds", path)
    if found:
        payload = MULTI_ROUNDS.get(int(found.group(1)))
        return json_response(payload) if payload is not None else json_response({"error": {"code": 404}}, status=404)
    found = re.fullmatch(rf"/api/v1/unique-tournament/{TOURNAMENT_B}/season/(\d+)/events/round/(\d+)", path)
    if found:
        return json_response({"events": MULTI_EVENTS.get((int(found.group(1)), int(found.group(2))), [])})
    if re.fullmatch(r"/api/v1/event/(\d+)/statistics", path):
        return json_response({"error": {"code": 404}}, status=404)
    return json_response({"error": {"code": 404, "message": f"no route for {path}"}}, status=404)


def _season_requests(transport: CountingTransport, season_id: int) -> list[str]:
    return [str(r.url) for r in transport.requests if f"/season/{season_id}/" in r.url.path]


def test_season_label_helpers() -> None:
    assert season_label_from_year("26/27") == "2026/27" and season_label_from_year("2026/27") == "2026/27"
    assert season_label_from_year("2026_27") == "2026/27" and season_label_from_year("2026-2027") == "2026/27"
    assert season_label_from_year("99/00") == "1999/00"
    assert season_label_from_year("26/28") is None and season_label_from_year("Serie C") is None
    assert season_label_from_year("") is None and season_label_from_year(None) is None
    assert hv._sort_season_labels(["2024/25", "season 7", "2026/27", "2025/26", "2026/27"]) == ["2026/27", "2025/26", "2024/25", "season 7"]
    assert hv._events_season_label([MULTI_EVENTS[(SEASON_B_2526, 1)], MULTI_EVENTS[(SEASON_B_2627, 1)][:1]]) == "2025/26"
    assert hv._events_season_label([[{"id": 1}]]) is None
    assert season_of(date(2026, 8, 23)) == "2026/27" and season_of(date(2026, 6, 30)) == "2025/26"


def test_registry_with_history_merges_older_dictionaries(current_registry: TeamRegistry, tmp_path: Path) -> None:
    merged = Harvester.registry_with_history(current_registry)
    assert merged.canonical_names == current_registry.canonical_names  # perimeter unchanged
    # Vicenza and Arezzo play Serie B in the 2026/27 perimeter (code SB), so they are current, not historical.
    assert {"Ternana", "Pontedera", "Pro Patria", "Rimini"} <= set(merged.historical_names)
    assert merged.is_current_team("Vicenza") and merged.is_current_team("Arezzo")
    assert merged.exact("Ternana Calcio") == "Ternana" and merged.exact("Juventus U23") == "Juventus Next Gen"
    legacy = Harvester.registry_with_history(load_registry(LEGACY_SEASON))
    assert legacy.historical_names == ()  # nothing older than 2025/26 ships
    synthetic = TeamRegistry.from_teams("test", {"A": ["Alpha", "Beta"]})
    assert Harvester.registry_with_history(synthetic) is synthetic
    assert Harvester.registry_with_history(current_registry, config_dir=tmp_path) is current_registry  # no older files there

    cfg = load_endpoints()
    by_season = Harvester(endpoints=cfg, season=CURRENT_SEASON, cache_dir=tmp_path)
    assert by_season.registry.season_label == "2026/27" and "Ternana" in by_season.matcher.historical_names
    plain = Harvester(endpoints=cfg, season=CURRENT_SEASON, cache_dir=tmp_path, historical_aliases=False)
    assert plain.matcher.historical_names == frozenset()
    default = Harvester(endpoints=cfg, cache_dir=tmp_path)
    assert default.registry.season_label == "2026/27" == Harvester(endpoints=cfg, cache_dir=tmp_path, registry=current_registry).registry.season_label
    with pytest.raises(FileNotFoundError):
        Harvester(endpoints=cfg, season="1999_00", cache_dir=tmp_path)


def test_parse_event_historical_lookup_is_unrestricted(current_registry: TeamRegistry) -> None:
    matcher = TeamMatcher(Harvester.registry_with_history(current_registry))
    event = MULTI_EVENTS[(SEASON_B_2526, 1)][0]  # Ternana Calcio v Juventus U23, listed under girone B
    strict = SofaScoreProvider.parse_event(event, Girone.B, matcher)
    assert isinstance(strict, MatchRecord)
    assert (strict.home_team, strict.away_team) == ("Ternana Calcio", "Juventus U23")  # current composition of B knows neither
    historical = SofaScoreProvider.parse_event(event, Girone.B, matcher, historical=True)
    assert isinstance(historical, MatchRecord)
    assert (historical.home_team, historical.away_team) == ("Ternana", "Juventus Next Gen")
    assert historical.girone is Girone.B and historical.date == date(2025, 8, 24)  # the girone the match was played in


@pytest.mark.asyncio
async def test_harvest_all_three_seasons_on_current_registry(tmp_path: Path, current_registry: TeamRegistry) -> None:
    cfg = load_endpoints()
    transport = CountingTransport(multi_season_router)
    harvester = Harvester(endpoints=cfg, registry=current_registry, cache_dir=tmp_path / "raw", transport=transport, sleeper=RecordingSleeper())
    result = await harvester.harvest_all([Girone.B], seasons=3, with_stats=False)

    # all three seasons were fetched, the fourth listed season was not touched
    assert result.seasons == ("2026/27", "2025/26", "2024/25")
    assert result.n_matches_by_season == {"2026/27": 3, "2025/26": 3, "2024/25": 2}
    assert sum(result.n_matches_by_season.values()) == len(result.matches) == 8
    assert all(_season_requests(transport, sid) for sid in (SEASON_B_2627, SEASON_B_2526, SEASON_B_2425))
    assert _season_requests(transport, SEASON_B_2324) == []
    assert transport.calls == 1 + 3 + 6  # seasons, rounds x3, events for 3 + 2 + 1 rounds
    assert result.failures == {} and result.provider == "sofascore"

    # fixtures only from the current (first) season: the 2025/26 not-started event is dropped
    assert [f.match_id for f in result.fixtures] == ["SS-90012", "SS-90021", "SS-90022"]
    assert all(season_of(f.date) == "2026/27" for f in result.fixtures)

    # one frame, fixed schema, no season column; the season is a function of the date
    df = result.matches
    assert list(df.columns) == list(MATCH_COLUMNS) and df["match_id"].is_unique
    assert df["girone"].unique().tolist() == ["B"]  # historical rows keep the girone they were played in
    assert Counter(season_of(d) for d in df["date"]) == result.n_matches_by_season
    teams = set(df["home_team"]) | set(df["away_team"])
    assert {"Juventus Next Gen", "Carpi", "Pescara", "Latina", "Spezia", "Atalanta U23"} <= teams  # current names whatever the girone then
    assert "Juventus U23" not in teams and "Delfino Pescara" not in teams

    # perimeter bookkeeping
    assert result.registry_season == "2026/27" == result.perimeter_season
    assert result.n_historical_matches == 5
    # Arezzo and Ascoli are Serie B teams of the 2026/27 perimeter (current); Ascoli only appears in the dropped fixture anyway
    assert result.historical_teams == ("Pontedera", "Rimini", "Ternana")
    assert result.unmapped_teams == {"SPAL": 1}
    assert set(result.historical_teams).isdisjoint(result.unmapped_teams)
    assert result.season_lines() == [
        "  2026/27: 3 matches (perimeter)",
        "  2025/26: 3 matches (history)",
        "  2024/25: 2 matches (history)",
    ]
    assert "season=2026/27 seasons=2026/27,2025/26,2024/25 historical_matches=5 historical_teams=3 unmapped_teams=1" in result.summary()

    # seasons=1 restricts to the current season
    only_current = await Harvester(
        endpoints=cfg, registry=current_registry, cache_dir=tmp_path / "raw1", transport=CountingTransport(multi_season_router), sleeper=RecordingSleeper()
    ).harvest_all([Girone.B], seasons=1, with_stats=False)
    assert only_current.seasons == ("2026/27",) and len(only_current.matches) == 3 and only_current.n_historical_matches == 0
    assert only_current.historical_teams == () and only_current.unmapped_teams == {}


@pytest.mark.asyncio
async def test_provider_harvest_seasons_structure_and_tuple_api(http_cfg: HttpConfig, current_registry: TeamRegistry) -> None:
    matcher = TeamMatcher(Harvester.registry_with_history(current_registry))
    client, transport, _ = make_client(multi_season_router, http_cfg)
    provider = SofaScoreProvider(client, load_endpoints().sofascore, matcher)
    async with client:
        harvested = await provider.harvest_seasons(Girone.B, seasons=3, with_stats=False)
        assert [type(item) for item in harvested] == [SeasonHarvest] * 3
        assert [item.label for item in harvested] == ["2026/27", "2025/26", "2024/25"]
        assert [item.season.id for item in harvested] == [SEASON_B_2627, SEASON_B_2526, SEASON_B_2425]
        assert [item.current for item in harvested] == [True, False, False]
        assert [item.historical for item in harvested] == [False, True, True]
        assert [len(item.matches) for item in harvested] == [3, 3, 2] and [len(item.fixtures) for item in harvested] == [3, 0, 0]
        assert [item.rounds for item in harvested] == [3, 2, 1] and all(item.error is None for item in harvested)

        matches, fixtures = await provider.harvest("B", seasons=2, with_stats=False)
        assert [m.match_id for m in matches] == ["SS-90001", "SS-90002", "SS-90011", "SS-80001", "SS-80002", "SS-80011"]
        assert [f.match_id for f in fixtures] == ["SS-90012", "SS-90021", "SS-90022"]
        single, _ = await provider.harvest(Girone.B, with_stats=False)  # default: one season, as before
        assert [m.match_id for m in single] == ["SS-90001", "SS-90002", "SS-90011"]
        with pytest.raises(ValueError, match="seasons"):
            await provider.harvest_seasons(Girone.B, seasons=0)


@pytest.mark.asyncio
async def test_pinned_past_season_is_historical_relative_to_the_perimeter(
    http_cfg: HttpConfig, tmp_path: Path, current_registry: TeamRegistry
) -> None:
    matcher = TeamMatcher(Harvester.registry_with_history(current_registry))
    client, transport, _ = make_client(multi_season_router, http_cfg)
    provider = SofaScoreProvider(client, load_endpoints().sofascore, matcher)
    async with client:
        [item] = await provider.harvest_seasons(Girone.B, season_id=SEASON_B_2526, seasons=3, with_stats=False)
    assert item.label == "2025/26" and item.season == SeasonInfo(SEASON_B_2526, "", "")  # label from the match dates
    assert item.current and item.historical  # first (only) season of the run, yet outside the 2026/27 perimeter
    assert {m.home_team for m in item.matches} == {"Ternana", "Arezzo", "Rimini"}
    assert [f.match_id for f in item.fixtures] == ["SS-80012"]  # pinned season: its not-started events are the fixtures
    assert item.fixtures[0].home_team == "Pontedera" and item.fixtures[0].away_team == "Ascoli"
    assert not any(r.url.path.endswith("/seasons") for r in transport.requests)  # no seasons call when pinned

    harvester = Harvester(
        endpoints=load_endpoints(), registry=current_registry, cache_dir=tmp_path / "raw", transport=CountingTransport(multi_season_router), sleeper=RecordingSleeper()
    )
    result = await harvester.harvest_all([Girone.B], season_id=SEASON_B_2526, with_stats=False)
    assert result.seasons == ("2025/26",) and result.registry_season == "2026/27"
    assert result.n_historical_matches == 3 == len(result.matches)  # everything is history for the perimeter
    assert result.historical_teams == ("Pontedera", "Rimini", "Ternana")  # Arezzo and Ascoli: current Serie B teams


@pytest.mark.asyncio
async def test_more_seasons_than_available_records_the_failure(
    http_cfg: HttpConfig, tmp_path: Path, current_registry: TeamRegistry, caplog: pytest.LogCaptureFixture
) -> None:
    matcher = TeamMatcher(Harvester.registry_with_history(current_registry))
    client, transport, _ = make_client(multi_season_router, http_cfg)
    provider = SofaScoreProvider(client, load_endpoints().sofascore, matcher)
    with caplog.at_level(logging.WARNING, logger="src.ingestion.harvester"):
        async with client:
            harvested = await provider.harvest_seasons(Girone.B, seasons=5, with_stats=False)
            with pytest.raises(NotFoundError):  # the tuple API is strict
                await provider.harvest(Girone.B, seasons=5, with_stats=False)
    assert [item.label for item in harvested] == ["2026/27", "2025/26", "2024/25", "2023/24"]
    failed = harvested[-1]
    assert failed.error and "404" in failed.error and failed.matches == () and failed.historical and not failed.current
    assert any(r.getMessage() == "fewer seasons available than requested" and r.requested == 5 and r.available == 4 for r in caplog.records)
    assert any(r.getMessage() == "historical season failed, skipped" and r.season == "2023/24" for r in caplog.records)

    harvester = Harvester(endpoints=load_endpoints(), registry=current_registry, cache_dir=tmp_path, transport=CountingTransport(multi_season_router), sleeper=RecordingSleeper())
    result = await harvester.harvest_all([Girone.B], seasons=4, with_stats=False)
    assert len(result.matches) == 8 and result.seasons == ("2026/27", "2025/26", "2024/25", "2023/24")
    assert result.n_matches_by_season["2023/24"] == 0 and sum(result.n_matches_by_season.values()) == 8
    assert list(result.failures) == ["B:sofascore:2023/24"] and "404" in result.failures["B:sofascore:2023/24"]
    write, reason = hv._write_decision(result, tmp_path / "matches.csv", allow_partial=False)
    assert write is False and "partial harvest" in (reason or "")  # history gaps are not silently written over good data


@pytest.mark.asyncio
async def test_current_season_failure_still_raises(http_cfg: HttpConfig, current_registry: TeamRegistry) -> None:
    def router(request: httpx.Request) -> httpx.Response:
        if f"/season/{SEASON_B_2627}/rounds" in request.url.path:
            return json_response({"error": "down"}, status=503)
        return multi_season_router(request)

    client, _, _ = make_client(router, http_cfg)
    provider = SofaScoreProvider(client, load_endpoints().sofascore, TeamMatcher(current_registry))
    async with client:
        with pytest.raises(HarvestError) as info:
            await provider.harvest_seasons(Girone.B, seasons=3, with_stats=False)
    assert info.value.status == 503


def test_cli_seasons_flags_and_season_report(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    parsed = hv._build_parser().parse_args(["--seasons", "2", "--season", "2025_26"])
    assert parsed.seasons == 2 and parsed.season == "2025_26"
    with pytest.raises(SystemExit) as stop:
        hv.main(["--seasons", "0", "--cache-dir", str(tmp_path / "raw")])
    assert stop.value.code == 2

    seen: dict[str, Any] = {}

    async def fake_harvest_all(self: Harvester, gironi, *, season_id=None, seasons=3, with_stats=True, with_odds=True) -> HarvestResult:
        seen.update(seasons=seasons, season_id=season_id, registry=self.registry.season_label, historical=sorted(self.matcher.historical_names)[:1])
        matches = hv.matches_to_frame(
            [
                MatchRecord("SS-90001", date(2026, 8, 23), 1, Girone.B, "Perugia", "Ravenna", 2, 1),
                MatchRecord("SS-80001", date(2025, 8, 24), 1, Girone.B, "Ternana", "Juventus Next Gen", 2, 0),
                MatchRecord("SS-70001", date(2024, 8, 25), 1, Girone.B, "Ternana", "Pescara", 3, 1),
            ]
        )
        return HarvestResult(
            matches=matches,
            fixtures=[Fixture("SS-90021", date(2026, 11, 8), 3, Girone.B, "Torres", "Vis Pesaro")],
            unmapped_teams={"SPAL": 1},
            stats={"requests": 10, "retries": 0, "cache_hits": 0, "errors": 0},
            provider="sofascore",
            seasons=("2026/27", "2025/26", "2024/25"),
            n_matches_by_season={"2026/27": 1, "2025/26": 1, "2024/25": 1},
            historical_teams=("Ternana",),
            registry_season="2026/27",
        )

    monkeypatch.setattr(Harvester, "harvest_all", fake_harvest_all)
    out, fixtures_out = tmp_path / "matches.csv", tmp_path / "fixtures.csv"
    code = hv.main(["--gironi", "B", "--seasons", "2", "--out", str(out), "--fixtures-out", str(fixtures_out), "--cache-dir", str(tmp_path / "raw")])
    assert code == 0
    assert seen == {"seasons": 2, "season_id": None, "registry": "2026/27", "historical": ["Bra"]}
    loaded = Harvester.load_processed(out)
    assert list(loaded.columns) == list(MATCH_COLUMNS) and len(loaded) == 3  # one CSV, no season column
    printed = capsys.readouterr().out
    assert "perimeter: season 2026/27 (config/teams_2026_27.json, 80 current teams)" in printed  # 60 Serie C + 20 Serie B
    assert "  2026/27: 1 matches (perimeter)" in printed and "  2024/25: 1 matches (history)" in printed
    assert "historical matches used: 2" in printed
    assert "historical teams (not in the 2026/27 perimeter, history only): Ternana (1)" in printed
    assert "unmapped teams: SPAL x1" in printed
    assert "season=2026/27 seasons=2026/27,2025/26,2024/25 historical_matches=2" in printed

    code = hv.main(["--gironi", "B", "--season", "1999_00", "--out", str(out), "--fixtures-out", str(fixtures_out), "--cache-dir", str(tmp_path / "raw")])
    assert code == 1 and "harvest failed" in capsys.readouterr().err  # unknown dictionary: clean failure, no traceback


# ============================================================================ review findings: ingestion
def test_cli_missing_or_invalid_endpoints_file_is_diagnosed_as_such(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """A missing --endpoints file used to be reported as 'team dictionary not found' (finding 1).

    ``hv.main`` reconfigures the root logger onto stderr (``force=True``), so the
    structured log lines are checked on the captured stderr, as the other CLI tests do.
    """
    missing = tmp_path / "nonexistent.toml"
    common = ["--cache-dir", str(tmp_path / "raw"), "--out", str(tmp_path / "m.csv"), "--fixtures-out", str(tmp_path / "f.csv")]
    assert hv.main(["--endpoints", str(missing), *common]) == 1
    err = capsys.readouterr().err
    assert f"endpoints file not found path={missing}" in err and "team dictionary" not in err
    assert f"harvest failed: endpoints file not found: {missing}" in err
    assert not (tmp_path / "m.csv").exists()

    malformed = tmp_path / "broken.toml"
    malformed.write_text("[sofascore\nbase_url = ", encoding="utf-8")
    assert hv.main(["--endpoints", str(malformed), *common]) == 1
    err = capsys.readouterr().err
    assert f"endpoints file invalid path={malformed}" in err and "team dictionary" not in err and "Traceback" not in err
    assert f"harvest failed: invalid endpoints file {malformed}" in err

    incomplete = tmp_path / "incomplete.toml"
    incomplete.write_text('[sofascore]\nbase_url = "https://api.sofascore.com"\n', encoding="utf-8")  # keys missing
    assert hv.main(["--endpoints", str(incomplete), *common]) == 1
    err = capsys.readouterr().err
    assert f"endpoints file invalid path={incomplete}" in err and "team dictionary" not in err

    # The dictionary diagnosis is kept for the dictionary problem only.
    assert hv.main(["--season", "1999_00", *common]) == 1
    err = capsys.readouterr().err
    assert "team dictionary not found season=1999_00" in err and "endpoints file" not in err and "harvest failed" in err


def test_http_config_history_cache_ttl() -> None:
    cfg = load_endpoints()
    assert cfg.http.history_cache_ttl_hours == 8760.0 and cfg.http.history_cache_ttl == timedelta(days=365)
    assert cfg.http.cache_ttl == timedelta(hours=6) < cfg.http.history_cache_ttl
    assert HttpConfig().history_cache_ttl == timedelta(hours=8760)
    assert HttpConfig(history_cache_ttl_hours=0).history_cache_ttl == timedelta(0)
    with pytest.raises(ValueError, match="history_cache_ttl_hours"):
        HttpConfig(history_cache_ttl_hours=-1)


def test_load_endpoints_reads_history_cache_ttl_hours(tmp_path: Path) -> None:
    base = (Path(__file__).resolve().parents[1] / "config" / "endpoints.toml").read_text(encoding="utf-8")
    custom = tmp_path / "endpoints.toml"
    custom.write_text(base.replace("history_cache_ttl_hours = 8760", "history_cache_ttl_hours = 48"), encoding="utf-8")
    assert load_endpoints(custom).http.history_cache_ttl == timedelta(hours=48)
    without = tmp_path / "without.toml"
    without.write_text(base.replace("history_cache_ttl_hours = 8760\n", ""), encoding="utf-8")
    assert load_endpoints(without).http.history_cache_ttl_hours == 8760.0  # default when the key is absent


def test_raw_cache_stable_reads_use_the_stable_ttl(tmp_path: Path) -> None:
    cache = RawCache(tmp_path, ttl=timedelta(0))  # volatile payloads expire at once, stable ones never (default)
    assert cache.ttl == timedelta(0) and cache.stable_ttl is None
    assert cache.ttl_for() == timedelta(0) and cache.ttl_for(stable=True) is None
    cache.put("https://a/season/1/rounds", {"rounds": [1]})
    assert cache.get("https://a/season/1/rounds") is None  # expired as volatile
    assert cache.get("https://a/season/1/rounds", stable=True) == {"rounds": [1]}  # same file, served as stable
    assert cache.stats == {"hits": 1, "misses": 1}

    bounded = RawCache(tmp_path / "bounded", ttl=None, stable_ttl=timedelta(0))
    bounded.put("https://a/x", [1])
    assert bounded.get("https://a/x") == [1] and bounded.get("https://a/x", stable=True) is None

    # A stale timestamp expires under a bounded stable TTL but not under ttl=None.
    aged = RawCache(tmp_path / "aged", ttl=timedelta(hours=6), stable_ttl=timedelta(hours=8760))
    path = aged.put("https://a/y", {"v": 1})
    record = json.loads(path.read_text(encoding="utf-8"))
    record["fetched_at"] = (datetime.now(UTC) - timedelta(days=30)).isoformat(timespec="seconds")
    path.write_text(json.dumps(record), encoding="utf-8")
    assert aged.get("https://a/y") is None and aged.get("https://a/y", stable=True) == {"v": 1}
    record["fetched_at"] = (datetime.now(UTC) - timedelta(days=400)).isoformat(timespec="seconds")
    path.write_text(json.dumps(record), encoding="utf-8")
    assert aged.get("https://a/y", stable=True) is None


def test_season_is_completed_rule() -> None:
    today = date(2026, 11, 6)
    assert hv._season_is_completed("2025/26", current=False, today=today) is True
    assert hv._season_is_completed("2024/25", current=False, today=today) is True
    assert hv._season_is_completed("2026/27", current=False, today=today) is False  # the running season, even if not listed first
    assert hv._season_is_completed("2027/28", current=False, today=today) is False
    assert hv._season_is_completed(None, current=False, today=today) is True  # not the newest listed season, label unknown
    assert hv._season_is_completed("2024/25", current=True, today=today) is False  # first listed / pinned: may be in progress
    assert hv._season_is_completed(None, current=True, today=today) is False
    # the July-June boundary of season_of: 2025/26 is still running on 30 June 2026
    assert hv._season_is_completed("2025/26", current=False, today=date(2026, 6, 30)) is False
    assert hv._season_is_completed("2025/26", current=False, today=date(2026, 7, 1)) is True


def _multi_season_router_with_stats(request: httpx.Request) -> httpx.Response:
    """The multi-season girone B with one statistics payload for every finished event."""
    if re.fullmatch(r"/api/v1/event/(\d+)/statistics", request.url.path):
        return json_response(load_fixture("statistics_13579101.json"))
    return multi_season_router(request)


def _paths(transport: CountingTransport, start: int) -> list[str]:
    return [r.url.path for r in transport.requests[start:]]


@pytest.mark.asyncio
async def test_completed_seasons_are_cached_as_stable_across_runs(tmp_path: Path, current_registry: TeamRegistry) -> None:
    """Finding 2: with the volatile TTL elapsed, a repeated multi-season run re-downloads the current season only."""
    cfg = load_endpoints()
    transport = CountingTransport(_multi_season_router_with_stats)
    make = lambda **kw: Harvester(  # noqa: E731 - small local factory
        endpoints=cfg, registry=current_registry, cache_dir=tmp_path / "raw", transport=transport, sleeper=RecordingSleeper(), **kw
    )
    first = make(cache_ttl=timedelta(0))  # the current season's payloads expire immediately
    assert first.cache.ttl == timedelta(0) and first.cache.stable_ttl == timedelta(days=365)
    result = await first.harvest_all([Girone.B], seasons=3, with_stats=True)
    assert result.n_matches_by_season == {"2026/27": 3, "2025/26": 3, "2024/25": 2}
    assert transport.calls == 1 + 3 + 6 + 8  # seasons, rounds x3, events for 3+2+1 rounds, statistics x8
    assert result.stats["cache_hits"] == 0

    before = transport.calls
    again = await make(cache_ttl=timedelta(0)).harvest_all([Girone.B], seasons=3, with_stats=True)
    pd.testing.assert_frame_equal(again.matches, result.matches)
    refetched = _paths(transport, before)
    # the seasons listing and the current season (rounds, 3 event rounds, 3 statistics) are volatile: downloaded again
    assert f"/api/v1/unique-tournament/{TOURNAMENT_B}/seasons" in refetched
    assert sum(f"/season/{SEASON_B_2627}/" in path for path in refetched) == 1 + 3
    assert {p for p in refetched if p.endswith("/statistics")} == {f"/api/v1/event/{e}/statistics" for e in (90001, 90002, 90011)}
    # the two completed seasons (2 rounds payloads, 3 event rounds, 5 statistics) come from the cache
    assert not any(f"/season/{sid}/" in path for sid in (SEASON_B_2526, SEASON_B_2425) for path in refetched)
    assert transport.calls - before == 1 + 1 + 3 + 3 and again.stats["cache_hits"] == 2 + 3 + 5

    # history TTL 0 (CLI --history-cache-ttl-hours 0) forces the completed seasons to be downloaded again
    before = transport.calls
    forced = await make(cache_ttl=timedelta(0), history_cache_ttl=timedelta(0)).harvest_all([Girone.B], seasons=3, with_stats=True)
    assert forced.stats["cache_hits"] == 0 and transport.calls - before == 1 + 3 + 6 + 8

    # the long TTL (default from endpoints.toml) also covers the current season's payloads within 6 h: everything is served from cache
    before = transport.calls
    cached = await make().harvest_all([Girone.B], seasons=3, with_stats=True)
    assert transport.calls == before and cached.stats["cache_hits"] == 1 + 3 + 6 + 8


@pytest.mark.asyncio
async def test_season_harvest_reports_completed_and_pinned_seasons_stay_volatile(
    http_cfg: HttpConfig, tmp_path: Path, current_registry: TeamRegistry
) -> None:
    matcher = TeamMatcher(Harvester.registry_with_history(current_registry))
    cache = RawCache(tmp_path / "raw", ttl=timedelta(0))  # volatile entries expire immediately
    client, transport, _ = make_client(_multi_season_router_with_stats, http_cfg, cache=cache)
    provider = SofaScoreProvider(client, load_endpoints().sofascore, matcher)
    async with client:
        harvested = await provider.harvest_seasons(Girone.B, seasons=4, with_stats=True)
        assert [item.completed for item in harvested] == [False, True, True, True]  # the failed 2023/24 season is completed too
        assert harvested[-1].error and harvested[-1].completed
        # a pinned past season is current for the run, hence volatile: its payloads are fetched again
        before = transport.calls
        [pinned] = await provider.harvest_seasons(Girone.B, season_id=SEASON_B_2526, with_stats=True)
        assert pinned.current and pinned.historical and not pinned.completed
        assert transport.calls - before == 1 + 2 + 3  # rounds, 2 event rounds, 3 statistics: nothing served as stable
        # the same season harvested through the listing is completed: everything now comes from the cache
        before = transport.calls
        listed = await provider.harvest_seasons(Girone.B, seasons=2, with_stats=True)
        assert [item.completed for item in listed] == [False, True]
        assert not any(f"/season/{SEASON_B_2526}/" in r.url.path for r in transport.requests[before:])
        assert sum(f"/season/{SEASON_B_2627}/" in r.url.path for r in transport.requests[before:]) == 1 + 3


def test_cli_history_cache_ttl_flag(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    parsed = hv._build_parser().parse_args([])
    assert parsed.cache_ttl_hours is None and parsed.history_cache_ttl_hours is None
    parsed = hv._build_parser().parse_args(["--cache-ttl-hours", "1", "--history-cache-ttl-hours", "0"])
    assert parsed.cache_ttl_hours == 1.0 and parsed.history_cache_ttl_hours == 0.0

    seen: dict[str, Any] = {}

    async def fake_harvest_all(self: Harvester, gironi, *, season_id=None, seasons=3, with_stats=True, with_odds=True) -> HarvestResult:
        seen.update(ttl=self.cache.ttl, stable_ttl=self.cache.stable_ttl)
        matches = hv.matches_to_frame([MatchRecord("SS-90001", date(2026, 8, 23), 1, Girone.B, "Perugia", "Ravenna", 2, 1)])
        return HarvestResult(matches=matches, fixtures=[], unmapped_teams={}, stats={"requests": 1, "retries": 0, "cache_hits": 0, "errors": 0}, provider="sofascore")

    monkeypatch.setattr(Harvester, "harvest_all", fake_harvest_all)
    common = ["--gironi", "B", "--out", str(tmp_path / "m.csv"), "--fixtures-out", str(tmp_path / "f.csv"), "--cache-dir", str(tmp_path / "raw")]
    assert hv.main(common) == 0
    assert seen == {"ttl": timedelta(hours=6), "stable_ttl": timedelta(hours=8760)}  # defaults from endpoints.toml
    assert hv.main([*common, "--cache-ttl-hours", "1", "--history-cache-ttl-hours", "0"]) == 0
    assert seen == {"ttl": timedelta(hours=1), "stable_ttl": timedelta(0)}
    capsys.readouterr()


# ============================================================================ facade: FootyStats provider
FS_FIXTURES = Path(__file__).parent / "fixtures" / "footystats"
FS_HOST = "api.football-data-api.com"
FS_KEY = "facade-test-key-9f8e7d6c"
FS_NOW = datetime(2026, 11, 6, 12, 0, tzinfo=UTC)  # the FootyStats fixture's open matches kick off on 2027-01-15


def footystats_router(request: httpx.Request) -> httpx.Response:
    """Serve the FootyStats fixture files (one season, 17 matches) for any season id."""
    assert request.url.params.get("key") == FS_KEY, "the key must reach the transport"
    files = {"/league-list": "league_list_sample.json", "/league-teams": "league_teams_sample.json", "/league-matches": "league_matches_sample.json"}
    name = files.get(request.url.path)
    if name is None:
        return json_response({"success": False, "message": "not found"}, status=404)
    return json_response(json.loads((FS_FIXTURES / name).read_text(encoding="utf-8")))


def dual_router(request: httpx.Request) -> httpx.Response:
    return footystats_router(request) if request.url.host == FS_HOST else sofascore_router(request)


def test_default_providers_depend_on_the_key(tmp_path: Path, registry: TeamRegistry, monkeypatch: pytest.MonkeyPatch) -> None:
    from src.ingestion import footystats  # noqa: PLC0415

    cfg = load_endpoints()
    assert hv.PROVIDER_NAMES == ("footystats", "sofascore", "fotmob")
    assert hv.default_providers() == ("sofascore",)  # no key anywhere (autouse fixture)
    assert hv.default_providers("k") == ("footystats", "sofascore")
    assert Harvester(endpoints=cfg, registry=registry, cache_dir=tmp_path).providers == ("sofascore",)
    assert Harvester(endpoints=cfg, registry=registry, cache_dir=tmp_path, api_key="k").providers == ("footystats", "sofascore")
    monkeypatch.setattr(footystats, "find_api_key", lambda *args, **kwargs: "from-env")
    assert hv.default_providers() == ("footystats", "sofascore")
    assert Harvester(endpoints=cfg, registry=registry, cache_dir=tmp_path).providers == ("footystats", "sofascore")
    assert Harvester(endpoints=cfg, registry=registry, cache_dir=tmp_path, providers=("sofascore",)).providers == ("sofascore",)
    assert hv._build_parser().parse_args([]).providers is None
    assert hv._build_parser().parse_args(["--providers", "footystats", "sofascore"]).providers == ["footystats", "sofascore"]
    assert cfg.footystats.base_url == "https://api.football-data-api.com" and cfg.footystats.key_env == "FOOTYSTATS_API_KEY"


@pytest.mark.asyncio
async def test_harvest_all_prefers_footystats_and_collects_odds(tmp_path: Path, registry: TeamRegistry, monkeypatch: pytest.MonkeyPatch) -> None:
    from src.ingestion import footystats  # noqa: PLC0415

    monkeypatch.setattr(footystats, "_utc_now", lambda: FS_NOW)
    cfg = load_endpoints()
    transport = CountingTransport(dual_router)
    harvester = Harvester(
        endpoints=cfg,
        registry=registry,
        cache_dir=tmp_path / "raw",
        providers=("footystats", "sofascore"),
        api_key=FS_KEY,
        transport=transport,
        sleeper=RecordingSleeper(),
        mapping_path=tmp_path / "footystats_teams.json",
    )
    result = await harvester.harvest_all([Girone.A], seasons=1)
    assert result.provider == "footystats" and result.failures == {}
    assert list(result.matches.columns) == list(MATCH_COLUMNS) and len(result.matches) == 13
    assert result.matches["girone"].unique().tolist() == ["A"] and result.matches["match_id"].str.startswith("FS-").all()
    assert [f.match_id for f in result.fixtures] == ["FS-900001", "FS-900002"]
    assert len(result.odds) == 5 + 3 and {q.match_id for q in result.odds} == {"FS-900001", "FS-900002"}
    assert all(r.url.host == FS_HOST for r in transport.requests) and transport.calls == 3  # list, teams, matches
    assert "2026/27" in result.seasons and result.n_matches_by_season["2018/19"] == 13  # demo payload dates
    assert "odds=8" in result.summary() and result.stats["requests"] == 3
    assert (tmp_path / "footystats_teams.json").exists()
    assert FS_KEY not in (tmp_path / "footystats_teams.json").read_text(encoding="utf-8")
    assert not [p for p in (tmp_path / "raw").glob("*.json") if FS_KEY in p.read_text(encoding="utf-8")]

    # --no-odds equivalent
    dry = await Harvester(
        endpoints=cfg, registry=registry, cache_dir=tmp_path / "raw", providers=("footystats",), api_key=FS_KEY,
        transport=transport, sleeper=RecordingSleeper(), mapping_path=tmp_path / "footystats_teams.json",
    ).harvest_all([Girone.A], seasons=1, with_odds=False)
    assert dry.odds == [] and len(dry.fixtures) == 2


@pytest.mark.asyncio
async def test_harvest_all_falls_back_to_sofascore_when_footystats_fails(tmp_path: Path, registry: TeamRegistry, caplog: pytest.LogCaptureFixture) -> None:
    cfg = load_endpoints()

    def footystats_down(request: httpx.Request) -> httpx.Response:
        if request.url.host == FS_HOST:
            return json_response({"error": "down"}, status=503)
        return sofascore_router(request)

    harvester = Harvester(
        endpoints=cfg, registry=registry, cache_dir=tmp_path / "raw", providers=("footystats", "sofascore"), api_key=FS_KEY,
        transport=CountingTransport(footystats_down), sleeper=RecordingSleeper(), rng=random.Random(3), mapping_path=tmp_path / "map.json",
    )
    with caplog.at_level(logging.ERROR):
        result = await harvester.harvest_all([Girone.A], seasons=1)
    assert result.provider == "sofascore" and len(result.matches) == 5 and result.odds == []
    assert list(result.failures) == ["A:footystats"] and "503" in result.failures["A:footystats"]
    assert FS_KEY not in result.failures["A:footystats"]
    assert FS_KEY not in "\n".join(f"{r.getMessage()} {r.__dict__}" for r in caplog.records)

    # No key at all: FootyStats is recorded as unavailable and SofaScore still delivers.
    keyless = Harvester(
        endpoints=cfg, registry=registry, cache_dir=tmp_path / "raw2", providers=("footystats", "sofascore"),
        transport=CountingTransport(sofascore_router), sleeper=RecordingSleeper(), rng=random.Random(3), mapping_path=tmp_path / "map.json",
    )
    second = await keyless.harvest_all([Girone.A], seasons=1)
    assert second.provider == "sofascore" and "API key" in second.failures["A:footystats"]

    # FootyStats alone and down: the HarvestError propagates like any single-provider failure.
    only = Harvester(
        endpoints=cfg, registry=registry, cache_dir=tmp_path / "raw3", providers=("footystats",), api_key=FS_KEY,
        transport=CountingTransport(footystats_down), sleeper=RecordingSleeper(), mapping_path=tmp_path / "map.json",
    )
    with pytest.raises(HarvestError):
        await only.harvest_all([Girone.A], seasons=1)


@pytest.mark.asyncio
async def test_harvest_all_reports_gironi_a_provider_cannot_serve(tmp_path: Path, registry: TeamRegistry) -> None:
    cfg = load_endpoints()
    serie_b = Girone.parse("SB")
    assert serie_b not in cfg.sofascore.tournament_ids  # SofaScore knows the three Serie C gironi only
    transport = CountingTransport(sofascore_router)
    harvester = Harvester(endpoints=cfg, registry=registry, cache_dir=tmp_path, providers=("sofascore",), transport=transport, sleeper=RecordingSleeper())
    result = await harvester.harvest_all([serie_b], seasons=1)
    assert result.matches.empty and result.failures == {"SB:sofascore": "no SofaScore tournament id configured for girone SB"}
    assert transport.calls == 0


def test_save_odds_round_trip_and_aliases(tmp_path: Path) -> None:
    from src.common.schemas import Market, OddsQuote  # noqa: PLC0415
    from src.dashboard.state import load_odds_quotes  # noqa: PLC0415
    import src.ingestion as package  # noqa: PLC0415

    quotes = [
        OddsQuote("FS-1", "FootyStats", Market.MATCH_ODDS, {"1": 2.1, "X": 3.3, "2": 3.5}),
        OddsQuote("FS-1", "FootyStats", Market.BTTS, {"GG": 1.8, "NG": 1.95}),
        OddsQuote("FS-2", "Book", Market.OVER_UNDER_25, {"Over 2.5": 1.9, "Under 2.5": 1.9}),
    ]
    path = hv.save_odds(quotes, tmp_path / "odds" / "footystats.csv")
    lines = path.read_text(encoding="utf-8").splitlines()
    assert lines[0] == ",".join(hv.ODDS_COLUMNS) and lines[1] == "FS-1,FootyStats,1X2,1,2.1" and len(lines) == 1 + 7
    loaded = load_odds_quotes(path.parent)
    assert [(q.match_id, q.bookmaker, q.market, q.odds) for q in loaded] == [(q.match_id, q.bookmaker, q.market, q.odds) for q in quotes]
    empty = hv.save_odds([], tmp_path / "odds" / "footystats.csv")
    assert empty.read_text(encoding="utf-8").splitlines() == [",".join(hv.ODDS_COLUMNS)] and load_odds_quotes(path.parent) == []
    assert hv.save_odds is Harvester.save_odds and package.save_odds is Harvester.save_odds and package.ODDS_COLUMNS is hv.ODDS_COLUMNS
    assert HarvestResult(matches=hv.empty_matches_frame(), fixtures=[], unmapped_teams={}, stats={}, provider="x").odds == []


def test_cli_odds_out_and_no_odds(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    from src.common.schemas import Market, OddsQuote  # noqa: PLC0415

    seen: dict[str, Any] = {}

    async def fake_harvest_all(self: Harvester, gironi, *, season_id=None, seasons=3, with_stats=True, with_odds=True) -> HarvestResult:
        seen.update(with_odds=with_odds, providers=self.providers)
        return HarvestResult(
            matches=hv.matches_to_frame([MatchRecord("FS-1", date(2026, 8, 23), 1, Girone.A, "Pro Vercelli", "Vicenza", 0, 2)]),
            fixtures=[Fixture("FS-2", date(2026, 11, 8), 3, Girone.A, "AlbinoLeffe", "Lumezzane")],
            unmapped_teams={},
            stats={"requests": 3, "retries": 0, "cache_hits": 0, "errors": 0},
            provider="footystats",
            odds=[OddsQuote("FS-2", "FootyStats", Market.MATCH_ODDS, {"1": 2.1, "X": 3.3, "2": 3.5})] if with_odds else [],
        )

    monkeypatch.setattr(Harvester, "harvest_all", fake_harvest_all)
    out, fixtures_out, odds_out = tmp_path / "m.csv", tmp_path / "f.csv", tmp_path / "odds" / "fs.csv"
    base = ["--gironi", "A", "--out", str(out), "--fixtures-out", str(fixtures_out), "--odds-out", str(odds_out), "--cache-dir", str(tmp_path / "raw")]
    assert hv.main([*base, "--providers", "footystats", "sofascore"]) == 0
    assert seen == {"with_odds": True, "providers": ("footystats", "sofascore")}
    assert odds_out.exists() and f"odds -> {odds_out}" in capsys.readouterr().out
    assert hv._build_parser().parse_args(base).odds_out == odds_out
    odds_out.unlink()
    assert hv.main([*base, "--no-odds"]) == 0 and seen["with_odds"] is False and not odds_out.exists()
    assert "odds=0" in capsys.readouterr().out


def test_default_providers_honour_the_configured_key_env(tmp_path: Path, registry: TeamRegistry, monkeypatch: pytest.MonkeyPatch) -> None:
    """``[footystats] key_env`` renames the variable: the facade must look the key up under that name."""
    from dataclasses import replace  # noqa: PLC0415
    from src.ingestion import footystats  # noqa: PLC0415

    def keyed(explicit: str | None = None, *, key_env: str = "FOOTYSTATS_API_KEY", **kwargs: Any) -> str | None:
        return "k" if key_env == "MY_FS_KEY" else None

    monkeypatch.setattr(footystats, "find_api_key", keyed)
    cfg = load_endpoints()
    custom = replace(cfg, footystats=replace(cfg.footystats, key_env="MY_FS_KEY"))
    assert hv.default_providers() == ("sofascore",)
    assert hv.default_providers(key_env="MY_FS_KEY") == ("footystats", "sofascore")
    assert Harvester(endpoints=cfg, registry=registry, cache_dir=tmp_path).providers == ("sofascore",)
    assert Harvester(endpoints=custom, registry=registry, cache_dir=tmp_path).providers == ("footystats", "sofascore")
    assert Harvester(endpoints=custom, registry=registry, cache_dir=tmp_path, providers=("sofascore",)).providers == ("sofascore",)


@pytest.mark.asyncio
async def test_harvest_all_default_gironi_are_those_the_providers_can_serve(tmp_path: Path, registry: TeamRegistry) -> None:
    """A default call on a SofaScore-only installation must not record Serie B (no tournament id) as a failure."""
    from dataclasses import replace  # noqa: PLC0415

    cfg = load_endpoints()
    serie_b = Girone.parse("SB")
    sofa_only = Harvester(endpoints=cfg, registry=registry, cache_dir=tmp_path / "a", providers=("sofascore",), transport=CountingTransport(sofascore_router), sleeper=RecordingSleeper())
    assert sofa_only.supported_gironi() == tuple(g for g in Girone if g in cfg.sofascore.tournament_ids)
    assert Girone.A in sofa_only.supported_gironi() and serie_b not in sofa_only.supported_gironi()
    with_footystats = Harvester(endpoints=cfg, registry=registry, cache_dir=tmp_path / "b", providers=("footystats", "sofascore"), api_key=FS_KEY, transport=CountingTransport(dual_router), sleeper=RecordingSleeper())
    assert with_footystats.supported_gironi() == tuple(Girone)
    assert Harvester(endpoints=cfg, registry=registry, cache_dir=tmp_path / "c", providers=("fotmob",), transport=CountingTransport(sofascore_router)).supported_gironi() == tuple(Girone)

    # end to end: the fixture router serves girone A only, so pin the SofaScore ids to A and call with the default
    a_only = replace(cfg, sofascore=replace(cfg.sofascore, tournament_ids={Girone.A: cfg.sofascore.tournament_ids[Girone.A]}))
    harvester = Harvester(endpoints=a_only, registry=registry, cache_dir=tmp_path / "d", providers=("sofascore",), transport=CountingTransport(sofascore_router), sleeper=RecordingSleeper(), rng=random.Random(3))
    assert harvester.supported_gironi() == (Girone.A,)
    result = await harvester.harvest_all(seasons=1)
    assert result.failures == {} and len(result.matches) == 5 and result.matches["girone"].unique().tolist() == ["A"]
    assert hv._write_decision(result, tmp_path / "m.csv", allow_partial=False) == (True, None)
    # an explicit request is still attempted and reported
    explicit = await harvester.harvest_all([serie_b], seasons=1)
    assert explicit.failures == {"SB:sofascore": "no SofaScore tournament id configured for girone SB"}


def test_save_odds_keeps_the_quoted_precision(tmp_path: Path) -> None:
    """Odds are written as quoted (101.25, 10.125), not rounded to four significant digits (101.2, 10.12)."""
    from src.common.schemas import Market, OddsQuote  # noqa: PLC0415
    from src.dashboard.state import load_odds_quotes  # noqa: PLC0415

    quotes = [
        OddsQuote("FS-1", "FootyStats", Market.MATCH_ODDS, {"1": 1.95, "X": 10.125, "2": 101.25}),
        OddsQuote("FS-2", "FootyStats", Market.BTTS, {"GG": 2.1, "NG": 1.666}),
    ]
    path = hv.save_odds(quotes, tmp_path / "odds.csv")
    lines = path.read_text(encoding="utf-8").splitlines()
    assert lines[1:] == ["FS-1,FootyStats,1X2,1,1.95", "FS-1,FootyStats,1X2,X,10.125", "FS-1,FootyStats,1X2,2,101.25", "FS-2,FootyStats,GG/NG,GG,2.1", "FS-2,FootyStats,GG/NG,NG,1.666"]
    loaded = load_odds_quotes(path.parent)
    assert [q.odds for q in loaded] == [q.odds for q in quotes]

