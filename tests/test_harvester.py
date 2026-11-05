"""Offline tests for :mod:`src.ingestion.harvester` (mock transports, no sleeping)."""
from __future__ import annotations

import asyncio
import dataclasses
import importlib.util
import json
import logging
import random
import re
from collections.abc import Callable
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import httpx
import pandas as pd
import pytest

from src.common.schemas import MATCH_COLUMNS, MATCH_DTYPES, Fixture, Girone, MatchRecord
from src.common.teams import TeamRegistry, load_registry
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
    SeasonInfo,
    SofaScoreProvider,
    load_endpoints,
    parse_retry_after,
    parse_stat_value,
)
from src.ingestion.team_matcher import TeamMatcher

FIXTURES = Path(__file__).parent / "fixtures" / "sofascore"
SEASON_ID = 77030
TOURNAMENT_A = 141
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
    return load_registry()


@pytest.fixture(scope="module")
def matcher(registry: TeamRegistry) -> TeamMatcher:
    return TeamMatcher(registry)


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
    result = await harvester.harvest_all([Girone.A])
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

    # Second run on the same cache directory: only the uncacheable failures hit the transport again.
    before = transport.calls
    second = Harvester(endpoints=cfg, registry=registry, cache_dir=tmp_path / "raw", transport=transport, sleeper=RecordingSleeper())
    again = await second.harvest_all([Girone.A], with_stats=True)
    assert transport.calls - before == (cfg.http.max_retries + 1) + 1
    assert again.stats["cache_hits"] == 2 + 3 + 3
    pd.testing.assert_frame_equal(again.matches, df)

    assert (await Harvester(endpoints=cfg, registry=registry, cache_dir=tmp_path / "raw", transport=transport).harvest_all([])).matches.empty


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
    assert args.out == hv.PROCESSED_DIR / "matches.csv" and args.fixtures_out == hv.PROCESSED_DIR / "fixtures.csv"
    custom = hv._build_parser().parse_args(["--gironi", "A", "C", "--season-id", "77030", "--no-stats", "--log-json", "--providers", "fotmob", "sofascore"])
    assert custom.gironi == ["A", "C"] and custom.season_id == 77030 and custom.no_stats and custom.log_json
    assert custom.providers == ["fotmob", "sofascore"]


def test_cli_main_writes_outputs(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    seen: dict[str, Any] = {}

    async def fake_harvest_all(self: Harvester, gironi, *, season_id=None, with_stats=True) -> HarvestResult:
        seen.update(gironi=list(gironi), season_id=season_id, with_stats=with_stats, providers=self.providers)
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
    assert seen == {"gironi": [Girone.A], "season_id": 77030, "with_stats": False, "providers": ("sofascore",)}
    assert Harvester.load_processed(out)["match_id"].tolist() == ["SS-1"]
    assert Harvester.load_fixtures(fixtures_out)[0].match_id == "SS-2"
    printed = capsys.readouterr().out
    assert "matches=1" in printed and "Sestri Levante x1" in printed
