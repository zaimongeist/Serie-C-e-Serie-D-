"""Offline tests for :mod:`src.ingestion.footystats` (``httpx.MockTransport``, no sleeping).

The fixture files in ``tests/fixtures/footystats/`` are real responses of the
public demo key (England 2018/19) with the Italian ``/league-list`` entries and
extra ``incomplete`` / ``suspended`` / ``canceled`` matches. The router serves
them for the Italian season ids of the league list, offsetting match ids per
season so three seasons of the same payload stay distinct. Team names are
English, so the mapping tests build small synthetic dictionaries.
"""
from __future__ import annotations

import dataclasses
import json
import logging
import os
import random
import re
import subprocess
import sys
from collections.abc import Callable
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import httpx
import pandas as pd
import pytest

from src.common.schemas import MATCH_COLUMNS, MATCH_DTYPES, Fixture, Girone, Market, MatchRecord
from src.common.teams import TeamRegistry, load_registry
from src.dashboard.state import load_odds_quotes
from src.ingestion import footystats as fs
from src.ingestion.harvester import FootyStatsEndpoints, HarvestError, HttpConfig, RawCache, load_endpoints
from src.ingestion.team_matcher import TeamMatcher

FIXTURES = Path(__file__).parent / "fixtures" / "footystats"
KEY = "sk-test-0123456789abcdefABCDEF-secret"
WRONG_KEY = "wrong-key-xyz"
HOST = "api.football-data-api.com"
UTC = timezone.utc
NOW = datetime(2026, 11, 6, 12, 0, tzinfo=UTC)  # the fixture's incomplete matches kick off on 2027-01-15
SEASONS_A = {"2026/27": 17487, "2025/26": 15535, "2024/25": 12659}
SEASONS_SB = {"2026/27": 17404, "2025/26": 15632, "2024/25": 12621}
CURRENT_A, PAST_A, OLD_A = SEASONS_A["2026/27"], SEASONS_A["2025/26"], SEASONS_A["2024/25"]


# ============================================================================ helpers
class RecordingSleeper:
    def __init__(self) -> None:
        self.calls: list[float] = []

    async def __call__(self, delay: float) -> None:
        self.calls.append(delay)


class CountingTransport(httpx.MockTransport):
    def __init__(self, handler: Callable[[httpx.Request], Any]) -> None:
        super().__init__(handler)
        self.calls = 0
        self.requests: list[httpx.Request] = []

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        self.calls += 1
        self.requests.append(request)
        return await super().handle_async_request(request)


def load_fixture(name: str) -> dict[str, Any]:
    return json.loads((FIXTURES / name).read_text(encoding="utf-8"))


def json_response(payload: Any, status: int = 200) -> httpx.Response:
    return httpx.Response(status, json=payload)


def _offset(payload: dict[str, Any], season_id: int) -> dict[str, Any]:
    """Make match ids unique per season (the demo payload is the same for every season)."""
    out = dict(payload)
    out["data"] = [{**m, "id": int(m["id"]) + season_id * 1_000_000} for m in payload["data"]]
    return out


def _single_page(payload: dict[str, Any], rows: int) -> dict[str, Any]:
    out = dict(payload)
    out["data"] = payload["data"][:rows]
    out["pager"] = {"current_page": 1, "max_page": 1, "results_per_page": 600, "total_results": rows}
    return out


def make_router(*, failing_seasons: set[int] = frozenset(), league_list: dict[str, Any] | None = None) -> Callable[[httpx.Request], httpx.Response]:
    """Serve the fixture files for the Italian season ids; the key must be present and right."""

    def router(request: httpx.Request) -> httpx.Response:
        assert request.url.host == HOST
        key = request.url.params.get("key")
        if key is None:
            return json_response({"success": False, "message": "No key"}, status=401)
        if key != KEY:
            return json_response({"success": False, "message": "Invalid API key, visit footystats.org"})
        path = request.url.path
        page = int(request.url.params.get("page", "1"))
        if path == "/league-list":
            assert request.url.params.get("chosen_leagues_only") == "true"
            return json_response(league_list if league_list is not None else load_fixture("league_list_sample.json"))
        season_id = int(request.url.params.get("season_id", "0"))
        if path == "/league-teams":
            return json_response(load_fixture("league_teams_sample.json"))
        if path == "/league-matches":
            if season_id in failing_seasons:
                return json_response({"error": "boom"}, status=500)
            if season_id in (SEASONS_A["2025/26"], SEASONS_SB["2025/26"]):
                return json_response(_offset(load_fixture(f"league_matches_page{page}.json"), season_id))
            if season_id in (SEASONS_A["2024/25"], SEASONS_SB["2024/25"]):
                return json_response(_offset(_single_page(load_fixture("league_matches_page1.json"), 10), season_id))
            return json_response(_offset(load_fixture("league_matches_sample.json"), season_id))
        return json_response({"success": False, "message": "not found"}, status=404)

    return router


@pytest.fixture(autouse=True)
def _restore_root_logging() -> Any:
    root = logging.getLogger()
    handlers, level = list(root.handlers), root.level
    yield
    for handler in list(root.handlers):
        root.removeHandler(handler)
    for handler in handlers:
        root.addHandler(handler)
    root.setLevel(level)


def _repository_state() -> dict[str, str]:
    """Digest of every tracked dictionary and of every file under ``data/`` (tests must never touch them)."""
    import hashlib  # noqa: PLC0415

    paths = [*sorted((fs.ROOT / "config").glob("*.json")), *sorted(p for p in (fs.ROOT / "data").rglob("*") if p.is_file())]
    return {str(path.relative_to(fs.ROOT)): hashlib.sha256(path.read_bytes()).hexdigest() for path in paths}


@pytest.fixture(autouse=True)
def _repository_untouched() -> Any:
    """The registry sync writes the dictionary it was read from: no test may let it write the repository's files."""
    before = _repository_state()
    yield
    assert _repository_state() == before, "a test wrote config/*.json or data/ of the repository"


@pytest.fixture
def http_cfg() -> HttpConfig:
    return HttpConfig(timeout_seconds=5.0, max_retries=2, backoff_base_seconds=1.0, backoff_max_seconds=5.0, max_concurrency=2, min_delay_seconds=0.0, cache_ttl_hours=1.0)


@pytest.fixture
def endpoints() -> FootyStatsEndpoints:
    return load_endpoints().footystats


def make_client(
    http: HttpConfig,
    *,
    router: Callable[[httpx.Request], httpx.Response] | None = None,
    cache: RawCache | None = None,
    key: str = KEY,
    endpoints: FootyStatsEndpoints | None = None,
) -> tuple[fs.FootyStatsClient, CountingTransport]:
    transport = CountingTransport(router or make_router())
    client = fs.FootyStatsClient(
        api_key=key,
        endpoints=endpoints or load_endpoints().footystats,
        http=http,
        cache=cache,
        transport=transport,
        sleeper=RecordingSleeper(),
        rng=random.Random(1),
    )
    return client, transport


def synthetic_registry(*, with_history: bool = False) -> TeamRegistry:
    """Small dictionary over the demo teams: exact, alias, fuzzy and historical cases."""
    current = TeamRegistry.from_mapping(
        "2026/27",
        {
            "A": {
                "Arsenal": ["The Gunners"],
                "Tottenham": ["Tottenham Hotspur FC"],
                "Man City": ["Manchester City"],
                "Leicester": [],
                "Southhampton": [],  # deliberate typo: Southampton must reach it through the fuzzy step
            },
            "SB": {"Chelsea": [], "Liverpool": []},
        },
    )
    if not with_history:
        return current
    older = TeamRegistry.from_mapping("2025/26", {"A": {"Cardiff": ["Cardiff City"], "Arsenal": ["Arsenal FC"]}})
    return current.with_historical_aliases(older)


def sample_teams() -> list[fs.TeamInfo]:
    teams = [fs.TeamInfo.from_payload(t) for t in load_fixture("league_teams_sample.json")["data"]]
    return [t for t in teams if t is not None]


def team(teams: list[fs.TeamInfo], clean_name: str) -> fs.TeamInfo:
    return next(t for t in teams if t.clean_name == clean_name)


def _scan_for_key(directory: Path, key: str) -> list[Path]:
    return [p for p in directory.rglob("*") if p.is_file() and key in p.read_text(encoding="utf-8", errors="ignore")]


# ============================================================================ key handling
def test_read_env_file_parses_quotes_exports_and_comments(tmp_path: Path) -> None:
    env = tmp_path / ".env"
    env.write_text(
        "# comment\n\nexport FOOTYSTATS_API_KEY='abc def'\nOTHER=\"quoted\"\nPLAIN=value # trailing\nBAD LINE\n=novalue\n",
        encoding="utf-8",
    )
    assert fs.read_env_file(env) == {"FOOTYSTATS_API_KEY": "abc def", "OTHER": "quoted", "PLAIN": "value"}
    assert fs.read_env_file(tmp_path / "missing.env") == {}


def test_read_env_file_accepts_bom_crlf_and_quoted_values_with_comments(tmp_path: Path) -> None:
    """Windows Notepad writes 'UTF-8 with BOM' and CRLF; a quoted value may be followed by a comment."""
    env = tmp_path / ".env"
    env.write_bytes(
        b"\xef\xbb\xbfFOOTYSTATS_API_KEY=abc123\r\n"
        b'QUOTED="q-key" # my key\r\n'
        b"SINGLE='a # b' # c\r\n"
        b"EMPTY= # only a comment\r\n"
        b"HASH=a#b\r\n"
        b"UNTERMINATED=\"abc # note\r\n"
    )
    assert fs.read_env_file(env) == {
        "FOOTYSTATS_API_KEY": "abc123",
        "QUOTED": "q-key",
        "SINGLE": "a # b",
        "EMPTY": "",
        "HASH": "a#b",
        "UNTERMINATED": '"abc',
    }
    assert fs.find_api_key(None, environ={}, env_file=env) == "abc123"
    quoted = tmp_path / "quoted.env"
    quoted.write_text('FOOTYSTATS_API_KEY="abc123" # my key\n', encoding="utf-8")
    assert fs.find_api_key(None, environ={}, env_file=quoted) == "abc123"
    bom_only = tmp_path / "bom.env"
    bom_only.write_text("FOOTYSTATS_API_KEY=from-bom\n", encoding="utf-8-sig")
    assert fs.resolve_api_key(None, environ={}, env_file=bom_only) == "from-bom"


def test_find_and_resolve_api_key_precedence(tmp_path: Path) -> None:
    env_file = tmp_path / ".env"
    env_file.write_text("FOOTYSTATS_API_KEY=from-file\n", encoding="utf-8")
    assert fs.find_api_key("  explicit ", environ={"FOOTYSTATS_API_KEY": "from-env"}, env_file=env_file) == "explicit"
    assert fs.find_api_key(None, environ={"FOOTYSTATS_API_KEY": "from-env"}, env_file=env_file) == "from-env"
    assert fs.find_api_key("", environ={"FOOTYSTATS_API_KEY": "  "}, env_file=env_file) == "from-file"
    assert fs.find_api_key(None, environ={}, env_file=None) is None
    assert fs.find_api_key(None, key_env="OTHER_KEY", environ={"OTHER_KEY": "k"}, env_file=None) == "k"
    assert fs.resolve_api_key(None, environ={}, env_file=env_file) == "from-file"
    with pytest.raises(fs.MissingApiKeyError) as caught:
        fs.resolve_api_key(None, environ={}, env_file=tmp_path / "absent.env")
    assert isinstance(caught.value, HarvestError)
    assert "FOOTYSTATS_API_KEY" in caught.value.instructions and "footystats.org" in caught.value.instructions
    assert "[url=" not in str(caught.value)


def test_mask_key_hides_query_parameters_mappings_and_literals() -> None:
    assert fs.mask_key("https://x/y?season_id=1&key=abc123&page=2") == "https://x/y?season_id=1&key=***&page=2"
    assert fs.mask_key("params={'season_id': 1, 'key': 'abc123'}") == "params={'season_id': 1, 'key': '***'}"
    assert fs.mask_key("Key=ABC", None) == "Key=***"
    assert fs.mask_key("the secret sk-1 leaked", "sk-1") == "the secret *** leaked"
    assert fs.mask_key("nothing here", "sk-1") == "nothing here"


@pytest.mark.asyncio
async def test_key_injecting_transport_adds_the_key_once() -> None:
    seen: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(str(request.url))
        return json_response({"ok": True})

    transport = fs.KeyInjectingTransport(KEY, httpx.MockTransport(handler))
    async with httpx.AsyncClient(transport=transport) as client:
        await client.get(f"https://{HOST}/league-list", params={"chosen_leagues_only": "true"})
        await client.get(f"https://{HOST}/league-list", params={"key": "already-there"})
    assert seen == [f"https://{HOST}/league-list?chosen_leagues_only=true&key={KEY}", f"https://{HOST}/league-list?key=already-there"]
    assert KEY not in repr(transport)
    with pytest.raises(ValueError):
        fs.KeyInjectingTransport("  ", httpx.MockTransport(handler))


# ============================================================================ client
@pytest.mark.asyncio
async def test_get_all_paginates_and_concatenates(http_cfg: HttpConfig) -> None:
    client, transport = make_client(http_cfg)
    async with client:
        rows = await client.get_all("/league-matches", {"season_id": PAST_A})
    assert len(rows) == 17 and client.pages == 2
    pages = [r.url.params.get("page") for r in transport.requests]
    assert pages == ["1", "2"]
    assert all(r.url.params["key"] == KEY for r in transport.requests)
    assert client.stats["requests"] == 2 and client.stats["pages"] == 2 and client.stats["api_errors"] == 0


@pytest.mark.asyncio
async def test_envelope_success_false_raises_and_is_not_cached(http_cfg: HttpConfig, tmp_path: Path) -> None:
    cache = RawCache(tmp_path / "raw", ttl=timedelta(hours=1))
    client, _ = make_client(http_cfg, cache=cache, key=WRONG_KEY)
    async with client:
        with pytest.raises(fs.FootyStatsError, match="refused") as caught:
            await client.get("/league-list", {"chosen_leagues_only": "true"})
    assert isinstance(caught.value, HarvestError)
    assert WRONG_KEY not in str(caught.value) and "key=" not in str(caught.value)
    assert len(cache) == 0 and client.stats["api_errors"] == 1

    def no_list(request: httpx.Request) -> httpx.Response:
        return json_response({"success": True, "data": {"not": "a list"}})

    other, _ = make_client(http_cfg, router=no_list, cache=cache)
    async with other:
        with pytest.raises(fs.FootyStatsError, match="data list"):
            await other.get("/league-teams", {"season_id": 1})
    assert len(cache) == 0

    def not_json_object(request: httpx.Request) -> httpx.Response:
        return json_response([1, 2, 3])

    third, _ = make_client(http_cfg, router=not_json_object)
    async with third:
        with pytest.raises(fs.FootyStatsError, match="not a JSON object"):
            await third.get("/league-teams", {"season_id": 1})


@pytest.mark.asyncio
async def test_http_errors_surface_as_harvest_errors_without_the_key(http_cfg: HttpConfig) -> None:
    client, transport = make_client(http_cfg, router=lambda request: json_response({"error": "x"}, status=503))
    async with client:
        with pytest.raises(HarvestError) as caught:
            await client.get("/league-list", {"chosen_leagues_only": "true"})
    assert KEY not in str(caught.value) and caught.value.status == 503
    assert transport.calls == http_cfg.max_retries + 1


@pytest.mark.asyncio
async def test_key_never_reaches_cache_logs_or_errors(http_cfg: HttpConfig, tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    caplog.set_level(logging.DEBUG)
    cache_dir = tmp_path / "raw"
    cache = RawCache(cache_dir, ttl=timedelta(hours=1))
    client, transport = make_client(http_cfg, cache=cache)
    async with client:
        leagues = await client.league_list()
        rows = await client.get_all("/league-matches", {"season_id": PAST_A})
        with pytest.raises(HarvestError) as caught:
            await client.get("/missing", {"season_id": 1})
    assert leagues and len(rows) == 17
    # The real request carried the key ...
    assert all(r.url.params["key"] == KEY for r in transport.requests)
    # ... but no cache file name or content, no log record and no exception does.
    assert len(cache) == 3 and _scan_for_key(cache_dir, KEY) == []  # the 404 is never cached
    for record in cache_dir.glob("*.json"):
        assert "key=" not in json.loads(record.read_text(encoding="utf-8"))["url"]
    rendered = "\n".join(f"{r.getMessage()} {r.__dict__}" for r in caplog.records)
    assert "HTTP Request" in rendered  # httpx's own request log was captured and is clean
    assert KEY not in rendered
    assert KEY not in str(caught.value) and KEY not in caught.value.url


@pytest.mark.asyncio
async def test_metadata_budget_warning(http_cfg: HttpConfig, caplog: pytest.LogCaptureFixture) -> None:
    def low_budget(request: httpx.Request) -> httpx.Response:
        payload = load_fixture("league_teams_sample.json")
        payload["metadata"] = {"request_limit": "1800", "request_remaining": "3"}
        return json_response(payload)

    client, _ = make_client(http_cfg, router=low_budget)
    with caplog.at_level(logging.WARNING):
        async with client:
            await client.league_teams(1)
            await client.league_teams(2)
    assert client.request_remaining == 3 and client.stats["request_remaining"] == 3
    assert sum("budget" in r.getMessage() for r in caplog.records) == 1


def test_season_label_from_footystats_year() -> None:
    assert fs.season_label_from_footystats_year(20262027) == "2026/27"
    assert fs.season_label_from_footystats_year("20252026") == "2025/26"
    assert fs.season_label_from_footystats_year(2026) == "2026"
    assert fs.season_label_from_footystats_year(19992000) == "1999/00"
    assert fs.season_label_from_footystats_year(20262028) is None
    assert fs.season_label_from_footystats_year(None) is None
    assert fs.season_label_from_footystats_year("abc") is None


def test_season_completed_rule() -> None:
    today = date(2026, 11, 6)
    assert fs.season_completed("2025/26", today) and not fs.season_completed("2026/27", today)
    assert fs.season_completed("2025", today) and not fs.season_completed("2026", today)
    assert not fs.season_completed("garbage", today)


@pytest.mark.asyncio
async def test_league_list_parses_italian_entries(http_cfg: HttpConfig) -> None:
    client, _ = make_client(http_cfg)
    async with client:
        leagues = await client.league_list()
    by_name = {league.name: league for league in leagues}
    assert "Italy Serie C Group A" in by_name and "Italy Serie B" in by_name
    group_a = by_name["Italy Serie C Group A"]
    assert group_a.league_name == "Serie C Group A" and group_a.country == "Italy"
    assert {ref.label: ref.season_id for ref in group_a.newest_seasons(3)} == SEASONS_A
    assert [ref.label for ref in group_a.newest_seasons(3)] == ["2026/27", "2025/26", "2024/25"]
    assert by_name["Italy Serie B"].newest_seasons(1) == [fs.SeasonRef("2026/27", 17404)]


@pytest.mark.asyncio
async def test_resolve_seasons_newest_three_and_missing_competition(http_cfg: HttpConfig, caplog: pytest.LogCaptureFixture) -> None:
    client, _ = make_client(http_cfg)
    async with client:
        resolved = await client.resolve_seasons(["A", Girone.SB, "a"], seasons=3)
        assert list(resolved) == ["A", "SB"]
        assert [(r.label, r.season_id) for r in resolved["A"]] == [("2026/27", 17487), ("2025/26", 15535), ("2024/25", 12659)]
        assert [(r.label, r.season_id) for r in resolved["SB"]] == [("2026/27", 17404), ("2025/26", 15632), ("2024/25", 12621)]
        two = await client.resolve_seasons(["B"], seasons=2)
        assert [r.season_id for r in two["B"]] == [17492, 15532]
        with pytest.raises(ValueError):
            await client.resolve_seasons(["A"], seasons=0)
        with pytest.raises(ValueError, match="unknown competition code"):
            await client.resolve_seasons(["ZZ"])

    # a league that is not among the chosen ones cannot be resolved: FootyStats refuses its data whatever season id is
    # passed, so season_resolution reports it in not_chosen (with a warning naming it) instead of using the configured
    # ids, and the strict resolve_seasons raises with the advice naming the league and the leagues chosen now
    without_serie_b = load_fixture("league_list_sample.json")
    without_serie_b["data"] = [entry for entry in without_serie_b["data"] if entry["name"] != "Italy Serie B"]
    client, _ = make_client(http_cfg, router=make_router(league_list=without_serie_b))
    with caplog.at_level(logging.WARNING):
        async with client:
            resolution = await client.season_resolution(["A", "SB"])
            with pytest.raises(fs.FootyStatsError) as caught:
                await client.resolve_seasons(["A", "SB"])
    assert resolution.codes == ("A",) and resolution.not_chosen == {"SB": "Italy Serie B"} and not resolution.from_configured_ids
    assert [(r.label, r.season_id) for r in resolution.seasons["A"]] == [("2026/27", 17487), ("2025/26", 15535), ("2024/25", 12659)]
    assert "Italy Serie C Group A" in resolution.chosen_leagues and "Italy Serie B" not in resolution.chosen_leagues
    assert resolution.reason("SB") == fs.not_chosen_reason("Italy Serie B") and resolution.reason("A") is None
    assert fs.competition_season_ids("SB"), "the configured ids exist but are never used for a league the account has not chosen"
    warning = next(r for r in caplog.records if r.getMessage() == "league not among the account's chosen leagues, competition skipped")
    assert warning.league == "Italy Serie B" and warning.competition == "SB" and "footystats.org" in warning.advice
    message = str(caught.value)
    assert "'Italy Serie B'" in message and "footystats.org" in message and "Italy Serie C Group A" in message


@pytest.mark.asyncio
async def test_resolve_seasons_falls_back_to_configured_ids_when_list_is_down(http_cfg: HttpConfig, endpoints: FootyStatsEndpoints, caplog: pytest.LogCaptureFixture, monkeypatch: pytest.MonkeyPatch) -> None:
    def list_down(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/league-list":
            return json_response({"error": "down"}, status=503)
        return make_router()(request)

    client, _ = make_client(http_cfg, router=list_down, endpoints=endpoints)
    with caplog.at_level(logging.WARNING):
        async with client:
            resolved = await client.resolve_seasons(["SB", "A"], seasons=2)
    assert [(r.label, r.season_id) for r in resolved["SB"]] == [("2026/27", 17404), ("2025/26", 15632)]
    assert [(r.label, r.season_id) for r in resolved["A"]] == [("2026/27", 17487), ("2025/26", 15535)]
    assert any("configured season ids" in r.getMessage() for r in caplog.records)
    bare = FootyStatsEndpoints()  # no ids in endpoints.toml: those of config/competitions.toml still apply
    client, _ = make_client(http_cfg, router=list_down, endpoints=bare)
    async with client:
        assert [r.season_id for r in (await client.resolve_seasons(["DA"], seasons=3))["DA"]] == [17664, 15710, 13475]
    assert fs.competition_season_ids("DA", bare) == {"2024/25": 13475, "2025/26": 15710, "2026/27": 17664}
    monkeypatch.setattr(fs, "competition_season_ids", lambda code, endpoints=None: {})  # no ids anywhere: the error propagates
    client, _ = make_client(http_cfg, router=list_down, endpoints=bare)
    async with client:
        with pytest.raises(HarvestError):
            await client.resolve_seasons(["A"])


@pytest.mark.asyncio
async def test_league_teams_and_matches(http_cfg: HttpConfig) -> None:
    client, transport = make_client(http_cfg)
    async with client:
        teams = await client.league_teams(CURRENT_A, stable=True)
        matches = await client.league_matches(CURRENT_A)
    assert len(teams) == 20 and len(matches) == 17
    spurs = team(teams, "Tottenham Hotspur")
    assert spurs == fs.TeamInfo(92, "Tottenham Hotspur FC", "Tottenham Hotspur", "Tottenham Hotspur FC", ("Spurs", "The Lilywhites"))
    assert spurs.spellings == ("Tottenham Hotspur", "Tottenham Hotspur FC", "Spurs", "The Lilywhites")
    assert team(teams, "Arsenal").spellings == ("Arsenal", "Arsenal FC", "The Gunners")
    assert [r.url.params["season_id"] for r in transport.requests] == [str(CURRENT_A), str(CURRENT_A)]


# ============================================================================ parsers
def _match(match_id: int, season_payload: str = "league_matches_sample.json") -> dict[str, Any]:
    return next(m for m in load_fixture(season_payload)["data"] if m["id"] == match_id)


def test_parse_match_complete_with_statistics() -> None:
    names = {149: "Manchester Utd", 108: "Leicester"}
    record = fs.parse_match(_match(453873), "A", names, now=NOW)
    assert isinstance(record, MatchRecord)
    assert record == MatchRecord("FS-453873", date(2018, 8, 10), 1, Girone.A, "Manchester Utd", "Leicester", 2, 1, 7.0, 7.0, 6.0, 4.0, 2.0, 5.0, 1.12, 1.33)
    # names fall back to the payload spelling when the id is unknown; the competition code is stored as given
    other = fs.parse_match(_match(453874), Girone.SB, {}, now=NOW)
    assert isinstance(other, MatchRecord) and other.home_team == "Newcastle United" and other.girone == Girone.SB


def test_parse_match_missing_statistics_rules() -> None:
    report = fs.ParseReport()
    record = fs.parse_match(_match(900005), "A", {}, now=NOW, report=report)
    assert isinstance(record, MatchRecord) and (record.home_goals, record.away_goals) == (0, 3)
    assert all(getattr(record, f) is None for f in ("home_shots", "away_shots", "home_shots_target", "away_shots_target", "home_corners", "away_corners", "home_xg", "away_xg"))
    assert report.complete == 1 and report.missing_shots == 1 and report.missing_xg == 1
    # shots present, xG not covered: only xG becomes missing; a lone -1 is missing on its own
    payload = {**_match(453873), "team_a_xg": 0, "team_b_xg": 0, "team_b_corners": -1}
    stats, shots_missing, xg_missing = fs.parse_stats(payload)
    assert not shots_missing and xg_missing
    assert stats["home_shots"] == 7.0 and stats["home_xg"] is None and stats["away_xg"] is None
    assert stats["home_corners"] == 2.0 and stats["away_corners"] is None


def test_parse_match_incomplete_suspended_canceled(caplog: pytest.LogCaptureFixture) -> None:
    report = fs.ParseReport()
    fixture = fs.parse_match(_match(900001), "A", {149: "Manchester Utd"}, now=NOW, report=report)
    kickoff = datetime.fromtimestamp(int(_match(900001)["date_unix"]), tz=UTC)
    assert fixture == Fixture("FS-900001", date(2027, 1, 15), 38, Girone.A, "Manchester Utd", "Leicester City", kickoff=kickoff, status="scheduled")
    assert isinstance(fixture, Fixture) and fixture.kickoff is not None and fixture.kickoff.tzinfo is not None
    # the same incomplete match seen more than three hours after kick-off is POSTPONED: kept, with its original kick-off
    late = fs.parse_match(_match(900001), "A", {}, now=kickoff + timedelta(hours=3, minutes=1), report=report)
    assert isinstance(late, Fixture) and late.status == "postponed" and late.kickoff == kickoff and late.date == date(2027, 1, 15) and late.round == 38
    # within three hours of kick-off it is in play: kept as a SCHEDULED fixture with its (past) kick-off, never
    # dropped, so the round still lists it (the score comes with the next harvest)
    for elapsed in (timedelta(hours=2), fs.IN_PLAY_WINDOW):
        playing = fs.parse_match(_match(900001), "A", {}, now=kickoff + elapsed, report=report)
        assert isinstance(playing, Fixture) and playing.status == "scheduled" and playing.kickoff == kickoff and playing.round == 38
    # the goal fields do not matter any more: the kick-off decides
    without_goals = {**_match(900001), "homeGoalCount": None, "awayGoalCount": None}
    assert fs.parse_match(without_goals, "A", {}, now=datetime(2027, 1, 16, tzinfo=UTC), report=report).status == "postponed"
    assert fs.parse_match(_match(900003), "A", {}, now=NOW, report=report) is None  # suspended
    assert fs.parse_match(_match(900004), "A", {}, now=NOW, report=report) is None  # canceled
    assert fs.parse_match({**_match(453873), "status": "live"}, "A", {}, now=NOW, report=report) is None
    assert report.fixtures == 5 and report.postponed == 2 and report.in_play == 2 and report.incomplete_past == 4
    assert report.as_dict()["postponed"] == 2 and report.as_dict()["in_play"] == 2
    assert dict(report.skipped_status) == {"suspended": 1, "canceled": 1, "live": 1}
    with caplog.at_level(logging.WARNING):
        assert fs.parse_match({**_match(453873), "id": None}, "A", {}, now=NOW, report=report) is None
        assert fs.parse_match({**_match(453873), "date_unix": 0}, "A", {}, now=NOW, report=report) is None
        assert fs.parse_match({**_match(453873), "home_name": "", "homeID": 999}, "A", {}, now=NOW, report=report) is None
        assert fs.parse_match(_match(453873), "A", {149: "Same", 108: "Same"}, now=NOW, report=report) is None
        assert fs.parse_match({**_match(453873), "homeGoalCount": None}, "A", {}, now=NOW, report=report) is None
    assert report.skipped_invalid == 5
    assert {"match without id skipped", "match without kick-off time skipped", "match without team names skipped", "match with identical teams skipped", "complete match without a score skipped"} <= {r.getMessage() for r in caplog.records}


def test_parse_match_round_fallbacks() -> None:
    base = _match(453873)
    assert fs.parse_match(base, "A", {}, now=NOW).round == 1
    assert fs.parse_match({**base, "game_week": -1, "revised_game_week": 5}, "A", {}, now=NOW).round == 5
    assert fs.parse_match({**base, "game_week": None, "revised_game_week": None}, "A", {}, now=NOW).round == 0
    assert fs.parse_match({**base, "game_week": "7"}, "A", {}, now=NOW).round == 7


def test_parse_odds_markets_and_skipping() -> None:
    quotes = fs.parse_odds(_match(453873), "FS-453873")
    assert [q.market for q in quotes] == [Market.MATCH_ODDS, Market.OVER_UNDER_15, Market.OVER_UNDER_25, Market.OVER_UNDER_35, Market.BTTS]
    assert quotes[0].odds == {"1": 1.37, "X": 4.98, "2": 9.81} and quotes[0].bookmaker == "FootyStats"
    assert quotes[2].odds == {"Over 2.5": 2.0, "Under 2.5": 1.75} and quotes[4].odds == {"GG": 2.05, "NG": 1.69}
    partial = fs.parse_odds(_match(900002), "FS-900002", bookmaker="Book")
    assert [q.market.value for q in partial] == ["OU1.5", "OU2.5", "OU3.5"] and all(q.bookmaker == "Book" for q in partial)
    assert fs.parse_odds({**_match(453873), "odds_ft_over25": 1.0, "odds_btts_no": None, "odds_ft_x": "bad"}, "x") == [
        fs.OddsQuote("x", "FootyStats", Market.OVER_UNDER_15, {"Over 1.5": 1.33, "Under 1.5": 3.25}),
        fs.OddsQuote("x", "FootyStats", Market.OVER_UNDER_35, {"Over 3.5": 3.6, "Under 3.5": 1.29}),
    ]
    assert fs.parse_odds({}, "x") == []


# ============================================================================ mapper
def test_mapper_exact_alias_fuzzy_and_unmapped(tmp_path: Path) -> None:
    registry = synthetic_registry()
    mapper = fs.FootyStatsTeamMapper(registry, TeamMatcher(registry), mapping_path=tmp_path / "map.json")
    teams = sample_teams()
    arsenal = mapper.resolve(team(teams, "Arsenal"), "A", "2026/27")
    assert (arsenal.canonical, arsenal.method, arsenal.current) == ("Arsenal", "exact", True)
    spurs = mapper.resolve(team(teams, "Tottenham Hotspur"), "A", "2026/27")
    assert (spurs.canonical, spurs.method) == ("Tottenham", "alias")
    city = mapper.resolve(team(teams, "Manchester City"), Girone.A, "2026/27")
    assert (city.canonical, city.method) == ("Man City", "alias")
    leicester = mapper.resolve(team(teams, "Leicester City"), "A", "2026/27")
    assert (leicester.canonical, leicester.method) == ("Leicester", "exact")  # through alt_names
    saints = mapper.resolve(team(teams, "Southampton"), "A", "2026/27")
    assert (saints.canonical, saints.method) == ("Southhampton", "fuzzy")
    burnley = mapper.resolve(team(teams, "Burnley"), "A", "2026/27")
    assert (burnley.canonical, burnley.method, burnley.current, burnley.historical) == ("Burnley", "none", False, False)
    assert mapper.unmapped_teams == {"Burnley": 1}
    # a Serie B team restricted to the SB composition; in the current season a club plays in ONE competition, so a
    # spelling of a current Serie B club listed in girone A is not merged with it: it takes the sync's collision name
    chelsea = mapper.resolve(team(teams, "Chelsea"), "SB", "2026/27")
    assert (chelsea.canonical, chelsea.method) == ("Chelsea", "exact")
    cross = mapper.resolve(team(teams, "Liverpool"), "A", "2026/27")
    assert (cross.canonical, cross.method, cross.current) == ("Liverpool Serie C", "none", False)
    assert mapper.unmapped_teams == {"Burnley": 1, "Liverpool Serie C": 1}
    # in a past season clubs do change competition: the same spelling is accepted there
    past = fs.FootyStatsTeamMapper(registry, TeamMatcher(registry), mapping_path=tmp_path / "past.json")
    assert past.resolve(team(teams, "Liverpool"), "A", "2025/26", current=False).canonical == "Liverpool"
    names = mapper.names_for(teams, "A", "2026/27")
    assert names[59] == "Arsenal" and names[92] == "Tottenham" and names[145] == "Burnley" and len(names) == 20
    path = mapper.save()
    saved = json.loads(path.read_text(encoding="utf-8"))["teams"]
    assert saved["92"]["canonical"] == "Tottenham" and saved["92"]["method"] == "alias" and saved["92"]["seasons"] == ["2026/27"]
    assert saved["146"] == {"canonical": "Southhampton", "clean_name": "Southampton", "competition": "A", "seasons": ["2026/27"], "method": "fuzzy"}


def test_mapper_persisted_ids_beat_fuzzy_but_not_dictionary_hits(tmp_path: Path) -> None:
    mapping = tmp_path / "map.json"
    mapping.write_text(
        json.dumps(
            {
                "teams": {
                    "59": {"canonical": "Wrong Arsenal", "clean_name": "Arsenal", "competition": "A", "seasons": ["2025/26"]},
                    "146": {"canonical": "Saints", "clean_name": "Southampton", "competition": "A", "seasons": ["2025/26"]},
                    "145": {"canonical": "Burnley Persisted", "clean_name": "Burnley", "competition": "A", "seasons": ["2025/26"]},
                    "bad": {"canonical": ""},
                }
            }
        ),
        encoding="utf-8",
    )
    registry = synthetic_registry()
    mapper = fs.FootyStatsTeamMapper(registry, mapping_path=mapping)
    teams = sample_teams()
    assert mapper.resolve(team(teams, "Arsenal"), "A", "2026/27").canonical == "Arsenal"  # exact beats persisted
    saints = mapper.resolve(team(teams, "Southampton"), "A", "2026/27")
    assert (saints.canonical, saints.method) == ("Saints", "persisted")  # persisted beats fuzzy
    burnley = mapper.resolve(team(teams, "Burnley"), "A", "2026/27")
    assert (burnley.canonical, burnley.method) == ("Burnley Persisted", "persisted")  # persisted beats unmapped
    assert mapper.unmapped_teams == {}
    mapper.save()
    saved = json.loads(mapping.read_text(encoding="utf-8"))["teams"]
    assert saved["59"]["canonical"] == "Arsenal" and saved["59"]["seasons"] == ["2026/27", "2025/26"]
    assert "bad" not in saved and saved["146"]["method"] == "persisted"
    # the merged file is read back identically on the next run
    again = fs.FootyStatsTeamMapper(registry, mapping_path=mapping)
    assert again.persisted[59]["canonical"] == "Arsenal" and again.persisted[145]["canonical"] == "Burnley Persisted"


def test_mapper_historical_teams_and_unreadable_mapping(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    (tmp_path / "map.json").write_text("{not json", encoding="utf-8")
    registry = synthetic_registry(with_history=True)
    with caplog.at_level(logging.WARNING):
        mapper = fs.FootyStatsTeamMapper(registry, TeamMatcher(registry), mapping_path=tmp_path / "map.json")
    assert any("unreadable" in r.getMessage() for r in caplog.records)
    teams = sample_teams()
    cardiff = mapper.resolve(team(teams, "Cardiff City"), "A", "2025/26", current=False)
    assert (cardiff.canonical, cardiff.method, cardiff.historical, cardiff.current) == ("Cardiff", "alias", True, False)
    assert mapper.resolve(team(teams, "Arsenal"), "A", "2025/26", current=False).canonical == "Arsenal"
    assert mapper.historical_teams == ("Cardiff",)
    # restricted current-season lookup does not return a historical team ...
    restricted = mapper.resolve(team(teams, "Cardiff City"), "A", "2026/27", current=True)
    # ... unless nothing else matches: the unrestricted retry then finds the historical spelling
    assert restricted.canonical == "Cardiff" and restricted.historical


# ============================================================================ provider
@pytest.fixture
def provider_factory(http_cfg: HttpConfig, tmp_path: Path) -> Callable[..., tuple[fs.FootyStatsProvider, CountingTransport]]:
    def make(
        *,
        router: Callable[[httpx.Request], httpx.Response] | None = None,
        registry: TeamRegistry | None = None,
        cache: RawCache | None = None,
        now: datetime = NOW,
    ) -> tuple[fs.FootyStatsProvider, CountingTransport]:
        client, transport = make_client(http_cfg, router=router, cache=cache)
        reg = registry if registry is not None else synthetic_registry(with_history=True)
        mapper = fs.FootyStatsTeamMapper(reg, TeamMatcher(reg), mapping_path=tmp_path / "footystats_teams.json")
        # an in-memory registry is never synchronised (it was not read from a dictionary file); config_dir points
        # at an empty temporary directory anyway, so nothing could be written to the repository's config/
        return fs.FootyStatsProvider(client, mapper, client.endpoints, now=lambda: now, config_dir=tmp_path / "no-config"), transport

    return make


@pytest.mark.asyncio
async def test_harvest_end_to_end(provider_factory: Callable[..., Any], tmp_path: Path) -> None:
    cache = RawCache(tmp_path / "raw", ttl=timedelta(hours=1))
    provider, transport = provider_factory(cache=cache)
    async with provider.client:
        result = await provider.harvest(["A", "SB"], seasons=3)
    df = result.matches
    assert list(df.columns) == list(MATCH_COLUMNS)
    for column, dtype in MATCH_DTYPES.items():
        assert str(df[column].dtype) == dtype, column
    assert df["match_id"].is_unique and df["match_id"].str.startswith("FS-").all()
    assert sorted(df["girone"].unique().tolist()) == ["A", "SB"]
    assert result.seasons == {"A": ["2026/27", "2025/26", "2024/25"], "SB": ["2026/27", "2025/26", "2024/25"]}
    assert result.n_matches_by_competition == {"A": {"2026/27": 13, "2025/26": 13, "2024/25": 10}, "SB": {"2026/27": 13, "2025/26": 13, "2024/25": 10}}
    assert result.n_matches_by_season == {"2026/27": 26, "2025/26": 26, "2024/25": 20}
    assert len(df) == 72 == len(result.records) and sum(result.n_matches_by_season.values()) == 72
    # fixtures only from the current season of each competition, deduplicated; odds only for fixtures
    assert [f.match_id for f in result.fixtures] == [f"FS-{CURRENT_A * 1_000_000 + 900001}", f"FS-{CURRENT_A * 1_000_000 + 900002}", f"FS-{17404 * 1_000_000 + 900001}", f"FS-{17404 * 1_000_000 + 900002}"]
    assert all(f.date == date(2027, 1, 15) and f.status == "scheduled" and f.kickoff is not None for f in result.fixtures)
    assert {f.girone for f in result.fixtures} == {Girone.A, Girone.SB}
    assert result.registry_sync is None and result.n_postponed == 0  # an in-memory registry is never synchronised
    fixture_ids = {f.match_id for f in result.fixtures}
    assert result.odds and {q.match_id for q in result.odds} <= fixture_ids
    assert len(result.odds) == 2 * (5 + 3)  # per competition: full book for 900001, three OU markets for 900002
    assert sorted({q.market.value for q in result.odds}) == ["1X2", "GG/NG", "OU1.5", "OU2.5", "OU3.5"]
    # the 900005 match without statistics kept its result but no stats
    broken = df.loc[df["match_id"] == f"FS-{CURRENT_A * 1_000_000 + 900005}"].iloc[0]
    assert (broken["home_goals"], broken["away_goals"]) == (0, 3) and pd.isna(broken["home_shots"]) and pd.isna(broken["home_xg"])
    good = df.loc[df["match_id"] == f"FS-{CURRENT_A * 1_000_000 + 453873}"].iloc[0]
    # Manchester United is in no dictionary: the raw clean name is kept and the team is reported as unmapped
    assert good["home_team"] == "Manchester United" and result.unmapped_teams["Manchester United"] >= 1
    assert good["away_team"] == "Leicester" and good["home_xg"] == 1.12 and good["away_shots"] == 7.0
    assert "Cardiff" in result.historical_teams and "Arsenal" not in result.historical_teams
    assert result.unmapped_teams["Burnley"] >= 1 and "Arsenal" not in result.unmapped_teams
    assert result.failures == {} and result.skipped == {} and result.problem_for("A") is None and result.problem_for(Girone.SB) is None
    assert result.competitions == ("A", "SB")
    assert set(result.stats) >= {"requests", "retries", "cache_hits", "errors", "pages", "api_errors", "request_remaining"}
    assert result.stats["requests"] == 1 + 2 * (3 * 2 + 1)  # league list + per competition: 3 teams + 3 matches payloads, one extra page
    assert "matches=72 fixtures=4 odds=16" in result.summary()
    assert result.season_lines()[0] == "  A: 2026/27: 13 matches, 2025/26: 13 matches, 2024/25: 10 matches"
    mapping = json.loads((tmp_path / "footystats_teams.json").read_text(encoding="utf-8"))["teams"]
    assert mapping["59"]["canonical"] == "Arsenal" and sorted(mapping["59"]["seasons"], reverse=True) == ["2026/27", "2025/26", "2024/25"]
    assert KEY not in (tmp_path / "footystats_teams.json").read_text(encoding="utf-8") and _scan_for_key(tmp_path, KEY) == []

    # Second run on the same cache: every payload is served from disk (completed seasons as stable entries).
    before = transport.calls
    again, _ = provider_factory(cache=cache)
    async with again.client:
        second = await again.harvest(["A", "SB"], seasons=3)
    assert transport.calls == before and second.stats["cache_hits"] == result.stats["requests"]
    pd.testing.assert_frame_equal(second.matches, df)


@pytest.mark.asyncio
async def test_harvest_skips_unsupported_codes_and_records_season_failures(provider_factory: Callable[..., Any]) -> None:
    provider, _ = provider_factory(router=make_router(failing_seasons={OLD_A}))
    async with provider.client:
        result = await provider.harvest(["A", "ZZ"], seasons=3)
    assert list(result.failures) == ["A:2024/25"] and "HTTP 500" in result.failures["A:2024/25"] and KEY not in result.failures["A:2024/25"]
    assert result.skipped == {"ZZ": "the match schema (Girone) does not accept competition code 'ZZ' yet"}
    assert result.n_matches_by_competition == {"A": {"2026/27": 13, "2025/26": 13}} and len(result.matches) == 26
    assert result.problem_for("A") is None and result.problem_for("ZZ").startswith("the match schema")
    assert result.problem_for("B") == "competition B was not harvested"
    assert result.season_lines()[-1] == "  ZZ: skipped (the match schema (Girone) does not accept competition code 'ZZ' yet)"

    only_bad, _ = provider_factory(router=make_router(failing_seasons=set(SEASONS_A.values())))
    async with only_bad.client:
        with pytest.raises(HarvestError):
            await only_bad.harvest(["A"], seasons=2)

    nothing, transport = provider_factory()
    async with nothing.client:
        empty = await nothing.harvest(["ZZ"])
    assert empty.matches.empty and empty.skipped and transport.calls == 0
    with pytest.raises(ValueError):
        await nothing.harvest(["A"], seasons=0)


@pytest.mark.asyncio
async def test_harvest_without_odds_and_missing_competition_is_skipped(provider_factory: Callable[..., Any], caplog: pytest.LogCaptureFixture) -> None:
    provider, _ = provider_factory()
    async with provider.client:
        result = await provider.harvest(["A"], seasons=1, with_odds=False)
    assert len(result.fixtures) == 2 and result.odds == [] and len(result.matches) == 13
    assert result.n_matches_by_season == {"2026/27": 13}

    # the league is not among the account's chosen leagues: nothing of it is fetched (the API would refuse it) and the
    # competition is a skip, not a failure or an exception -- the configured season ids are never used for it
    only_serie_a = load_fixture("league_list_sample.json")
    only_serie_a["data"] = [e for e in only_serie_a["data"] if e["name"] == "Italy Serie A"]
    provider, transport = provider_factory(router=make_router(league_list=only_serie_a))
    with caplog.at_level(logging.WARNING):
        async with provider.client:
            skipped = await provider.harvest(["A"], seasons=1, with_odds=False)
    assert skipped.matches.empty and skipped.fixtures == [] and skipped.failures == {} and skipped.seasons == {}
    assert skipped.not_chosen == {"A": "Italy Serie C Group A"} and skipped.skipped == {"A": fs.not_chosen_reason("Italy Serie C Group A")}
    assert skipped.problem_for("A") == skipped.skipped["A"] and "not_chosen=A" in skipped.summary() and skipped.initialised_competitions == ()
    assert [r.url.path for r in transport.requests] == ["/league-list"]
    assert any(r.getMessage() == "competitions skipped: leagues not among the account's chosen leagues" for r in caplog.records)
    # the strict resolution (used by --init-registry) still stops with the advice
    provider, _ = provider_factory(router=make_router(league_list=only_serie_a))
    async with provider.client:
        with pytest.raises(fs.FootyStatsError, match="chosen leagues"):
            await provider.init_registry(["A"])


@pytest.mark.asyncio
async def test_init_registry_and_write_registry_update(provider_factory: Callable[..., Any], tmp_path: Path) -> None:
    provider, _ = provider_factory()
    async with provider.client:
        teams = await provider.init_registry(["A", "SB"], "2026_27")
        newest = await provider.init_registry(["A"])
        with pytest.raises(fs.FootyStatsError, match="not listed"):
            await provider.init_registry(["A"], "1999/00")
    assert set(teams) == {"A", "SB"} and len(teams["A"]) == 20 and len(newest["A"]) == 20

    path = tmp_path / "teams_2026_27.json"
    path.write_text(
        json.dumps({"season": "2026/27", "gironi": {"A": {"Arsenal": ["Gunners (old alias)"], "Tottenham": ["Spurs"]}, "SB": {"Chelsea": []}}}),
        encoding="utf-8",
    )
    registry = synthetic_registry(with_history=True)
    diff = fs.write_registry_update(path, "2026_27", {"A": teams["A"], "ZZ": teams["A"][:1]}, matcher=TeamMatcher(registry))
    saved = json.loads(path.read_text(encoding="utf-8"))
    group_a = saved["gironi"]["A"]
    assert saved["season"] == "2026/27" and not diff.created
    # existing aliases kept, new ones appended; a spelling that normalises like the canonical ("Arsenal FC") adds nothing
    assert group_a["Arsenal"] == ["Gunners (old alias)", "The Gunners"]
    assert group_a["Tottenham"] == ["Spurs", "Tottenham Hotspur", "The Lilywhites"]
    assert "Cardiff" in group_a and group_a["Cardiff"] == ["Cardiff City", "Bluebirds"]  # historical name revived
    assert "Burnley" in group_a and group_a["Burnley"] == ["The Clarets"]
    assert "Chelsea" not in group_a and saved["gironi"]["SB"] == {"Chelsea": []}  # listed in SB by the file: conflict, left alone
    assert diff.conflicts == {"Chelsea": "FootyStats lists it in A, the file in SB; left unchanged"}
    assert "Cardiff" in diff.added_teams["A"] and "Burnley" in diff.added_teams["A"] and "Arsenal" not in diff.added_teams["A"]
    assert diff.new_aliases["Arsenal"] == ["The Gunners"]
    assert diff.skipped == {"ZZ": "the match schema (Girone) does not accept competition code 'ZZ' yet"}
    text = diff.describe()
    assert "updated" in text and "conflict Chelsea" in text and "ZZ: skipped" in text
    # a brand new file
    fresh = fs.write_registry_update(tmp_path / "teams_2027_28.json", "2027/28", {"A": teams["A"][:2]}, matcher=None)
    assert fresh.created and json.loads((tmp_path / "teams_2027_28.json").read_text(encoding="utf-8"))["gironi"]["A"] == {
        "Arsenal": ["The Gunners"],
        "Tottenham Hotspur": ["Spurs", "The Lilywhites"],
    }
    # idempotent: a second pass changes nothing
    second = fs.write_registry_update(tmp_path / "teams_2027_28.json", "2027/28", {"A": teams["A"][:2]}, matcher=None)
    assert second.added_teams == {} and second.new_aliases == {} and second.unchanged_teams == 2
    with pytest.raises(ValueError):
        (tmp_path / "broken.json").write_text("[]", encoding="utf-8")
        fs.write_registry_update(tmp_path / "broken.json", "2027/28", {"A": []})


def test_write_registry_update_can_replace_aliases_and_reports_unlisted_teams(tmp_path: Path) -> None:
    teams = sample_teams()[:2]  # Arsenal, Tottenham Hotspur
    path = tmp_path / "teams_2026_27.json"
    original = {"season": "2026/27", "competition": "hand-written label", "gironi": {"A": {"Arsenal": ["Old Alias"], "Ghost FC": ["Ghosts"]}}}
    path.write_text(json.dumps(original), encoding="utf-8")
    # default: existing aliases are kept, teams FootyStats no longer lists are reported but kept
    kept = fs.write_registry_update(path, "2026_27", {"A": teams}, matcher=None)
    saved = json.loads(path.read_text(encoding="utf-8"))
    assert saved["gironi"]["A"]["Arsenal"] == ["Old Alias", "The Gunners"] and saved["gironi"]["A"]["Ghost FC"] == ["Ghosts"]
    assert saved["competition"] == "hand-written label"  # an existing file keeps its metadata
    assert kept.removed_aliases == {} and kept.unlisted_teams == {"A": ["Ghost FC"]}
    assert "A: 1 teams in the file no longer listed by FootyStats" in kept.describe() and "Ghost FC" in kept.describe()
    # keep_existing_aliases=False replaces the aliases with the FootyStats spellings and reports what went
    replaced = fs.write_registry_update(path, "2026_27", {"A": teams}, matcher=None, keep_existing_aliases=False)
    saved = json.loads(path.read_text(encoding="utf-8"))
    assert saved["gironi"]["A"]["Arsenal"] == ["The Gunners"] and "Old Alias" not in json.dumps(saved["gironi"]["A"]["Arsenal"])
    assert replaced.removed_aliases == {"Arsenal": ["Old Alias"]} and replaced.new_aliases == {} and replaced.added_teams == {}
    assert replaced.unchanged_teams == 1  # Tottenham Hotspur: nothing added, nothing dropped
    assert "aliases removed from 1 teams: Arsenal -> Old Alias" in replaced.describe()
    assert saved["gironi"]["A"]["Ghost FC"] == ["Ghosts"]  # never removed, only reported
    # a competition that is not part of the listing is not reported as unlisted
    sb = fs.write_registry_update(path, "2026_27", {"SB": teams[:1]}, matcher=None)
    assert sb.unlisted_teams == {} and sb.conflicts == {"Arsenal": "FootyStats lists it in SB, the file in A; left unchanged"}


def test_configuration_drives_default_competitions_labels_and_key_instructions(tmp_path: Path) -> None:
    assert fs.DEFAULT_COMPETITIONS == tuple(member.value for member in Girone) and "SB" in fs.DEFAULT_COMPETITIONS
    assert fs.competition_label(["A", "B", "C"]) == "Serie C" and fs.competition_label(["SB"]) == "Serie B"
    assert fs.competition_label(["SB", "A"]) == "Serie C + Serie B" and fs.competition_label([Girone.A, "ZZ"]) == "Serie C + ZZ"
    assert fs.competition_label([]) == ""
    assert fs.configured_league_names() == [member.footystats_name for member in Girone]
    custom = FootyStatsEndpoints(league_names={"ZZ": "Zed League"})
    assert fs.configured_league_names(custom, codes=["A", "ZZ", "QQ", "A"]) == ["Italy Serie C Group A", "Zed League"]
    assert fs.configured_league_names(codes=["QQ"]) == list(fs._FALLBACK_LEAGUE_NAMES.values())  # last resort
    with pytest.raises(ValueError, match="competitions.toml"):
        fs.footystats_league_name("QQ")
    # the key instructions list the leagues of the configured competitions, not a frozen table
    error = fs.MissingApiKeyError("FootyStats API key not configured")
    assert error.league_names == tuple(member.footystats_name for member in Girone)
    assert all(member.footystats_name in error.instructions for member in Girone)
    with pytest.raises(fs.MissingApiKeyError) as caught:
        fs.resolve_api_key(None, environ={}, env_file=None, endpoints=custom)
    assert "Italy Serie B" in caught.value.instructions
    only = fs.MissingApiKeyError("x", league_names=["Only League"])
    assert "Only League" in only.instructions and "Italy Serie B" not in only.instructions
    # a new dictionary file is labelled after the families of the codes it holds
    teams = sample_teams()[:1]
    fs.write_registry_update(tmp_path / "teams_sb.json", "2026/27", {"SB": teams}, matcher=None)
    assert json.loads((tmp_path / "teams_sb.json").read_text(encoding="utf-8"))["competition"] == "Serie B"
    fs.write_registry_update(tmp_path / "teams_mixed.json", "2026/27", {"A": teams, "SB": teams, "ZZ": teams}, matcher=None)
    assert json.loads((tmp_path / "teams_mixed.json").read_text(encoding="utf-8"))["competition"] == "Serie C + Serie B"
    help_text = " ".join(fs._build_parser().format_help().split())  # argparse wraps lines at the terminal width
    assert "SB = Serie B" in help_text and "A = Serie C Girone A" in help_text


# ============================================================================ launchers
def _launcher_snippet(path: Path, quote: str) -> str:
    found = re.search(rf"python -c {quote}([^{quote}]*find_api_key[^{quote}]*){quote}", path.read_text(encoding="utf-8"))
    assert found is not None, f"{path.name} does not delegate the key check to Python"
    return found.group(1)


def test_launchers_detect_the_key_with_the_python_parser() -> None:
    """avvia.bat and avvia.command must agree with the CLI on whether a key is configured (finding: findstr
    at column 0 rejected 'export', leading spaces and a BOM that read_env_file accepts)."""
    bat, cmd = _launcher_snippet(fs.ROOT / "avvia.bat", '"'), _launcher_snippet(fs.ROOT / "avvia.command", "'")
    assert bat == cmd and "find_api_key" in bat and "key_env" in bat and "findstr" not in (fs.ROOT / "avvia.bat").read_text(encoding="utf-8")
    base_env = {k: v for k, v in os.environ.items() if k != "FOOTYSTATS_API_KEY"}
    with_key = subprocess.run([sys.executable, "-c", bat], cwd=fs.ROOT, env={**base_env, "FOOTYSTATS_API_KEY": KEY}, capture_output=True, text=True)
    assert with_key.returncode == 0, with_key.stderr
    assert KEY not in with_key.stdout + with_key.stderr
    if not fs.DEFAULT_ENV_FILE.exists():  # a developer's own .env legitimately makes the check succeed
        without = subprocess.run([sys.executable, "-c", bat], cwd=fs.ROOT, env=base_env, capture_output=True, text=True)
        assert without.returncode == 1, without.stderr


# ============================================================================ CLI
@pytest.fixture
def cli_env(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> CountingTransport:
    """No key in the environment or in a .env file; the client talks to the fixture router.

    The CLI defaults (``--config-dir``, ``--endpoints``, ``--mapping``) point at a private copy of ``config/``
    and at a temporary mapping: the registry sync writes the dictionary it reads, never the repository's.
    """
    import shutil  # noqa: PLC0415

    config_copy = tmp_path / "config-copy"
    shutil.copytree(fs.CONFIG_DIR, config_copy)
    monkeypatch.setattr(fs, "CONFIG_DIR", config_copy)
    monkeypatch.setattr(fs, "DEFAULT_MAPPING_PATH", tmp_path / "data" / "footystats_teams.json")
    monkeypatch.delenv("FOOTYSTATS_API_KEY", raising=False)
    monkeypatch.setattr(fs, "DEFAULT_ENV_FILE", tmp_path / "absent.env")
    monkeypatch.setattr(fs, "_utc_now", lambda: NOW)
    transport = CountingTransport(make_router())
    real_client = fs.FootyStatsClient

    def with_transport(**kwargs: Any) -> fs.FootyStatsClient:
        kwargs.setdefault("transport", transport)
        kwargs.setdefault("sleeper", RecordingSleeper())
        return real_client(**kwargs)

    monkeypatch.setattr(fs, "FootyStatsClient", with_transport)
    return transport


def test_cli_exit_code_2_without_key(cli_env: CountingTransport, tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    code = fs.main(["--competitions", "A", "--cache-dir", str(tmp_path / "raw")])
    captured = capsys.readouterr()
    assert code == 2 and cli_env.calls == 0
    assert "FOOTYSTATS_API_KEY" in captured.err and "footystats.org" in captured.err and "--key" in captured.err
    # a .env file next to the code is enough
    env_file = tmp_path / "present.env"
    env_file.write_text(f"FOOTYSTATS_API_KEY={KEY}\n", encoding="utf-8")
    fs.DEFAULT_ENV_FILE = env_file  # the cli_env fixture restores the attribute
    assert fs.main(["--list-leagues", "--cache-dir", str(tmp_path / "raw")]) == 0
    assert KEY not in capsys.readouterr().out


def test_cli_list_leagues(cli_env: CountingTransport, tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    code = fs.main(["--list-leagues", "--key", KEY, "--cache-dir", str(tmp_path / "raw")])
    out = capsys.readouterr().out
    assert code == 0 and cli_env.calls == 1
    assert "7 leagues chosen" in out and "Italy Serie C Group A [Italy]" in out and "2026/27=17487" in out and "Italy Serie B" in out
    assert KEY not in out


def test_cli_rejects_unknown_competition_and_bad_seasons(cli_env: CountingTransport, tmp_path: Path) -> None:
    with pytest.raises(SystemExit) as stop:
        fs.main(["--competitions", "ZZ", "--key", KEY])
    assert stop.value.code == 2
    with pytest.raises(SystemExit):
        fs.main(["--seasons", "0", "--key", KEY])
    assert cli_env.calls == 0


def test_cli_harvest_writes_csvs_and_odds(cli_env: CountingTransport, tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    out, fixtures_out, odds_out = tmp_path / "p" / "matches.csv", tmp_path / "p" / "fixtures.csv", tmp_path / "o" / "footystats.csv"
    code = fs.main(
        [
            "--competitions", "A", "SB", "--seasons", "2", "--key", KEY, "--no-sync",
            "--out", str(out), "--fixtures-out", str(fixtures_out), "--odds-out", str(odds_out),
            "--cache-dir", str(tmp_path / "raw"), "--mapping", str(tmp_path / "map.json"),
        ]
    )  # fmt: skip
    captured = capsys.readouterr()
    assert code == 0, captured.err
    assert KEY not in captured.out and KEY not in captured.err
    assert "matches=52 fixtures=4 odds=16" in captured.out and "perimeter: season 2026/27 (242 current teams)" in captured.out
    assert "registry_sync=-" in captured.out and f"fixtures -> {fixtures_out} (4 fixtures, 0 postponed)" in captured.out
    header = fixtures_out.read_text(encoding="utf-8").splitlines()[0]
    assert header == "match_id,date,round,girone,home_team,away_team,kickoff,status"
    assert "  A: 2026/27: 13 matches, 2025/26: 13 matches" in captured.out and f"odds -> {odds_out}" in captured.out
    from src.ingestion.harvester import load_fixtures, load_processed  # noqa: PLC0415

    frame = load_processed(out)
    assert len(frame) == 52 and sorted(frame["girone"].unique().tolist()) == ["A", "SB"]
    fixtures = load_fixtures(fixtures_out)
    assert len(fixtures) == 4 and all(f.date == date(2027, 1, 15) and f.kickoff is not None and f.status == "scheduled" for f in fixtures)
    quotes = load_odds_quotes(odds_out.parent)  # the dashboard / main.py reader understands the file
    assert len(quotes) == 16 and {q.match_id for q in quotes} == {f.match_id for f in fixtures}
    assert all(q.bookmaker == "FootyStats" for q in quotes)
    header = odds_out.read_text(encoding="utf-8").splitlines()[0]
    assert header == "match_id,bookmaker,market,selection,odds"
    assert (tmp_path / "map.json").exists() and _scan_for_key(tmp_path, KEY) == []

    # --no-odds leaves the odds file alone; a failing history season is reported and keeps the CSV untouched
    fs.DEFAULT_ENV_FILE = tmp_path / "absent.env"
    code = fs.main(["--competitions", "A", "--seasons", "1", "--key", KEY, "--no-odds", "--no-sync", "--out", str(out), "--fixtures-out", str(fixtures_out), "--odds-out", str(tmp_path / "o" / "none.csv"), "--cache-dir", str(tmp_path / "raw"), "--mapping", str(tmp_path / "map.json")])
    assert code == 0 and not (tmp_path / "o" / "none.csv").exists() and len(load_processed(out)) == 13


def test_cli_no_odds_removes_the_stale_quotes_of_postponed_matches(
    cli_env: CountingTransport, monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Regression: with ``--no-odds`` the quotes of an earlier run kept pricing matches that were since postponed."""
    out, fixtures_out, odds_out = tmp_path / "p" / "matches.csv", tmp_path / "p" / "fixtures.csv", tmp_path / "o" / "footystats.csv"
    common = ["--competitions", "A", "--seasons", "1", "--key", KEY, "--no-sync", "--out", str(out), "--fixtures-out", str(fixtures_out), "--odds-out", str(odds_out), "--mapping", str(tmp_path / "map.json")]
    assert fs.main([*common, "--cache-dir", str(tmp_path / "raw1")]) == 0
    before = load_odds_quotes(odds_out.parent)
    assert {q.match_id for q in before} == {f"FS-{CURRENT_A * 1_000_000 + 900001}", f"FS-{CURRENT_A * 1_000_000 + 900002}"}
    with odds_out.open("a", encoding="utf-8") as handle:
        handle.write("USER-1,MyBook,1X2,1,2.0\nUSER-1,MyBook,1X2,X,3.0\nUSER-1,MyBook,1X2,2,4.0\n")  # not a fixture of the run: kept
    capsys.readouterr()
    # a day after their kick-off the two matches are still incomplete: postponed, and their old prices are stale
    monkeypatch.setattr(fs, "_utc_now", lambda: datetime(2027, 1, 16, 12, 0, tzinfo=UTC))
    assert fs.main([*common, "--cache-dir", str(tmp_path / "raw2"), "--no-odds"]) == 0
    captured = capsys.readouterr()
    assert "(2 fixtures, 2 postponed)" in captured.out and KEY not in captured.out
    assert f"odds: 17 stale quote rows removed from {odds_out}" in captured.out  # 11 selections of 900001, 6 of 900002
    assert {q.match_id for q in load_odds_quotes(odds_out.parent)} == {"USER-1"}


def test_cli_partial_harvest_exit_code_and_allow_partial(cli_env: CountingTransport, monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    failing = CountingTransport(make_router(failing_seasons={PAST_A}))
    real_client = fs.FootyStatsClient

    def with_failing(**kwargs: Any) -> fs.FootyStatsClient:
        kwargs["transport"] = failing
        kwargs.setdefault("sleeper", RecordingSleeper())
        return real_client(**kwargs)

    monkeypatch.setattr(fs, "FootyStatsClient", with_failing)
    out = tmp_path / "matches.csv"
    args = ["--competitions", "A", "--seasons", "2", "--key", KEY, "--no-sync", "--out", str(out), "--fixtures-out", str(tmp_path / "f.csv"), "--odds-out", str(tmp_path / "o.csv"), "--cache-dir", str(tmp_path / "raw"), "--mapping", str(tmp_path / "map.json")]
    assert fs.main(args) == 1
    captured = capsys.readouterr()
    assert "failure A:2025/26" in captured.err and "harvest incomplete: 1 failures" in captured.err and not out.exists()
    assert KEY not in captured.err
    assert fs.main([*args, "--allow-partial"]) == 1 and out.exists()


def test_cli_init_registry(cli_env: CountingTransport, tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    config_dir = tmp_path / "config"
    config_dir.mkdir()
    (config_dir / "teams_2025_26.json").write_text(
        json.dumps({"season": "2025/26", "gironi": {"A": {"Arsenal": ["Arsenal FC"], "Cardiff": ["Cardiff City"]}}}), encoding="utf-8"
    )
    code = fs.main(["--init-registry", "--competitions", "A", "--season", "2026_27", "--key", KEY, "--config-dir", str(config_dir), "--cache-dir", str(tmp_path / "raw"), "--mapping", str(tmp_path / "map.json")])
    out = capsys.readouterr().out
    assert code == 0 and f"created {config_dir / 'teams_2026_27.json'}" in out and "A: added 20 teams" in out
    saved = json.loads((config_dir / "teams_2026_27.json").read_text(encoding="utf-8"))
    assert saved["season"] == "2026/27" and len(saved["gironi"]["A"]) == 20
    assert saved["gironi"]["A"]["Arsenal"] == ["The Gunners"] and saved["gironi"]["A"]["Cardiff"] == ["Cardiff City", "Bluebirds"]
    # a missing dictionary is only tolerated for --init-registry
    assert fs.main(["--competitions", "A", "--season", "1999_00", "--key", KEY, "--config-dir", str(config_dir), "--cache-dir", str(tmp_path / "raw")]) == 1
    assert "footystats failed" in capsys.readouterr().err


def test_cli_harvest_failure_is_masked_and_exit_1(cli_env: CountingTransport, monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    down = CountingTransport(lambda request: json_response({"error": "down"}, status=503))
    real_client = fs.FootyStatsClient
    monkeypatch.setattr(fs, "FootyStatsClient", lambda **kw: real_client(**{**kw, "transport": down, "sleeper": RecordingSleeper()}))
    code = fs.main(["--competitions", "A", "--key", KEY, "--cache-dir", str(tmp_path / "raw"), "--mapping", str(tmp_path / "map.json"), "--out", str(tmp_path / "m.csv")])
    captured = capsys.readouterr()
    assert code == 1 and "footystats failed" in captured.err and "503" in captured.err and KEY not in captured.err


def test_cli_parser_defaults_and_endpoints_section(endpoints: FootyStatsEndpoints) -> None:
    args = fs._build_parser().parse_args([])
    assert args.competitions == list(fs.DEFAULT_COMPETITIONS) == [m.value for m in Girone] and len(args.competitions) == 13
    assert args.seasons == 3 and args.key is None and not args.no_odds and not args.no_auto_init and not args.no_sync and not args.sync_registry
    assert args.odds_out == fs.ODDS_DIR / "footystats.csv" and args.mapping == fs.DEFAULT_MAPPING_PATH
    assert fs.DEFAULT_MAPPING_PATH == fs.ROOT / "data" / "footystats_teams.json" and fs.LEGACY_MAPPING_PATH == fs.ROOT / "config" / "footystats_teams.json"
    assert endpoints.base_url == "https://api.football-data-api.com" and endpoints.key_env == "FOOTYSTATS_API_KEY"
    assert endpoints.url_league_list() == "https://api.football-data-api.com/league-list"
    assert endpoints.url_league_matches().endswith("/league-matches") and endpoints.url_league_teams().endswith("/league-teams")
    assert endpoints.max_requests_per_hour == 1800
    assert endpoints.league_names == {"A": "Italy Serie C Group A", "B": "Italy Serie C Group B", "C": "Italy Serie C Group C", "SB": "Italy Serie B"}
    assert endpoints.known_season_ids("sb") == SEASONS_SB and endpoints.known_season_ids("ZZ") == {}
    assert FootyStatsEndpoints().league_names == {} and fs.footystats_league_name("B", FootyStatsEndpoints()) == "Italy Serie C Group B"
    with pytest.raises(ValueError):
        FootyStatsEndpoints(max_requests_per_hour=0)


def test_load_endpoints_rejects_malformed_footystats_section(tmp_path: Path) -> None:
    base = (fs.CONFIG_DIR / "endpoints.toml").read_text(encoding="utf-8").split("[footystats]")[0]
    bad = tmp_path / "endpoints.toml"
    bad.write_text(base + '[footystats]\nbase_url = "https://x"\n[footystats.league_names]\nA = 3\n', encoding="utf-8")
    with pytest.raises(ValueError, match="league_names"):
        load_endpoints(bad)
    bad.write_text(base + '[footystats]\n[footystats.season_ids]\nA = 5\n', encoding="utf-8")
    with pytest.raises(ValueError, match="season_ids"):
        load_endpoints(bad)
    bad.write_text(base, encoding="utf-8")  # no section at all: defaults
    assert load_endpoints(bad).footystats == FootyStatsEndpoints()


def test_real_registry_mapper_keeps_one_name_across_divisions(tmp_path: Path) -> None:
    """With the shipped dictionaries a FootyStats listing of Serie B / Serie C spellings resolves to the perimeter names."""
    registry = load_registry("2026_27", with_history=True)
    mapper = fs.FootyStatsTeamMapper(registry, TeamMatcher(registry), mapping_path=tmp_path / "map.json")
    vicenza = fs.TeamInfo(1, "L.R. Vicenza", "Vicenza", "LR Vicenza", ())
    pescara = fs.TeamInfo(2, "Delfino Pescara 1936", "Pescara", "Pescara", ("Delfino Pescara",))
    resolved = {t.id: mapper.resolve(t, code, "2026/27") for t, code in ((vicenza, "SB"), (pescara, "B"))}
    assert resolved[1].canonical in registry.canonical_names and resolved[2].canonical in registry.canonical_names
    assert registry.girone_of(resolved[1].canonical) == Girone.parse("SB") and registry.girone_of(resolved[2].canonical) == Girone.B


# ============================================================================ calendar as of the download, in-play matches, stale quotes
_LIVE_SEASON = 17487  # Serie C girone A 2026/27 in the league list below
_LIVE_TEAMS = ((1, "Renate"), (2, "Trento"), (3, "Lecco"), (4, "Novara"))


class LiveWorld:
    """A one-competition FootyStats account whose ``/league-matches`` changes between two harvests."""

    def __init__(self, matches: list[dict[str, Any]]) -> None:
        self.matches = matches
        self.paths: list[str] = []

    @staticmethod
    def match(match_id: int, home: int, away: int, kickoff: datetime, status: str = "incomplete", *, goals: tuple[int, int] = (-1, -1), odds: bool = False) -> dict[str, Any]:
        row: dict[str, Any] = {
            "id": match_id, "homeID": home, "awayID": away, "home_name": "", "away_name": "", "status": status,
            "date_unix": int(kickoff.timestamp()), "game_week": 13, "homeGoalCount": goals[0], "awayGoalCount": goals[1],
        }  # fmt: skip
        if odds:
            row.update(odds_ft_1=2.1, odds_ft_x=3.2, odds_ft_2=3.4)
        return row

    def router(self, request: httpx.Request) -> httpx.Response:
        assert request.url.params.get("key") == KEY
        self.paths.append(request.url.path)
        envelope = {"success": True, "pager": {"current_page": 1, "max_page": 1}}
        if request.url.path == "/league-list":
            return json_response({**envelope, "data": [{"name": "Italy Serie C Group A", "country": "Italy", "season": [{"id": _LIVE_SEASON, "year": 20262027}]}]})
        if request.url.path == "/league-teams":
            return json_response({**envelope, "data": [{"id": i, "name": n, "cleanName": n, "english_name": n} for i, n in _LIVE_TEAMS]})
        if request.url.path == "/league-matches":
            return json_response({**envelope, "data": [dict(m) for m in self.matches]})
        return json_response({"success": False, "message": "not found"}, status=404)


def _live_provider(world: LiveWorld, http: HttpConfig, tmp_path: Path, *, now: datetime, cache: RawCache | None, live_minutes: float = 10.0) -> fs.FootyStatsProvider:
    endpoints = dataclasses.replace(load_endpoints().footystats, live_matches_cache_minutes=live_minutes)
    client, _ = make_client(http, router=world.router, cache=cache, endpoints=endpoints)
    registry = TeamRegistry.from_teams("2026/27", {"A": [n for _, n in _LIVE_TEAMS]})
    mapper = fs.FootyStatsTeamMapper(registry, TeamMatcher(registry), mapping_path=tmp_path / "map.json")
    return fs.FootyStatsProvider(client, mapper, endpoints, now=lambda: now, config_dir=tmp_path / "no-config")


def _back_date(cache_dir: Path, when: datetime) -> int:
    count = 0
    for path in cache_dir.glob("*.json"):
        record = json.loads(path.read_text(encoding="utf-8"))
        record["fetched_at"] = when.isoformat(timespec="seconds")
        path.write_text(json.dumps(record), encoding="utf-8")
        count += 1
    return count


@pytest.mark.asyncio
async def test_a_cached_calendar_is_judged_as_of_its_download_never_as_postponed(http_cfg: HttpConfig, tmp_path: Path) -> None:
    """Regression: a five-hour-old cached payload made a match that kicked off 3h30 ago look 'postponed'."""
    real_now = datetime.now(UTC).replace(microsecond=0)
    kickoff = real_now - timedelta(hours=3, minutes=30)  # kicked off after the download below, played by now
    world = LiveWorld([LiveWorld.match(100, 1, 2, kickoff), LiveWorld.match(101, 3, 4, real_now + timedelta(days=1))])
    cache = RawCache(tmp_path / "raw", ttl=timedelta(hours=6))
    first = _live_provider(world, http_cfg, tmp_path, now=real_now - timedelta(hours=5), cache=cache)
    async with first.client:
        earlier = await first.harvest(["A"], seasons=1, with_odds=False)
    assert [(f.match_id, f.status) for f in earlier.fixtures] == [("FS-100", "scheduled"), ("FS-101", "scheduled")]
    assert _back_date(tmp_path / "raw", real_now - timedelta(hours=5)) >= 3  # the download really happened five hours ago
    world.matches[0].update(status="complete", homeGoalCount=2, awayGoalCount=1)  # meanwhile FootyStats has the result

    # the cache may serve the old payload (a cap longer than its age): its matches are judged as of the download
    cached = _live_provider(world, http_cfg, tmp_path, now=real_now, cache=cache, live_minutes=600)
    world.paths.clear()
    async with cached.client:
        stale = await cached.harvest(["A"], seasons=1, with_odds=False)
    assert world.paths == [] and stale.stats["cache_hits"] >= 3, "served from the raw cache"
    assert [(f.match_id, f.status) for f in stale.fixtures] == [("FS-100", "scheduled"), ("FS-101", "scheduled")]
    assert stale.n_postponed == 0, "a match still to play in an old payload is never 'rinviata'"

    # the default cap (10 minutes) re-downloads the season being played: the result is there
    fresh = _live_provider(world, http_cfg, tmp_path, now=real_now, cache=cache)
    async with fresh.client:
        result = await fresh.harvest(["A"], seasons=1, with_odds=False)
    assert "/league-matches" in world.paths
    assert result.matches["match_id"].tolist() == ["FS-100"] and (int(result.matches["home_goals"].iloc[0]), int(result.matches["away_goals"].iloc[0])) == (2, 1)
    assert [f.match_id for f in result.fixtures] == ["FS-101"] and result.n_postponed == 0


def test_matches_snapshot_reference_time_is_never_after_the_download() -> None:
    now = datetime(2026, 11, 7, 15, 30, tzinfo=UTC)
    assert fs.MatchesSnapshot([], None).reference_time(now) == now
    assert fs.MatchesSnapshot([], now - timedelta(hours=5)).reference_time(now) == now - timedelta(hours=5)
    assert fs.MatchesSnapshot([], now + timedelta(hours=1)).reference_time(now) == now  # an injected clock earlier than the file
    assert fs.MatchesSnapshot([], datetime(2026, 11, 7, 10, 0)).reference_time(now.replace(tzinfo=None)) == datetime(2026, 11, 7, 10, 0, tzinfo=UTC)
    assert load_endpoints().footystats.live_matches_cache_ttl == timedelta(minutes=10)
    with pytest.raises(ValueError, match="live_matches_cache_minutes"):
        FootyStatsEndpoints(live_matches_cache_minutes=-1)


@pytest.mark.asyncio
async def test_matches_in_play_stay_in_the_round_without_odds(http_cfg: HttpConfig, tmp_path: Path) -> None:
    """Regression: a match that kicked off less than 3 hours ago was dropped (neither played nor a fixture)."""
    now = datetime(2026, 11, 7, 14, 30, tzinfo=UTC)  # 15:30 in Rome
    world = LiveWorld([
        LiveWorld.match(200, 1, 2, now - timedelta(hours=1), odds=True),  # in play
        LiveWorld.match(201, 3, 4, now + timedelta(hours=2), odds=True),  # later today
        LiveWorld.match(202, 2, 3, now - timedelta(days=4), odds=True),  # postponed
    ])  # fmt: skip
    provider = _live_provider(world, http_cfg, tmp_path, now=now, cache=None)
    async with provider.client:
        result = await provider.harvest(["A"], seasons=1, with_odds=True)
    by_id = {f.match_id: f for f in result.fixtures}
    assert set(by_id) == {"FS-200", "FS-201", "FS-202"}, "the match in play is still a fixture of its round"
    assert by_id["FS-200"].status == "scheduled" and by_id["FS-200"].kickoff == now - timedelta(hours=1)
    assert by_id["FS-202"].status == "postponed" and result.n_postponed == 1
    assert {q.match_id for q in result.odds} == {"FS-201"}, "no pre-match price for a match in play or postponed"


def test_prune_stale_odds_drops_quotes_of_postponed_and_kicked_off_matches(tmp_path: Path) -> None:
    now = datetime(2026, 11, 7, 14, 30, tzinfo=UTC)
    fixtures = [
        Fixture("FS-1", date(2026, 11, 8), 13, "A", "Lecco", "Trento", kickoff=now + timedelta(days=1)),
        Fixture("FS-2", date(2026, 11, 7), 13, "A", "Renate", "Novara", kickoff=now - timedelta(minutes=30)),
        Fixture("FS-3", date(2026, 9, 30), 1, "A", "Trento", "Renate", kickoff=datetime(2026, 9, 30, 13, 0, tzinfo=UTC), status="postponed"),
        Fixture("FS-4", date(2026, 11, 8), 13, "A", "Novara", "Lecco"),  # no kick-off time: judged by nothing, kept
    ]
    assert fs.stale_quote_ids(fixtures, now=now) == {"FS-2", "FS-3"}
    path = tmp_path / "footystats.csv"
    pd.DataFrame(
        [{"match_id": mid, "bookmaker": "FootyStats", "market": "1X2", "selection": sel, "odds": 2.5} for mid in ("FS-1", "FS-2", "FS-3", "FS-4", "FS-9") for sel in ("1", "X", "2")]
    ).to_csv(path, index=False)
    assert fs.prune_stale_odds(path, fixtures, now=now) == 6
    kept = pd.read_csv(path, dtype="string")
    assert sorted(set(kept["match_id"])) == ["FS-1", "FS-4", "FS-9"] and len(kept) == 9 and list(kept.columns) == ["match_id", "bookmaker", "market", "selection", "odds"]
    assert fs.prune_stale_odds(path, fixtures, now=now) == 0 and fs.prune_stale_odds(tmp_path / "absent.csv", fixtures, now=now) == 0
    # the quotes the dashboard reads no longer price the postponed match
    assert {q.match_id for q in load_odds_quotes(tmp_path)} == {"FS-1", "FS-4", "FS-9"}
