"""Tests for ``src.dashboard.data_refresh``: key lookup, data age, and the runtime FootyStats refresh with a fake provider (no network)."""
from __future__ import annotations

import json
import logging
import os
import time
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import pandas as pd
import pytest

from src.common.schemas import Fixture, OddsQuote
from src.common.teams import TeamRegistry, load_registry
from src.dashboard import data_refresh
from src.dashboard.data_refresh import (
    DEFAULT_MAX_AGE,
    KEY_ENV,
    ODDS_FILE,
    STAMP_FILE,
    KeyLocation,
    RefreshResult,
    data_age,
    find_key,
    last_refresh,
    locate_key,
    needs_refresh,
    refresh,
    refresh_label,
    supported_competitions,
)
from src.ingestion.footystats import DEFAULT_COMPETITIONS
from src.ingestion.harvester import load_fixtures, load_processed
from src.simulation.generator import simulate_league

#: A fake key: it must never appear in a log record, a file, a result or an error message.
SECRET = "fs-test-key-0123456789abcdef"


def bc_registry() -> TeamRegistry:
    """The Serie B and Serie C part of the shipped 2026/27 dictionary (80 teams)."""
    full = load_registry()
    return TeamRegistry.from_teams(full.season, {g.value: full.teams_by_girone[g] for g in full.gironi if g.family in ("Serie C", "Serie B")})


@pytest.fixture(scope="module")
def league():  # noqa: ANN201 - SimulatedLeague
    return simulate_league(seed=3, registry=bc_registry(), n_rounds=3)


@dataclass
class FakeReport:
    """A registry-sync report as the provider returns it (only ``describe`` is read)."""

    text: str

    def describe(self) -> str:
        return self.text


@dataclass
class FakeResult:
    """The attributes of :class:`~src.ingestion.footystats.FootyStatsHarvest` that ``refresh`` reads."""

    matches: pd.DataFrame
    fixtures: list[Fixture]
    odds: list[OddsQuote]
    failures: dict[str, str] = field(default_factory=dict)
    skipped: dict[str, str] = field(default_factory=dict)
    not_chosen: dict[str, str] = field(default_factory=dict)
    registry_sync: Any = None
    registry_sync_error: str | None = None

    def problem_for(self, code: str) -> str | None:
        return self.skipped.get(code)


class FakeClient:
    def __init__(self) -> None:
        self.stats = {"requests": 0}


class FakeProvider:
    """Serves the simulated league for the requested codes; ``not_chosen`` codes deliver nothing, ``fail`` raises."""

    def __init__(self, league: Any, *, not_chosen: dict[str, str] | None = None, fail: Exception | None = None, delay: float = 0.0) -> None:
        self.league = league
        self.not_chosen = not_chosen or {}
        self.fail = fail
        self.delay = delay
        self.client = FakeClient()
        self.calls: list[tuple[list[str], dict[str, Any]]] = []

    async def harvest(self, codes, *, seasons: int = 3, with_odds: bool = True, sync_registry: bool = True):  # noqa: ANN001, ANN201
        import asyncio

        self.calls.append((list(codes), {"seasons": seasons, "with_odds": with_odds, "sync_registry": sync_registry}))
        if self.delay:
            for _ in range(3):
                if self.client is not None:
                    self.client.stats["requests"] += 5
                await asyncio.sleep(self.delay / 3)
        if self.fail is not None:
            raise self.fail
        wanted = {c for c in codes if c not in self.not_chosen}
        matches = self.league.matches[self.league.matches["girone"].isin(wanted)].reset_index(drop=True)
        fixtures = [f for f in self.league.fixtures if f.girone.value in wanted]
        ids = {f.match_id for f in fixtures}
        odds = [q for q in self.league.odds if q.match_id in ids]
        skipped = {c: f"league {name!r} is not among the leagues chosen" for c, name in self.not_chosen.items() if c in codes}
        return FakeResult(
            matches,
            fixtures,
            odds,
            skipped=skipped,
            not_chosen={c: n for c, n in self.not_chosen.items() if c in codes},
            registry_sync=FakeReport("Dizionario squadre 2026/27 sincronizzato con FootyStats: 1 variazioni.\nGirone B: aggiunta Nuova Monterosi"),
        )


def factory_for(provider: FakeProvider, seen: dict[str, Any]):  # noqa: ANN201 - async context manager factory
    @asynccontextmanager
    async def factory(key: str, *, season: str, config_dir: Path):  # noqa: ANN202
        seen["key"] = key
        seen["season"] = season
        yield provider

    return factory


# ----------------------------------------------------------------------------- key
def test_find_key_reads_environment_then_env_file_then_secrets(tmp_path: Path) -> None:
    env_file = tmp_path / ".env"
    assert find_key(environ={}, env_file=env_file, use_streamlit_secrets=False) is None
    assert find_key(environ={}, env_file=None, secrets={}, use_streamlit_secrets=False) is None
    env_file.write_text(f"# chiave\n{KEY_ENV}=\"from-file\"  # comment\n", encoding="utf-8")
    assert find_key(environ={}, env_file=env_file, use_streamlit_secrets=False) == "from-file"
    assert locate_key(environ={}, env_file=env_file, use_streamlit_secrets=False).source == ".env"
    assert find_key(environ={KEY_ENV: " from-env "}, env_file=env_file) == "from-env"
    assert locate_key(environ={KEY_ENV: "x"}, env_file=env_file).source == "ambiente"
    assert find_key(environ={}, env_file=None, secrets={KEY_ENV: "from-secrets"}) == "from-secrets"
    assert locate_key(environ={}, env_file=None, secrets={KEY_ENV: "s"}).source == "Streamlit secrets"
    assert find_key(environ={KEY_ENV: "  "}, env_file=None, secrets={KEY_ENV: ""}) is None
    # without a secrets file Streamlit raises: the lookup says "no key" instead
    assert data_refresh._streamlit_secret("NOPE_NOT_A_SECRET") is None
    # the key is never part of a repr
    location = KeyLocation(SECRET, "ambiente")
    assert SECRET not in repr(location) and "ambiente" in repr(location)


# ----------------------------------------------------------------------------- age
def test_data_age_and_needs_refresh(tmp_path: Path) -> None:
    processed = tmp_path / "processed"
    now = datetime(2026, 11, 7, 12, 0, tzinfo=UTC)
    assert data_age(processed, now=now) is None and needs_refresh(processed, now=now) and refresh_label(processed) is None
    processed.mkdir()
    (processed / "matches.csv").write_text("match_id\n", encoding="utf-8")
    # without a stamp the modification time of matches.csv counts
    os.utime(processed / "matches.csv", (now.timestamp() - 3600, now.timestamp() - 3600))
    assert data_age(processed, now=now) == timedelta(hours=1) and not needs_refresh(processed, now=now)
    assert needs_refresh(processed, now=now + DEFAULT_MAX_AGE) and DEFAULT_MAX_AGE == timedelta(hours=6)
    # the stamp written by refresh() records the file time it wrote: while matches.csv is that file, the stamp's time counts
    written_mtime = (processed / "matches.csv").stat().st_mtime
    stamp = {"refreshed_at": (now - timedelta(hours=7)).isoformat(), "matches_mtime": written_mtime}
    (processed / STAMP_FILE).write_text(json.dumps(stamp), encoding="utf-8")
    assert data_age(processed, now=now) == timedelta(hours=7) and needs_refresh(processed, now=now)
    assert not needs_refresh(processed, now=now, max_age=timedelta(hours=8))
    assert refresh_label(processed) == "Dati FootyStats aggiornati alle 06:00 del 07/11"  # 05:00 UTC = 06:00 CET
    assert last_refresh(processed)["refreshed_at"].startswith("2026-11-07T05:00")
    # a clock behind the file reads as age zero
    assert data_age(processed, now=now - timedelta(hours=10)) == timedelta(0)
    # a broken stamp falls back to the file time
    (processed / STAMP_FILE).write_text("{not json", encoding="utf-8")
    assert last_refresh(processed) is None and data_age(processed, now=now) == timedelta(hours=1)
    (processed / STAMP_FILE).write_text(json.dumps(stamp), encoding="utf-8")
    # regression: the launchers' `python -m src.ingestion.footystats` rewrites matches.csv without a stamp -> the newer
    # file time counts, so the dashboard neither shows the old time nor downloads everything again right after the launcher
    os.utime(processed / "matches.csv", (now.timestamp() - 1800, now.timestamp() - 1800))
    assert data_age(processed, now=now) == timedelta(minutes=30) and not needs_refresh(processed, now=now)
    assert refresh_label(processed) == "Dati FootyStats aggiornati alle 12:30 del 07/11"
    # a stamp of an earlier release (no matches_mtime): the newer of the stamp and the file
    (processed / STAMP_FILE).write_text(json.dumps({"refreshed_at": (now - timedelta(days=2)).isoformat()}), encoding="utf-8")
    assert data_age(processed, now=now) == timedelta(minutes=30) and not needs_refresh(processed, now=now)
    (processed / STAMP_FILE).write_text(json.dumps({"refreshed_at": (now - timedelta(minutes=10)).isoformat()}), encoding="utf-8")
    assert data_age(processed, now=now) == timedelta(minutes=10)
    assert supported_competitions() == tuple(DEFAULT_COMPETITIONS)


# ----------------------------------------------------------------------------- refresh
def test_refresh_saves_locally_reports_counts_and_never_leaks_the_key(tmp_path: Path, league, caplog: pytest.LogCaptureFixture, monkeypatch: pytest.MonkeyPatch) -> None:
    processed, odds_dir = tmp_path / "processed", tmp_path / "odds"
    monkeypatch.setattr(data_refresh, "PROGRESS_INTERVAL", 0.005)
    provider = FakeProvider(league, not_chosen={"DA": "Italy Serie D Group A"}, delay=0.05)
    seen: dict[str, Any] = {}
    steps: list[tuple[float, str]] = []
    clock = iter([datetime(2026, 11, 7, 13, 0, tzinfo=UTC), datetime(2026, 11, 7, 13, 2, tzinfo=UTC), datetime(2026, 11, 7, 13, 2, 30, tzinfo=UTC)])
    caplog.set_level(logging.DEBUG)
    result = refresh(
        processed,
        odds_dir,
        competitions=["A", "B", "C", "SB", "DA"],
        key=SECRET,
        provider_factory=factory_for(provider, seen),
        progress=lambda fraction, message: steps.append((fraction, message)),
        now=lambda: next(clock),
    )
    assert isinstance(result, RefreshResult) and result.written and result.partial and not result.ok and result.error is None
    assert seen["key"] == SECRET, "the key reaches the provider only"
    # one harvest for every competition, with the dictionary sync and the odds
    assert provider.calls == [(["A", "B", "C", "SB", "DA"], {"seasons": 3, "with_odds": True, "sync_registry": True})]
    assert result.competitions == ("A", "B", "C", "SB", "DA") and set(result.per_competition) == {"A", "B", "C", "SB"}
    assert result.per_competition["SB"] == {"partite": 30, "in_calendario": 10, "quote": len([q for q in league.odds if q.match_id.startswith("SIM-SB")])}
    assert set(result.errors) == {"DA"} and result.not_chosen == {"DA": "Italy Serie D Group A"}
    warnings = result.warnings()
    assert len(warnings) == 1 and "Serie D Girone A" in warnings[0] and "Italy Serie D Group A" in warnings[0] and "Choose leagues" in warnings[0]
    assert result.registry_sync == ("Dizionario squadre 2026/27 sincronizzato con FootyStats: 1 variazioni.", "Girone B: aggiunta Nuova Monterosi")
    assert (result.n_matches, result.n_fixtures, result.n_odds) == (len(league.matches), len(league.fixtures), len(league.odds))
    assert result.timestamp_text == "14:02 del 07/11" and "14:02 del 07/11" in result.summary_text() and "1 competizioni non scaricate" in result.summary_text()
    # the files are written locally and read back with the harvester's readers
    assert load_processed(processed / "matches.csv").shape[0] == len(league.matches)
    assert {f.match_id for f in load_fixtures(processed / "fixtures.csv")} == {f.match_id for f in league.fixtures}
    odds = pd.read_csv(odds_dir / ODDS_FILE)
    assert len(odds) == sum(len(q.odds) for q in league.odds) and set(odds.columns) == {"match_id", "bookmaker", "market", "selection", "odds"}
    stamp = last_refresh(processed)
    assert stamp is not None and stamp["n_matches"] == len(league.matches) and stamp["errors"] == {"DA": result.errors["DA"]}
    assert stamp["registry_sync"] == list(result.registry_sync) and stamp["refreshed_at"].startswith("2026-11-07T13:02")
    assert not needs_refresh(processed, now=datetime(2026, 11, 7, 14, 0, tzinfo=UTC))
    assert not list(processed.glob(".*.tmp")) and not list(odds_dir.glob(".*.tmp")), "temporary files are replaced atomically"
    # progress: within [0, 1], monotone, ticking while the provider worked, ending at 1.0
    fractions = [f for f, _ in steps]
    assert fractions[0] < 0.1 and fractions[-1] == 1.0 and all(0.0 <= f <= 1.0 for f in fractions) and fractions == sorted(fractions)
    assert any("richieste" in message for _, message in steps) and steps[-1][1] == "Dati aggiornati."
    # the key appears nowhere: logs, files, result
    assert SECRET not in caplog.text
    for path in [*processed.iterdir(), *odds_dir.iterdir()]:
        assert SECRET not in path.read_text(encoding="utf-8"), path
    assert SECRET not in repr(result) and SECRET not in json.dumps(result.to_stamp())


def test_refresh_keeps_the_previous_data_of_a_competition_that_failed(tmp_path: Path, league) -> None:
    processed, odds_dir = tmp_path / "processed", tmp_path / "odds"
    first = refresh(processed, odds_dir, competitions=["A", "B", "C", "SB"], key=SECRET, provider_factory=factory_for(FakeProvider(league), {}))
    assert first.ok and first.n_matches == len(league.matches)
    # the next run: Serie B is refused (league no longer chosen) -> the other competitions are rewritten, Serie B is kept
    second = refresh(processed, odds_dir, competitions=["A", "B", "C", "SB"], key=SECRET, provider_factory=factory_for(FakeProvider(league, not_chosen={"SB": "Italy Serie B"}), {}))
    assert second.written and set(second.errors) == {"SB"} and set(second.per_competition) == {"A", "B", "C"}
    matches = load_processed(processed / "matches.csv")
    assert len(matches) == len(league.matches) and (matches["girone"] == "SB").sum() == 30
    fixtures = load_fixtures(processed / "fixtures.csv")
    assert len(fixtures) == len(league.fixtures) and sum(1 for f in fixtures if f.girone.value == "SB") == 10
    odds = pd.read_csv(odds_dir / ODDS_FILE)
    assert len(odds) == sum(len(q.odds) for q in league.odds) and second.n_odds == len(league.odds)


def test_refresh_without_key_or_with_a_failing_provider_writes_nothing_and_never_raises(tmp_path: Path, league, caplog: pytest.LogCaptureFixture, monkeypatch: pytest.MonkeyPatch) -> None:
    processed, odds_dir = tmp_path / "processed", tmp_path / "odds"
    monkeypatch.delenv(KEY_ENV, raising=False)
    monkeypatch.setattr(data_refresh, "find_key", lambda **_: None)
    no_key = refresh(processed, odds_dir, provider_factory=factory_for(FakeProvider(league), {}))
    assert not no_key.written and KEY_ENV in no_key.error and not processed.exists()
    assert no_key.warnings() == [no_key.error] and no_key.summary_text().startswith("Aggiornamento FootyStats non riuscito")
    caplog.set_level(logging.DEBUG)
    boom = refresh(processed, odds_dir, key=SECRET, provider_factory=factory_for(FakeProvider(league, fail=RuntimeError(f"HTTP 500 at /league-matches?key={SECRET}")), {}))
    assert not boom.written and "RuntimeError" in boom.error and SECRET not in boom.error and SECRET not in caplog.text
    assert not (processed / "matches.csv").exists()
    # nothing delivered at all (every league refused): the local data is left untouched
    processed.mkdir(parents=True, exist_ok=True)
    (processed / "matches.csv").write_text("sentinel\n", encoding="utf-8")
    refused = refresh(processed, odds_dir, competitions=["A", "SB"], key=SECRET, provider_factory=factory_for(FakeProvider(league, not_chosen={"A": "Italy Serie C Group A", "SB": "Italy Serie B"}), {}))
    assert not refused.written and set(refused.errors) == {"A", "SB"} and "nessuna competizione scaricata" in refused.error
    assert (processed / "matches.csv").read_text(encoding="utf-8") == "sentinel\n"
    assert len(refused.warnings()) == 3


def test_refresh_progress_without_request_counts_uses_elapsed_time(tmp_path: Path, league, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(data_refresh, "PROGRESS_INTERVAL", 0.05)
    provider = FakeProvider(league, delay=0.3)
    provider.client = None  # a provider that does not expose its client
    steps: list[float] = []
    started = time.perf_counter()
    result = refresh(tmp_path / "p", tmp_path / "o", competitions=["A"], key=SECRET, provider_factory=factory_for(provider, {}), progress=lambda f, _m: steps.append(f))
    assert result.ok and time.perf_counter() - started >= 0.25
    assert steps[0] == pytest.approx(0.02) and steps[-1] == 1.0 and all(0.0 <= f <= 1.0 for f in steps) and len(steps) >= 3


# ----------------------------------------------------------------------------- review fixes: one season failing, old data warnings
class ScriptedProvider:
    """Returns a prepared harvest result (``FakeResult``-like), whatever the codes."""

    def __init__(self, result: Any) -> None:
        self.result = result
        self.client = FakeClient()

    async def harvest(self, codes, *, seasons: int = 3, with_odds: bool = True, sync_registry: bool = True):  # noqa: ANN001, ANN201
        return self.result


def test_refresh_keeps_the_previous_current_season_of_a_competition_whose_current_season_failed(tmp_path: Path, league, caplog: pytest.LogCaptureFixture) -> None:
    """Regression: a transient error (HTTP 503) on the current season of Girone A, with its history served from the cache,
    erased A's current-season matches, its fixtures and its odds, while the sidebar said the data was fresh."""
    from src.common.schemas import validate_matches_df
    from src.dashboard.data_refresh import kept_data_warnings
    from src.simulation.generator import simulate_history

    history = simulate_history(seed=3, registry=bc_registry(), seasons=("2025/26",)).matches
    full = validate_matches_df(pd.concat([history, league.matches], ignore_index=True))
    processed, odds_dir = tmp_path / "processed", tmp_path / "odds"
    first = refresh(processed, odds_dir, competitions=["A", "B", "C", "SB"], key=SECRET, provider_factory=factory_for(FakeProvider(league), {}))
    assert first.ok
    # the first run wrote the current season only; add the history so the merge has both seasons to tell apart
    from src.ingestion.harvester import save_processed

    save_processed(full, processed / "matches.csv")
    a_current = full[(full["girone"] == "A") & (full["date"] >= pd.Timestamp(2026, 7, 1))]
    a_fixtures = [f for f in league.fixtures if f.girone.value == "A"]
    a_ids = {f.match_id for f in a_fixtures}
    # the second run: everything delivered but Girone A 2026/27 (history of A still served)
    delivered = full.drop(a_current.index).reset_index(drop=True)
    result = FakeResult(
        delivered,
        [f for f in league.fixtures if f.match_id not in a_ids],
        [q for q in league.odds if q.match_id not in a_ids],
        failures={"A:2026/27": f"HTTP 503 at /league-matches?key={SECRET}"},
    )
    result.seasons = {code: ["2026/27", "2025/26"] for code in ("A", "B", "C", "SB")}
    caplog.set_level(logging.DEBUG)
    second = refresh(processed, odds_dir, competitions=["A", "B", "C", "SB"], key=SECRET, provider_factory=factory_for(ScriptedProvider(result), {}))
    assert second.written and second.partial and not second.ok and not second.errors and set(second.season_failures) == {"A:2026/27"}
    matches = load_processed(processed / "matches.csv")
    assert len(matches) == len(full) and set(a_current["match_id"]) <= set(matches["match_id"]), "A's current season is kept"
    assert matches["match_id"].is_unique
    fixtures = load_fixtures(processed / "fixtures.csv")
    assert {f.match_id for f in fixtures} == {f.match_id for f in league.fixtures}, "A's fixtures are kept"
    odds = pd.read_csv(odds_dir / ODDS_FILE)
    assert set(odds["match_id"]) == {q.match_id for q in league.odds} and second.n_odds == len(league.odds)
    assert second.kept == {"A": {"stagioni": ["2026/27"], "partite": len(a_current), "in_calendario": len(a_fixtures), "motivo": second.season_failures["A:2026/27"]}}
    lines = second.warnings()
    assert len(lines) == 1 and lines[0].startswith("Serie C Girone A: stagione 2026/27 non scaricata (HTTP 503") and "Tenuti i dati del download precedente" in lines[0]
    kept = kept_data_warnings(processed)
    assert len(kept) == 1 and kept[0].startswith("Serie C Girone A, stagione 2026/27: non aggiornata all'ultimo download (HTTP 503")
    assert f"{len(a_current)} partite, {len(a_fixtures)} in calendario" in kept[0]
    assert SECRET not in caplog.text and SECRET not in " ".join(lines) and SECRET not in (processed / STAMP_FILE).read_text(encoding="utf-8")
    # a history season failing keeps its previous rows only: the current season and the fixtures come from the new run
    a_history = full[(full["girone"] == "B") & (full["date"] < pd.Timestamp(2026, 7, 1))]
    third_result = FakeResult(full.drop(a_history.index).reset_index(drop=True), list(league.fixtures), list(league.odds), failures={"B:2025/26": "timeout"})
    third_result.seasons = {code: ["2026/27", "2025/26"] for code in ("A", "B", "C", "SB")}
    third = refresh(processed, odds_dir, competitions=["A", "B", "C", "SB"], key=SECRET, provider_factory=factory_for(ScriptedProvider(third_result), {}))
    assert third.kept == {"B": {"stagioni": ["2025/26"], "partite": len(a_history), "in_calendario": 0, "motivo": "timeout"}}
    assert len(load_processed(processed / "matches.csv")) == len(full) and len(load_fixtures(processed / "fixtures.csv")) == len(league.fixtures)
    # without the provider's season list the registry season counts as the current one
    plain = FakeResult(delivered, [f for f in league.fixtures if f.match_id not in a_ids], [], failures={"A:2026/27": "HTTP 503"})
    fourth = refresh(processed, odds_dir, competitions=["A", "B", "C", "SB"], key=SECRET, provider_factory=factory_for(ScriptedProvider(plain), {}))
    assert "A" in fourth.kept and {f.match_id for f in load_fixtures(processed / "fixtures.csv")} >= a_ids


def test_stale_and_kept_data_warnings_age_text(tmp_path: Path) -> None:
    """The warnings the dashboard repeats on every run while the data is old (finding: old data hardly flagged)."""
    from src.dashboard.data_refresh import age_text, kept_data_warnings, stale_warning

    assert (age_text(timedelta(minutes=1)), age_text(timedelta(minutes=45)), age_text(timedelta(hours=1, minutes=5))) == ("1 minuto", "45 minuti", "1 ora")
    assert (age_text(timedelta(hours=30)), age_text(timedelta(days=2, hours=1)), age_text(timedelta(seconds=-5))) == ("30 ore", "2 giorni", "0 minuti")
    processed = tmp_path / "processed"
    now = datetime(2026, 11, 7, 12, 0, tzinfo=UTC)
    assert stale_warning(processed, key_available=False, now=now) is None and kept_data_warnings(processed) == [], "no data: the setup page says it"
    processed.mkdir()
    (processed / "matches.csv").write_text("match_id\n", encoding="utf-8")
    two_days = now - timedelta(days=2, minutes=14)
    os.utime(processed / "matches.csv", (two_days.timestamp(), two_days.timestamp()))
    text = stale_warning(processed, key_available=False, now=now)
    assert text is not None and text.startswith("Dati vecchi di 2 giorni (ultimo aggiornamento FootyStats alle 12:46 del 05/11/2026): chiave FootyStats mancante")
    assert KEY_ENV in text and "in corso / risultato in attesa" in text
    failed = stale_warning(processed, key_available=True, last_error="aggiornamento non riuscito: HTTP 429", now=now)
    assert failed is not None and "aggiornamento non riuscito: HTTP 429" in failed and "chiave FootyStats mancante" not in failed
    assert "aggiornamento non ancora eseguito" in stale_warning(processed, key_available=True, now=now)
    assert stale_warning(processed, key_available=False, now=two_days + timedelta(hours=5)) is None, "younger than 6 hours: fresh"
    (processed / STAMP_FILE).write_text(json.dumps({"refreshed_at": now.isoformat(), "kept": {"SB": {"stagioni": [], "partite": 30, "in_calendario": 10, "motivo": "lega non scelta"}}}), encoding="utf-8")
    assert kept_data_warnings(processed) == [
        "Serie B: non aggiornata all'ultimo download (lega non scelta): restano i dati del download precedente (30 partite, 10 in calendario)."
    ]


# ----------------------------------------------------------------------------- odds history
def _history_for(matches: pd.DataFrame) -> list[OddsQuote]:
    """A complete 1X2 and GG/NG book for every played match (what FootyStats keeps on completed matches)."""
    from src.common.schemas import Market

    quotes: list[OddsQuote] = []
    for match_id in matches["match_id"].astype(str):
        quotes.append(OddsQuote(match_id, "FootyStats", Market.MATCH_ODDS, {"1": 2.2, "X": 3.25, "2": 3.4}))
        quotes.append(OddsQuote(match_id, "FootyStats", Market.BTTS, {"GG": 1.9, "NG": 1.9}))
    return quotes


def test_refresh_writes_the_odds_history_apart_from_the_current_prices(tmp_path: Path, league) -> None:
    from src.dashboard.data_refresh import odds_history_file
    from src.dashboard.state import load_odds_quotes
    from src.ingestion.harvester import load_odds_history

    processed, odds_dir = tmp_path / "processed", tmp_path / "odds"
    result = FakeResult(league.matches, list(league.fixtures), list(league.odds))
    result.odds_history = _history_for(league.matches)  # type: ignore[attr-defined]
    first = refresh(processed, odds_dir, competitions=["A", "B", "C", "SB"], key=SECRET, provider_factory=factory_for(ScriptedProvider(result), {}))
    path = odds_history_file(odds_dir)
    assert path == odds_dir / "history" / "odds_history.csv" and path.is_file()
    assert first.n_odds_history == 2 * len(league.matches) and last_refresh(processed)["n_odds_history"] == first.n_odds_history
    history = load_odds_history(path)
    assert len(history) == 5 * len(league.matches) and set(history["match_id"]) == set(league.matches["match_id"].astype(str))
    # the current prices are unchanged: the history never reaches the value engine
    assert {q.match_id for q in load_odds_quotes(odds_dir)} == {q.match_id for q in league.odds}
    assert not list(path.parent.glob(".*.tmp")) and SECRET not in path.read_text(encoding="utf-8")
    # next run: Serie B is refused -> its history rows are kept, the others rewritten
    second_result = FakeResult(
        league.matches[league.matches["girone"] != "SB"].reset_index(drop=True),
        [f for f in league.fixtures if f.girone.value != "SB"],
        [q for q in league.odds if not q.match_id.startswith("SIM-SB")],
        skipped={"SB": "league 'Italy Serie B' is not among the leagues chosen"},
        not_chosen={"SB": "Italy Serie B"},
    )
    second_result.odds_history = _history_for(second_result.matches)  # type: ignore[attr-defined]
    second = refresh(processed, odds_dir, competitions=["A", "B", "C", "SB"], key=SECRET, provider_factory=factory_for(ScriptedProvider(second_result), {}))
    assert second.written and set(second.errors) == {"SB"} and second.n_odds_history == 2 * len(second_result.matches)
    assert set(load_odds_history(path)["match_id"]) == set(league.matches["match_id"].astype(str))


def test_refresh_without_odds_history_leaves_no_history_file(tmp_path: Path, league) -> None:
    from src.dashboard.data_refresh import odds_history_file

    processed, odds_dir = tmp_path / "processed", tmp_path / "odds"
    result = refresh(processed, odds_dir, competitions=["A", "B", "C", "SB"], key=SECRET, provider_factory=factory_for(FakeProvider(league), {}))
    assert result.ok and result.n_odds_history == 0 and not odds_history_file(odds_dir).exists() and not (odds_dir / "history").exists()
    assert last_refresh(processed)["n_odds_history"] == 0
