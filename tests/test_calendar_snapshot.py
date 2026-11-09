"""Calendar fidelity end to end: the snapshot of the FootyStats calendar, the round view of the product, the CI
workflow that prints both, and the rules that keep FootyStats data out of the (public) repository.

The calendars are synthetic but shaped like the ones read live on 2026-11-07 (Saturday): Serie C girone A round 13
with two matches played on Friday and eight still to play over the weekend; Serie D girone G round 10 played on
31/10-01/11 except one recovery on Wednesday 11/11 while round 11 is this weekend; Serie D girone A with a round 1
match postponed on 30/09 and never played. Offline: the API runs on ``httpx.MockTransport``.
"""
from __future__ import annotations

import hashlib
import json
import random
import shutil
import subprocess
import sys
from collections.abc import Callable
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import httpx
import pytest

from src.common.competitions import competition
from src.common.paths import CONFIG_DIR, DATA_DIR, ROOT
from src.common.schemas import Fixture, Girone, MatchRecord, matches_to_frame
from src.ingestion import calendar_snapshot as cs
from src.ingestion import footystats as fs
from src.ingestion.harvester import HttpConfig, save_fixtures, save_processed

ROME = ZoneInfo("Europe/Rome")
UTC = timezone.utc
NOW = datetime(2026, 11, 7, 12, 0, tzinfo=UTC)  # Saturday 13:00 in Rome
KEY = "calendar-snapshot-test-key-0123456789"
HOST = "api.football-data-api.com"
WORKFLOW = ROOT / ".github" / "workflows" / "verifica-calendari.yml"


def _unix(day: int, hour: int, minute: int = 0, month: int = 11) -> int:
    return int(datetime(2026, month, day, hour, minute, tzinfo=ROME).timestamp())


def _teams(prefix: str, n: int, first_id: int) -> list[dict[str, Any]]:
    return [{"id": first_id + i, "name": f"{prefix} {i + 1:02d}", "cleanName": f"{prefix} {i + 1:02d}", "english_name": ""} for i in range(n)]


def _row(match_id: int, rnd: int, home: dict[str, Any], away: dict[str, Any], kickoff: int, *, score: tuple[int, int] | None = None) -> dict[str, Any]:
    complete = score is not None
    return {
        "id": match_id,
        "game_week": rnd,
        "revised_game_week": -1,
        "status": "complete" if complete else "incomplete",
        "date_unix": kickoff,
        "homeID": home["id"],
        "awayID": away["id"],
        "home_name": home["cleanName"],
        "away_name": away["cleanName"],
        "homeGoalCount": score[0] if complete else 0,
        "awayGoalCount": score[1] if complete else 0,
    }


def serie_c_a_check() -> cs.CompetitionCheck:
    """Round 12 fully played (31/10-01/11); round 13: two played on Friday 06/11, eight on 07-08/11."""
    teams = _teams("Club A", 20, 1000)
    rows = [_row(12_000 + k, 12, teams[2 * k], teams[2 * k + 1], _unix(1, 15) if k else _unix(31, 17, month=10), score=(1, 1)) for k in range(10)]
    for k in range(10):
        home, away = teams[2 * k + 1], teams[(2 * k + 2) % 20]
        if k < 2:
            rows.append(_row(13_000 + k, 13, home, away, _unix(6, 20, 30), score=(2, 1) if k == 0 else (0, 0)))
        else:
            rows.append(_row(13_000 + k, 13, home, away, _unix(7, 15) if k < 6 else _unix(8, 17, 30)))
    return cs.CompetitionCheck("A", "Serie C Girone A", "Italy Serie C Group A", 17487, teams=teams, matches=rows)


def serie_d_g_check() -> cs.CompetitionCheck:
    """Round 10: eight played on 31/10-01/11, Trastevere-Anzio recovered on Wednesday 11/11; round 11 this weekend."""
    teams = _teams("Club G", 18, 2000)
    teams[0].update(name="Trastevere", cleanName="Trastevere")
    teams[1].update(name="Anzio", cleanName="Anzio")
    rows = []
    for k in range(9):
        home, away = teams[2 * k], teams[2 * k + 1]
        rows.append(_row(10_000 + k, 10, home, away, _unix(11, 15) if k == 0 else _unix(1, 14, 30), score=None if k == 0 else (1, 0)))
    for k in range(9):
        rows.append(_row(11_000 + k, 11, teams[2 * k + 1], teams[(2 * k + 2) % 18], _unix(8, 14, 30)))
    return cs.CompetitionCheck("DG", "Serie D Girone G", "Italy Serie D Group G", 17672, teams=teams, matches=rows)


def serie_d_a_check() -> cs.CompetitionCheck:
    """Four teams: round 1 Chisola-Borgosesia postponed on 30/09 (still incomplete), the rest played; round 2 tomorrow."""
    teams = _teams("Club DA", 4, 3000)
    teams[0].update(name="Chisola", cleanName="Chisola")
    teams[1].update(name="Borgosesia", cleanName="Borgosesia")
    rows = [
        _row(1, 1, teams[0], teams[1], _unix(30, 15, month=9)),
        _row(2, 1, teams[2], teams[3], _unix(30, 15, month=9), score=(3, 0)),
        _row(3, 2, teams[1], teams[2], _unix(8, 14, 30)),
        _row(4, 2, teams[3], teams[0], _unix(8, 14, 30)),
    ]
    return cs.CompetitionCheck("DA", "Serie D Girone A", "Italy Serie D Group A", 17664, teams=teams, matches=rows)


def _repository_state() -> dict[str, str]:
    paths = [*sorted(CONFIG_DIR.glob("*.json")), *sorted(p for p in DATA_DIR.rglob("*") if p.is_file())]
    return {str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest() for path in paths}


@pytest.fixture(autouse=True)
def _repository_untouched() -> Any:
    before = _repository_state()
    yield
    assert _repository_state() == before, "a test wrote config/*.json or data/ of the repository"


# ============================================================================ the snapshot checks
def test_current_round_keeps_the_friday_matches_with_their_score_and_the_weekend_ones_with_their_time() -> None:
    check = serie_c_a_check()
    cs.check_calendar(check, now=NOW)
    assert check.next_round == 13 and len(check.next_round_lines) == 10 and check.issues == [] and check.postponed == []
    assert check.next_round_lines[0] == "ven 06/11  Club A 02 - Club A 03  giocata 2-1"
    assert check.next_round_lines[1] == "ven 06/11  Club A 04 - Club A 05  giocata 0-0"
    assert check.next_round_lines[2] == "sab 07/11 15:00  Club A 06 - Club A 07  da giocare"
    assert check.next_round_lines[-1].startswith("dom 08/11 17:30") and check.next_round_lines[-1].endswith("da giocare")
    assert check.status_counts == {"complete": 12, "incomplete": 8} and check.recovery_lines == []


def test_a_midweek_recovery_never_becomes_the_current_round() -> None:
    check = serie_d_g_check()
    cs.check_calendar(check, now=NOW)
    assert check.next_round == 11 and len(check.next_round_lines) == 9
    assert all(line.startswith("dom 08/11 14:30") for line in check.next_round_lines)
    assert check.recovery_lines == ["g.10 mer 11/11 15:00  Trastevere - Anzio  da giocare, in origine dom 01/11"]
    assert check.postponed == [] and check.issues == []


def test_postponed_matches_are_listed_and_round_anomalies_are_reported() -> None:
    check = serie_d_a_check()
    cs.check_calendar(check, now=NOW)
    assert check.postponed == ["g.1 mer 30/09 15:00 Chisola - Borgosesia (data passata, non giocata)"]
    assert check.next_round == 2 and check.next_round_lines == ["dom 08/11 14:30  Borgosesia - Club DA 03  da giocare", "dom 08/11 14:30  Club DA 04 - Chisola  da giocare"]
    assert check.issues == []
    broken = serie_d_a_check()
    broken.matches.append(_row(5, 2, broken.teams[1], broken.teams[3], _unix(9, 20)))  # Borgosesia twice in round 2
    broken.matches.append({**_row(6, 3, broken.teams[0], broken.teams[2], 0), "homeID": 99})  # no date, unknown team
    cs.check_calendar(broken, now=NOW)
    assert "giornata 2: squadre presenti più volte: Borgosesia, Club DA 04" in broken.issues and "giornata 2: 3 partite invece di 2" in broken.issues
    assert "giornata 3: id squadra non presenti in league-teams: [99]" in broken.issues and "partita 6 senza data" in broken.issues


def test_snapshot_uses_the_same_rule_as_the_product() -> None:
    """The round of the snapshot is :func:`src.engine.calendar.current_round` on the product's own parse."""
    from src.engine.calendar import current_round  # noqa: PLC0415

    for check in (serie_c_a_check(), serie_d_g_check(), serie_d_a_check()):
        played, fixtures, report = cs._parsed_calendar(check, now=NOW)
        cs.check_calendar(check, now=NOW)
        view = current_round(check.code, played, fixtures, now=NOW)
        assert view is not None and view.round == check.next_round and cs.round_lines(view) == check.next_round_lines
    assert report.postponed == 1 and any(f.status == "postponed" for f in fixtures)


def test_compare_with_dictionary_is_a_dry_run_of_the_registry_sync(tmp_path: Path) -> None:
    config = tmp_path / "config"
    config.mkdir()
    shutil.copy(CONFIG_DIR / "teams_2026_27.json", config / "teams_2026_27.json")
    before = (config / "teams_2026_27.json").read_bytes()
    shipped = json.loads(before)["gironi"]
    di = cs.CompetitionCheck("DI", "Serie D Girone I", "Italy Serie D Group I", 17675)
    names = ["AC Palermo" if n == "AC Palermo Serie D" else n for n in shipped["DI"]][:-1] + ["Nuova Squadra"]
    di.teams = [{"id": 500 + i, "name": n, "cleanName": n, "english_name": ""} for i, n in enumerate(names)]
    report = cs.compare_with_dictionary([di], season="2026_27", registry_path=config / "teams_2026_27.json", mapping_path=tmp_path / "map.json")
    assert report is not None and not report.written and (config / "teams_2026_27.json").read_bytes() == before and not (tmp_path / "map.json").exists()
    assert di.registry_missing == ["Nuova Squadra"] and di.registry_extra == [list(shipped["DI"])[-1]]
    assert "AC Palermo" not in di.registry_missing, "the Serie D club is matched by the sync rules, not mistaken for Serie B's Palermo"
    assert cs.compare_with_dictionary([di], season="2026_27", registry_path=tmp_path / "absent.json") is None
    text = cs.summary_markdown([di], season="2026_27", now=NOW, sync=report)
    assert "## Dizionario squadre (prova a secco della sincronizzazione)" in text and "aggiunte Nuova Squadra" in text
    assert "**Squadre FootyStats che la sincronizzazione aggiungerebbe:** Nuova Squadra" in text


def test_summary_markdown_has_the_table_and_no_odds_or_statistics() -> None:
    checks = [serie_c_a_check(), serie_d_g_check(), serie_d_a_check()]
    for check in checks:
        cs.check_calendar(check, now=NOW)
    error = cs.CompetitionCheck("DB", "Serie D Girone B", "Italy Serie D Group B", None, chosen=False, error="lega non scelta nell'account FootyStats (Choose leagues)")
    text = cs.summary_markdown([*checks, error], season="2026_27", now=NOW)
    assert text.startswith("# Verifica calendari FootyStats · stagione 2026/27 · 07/11/2026 13:00")
    assert "| Serie C Girone A | Italy Serie C Group A | ? | 20 | 20 | 12 | 8 | 0 | 13 | 0 | +0/-0 |" in text
    assert "| Serie D Girone A | Italy Serie D Group A | ? | 4 | 4 | 1 | 2 | 1 | 2 | 0 | +0/-0 |" in text
    assert "| Serie D Girone B | Italy Serie D Group B | NO |" in text and "ERRORE: lega non scelta" in text
    assert "**Giornata corrente (g.13) come la mostra la dashboard, orari Europe/Rome:**" in text
    assert "**Recuperi fuori dalla giornata corrente:**" in text and "**Partite rinviate (data passata, non giocate):**" in text
    for forbidden in ("odds", "quota", "xg", "shots", "corners"):
        assert forbidden not in text.lower()
    assert set(cs.MATCH_FIELDS) == {"id", "game_week", "revised_game_week", "status", "date_unix", "homeID", "awayID", "home_name", "away_name", "homeGoalCount", "awayGoalCount"}
    assert cs.kickoff_rome(0) is None and cs.kickoff_rome("x") is None and cs.kickoff_rome(_unix(7, 15)).hour == 15
    assert cs.season_key("2026_27") == "2026/27" and cs.round_lines(None) == []


# ============================================================================ the product view (CI step 2)
def _product_csvs(tmp_path: Path) -> tuple[Path, Path]:
    records = [
        MatchRecord("FS-1", date(2026, 11, 6), 13, Girone.A, "Lecco", "Trento", 2, 1),
        MatchRecord("FS-2", date(2026, 11, 6), 13, Girone.A, "Novara", "Renate", 0, 0),
        MatchRecord("FS-0", date(2025, 11, 9), 13, Girone.A, "Lecco", "Novara", 1, 1),  # last season: ignored
        *(MatchRecord(f"FS-G{k}", date(2026, 11, 1), 10, Girone.DG, f"G{2 * k}", f"G{2 * k + 1}", 1, 0) for k in range(1, 9)),
    ]
    fixtures = [
        Fixture("FS-3", date(2026, 11, 7), 13, Girone.A, "Pro Vercelli", "Cittadella", kickoff=datetime(2026, 11, 7, 14, 0, tzinfo=UTC)),
        Fixture("FS-4", date(2026, 11, 8), 13, Girone.A, "Lumezzane", "Treviso", kickoff=datetime(2026, 11, 8, 16, 30, tzinfo=UTC)),
        Fixture("FS-G0", date(2026, 11, 11), 10, Girone.DG, "Trastevere", "Anzio", kickoff=datetime(2026, 11, 11, 14, 0, tzinfo=UTC)),
        *(Fixture(f"FS-H{k}", date(2026, 11, 8), 11, Girone.DG, f"G{2 * k + 1}", f"G{(2 * k + 2) % 18}", kickoff=datetime(2026, 11, 8, 13, 30, tzinfo=UTC)) for k in range(9)),
        Fixture("FS-P", date(2026, 9, 30), 1, Girone.DA, "Chisola", "Borgosesia", kickoff=datetime(2026, 9, 30, 13, 0, tzinfo=UTC), status="postponed"),
    ]
    matches, fixtures_path = tmp_path / "ci" / "matches.csv", tmp_path / "ci" / "fixtures.csv"
    save_processed(matches_to_frame(records), matches)
    save_fixtures(fixtures, fixtures_path)
    return matches, fixtures_path


def test_product_view_prints_the_round_the_dashboard_shows(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    matches, fixtures = _product_csvs(tmp_path)
    code = cs.main(["--product-view", "--matches", str(matches), "--fixtures", str(fixtures), "--now", "2026-11-07T13:00:00+01:00", "--summary", str(tmp_path / "ci" / "giornate.md"), "--competitions", "A", "DG", "DA", "DB"])
    out = capsys.readouterr().out
    assert code == 0 and (tmp_path / "ci" / "giornate.md").read_text(encoding="utf-8") + "\n" == out
    assert out.startswith("# Giornate come le mostra il prodotto · stagione 2026/27 · 07/11/2026 13:00 (Europe/Rome)")
    assert "## Serie C Girone A\n**Giornata 13**: 2 giocate, 2 da giocare, 0 rinviate" in out
    assert "- ven 06/11  Lecco - Trento  giocata 2-1" in out and "- sab 07/11 15:00  Pro Vercelli - Cittadella  da giocare" in out
    assert "- dom 08/11 17:30  Lumezzane - Treviso  da giocare" in out and "Lecco - Novara" not in out
    assert "## Serie D Girone G\n**Giornata 11**: 0 giocate, 9 da giocare, 0 rinviate" in out
    assert "- g.10 mer 11/11 15:00  Trastevere - Anzio  da giocare, in origine dom 01/11" in out
    assert "## Serie D Girone A\nNessuna giornata in programma" in out and "- g.1 mer 30/09 15:00  Chisola - Borgosesia  rinviata, in origine mer 30/09" in out
    assert "## Serie D Girone B\nNessun dato per questa competizione" in out
    for forbidden in ("odds", "quota", "xg"):
        assert forbidden not in out.lower()
    # the CSVs were never written by the product: a clear message, exit code 1
    assert cs.main(["--product-view", "--matches", str(tmp_path / "none.csv"), "--fixtures", str(fixtures)]) == 1
    assert "la raccolta FootyStats non li ha scritti" in capsys.readouterr().err
    with pytest.raises(SystemExit):
        cs.main(["--product-view", "--matches", str(matches)])


def _round_through_the_product(check: cs.CompetitionCheck, now: datetime, tmp_path: Path) -> Any:
    """``parse_match`` -> the product's CSVs -> :func:`current_round`, exactly as the dashboard reads them."""
    from src.engine.calendar import current_round  # noqa: PLC0415
    from src.ingestion.harvester import load_fixtures, load_processed  # noqa: PLC0415

    played, fixtures, report = cs._parsed_calendar(check, now=now)
    save_processed(played, tmp_path / "matches.csv")
    save_fixtures(fixtures, tmp_path / "fixtures.csv")
    view = current_round(check.code, load_processed(tmp_path / "matches.csv"), load_fixtures(tmp_path / "fixtures.csv"), now=now)
    return view, report


def test_matches_in_play_never_vanish_from_their_round(tmp_path: Path) -> None:
    """Regression: during the Saturday 15:00 matches (and the last Monday match) the round lost its matches in play."""
    check = serie_c_a_check()
    during = datetime(2026, 11, 7, 16, 30, tzinfo=ROME)  # the four 15:00 matches are being played
    view, report = _round_through_the_product(check, during, tmp_path)
    assert report.in_play == 4 and report.postponed == 0
    assert view is not None and view.round == 13 and view.n_matches == 10 and view.n_played == 2 and view.n_postponed == 0
    assert {m.match_id for m in view.matches} == {f"FS-{13_000 + k}" for k in range(10)}
    cs.check_calendar(check, now=during)
    assert check.next_round == 13 and len(check.next_round_lines) == 10 and check.postponed == []

    # Monday 21:30: everything of round 13 played but the 20:30 match, round 14 next weekend
    monday = serie_c_a_check()
    for row in monday.matches:
        if row["game_week"] == 13:
            row.update(status="complete", homeGoalCount=1, awayGoalCount=1)
    last = monday.matches[-1]
    last.update(status="incomplete", homeGoalCount=0, awayGoalCount=0, date_unix=_unix(9, 20, 30))
    teams = monday.teams
    monday.matches += [_row(14_000 + k, 14, teams[2 * k], teams[(2 * k + 3) % 20], _unix(15, 15)) for k in range(10)]
    view, report = _round_through_the_product(monday, datetime(2026, 11, 9, 21, 30, tzinfo=ROME), tmp_path)
    assert report.in_play == 1 and view is not None and view.round == 13, "round 13 stays current while its last match is played"
    assert last["id"] and f"FS-{last['id']}" in {m.match_id for m in view.matches} and view.n_played == 9 and view.n_postponed == 0
    # three hours later the match has not been reported played: it is postponed (rinviata) and round 14 is current
    late, report = _round_through_the_product(monday, datetime(2026, 11, 10, 0, 31, tzinfo=ROME), tmp_path)
    assert report.postponed == 1 and late is not None and late.round == 14


# ============================================================================ the snapshot against a mock API
def mock_api() -> Callable[[httpx.Request], httpx.Response]:
    serie_a = serie_c_a_check()

    def router(request: httpx.Request) -> httpx.Response:
        assert request.url.host == HOST and request.url.params.get("key") == KEY
        if request.url.path == "/league-list":
            entry = {"name": "Italy Serie C Group A", "country": "Italy", "season": [{"id": 17487, "year": 20262027}]}
            return httpx.Response(200, json={"success": True, "pager": {"current_page": 1, "max_page": 1}, "data": [entry]})
        if request.url.params.get("season_id") != "17487":
            return httpx.Response(200, json={"success": False, "message": "League is not chosen by the user."})
        rows = serie_a.teams if request.url.path == "/league-teams" else [{**m, "odds_ft_1": 2.1, "team_a_xg": 1.4} for m in serie_a.matches]
        return httpx.Response(200, json={"success": True, "pager": {"current_page": 1, "max_page": 1}, "data": rows})

    return router


@pytest.mark.asyncio
async def test_snapshot_downloads_only_calendar_facts_and_flags_unchosen_leagues() -> None:
    http = HttpConfig(timeout_seconds=5.0, max_retries=1, backoff_base_seconds=0.1, backoff_max_seconds=0.2, max_concurrency=2, min_delay_seconds=0.0)
    checks = await cs.snapshot([competition("A"), competition("DB")], season="2026_27", api_key=KEY, now=NOW, http=http, transport=httpx.MockTransport(mock_api()))
    a, db = checks
    assert a.chosen is True and a.error is None and a.next_round == 13 and len(a.matches) == 20 and len(a.teams) == 20
    assert all(set(m) == set(cs.MATCH_FIELDS) for m in a.matches), "odds and statistics are never copied into the snapshot"
    assert db.chosen is False and db.error == "lega non scelta nell'account FootyStats (Choose leagues)" and db.matches == []
    assert KEY not in json.dumps([c.to_dict() for c in checks])


def test_cli_snapshot_prints_the_summary_never_the_key(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    transport = httpx.MockTransport(mock_api())
    real = cs.FootyStatsClient
    monkeypatch.setattr(cs, "FootyStatsClient", lambda **kwargs: real(**{**kwargs, "transport": transport, "rng": random.Random(1)}))
    config = tmp_path / "config"
    config.mkdir()
    shutil.copy(CONFIG_DIR / "teams_2026_27.json", config / "teams_2026_27.json")
    args = ["--key", KEY, "--competitions", "A", "--now", "2026-11-07T13:00:00+01:00", "--config-dir", str(config), "--mapping", str(tmp_path / "map.json"), "--log-level", "ERROR"]
    assert cs.main([*args, "--out", str(tmp_path / "snap.json"), "--summary", str(tmp_path / "riepilogo.md")]) == 0
    out = capsys.readouterr()
    assert "| Serie C Girone A | Italy Serie C Group A | sì | 20 | 20 | 12 | 8 | 0 | 13 |" in out.out
    assert "+20/-20" in out.out, "the synthetic Club A 01..20 would replace the 20 real clubs: the dry run says so"
    assert KEY not in out.out + out.err + (tmp_path / "snap.json").read_text(encoding="utf-8") + (tmp_path / "riepilogo.md").read_text(encoding="utf-8")
    assert not (tmp_path / "map.json").exists() and (config / "teams_2026_27.json").read_bytes() == (CONFIG_DIR / "teams_2026_27.json").read_bytes()
    # a competition that cannot be checked makes the exit code 1; without --out/--summary nothing is written
    assert cs.main([*args[:2], "--competitions", "A", "DB", *args[4:]]) == 1
    assert "ERRORE: lega non scelta" in capsys.readouterr().out and sorted(p.name for p in tmp_path.iterdir()) == ["config", "riepilogo.md", "snap.json"]


def test_cli_snapshot_without_a_key_explains_how_to_set_it(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    if fs.DEFAULT_ENV_FILE.exists():
        pytest.skip("a developer's own .env provides a key")
    monkeypatch.delenv("FOOTYSTATS_API_KEY", raising=False)
    assert cs.main(["--competitions", "A"]) == 2
    assert "FOOTYSTATS_API_KEY" in capsys.readouterr().err


# ============================================================================ CI workflow and data policy
def test_this_module_is_collected_without_pyyaml() -> None:
    # PyYAML is not in requirements.txt: a clean install must still collect (and run) this file
    code = "import sys; sys.modules['yaml'] = None; import tests.test_calendar_snapshot as m; print(m.WORKFLOW.name)"
    done = subprocess.run([sys.executable, "-c", code], cwd=ROOT, capture_output=True, text=True, timeout=120)
    assert done.returncode == 0, done.stderr
    assert done.stdout.strip() == "verifica-calendari.yml"


def test_verifica_calendari_text_never_echoes_the_key_nor_writes_to_the_repository() -> None:
    # plain-text checks, no YAML parser needed (they run on a clean install too)
    text = WORKFLOW.read_text(encoding="utf-8")
    for path in ("src/ingestion/calendar_snapshot.py", "src/ingestion/footystats.py", "src/engine/calendar.py"):
        assert f"- {path}" in text, path
    # the product path now keeps the pre-match odds of the played matches (inside ci/, deleted at the end) for the
    # aggregated model-vs-market metrics of the reliability step, which never prints a match or a price
    for option in ("--odds-history-out ci/history/odds_history.csv", "--out ci/matches.csv", "--fixtures-out ci/fixtures.csv", "--odds-out ci/odds.csv", "--cache-dir ci/raw", "--mapping ci/footystats_teams.json"):
        assert option in text, option
    assert "--reliability --matches ci/matches.csv" in text and "--odds-history ci/history/odds_history.csv" in text
    for path in ("src/engine/backtest.py", "src/engine/calibration.py", "src/engine/slips.py", "src/engine/fitgroups.py", "src/dashboard/state.py"):
        assert f"- {path}" in text, path
    assert "rm -rf ci" in text and "upload-artifact" not in text
    for line in text.splitlines():
        if "FOOTYSTATS_API_KEY" in line and "echo" in line:
            assert "${FOOTYSTATS_API_KEY}" not in line and "$FOOTYSTATS_API_KEY" not in line and "secrets." not in line, line
    assert text.count("secrets.FOOTYSTATS_API_KEY") == 1
    for forbidden in ("git commit", "git push", "git add", "set -x"):
        assert forbidden not in text


def test_verifica_calendari_runs_the_snapshot_and_the_product_path_without_writing_or_echoing_the_key() -> None:
    yaml = pytest.importorskip("yaml")  # optional: PyYAML is not a dependency of the project
    text = WORKFLOW.read_text(encoding="utf-8")
    workflow = yaml.safe_load(text)
    triggers = workflow.get("on", workflow.get(True))  # PyYAML reads the bare key `on` as the boolean True
    paths = triggers["push"]["paths"]
    for required in ("src/ingestion/calendar_snapshot.py", "src/ingestion/footystats.py", "src/engine/calendar.py", ".github/workflows/verifica-calendari.yml"):
        assert required in paths
    assert "workflow_dispatch" in triggers and workflow["permissions"] == {"contents": "read"}
    job = workflow["jobs"]["verifica"]
    assert job["env"]["FOOTYSTATS_API_KEY"] == "${{ secrets.FOOTYSTATS_API_KEY }}"
    steps = {step["name"]: step for step in job["steps"]}
    assert not any("upload-artifact" in str(step.get("uses", "")) for step in job["steps"]), "no artifact: nothing of FootyStats leaves the run"
    snapshot = steps["Istantanea dei calendari"]["run"]
    assert "python -m src.ingestion.calendar_snapshot" in snapshot and "--summary ci/riepilogo.md" in snapshot
    product = steps["Percorso del prodotto (raccolta FootyStats + sincronizzazione del dizionario)"]["run"]
    assert "python -m src.ingestion.footystats --seasons 3" in product and "--competitions" not in product
    for option in ("--out ci/matches.csv", "--fixtures-out ci/fixtures.csv", "--odds-out ci/odds.csv", "--odds-history-out ci/history/odds_history.csv", "--cache-dir ci/raw", "--mapping ci/footystats_teams.json"):
        assert option in product, option
    assert "--product-view --matches ci/matches.csv --fixtures ci/fixtures.csv" in product and "data/" not in product
    reliability = steps["Affidabilità del modello (backtest walk-forward, solo metriche aggregate)"]
    assert reliability.get("if") == "always()" and "--reliability --matches ci/matches.csv" in reliability["run"] and "data/" not in reliability["run"]
    assert "> /dev/null" in reliability["run"], "only the aggregated Markdown summary reaches the log"
    for required in ("src/engine/backtest.py", "src/engine/calibration.py", "src/engine/slips.py", "src/engine/fitgroups.py", "src/dashboard/state.py"):
        assert required in paths, required
    assert "rm -rf ci" in steps["Pulizia (nessun dato FootyStats resta sul runner)"]["run"]
    # the key is only ever tested for emptiness, never printed; nothing is committed or pushed
    for line in text.splitlines():
        if "FOOTYSTATS_API_KEY" in line and "echo" in line:
            assert "${FOOTYSTATS_API_KEY}" not in line and "$FOOTYSTATS_API_KEY" not in line and "secrets." not in line, line
    assert text.count("secrets.FOOTYSTATS_API_KEY") == 1
    for forbidden in ("git commit", "git push", "git add", "set -x"):
        assert forbidden not in text


def test_no_footystats_data_can_reach_the_repository() -> None:
    for gone in (".github/workflows/aggiorna-dati.yml", "pubblica_dati.bat", "pubblica_dati.command"):
        assert not (ROOT / gone).exists(), gone
    gitignore = (ROOT / ".gitignore").read_text(encoding="utf-8")
    assert "!data/processed/*.csv" not in gitignore and "!data/odds/*.csv" not in gitignore
    if shutil.which("git") is None or not (ROOT / ".git").exists():
        pytest.skip("git is not available")

    def ignored(path: str) -> bool:
        return subprocess.run(["git", "check-ignore", "--no-index", "-q", path], cwd=ROOT).returncode == 0

    for path in ("data/processed/matches.csv", "data/processed/fixtures.csv", "data/odds/footystats.csv", "data/raw/0a1b2c.json", "data/footystats_teams.json", "data/last_refresh.json", ".env", "ci/matches.csv"):
        assert ignored(path), f"{path} must be git-ignored"
    for path in ("data/processed/.gitkeep", "data/odds/.gitkeep", "data/raw/.gitkeep", ".env.example", "config/teams_2026_27.json", "config/competitions.toml"):
        assert not ignored(path), f"{path} must stay tracked"
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    assert "Dati FootyStats: solo in locale, mai nel repository" in readme and "pubblico" in readme and "licenza" in readme
    assert "aggiorna-dati.yml" in readme and "Secrets" in readme and 'FOOTYSTATS_API_KEY = "' in readme
    assert "I CSV in `data/processed/` e `data/odds/` sono versionati" not in readme and "pubblica_dati.bat` / `pubblica_dati.command` (serve git" not in readme


def test_snapshot_dates_are_shown_in_rome_across_the_dst_change() -> None:
    summer = cs.kickoff_rome(int(datetime(2026, 10, 24, 13, 0, tzinfo=UTC).timestamp()))
    winter = cs.kickoff_rome(int(datetime(2026, 10, 31, 14, 0, tzinfo=UTC).timestamp()))
    assert summer is not None and winter is not None and (summer.hour, winter.hour) == (15, 15)
    assert NOW.astimezone(ROME).utcoffset() == timedelta(hours=1)


def test_public_summary_hides_odds_aggregates_below_the_minimum_sample():
    from src.ingestion.calendar_snapshot import PUBLIC_MIN_QUOTED, _public

    assert _public("1.234", PUBLIC_MIN_QUOTED) == "1.234"
    assert _public("1.234", PUBLIC_MIN_QUOTED - 1) == f"n<{PUBLIC_MIN_QUOTED}"
    assert _public("12.5%", 1) == f"n<{PUBLIC_MIN_QUOTED}"
    assert _public("12.5%", None) == f"n<{PUBLIC_MIN_QUOTED}"
