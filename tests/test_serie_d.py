"""Serie D (nine gironi ``DA`` .. ``DI``, family ``'Serie D'``): configuration, registries that ship empty,
simulator and engine behaviour, FootyStats auto-init of the team lists, automation and launchers.

The FootyStats side runs offline on ``httpx.MockTransport``: the router serves a league list with the Serie D
league, a synthetic 18-team ``/league-teams`` listing and ``/league-matches`` rows with the thin statistics
FootyStats publishes for Serie D (shots and xG at ``0``, no odds fields at all).
"""
from __future__ import annotations

import hashlib
import json
import logging
import random
import re
import shutil
from collections.abc import Callable
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any

import httpx
import numpy as np
import pandas as pd
import pytest
import yaml

from src.common import competitions as competitions_module
from src.common.competitions import (
    COMPETITIONS,
    Competition,
    competition,
    competitions_by_family,
    competitions_of_family,
    describe_competitions,
    families,
    family_label,
    find_competition,
)
from src.common.paths import CONFIG_DIR, ROOT
from src.common.schemas import Girone, validate_matches_df
from src.common.teams import PerimeterLabel, TeamRegistry, load_registry, normalize_name
from src.engine.model import DixonColesConfig, DixonColesEngine, build_team_index
from src.ingestion import footystats as fs
from src.ingestion.harvester import Harvester, HttpConfig, load_endpoints
from src.ingestion.team_matcher import TeamMatcher
from src.simulation.generator import (
    DEFAULT_HOME_ADVANTAGE,
    SERIE_D_TEAM_PREFIX,
    default_home_advantage,
    sample_true_params,
    simulate_history,
    simulate_league,
    simulate_serie_d_registry,
)

SERIE_D_CODES = tuple(f"D{letter}" for letter in "ABCDEFGHI")
ALL_CODES = ("A", "B", "C", "SB", *SERIE_D_CODES)
SEASON_IDS = {
    "DA": {"2024/25": 13475, "2025/26": 15710, "2026/27": 17664},
    "DB": {"2024/25": 13232, "2025/26": 15706, "2026/27": 17667},
    "DC": {"2024/25": 13479, "2025/26": 15718, "2026/27": 17666},
    "DD": {"2024/25": 13481, "2025/26": 15714, "2026/27": 17676},
    "DE": {"2024/25": 13473, "2025/26": 15685, "2026/27": 17669},
    "DF": {"2024/25": 13470, "2025/26": 15707, "2026/27": 17671},
    "DG": {"2024/25": 13469, "2025/26": 15712, "2026/27": 17672},
    "DH": {"2024/25": 13477, "2025/26": 15693, "2026/27": 17674},
    "DI": {"2024/25": 13472, "2025/26": 15705, "2026/27": 17675},
}
KEY = "serie-d-test-key-0123456789"
HOST = "api.football-data-api.com"
NOW = datetime(2026, 11, 7, 12, 0, tzinfo=timezone.utc)
#: Kick-offs of the synthetic Serie D matches: three played in September 2026, one fixture in January 2027.
PLAYED_UNIX = tuple(int(datetime(2026, 9, day, 13, 0, tzinfo=timezone.utc).timestamp()) for day in (14, 21, 28))
FIXTURE_UNIX = int(datetime(2027, 1, 15, 13, 30, tzinfo=timezone.utc).timestamp())


# ============================================================================ configuration
def test_enum_has_thirteen_members_in_configuration_order() -> None:
    assert [m.value for m in Girone] == list(ALL_CODES) == [c.code for c in COMPETITIONS]
    assert len(Girone) == 13 and Girone.DA.label == "Serie D Girone A" and Girone.DI.short == "D-I"
    assert all(Girone(code).family == "Serie D" and Girone(code).tier == 4 for code in SERIE_D_CODES)
    assert Girone.DA.footystats_name == "Italy Serie D Group A" and Girone.DA.country == "Italy"
    assert families() == ("Serie C", "Serie B", "Serie D") and family_label("serie_d") == "Serie D" == family_label("SERIE-D")
    with pytest.raises(ValueError, match="Unknown family"):
        family_label("Serie A")
    assert competitions_of_family("Serie D") == COMPETITIONS[4:] == competitions_by_family()["Serie D"]
    assert [c.girone_letter for c in COMPETITIONS[4:]] == list("ABCDEFGHI") and competition("SB").girone_letter == ""


@pytest.mark.parametrize(
    ("spelling", "expected"),
    [
        ("DA", "DA"),
        ("da", "DA"),
        ("D-A", "DA"),
        ("d_a", "DA"),
        ("Serie D Girone A", "DA"),
        ("serie_d_girone_a", "DA"),
        ("Italy Serie D Group A", "DA"),
        ("Italy Serie D Group I", "DI"),
        ("D-I", "DI"),
        ("Serie D Girone H", "DH"),
        # the Serie C spellings are untouched by the nine new members
        ("A", "A"),
        ("B", "B"),
        ("C", "C"),
        ("Girone A", "A"),
        ("Group A", "A"),
        ("Serie C Girone A", "A"),
        ("SB", "SB"),
        ("Serie B", "SB"),
    ],
)
def test_girone_parse_accepts_serie_d_spellings_and_keeps_serie_c_letters(spelling: str, expected: str) -> None:
    assert Girone.parse(spelling) is Girone(expected) and find_competition(spelling).code == expected


@pytest.mark.parametrize("spelling", ["D", "d", "Serie D", "Girone D", "Group D", "Gruppo D", "Serie D Girone", "Serie C Girone DA", "Serie D A"])
def test_d_alone_and_contradictory_serie_d_spellings_are_rejected(spelling: str) -> None:
    """``'D'`` must never resolve to a Serie D girone (there are nine) and the Serie C family must not wrap a Serie D code."""
    with pytest.raises(ValueError, match="Unknown girone"):
        Girone.parse(spelling)
    with pytest.raises(ValueError, match="Unknown girone"):
        find_competition(spelling)


def test_competition_records_carry_season_ids_and_provider_support() -> None:
    for code, ids in SEASON_IDS.items():
        comp = competition(code)
        assert comp.season_ids == ids and comp.sofascore_tournament_id == 0 and comp.tier == 4 and comp.family == "Serie D"
        assert not comp.supports_provider("sofascore") and comp.supports_provider("footystats") and comp.supports_provider("FotMob")
        assert comp.to_dict()["season_ids"] == ids and comp.to_dict()["season_ids"] is not comp.season_ids
    assert competition("A").season_ids == {"2024/25": 12659, "2025/26": 15535, "2026/27": 17487} and competition("A").supports_provider("sofascore")
    assert competition("SB").season_ids == {"2024/25": 12621, "2025/26": 15632, "2026/27": 17404}
    assert all(c.season_ids for c in COMPETITIONS), "every competition has fallback season ids"
    with pytest.raises(ValueError):
        competition("DA").supports_provider("")
    # the record stays hashable (frozen) although season_ids is a dict, and the season labels are normalised
    assert len({competition("DA"), competition("DB"), competition("DA")}) == 2
    assert Competition("ZZ", "L", "S", "Italy", "FS", 0, 4, "F", {"2026_27": 5}).season_ids == {"2026/27": 5}
    with pytest.raises(ValueError, match="positive integer"):
        Competition("ZZ", "L", "S", "Italy", "FS", 0, 4, "F", {"2026/27": 0})
    with pytest.raises(ValueError, match="season label"):
        Competition("ZZ", "L", "S", "Italy", "FS", 0, 4, "F", {"next": 5})
    # the TOML sub-table is optional and unknown keys are still rejected
    table = {"code": "ZZ", "label": "L", "short": "S", "country": "Italy", "footystats_name": "FS", "sofascore_tournament_id": 0, "tier": 4, "family": "F"}
    assert competitions_module.parse_competitions({"competition": [table]})[0].season_ids == {}
    assert competitions_module.parse_competitions({"competition": [{**table, "season_ids": {"2026/27": 7}}]})[0].season_ids == {"2026/27": 7}
    with pytest.raises(ValueError, match="unknown keys"):
        competitions_module.parse_competitions({"competition": [{**table, "seasons_ids": {}}]})


def test_describe_competitions_per_family() -> None:
    assert describe_competitions() == "Serie C Gironi A/B/C + Serie B + Serie D Gironi A-I"
    assert describe_competitions(family="Serie D") == "Serie D Gironi A-I"
    assert describe_competitions(family="Serie D", season="2026/27") == "Serie D 2026/27 Gironi A-I"
    assert describe_competitions(family="Serie C", season="2026/27") == "Serie C 2026/27 Gironi A/B/C"
    assert describe_competitions(family="serie b") == "Serie B"
    assert describe_competitions(COMPETITIONS[4:7]) == "Serie D Gironi A/B/C"  # fewer than four consecutive letters: no range
    assert describe_competitions([COMPETITIONS[4], COMPETITIONS[6]]) == "Serie D Gironi A/C"
    assert describe_competitions([COMPETITIONS[4]]) == "Serie D"
    assert describe_competitions([], family="Serie D") == "" and describe_competitions([], season="2026/27") == "2026/27"
    with pytest.raises(ValueError, match="Unknown family"):
        describe_competitions(family="Serie A")


# ============================================================================ registry
def test_registry_with_empty_serie_d_competitions() -> None:
    registry = load_registry()
    assert registry.n_teams == 80 and registry.gironi == (Girone.A, Girone.B, Girone.C, Girone.SB)
    assert registry.missing_competitions() == tuple(Girone(code) for code in SERIE_D_CODES)
    assert registry.families() == ("Serie C", "Serie B")
    assert all(registry.teams(code) == () for code in SERIE_D_CODES) and registry.teams("Serie D Girone I") == ()
    with pytest.raises(ValueError, match="Unknown girone"):
        registry.teams("D")
    label = registry.perimeter_label
    assert label == "Serie C 2026/27 Gironi A/B/C + Serie B (80 squadre)" and isinstance(label, PerimeterLabel) and isinstance(label, str)
    assert label("Serie D") == registry.perimeter_label_for("Serie D") == "Serie D 2026/27 (0 squadre)"
    assert label(family="Serie C") == "Serie C 2026/27 Gironi A/B/C (60 squadre)" and label("Serie B") == "Serie B 2026/27 (20 squadre)"
    assert label(None) == label and f"Perimetro: {label}" == "Perimetro: Serie C 2026/27 Gironi A/B/C + Serie B (80 squadre)"
    with pytest.raises(ValueError, match="Unknown family"):
        registry.perimeter_label_for("Serie A")
    # the shipped file lists the nine codes with empty objects, and loading with history keeps them empty
    payload = json.loads((CONFIG_DIR / "teams_2026_27.json").read_text(encoding="utf-8"))
    assert list(payload["gironi"]) == list(ALL_CODES) and all(payload["gironi"][code] == {} for code in SERIE_D_CODES)
    assert "auto-init" in payload["note"] or "harvester" in payload["note"]
    merged = load_registry(with_history=True)
    assert merged.n_teams == 80 and merged.missing_competitions() == registry.missing_competitions()
    # absent and empty mean the same thing
    absent = TeamRegistry.from_mapping("2026/27", {"A": {"x": []}, "SB": {"y": []}})
    empty = TeamRegistry.from_mapping("2026/27", {"A": {"x": []}, "SB": {"y": []}, "DA": {}, "DI": {}})
    assert absent.gironi == empty.gironi == (Girone.A, Girone.SB) and absent.missing_competitions() == empty.missing_competitions()
    assert empty.teams("DA") == () and Girone.DA not in empty.teams_by_girone
    assert empty.perimeter_label == absent.perimeter_label == "Serie C 2026/27 + Serie B (2 squadre)"
    with pytest.raises(ValueError, match="listed twice"):
        TeamRegistry.from_mapping("2026/27", {"DA": {}, "D-A": {}})
    # a populated Serie D registry describes its family only when asked for it
    populated = TeamRegistry.from_mapping("2026/27", {"A": {"x": []}, "DA": {f"Club {i:02d}": [] for i in range(1, 19)}, "DB": {}})
    assert populated.gironi == (Girone.A, Girone.DA) and populated.families() == ("Serie C", "Serie D")
    assert populated.perimeter_label == "Serie C 2026/27 + Serie D (19 squadre)" and populated.perimeter_label("Serie D") == "Serie D 2026/27 (18 squadre)"
    assert populated.missing_competitions() == (Girone.B, Girone.C, Girone.SB, *(Girone(code) for code in SERIE_D_CODES[1:]))
    assert populated.is_current_team("Club 01") and populated.girone_of("Club 01") is Girone.DA


# ============================================================================ simulator and engine
def test_simulate_league_and_history_skip_competitions_without_teams() -> None:
    assert DEFAULT_HOME_ADVANTAGE == {"A": 0.30, "B": 0.22, "C": 0.38, "SB": 0.25, **{code: 0.25 for code in SERIE_D_CODES}}
    assert default_home_advantage() == {g: DEFAULT_HOME_ADVANTAGE[g.value] for g in Girone} and default_home_advantage(["DA"]) == {Girone.DA: 0.25}
    league = simulate_league(seed=7, as_of=date(2026, 11, 6))
    assert sorted(set(league.matches["girone"])) == ["A", "B", "C", "SB"] and len(league.matches) == 440 and len(league.fixtures) == 40
    assert set(league.true_params.gamma) == set(Girone), "the true parameters cover every configured competition"
    history = simulate_history(seed=7, seasons=("2025/26",))
    assert sorted(set(history.matches["girone"])) == ["A", "B", "C", "SB"] and len(history.matches) == 1520
    assert history.season_registries["2025/26"].gironi == (Girone.A, Girone.B, Girone.C, Girone.SB)
    params = sample_true_params(load_registry(), np.random.default_rng(0))
    assert len(params.alpha) == 80 and set(params.gamma) == set(Girone)
    with pytest.raises(ValueError, match="no teams"):
        simulate_league(seed=1, registry=TeamRegistry.from_mapping("2026/27", {"DA": {}}))


def test_simulate_serie_d_registry_and_the_joint_fit_has_nine_gammas() -> None:
    registry = simulate_serie_d_registry(seed=3)
    assert registry.n_teams == 162 and registry.gironi == tuple(Girone(code) for code in SERIE_D_CODES)
    assert registry.missing_competitions() == (Girone.A, Girone.B, Girone.C, Girone.SB) and registry.families() == ("Serie D",)
    assert registry.perimeter_label == "Serie D 2026/27 Gironi A-I (162 squadre)" and registry.season_label == "2026/27"
    names = registry.teams("DA")
    assert len(names) == 18 and set(names) == {f"{SERIE_D_TEAM_PREFIX} DA {i:02d}" for i in range(1, 19)} and "Squadra DA 01" in names
    assert len({normalize_name(t) for t in registry.canonical_names}) == 162, "normalised names stay unique (dash-free codes, zero-padded numbers)"
    assert simulate_serie_d_registry(seed=3).teams("DA") == names and simulate_serie_d_registry(seed=4).teams("DA") != names
    assert set(simulate_serie_d_registry(seed=4).teams("DA")) == set(names), "the seed only shuffles the order"
    assert simulate_serie_d_registry(seed=1, n_teams=6, season="2025_26").n_teams == 54
    with pytest.raises(ValueError):
        simulate_serie_d_registry(n_teams=1)

    league = simulate_league(seed=5, registry=registry, n_rounds=14, with_odds=True)
    assert sorted(set(league.matches["girone"])) == list(SERIE_D_CODES) and len(league.matches) == 9 * 14 * 9
    assert len(league.fixtures) == 9 * 9 and {fx.round for fx in league.fixtures} == {15} and league.odds
    assert league.matches["round"].max() == 14 and league.registry is registry
    full = simulate_league(seed=5, registry=registry, with_odds=False)
    assert full.matches["round"].max() == 33 and {fx.round for fx in full.fixtures} == {34}, "18 teams: 34 rounds, the last one is the fixture list"

    params = DixonColesEngine(DixonColesConfig(xi=0.0)).fit(league.matches, registry=registry)
    assert params.converged and set(params.gamma) == set(registry.gironi) and len(params.gamma) == 9
    assert set(params.current_teams) == set(registry.canonical_names) and params.historical_teams == ()
    for g, members in params.teams_by_girone.items():
        assert len(members) == 18 and sum(params.alpha[t] for t in members) == pytest.approx(0.0, abs=1e-6), g
        assert np.isfinite(params.gamma[g]), g  # 126 matches per girone: a single gamma is still noisy (18 teams x 2 strengths)
    assert abs(np.mean(list(params.gamma.values())) - 0.25) < 0.2, "the nine gammas average to the true home advantage"
    assert np.corrcoef([params.alpha[t] for t in params.teams], [league.true_params.alpha[t] for t in params.teams])[0, 1] > 0.5


def test_model_handles_a_competition_with_matches_but_no_registry_teams(caplog: pytest.LogCaptureFixture) -> None:
    """The auto-init raced with the harvest: Serie D matches exist, the registry has no Serie D teams yet. The
    index must not crash: those teams are historical of their own competition (own gamma, own group)."""
    registry = load_registry()
    serie_d = simulate_serie_d_registry(seed=2)
    da_only = simulate_league(seed=9, registry=serie_d, n_rounds=8, with_odds=False).matches
    da_only = validate_matches_df(da_only.loc[da_only["girone"] == "DA"].reset_index(drop=True))
    with caplog.at_level(logging.WARNING, logger="src.engine.model"):
        index = build_team_index(da_only, registry=registry)
    assert index.gironi == (Girone.DA,) and index.n_gammas == 1 and len(index.historical) == 18 and index.current_teams == ()
    assert set(index.teams_by_girone) == {Girone.DA} and index.constraint_groups() == {Girone.DA: index.teams_by_girone[Girone.DA]}
    record = next(r for r in caplog.records if r.getMessage().startswith("competitions with matches but no current teams"))
    assert record.gironi == ["DA"] and record.teams == {"DA": 18}
    params = DixonColesEngine(DixonColesConfig(xi=0.0)).fit(da_only, registry=registry)
    assert params.converged and set(params.gamma) == {Girone.DA} and len(params.historical_teams) == 18 and params.current_teams == ()
    # competitions with neither teams nor matches are simply absent: no gamma, no group
    current = simulate_league(seed=7, n_rounds=6, with_odds=False).matches
    both = validate_matches_df(pd.concat([current, da_only], ignore_index=True))
    with caplog.at_level(logging.WARNING, logger="src.engine.model"):
        joint = DixonColesEngine(DixonColesConfig(xi=0.0)).fit(both, registry=registry)
    assert set(joint.gamma) == {Girone.A, Girone.B, Girone.C, Girone.SB, Girone.DA} and len(joint.current_teams) == 80
    assert Girone.DB not in joint.teams_by_girone and len(joint.historical_teams) == 18


# ============================================================================ FootyStats auto-init
class CountingTransport(httpx.MockTransport):
    def __init__(self, handler: Callable[[httpx.Request], Any]) -> None:
        super().__init__(handler)
        self.requests: list[httpx.Request] = []

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        return await super().handle_async_request(request)


class RecordingSleeper:
    def __init__(self) -> None:
        self.calls: list[float] = []

    async def __call__(self, delay: float) -> None:
        self.calls.append(delay)


def _envelope(rows: list[dict[str, Any]]) -> httpx.Response:
    return httpx.Response(
        200,
        json={
            "success": True,
            "pager": {"current_page": 1, "max_page": 1, "results_per_page": 600, "total_results": len(rows)},
            "metadata": {"request_limit": 1800, "request_remaining": 1700},
            "data": rows,
        },
    )


def _league_entry(name: str, league_name: str, ids: dict[str, int]) -> dict[str, Any]:
    return {
        "name": name,
        "country": "Italy",
        "league_name": league_name,
        "season": [{"id": season_id, "year": int(label[:4]) * 10000 + int(label[:4]) + 1} for label, season_id in ids.items()],
    }


#: (id, name, cleanName, english_name, alt_names): the spellings differ after ``normalize_name`` on purpose (``"Club"`` and legal
#: suffixes are dropped by it, so ``"ASD Borgo 01 Calcio"`` would be the same spelling as ``"Borgo 01"`` and no alias at all).
SERIE_D_TEAMS: tuple[tuple[int, str, str, str, list[str]], ...] = tuple(
    (9000 + i, f"Unione Sportiva Borgo {i:02d}", f"Borgo {i:02d}", f"Borgo {i:02d} Town", [f"Borgo{i:02d}"]) for i in range(1, 19)
)
#: Two clubs sharing a nickname: the auto-init must write it for one of them only (one spelling -> one club).
SHARED_NICKNAME = "I Biancazzurri"


def _serie_d_teams_payload() -> list[dict[str, Any]]:
    rows = []
    for team_id, name, clean, english, alt in SERIE_D_TEAMS:
        extra = [SHARED_NICKNAME] if team_id in (9001, 9002) else []
        rows.append({"id": team_id, "name": name, "cleanName": clean, "english_name": english, "alt_names": [*alt, *extra]})
    return rows


def _season_shift(season_id: int) -> int:
    """Seconds to subtract from the 2026/27 kick-offs so that a past season's matches fall in their own season."""
    for ids in SEASON_IDS.values():
        for label, candidate in ids.items():
            if candidate == season_id:
                return (2026 - int(label[:4])) * 365 * 86400
    return 0


def _serie_d_matches_payload(season_id: int) -> list[dict[str, Any]]:
    """Three complete matches with the thin Serie D statistics (no shots, no xG, no odds fields) and one fixture."""
    rows: list[dict[str, Any]] = []
    teams = SERIE_D_TEAMS
    shift = _season_shift(season_id)
    for k, kickoff in enumerate(unix - shift for unix in PLAYED_UNIX):
        home, away = teams[2 * k], teams[2 * k + 1]
        rows.append(
            {
                "id": season_id * 100 + k + 1,
                "status": "complete",
                "date_unix": kickoff,
                "game_week": k + 1,
                "homeID": home[0],
                "awayID": away[0],
                "home_name": home[1],
                "away_name": away[1],
                "homeGoalCount": 2 - k if k < 2 else 0,
                "awayGoalCount": 1,
                "team_a_shots": 0 if k else -1,
                "team_b_shots": 0,
                "team_a_shotsOnTarget": 0,
                "team_b_shotsOnTarget": 0,
                "team_a_corners": 0,
                "team_b_corners": 0,
                "team_a_xg": 0,
                "team_b_xg": 0,
            }
        )
    home, away = teams[6], teams[7]
    rows.append(
        {
            "id": season_id * 100 + 50,
            "status": "incomplete",
            "date_unix": FIXTURE_UNIX - shift,
            "game_week": 17,
            "homeID": home[0],
            "awayID": away[0],
            "home_name": home[1],
            "away_name": away[1],
            "homeGoalCount": 0,
            "awayGoalCount": 0,
            # only a partial 1X2 book: odds_ft_2 missing, no Over/Under, no BTTS
            "odds_ft_1": 2.1,
            "odds_ft_x": 3.2,
        }
    )
    return rows


def make_router(*, chosen: tuple[str, ...] = ("DA",), teams_payload: list[dict[str, Any]] | None = None) -> Callable[[httpx.Request], httpx.Response]:
    """Serve the Serie D leagues in ``chosen``; every Serie D season id answers with the synthetic teams and matches."""
    all_ids = {season_id for ids in SEASON_IDS.values() for season_id in ids.values()}

    def router(request: httpx.Request) -> httpx.Response:
        assert request.url.host == HOST and request.url.params.get("key") == KEY
        path = request.url.path
        if path == "/league-list":
            entries = [_league_entry(f"Italy Serie D Group {code[1]}", f"Serie D Group {code[1]}", SEASON_IDS[code]) for code in chosen]
            entries.append(_league_entry("Italy Serie C Group A", "Serie C Group A", {"2026/27": 17487, "2025/26": 15535, "2024/25": 12659}))
            return _envelope(entries)
        season_id = int(request.url.params.get("season_id", "0"))
        if season_id not in all_ids:
            return httpx.Response(200, json={"success": False, "message": "unknown season"})
        if path == "/league-teams":
            return _envelope(teams_payload if teams_payload is not None else _serie_d_teams_payload())
        if path == "/league-matches":
            return _envelope(_serie_d_matches_payload(season_id))
        return httpx.Response(404, json={"success": False, "message": "not found"})

    return router


@pytest.fixture
def http_cfg() -> HttpConfig:
    return HttpConfig(timeout_seconds=5.0, max_retries=1, backoff_base_seconds=0.1, backoff_max_seconds=0.2, max_concurrency=2, min_delay_seconds=0.0, cache_ttl_hours=1.0)


@pytest.fixture
def config_copy(tmp_path: Path) -> Path:
    """A private copy of the shipped dictionaries: the auto-init must never touch the repository's files in tests."""
    config_dir = tmp_path / "config"
    config_dir.mkdir()
    for name in ("teams_2026_27.json", "teams_2025_26.json"):
        shutil.copy(CONFIG_DIR / name, config_dir / name)
    return config_dir


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def make_provider(
    http: HttpConfig,
    config_dir: Path,
    mapping_path: Path,
    *,
    router: Callable[[httpx.Request], httpx.Response] | None = None,
    registry_path: Path | None = None,
) -> tuple[fs.FootyStatsProvider, CountingTransport]:
    transport = CountingTransport(router or make_router())
    endpoints = load_endpoints().footystats
    client = fs.FootyStatsClient(api_key=KEY, endpoints=endpoints, http=http, transport=transport, sleeper=RecordingSleeper(), rng=random.Random(1))
    registry = TeamRegistry.from_json(registry_path or config_dir / "teams_2026_27.json")
    matcher = TeamMatcher(Harvester.registry_with_history(registry, config_dir))
    mapper = fs.FootyStatsTeamMapper(registry, matcher, mapping_path=mapping_path)
    provider = fs.FootyStatsProvider(client, mapper, endpoints, now=lambda: NOW, config_dir=config_dir, registry_path=registry_path)
    return provider, transport


@pytest.mark.asyncio
async def test_harvest_auto_initialises_an_empty_serie_d_girone(http_cfg: HttpConfig, config_copy: Path, tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    shipped_before = _digest(CONFIG_DIR / "teams_2026_27.json")
    provider, transport = make_provider(http_cfg, config_copy, tmp_path / "map.json")
    assert provider.registry.teams("DA") == () and provider.registry_path == config_copy / "teams_2026_27.json"
    with caplog.at_level(logging.INFO):
        async with provider.client:
            result = await provider.harvest(["DA"], seasons=2)
    # league-teams was called for the current season once (the listing is reused by the harvest) and once for the past one
    teams_calls = [int(r.url.params["season_id"]) for r in transport.requests if r.url.path == "/league-teams"]
    assert teams_calls == [SEASON_IDS["DA"]["2026/27"], SEASON_IDS["DA"]["2025/26"]]
    assert result.initialised_competitions == ("DA",) and result.registry_path == config_copy / "teams_2026_27.json"
    assert "initialised=DA" in result.summary()
    # the dictionary copy gained the 18 teams (canonical = cleanName, aliases = name / english name / alt_names) ...
    saved = json.loads((config_copy / "teams_2026_27.json").read_text(encoding="utf-8"))
    group = saved["gironi"]["DA"]
    assert len(group) == 18 and group["Borgo 01"] == ["Unione Sportiva Borgo 01", "Borgo 01 Town", "Borgo01", SHARED_NICKNAME]
    assert group["Borgo 02"] == ["Unione Sportiva Borgo 02", "Borgo 02 Town", "Borgo02"], "a nickname shared by two clubs is written for the first only"
    assert saved["gironi"]["DB"] == {} and len(saved["gironi"]["A"]) == 20 and saved["season"] == "2026/27"
    reloaded = TeamRegistry.from_json(config_copy / "teams_2026_27.json")  # the file loads: no alias collision
    assert reloaded.n_teams == 98 and reloaded.gironi == (Girone.A, Girone.B, Girone.C, Girone.SB, Girone.DA)
    assert reloaded.exact(SHARED_NICKNAME) == "Borgo 01" and reloaded.exact("Borgo 05 Town") == "Borgo 05" and reloaded.perimeter_label("Serie D") == "Serie D 2026/27 (18 squadre)"
    # ... the provider reloaded it, so the Serie D teams are current, not unmapped ...
    assert provider.registry.teams("DA") == tuple(f"Borgo {i:02d}" for i in range(1, 19)) and provider.registry.n_teams == 98
    assert result.unmapped_teams == {} and result.historical_teams == ()
    record = next(r for r in caplog.records if r.getMessage() == "registry initialised from FootyStats league-teams")
    assert record.competitions == ["DA"] and record.teams == {"DA": 18}
    # ... the matches carry the thin Serie D statistics as NaN and the partial odds book is simply absent
    df = result.matches
    assert len(df) == 6 and set(df["girone"]) == {"DA"} and df["home_team"].isin(provider.registry.teams("DA")).all()
    assert df[["home_shots", "away_shots", "home_shots_target", "home_corners", "home_xg", "away_xg"]].isna().all().all()
    assert df["home_goals"].tolist() == [2, 1, 0, 2, 1, 0] and result.n_matches_by_competition == {"DA": {"2026/27": 3, "2025/26": 3}}
    assert df["date"].dt.year.tolist() == [2025, 2025, 2025, 2026, 2026, 2026]  # the past season's matches sit in their own season
    assert len(result.fixtures) == 1 and result.fixtures[0].date == date(2027, 1, 15) and result.fixtures[0].girone is Girone.DA
    assert result.odds == [] and result.failures == {} and result.skipped == {}
    # the repository's own dictionary was never touched
    assert _digest(CONFIG_DIR / "teams_2026_27.json") == shipped_before
    # a second run finds the teams and initialises nothing
    again, transport2 = make_provider(http_cfg, config_copy, tmp_path / "map.json")
    async with again.client:
        second = await again.harvest(["DA"], seasons=1)
    assert second.initialised_competitions == () and second.registry_path is None and len(second.matches) == 3
    assert [r.url.path for r in transport2.requests] == ["/league-list", "/league-teams", "/league-matches"]


@pytest.mark.asyncio
async def test_harvest_without_auto_init_keeps_serie_d_teams_historical(http_cfg: HttpConfig, config_copy: Path, tmp_path: Path) -> None:
    before = _digest(config_copy / "teams_2026_27.json")
    provider, transport = make_provider(http_cfg, config_copy, tmp_path / "map.json")
    async with provider.client:
        result = await provider.harvest(["DA"], seasons=1, auto_init_registry=False)
    assert result.initialised_competitions == () and _digest(config_copy / "teams_2026_27.json") == before
    assert len(result.matches) == 3 and set(result.unmapped_teams) == {f"Borgo {i:02d}" for i in range(1, 19)}  # every listed team is unknown
    assert sum(1 for r in transport.requests if r.url.path == "/league-teams") == 1
    # a registry whose season is not a season label is never auto-initialised
    synthetic = TeamRegistry.from_teams("sim", {"A": ("x", "y")})
    mapper = fs.FootyStatsTeamMapper(synthetic, TeamMatcher(synthetic), mapping_path=tmp_path / "map2.json")
    client = fs.FootyStatsClient(api_key=KEY, endpoints=load_endpoints().footystats, http=http_cfg, transport=CountingTransport(make_router()), sleeper=RecordingSleeper())
    sim_provider = fs.FootyStatsProvider(client, mapper, now=lambda: NOW, config_dir=config_copy)
    async with client:
        sim_result = await sim_provider.harvest(["DA"], seasons=1)
    assert sim_result.initialised_competitions == () and not (config_copy / "teams_sim.json").exists()


@pytest.mark.asyncio
async def test_resolve_seasons_falls_back_to_the_configured_serie_d_ids(http_cfg: HttpConfig, config_copy: Path, tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    """``Italy Serie D Group B`` is not among the chosen leagues: its ids come from competitions.toml, with a warning."""
    provider, _ = make_provider(http_cfg, config_copy, tmp_path / "map.json", router=make_router(chosen=("DA",)))
    with caplog.at_level(logging.WARNING):
        async with provider.client:
            resolved = await provider.client.resolve_seasons(["DA", "DB"], seasons=3)
    assert [r.season_id for r in resolved["DA"]] == [17664, 15710, 13475] and [r.season_id for r in resolved["DB"]] == [17667, 15706, 13232]
    warning = next(r for r in caplog.records if r.getMessage() == "league not among the account's chosen leagues, using configured season ids")
    assert warning.competition == "DB" and warning.league == "Italy Serie D Group B" and "footystats.org" in warning.advice
    assert fs.competition_season_ids("DB") == SEASON_IDS["DB"] and fs.competition_season_ids(Girone.DI) == SEASON_IDS["DI"]
    assert fs.competition_season_ids("A", load_endpoints().footystats) == {"2024/25": 12659, "2025/26": 15535, "2026/27": 17487}
    assert fs.competition_season_ids("ZZ") == {} and fs.competition_season_ids("ZZ", load_endpoints().footystats) == {}


def test_write_registry_update_never_writes_a_dictionary_that_would_not_load(tmp_path: Path) -> None:
    teams = [fs.TeamInfo.from_payload(row) for row in _serie_d_teams_payload()]
    assert all(t is not None for t in teams)
    path = tmp_path / "teams_2026_27.json"
    path.write_text(json.dumps({"season": "2026/27", "gironi": {"A": {"Borgo 03": ["Terzo"]}, "DA": {}}}), encoding="utf-8")
    diff = fs.write_registry_update(path, "2026_27", {"DA": [t for t in teams if t is not None]}, matcher=None)
    saved = json.loads(path.read_text(encoding="utf-8"))
    assert len(saved["gironi"]["DA"]) == 17 and diff.conflicts == {"Borgo 03": "FootyStats lists it in DA, the file in A; left unchanged"}
    assert diff.skipped_aliases == {"Borgo 02": [SHARED_NICKNAME]} and "aliases not written for 1 teams" in diff.describe()
    assert TeamRegistry.from_json(path).n_teams == 18  # loads: the shared nickname belongs to one club
    # identity comes from the club's own names, never from a shared nickname
    everton_like = fs.TeamInfo(1, "Everton FC", "Everton", "Everton FC", ("The Blues",))
    chelsea_like = fs.TeamInfo(2, "Chelsea FC", "Chelsea", "Chelsea FC", ("The Blues",))
    assert everton_like.identity_spellings == ("Everton", "Everton FC") and everton_like.spellings == ("Everton", "Everton FC", "The Blues")
    fresh = fs.write_registry_update(tmp_path / "teams_2027_28.json", "2027/28", {"A": [everton_like, chelsea_like]}, matcher=None)
    assert fresh.added_teams == {"A": ["Everton", "Chelsea"]} and fresh.skipped_aliases == {"Chelsea": ["The Blues"]}
    assert TeamRegistry.from_json(tmp_path / "teams_2027_28.json").exact("The Blues") == "Everton"


def test_cli_defaults_cover_the_thirteen_competitions_and_the_auto_init_flag() -> None:
    args = fs._build_parser().parse_args([])
    assert args.competitions == list(ALL_CODES) and fs.DEFAULT_COMPETITIONS == ALL_CODES and not args.no_auto_init
    assert fs._build_parser().parse_args(["--no-auto-init", "--competitions", "DA", "DB"]).no_auto_init
    help_text = " ".join(fs._build_parser().format_help().split())
    assert "DA = Serie D Girone A" in help_text and "DI = Serie D Girone I" in help_text and "--no-auto-init" in help_text
    assert fs.configured_league_names()[-9:] == [f"Italy Serie D Group {letter}" for letter in "ABCDEFGHI"]
    assert fs.competition_label(["DA", "DB"]) == "Serie D" and fs.competition_label(["A", "SB", "DI"]) == "Serie C + Serie B + Serie D"
    assert all(name in fs.MissingApiKeyError("x").instructions for name in ("Italy Serie D Group A", "Italy Serie D Group I"))


def test_harvester_facade_supports_serie_d_through_footystats_only(tmp_path: Path) -> None:
    cfg = load_endpoints()
    registry = load_registry()
    sofa = Harvester(endpoints=cfg, registry=registry, cache_dir=tmp_path / "a", providers=("sofascore",))
    assert sofa.supported_gironi() == (Girone.A, Girone.B, Girone.C) and sofa.supported_gironi("sofascore") == (Girone.A, Girone.B, Girone.C)
    assert sofa.supported_gironi("footystats") == tuple(Girone) and sofa.supported_gironi("fotmob") == tuple(Girone)
    footy = Harvester(endpoints=cfg, registry=registry, cache_dir=tmp_path / "b", providers=("footystats", "sofascore"), api_key=KEY)
    assert footy.supported_gironi() == tuple(Girone) and len(footy.supported_gironi()) == 13
    assert footy._supports("sofascore", Girone.DA) is not None and footy._supports("footystats", Girone.DA) is None
    with pytest.raises(ValueError, match="unknown provider"):
        sofa.supported_gironi("opta")


@pytest.mark.asyncio
async def test_harvester_facade_adopts_the_registry_filled_by_the_auto_init(http_cfg: HttpConfig, config_copy: Path, tmp_path: Path) -> None:
    from dataclasses import replace  # noqa: PLC0415

    cfg = load_endpoints()
    cfg = replace(cfg, http=http_cfg)
    registry = TeamRegistry.from_json(config_copy / "teams_2026_27.json")
    harvester = Harvester(
        endpoints=cfg,
        registry=registry,
        cache_dir=tmp_path / "raw",
        providers=("footystats",),
        api_key=KEY,
        transport=CountingTransport(make_router()),
        sleeper=RecordingSleeper(),
        config_dir=config_copy,
        mapping_path=tmp_path / "map.json",
    )
    assert harvester.registry.teams("DA") == ()
    result = await harvester.harvest_all([Girone.DA], seasons=1)
    assert len(result.matches) == 3 and result.failures == {} and result.unmapped_teams == {} and result.provider == "footystats"
    assert harvester.registry.teams("DA") == tuple(f"Borgo {i:02d}" for i in range(1, 19))  # adopted after the auto-init
    assert harvester.matcher.is_current("Borgo 01") and TeamRegistry.from_json(config_copy / "teams_2026_27.json").n_teams == 98


# ============================================================================ automation and launchers
def test_workflow_runs_footystats_for_every_competition_and_commits_the_dictionaries() -> None:
    path = ROOT / ".github" / "workflows" / "aggiorna-dati.yml"
    workflow = yaml.safe_load(path.read_text(encoding="utf-8"))
    triggers = workflow.get("on", workflow.get(True))  # PyYAML reads the bare key `on` as the boolean True
    assert workflow["name"] and "schedule" in triggers and "workflow_dispatch" in triggers
    competitions_input = triggers["workflow_dispatch"]["inputs"]["competitions"]
    assert competitions_input["default"] == "" and "DA" in competitions_input["description"], "empty = every configured competition"
    steps = {step["name"]: step for step in workflow["jobs"]["harvest"]["steps"]}
    footy = steps["Raccolta dati (FootyStats)"]["run"]
    assert "python -m src.ingestion.footystats" in footy and "--seasons" in footy
    assert "--competitions A B C SB" not in footy, "the default (every configured competition, Serie D included) is used unless narrowed"
    sofa = steps["Raccolta dati (SofaScore, nessun segreto FOOTYSTATS_API_KEY)"]["run"]
    assert "--gironi A B C" in sofa and "DA" not in sofa
    commit = steps["Commit dei CSV aggiornati"]["run"]
    assert "config/teams_2026_27.json" in commit and "config/footystats_teams.json" in commit
    summary = steps["Riepilogo (partite per competizione e stagione)"]["run"]
    assert "registry initialised" in summary and "Serie D" in path.read_text(encoding="utf-8")


def test_launchers_harvest_every_competition_with_footystats() -> None:
    bat = (ROOT / "avvia.bat").read_text(encoding="utf-8")
    cmd = (ROOT / "avvia.command").read_text(encoding="utf-8")
    for text in (bat, cmd):
        footy_line = next(line for line in text.splitlines() if "src.ingestion.footystats" in line and "--seasons 3" in line)
        assert "--competitions A B C SB" not in footy_line, "FootyStats runs for every configured competition (Serie D included)"
        assert "Serie D" in text
        assert re.search(r"src\.ingestion\.harvester --gironi A B C --seasons 3", text), "the SofaScore path stays A B C"
