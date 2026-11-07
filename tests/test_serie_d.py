"""Serie D (nine gironi ``DA`` .. ``DI``, family ``'Serie D'``): configuration, the shipped 2026/27 lists, registries
with empty competitions, simulator and engine behaviour, the FootyStats registry sync of the team lists and the launchers.

The FootyStats side runs offline on ``httpx.MockTransport``: the router serves a league list with the Serie D
league, a synthetic 18-team ``/league-teams`` listing and ``/league-matches`` rows with the thin statistics
FootyStats publishes for Serie D (shots and xG at ``0``, no odds fields at all). Like the live API (verified
2026-11-07) it refuses ``/league-teams`` and ``/league-matches`` of a league the account has not chosen with the
``success: false`` envelope *League is not chosen by the user*, and it always lists ``Italy Serie C Group A``
(served with the shipped girone A teams) so the tests can check that Serie C is still written when Serie D is not.
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
from src.common.teams import PerimeterLabel, TeamRegistry, dumps_registry_payload, load_registry, normalize_name
from src.engine.model import DixonColesConfig, DixonColesEngine, build_team_index
from src.ingestion import footystats as fs
from src.ingestion.harvester import Harvester, HttpConfig, _write_decision, load_endpoints
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
#: ``Italy Serie C Group A``: always among the chosen leagues of the mock account (ids of config/competitions.toml).
SERIE_C_A_IDS = {"2024/25": 12659, "2025/26": 15535, "2026/27": 17487}
#: ``message`` of the ``success: false`` envelope the live API answers for a league the account has not chosen.
NOT_CHOSEN_MESSAGE = (
    "League is not chosen by the user. Please contact support or check your account settings. "
    "League may not exist, or is not available to this user."
)
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
def test_shipped_registry_lists_the_nine_serie_d_gironi_of_footystats() -> None:
    registry = load_registry()
    assert registry.n_teams == 242 and registry.gironi == tuple(Girone) and registry.missing_competitions() == ()
    assert registry.families() == ("Serie C", "Serie B", "Serie D")
    assert all(len(registry.teams(code)) == 18 for code in SERIE_D_CODES) and len(registry.teams("Serie D Girone I")) == 18
    with pytest.raises(ValueError, match="Unknown girone"):
        registry.teams("D")
    label = registry.perimeter_label
    assert label == "Serie C 2026/27 Gironi A/B/C + Serie B + Serie D Gironi A-I (242 squadre)" and isinstance(label, PerimeterLabel) and isinstance(label, str)
    assert label("Serie D") == registry.perimeter_label_for("Serie D") == "Serie D 2026/27 Gironi A-I (162 squadre)"
    assert label(family="Serie C") == "Serie C 2026/27 Gironi A/B/C (60 squadre)" and label("Serie B") == "Serie B 2026/27 (20 squadre)"
    assert label(None) == label and f"Perimetro: {label}" == "Perimetro: Serie C 2026/27 Gironi A/B/C + Serie B + Serie D Gironi A-I (242 squadre)"
    with pytest.raises(ValueError, match="Unknown family"):
        registry.perimeter_label_for("Serie A")
    # the shipped file lists every code; Serie D canonical names are the FootyStats cleanName (two documented exceptions)
    payload = json.loads((CONFIG_DIR / "teams_2026_27.json").read_text(encoding="utf-8"))
    assert list(payload["gironi"]) == list(ALL_CODES) and all(len(payload["gironi"][code]) == 18 for code in SERIE_D_CODES)
    assert "sync_registry" in payload["note"] and "AC Palermo Serie D" in payload["note"]
    assert registry.girone_of("AC Palermo Serie D") is Girone.DI and registry.girone_of("Palermo") is Girone.SB
    assert registry.girone_of("Trapani") is Girone.DI and registry.exact("Trapani 1905") == "Trapani"  # Serie C 2025/26: one name across divisions
    assert registry.girone_of("Milan II") is Girone.DB and registry.girone_of("Ternana") is Girone.DE and registry.girone_of("Triestina") is Girone.DC
    merged = load_registry(with_history=True)
    assert merged.n_teams == 242 and merged.missing_competitions() == ()
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
    serie_c_b = serie_c_b_registry()  # Serie D configured but without teams
    league = simulate_league(seed=7, as_of=date(2026, 11, 6), registry=serie_c_b)
    assert sorted(set(league.matches["girone"])) == ["A", "B", "C", "SB"] and len(league.matches) == 440 and len(league.fixtures) == 40
    assert set(league.true_params.gamma) == set(Girone), "the true parameters cover every configured competition"
    history = simulate_history(seed=7, seasons=("2025/26",), registry=serie_c_b)
    assert sorted(set(history.matches["girone"])) == ["A", "B", "C", "SB"] and len(history.matches) == 1520
    assert history.season_registries["2025/26"].gironi == (Girone.A, Girone.B, Girone.C, Girone.SB)
    params = sample_true_params(load_registry(), np.random.default_rng(0))
    assert len(params.alpha) == 242 and set(params.gamma) == set(Girone)
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
    """Serie D matches exist but the registry has no Serie D teams (an older dictionary, or the sync disabled). The
    index must not crash: those teams are historical of their own competition (own gamma, own group)."""
    registry = serie_c_b_registry()
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
    current = simulate_league(seed=7, n_rounds=6, with_odds=False, registry=registry).matches
    both = validate_matches_df(pd.concat([current, da_only], ignore_index=True))
    with caplog.at_level(logging.WARNING, logger="src.engine.model"):
        joint = DixonColesEngine(DixonColesConfig(xi=0.0)).fit(both, registry=registry)
    assert set(joint.gamma) == {Girone.A, Girone.B, Girone.C, Girone.SB, Girone.DA} and len(joint.current_teams) == 80
    assert Girone.DB not in joint.teams_by_girone and len(joint.historical_teams) == 18


# ============================================================================ FootyStats registry sync
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


#: The shipped Serie C girone A clubs as FootyStats would list them (ids 8001..): already in the dictionary, so no auto-init.
SERIE_C_A_TEAMS: tuple[tuple[int, str, str, str, list[str]], ...] = tuple(
    (8000 + i, f"{name} Calcio", name, name, []) for i, name in enumerate(load_registry().teams("A"), start=1)
)


def _teams_payload(teams: tuple[tuple[int, str, str, str, list[str]], ...]) -> list[dict[str, Any]]:
    return [{"id": team_id, "name": name, "cleanName": clean, "english_name": english, "alt_names": list(alt)} for team_id, name, clean, english, alt in teams]


def _serie_d_teams_payload() -> list[dict[str, Any]]:
    rows = []
    for team_id, name, clean, english, alt in SERIE_D_TEAMS:
        extra = [SHARED_NICKNAME] if team_id in (9001, 9002) else []
        rows.append({"id": team_id, "name": name, "cleanName": clean, "english_name": english, "alt_names": [*alt, *extra]})
    return rows


def _season_shift(season_id: int) -> int:
    """Seconds to subtract from the 2026/27 kick-offs so that a past season's matches fall in their own season."""
    for ids in (*SEASON_IDS.values(), SERIE_C_A_IDS):
        for label, candidate in ids.items():
            if candidate == season_id:
                return (2026 - int(label[:4])) * 365 * 86400
    return 0


def _serie_d_matches_payload(season_id: int, teams: tuple[tuple[int, str, str, str, list[str]], ...] = SERIE_D_TEAMS) -> list[dict[str, Any]]:
    """Three complete matches with the thin Serie D statistics (no shots, no xG, no odds fields) and one fixture."""
    rows: list[dict[str, Any]] = []
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


def make_router(
    *,
    chosen: tuple[str, ...] = ("DA",),
    teams_payload: list[dict[str, Any]] | None = None,
    league_list_status: int | None = None,
) -> Callable[[httpx.Request], httpx.Response]:
    """A mock FootyStats account holding ``Italy Serie C Group A`` plus the Serie D leagues in ``chosen``.

    Every known season id answers with synthetic teams and matches (the shipped girone A clubs for Serie C, the
    ``SERIE_D_TEAMS`` for Serie D) **when its league is chosen**; the season ids of a Serie D league that is not
    chosen get the live refusal envelope (``success: false``, :data:`NOT_CHOSEN_MESSAGE`), exactly like the real
    API. ``league_list_status`` makes ``/league-list`` fail with that HTTP status (the only case where the
    configured season ids may stand in).
    """
    serie_d_ids = {season_id: code for code, ids in SEASON_IDS.items() for season_id in ids.values()}
    serie_c_ids = set(SERIE_C_A_IDS.values())

    def router(request: httpx.Request) -> httpx.Response:
        assert request.url.host == HOST and request.url.params.get("key") == KEY
        path = request.url.path
        if path == "/league-list":
            if league_list_status is not None:
                return httpx.Response(league_list_status, json={"error": "down"})
            entries = [_league_entry(f"Italy Serie D Group {code[1]}", f"Serie D Group {code[1]}", SEASON_IDS[code]) for code in chosen]
            entries.append(_league_entry("Italy Serie C Group A", "Serie C Group A", SERIE_C_A_IDS))
            return _envelope(entries)
        season_id = int(request.url.params.get("season_id", "0"))
        if season_id not in serie_d_ids and season_id not in serie_c_ids:
            return httpx.Response(200, json={"success": False, "message": "unknown season"})
        if season_id in serie_d_ids and serie_d_ids[season_id] not in chosen:
            return httpx.Response(200, json={"success": False, "message": NOT_CHOSEN_MESSAGE})
        teams = SERIE_C_A_TEAMS if season_id in serie_c_ids else SERIE_D_TEAMS
        if path == "/league-teams":
            if season_id in serie_c_ids:
                return _envelope(_teams_payload(SERIE_C_A_TEAMS))
            return _envelope(teams_payload if teams_payload is not None else _serie_d_teams_payload())
        if path == "/league-matches":
            return _envelope(_serie_d_matches_payload(season_id, teams))
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


@pytest.fixture
def empty_da_copy(config_copy: Path) -> Path:
    """The private copy with Serie D girone A emptied: the state of a competition just added to the configuration."""
    path = config_copy / "teams_2026_27.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["gironi"]["DA"] = {}
    path.write_text(dumps_registry_payload(payload), encoding="utf-8")
    return config_copy


def serie_c_b_registry() -> TeamRegistry:
    """The shipped 2026/27 dictionary restricted to Serie C and Serie B (80 teams): a registry whose Serie D is empty."""
    payload = json.loads((CONFIG_DIR / "teams_2026_27.json").read_text(encoding="utf-8"))
    return TeamRegistry.from_mapping("2026/27", {code: payload["gironi"][code] for code in ("A", "B", "C", "SB")})


def _repository_state() -> dict[str, str]:
    paths = [*sorted(CONFIG_DIR.glob("*.json")), *sorted(p for p in (ROOT / "data").rglob("*") if p.is_file())]
    return {str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest() for path in paths}


@pytest.fixture(autouse=True)
def _repository_untouched() -> Any:
    """The registry sync writes the dictionary it was read from: no test may let it write the repository's files."""
    before = _repository_state()
    yield
    assert _repository_state() == before, "a test wrote config/*.json or data/ of the repository"


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
async def test_harvest_sync_fills_an_empty_serie_d_girone(http_cfg: HttpConfig, empty_da_copy: Path, tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    provider, transport = make_provider(http_cfg, empty_da_copy, tmp_path / "map.json")
    assert provider.registry.teams("DA") == () and provider.registry_path == empty_da_copy / "teams_2026_27.json"
    with caplog.at_level(logging.INFO):
        async with provider.client:
            result = await provider.harvest(["DA"], seasons=2)
    # league-teams was called for the current season once (the listing is reused by the harvest) and once for the past one
    teams_calls = [int(r.url.params["season_id"]) for r in transport.requests if r.url.path == "/league-teams"]
    assert teams_calls == [SEASON_IDS["DA"]["2026/27"], SEASON_IDS["DA"]["2025/26"]]
    assert result.initialised_competitions == ("DA",) and result.registry_path == empty_da_copy / "teams_2026_27.json"
    sync = result.registry_sync
    assert sync is not None and sync.changed and sync.written and sync.competitions["DA"].added == tuple(f"Borgo {i:02d}" for i in range(1, 19))
    assert sync.competitions["DA"].removed == () and sync.total == 242 and "registry_sync=changed:18" in result.summary()
    # the dictionary copy gained the 18 teams (canonical = cleanName, aliases = name / english name / alt_names) ...
    saved = json.loads((empty_da_copy / "teams_2026_27.json").read_text(encoding="utf-8"))
    group = saved["gironi"]["DA"]
    assert len(group) == 18 and group["Borgo 01"] == ["Unione Sportiva Borgo 01", "Borgo 01 Town", "Borgo01", SHARED_NICKNAME]
    assert group["Borgo 02"] == ["Unione Sportiva Borgo 02", "Borgo 02 Town", "Borgo02"], "a nickname shared by two clubs is written for the first only"
    assert sync.skipped_aliases == {"Borgo 02": (SHARED_NICKNAME,)}
    assert len(saved["gironi"]["DB"]) == 18 and len(saved["gironi"]["A"]) == 20 and saved["season"] == "2026/27"
    reloaded = TeamRegistry.from_json(empty_da_copy / "teams_2026_27.json")  # the file loads: no alias collision
    assert reloaded.n_teams == 242 and reloaded.gironi == tuple(Girone)
    assert reloaded.exact(SHARED_NICKNAME) == "Borgo 01" and reloaded.exact("Borgo 05 Town") == "Borgo 05"
    # ... the provider reloaded it, so the Serie D teams are current, not unmapped ...
    assert provider.registry.teams("DA") == tuple(f"Borgo {i:02d}" for i in range(1, 19)) and provider.registry.n_teams == 242
    assert result.unmapped_teams == {} and result.historical_teams == ()
    record = next(r for r in caplog.records if r.getMessage() == "registry initialised from FootyStats league-teams")
    assert record.competitions == ["DA"] and record.teams == {"DA": 18}
    synced = next(r for r in caplog.records if r.getMessage() == "team dictionary synchronised with FootyStats")
    assert synced.added == {"DA": [f"Borgo {i:02d}" for i in range(1, 19)]} and synced.changed and synced.total == 242
    # ... the ids are persisted as the clubs' identity (method 'sync') ...
    mapping = json.loads((tmp_path / "map.json").read_text(encoding="utf-8"))["teams"]
    assert mapping["9001"]["canonical"] == "Borgo 01" and mapping["9001"]["method"] == fs.SYNC_METHOD and mapping["9001"]["competition"] == "DA"
    # ... the matches carry the thin Serie D statistics as NaN and the partial odds book is simply absent
    df = result.matches
    assert len(df) == 6 and set(df["girone"]) == {"DA"} and df["home_team"].isin(provider.registry.teams("DA")).all()
    assert df[["home_shots", "away_shots", "home_shots_target", "home_corners", "home_xg", "away_xg"]].isna().all().all()
    assert df["home_goals"].tolist() == [2, 1, 0, 2, 1, 0] and result.n_matches_by_competition == {"DA": {"2026/27": 3, "2025/26": 3}}
    assert df["date"].dt.year.tolist() == [2025, 2025, 2025, 2026, 2026, 2026]  # the past season's matches sit in their own season
    assert len(result.fixtures) == 1 and result.fixtures[0].date == date(2027, 1, 15) and result.fixtures[0].girone is Girone.DA
    assert result.fixtures[0].status == "scheduled" and result.fixtures[0].kickoff == datetime.fromtimestamp(FIXTURE_UNIX, tz=timezone.utc)
    assert result.odds == [] and result.failures == {} and result.skipped == {}
    # a second run finds the dictionary aligned: nothing written, one league-teams call
    again, transport2 = make_provider(http_cfg, empty_da_copy, tmp_path / "map.json")
    async with again.client:
        second = await again.harvest(["DA"], seasons=1)
    assert second.initialised_competitions == () and second.registry_path is None and len(second.matches) == 3
    assert second.registry_sync is not None and not second.registry_sync.changed and "registry_sync=unchanged" in second.summary()
    assert [r.url.path for r in transport2.requests] == ["/league-list", "/league-teams", "/league-matches"]


@pytest.mark.asyncio
async def test_harvest_sync_replaces_the_clubs_footystats_no_longer_lists(http_cfg: HttpConfig, config_copy: Path, tmp_path: Path) -> None:
    """The shipped DA (18 real clubs) against a listing of 18 other clubs: the dictionary becomes the listing."""
    shipped_da = load_registry().teams("DA")
    provider, _ = make_provider(http_cfg, config_copy, tmp_path / "map.json")
    async with provider.client:
        result = await provider.harvest(["DA"], seasons=1)
    da = result.registry_sync.competitions["DA"]
    assert da.removed == shipped_da and da.added == tuple(f"Borgo {i:02d}" for i in range(1, 19)) and da.total == 18
    assert "rimosse AS Biellese 1902" in da.describe() and "aggiunte Borgo 01" in da.describe()
    assert provider.registry.teams("DA") == da.added and result.initialised_competitions == ()
    assert len(TeamRegistry.from_json(config_copy / "teams_2026_27.json").teams("DB")) == 18, "competitions not synchronised are untouched"


@pytest.mark.asyncio
async def test_harvest_without_sync_keeps_serie_d_teams_historical(http_cfg: HttpConfig, empty_da_copy: Path, tmp_path: Path) -> None:
    before = _digest(empty_da_copy / "teams_2026_27.json")
    for index, flag in enumerate(({"auto_init_registry": False}, {"sync_registry": False})):
        provider, transport = make_provider(http_cfg, empty_da_copy, tmp_path / f"map{index}.json")
        async with provider.client:
            result = await provider.harvest(["DA"], seasons=1, **flag)
        assert result.initialised_competitions == () and result.registry_sync is None and _digest(empty_da_copy / "teams_2026_27.json") == before
        assert len(result.matches) == 3 and set(result.unmapped_teams) == {f"Borgo {i:02d}" for i in range(1, 19)}  # every listed team is unknown
        assert sum(1 for r in transport.requests if r.url.path == "/league-teams") == 1
    # the former keyword wins over the new one when both are given
    provider, _ = make_provider(http_cfg, empty_da_copy, tmp_path / "map.json")
    async with provider.client:
        both = await provider.harvest(["DA"], seasons=1, sync_registry=True, auto_init_registry=False)
    assert both.registry_sync is None and _digest(empty_da_copy / "teams_2026_27.json") == before
    # a registry whose season is not a season label is never synchronised
    synthetic = TeamRegistry.from_teams("sim", {"A": ("x", "y")})
    mapper = fs.FootyStatsTeamMapper(synthetic, TeamMatcher(synthetic), mapping_path=tmp_path / "map2.json")
    client = fs.FootyStatsClient(api_key=KEY, endpoints=load_endpoints().footystats, http=http_cfg, transport=CountingTransport(make_router()), sleeper=RecordingSleeper())
    sim_provider = fs.FootyStatsProvider(client, mapper, now=lambda: NOW, config_dir=empty_da_copy)
    async with client:
        sim_result = await sim_provider.harvest(["DA"], seasons=1)
    assert sim_result.initialised_competitions == () and sim_result.registry_sync is None and not (empty_da_copy / "teams_sim.json").exists()


@pytest.mark.asyncio
async def test_season_resolution_skips_a_serie_d_league_the_account_has_not_chosen(http_cfg: HttpConfig, config_copy: Path, tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    """``Italy Serie D Group B`` is not among the chosen leagues: it is reported as not chosen (the API refuses its data),
    never resolved from the configured ids; those ids stand in only when ``/league-list`` itself is unreachable."""
    provider, transport = make_provider(http_cfg, config_copy, tmp_path / "map.json", router=make_router(chosen=("DA",)))
    with caplog.at_level(logging.WARNING):
        async with provider.client:
            resolution = await provider.client.season_resolution(["DA", "DB"], seasons=3)
            with pytest.raises(fs.FootyStatsError, match="'Italy Serie D Group B'") as caught:
                await provider.client.resolve_seasons(["DA", "DB"], seasons=3)
    assert [r.season_id for r in resolution.seasons["DA"]] == [17664, 15710, 13475] and resolution.codes == ("DA",)
    assert resolution.not_chosen == {"DB": "Italy Serie D Group B"} and not resolution.from_configured_ids
    assert resolution.chosen_leagues == ("Italy Serie C Group A", "Italy Serie D Group A")
    assert resolution.reason("DB") == fs.not_chosen_reason("Italy Serie D Group B") and "footystats.org" in resolution.reason("DB")
    assert "Italy Serie D Group A" in str(caught.value) and "Choose leagues" in str(caught.value)
    warning = next(r for r in caplog.records if r.getMessage() == "league not among the account's chosen leagues, competition skipped")
    assert warning.competition == "DB" and warning.league == "Italy Serie D Group B" and "footystats.org" in warning.advice
    assert all(r.url.path == "/league-list" for r in transport.requests), "nothing of an unchosen league is ever requested"
    # the configured ids exist for every competition ...
    assert fs.competition_season_ids("DB") == SEASON_IDS["DB"] and fs.competition_season_ids(Girone.DI) == SEASON_IDS["DI"]
    assert fs.competition_season_ids("A", load_endpoints().footystats) == SERIE_C_A_IDS
    assert fs.competition_season_ids("ZZ") == {} and fs.competition_season_ids("ZZ", load_endpoints().footystats) == {}
    # ... and are used only when /league-list cannot be reached (then for every requested competition, with a warning)
    down, _ = make_provider(http_cfg, config_copy, tmp_path / "map2.json", router=make_router(chosen=("DA",), league_list_status=503))
    with caplog.at_level(logging.WARNING):
        async with down.client:
            offline = await down.client.season_resolution(["DA", "DB"], seasons=3)
    assert offline.from_configured_ids and offline.not_chosen == {} and offline.chosen_leagues == ()
    assert [r.season_id for r in offline.seasons["DB"]] == [17667, 15706, 13232] and [r.season_id for r in offline.seasons["DA"]] == [17664, 15710, 13475]
    assert any(r.getMessage() == "league-list unavailable, using configured season ids" for r in caplog.records)


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


def test_cli_defaults_cover_the_thirteen_competitions_and_the_sync_flags() -> None:
    args = fs._build_parser().parse_args([])
    assert args.competitions == list(ALL_CODES) and fs.DEFAULT_COMPETITIONS == ALL_CODES and not args.no_auto_init and not args.no_sync
    assert fs._build_parser().parse_args(["--no-auto-init", "--competitions", "DA", "DB"]).no_auto_init
    assert fs._build_parser().parse_args(["--no-sync"]).no_sync and fs._build_parser().parse_args(["--sync-registry"]).sync_registry
    help_text = " ".join(fs._build_parser().format_help().split())
    assert "DA = Serie D Girone A" in help_text and "DI = Serie D Girone I" in help_text and "--no-sync" in help_text and "--sync-registry" in help_text
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
async def test_harvester_facade_adopts_the_registry_synchronised_by_footystats(http_cfg: HttpConfig, empty_da_copy: Path, tmp_path: Path) -> None:
    from dataclasses import replace  # noqa: PLC0415

    config_copy = empty_da_copy
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
    assert harvester.registry.teams("DA") == tuple(f"Borgo {i:02d}" for i in range(1, 19))  # adopted after the sync
    assert harvester.matcher.is_current("Borgo 01") and TeamRegistry.from_json(config_copy / "teams_2026_27.json").n_teams == 242
    assert harvester.mapping_path == tmp_path / "map.json" and Harvester(endpoints=cfg, registry=registry, cache_dir=tmp_path / "x").mapping_path == ROOT / "data" / "footystats_teams.json"


# ============================================================================ launchers
def test_launchers_harvest_every_competition_with_footystats() -> None:
    bat = (ROOT / "avvia.bat").read_text(encoding="utf-8")
    cmd = (ROOT / "avvia.command").read_text(encoding="utf-8")
    for text in (bat, cmd):
        footy_line = next(line for line in text.splitlines() if "src.ingestion.footystats" in line and "--seasons 3" in line)
        assert "--competitions" not in footy_line, "FootyStats runs for every configured competition (Serie D included)"
        assert "--odds-out" not in footy_line and "--no-sync" not in footy_line, "default outputs under data/, registry sync included"
        assert "Serie D" in text and "sincronizzazione" in text and "Chiave:" in text, "the key prompt stays"
        assert "src.ingestion.harvester --gironi" not in text, "FootyStats is the only source of real data: no silent SofaScore fallback"
        assert "pubblica_dati" not in text and "git add" not in text, "nothing is ever published to the repository"
    assert cmd.index("src.ingestion.footystats --seasons 3") < cmd.index("streamlit run")
    assert bat.index("src.ingestion.footystats --seasons 3") < bat.index("streamlit run")


# ============================================================================ review findings: unchosen leagues, fuzzy names, --config-dir
def _patch_cli_client(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, router: Callable[[httpx.Request], httpx.Response]) -> CountingTransport:
    """Make ``fs.main`` talk to ``router``: no key in the environment, the mock transport injected into every client."""
    monkeypatch.delenv("FOOTYSTATS_API_KEY", raising=False)
    monkeypatch.setattr(fs, "DEFAULT_ENV_FILE", tmp_path / "absent.env")
    monkeypatch.setattr(fs, "_utc_now", lambda: NOW)
    transport = CountingTransport(router)
    real_client = fs.FootyStatsClient

    def with_transport(**kwargs: Any) -> fs.FootyStatsClient:
        kwargs.setdefault("transport", transport)
        kwargs.setdefault("sleeper", RecordingSleeper())
        return real_client(**kwargs)

    monkeypatch.setattr(fs, "FootyStatsClient", with_transport)
    return transport


def _cli_args(config_copy: Path, tmp_path: Path, *competitions: str) -> list[str]:
    return [
        "--competitions", *competitions, "--seasons", "1", "--key", KEY, "--config-dir", str(config_copy),
        "--out", str(tmp_path / "p" / "matches.csv"), "--fixtures-out", str(tmp_path / "p" / "fixtures.csv"),
        "--odds-out", str(tmp_path / "o" / "footystats.csv"), "--cache-dir", str(tmp_path / "raw"), "--mapping", str(tmp_path / "map.json"),
    ]  # fmt: skip


@pytest.mark.asyncio
async def test_harvest_skips_serie_d_leagues_the_account_has_not_chosen_and_still_delivers_serie_c(
    http_cfg: HttpConfig, config_copy: Path, tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """Finding 1: an account holding Serie C Group A but no Serie D league must still get Serie C, with the Serie D gironi
    reported as skipped (not chosen), never as failures that block the CSVs, and nothing of them requested."""
    before = _digest(config_copy / "teams_2026_27.json")
    provider, transport = make_provider(http_cfg, config_copy, tmp_path / "map.json", router=make_router(chosen=()))
    with caplog.at_level(logging.WARNING):
        async with provider.client:
            result = await provider.harvest(["A", "DA", "DB"], seasons=1)
    assert len(result.matches) == 3 and set(result.matches["girone"]) == {"A"} and result.seasons == {"A": ["2026/27"]}
    assert result.matches["home_team"].isin(provider.registry.teams("A")).all() and result.unmapped_teams == {}
    assert result.failures == {} and result.initialised_competitions == () and result.registry_path is None
    assert result.not_chosen == {"DA": "Italy Serie D Group A", "DB": "Italy Serie D Group B"}
    assert result.skipped == {"DA": fs.not_chosen_reason("Italy Serie D Group A"), "DB": fs.not_chosen_reason("Italy Serie D Group B")}
    assert result.problem_for("DA") == result.skipped["DA"] and result.problem_for("A") is None
    assert "not_chosen=DA,DB" in result.summary() and "  DA: skipped (" in "\n".join(result.season_lines())
    assert "footystats.org" in result.skipped["DA"] and "Choose leagues" in result.skipped["DA"]
    # nothing of the unchosen leagues was requested (the API would refuse it) and the dictionary was not touched
    serie_d_ids = {season_id for ids in SEASON_IDS.values() for season_id in ids.values()}
    assert not any(int(r.url.params.get("season_id", "0")) in serie_d_ids for r in transport.requests)
    assert [r.url.path for r in transport.requests] == ["/league-list", "/league-teams", "/league-matches"]
    assert _digest(config_copy / "teams_2026_27.json") == before
    # a skipped competition is not a partial harvest: the CLI writes the CSVs without --allow-partial
    assert _write_decision(result, tmp_path / "absent.csv", allow_partial=False) == (True, None)
    summary = next(r for r in caplog.records if r.getMessage() == "competitions skipped: leagues not among the account's chosen leagues")
    assert summary.competitions == ["DA", "DB"] and summary.leagues == ["Italy Serie D Group A", "Italy Serie D Group B"]


@pytest.mark.asyncio
async def test_refusal_of_an_unchosen_league_during_auto_init_and_harvest_is_a_skip(
    http_cfg: HttpConfig, config_copy: Path, tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """Finding 1, league-list unreachable: the configured ids stand in, FootyStats then refuses the unchosen Serie D league
    (live envelope) both in the auto-init and in the harvest -- a warning and a skip each time, Serie C still delivered."""
    before = _digest(config_copy / "teams_2026_27.json")
    provider, transport = make_provider(http_cfg, config_copy, tmp_path / "map.json", router=make_router(chosen=(), league_list_status=503))
    with caplog.at_level(logging.WARNING):
        async with provider.client:
            result = await provider.harvest(["A", "DA"], seasons=1)
    assert len(result.matches) == 3 and result.seasons == {"A": ["2026/27"]} and result.failures == {}
    assert result.not_chosen == {"DA": "Italy Serie D Group A"} and result.skipped == {"DA": fs.not_chosen_reason("Italy Serie D Group A")}
    assert result.initialised_competitions == () and result.registry_path is None and provider.registry.teams("DA") == load_registry().teams("DA")
    assert _digest(config_copy / "teams_2026_27.json") == before, "a competition without a listing is never synchronised (never emptied)"
    assert result.registry_sync is not None and "DA" in result.registry_sync.skipped and "non scelta" in result.registry_sync.skipped["DA"]
    init_warning = next(r for r in caplog.records if r.getMessage() == "registry sync: league-teams unavailable, competition left unchanged")
    assert init_warning.competition == "DA" and init_warning.season_id == 17664 and KEY not in init_warning.error
    refused = next(r for r in caplog.records if r.getMessage() == "competition skipped: FootyStats refused the league as not chosen by the account")
    assert refused.competition == "DA" and refused.league == "Italy Serie D Group A"
    assert not any(r.getMessage() == "footystats season failed" for r in caplog.records), "a refusal is a skip, not a failure"
    refusals = [r for r in transport.requests if int(r.url.params.get("season_id", "0")) == 17664]
    assert [r.url.path for r in refusals] == ["/league-teams", "/league-teams"], "sync and harvest each asked once; no league-matches after the refusal"
    assert _write_decision(result, tmp_path / "absent.csv", allow_partial=False) == (True, None)
    assert fs.is_not_chosen_refusal(fs.FootyStatsError(f"API refused the request: {NOT_CHOSEN_MESSAGE}", url="u"))
    assert not fs.is_not_chosen_refusal(fs.FootyStatsError("API refused the request: Invalid API key", url="u"))
    assert not fs.is_not_chosen_refusal(RuntimeError(NOT_CHOSEN_MESSAGE))


def test_cli_writes_serie_c_and_exits_0_when_the_serie_d_leagues_are_not_chosen(
    monkeypatch: pytest.MonkeyPatch, config_copy: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Finding 1, end to end: the default run on an account without the Serie D leagues writes Serie C and exits 0."""
    shipped_before = _digest(CONFIG_DIR / "teams_2026_27.json")
    _patch_cli_client(monkeypatch, tmp_path, make_router(chosen=()))
    code = fs.main(_cli_args(config_copy, tmp_path, "A", "DA", "DB", "DC"))
    captured = capsys.readouterr()
    assert code == 0, captured.err
    assert "matches=3 fixtures=1" in captured.out and "  A: 2026/27: 3 matches" in captured.out and "  DA: skipped (" in captured.out
    assert "skipped DA: 'Italy Serie D Group A' is not among the account's chosen leagues" in captured.err
    assert "leagues not chosen on the FootyStats account (competitions skipped, not failures): DA = Italy Serie D Group A, DB = Italy Serie D Group B, DC = Italy Serie D Group C" in captured.err
    assert "harvest incomplete" not in captured.err and "failure " not in captured.err and KEY not in captured.out + captured.err
    frame = pd.read_csv(tmp_path / "p" / "matches.csv")
    assert len(frame) == 3 and set(frame["girone"]) == {"A"} and (tmp_path / "p" / "fixtures.csv").exists()
    assert _digest(CONFIG_DIR / "teams_2026_27.json") == shipped_before
    assert json.loads((config_copy / "teams_2026_27.json").read_text(encoding="utf-8"))["gironi"]["DA"] == json.loads((CONFIG_DIR / "teams_2026_27.json").read_text(encoding="utf-8"))["gironi"]["DA"]


def test_cli_sync_writes_the_dictionary_of_config_dir_not_the_repository(
    monkeypatch: pytest.MonkeyPatch, config_copy: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """The registry sync fills <config-dir>/teams_2026_27.json and prints its report; the shipped file is untouched."""
    shipped_before = _digest(CONFIG_DIR / "teams_2026_27.json")
    _patch_cli_client(monkeypatch, tmp_path, make_router(chosen=("DA",)))
    code = fs.main(_cli_args(config_copy, tmp_path, "DA"))
    captured = capsys.readouterr()
    assert code == 0, captured.err
    assert f"Dizionario squadre 2026/27 sincronizzato con FootyStats ({config_copy / 'teams_2026_27.json'}): 36 variazioni" in captured.out
    assert "Serie D Girone A (18 squadre): aggiunte Borgo 01" in captured.out and "rimosse AS Biellese 1902" in captured.out
    assert "perimeter: season 2026/27 (242 current teams)" in captured.out and "registry initialised" not in captured.out
    saved = json.loads((config_copy / "teams_2026_27.json").read_text(encoding="utf-8"))
    assert set(saved["gironi"]["DA"]) == {f"Borgo {i:02d}" for i in range(1, 19)} and len(saved["gironi"]["DB"]) == 18
    assert _digest(CONFIG_DIR / "teams_2026_27.json") == shipped_before, "the repository's dictionary must not change behind the user's back"
    frame = pd.read_csv(tmp_path / "p" / "matches.csv")
    assert len(frame) == 3 and set(frame["girone"]) == {"DA"} and set(frame["home_team"]) <= set(saved["gironi"]["DA"])
    # --sync-registry only aligns the dictionary (no CSV), --no-sync leaves it alone
    (tmp_path / "p" / "matches.csv").unlink()
    assert fs.main([*_cli_args(config_copy, tmp_path, "DA"), "--sync-registry"]) == 0
    out = capsys.readouterr().out
    assert "già allineato a FootyStats" in out and "Serie D Girone A (18 squadre): invariata" in out and not (tmp_path / "p" / "matches.csv").exists()
    before = _digest(config_copy / "teams_2026_27.json")
    assert fs.main([*_cli_args(config_copy, tmp_path, "DA"), "--no-sync"]) == 0
    assert _digest(config_copy / "teams_2026_27.json") == before and "registry_sync=-" in capsys.readouterr().out


def test_write_registry_update_never_adopts_a_fuzzy_match_as_canonical(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    """'Trestina' resembles 'Triestina' (a club of the 2025/26 dictionary): --init-registry writes the clean name and only
    reports the resemblance; a club the dictionary really knows (exact / alias hit) keeps its dictionary name."""
    registry = load_registry("2025_26", CONFIG_DIR)
    matcher = TeamMatcher(registry)
    probe = matcher.match("Trestina", None)
    assert probe.canonical == "Triestina" and probe.method == "fuzzy", "precondition: the fuzzy trap exists"
    path = tmp_path / "teams_2026_27.json"
    path.write_text(json.dumps({"season": "2026/27", "gironi": {"DA": {}, "DE": {}}}), encoding="utf-8")
    trestina = fs.TeamInfo(id=7001, name="Sporting Club Trestina", clean_name="Trestina")
    triestina = fs.TeamInfo(id=7002, name="US Triestina", clean_name="Triestina")  # alias hit in teams_2025_26.json
    with caplog.at_level(logging.WARNING):
        diff = fs.write_registry_update(path, "2026/27", {"DE": [trestina], "DA": [triestina]}, matcher=matcher)
    saved = json.loads(path.read_text(encoding="utf-8"))
    assert saved["gironi"]["DE"] == {"Trestina": ["Sporting Club Trestina"]} and "Triestina" not in saved["gironi"]["DE"]
    assert saved["gironi"]["DA"] == {"Triestina": []}, "continuity works through the dictionary's aliases ('US Triestina' normalises like the canonical: no new alias)"
    assert diff.added_teams == {"DE": ["Trestina"], "DA": ["Triestina"]} and diff.suggestions == {"Trestina": "Triestina"}
    assert "Trestina ~ Triestina" in diff.describe() and "check by hand" in diff.describe()
    record = next(r for r in caplog.records if r.getMessage().startswith("FootyStats team written with its clean name"))
    assert record.team == "Trestina" and record.resembles == "Triestina" and record.competition == "DE"
    reloaded = TeamRegistry.from_json(path)
    assert reloaded.exact("Trestina") == "Trestina" and reloaded.exact("Sporting Club Trestina") == "Trestina"
    assert reloaded.exact("US Triestina") == "Triestina" and reloaded.is_current_team("Trestina") and reloaded.is_current_team("Triestina")
    # a second pass is idempotent and reports nothing new
    again = fs.write_registry_update(path, "2026/27", {"DE": [trestina]}, matcher=matcher)
    assert again.added_teams == {} and again.suggestions == {} and again.unchanged_teams == 1


def test_env_example_lists_every_configured_footystats_league() -> None:
    """Finding 7: the .env template names the nine Serie D leagues (every footystats_name of competitions.toml)."""
    text = (ROOT / ".env.example").read_text(encoding="utf-8")
    for comp in COMPETITIONS:
        assert comp.footystats_name in text, f".env.example must list {comp.footystats_name!r}"
    assert "FOOTYSTATS_API_KEY=" in text and "Choose leagues" in text
    assert "Secrets" in text and 'FOOTYSTATS_API_KEY = "' in text, "Streamlit Cloud: the key goes in the app Secrets"
    assert "PUBLIC" in text and "data/ is git-ignored" in text


def test_docs_say_that_every_league_must_be_chosen_and_that_ids_stand_in_only_for_a_league_list_outage() -> None:
    """Finding 4: README, workflow comment, endpoints.toml and competitions.toml agree with the live API."""
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    endpoints = (ROOT / "config" / "endpoints.toml").read_text(encoding="utf-8")
    competitions = (ROOT / "config" / "competitions.toml").read_text(encoding="utf-8")
    module_doc = fs.__doc__ or ""
    for text, name in ((readme, "README"), (endpoints, "endpoints.toml"), (competitions, "competitions.toml"), (module_doc, "module docstring")):
        assert not re.search(r"(?i)non scelta ripiega|lega non scelta ripiega|not chosen falls back|non è fra quelle scelte nell'account \(la lista live", text), name
        assert "not chosen by the user" in text or "League is not chosen" in text or "non scelta" in text, f"{name} must say what the API does with an unchosen league"
    assert "League is not chosen by the user" in readme and "saltata" in readme and "Choose leagues" in readme
    assert re.search(r"(?i)limitano il numero di leghe|limite .*leghe", readme), "the README mentions the per-plan cap on chosen leagues"
    assert "only when /league-list itself cannot be reached" in endpoints and "only when /league-list itself cannot be reached" in competitions
