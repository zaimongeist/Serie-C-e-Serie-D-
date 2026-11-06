"""Config-driven competitions: ``config/competitions.toml`` -> dynamic ``Girone`` enum -> registry,
engine, simulator, console and dashboard (Serie C Gironi A/B/C + Serie B, more leagues later).
"""
from __future__ import annotations

import logging
import pickle
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import main as pipeline
from src.common import competitions as competitions_module
from src.common.competitions import (
    COMPETITIONS,
    COMPETITIONS_PATH,
    DEFAULT_COMPETITIONS,
    Competition,
    competition,
    competitions_by_family,
    describe_competitions,
    find_competition,
    load_competitions,
    normalize_competition_name,
    parse_competitions,
)
from src.common.schemas import CompetitionBase, Fixture, Girone, MatchRecord, build_girone_enum, matches_to_frame, validate_matches_df
from src.common.teams import TeamRegistry, load_registry
from src.dashboard.state import (
    PRIOR_MIN_MATCHES_PER_TEAM,
    RANKING_COLUMNS,
    VALUE_COLUMNS,
    CoverageReport,
    build_state,
    competition_label,
    coverage_report,
    fixtures_for,
    matches_per_team,
    prior_recommended,
    ranking_table,
    traps_table,
    value_table,
)
from src.engine.model import DixonColesConfig, DixonColesEngine, TeamIndex, build_team_index
from src.simulation.generator import (
    DEFAULT_HOME_ADVANTAGE,
    FALLBACK_HOME_ADVANTAGE,
    HISTORICAL_ABSENCES,
    HISTORICAL_MOVES,
    default_home_advantage,
    sample_true_params,
    simulate_history,
    simulate_league,
)

TODAY = date(2026, 11, 6)
EXTRA_TOML = """
[[competition]]
code = "A"
label = "Serie C Girone A"
short = "Girone A"
country = "Italy"
footystats_name = "Italy Serie C Group A"
sofascore_tournament_id = 141
tier = 3
family = "Serie C"

[[competition]]
code = "B"
label = "Serie C Girone B"
short = "Girone B"
country = "Italy"
footystats_name = "Italy Serie C Group B"
sofascore_tournament_id = 142
tier = 3
family = "Serie C"

[[competition]]
code = "C"
label = "Serie C Girone C"
short = "Girone C"
country = "Italy"
footystats_name = "Italy Serie C Group C"
sofascore_tournament_id = 143
tier = 3
family = "Serie C"

[[competition]]
code = "SB"
label = "Serie B"
short = "Serie B"
country = "Italy"
footystats_name = "Italy Serie B"
sofascore_tournament_id = 53
tier = 2
family = "Serie B"

[[competition]]
code = "SA"
label = "Serie A"
short = "Serie A"
country = "Italy"
footystats_name = "Italy Serie A"
sofascore_tournament_id = 23
tier = 1
family = "Serie A"
"""


@pytest.fixture(scope="module")
def registry() -> TeamRegistry:
    return load_registry()


@pytest.fixture(scope="module")
def sim_state():
    """Simulated 2026/27 pipeline with the four competitions (15 rounds, no history, stationary)."""
    return build_state(simulate=True, seed=7, xi=0.0, n_rounds=15, history_seasons=0)


# ----------------------------------------------------------------------------- configuration file
def test_shipped_toml_defines_the_four_competitions_in_display_order() -> None:
    assert COMPETITIONS_PATH == Path(competitions_module.CONFIG_DIR) / "competitions.toml" and COMPETITIONS_PATH.is_file()
    loaded = load_competitions(COMPETITIONS_PATH)
    assert loaded == COMPETITIONS == DEFAULT_COMPETITIONS, "the file and the built-in fallback must agree"
    assert [c.code for c in COMPETITIONS] == ["A", "B", "C", "SB"]
    sb = competition("SB")
    assert sb == competition("sb") == Competition("SB", "Serie B", "Serie B", "Italy", "Italy Serie B", 53, 2, "Serie B")
    assert sb.to_dict()["footystats_name"] == "Italy Serie B" and sb.tier == 2
    assert competition("A").sofascore_tournament_id == 141 and competition("C").footystats_name == "Italy Serie C Group C"
    with pytest.raises(KeyError):
        competition("SA")
    assert competitions_by_family() == {"Serie C": COMPETITIONS[:3], "Serie B": (COMPETITIONS[3],)}
    assert describe_competitions() == "Serie C Gironi A/B/C + Serie B"
    assert describe_competitions(season="2026/27") == "Serie C 2026/27 Gironi A/B/C + Serie B"
    assert describe_competitions(COMPETITIONS[:3], season="2026/27") == "Serie C 2026/27 Gironi A/B/C"
    assert describe_competitions([COMPETITIONS[3]]) == "Serie B" and describe_competitions([]) == "" and describe_competitions([], season="2026/27") == "2026/27"


def test_custom_toml_with_an_extra_competition_builds_a_fresh_five_member_enum(tmp_path: Path) -> None:
    path = tmp_path / "competitions.toml"
    path.write_text(EXTRA_TOML, encoding="utf-8")
    comps = load_competitions(path)
    assert [c.code for c in comps] == ["A", "B", "C", "SB", "SA"] and comps[:4] == COMPETITIONS
    fresh = build_girone_enum(comps, name="GironeTest", module=__name__)
    assert issubclass(fresh, CompetitionBase) and fresh is not Girone
    assert [m.value for m in fresh] == ["A", "B", "C", "SB", "SA"] and len(fresh) == 5
    assert fresh.parse("Serie A") is fresh.SA is fresh("SA") and fresh.SA.short == "Serie A" and fresh.SA.tier == 1
    assert fresh.parse("B") is fresh.B and fresh.B.label == "Serie C Girone B"
    assert fresh.competitions() == comps and fresh.SA.competition == comps[-1]
    # a member of the global enum is accepted by value, and vice versa
    assert fresh.parse(Girone.C) is fresh.C and Girone.parse(fresh.A) is Girone.A
    with pytest.raises(ValueError):
        Girone.parse(fresh.SA)  # the global configuration has no Serie A
    # the global enum used by every other test is untouched
    assert [m.value for m in Girone] == ["A", "B", "C", "SB"] and Girone.competitions() == COMPETITIONS
    assert fresh.A is not Girone.A and fresh.A == Girone.A == "A"  # str mixin: equal as strings, distinct members
    assert describe_competitions(comps) == "Serie C Gironi A/B/C + Serie B + Serie A"
    with pytest.raises(ValueError):
        build_girone_enum([])
    with pytest.raises(ValueError, match="duplicate"):
        build_girone_enum([comps[0], comps[0]])


def test_loader_errors_and_fallback(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    with pytest.raises(FileNotFoundError):
        load_competitions(tmp_path / "missing.toml")
    bad = tmp_path / "bad.toml"
    bad.write_text("[[competition]\ncode = ", encoding="utf-8")
    with pytest.raises(ValueError, match="invalid TOML"):
        load_competitions(bad)
    table = {"code": "A", "label": "x", "short": "x", "country": "Italy", "footystats_name": "fs", "sofascore_tournament_id": 1, "tier": 3, "family": "f"}
    with pytest.raises(ValueError, match="non-empty"):
        parse_competitions({"competition": []})
    with pytest.raises(ValueError, match="duplicate competition code"):
        parse_competitions({"competition": [table, {**table, "label": "y", "short": "y", "footystats_name": "fs2"}]})
    with pytest.raises(ValueError, match="unknown keys"):
        parse_competitions({"competition": [{**table, "typo": 1}]})
    with pytest.raises(ValueError, match="missing keys"):
        parse_competitions({"competition": [{k: v for k, v in table.items() if k != "tier"}]})
    with pytest.raises(ValueError, match="belongs to both"):
        parse_competitions({"competition": [table, {**table, "code": "B", "label": "x"}]})
    with pytest.raises(ValueError, match="code must be"):
        Competition("serie-b", "Serie B", "Serie B", "Italy", "Italy Serie B", 53, 2, "Serie B")
    with pytest.raises(ValueError, match="tier"):
        Competition("SB", "Serie B", "Serie B", "Italy", "Italy Serie B", 53, 0, "Serie B")
    with pytest.raises(ValueError, match="integer"):
        Competition("SB", "Serie B", "Serie B", "Italy", "Italy Serie B", "53", 2, "Serie B")  # type: ignore[arg-type]
    assert Competition(" sb ", "Serie B", "Serie B", "Italy", "Italy Serie B", 53, 2, "Serie B").code == "SB"
    # imports never fail: a missing or broken file falls back to the built-in definitions (logged)
    with caplog.at_level(logging.INFO, logger="src.common.competitions"):
        assert competitions_module._load_or_default(tmp_path / "missing.toml") == DEFAULT_COMPETITIONS
        assert competitions_module._load_or_default(bad) == DEFAULT_COMPETITIONS
    messages = [r.getMessage() for r in caplog.records]
    assert "competitions file not found, using the built-in definitions" in messages
    assert "competitions file invalid, using the built-in definitions" in messages


def test_loading_the_shipped_file_is_silent_at_info_level(caplog: pytest.LogCaptureFixture) -> None:
    """Regression: ``COMPETITIONS`` is loaded at import time, before ``main.py`` or the harvesters configure
    logging, so the normal case must not print an INFO line on every CLI start (``--log-level ERROR`` included).
    Only the missing/invalid-file fallbacks stay at INFO/ERROR."""
    caplog.clear()
    with caplog.at_level(logging.INFO, logger="src.common.competitions"):
        assert competitions_module._load_or_default(COMPETITIONS_PATH) == COMPETITIONS
    assert [r.getMessage() for r in caplog.records] == []
    caplog.clear()
    with caplog.at_level(logging.DEBUG, logger="src.common.competitions"):
        competitions_module._load_or_default(COMPETITIONS_PATH)
    debug = [r for r in caplog.records if r.getMessage() == "competitions loaded"]
    assert len(debug) == 1 and debug[0].levelno == logging.DEBUG and debug[0].__dict__["codes"] == ["A", "B", "C", "SB"]


# ----------------------------------------------------------------------------- dynamic enum
@pytest.mark.parametrize(
    ("spelling", "expected"),
    [
        ("A", "A"),
        ("a", "A"),
        ("Girone A", "A"),
        ("GIRONE_A", "A"),
        ("girone-a", "A"),
        ("serie c girone a", "A"),
        ("Serie C Girone A", "A"),
        ("Italy Serie C Group A", "A"),
        ("Group B", "B"),
        ("B", "B"),
        ("SB", "SB"),
        ("sb", "SB"),
        ("Serie B", "SB"),
        ("serie_b", "SB"),
        ("SERIE B", "SB"),
        ("Italy Serie B", "SB"),
        ("Girone SB", "SB"),
        ("GIRONEA", "A"),  # filler glued to the code, accepted by the pre-2026/27 enum too
        ("Serie C C", "C"),
        ("Serie B SB", "SB"),  # redundant, not contradictory
        (Girone.C, "C"),
    ],
)
def test_girone_parse_accepts_codes_labels_and_footystats_names(spelling: object, expected: str) -> None:
    member = Girone.parse(spelling)  # type: ignore[arg-type]
    assert member is Girone(expected) and member.value == expected
    assert find_competition(spelling).code == expected


def test_girone_parse_b_means_serie_c_girone_b_and_unknown_spellings_raise() -> None:
    assert Girone.parse("B") is Girone.B and Girone.B.family == "Serie C" and Girone.parse("Serie B") is Girone.SB
    for bad in ("D", "SA", "", "Serie C", "Serie A", "Gironi A/B", "Italy"):
        with pytest.raises(ValueError, match="Unknown girone"):
            Girone.parse(bad)
    with pytest.raises(ValueError):
        Girone("Serie B")  # the plain constructor only knows the codes; parse() knows the labels
    assert normalize_competition_name(" serie_b ") == "SERIE B" and normalize_competition_name("Girone  A") == "GIRONE A"


@pytest.mark.parametrize(
    "spelling",
    ["Serie B Girone A", "Serie B B", "Serie C SB", "Italy A", "Italy Serie B Group A", "Serie C Girone SB", "Italy Girone A", "Serie B Group C"],
)
def test_girone_parse_rejects_contradictory_spellings(spelling: str) -> None:
    """Regression: only filler words and the competition's OWN family may wrap a code. ``'Serie B Girone A'``
    used to resolve to Serie C girone A and ``'Serie C SB'`` to Serie B; a country is not a filler word."""
    with pytest.raises(ValueError, match="Unknown girone"):
        Girone.parse(spelling)
    with pytest.raises(ValueError, match="Unknown girone"):
        find_competition(spelling)
    raw = pd.DataFrame({"match_id": ["m1"], "date": ["2026-08-23"], "round": [1], "girone": [spelling], "home_team": ["a"], "away_team": ["b"], "home_goals": [1], "away_goals": [0]})
    with pytest.raises(ValueError):
        validate_matches_df(raw)  # a mislabelled girone cell never passes as another competition


def test_girone_members_properties_order_and_pickling() -> None:
    assert [m.value for m in Girone] == ["A", "B", "C", "SB"] == [c.code for c in COMPETITIONS]
    assert Girone.A.value == "A" and Girone("SB").value == "SB" and Girone.SB.name == "SB"
    assert Girone.A == "A" and str(Girone.SB) == "SB" and Girone.SB in Girone and "SB" in Girone.__members__
    assert Girone.A.label == "Serie C Girone A" and Girone.A.short == "Girone A" and Girone.A.family == "Serie C" and Girone.A.tier == 3
    assert Girone.SB.label == "Serie B" and Girone.SB.short == "Serie B" and Girone.SB.family == "Serie B" and Girone.SB.tier == 2
    assert Girone.SB.country == "Italy" and Girone.SB.footystats_name == "Italy Serie B"
    assert Girone.C.competition is competition("C") and Girone.competitions() == COMPETITIONS
    assert pickle.loads(pickle.dumps(Girone.SB)) is Girone.SB
    assert sorted([Girone.SB, Girone.A]) == [Girone.A, Girone.SB]  # str ordering of the codes
    assert competition_label("SB") == "Serie B" and competition_label(Girone.A) == "Girone A"


def test_contracts_accept_serie_b() -> None:
    record = MatchRecord("SB-1", date(2026, 8, 22), 1, "Serie B", "Palermo", "Sampdoria", 2, 1)
    assert record.girone is Girone.SB and record.to_dict()["girone"] == "SB"
    fixture = Fixture("SB-2", date(2026, 11, 7), 12, "SB", "Cesena", "Modena")
    assert fixture.girone is Girone.SB
    frame = matches_to_frame([record])
    assert frame["girone"].tolist() == ["SB"]
    raw = pd.DataFrame({"match_id": ["m1", "m2"], "date": ["2026-08-23"] * 2, "round": [1, 1], "girone": ["Serie B", "Italy Serie B"], "home_team": ["a", "c"], "away_team": ["b", "d"], "home_goals": [1, 2], "away_goals": [0, 2]})
    assert validate_matches_df(raw)["girone"].tolist() == ["SB", "SB"]
    with pytest.raises(ValueError, match=r"expected one of \['A', 'B', 'C', 'SB'\]"):
        validate_matches_df(raw.assign(girone=["A", "SA"]))


# ----------------------------------------------------------------------------- registry
def test_registry_has_80_teams_in_four_competitions_and_a_config_derived_perimeter_label(registry: TeamRegistry) -> None:
    assert registry.n_teams == 80 and registry.gironi == (Girone.A, Girone.B, Girone.C, Girone.SB)
    assert all(len(registry.teams(g)) == 20 for g in Girone) and len(registry.teams("Serie B")) == 20
    assert registry.perimeter_label == "Serie C 2026/27 Gironi A/B/C + Serie B (80 squadre)"
    assert registry.canonical_names[60:] == registry.teams(Girone.SB)  # configuration order: Serie B last
    for team in ("Arezzo", "Ascoli", "Benevento", "Vicenza", "Hellas Verona", "Sampdoria", "Südtirol"):
        assert registry.girone_of(team) is Girone.SB and registry.is_current_team(team)
    for team in ("Pescara", "Spezia", "Reggiana"):
        assert registry.girone_of(team) is Girone.B
    assert registry.girone_of("Bari") is Girone.C
    assert registry.exact("FC Südtirol") == registry.exact("Sudtirol") == registry.exact("Alto Adige") == "Südtirol"
    assert registry.exact("LR Vicenza") == "Vicenza" and registry.exact("UC Sampdoria") == "Sampdoria" and registry.exact("Entella") == "Virtus Entella"
    # the 2025/26 file has Serie B too: one canonical name per club across divisions and seasons
    old = load_registry("2025_26")
    assert old.n_teams == 80 and old.girone_of("Pescara") is Girone.SB and old.girone_of("Vicenza") is Girone.A
    merged = load_registry(with_history=True)
    assert merged.exact("Delfino Pescara") == "Pescara" and merged.girone_of("Pescara") is Girone.B and merged.is_current_team("Pescara")
    assert merged.exact("L.R. Vicenza") == "Vicenza" and merged.girone_of("Vicenza") is Girone.SB
    assert {"Frosinone", "Monza", "Venezia", "Ternana"} <= set(merged.historical_names)
    assert not merged.is_current_team("Frosinone") and merged.exact("Frosinone Calcio") == "Frosinone"
    # smaller registries describe only the competitions they hold
    assert TeamRegistry.from_teams("2026/27", {"A": ("x", "y"), "C": ("z", "w")}).perimeter_label == "Serie C 2026/27 Gironi A/C (4 squadre)"
    assert TeamRegistry.from_teams("2026/27", {"SB": ("x", "y")}).perimeter_label == "Serie B 2026/27 (2 squadre)"
    assert TeamRegistry.from_teams("2026/27", {"Serie B": ("x", "y")}).girone_of("x") is Girone.SB
    with pytest.raises(ValueError, match="listed in both"):
        TeamRegistry.from_teams("2026/27", {"A": ("x", "y"), "SB": ("x",)})


def test_registry_rejects_alias_collisions_across_clubs() -> None:
    """Regression: a normalised spelling shared by two clubs used to be overwritten silently, the competition
    listed last winning (``'Verona'`` -> ``'Hellas Verona'`` and Virtus Verona lost its alias)."""
    with pytest.raises(ValueError, match=r"'Verona'.*'Hellas Verona' \[SB\] collides with 'Virtus Verona' \[A\]"):
        TeamRegistry.from_mapping("2026/27", {"A": {"Virtus Verona": ["V. Verona"]}, "SB": {"Hellas Verona": ["Hellas Verona FC", "Verona"]}})
    # the same collision inside one competition, and an alias equal to another club's canonical name
    with pytest.raises(ValueError, match="collides"):
        TeamRegistry.from_mapping("2026/27", {"A": {"Alpha": ["Shared"], "Beta": ["Shared FC"]}})
    with pytest.raises(ValueError, match="collides"):
        TeamRegistry.from_mapping("2026/27", {"A": {"Alpha": ["Beta Calcio"]}, "B": {"Beta": []}})
    # repeating a spelling for the SAME club is fine, and so are the shipped files
    reg = TeamRegistry.from_mapping("2026/27", {"A": {"Alpha": ["Alpha", "Alpha FC", "alpha"]}})
    assert reg.exact("Alpha FC") == "Alpha" and reg.aliases == {"alpha": "Alpha"}
    for season in ("2025_26", "2026_27"):
        assert load_registry(season).n_teams == 80 and load_registry(season, with_history=True).n_teams == 80


# ----------------------------------------------------------------------------- coverage rule
def test_coverage_rule_ignores_registry_competitions_that_were_never_harvested(registry: TeamRegistry) -> None:
    """Regression: Serie B in the registry but never harvested (every SofaScore-only installation) forced
    ``matches_per_team == 0.0`` and the MAP prior for ever, whatever Serie C showed."""
    label = registry.season_label
    full = simulate_league(seed=7, n_rounds=37).matches
    assert matches_per_team(full, registry=registry, season=label) == pytest.approx(37.0) and not prior_recommended(full, registry=registry, season=label)
    serie_c_only = full.loc[full["girone"] != "SB"].reset_index(drop=True)
    report = coverage_report(serie_c_only, registry=registry, season=label)
    assert isinstance(report, CoverageReport) and report.ignored == (Girone.SB,) and report.season == label
    assert set(report.per_competition) == {Girone.A, Girone.B, Girone.C} and report.thinnest == pytest.approx(37.0)
    assert report.thinnest_competition is Girone.A and report.describe_ignored() == "Serie B senza partite nei dati (ignorata nel calcolo della copertura)"
    assert matches_per_team(serie_c_only, registry=registry, season=label) == pytest.approx(37.0)
    assert not prior_recommended(serie_c_only, registry=registry, season=label)
    # the rule still fires on genuinely thin Serie C data ...
    thin = serie_c_only.loc[serie_c_only["round"] <= 4].reset_index(drop=True)
    assert matches_per_team(thin, registry=registry, season=label) == pytest.approx(4.0) and 4.0 < PRIOR_MIN_MATCHES_PER_TEAM
    assert prior_recommended(thin, registry=registry, season=label) and coverage_report(thin, registry=registry, season=label).ignored == (Girone.SB,)
    # ... and a competition with history but no current-season match keeps zero coverage (it exists, it is thin)
    history = simulate_history(seed=7, seasons=("2025/26",)).matches
    sb_history_only = validate_matches_df(pd.concat([history, serie_c_only], ignore_index=True))
    with_history = coverage_report(sb_history_only, registry=registry, season=label)
    assert with_history.ignored == () and with_history.per_competition[Girone.SB] == 0.0 and with_history.thinnest == 0.0
    assert with_history.thinnest_competition is Girone.SB and prior_recommended(sb_history_only, registry=registry, season=label)
    assert with_history.describe_ignored() == ""
    # several ignored competitions are listed in configuration order with a plural participle
    only_a = full.loc[full["girone"] == "A"].reset_index(drop=True)
    assert coverage_report(only_a, registry=registry, season=label).ignored == (Girone.B, Girone.C, Girone.SB)
    assert coverage_report(only_a, registry=registry, season=label).describe_ignored() == "Girone B e Girone C e Serie B senza partite nei dati (ignorate nel calcolo della copertura)"
    # without a registry nothing is ignored and the legacy behaviour is unchanged
    assert coverage_report(serie_c_only).ignored == () and matches_per_team(serie_c_only) == pytest.approx(37.0)
    empty = coverage_report(full.iloc[0:0], registry=registry, season=label)
    assert empty.thinnest == 0.0 and empty.thinnest_competition is None and empty.per_competition == {}
    assert matches_per_team(full.iloc[0:0], registry=registry, season=label) == 0.0


def test_dashboard_and_console_name_the_ignored_competition(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """Serie C only (330 current matches, 11 rounds) through the dashboard and the console: the prior is
    recommended because 11 < 15, not because Serie B is at zero, and the banners say Serie B was ignored."""
    league = simulate_league(seed=7, as_of=TODAY)
    serie_c = league.matches.loc[league.matches["girone"] != "SB"].reset_index(drop=True)
    assert len(serie_c) == 330
    processed = tmp_path / "processed"
    processed.mkdir()
    serie_c.to_csv(processed / "matches.csv", index=False)
    state = build_state(processed_dir=processed, odds_dir=tmp_path / "odds", simulate=False, history_seasons=0)
    assert state.coverage is not None and state.coverage.ignored == (Girone.SB,) and state.coverage.thinnest == pytest.approx(11.0)
    assert state.prior_enabled and state.n_current_matches == 330
    fit_note = next(n for n in state.notes if n.startswith("Stima Dixon-Coles"))
    assert "copertura 11.0 partite per squadra nel girone meno coperto; Serie B senza partite nei dati (ignorata nel calcolo della copertura)" in fit_note
    # with every competition harvested nothing is ignored and the note is unchanged
    league.matches.to_csv(processed / "matches.csv", index=False)
    full = build_state(processed_dir=processed, odds_dir=tmp_path / "odds", simulate=False, history_seasons=0)
    assert full.coverage is not None and full.coverage.ignored == () and "ignorata" not in " ".join(full.notes)
    # console: same wording after the threshold
    serie_c.to_csv(processed / "matches.csv", index=False)
    out = pipeline.main(["--processed", str(processed / "matches.csv"), "--no-export", "--log-level", "ERROR"])
    console = capsys.readouterr().out
    assert out.params.penalty > 0.0
    assert (
        "stima MAP con prior gaussiana (11.0 partite per squadra nel girone meno coperto della stagione 2026/27, soglia 15; "
        "Serie B senza partite nei dati (ignorata nel calcolo della copertura))" in console
    )


# ----------------------------------------------------------------------------- simulator + engine
def test_simulated_league_covers_every_competition_and_the_fit_has_a_separate_serie_b_gamma(registry: TeamRegistry) -> None:
    assert DEFAULT_HOME_ADVANTAGE["SB"] == 0.25 and default_home_advantage() == {Girone.A: 0.30, Girone.B: 0.22, Girone.C: 0.38, Girone.SB: 0.25}
    assert default_home_advantage([Girone.SB]) == {Girone.SB: 0.25} and FALLBACK_HOME_ADVANTAGE == 0.28
    tp = sample_true_params(registry, np.random.default_rng(0))
    assert set(tp.gamma) == set(Girone) and tp.gamma[Girone.SB] == 0.25
    for g in Girone:
        assert sum(tp.alpha[t] for t in registry.teams(g)) == pytest.approx(0.0, abs=1e-9)

    league = simulate_league(seed=7, as_of=TODAY)
    assert sorted(set(league.matches["girone"])) == ["A", "B", "C", "SB"]
    assert len(league.matches) == 11 * 40 and len(league.fixtures) == 40 and sum(1 for fx in league.fixtures if fx.girone is Girone.SB) == 10
    history = simulate_history(seed=7)
    assert len(history.matches) == 2 * 4 * 380
    s26, s25 = history.season_registries["2025/26"], history.season_registries["2024/25"]
    # cross-division movers: Pescara (Serie B 2025/26 -> girone B), Vicenza (girone A 2025/26 -> Serie B), Spezia, Arezzo ...
    assert s26.girone_of("Pescara") is Girone.SB and s26.girone_of("Vicenza") is Girone.A and s26.girone_of("Arezzo") is Girone.B
    assert s26.girone_of("Spezia") is Girone.SB and s26.girone_of("Bari") is Girone.SB and s26.girone_of("Benevento") is Girone.C
    assert "Cremonese" not in s26.canonical_names and "Pisa" not in s26.canonical_names  # Serie A in 2025/26
    assert s25.girone_of("Padova") is Girone.A and s25.girone_of("Cittadella") is Girone.SB and s25.girone_of("Salernitana") is Girone.SB
    assert "Empoli" not in s25.canonical_names and HISTORICAL_ABSENCES["2025/26"] == frozenset({"Cremonese", "Pisa", "Hellas Verona"})
    assert HISTORICAL_MOVES["2025/26"]["Pescara"] is Girone.SB and HISTORICAL_MOVES["2025/26"]["Vicenza"] is Girone.A

    frame = pd.concat([history.matches, league.matches], ignore_index=True)
    engine = DixonColesEngine(DixonColesConfig(xi=0.0))
    params = engine.fit(frame, registry=registry)
    assert params.converged and set(params.gamma) == set(Girone) and len(params.teams_by_girone) == 4
    assert set(params.current_teams) == set(registry.canonical_names) and len(params.current_teams) == 80
    for g, members in params.teams_by_girone.items():
        assert sum(params.alpha[t] for t in members) == pytest.approx(0.0, abs=1e-6), g
    # Serie B has its own home advantage, close to the generator's 0.25 and distinct from the Serie C ones
    assert abs(params.gamma[Girone.SB] - league.true_params.gamma[Girone.SB]) < 0.1
    assert all(abs(params.gamma[Girone.SB] - params.gamma[g]) > 1e-3 for g in (Girone.A, Girone.B, Girone.C))
    # one canonical name across divisions: the mover sits in its CURRENT competition, its old matches stay in the fit
    assert params.girone_of("Pescara") is Girone.B and params.girone_of("Vicenza") is Girone.SB and params.is_current("Vicenza")
    assert ((frame["home_team"] == "Pescara") & (frame["girone"] == "SB")).any()
    assert engine.ranking(Girone.SB)["team"].tolist() and set(engine.ranking("Serie B")["team"]) == set(registry.teams(Girone.SB))
    pred = engine.predict(next(fx for fx in league.fixtures if fx.girone is Girone.SB))
    assert pred.matrix.sum() == pytest.approx(1.0) and pred.fixture.girone is Girone.SB


def test_registry_competition_without_matches_keeps_its_teams_current_and_constrains_them_elsewhere(registry: TeamRegistry, caplog: pytest.LogCaptureFixture) -> None:
    """Serie B in the registry but only Serie C harvested: the promoted clubs (Arezzo, Vicenza, ...) appear in Serie C
    history only. They stay current teams of Serie B, no Serie B home advantage is estimated, and for the
    ``sum(alpha) == 0`` constraint they join the girone they played in instead of being centred on their own."""
    past = simulate_history(seed=7, seasons=("2025/26",)).matches
    serie_c_only = past.loc[past["girone"] != "SB"].reset_index(drop=True)
    with caplog.at_level(logging.WARNING, logger="src.engine.model"):
        index = build_team_index(validate_matches_df(serie_c_only), registry=registry)
    assert index.gironi == (Girone.A, Girone.B, Girone.C) and Girone.SB in index.teams_by_girone
    promoted = index.teams_by_girone[Girone.SB]
    assert {"Arezzo", "Ascoli", "Benevento", "Vicenza"} <= set(promoted) and all(index.is_current(t) for t in promoted)
    assert index.constraint_girone_of("Vicenza") is Girone.A and index.constraint_girone_of("Arezzo") is Girone.B
    assert index.girone_of("Vicenza") is Girone.SB and set(index.constraint_girone) == set(promoted)
    groups = index.constraint_groups()
    assert set(groups) == {Girone.A, Girone.B, Girone.C} and "Vicenza" in groups[Girone.A] and sum(len(m) for m in groups.values()) == index.n_teams
    assert any("registry gironi with current teams but no matches" in r.getMessage() for r in caplog.records)
    assert len(index.alpha_groups()) == 3 and index.team_girone_positions().max() == 2

    params = DixonColesEngine(DixonColesConfig(xi=0.0)).fit(serie_c_only, registry=registry)
    assert params.converged and set(params.gamma) == {Girone.A, Girone.B, Girone.C}
    assert params.girone_of("Vicenza") is Girone.SB and params.is_current("Vicenza") and "Vicenza" not in params.historical_teams
    for g, members in groups.items():
        assert sum(params.alpha[t] for t in members) == pytest.approx(0.0, abs=1e-6), g
    assert np.isnan(params.to_frame().set_index("team").loc["Vicenza", "gamma"])
    engine = DixonColesEngine(DixonColesConfig(xi=0.0))
    engine.fit(serie_c_only, registry=registry)
    with pytest.raises(KeyError, match="no home-advantage estimate for girone SB"):
        engine.predict(Fixture("SB-FX", date(2026, 11, 8), 12, Girone.SB, "Vicenza", "Arezzo"))
    # without the override the index is invalid: a group without matches has nothing to constrain it
    with pytest.raises(ValueError, match="has teams but no matches"):
        TeamIndex(teams=("x", "y"), teams_by_girone={Girone.A: ("x",), Girone.SB: ("y",)}, gironi=(Girone.A,))
    ok = TeamIndex(teams=("x", "y"), teams_by_girone={Girone.A: ("x",), Girone.SB: ("y",)}, gironi=(Girone.A,), constraint_girone={"y": "A"})
    assert ok.constraint_groups() == {Girone.A: ("x", "y")} and ok.constraint_girone_of("y") is Girone.A and ok.girone_of("y") is Girone.SB
    with pytest.raises(ValueError, match="girone with matches"):
        TeamIndex(teams=("x", "y"), teams_by_girone={Girone.A: ("x",), Girone.SB: ("y",)}, gironi=(Girone.A,), constraint_girone={"y": "C"})


def test_serie_b_moves_are_skipped_for_a_registry_without_serie_b() -> None:
    reg = TeamRegistry.from_teams("2026/27", {"A": ("Vicenza", "Lecco", "Novara", "Trento"), "B": ("Pescara", "Spezia", "Perugia", "Reggiana")})
    history = simulate_history(seed=3, registry=reg, seasons=("2025/26",), swap_per_girone=0)
    comp = history.season_registries["2025/26"]
    assert comp.girone_of("Pescara") is Girone.B and comp.girone_of("Vicenza") is Girone.A  # no Serie B to move to
    assert set(comp.canonical_names) == set(reg.canonical_names) and sorted(set(history.matches["girone"])) == ["A", "B"]


# ----------------------------------------------------------------------------- dashboard and console
def test_dashboard_tables_show_the_competition_label_and_filter_by_serie_b(sim_state) -> None:
    state = sim_state
    assert state.registry is not None and state.registry.n_teams == 80 and state.gironi == (Girone.A, Girone.B, Girone.C, Girone.SB)
    assert state.perimeter_label == "Serie C 2026/27 Gironi A/B/C + Serie B (80 squadre)"
    assert state.perimeter_text == "Perimetro: Serie C 2026/27 Gironi A/B/C + Serie B (80 squadre) · 600 partite stagione corrente · 0 partite storiche"
    assert "Competizione" in RANKING_COLUMNS and "Competizione" in VALUE_COLUMNS and "Girone" not in RANKING_COLUMNS
    full = ranking_table(state, None)
    assert len(full) == 80 and full["Competizione"].value_counts().to_dict() == {"Girone A": 20, "Girone B": 20, "Girone C": 20, "Serie B": 20}
    sb = ranking_table(state, Girone.SB)
    assert len(sb) == 20 and set(sb["Competizione"]) == {"Serie B"} and set(sb["Squadra"]) == set(state.registry.teams(Girone.SB))
    pd.testing.assert_frame_equal(ranking_table(state, "Serie B"), sb)
    assert len(fixtures_for(state, Girone.SB)) == 10 and all(p.fixture.girone is Girone.SB for p in fixtures_for(state, Girone.SB))
    value_sb = value_table(state, -1.0, Girone.SB)
    assert len(value_sb) == sum(1 for o in state.opportunities if o.girone is Girone.SB) > 0 and set(value_sb["Competizione"]) == {"Serie B"}
    assert set(value_table(state, -1.0, "Girone B")["Competizione"]) == {"Girone B"}
    assert set(traps_table(state)["Competizione"]) <= {g.short for g in Girone}
    assert any("Serie B" in n or "80 squadre" in n for n in state.notes)


def test_main_full_mode_prints_the_80_team_perimeter_and_a_serie_b_recovery_row(capsys: pytest.CaptureFixture[str]) -> None:
    out = pipeline.main(["--mode", "full", "--no-export", "--log-level", "ERROR", "--seed", "3", "--history", "0", "--as-of", "2026-11-06"])
    console = capsys.readouterr().out
    assert "Perimetro: Serie C 2026/27 Gironi A/B/C + Serie B (80 squadre) · stagione corrente: 440 partite · storico: nessuna partita di stagioni precedenti" in console
    assert out.registry is not None and out.registry.n_teams == 80 and len(out.current_matches) == 440 and len(out.predictions) == 40
    assert out.recovery is not None and out.recovery["competizione"].tolist() == ["Girone A", "Girone B", "Girone C", "Serie B"]
    assert set(out.recovery["n_teams"]) == {20} and out.recovery.loc[out.recovery["girone"] == "SB", "gamma_true"].item() == 0.25
    assert "--- Serie B (20 squadre) ---" in console and "Serie B [SB]: gamma (fattore campo)" in console
    assert pipeline.perimeter_banner(out.registry, out.current_matches, out.history_matches).startswith("Perimetro: Serie C 2026/27 Gironi A/B/C + Serie B (80 squadre)")
    # a registry without Serie B describes itself without it
    mini = TeamRegistry.from_teams("2026/27", {"A": ("x", "y"), "B": ("z", "w"), "C": ("k", "j")})
    assert pipeline.perimeter_banner(mini, out.current_matches.iloc[0:0], out.history_matches.iloc[0:0]) == (
        "Perimetro: Serie C 2026/27 Gironi A/B/C (6 squadre) · stagione corrente: 0 partite · storico: nessuna partita di stagioni precedenti"
    )
