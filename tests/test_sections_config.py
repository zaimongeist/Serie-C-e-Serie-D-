"""Sections, fit groups and disabled competitions (``config/competitions.toml`` -> ``src/common/competitions.py``)
and the per-group Dixon-Coles fit (``src/engine/fitgroups.py``).

* every ``[[competition]]`` carries ``enabled`` (default true) and ``section`` (default: the family); the shipped
  file splits the old "Serie B e Serie C" grouping into the sections ``Serie C``, ``Serie B`` and ``Serie D`` and
  carries Serie A plus six European top flights disabled, ready to be switched on;
* disabled competitions are configured but invisible: not in ``COMPETITIONS``, not members of ``Girone``, not
  harvested by default, skipped when a registry file still lists them;
* the fit group of a competition is its country: one Dixon-Coles model per country, so enabling the Premier League
  adds a second, independent fit while Serie A would join the Italian one;
* :func:`fit_by_group` with one group is the single engine (same parameters, predictions, rankings, residuals),
  with two groups each group is fitted on its own matches only.

All fits run on small synthetic leagues (six teams per competition) so the file takes a few seconds.
"""
from __future__ import annotations

import json
import logging
import pickle
import re
from dataclasses import replace
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.common import competitions as competitions_module
from src.common.competitions import (
    ALL_COMPETITIONS,
    COMPETITIONS,
    COMPETITIONS_PATH,
    DEFAULT_COMPETITIONS,
    Competition,
    competition,
    competitions_by_section,
    competitions_of_family,
    competitions_of_section,
    describe_competitions,
    disabled_competitions,
    enabled_competitions,
    families,
    find_competition,
    fit_group_of,
    fit_groups,
    load_competitions,
    parse_competitions,
    section_label,
    section_of,
    sections,
)
from src.common.paths import ROOT
from src.common.schemas import Fixture, Girone, build_girone_enum
from src.common.teams import TeamRegistry, load_registry
from src.engine.fitgroups import (
    DEFAULT_FIT_METHOD,
    DIAGNOSTICS_COLUMNS,
    RESIDUAL_COLUMNS,
    GroupedFit,
    fit_by_group,
    group_of_codes,
    split_by_group,
)
from src.engine.model import DixonColesConfig, DixonColesEngine, NotFittedError
from src.ingestion import footystats as fs
from src.simulation.generator import simulate_league

ENABLED_CODES = ("A", "B", "C", "SB", "DA", "DB", "DC", "DD", "DE", "DF", "DG", "DH", "DI")
DISABLED = {
    # code: (label = short = family = section, country, FootyStats name, season ids 2024/25, 2025/26, 2026/27)
    "SA": ("Serie A", "Italy", "Italy Serie A", (12530, 15068, 17084)),
    "EPL": ("Premier League", "England", "England Premier League", (12325, 15050, 17146)),
    "LL": ("La Liga", "Spain", "Spain La Liga", (12316, 14956, 17199)),
    "BL": ("Bundesliga", "Germany", "Germany Bundesliga", (12529, 14968, 17210)),
    "L1": ("Ligue 1", "France", "France Ligue 1", (12337, 14932, 17102)),
    "ERE": ("Eredivisie", "Netherlands", "Netherlands Eredivisie", (12322, 14936, 17097)),
    "PL1": ("Primeira Liga", "Portugal", "Portugal Liga NOS", (12931, 15115, 17217)),
}
SEASONS = ("2024/25", "2025/26", "2026/27")
#: Prior-regularised config of the synthetic fits (six-team competitions are small).
CONFIG = DixonColesConfig(xi=0.0018, l2_penalty=0.02)


def _teams(*codes: str, n: int = 6) -> dict[str, tuple[str, ...]]:
    return {code: tuple(f"{code}team{i}" for i in range(n)) for code in codes}


def _toml_with(enabled_codes: set[str]) -> str:
    """The shipped competitions.toml text with ``enabled = true`` on the tables of ``enabled_codes``."""
    text = COMPETITIONS_PATH.read_text(encoding="utf-8")
    head, *blocks = re.split(r"(?m)^\[\[competition\]\]$", text)  # table headers only, not the header comments
    out = [head]
    for block in blocks:
        code = re.search(r'(?m)^code = "([^"]+)"$', block).group(1)  # type: ignore[union-attr]
        out.append(block.replace("enabled = false", "enabled = true") if code in enabled_codes else block)
    return "[[competition]]".join(out)


def _probabilities(prediction) -> np.ndarray:
    return np.array([p for market in prediction.probabilities.values() for p in market.values()])


# ============================================================================ configuration
def test_shipped_sections_split_serie_b_and_serie_c_and_follow_the_configuration_order() -> None:
    assert [c.code for c in COMPETITIONS] == list(ENABLED_CODES) and COMPETITIONS == enabled_competitions()
    assert sections() == ("Serie C", "Serie B", "Serie D") and sections() == families()  # one section per family by default
    assert competitions_by_section() == {"Serie C": COMPETITIONS[:3], "Serie B": (COMPETITIONS[3],), "Serie D": COMPETITIONS[4:]}
    assert competitions_of_section("Serie C") == competitions_of_family("Serie C") == COMPETITIONS[:3]
    assert competitions_of_section("serie_b") == (competition("SB"),) and competitions_of_section("SERIE-D") == COMPETITIONS[4:]
    assert section_label("serie c") == "Serie C" and section_label("Serie D") == "Serie D"
    for bad in ("Serie B e Serie C", "Serie A", "Premier League", ""):
        with pytest.raises(ValueError, match="Unknown section"):
            section_label(bad)
    assert section_of("SB") == "Serie B" and section_of("a") == "Serie C" and section_of(Girone.C) == "Serie C" and section_of("DI") == "Serie D"
    with pytest.raises(KeyError):
        section_of("SA")  # disabled: reachable only through ALL_COMPETITIONS
    assert section_of("SA", ALL_COMPETITIONS) == "Serie A"
    assert all(c.section == c.family for c in COMPETITIONS) and all(c.enabled for c in COMPETITIONS)
    # the enum members expose section and fit group
    assert Girone.SB.section == "Serie B" and Girone.A.section == "Serie C" and Girone.DA.section == "Serie D"
    assert {m.fit_group for m in Girone} == {"Italy"} and Girone.DE.fit_group == "Italy"
    # descriptions per section
    assert describe_competitions(section="Serie C", season="2026/27") == "Serie C 2026/27 Gironi A/B/C"
    assert describe_competitions(section="serie_b") == "Serie B" and describe_competitions(section="Serie D") == "Serie D Gironi A-I"
    with pytest.raises(ValueError, match="Unknown section"):
        describe_competitions(section="Serie A")
    # with the disabled tables the sections are in file order: Serie A is listed first, the foreign leagues last
    assert sections(ALL_COMPETITIONS) == ("Serie A", "Serie C", "Serie B", "Serie D", *[v[0] for k, v in DISABLED.items() if k != "SA"])


def test_fit_groups_are_the_countries() -> None:
    assert fit_groups() == {"Italy": ENABLED_CODES}
    assert fit_group_of("A") == fit_group_of(Girone.SB) == fit_group_of("di") == "Italy" and competition("DA").fit_group == "Italy"
    with pytest.raises(KeyError):
        fit_group_of("EPL")
    assert fit_group_of("EPL", ALL_COMPETITIONS) == "England"
    groups = fit_groups(ALL_COMPETITIONS)
    assert groups["Italy"] == ("SA", *ENABLED_CODES)  # Serie A, once enabled, is fitted with B/C/D (promotions connect them)
    assert {g: c for g, c in groups.items() if g != "Italy"} == {
        "England": ("EPL",), "Spain": ("LL",), "Germany": ("BL",), "France": ("L1",), "Netherlands": ("ERE",), "Portugal": ("PL1",)
    }
    assert group_of_codes() == {code: "Italy" for code in ENABLED_CODES}


def test_disabled_competitions_are_configured_but_invisible() -> None:
    off = disabled_competitions()
    assert [c.code for c in off] == list(DISABLED) and ALL_COMPETITIONS == DEFAULT_COMPETITIONS
    assert len(ALL_COMPETITIONS) == 20 and tuple(c for c in ALL_COMPETITIONS if c.enabled) == COMPETITIONS
    for comp in off:
        label, country, footystats_name, ids = DISABLED[comp.code]
        assert not comp.enabled and comp.label == comp.short == comp.family == comp.section == label
        assert comp.country == country == comp.fit_group and comp.footystats_name == footystats_name
        assert comp.tier == 1 and comp.sofascore_tournament_id == 0 and comp.season_ids == dict(zip(SEASONS, ids))
        assert not comp.supports_provider("sofascore") and comp.supports_provider("footystats")
        # invisible to the rest of the code base: not a Girone member, not harvested by default, no lookup by default
        assert comp.code not in Girone.__members__ and comp.code not in fs.DEFAULT_COMPETITIONS
        with pytest.raises(ValueError, match="Unknown girone"):
            Girone.parse(comp.code)
        with pytest.raises(ValueError, match="Unknown girone"):
            Girone.parse(comp.footystats_name)
        with pytest.raises(KeyError):
            competition(comp.code)
        assert competition(comp.code, ALL_COMPETITIONS) is comp
        assert find_competition(comp.footystats_name, ALL_COMPETITIONS) is comp and find_competition(label, ALL_COMPETITIONS) is comp
    assert fs.DEFAULT_COMPETITIONS == ENABLED_CODES == tuple(m.value for m in Girone)
    # the single-letter Serie C codes keep winning over the new labels
    assert find_competition("A", ALL_COMPETITIONS).code == "A" and find_competition("Serie A", ALL_COMPETITIONS).code == "SA"
    assert find_competition("Serie C Girone A", ALL_COMPETITIONS).code == "A"


def test_disabled_season_ids_agree_with_the_footystats_league_list_sample() -> None:
    """Cross-check with the live-shaped ``/league-list`` sample used by the FootyStats tests (Serie A, Premier League)."""
    payload = json.loads((ROOT / "tests" / "fixtures" / "footystats" / "league_list_sample.json").read_text(encoding="utf-8"))
    listed = {entry["name"]: {f"{s['year'] // 10000}/{s['year'] % 100:02d}": s["id"] for s in entry["season"]} for entry in payload["data"]}
    checked = 0
    for comp in ALL_COMPETITIONS:
        if comp.footystats_name in listed:
            for season, season_id in comp.season_ids.items():
                assert listed[comp.footystats_name][season] == season_id, (comp.code, season)
            checked += 1
    assert checked >= 2 and "Italy Serie A" in listed and "England Premier League" in listed


def test_enabled_and_section_keys_parse_validate_and_round_trip() -> None:
    table = {"code": "ZZ", "label": "Z League", "short": "Z", "country": "Zland", "footystats_name": "Zland Z", "sofascore_tournament_id": 0, "tier": 1, "family": "Z League"}
    (plain,) = parse_competitions({"competition": [table]})
    assert plain.enabled and plain.section == "Z League" and plain.fit_group == "Zland"
    (custom,) = parse_competitions({"competition": [{**table, "section": " Estero ", "enabled": False}]})
    assert not custom.enabled and custom.section == "Estero"
    assert parse_competitions({"competition": [custom.to_dict()]}) == (custom,) and custom.to_dict()["enabled"] is False
    with pytest.raises(ValueError, match="section must not be blank"):
        parse_competitions({"competition": [{**table, "section": "  "}]})
    with pytest.raises(ValueError, match="enabled must be true or false"):
        parse_competitions({"competition": [{**table, "enabled": "yes"}]})
    with pytest.raises(ValueError, match="section must be a string"):
        parse_competitions({"competition": [{**table, "section": 3}]})
    with pytest.raises(ValueError, match="unknown keys"):
        parse_competitions({"competition": [{**table, "enable": True}]})
    # a disabled table still reserves its spellings: enabling it later can never make one ambiguous
    other = {**table, "code": "YY", "short": "Y", "footystats_name": "Yland Y", "enabled": False}
    with pytest.raises(ValueError, match="belongs to both"):
        parse_competitions({"competition": [table, other]})  # same label 'Z League'
    # the constructor defaults: a blank section means the family
    assert Competition("ZZ", "L", "S", "X", "FS", 0, 1, "Fam", section="  ").section == "Fam"
    assert Competition("ZZ", "L", "S", "X", "FS", 0, 1, "Fam").enabled is True
    # several families can share one section, in first-appearance order
    pair = parse_competitions(
        {"competition": [{**table, "section": "Estero"}, {**table, "code": "YY", "label": "Y League", "short": "Y", "footystats_name": "Yland Y", "family": "Y League", "country": "Yland", "section": "Estero"}]}
    )
    assert sections(pair) == ("Estero",) and families(pair) == ("Z League", "Y League")
    assert [c.code for c in competitions_of_section("estero", pair)] == ["ZZ", "YY"] and fit_groups(pair) == {"Zland": ("ZZ",), "Yland": ("YY",)}
    assert describe_competitions(pair, section="Estero") == "Z League + Y League"


def test_enabling_a_league_in_a_temp_toml_adds_a_section_and_a_second_fit_group(tmp_path: Path) -> None:
    path = tmp_path / "competitions.toml"
    path.write_text(_toml_with({"EPL"}), encoding="utf-8")
    comps = load_competitions(path)
    on = enabled_competitions(comps)
    assert [c.code for c in on] == [*ENABLED_CODES, "EPL"] and [c.code for c in disabled_competitions(comps)] == ["SA", "LL", "BL", "L1", "ERE", "PL1"]
    assert sections(on) == ("Serie C", "Serie B", "Serie D", "Premier League")
    assert fit_groups(on) == {"Italy": ENABLED_CODES, "England": ("EPL",)}
    enum = build_girone_enum(on, name="GironeEPL", module=__name__)
    assert enum.parse("Premier League") is enum.EPL is enum.parse("England Premier League") and enum.EPL.section == "Premier League"
    assert enum.EPL.fit_group == "England" and enum.parse("A") is enum.A
    # Serie A joins the Italian group and becomes the first section
    path.write_text(_toml_with({"SA", "EPL"}), encoding="utf-8")
    on = enabled_competitions(load_competitions(path))
    assert sections(on)[0] == "Serie A" and fit_groups(on) == {"Italy": ("SA", *ENABLED_CODES), "England": ("EPL",)}
    # the loader of this installation is untouched
    assert competitions_module._load_or_default(path) == load_competitions(path)
    assert [m.value for m in Girone] == list(ENABLED_CODES)


def test_a_file_that_enables_nothing_falls_back_to_the_built_in_definitions(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    text = COMPETITIONS_PATH.read_text(encoding="utf-8").replace("enabled = true", "enabled = false")
    path = tmp_path / "competitions.toml"
    path.write_text(text, encoding="utf-8")
    assert enabled_competitions(load_competitions(path)) == ()
    with caplog.at_level(logging.ERROR, logger="src.common.competitions"):
        assert competitions_module._load_or_default(path) == DEFAULT_COMPETITIONS
    assert any(r.getMessage() == "competitions file enables no competition, using the built-in definitions" for r in caplog.records)


def test_registry_keys_of_disabled_competitions_are_skipped(caplog: pytest.LogCaptureFixture) -> None:
    """Disabling a league whose teams were already synced must not break the dictionary: its key is skipped."""
    with caplog.at_level(logging.INFO, logger="src.common.teams"):
        registry = TeamRegistry.from_mapping(
            "2026/27", {"A": {"Alpha": ["Alpha FC"]}, "EPL": {"Arsenal": ["Arsenal FC"]}, "Serie A": {"Inter": []}}
        )
    assert registry.canonical_names == ("Alpha",) and registry.exact("Arsenal") is None and registry.exact("Alpha FC") == "Alpha"
    record = next(r for r in caplog.records if r.getMessage() == "registry competitions skipped: disabled in config/competitions.toml")
    assert record.teams_by_competition == {"EPL": 1, "SA": 1}
    with pytest.raises(ValueError, match="Unknown girone"):
        TeamRegistry.from_mapping("2026/27", {"ZZ": {"Alpha": []}})  # an unknown key still raises
    assert load_registry().n_teams == 242


# ============================================================================ fit groups
@pytest.fixture(scope="module")
def italy_league():
    """Four Italian competitions (Serie C A/B, Serie B, Serie D A), six teams each, eight of ten rounds played (round 9 is the fixture list)."""
    registry = TeamRegistry.from_teams("2026/27", _teams("A", "B", "SB", "DA"))
    return simulate_league(seed=21, registry=registry, n_rounds=8, with_odds=False)


@pytest.fixture(scope="module")
def two_countries():
    """Serie D re-labelled as another country ('Testland'): two fit groups, Italy (A, B) and Testland (DA, DB)."""
    comps = tuple(replace(c, country="Testland") if c.family == "Serie D" else c for c in COMPETITIONS)
    registry = TeamRegistry.from_teams("2026/27", _teams("A", "B", "DA", "DB"))
    league = simulate_league(seed=33, registry=registry, n_rounds=8, with_odds=False)
    assert {fx.girone.value for fx in league.fixtures} == {"A", "B", "DA", "DB"}
    return comps, league


def test_grouped_fit_with_one_group_is_the_single_engine(italy_league) -> None:
    league = italy_league
    registry = league.registry
    engine = DixonColesEngine(replace(CONFIG, method="L-BFGS-B"))
    params = engine.fit(league.matches, registry=registry)
    grouped = fit_by_group(league.matches, registry=registry, config=CONFIG, fixtures=league.fixtures)
    assert isinstance(grouped, GroupedFit) and grouped.groups == ("Italy",) and grouped.missing_groups == () and grouped.notes == ()
    assert DEFAULT_FIT_METHOD == "L-BFGS-B" and grouped.config == replace(CONFIG, method="L-BFGS-B") and grouped.is_fitted
    assert grouped.params_ is grouped.engines["Italy"].params_ is grouped.params_by_group["Italy"] is grouped.params_for("SB")
    got = grouped.params_
    assert got.converged and got.method == params.method == "L-BFGS-B" and got.teams == params.teams and got.n_matches == params.n_matches
    np.testing.assert_allclose([got.alpha[t] for t in params.teams], [params.alpha[t] for t in params.teams], atol=1e-10)
    np.testing.assert_allclose([got.beta[t] for t in params.teams], [params.beta[t] for t in params.teams], atol=1e-10)
    assert got.gamma.keys() == params.gamma.keys() and got.rho == pytest.approx(params.rho, abs=1e-10)
    # predictions, rates, matrices: same numbers
    assert league.fixtures
    for fx in league.fixtures:
        np.testing.assert_allclose(_probabilities(grouped.predict(fx)), _probabilities(engine.predict(fx)), atol=1e-10)
        assert grouped.rates(fx.home_team, fx.away_team, fx.girone) == pytest.approx(engine.rates(fx.home_team, fx.away_team, fx.girone))
        np.testing.assert_allclose(grouped.score_matrix(fx.home_team, fx.away_team, fx.girone), engine.score_matrix(fx.home_team, fx.away_team, fx.girone), atol=1e-12)
    assert [p.fixture for p in grouped.predict_fixtures(league.fixtures)] == list(league.fixtures)
    fx = league.fixtures[0]
    assert grouped.fair_odds(fx.home_team, fx.away_team, fx.girone).keys() == engine.fair_odds(fx.home_team, fx.away_team, fx.girone).keys()
    # rankings and residuals
    pd.testing.assert_frame_equal(grouped.ranking(), engine.ranking(), atol=1e-9)
    pd.testing.assert_frame_equal(grouped.ranking("Serie B"), engine.ranking("SB"), atol=1e-9)
    residuals = grouped.residuals(league.matches)
    assert list(residuals.columns) == [*RESIDUAL_COLUMNS, "fit_group", "rho"] and list(engine.residuals(league.matches).columns) == list(RESIDUAL_COLUMNS)
    pd.testing.assert_frame_equal(residuals[list(RESIDUAL_COLUMNS)], engine.residuals(league.matches), atol=1e-9)
    assert set(residuals["fit_group"]) == {"Italy"} and np.allclose(residuals["rho"], got.rho)
    # diagnostics and Italian notes
    diag = grouped.diagnostics()
    assert list(diag.columns) == list(DIAGNOSTICS_COLUMNS) and len(diag) == 1
    row = diag.iloc[0]
    assert row["group"] == "Italy" and row["competitions"] == "A, B, SB, DA" and row["n_matches"] == got.n_matches and bool(row["converged"])
    assert grouped.fitted_codes == {"Italy": ("A", "B", "SB", "DA")} and grouped.group_codes["Italy"] == ENABLED_CODES
    (line,) = grouped.describe()
    assert line.startswith("Gruppo di stima Italy (A, B, SB, DA): ") and "metodo L-BFGS-B" in line and "convergenza sì" in line
    # picklable (Streamlit caches pickle the pipeline state)
    restored = pickle.loads(pickle.dumps(grouped))
    assert restored.groups == ("Italy",) and restored.predict(fx).probabilities == grouped.predict(fx).probabilities
    # a competition of the group without matches answers like the single engine
    with pytest.raises(KeyError, match="no home-advantage estimate"):
        grouped.rates("Ateam0", "Ateam1", "C")


def test_method_override_and_warm_start(italy_league) -> None:
    league = italy_league
    fast = fit_by_group(league.matches, registry=league.registry, config=replace(CONFIG, method="SLSQP"))
    assert fast.config.method == "L-BFGS-B" and fast.params_.method == "L-BFGS-B"  # the default method overrides the config's
    slow = fit_by_group(league.matches, registry=league.registry, config=replace(CONFIG, method="SLSQP"), method=None)
    assert slow.config.method == "SLSQP" and slow.params_.method == "SLSQP"
    # same optimum within tolerance
    for fx in league.fixtures:
        np.testing.assert_allclose(_probabilities(fast.predict(fx)), _probabilities(slow.predict(fx)), atol=2e-3)
    warm = fit_by_group(league.matches, registry=league.registry, config=CONFIG, warm_start=fast)
    warm_map = fit_by_group(league.matches, registry=league.registry, config=CONFIG, warm_start={"Italy": fast.params_})
    for other in (warm, warm_map):
        assert other.params_.converged
        np.testing.assert_allclose(_probabilities(other.predict(league.fixtures[0])), _probabilities(fast.predict(league.fixtures[0])), atol=1e-4)
    with pytest.raises(ValueError, match="method must be one of"):
        fit_by_group(league.matches, config=CONFIG, method="Nelder-Mead")  # type: ignore[arg-type]


def test_grouped_fit_with_two_groups_fits_each_country_on_its_own_matches(two_countries) -> None:
    comps, league = two_countries
    matches, registry = league.matches, league.registry
    grouped = fit_by_group(matches, registry=registry, config=CONFIG, fixtures=league.fixtures, competitions=comps)
    assert grouped.groups == ("Italy", "Testland") and grouped.missing_groups == ()
    assert grouped.group_of("DA") == "Testland" and grouped.group_of(Girone.B) == "Italy"
    assert grouped.fitted_codes == {"Italy": ("A", "B"), "Testland": ("DA", "DB")}
    split = split_by_group(matches, comps)
    assert list(split) == ["Italy", "Testland"] and sum(len(f) for f in split.values()) == len(matches)
    assert set(split["Italy"]["girone"]) == {"A", "B"} and set(split["Testland"]["girone"]) == {"DA", "DB"}

    # each group equals a single engine fitted on that group's matches only
    engines = {}
    for group, frame in split.items():
        engines[group] = DixonColesEngine(replace(CONFIG, method="L-BFGS-B"))
        engines[group].fit(frame, registry=registry)
        own, ref = grouped.params_by_group[group], engines[group].params_
        assert own.teams == ref.teams and set(own.gamma) == set(ref.gamma)
        np.testing.assert_allclose([own.alpha[t] for t in ref.teams], [ref.alpha[t] for t in ref.teams], atol=1e-10)
        assert own.rho == pytest.approx(ref.rho, abs=1e-10)
    assert set(grouped.params_by_group["Italy"].alpha).isdisjoint(grouped.params_by_group["Testland"].alpha)
    for fx in league.fixtures:
        expected = engines[grouped.group_of(fx.girone)].predict(fx)
        np.testing.assert_allclose(_probabilities(grouped.predict(fx)), _probabilities(expected), atol=1e-10)

    # independence: changing Testland's results leaves the Italian fit exactly as it was
    changed = matches.copy()
    is_d = changed["girone"].isin(["DA", "DB"])
    changed.loc[is_d, "home_goals"] = changed.loc[is_d, "home_goals"] + 2
    regrouped = fit_by_group(changed, registry=registry, config=CONFIG, competitions=comps)
    italy_a, italy_b = grouped.params_by_group["Italy"], regrouped.params_by_group["Italy"]
    assert italy_a.alpha == italy_b.alpha and italy_a.beta == italy_b.beta and italy_a.rho == italy_b.rho
    assert regrouped.params_by_group["Testland"].gamma[Girone.DA] > grouped.params_by_group["Testland"].gamma[Girone.DA]

    # the combined summary: union of the parameters, each group's own home advantages, weighted rho
    combined = grouped.params_
    by_group = grouped.params_by_group
    assert combined is grouped.params_  # cached
    assert set(combined.alpha) == set(by_group["Italy"].alpha) | set(by_group["Testland"].alpha)
    assert set(combined.gamma) == {Girone.A, Girone.B, Girone.DA, Girone.DB}
    assert combined.gamma[Girone.DA] == by_group["Testland"].gamma[Girone.DA] and combined.gamma[Girone.A] == by_group["Italy"].gamma[Girone.A]
    assert combined.n_matches == len(matches) and combined.converged and combined.method == "L-BFGS-B"
    rhos = [by_group[g].rho for g in grouped.groups]
    assert min(rhos) - 1e-12 <= combined.rho <= max(rhos) + 1e-12
    assert combined.is_current("DAteam0") and combined.is_current("Ateam0") and combined.girone_of("DBteam3") is Girone.DB
    assert combined.message.startswith("Italy: ") and "; Testland: " in combined.message

    # rankings: one competition from its group, all competitions merged and sorted by alpha
    pd.testing.assert_frame_equal(grouped.ranking("DA"), engines["Testland"].ranking("DA"))
    merged = grouped.ranking()
    assert len(merged) == 24 and merged["alpha"].is_monotonic_decreasing and set(merged["girone"]) == {"A", "B", "DA", "DB"}

    # residuals: every row from its own group's engine, in validated order
    residuals = grouped.residuals(matches)
    assert list(residuals.columns) == [*RESIDUAL_COLUMNS, "fit_group", "rho"] and len(residuals) == len(matches)
    for group, engine in engines.items():
        mine = residuals.loc[residuals["fit_group"] == group].reset_index(drop=True)
        pd.testing.assert_frame_equal(mine[list(RESIDUAL_COLUMNS)], engine.residuals(split[group]), atol=1e-9)
        assert np.allclose(mine["rho"], engine.params_.rho)
    order = residuals[["date", "girone", "match_id"]].astype({"girone": str, "match_id": str})
    assert order.equals(order.sort_values(["date", "girone", "match_id"], kind="stable"))

    diag = grouped.diagnostics()
    assert diag["group"].tolist() == ["Italy", "Testland"] and diag["competitions"].tolist() == ["A, B", "DA, DB"]
    assert len(grouped.describe()) == 2 and grouped.describe()[1].startswith("Gruppo di stima Testland (DA, DB)")


def test_groups_without_matches_and_name_collisions_are_reported_not_fatal(two_countries, caplog: pytest.LogCaptureFixture) -> None:
    comps, league = two_countries
    # a third country (Serie D girone C re-labelled) with registry teams and a fixture but no played match
    comps3 = tuple(replace(c, country="Elsewhere") if c.code == "DC" else c for c in comps)
    teams = {**{g.value: league.registry.teams(g) for g in league.registry.gironi}, **_teams("DC")}
    registry = TeamRegistry.from_teams("2026/27", teams)
    pending = Fixture("DC-1", date(2026, 11, 15), 1, "DC", "DCteam0", "DCteam1")
    with caplog.at_level(logging.WARNING, logger="src.engine.fitgroups"):
        grouped = fit_by_group(league.matches, registry=registry, config=CONFIG, fixtures=[*league.fixtures, pending], competitions=comps3)
    assert grouped.groups == ("Italy", "Testland") and grouped.missing_groups == ("Elsewhere",)
    assert any(r.getMessage().startswith("fit groups without matches") for r in caplog.records)
    assert any("Gruppo di stima Elsewhere (DC)" in note and "nessuna partita giocata" in note for note in grouped.notes)
    assert grouped.notes[0] in grouped.describe()
    with pytest.raises(KeyError, match="no fit for group 'Elsewhere'"):
        grouped.predict(pending)
    residuals = grouped.residuals(pd.concat([league.matches, pd.DataFrame([{
        "match_id": "DC-played", "date": pd.Timestamp("2026-11-01"), "round": 1, "girone": "DC", "home_team": "DCteam0",
        "away_team": "DCteam1", "home_goals": 2, "away_goals": 1,
    }])], ignore_index=True))
    extra = residuals.loc[residuals["match_id"] == "DC-played"].iloc[0]
    assert extra["fit_group"] == "Elsewhere" and np.isnan(extra["lam"]) and np.isnan(extra["rho"]) and extra["actual_total"] == 3

    # the same club name in two countries: each group predicts its own, the merged summary refuses to guess
    clash = league.matches.copy()
    for column in ("home_team", "away_team"):
        clash[column] = clash[column].replace({"DAteam0": "Ateam0"})
    caplog.clear()
    with caplog.at_level(logging.WARNING, logger="src.engine.fitgroups"):
        collided = fit_by_group(clash, config=CONFIG, competitions=comps)
    assert collided.team_collisions == {"Ateam0": ("Italy", "Testland")}
    assert any(r.getMessage() == "team names fitted in more than one group" for r in caplog.records)
    assert any("stimati in più gruppi" in note for note in collided.notes)
    with pytest.raises(ValueError, match="more than one group"):
        _ = collided.params_
    assert collided.params_for("DA").alpha["Ateam0"] != collided.params_for("A").alpha["Ateam0"]
    collided.predict(Fixture("X", date(2026, 11, 15), 11, "DA", "Ateam0", "DAteam1"))  # Testland's A-T0


def test_fit_by_group_input_errors(two_countries, italy_league) -> None:
    comps, league = two_countries
    with pytest.raises(ValueError, match="no valid matches"):
        fit_by_group(league.matches.iloc[0:0], config=CONFIG)
    with pytest.raises(ValueError, match="without a fit group"):
        split_by_group(league.matches, comps[:3])  # A, B, C only: DA/DB rows have no group
    fitted = fit_by_group(italy_league.matches, config=CONFIG)
    with pytest.raises(ValueError, match="at least one fitted group"):
        GroupedFit(CONFIG, {}, group_codes={"Italy": ENABLED_CODES})
    with pytest.raises(NotFittedError):
        GroupedFit(CONFIG, {"Italy": DixonColesEngine(CONFIG)}, group_codes={"Italy": ENABLED_CODES})
    with pytest.raises(ValueError, match="belongs to both fit groups"):
        GroupedFit(CONFIG, fitted.engines, group_codes={"Italy": ("A", "B"), "Other": ("B",)})
    with pytest.raises(ValueError, match="no entry for the fitted groups"):
        GroupedFit(CONFIG, fitted.engines, group_codes={"Other": ("A",)})
    rebuilt = GroupedFit(fitted.config, fitted.engines, group_codes=fitted.group_codes)
    assert rebuilt.fitted_codes == fitted.fitted_codes == {"Italy": ("A", "B", "SB", "DA")}
