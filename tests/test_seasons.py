"""Seasons, perimeter and history: the 2026/27 registry (Serie C Gironi A/B/C + Serie B, 80 teams) is the
only set of teams that is ranked, predicted and priced; older seasons are history (and historical teams
opponents only). A club keeps one canonical name across divisions: the registry decides its competition.

Covers ``season_of``/``season_bounds``, the registry defaults and historical aliases, the
calendar-anchored simulator with synthetic past seasons, the registry-aware engine
(girone from the registry, historical teams in the likelihood but out of the ranking),
the console pipeline banner and the dashboard data layer.
"""
from __future__ import annotations

import dataclasses
import logging
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import main as pipeline
from src.common.schemas import Fixture, Girone, MatchRecord, matches_to_frame, season_bounds, season_of
from src.common.teams import DEFAULT_SEASON, TeamRegistry, list_seasons, load_registry
from src.common.teams import normalize_name
from src.dashboard.state import (
    MAX_HISTORY_SEASONS,
    build_state,
    fixtures_for,
    matches_per_team,
    other_season_reason,
    perimeter_matches,
    prior_recommended,
    ranking_table,
    simulate_current_season,
    split_by_season,
    traps_table,
    value_table,
)
from src.engine.model import DixonColesConfig, DixonColesEngine, FittedParams, build_team_index
from src.ingestion.harvester import save_fixtures, save_processed
from src.simulation.generator import (
    HISTORICAL_TEAMS,
    HISTORY_SEASONS,
    SEASON_2026_27_START,
    SYNTHETIC_HISTORICAL_PREFIX,
    history_seasons_for,
    previous_seasons,
    season_start_date,
    simulate_history,
    simulate_league,
    simulate_mini_dataset,
)

TODAY = date(2026, 11, 6)
#: Current teams that played elsewhere in 2025/26: girone moves inside Serie C and division moves (Pescara / Spezia from
#: Serie B to girone B, Vicenza / Arezzo from Serie C to Serie B). The 2026/27 registry decides their competition.
MOVED = {
    "Inter U23": Girone.C,
    "Atalanta U23": Girone.B,
    "Juventus Next Gen": Girone.A,
    "Latina": Girone.B,
    "Carpi": Girone.A,
    "Pescara": Girone.B,
    "Spezia": Girone.B,
    "Vicenza": Girone.SB,
    "Arezzo": Girone.SB,
}


# ----------------------------------------------------------------------------- fixtures
@pytest.fixture(scope="module")
def registry() -> TeamRegistry:
    return load_registry()


@pytest.fixture(scope="module")
def history():
    return simulate_history(seed=7)


@pytest.fixture(scope="module")
def current():
    return simulate_league(seed=7, as_of=TODAY)


@pytest.fixture(scope="module")
def all_matches(history, current) -> pd.DataFrame:
    return pd.concat([history.matches, current.matches], ignore_index=True)


@pytest.fixture(scope="module")
def fit(all_matches, registry):
    engine = DixonColesEngine(DixonColesConfig(xi=0.0))
    params = engine.fit(all_matches, registry=registry)
    return engine, params


# ----------------------------------------------------------------------------- season_of
def test_season_of_boundaries_and_inputs() -> None:
    assert season_of(date(2026, 6, 30)) == "2025/26"
    assert season_of(date(2026, 7, 1)) == "2026/27"
    assert season_of(pd.Timestamp("2026-11-06")) == "2026/27"
    assert season_of("2024-08-25") == "2024/25"
    assert season_of(date(1999, 12, 1)) == "1999/00"
    assert season_of(SEASON_2026_27_START) == "2026/27"
    with pytest.raises(ValueError):
        season_of(pd.NaT)
    with pytest.raises(TypeError):
        season_of(2026)  # type: ignore[arg-type]


def test_season_bounds_roundtrip_and_validation() -> None:
    assert season_bounds("2026/27") == (date(2026, 7, 1), date(2027, 6, 30))
    assert season_bounds("2026_27") == season_bounds("2026-27") == season_bounds("2026/2027") == season_bounds("2026/27")
    assert season_bounds("1999/00") == (date(1999, 7, 1), date(2000, 6, 30))
    first, last = season_bounds("2025/26")
    assert season_of(first) == season_of(last) == "2025/26"
    for bad in ("2026/28", "2026", "26/27", "2026/2028", ""):
        with pytest.raises(ValueError):
            season_bounds(bad)


# ----------------------------------------------------------------------------- registry
def test_default_registry_is_2026_27_with_80_teams(registry: TeamRegistry) -> None:
    assert DEFAULT_SEASON == "2026_27"
    assert registry.season_label == "2026/27" and registry.n_teams == 80
    assert all(len(registry.teams(g)) == 20 for g in registry.gironi) and registry.gironi == (Girone.A, Girone.B, Girone.C, Girone.SB)
    # the Serie D gironi are configured but ship empty (filled by the FootyStats auto-init): not part of the perimeter yet
    assert registry.missing_competitions() == tuple(g for g in Girone if g.family == "Serie D") and registry.teams("DA") == ()
    assert registry.perimeter_label == "Serie C 2026/27 Gironi A/B/C + Serie B (80 squadre)"
    assert list_seasons() == ["2025_26", "2026_27"]
    assert load_registry("2026/27").canonical_names == registry.canonical_names  # label with a slash works too
    for team, girone in MOVED.items():
        assert registry.girone_of(team) is girone and registry.is_current_team(team)
    assert not registry.is_current_team("Ternana") and not registry.is_current_team("Nowhere FC")
    assert registry.historical_names == ()
    with pytest.raises(FileNotFoundError):
        load_registry("2019_20")


def test_with_historical_aliases_keeps_current_teams_and_maps_old_spellings(registry: TeamRegistry) -> None:
    old = load_registry("2025_26")
    merged = registry.with_historical_aliases(old)
    assert merged.canonical_names == registry.canonical_names and merged.season == registry.season
    # a team that still exists: old aliases point to the current canonical name
    assert merged.exact("Juventus U23") == "Juventus Next Gen" and merged.is_current_team("Juventus U23")
    # a team that is gone: its old canonical spelling and aliases map to the historical spelling itself
    assert merged.exact("Ternana Calcio") == "Ternana" and merged.exact("Ternana") == "Ternana"
    assert not merged.is_current_team("Ternana Calcio")
    assert "Ternana" in merged.historical_names and "Frosinone" in merged.historical_names  # Frosinone: Serie B 2025/26, gone in 2026/27
    # clubs that changed division keep ONE canonical name: Vicenza (girone A -> Serie B), Pescara (Serie B -> girone B)
    assert merged.is_current_team("L.R. Vicenza") and merged.girone_of("Vicenza") is Girone.SB and old.girone_of("Vicenza") is Girone.A
    assert merged.exact("Delfino Pescara") == "Pescara" and merged.girone_of("Pescara") is Girone.B and old.girone_of("Pescara") is Girone.SB
    assert not set(merged.historical_names) & set(registry.canonical_names)
    # a current team that changed girone keeps the current girone
    assert merged.girone_of("Latina") is Girone.B and old.girone_of("Latina") is Girone.C
    # the original registry is untouched and the lookup table of current spellings is unchanged
    assert registry.exact("Ternana Calcio") is None
    assert load_registry(with_history=True) == merged


# ----------------------------------------------------------------------------- simulator
def test_calendar_anchored_to_the_real_season(current) -> None:
    assert SEASON_2026_27_START == date(2026, 8, 23) and SEASON_2026_27_START.weekday() == 6  # Sunday
    assert season_start_date("2026/27") == SEASON_2026_27_START
    assert season_start_date("2025/26") == date(2025, 8, 24) and season_start_date("2024/25") == date(2024, 8, 25)
    assert current.matches["date"].min().date() == SEASON_2026_27_START
    assert set(current.matches["date"].dt.weekday) == {6}
    # as of 2026-11-06: rounds dated before it are played (11), round 12 on 2026-11-08 is the fixture list
    assert int(current.matches["round"].max()) == 11 and len(current.matches) == 11 * 40  # 30 Serie C + 10 Serie B matches a round
    assert len(current.fixtures) == 40 and {fx.round for fx in current.fixtures} == {12}
    assert {fx.date for fx in current.fixtures} == {date(2026, 11, 8)}
    assert current.registry.season_label == "2026/27" and set(current.teams) == set(load_registry().canonical_names)
    # exactly on a match day the round is still to be played
    on_day = simulate_league(seed=7, as_of=date(2026, 11, 8), with_odds=False)
    assert int(on_day.matches["round"].max()) == 11 and {fx.round for fx in on_day.fixtures} == {12}
    before = simulate_league(seed=7, as_of=date(2026, 8, 1), with_odds=False)
    assert before.matches.empty and {fx.round for fx in before.fixtures} == {1}
    over = simulate_league(seed=7, as_of=date(2027, 6, 1), with_odds=False)
    assert int(over.matches["round"].max()) == 38 and over.fixtures == []
    # n_rounds keeps explicit control and wins over as_of
    explicit = simulate_league(seed=7, n_rounds=3, as_of=TODAY, with_odds=False)
    assert int(explicit.matches["round"].max()) == 3 and {fx.round for fx in explicit.fixtures} == {4}


def test_simulate_history_composition_and_determinism(history, current, registry: TeamRegistry) -> None:
    assert history.seasons == HISTORY_SEASONS == ("2024/25", "2025/26")
    assert history.fixtures == [] and history.odds == []
    assert len(history.matches) == 2 * 4 * 380  # two full double round robins of four 20-team competitions
    by_season = history.matches["date"].map(season_of)
    assert by_season.value_counts().to_dict() == {"2024/25": 1520, "2025/26": 1520}
    assert history.matches["date"].min().date() == date(2024, 8, 25)
    assert history.matches["match_id"].str.startswith("HIST").all()
    assert pd.concat([history.matches["match_id"], current.matches["match_id"]]).is_unique
    # same TrueParams drive history and current season; historical clubs have strengths too
    for team in registry.canonical_names:
        assert history.true_params.alpha[team] == current.true_params.alpha[team]
    assert set(HISTORICAL_TEAMS) <= set(history.true_params.alpha)
    historical = set(history.teams) - set(registry.canonical_names)
    assert historical and historical <= set(HISTORICAL_TEAMS)
    for label, comp in history.season_registries.items():
        assert comp.gironi == registry.gironi and all(len(comp.teams(g)) == 20 for g in registry.gironi), label
        assert len(set(comp.canonical_names) - set(registry.canonical_names)) >= 12  # 3 swaps per competition
    # real past compositions: Inter U23 in A (2025/26, absent in 2024/25), Atalanta U23 C/A, Juventus Next Gen B/C,
    # Pescara / Spezia / Reggiana / Bari in Serie B 2025/26, Vicenza / Arezzo / Ascoli / Benevento in Serie C
    s26, s25 = history.season_registries["2025/26"], history.season_registries["2024/25"]
    assert s26.girone_of("Inter U23") is Girone.A and s26.girone_of("Atalanta U23") is Girone.C
    assert s26.girone_of("Juventus Next Gen") is Girone.B and s26.girone_of("Latina") is Girone.C and s26.girone_of("Carpi") is Girone.B
    assert s26.girone_of("Pescara") is Girone.SB and s26.girone_of("Spezia") is Girone.SB and s26.girone_of("Bari") is Girone.SB
    assert s26.girone_of("Vicenza") is Girone.A and s26.girone_of("Arezzo") is Girone.B and s26.girone_of("Benevento") is Girone.C
    assert not {"Cremonese", "Pisa", "Hellas Verona"} & set(s26.canonical_names)  # Serie A in 2025/26
    assert "Inter U23" not in s25.canonical_names and not {"Empoli", "Hellas Verona"} & set(s25.canonical_names)
    assert s25.girone_of("Atalanta U23") is Girone.A and s25.girone_of("Juventus Next Gen") is Girone.C
    assert s25.girone_of("Padova") is Girone.A and s25.girone_of("Virtus Entella") is Girone.B and s25.girone_of("Cittadella") is Girone.SB
    # deterministic
    again = simulate_history(seed=7)
    pd.testing.assert_frame_equal(again.matches, history.matches)
    assert again.true_params == history.true_params
    # only the moves, no swaps: girone A loses Juventus Next Gen and Carpi but gains Inter U23 and Vicenza (balanced);
    # girone B loses five (Atalanta U23, Latina, Pescara, Reggiana, Spezia) and gains four, so one historical club
    # fills it; girone C loses Bari and Inter U23, gains three, so one current team is dropped; Serie B loses the four
    # promoted clubs and the three Serie A clubs of 2025/26, gains the four relegated ones, so three fills
    moves_only = simulate_history(seed=1, seasons=("2025/26",), swap_per_girone=0)
    comp = moves_only.season_registries["2025/26"]
    extra_hist = set(comp.canonical_names) - set(registry.canonical_names)
    assert len(extra_hist) == 4 and extra_hist <= set(HISTORICAL_TEAMS)
    assert sorted(comp.girone_of(t).value for t in extra_hist) == ["B", "SB", "SB", "SB"]
    assert comp.girone_of("Inter U23") is Girone.A and all(len(comp.teams(g)) == 20 for g in registry.gironi)
    assert set(registry.teams("A")) - set(comp.teams("A")) == {"Carpi", "Juventus Next Gen"}
    assert len(set(registry.teams("C")) - set(comp.teams("C"))) == 3  # Bari + Inter U23 (moved) + one dropped
    assert set(registry.teams("SB")) - set(comp.teams("SB")) == {"Arezzo", "Ascoli", "Benevento", "Vicenza", "Cremonese", "Pisa", "Hellas Verona"}


def test_mini_dataset_uses_current_teams() -> None:
    mini = simulate_mini_dataset(seed=42)
    reg = load_registry()
    assert len(mini.matches) == 20 and set(mini.teams) <= set(reg.canonical_names)
    assert mini.registry.season_label == "2026/27"
    assert mini.matches["date"].min().date() == SEASON_2026_27_START
    for team in mini.teams:
        assert mini.registry.girone_of(team) is reg.girone_of(team)


# ----------------------------------------------------------------------------- engine
def test_fit_with_registry_assigns_current_teams_to_registry_girone(fit, registry: TeamRegistry) -> None:
    engine, params = fit
    assert params.converged
    assert set(params.current_teams) == set(registry.canonical_names) and len(params.historical_teams) >= 12
    assert not set(params.historical_teams) & set(registry.canonical_names)
    for team, girone in MOVED.items():
        assert params.girone_of(team) is girone, team
        assert params.is_current(team)
    for team in params.historical_teams:
        assert not params.is_current(team) and team in params.alpha and team in params.beta
    assert not params.is_current("Nowhere FC")
    # the ranking lists only the perimeter, in the registry girone
    table = engine.ranking()
    assert len(table) == 80 and not set(table["team"]) & set(params.historical_teams)
    assert table.loc[table["team"] == "Inter U23", "girone"].item() == "C"
    assert table.loc[table["team"] == "Pescara", "girone"].item() == "B" and table.loc[table["team"] == "Vicenza", "girone"].item() == "SB"
    assert table.loc[table["team"] == "Atalanta U23", "girone"].item() == "B"
    assert table.loc[table["team"] == "Juventus Next Gen", "girone"].item() == "A"
    for g in registry.gironi:
        assert set(engine.ranking(g)["team"]) == set(registry.teams(g))
    everything = engine.ranking(include_historical=True)
    assert len(everything) == len(params.teams) and set(params.historical_teams) <= set(everything["team"])
    assert set(engine.ranking("A", include_historical=True)["team"]) == set(params.teams_by_girone[Girone.A])
    # sum(alpha) == 0 per constraint group (registry girone + the historical teams that played most there)
    for g, members in params.teams_by_girone.items():
        assert sum(params.alpha[t] for t in members) == pytest.approx(0.0, abs=1e-6), g
    assert params.current_teams_by_girone() == {g: tuple(sorted(registry.teams(g))) for g in registry.gironi}
    # gamma is indexed by the girone the match was played in: one per girone, as without a registry
    assert set(params.gamma) == set(registry.gironi)  # the empty Serie D gironi have no home advantage to estimate


def test_registry_fit_recovers_current_strengths(fit, current) -> None:
    _, params = fit
    for g, teams in params.current_teams_by_girone().items():
        a_hat = np.array([params.alpha[t] for t in teams])
        a_true = np.array([current.true_params.alpha[t] for t in teams])
        assert float(np.corrcoef(a_hat, a_true)[0, 1]) > 0.85, g
        assert abs(params.gamma[g] - current.true_params.gamma[g]) < 0.12, g


def test_predictions_work_for_current_fixtures_and_history_only_sharpens(fit, current, all_matches, registry) -> None:
    engine, params = fit
    preds = engine.predict_fixtures(current.fixtures)
    assert len(preds) == 40 and all(p.matrix.sum() == pytest.approx(1.0) for p in preds)
    assert sum(1 for p in preds if p.fixture.girone is Girone.SB) == 10
    assert all(params.is_current(p.fixture.home_team) and params.is_current(p.fixture.away_team) for p in preds)
    # the engine itself can still price any fitted team (the perimeter filter lives in the callers)
    historical = params.historical_teams[0]
    lam, mu = engine.rates(historical, registry.teams("A")[0], "A")
    assert lam > 0 and mu > 0
    # a fit on the current season alone has the same perimeter but a noisier estimate
    alone = DixonColesEngine(DixonColesConfig(xi=0.0)).fit(current.matches, registry=registry)
    assert alone.historical_teams == () and set(alone.current_teams) == set(registry.canonical_names)
    teams = list(registry.canonical_names)
    truth = np.array([current.true_params.alpha[t] for t in teams])
    err_with = np.sqrt(np.mean((np.array([params.alpha[t] for t in teams]) - truth) ** 2))
    err_alone = np.sqrt(np.mean((np.array([alone.alpha[t] for t in teams]) - truth) ** 2))
    assert err_with < err_alone


def test_fitted_params_roundtrip_and_legacy_payloads(fit, tmp_path: Path) -> None:
    engine, params = fit
    loaded = DixonColesEngine.load(engine.save(tmp_path / "engine.json"))
    assert loaded.params_ == params and loaded.params_.historical_teams == params.historical_teams
    assert loaded.params_.current_teams == params.current_teams
    payload = params.to_dict()
    assert payload["historical_teams"] == list(params.historical_teams) and payload["current_teams"] == list(params.current_teams)
    legacy = {k: v for k, v in payload.items() if k not in ("historical_teams", "current_teams")}
    restored = FittedParams.from_dict(legacy)
    assert restored.historical_teams == () and set(restored.current_teams) == set(params.teams)
    assert restored.is_current(params.historical_teams[0])  # every team is current without the field
    with pytest.raises(ValueError, match="historical_teams"):
        FittedParams.from_dict({**payload, "historical_teams": ["Nowhere FC"]})


def test_build_team_index_with_registry_and_without(all_matches, registry: TeamRegistry, caplog) -> None:
    from src.common.schemas import validate_matches_df

    df = validate_matches_df(all_matches)
    with caplog.at_level(logging.WARNING, logger="src.engine.model"):
        with_registry = build_team_index(df, registry=registry)
    assert set(with_registry.current_teams) == set(registry.canonical_names)
    assert with_registry.girone_of("Inter U23") is Girone.C and with_registry.is_current("Inter U23")
    assert with_registry.historical and all(not with_registry.is_current(t) for t in with_registry.historical)
    assert not any("Inter U23" in rec.getMessage() or rec.__dict__.get("team") == "Inter U23" for rec in caplog.records)
    plain = build_team_index(df)
    assert plain.historical == () and set(plain.current_teams) == set(plain.teams) == set(with_registry.teams)
    assert plain.girone_of("Inter U23") is Girone.A  # majority of its matches (2025/26 in A) without a registry
    # a current team whose only matches were played in another girone still goes to its registry girone ...
    only_hist = df.loc[df["date"] < pd.Timestamp("2026-07-01")].reset_index(drop=True)
    assert build_team_index(only_hist, registry=registry).girone_of("Inter U23") is Girone.C
    # ... and when its registry girone has no match at all (girone C not harvested) that home advantage cannot be
    # estimated: the team keeps its registry girone but joins, for the constraint only, the girone it played in
    no_c = only_hist.loc[only_hist["girone"] != "C"].reset_index(drop=True)
    with caplog.at_level(logging.WARNING, logger="src.engine.model"):
        fallback = build_team_index(no_c, registry=registry)
    assert Girone.C not in fallback.gironi and fallback.girone_of("Inter U23") is Girone.C and fallback.is_current("Inter U23")
    assert fallback.constraint_girone_of("Inter U23") is Girone.A and Girone.C not in fallback.constraint_groups()
    assert any("registry gironi with current teams but no matches" in rec.getMessage() for rec in caplog.records)


# ----------------------------------------------------------------------------- coverage rule
def test_prior_rule_counts_current_season_matches_of_current_teams_only(all_matches, current, registry) -> None:
    label = registry.season_label
    cur, hist = split_by_season(all_matches, label)
    assert len(cur) == len(current.matches) == 440 and len(hist) == 3040
    assert perimeter_matches(all_matches, registry=registry, season=label).equals(cur)
    # without the perimeter arguments the legacy rule sees three seasons (43+ per team)
    assert matches_per_team(all_matches) > 40 and not prior_recommended(all_matches)
    # with them, only the 11 current rounds count: the prior is recommended
    assert matches_per_team(all_matches, registry=registry, season=label) == pytest.approx(11.0)
    assert prior_recommended(all_matches, registry=registry, season=label)
    # a girone without current-season matches is zero coverage even with a long history
    no_c = all_matches.loc[~((all_matches["girone"] == "C") & (all_matches["date"] >= pd.Timestamp("2026-07-01")))]
    assert matches_per_team(no_c, registry=registry, season=label) == 0.0
    assert matches_per_team(all_matches.iloc[0:0], registry=registry, season=label) == 0.0


# ----------------------------------------------------------------------------- console
def test_main_mini_and_full_print_the_perimeter_banner(capsys: pytest.CaptureFixture[str]) -> None:
    out = pipeline.main(["--mode", "mini", "--no-export", "--log-level", "ERROR"])
    console = capsys.readouterr().out
    assert "Perimetro: Serie C 2026/27 Gironi A/B/C (14 squadre) · stagione corrente: 20 partite · storico: 108 partite 2024/25-2025/26" in console
    assert out.season_label == "2026/27" and out.registry is not None and out.registry.season_label == "2026/27"
    assert len(out.current_matches) == 20 and len(out.history_matches) == 108 and len(out.matches) == 128 == out.params.n_matches
    assert out.params.historical_teams and not set(out.params.historical_teams) & set(out.registry.canonical_names)
    assert all(out.params.is_current(t) for p in out.predictions for t in (p.fixture.home_team, p.fixture.away_team))
    assert "squadre storiche escluse dal ranking" in console and "per stagione" in console
    for label in ("2024/25", "2025/26", "2026/27"):
        assert label in console
    assert "stima MAP con prior gaussiana (2.5 partite per squadra nel girone meno coperto della stagione 2026/27" in console

    no_history = pipeline.main(["--mode", "mini", "--no-export", "--log-level", "ERROR", "--history", "0"])
    console = capsys.readouterr().out
    assert "storico: nessuna partita di stagioni precedenti" in console
    assert len(no_history.matches) == 20 and no_history.params.historical_teams == () and no_history.history_matches.empty

    assert "Demo mini: istantanea fissa di inizio stagione 2026/27 (giornate 1-3 giocate, 20 partite; turno da prevedere: 4ª giornata del 2026-09-13)" in console

    full = pipeline.main(["--mode", "full", "--no-export", "--log-level", "ERROR", "--seed", "3", "--history", "1", "--as-of", "2026-11-06"])
    console = capsys.readouterr().out
    assert "Perimetro: Serie C 2026/27 Gironi A/B/C + Serie B (80 squadre) · stagione corrente: 440 partite · storico: 1520 partite 2025/26" in console
    assert "Calendario simulato al 2026-11-06: giornate con data precedente giocate (11), la prima giornata in data uguale o successiva (2026-11-08)" in console
    assert full.recovery is not None and set(full.recovery["n_teams"]) == {20} and len(full.recovery) == 4
    assert full.recovery["competizione"].tolist() == ["Girone A", "Girone B", "Girone C", "Serie B"]
    assert (full.recovery["corr_alpha"] > 0.75).all()  # 11 rounds + one synthetic season: a thin sample
    assert full.params.girone_of("Inter U23") is Girone.C and full.params.girone_of("Atalanta U23") is Girone.B
    assert full.params.girone_of("Pescara") is Girone.B and full.params.girone_of("Vicenza") is Girone.SB
    assert "RECUPERO DEI PARAMETRI VERI DEL SIMULATORE (squadre 2026/27)" in console
    assert {p.fixture.date for p in full.predictions} == {date(2026, 11, 8)} and len(full.predictions) == 40

    fixed = pipeline.main(["--mode", "full", "--no-export", "--log-level", "ERROR", "--seed", "3", "--history", "1", "--rounds", "30"])
    console = capsys.readouterr().out
    assert "Perimetro: Serie C 2026/27 Gironi A/B/C + Serie B (80 squadre) · stagione corrente: 1200 partite · storico: 1520 partite 2025/26" in console
    assert "30 giornate giocate (--rounds)" in console and fixed.recovery is not None and (fixed.recovery["corr_alpha"] > 0.85).all()

    with pytest.raises(SystemExit) as exc:
        pipeline.main(["--season", "2019_20", "--no-export"])
    assert exc.value.code == 2 and "teams_<stagione>.json" in capsys.readouterr().err


def test_main_real_data_skips_fixtures_outside_the_perimeter(tmp_path: Path, history, current, capsys) -> None:
    frame = pd.concat([history.matches, current.matches], ignore_index=True)
    processed, fixtures = tmp_path / "matches.csv", tmp_path / "fixtures.csv"
    save_processed(frame, processed)
    historical = sorted(set(history.teams) - set(current.registry.canonical_names))[0]
    extra = Fixture("HIST-FX", current.fixtures[0].date, 12, Girone.A, historical, current.registry.teams("A")[0])
    save_fixtures([*current.fixtures, extra], fixtures)
    out = pipeline.main(["--processed", str(processed), "--fixtures", str(fixtures), "--no-export", "--log-level", "ERROR"])
    console = capsys.readouterr().out
    assert len(out.matches) == 3480 and len(out.current_matches) == 440 and len(out.history_matches) == 3040
    assert len(out.fixtures) == 41 and len(out.predictions) == 40
    assert out.skipped_fixtures == [extra.label]
    assert f"partita saltata, squadra fuori dal perimetro 2026/27 ({historical})" in console
    assert any("fuori dal perimetro 2026/27" in n for n in out.notes)
    assert out.params.xi == pytest.approx(0.0018) and historical in out.params.historical_teams
    assert "Perimetro: Serie C 2026/27 Gironi A/B/C + Serie B (80 squadre) · stagione corrente: 440 partite · storico: 3040 partite 2024/25-2025/26" in console


# ----------------------------------------------------------------------------- dashboard data layer
@pytest.fixture(scope="module")
def sim_state():
    return build_state(simulate=True, seed=7, xi=0.0, as_of=TODAY)


def test_build_state_simulated_uses_the_2026_27_perimeter(sim_state) -> None:
    state = sim_state
    assert state.source == "simulated" and state.season_label == "2026/27"
    assert state.registry is not None and state.registry.n_teams == 80
    assert state.n_current_matches == 440 and state.n_history_matches == 3040 and state.n_matches == 3480
    assert state.history_label == "2024/25-2025/26"
    assert state.perimeter_text == "Perimetro: Serie C 2026/27 Gironi A/B/C + Serie B (80 squadre) · 440 partite stagione corrente · 3040 partite storiche (2024/25-2025/26)"
    assert state.n_current_teams == 80 and state.n_historical_teams >= 12
    historical = set(state.params.historical_teams)
    assert len(state.fixtures) == 40 == len(state.predictions) and state.skipped_fixtures == []
    assert {p.fixture.date for p in state.predictions} == {date(2026, 11, 8)}
    assert not any(t in historical for p in fixtures_for(state, None) for t in (p.fixture.home_team, p.fixture.away_team))
    teams_in_value = set(value_table(state, -1.0)["Partita"].str.split(" - ").explode())
    assert teams_in_value and not teams_in_value & historical
    assert not set(traps_table(state)["Squadra"]) & historical and state.traps
    ranking = ranking_table(state, None)
    assert len(ranking) == 80 and not set(ranking["Squadra"]) & historical
    assert ranking.loc[ranking["Squadra"] == "Inter U23", "Competizione"].item() == "Girone C"
    assert ranking.loc[ranking["Squadra"] == "Juventus Next Gen", "Competizione"].item() == "Girone A"
    assert ranking.loc[ranking["Squadra"] == "Vicenza", "Competizione"].item() == "Serie B"
    assert state.prior_enabled, "11 current-season rounds are below the 15-per-team threshold, history does not count"
    joined = "\n".join(state.notes)
    assert "Calendario simulato al 2026-11-06" in joined and "11 giornate giocate" in joined
    assert "Storico simulato: 3040 partite delle stagioni 2024/25, 2025/26" in joined
    assert "Perimetro: Serie C 2026/27 Gironi A/B/C + Serie B (80 squadre), 80 squadre con partite nei dati" in joined


def test_build_state_history_options_and_validation() -> None:
    none = build_state(simulate=True, seed=7, xi=0.0, as_of=TODAY, history_seasons=0)
    assert none.n_history_matches == 0 and none.params.historical_teams == () and none.n_matches == 440
    assert none.perimeter_text == "Perimetro: Serie C 2026/27 Gironi A/B/C + Serie B (80 squadre) · 440 partite stagione corrente · 0 partite storiche"
    one = build_state(simulate=True, seed=7, xi=0.0, as_of=TODAY, history_seasons=1)
    assert one.n_history_matches == 1520 and one.history_label == "2025/26"
    with pytest.raises(ValueError, match="history_seasons"):
        build_state(simulate=True, seed=7, xi=0.0, history_seasons=MAX_HISTORY_SEASONS + 1)
    # before the kick-off one round is simulated anyway; after the end the last round stays as fixtures
    early = build_state(simulate=True, seed=7, xi=0.0, as_of=date(2026, 8, 1), history_seasons=0)
    assert early.n_current_matches == 40 and len(early.fixtures) == 40 and any("non è ancora iniziata" in n for n in early.notes)
    late = build_state(simulate=True, seed=7, xi=0.0, as_of=date(2027, 7, 1), history_seasons=0)
    assert late.n_current_matches == 37 * 40 and len(late.fixtures) == 40 and any("è conclusa" in n for n in late.notes)


def test_build_state_processed_three_seasons(tmp_path: Path, history, current) -> None:
    processed = tmp_path / "processed"
    frame = pd.concat([history.matches, current.matches], ignore_index=True)
    save_processed(frame, processed / "matches.csv")
    historical = sorted(set(history.teams) - set(current.registry.canonical_names))[0]
    extra = Fixture("HIST-FX", current.fixtures[0].date, 12, Girone.A, historical, current.registry.teams("A")[0])
    save_fixtures([*current.fixtures, extra], processed / "fixtures.csv")
    state = build_state(processed_dir=processed, odds_dir=tmp_path / "odds", xi=0.0)
    assert state.source == "processed" and state.season_label == "2026/27"
    assert state.n_current_matches == 440 and state.n_history_matches == 3040
    assert len(state.fixtures) == 41 and len(state.predictions) == 40 and state.skipped_fixtures == [extra.label]
    assert any("fuori dal perimetro 2026/27" in n for n in state.notes)
    assert historical in state.params.historical_teams and state.prior_enabled
    # the registry season can be changed: with 2025/26 the current season is the middle one
    old = build_state(processed_dir=processed, odds_dir=tmp_path / "odds", xi=0.0, season="2025_26")
    assert old.season_label == "2025/26" and old.n_current_matches == 1520 and old.n_history_matches == 1960
    assert old.registry is not None and old.registry.n_teams == 80 and "Vicenza" in old.params.current_teams
    assert old.params.girone_of("Vicenza") is Girone.A and old.params.girone_of("Pescara") is Girone.SB  # 2025/26 competitions
    # the fixtures are dated November 2026: for a 2025/26 perimeter they belong to another season and are all skipped
    assert old.predictions == [] and len(old.skipped_fixtures) == 41
    assert any("partita di una stagione successiva (2026/27), fuori dal perimetro 2025/26" in n for n in old.notes)
    # 2026/27 newcomers (Desenzano, Spezia, Bari, ...) are historical for that perimeter
    assert "Desenzano" in old.params.historical_teams and old.params.girone_of("Inter U23") is Girone.A


def test_split_by_season_edge_cases() -> None:
    records = [
        MatchRecord("A1", date(2026, 6, 30), 38, Girone.A, "Lecco", "Novara", 1, 0),
        MatchRecord("A2", date(2026, 7, 1), 1, Girone.A, "Novara", "Lecco", 0, 0),
    ]
    frame = matches_to_frame(records)
    cur, hist = split_by_season(frame, "2026/27")
    assert cur["match_id"].tolist() == ["A2"] and hist["match_id"].tolist() == ["A1"]
    cur_u, hist_u = split_by_season(frame, "2026_27")
    pd.testing.assert_frame_equal(cur_u, cur)
    pd.testing.assert_frame_equal(hist_u, hist)
    empty_cur, empty_hist = split_by_season(frame.iloc[0:0], "2026/27")
    assert empty_cur.empty and empty_hist.empty and list(empty_cur.columns) == list(frame.columns)


# ----------------------------------------------------------------------------- review findings (seasons chain)
def test_previous_seasons_and_history_seasons_for(registry: TeamRegistry) -> None:
    assert previous_seasons("2026/27", 2) == ("2024/25", "2025/26") == HISTORY_SEASONS
    assert previous_seasons("2025_26", 1) == ("2024/25",) and previous_seasons("2026/27", 0) == ()
    assert previous_seasons("2000/01", 2) == ("1998/99", "1999/00")
    with pytest.raises(ValueError):
        previous_seasons("sim", 2)
    assert history_seasons_for(registry, 2) == ("2024/25", "2025/26") and history_seasons_for(registry, 1) == ("2025/26",)
    assert history_seasons_for(load_registry("2025_26"), 2) == ("2023/24", "2024/25")
    sim = TeamRegistry.from_teams("sim", {"A": ("x", "y")})  # not a season label: the default history
    assert history_seasons_for(sim, 2) == HISTORY_SEASONS and history_seasons_for(sim, 1) == ("2025/26",)
    assert history_seasons_for(sim, 0) == () and history_seasons_for(sim, 5) == HISTORY_SEASONS


def test_simulated_calendar_and_history_follow_the_registry_season() -> None:
    """Finding: a 2025/26 registry was simulated on the 2026/27 calendar, so its 'current' matches were dated in
    the following season and the synthetic 2025/26 history became the current season."""
    old = load_registry("2025_26")
    current = simulate_league(seed=7, registry=old, as_of=date(2025, 11, 6), with_odds=False)
    assert current.matches["date"].min().date() == date(2025, 8, 24) == season_start_date("2025/26")
    assert set(current.matches["date"].map(season_of)) == {"2025/26"} and current.seasons == ("2025/26",)
    assert int(current.matches["round"].max()) == 11 and {fx.date for fx in current.fixtures} == {date(2025, 11, 9)}
    history = simulate_history(seed=7, registry=old)
    assert history.seasons == ("2023/24", "2024/25")
    assert history.matches["date"].max().date() < date(2025, 7, 1)
    assert set(history.matches["date"].map(season_of)) == {"2023/24", "2024/25"}
    assert "Inter U23" not in history.season_registries["2023/24"].canonical_names  # founded in 2025
    assert history.season_registries["2024/25"].girone_of("Atalanta U23") is Girone.A
    assert history.season_registries["2024/25"].girone_of("Juventus Next Gen") is Girone.C
    # the mini snapshot of another season sits on that season's calendar too
    mini = simulate_mini_dataset(seed=42, season="2025/26")
    assert mini.matches["date"].min().date() == date(2025, 8, 24) and {fx.date for fx in mini.fixtures} == {date(2025, 9, 14)}
    # a registry whose label is not a season keeps the 2026/27 calendar; an explicit start_date still wins
    sim = simulate_league(seed=1, teams_by_girone={"A": ("x", "y", "z", "w")}, n_rounds=1, with_odds=False)
    assert sim.matches["date"].min().date() == SEASON_2026_27_START
    explicit = simulate_league(seed=1, registry=old, n_rounds=1, start_date=date(2030, 1, 6), with_odds=False)
    assert explicit.matches["date"].min().date() == date(2030, 1, 6)


def test_historical_pool_is_size_safe(caplog: pytest.LogCaptureFixture, registry: TeamRegistry) -> None:
    """Finding: 12 of the 20 HISTORICAL_TEAMS play Serie C 2025/26, so simulate_history(registry=2025_26) raised
    'pool exhausted'. The pool now falls back to the registry's historical names and then to synthetic clubs."""
    old = load_registry("2025_26")
    with caplog.at_level(logging.WARNING, logger="src.simulation.generator"):
        history = simulate_history(seed=7, registry=old)
    assert any("historical pool exhausted" in rec.getMessage() for rec in caplog.records)
    assert len(history.matches) == 2 * 4 * 380
    for label, comp in history.season_registries.items():
        assert comp.gironi == old.gironi and all(len(comp.teams(g)) == 20 for g in old.gironi), label
        extra = set(comp.canonical_names) - set(old.canonical_names)
        assert len(extra) >= 12 and not extra & set(old.canonical_names)
    synthetic = {t for t in history.teams if t.startswith(SYNTHETIC_HISTORICAL_PREFIX)}
    assert synthetic == {f"Storica {i:02d}" for i in range(1, 7)}  # 8 real names left for 4 competitions x 3 swaps + fills
    # zero-padded on purpose: normalize_name drops single-character tokens, so 'Storica 1'/'Storica 2' would collide
    assert len({normalize_name(t) for t in synthetic}) == len(synthetic) == 6
    # every team that plays has true strengths (the extra names draw from the same distribution)
    assert set(history.teams) <= set(history.true_params.alpha) and set(history.teams) <= set(history.true_params.beta)
    # the eight real names left (15 of the 23 play in the 2025/26 perimeter) are used before any synthetic one
    real_left = set(HISTORICAL_TEAMS) - set(old.canonical_names)
    assert len(real_left) == 8 and real_left <= set(history.teams)
    # without swaps the moves alone fit in the real pool: no synthetic club
    moves_only = simulate_history(seed=7, registry=old, swap_per_girone=0)
    assert not any(t.startswith(SYNTHETIC_HISTORICAL_PREFIX) for t in moves_only.teams)
    # the registry's own historical names come before the synthetic ones
    crowded = TeamRegistry.from_teams("2026/27", {"A": HISTORICAL_TEAMS[:10], "B": HISTORICAL_TEAMS[10:]})
    crowded = dataclasses.replace(crowded, historical_names=("Club Uno", "Club Due"))
    one_swap = simulate_history(seed=3, registry=crowded, seasons=("2025/26",), swap_per_girone=1)
    extra = set(one_swap.teams) - set(crowded.canonical_names)
    assert extra == {"Club Uno", "Club Due"}
    two_swaps = simulate_history(seed=3, registry=crowded, seasons=("2025/26",), swap_per_girone=2)
    extra = set(two_swaps.teams) - set(crowded.canonical_names)
    assert {"Club Uno", "Club Due"} <= extra and len(extra) == 4 and len([t for t in extra if t.startswith("Storica")]) == 2
    # the default registry is untouched: same history as before, no synthetic club
    assert not any(t.startswith(SYNTHETIC_HISTORICAL_PREFIX) for t in simulate_history(seed=7, registry=registry).teams)


def test_non_default_season_runs_end_to_end_and_is_consistent(capsys: pytest.CaptureFixture[str]) -> None:
    """Findings 1 + 3 on every surface: ``--season 2025_26`` neither crashes nor swaps current and history."""
    mini = pipeline.main(["--mode", "mini", "--no-export", "--log-level", "ERROR", "--season", "2025_26"])
    console = capsys.readouterr().out
    assert "Perimetro: Serie C 2025/26 Gironi A/B/C (14 squadre) · stagione corrente: 20 partite · storico: 108 partite 2023/24-2024/25" in console
    assert "Demo mini: istantanea fissa di inizio stagione 2025/26 (giornate 1-3 giocate, 20 partite; turno da prevedere: 4ª giornata del 2025-09-14)" in console
    assert set(mini.current_matches["date"].map(season_of)) == {"2025/26"} and {p.fixture.date for p in mini.predictions} == {date(2025, 9, 14)}
    assert set(mini.history_matches["date"].map(season_of)) == {"2023/24", "2024/25"}

    full = pipeline.main(["--mode", "full", "--no-export", "--log-level", "ERROR", "--season", "2025_26", "--as-of", "2025-11-06", "--history", "1"])
    console = capsys.readouterr().out
    assert "Perimetro: Serie C 2025/26 Gironi A/B/C + Serie B (80 squadre) · stagione corrente: 440 partite · storico: 1520 partite 2024/25" in console
    assert "Calendario simulato al 2025-11-06: giornate con data precedente giocate (11), la prima giornata in data uguale o successiva (2025-11-09)" in console
    assert full.registry is not None and full.registry.season_label == "2025/26" and "Vicenza" in full.params.current_teams
    assert full.params.girone_of("Atalanta U23") is Girone.C and full.params.girone_of("Juventus Next Gen") is Girone.B  # 2025/26 gironi
    assert full.params.girone_of("Pescara") is Girone.SB and full.params.girone_of("Vicenza") is Girone.A  # 2025/26 divisions
    assert any(t.startswith(SYNTHETIC_HISTORICAL_PREFIX) for t in full.params.historical_teams)
    assert len(full.predictions) == 40 and {p.fixture.date for p in full.predictions} == {date(2025, 11, 9)} and full.skipped_fixtures == []
    assert full.params.penalty > 0.0, "11 current-season rounds: the prior is on, the history does not count as coverage"

    state = build_state(simulate=True, seed=7, xi=0.0, as_of=date(2025, 11, 6), season="2025_26")
    assert state.season_label == "2025/26" and state.registry is not None and state.registry.n_teams == 80
    assert state.n_current_matches == 440 and state.n_history_matches == 3040 and state.history_label == "2023/24-2024/25"
    assert state.perimeter_text == "Perimetro: Serie C 2025/26 Gironi A/B/C + Serie B (80 squadre) · 440 partite stagione corrente · 3040 partite storiche (2023/24-2024/25)"
    assert len(state.predictions) == 40 and {p.fixture.date for p in state.predictions} == {date(2025, 11, 9)} and state.skipped_fixtures == []
    assert state.prior_enabled and not set(ranking_table(state, None)["Squadra"]) & set(state.params.historical_teams)
    assert any("Calendario simulato al 2025-11-06" in n for n in state.notes)
    # as of today the 2025/26 season is over: the last round is kept as fixtures, dated in that season
    over = build_state(simulate=True, seed=7, xi=0.0, as_of=TODAY, season="2025_26", history_seasons=0)
    assert over.n_current_matches == 37 * 40 and {p.fixture.date for p in over.predictions} == {date(2026, 5, 10)}
    assert any("la stagione 2025/26 è conclusa" in n for n in over.notes)


def test_stale_fixtures_of_last_season_are_never_predicted(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """Finding: a harvest of 2025/26 (matches + the last round as fixtures, dated May 2026) with the default 2026/27
    registry produced 18 predictions six months in the past, selectable in Match & Heatmap and priced in the value
    panel. Fixtures dated in another season are skipped on both surfaces."""
    old = load_registry("2025_26")
    league = simulate_league(seed=5, registry=old, n_rounds=37, start_date=season_start_date("2025/26"))
    assert {fx.date for fx in league.fixtures} == {date(2026, 5, 10)} and len(league.fixtures) == 40
    processed = tmp_path / "processed"
    save_processed(league.matches, processed / "matches.csv")
    # one current-season fixture between current teams among the stale ones: it is the only one predicted
    reg = load_registry()
    fresh = Fixture("FRESH-1", date(2026, 11, 8), 12, Girone.A, reg.teams("A")[0], reg.teams("A")[1])
    save_fixtures([*league.fixtures, fresh], processed / "fixtures.csv")

    state = build_state(processed_dir=processed, odds_dir=tmp_path / "odds", xi=0.0)
    assert state.season_label == "2026/27" and state.n_current_matches == 0 and state.n_history_matches == 1480
    assert [p.fixture.match_id for p in state.predictions] == ["FRESH-1"]
    assert len(state.skipped_fixtures) == 40 and set(state.skipped_fixtures) == {fx.label for fx in league.fixtures}
    assert any("40 partite in programma saltate (partita di una stagione precedente (2025/26), fuori dal perimetro 2026/27)" in n for n in state.notes)
    assert value_table(state, -1.0).empty and all(p.fixture.date >= date(2026, 7, 1) for p in fixtures_for(state, None))

    out = pipeline.main(["--processed", str(processed / "matches.csv"), "--fixtures", str(processed / "fixtures.csv"), "--no-export", "--log-level", "ERROR"])
    console = capsys.readouterr().out
    assert [p.fixture.match_id for p in out.predictions] == ["FRESH-1"] and len(out.skipped_fixtures) == 40
    assert "partita saltata, partita di una stagione precedente (2025/26), fuori dal perimetro 2026/27" in console
    assert any(n.startswith("40 partite in programma saltate (partite di un'altra stagione, non della 2026/27)") for n in out.notes)
    assert "Perimetro: Serie C 2026/27 Gironi A/B/C + Serie B (80 squadre) · stagione corrente: 0 partite · storico: 1480 partite 2025/26" in console
    assert other_season_reason("2025/26", "2026/27") == "partita di una stagione precedente (2025/26), fuori dal perimetro 2026/27"
    assert other_season_reason("2027/28", "2026/27").startswith("partita di una stagione successiva (2027/28)")


def test_main_full_mode_follows_the_calendar(capsys: pytest.CaptureFixture[str]) -> None:
    """Finding: full mode hard-coded 30 rounds (into March 2027) whatever the date; the mini demo did not say that
    its round 4 of September is a fixed snapshot."""
    early = pipeline.main(["--mode", "full", "--no-export", "--log-level", "ERROR", "--history", "0", "--as-of", "2026-08-01"])
    console = capsys.readouterr().out
    assert len(early.current_matches) == 40 and {p.fixture.date for p in early.predictions} == {date(2026, 8, 30)}
    assert "Calendario simulato al 2026-08-01: la stagione 2026/27 non è ancora iniziata; simulo comunque 1 giornata" in console
    assert "1 giornate giocate al 2026-08-01" in console
    late = pipeline.main(["--mode", "full", "--no-export", "--log-level", "ERROR", "--history", "0", "--as-of", "2027-07-01"])
    console = capsys.readouterr().out
    assert len(late.current_matches) == 37 * 40 and {p.fixture.date for p in late.predictions} == {date(2027, 5, 9)}
    assert "la stagione 2026/27 è conclusa; tengo l'ultima giornata come partite in programma (37 giornate giocate)" in console
    clamped = pipeline.main(["--mode", "full", "--no-export", "--log-level", "ERROR", "--history", "0", "--rounds", "50"])
    console = capsys.readouterr().out
    assert len(clamped.current_matches) == 37 * 40 and "valore limitato a 37" in console and len(clamped.predictions) == 40
    assert any("limitato a 37" in n for n in clamped.notes)
    # invalid values are CLI errors, not tracebacks
    for bad in (["--rounds", "0"], ["--as-of", "yesterday"]):
        with pytest.raises(SystemExit) as exc:
            pipeline.main(["--mode", "full", "--no-export", *bad])
        assert exc.value.code == 2
    capsys.readouterr()
    # the shared helper is what both surfaces use
    reg = load_registry()
    notes: list[str] = []
    league, played = simulate_current_season(7, reg, as_of=TODAY, notes=notes)
    assert played == 11 and len(league.matches) == 440 and {fx.date for fx in league.fixtures} == {date(2026, 11, 8)}
    assert notes == ["Calendario simulato al 2026-11-06: giornate con data precedente giocate (11), la prima giornata in data uguale o successiva (2026-11-08) è il turno da prevedere."]
    fixed, played = simulate_current_season(7, reg, n_rounds=50, as_of=TODAY, notes=notes)
    assert played == 37 and len(fixed.fixtures) == 40 and "limitato a 37" in notes[-1]
