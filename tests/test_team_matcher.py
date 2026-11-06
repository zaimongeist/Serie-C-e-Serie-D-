"""Tests for :mod:`src.ingestion.team_matcher` (pure, offline).

The matching-logic tests run on the 2025/26 dictionary (``LEGACY_SEASON``): its
spellings (``Vicenza``, ``Virtus Verona``, ...) are the ones the assertions were
written for, and pinning the season keeps them independent of
:data:`src.common.teams.DEFAULT_SEASON`. The historical-team tests use the
current 2026/27 perimeter extended with the 2025/26 aliases.
"""
from __future__ import annotations

import pandas as pd
import pytest

from src.common.schemas import Girone
from src.common.teams import DEFAULT_SEASON, TeamRegistry, load_registry
from src.ingestion import team_matcher as tm
from src.ingestion.team_matcher import MatchResult, TeamMatcher

LEGACY_SEASON = "2025_26"
CURRENT_SEASON = "2026_27"


@pytest.fixture(scope="module")
def registry() -> TeamRegistry:
    return load_registry(LEGACY_SEASON)


@pytest.fixture(scope="module")
def history_matcher() -> TeamMatcher:
    """Current perimeter (2026/27) plus the aliases of the older dictionaries."""
    return TeamMatcher(load_registry(CURRENT_SEASON, with_history=True))


@pytest.fixture
def matcher(registry: TeamRegistry) -> TeamMatcher:
    return TeamMatcher(registry)


# --------------------------------------------------------------------- lookups
def test_exact_canonical(matcher: TeamMatcher) -> None:
    result = matcher.match("Vicenza")
    assert result == MatchResult("Vicenza", "Vicenza", 100, "exact", None, (("Vicenza", 100),))
    assert result.matched


def test_exact_after_normalisation(matcher: TeamMatcher) -> None:
    # "Pergolettese Calcio 1932" normalises to "pergolettese" -> canonical hit.
    result = matcher.match("Pergolettese Calcio 1932")
    assert result.canonical == "Pergolettese"
    assert result.method in {"exact", "alias", "fuzzy"}
    assert result.score >= matcher.threshold


def test_alias_lookup(matcher: TeamMatcher) -> None:
    result = matcher.match("Juventus U23")
    assert result.canonical == "Juventus Next Gen"
    assert result.method == "alias"
    assert result.score == 100
    assert matcher.match("L.R. Vicenza").canonical == "Vicenza"
    assert matcher.match("Brescia").canonical == "Union Brescia"


def test_fuzzy_lookup(matcher: TeamMatcher) -> None:
    result = matcher.match("Juve Next Gen")
    assert result.canonical == "Juventus Next Gen"
    assert result.method == "fuzzy"
    assert matcher.threshold <= result.score < 100
    assert result.candidates[0] == ("Juventus Next Gen", result.score)

    typo = matcher.match("Pergoletese")
    assert typo.canonical == "Pergolettese"
    assert typo.method == "fuzzy"


def test_team_not_in_serie_c_is_none(matcher: TeamMatcher) -> None:
    result = matcher.match("Virtus Entella")
    assert result.canonical is None
    assert result.method == "none"
    assert not result.matched
    assert result.score < matcher.threshold
    assert result.candidates and result.candidates[0][0] == "Virtus Verona"


def test_empty_name(matcher: TeamMatcher) -> None:
    assert matcher.match("") == MatchResult("", None, 0, "none", None, ())
    assert matcher.match("   ").canonical is None


def test_girone_restriction(matcher: TeamMatcher) -> None:
    assert matcher.match("Vicenza", Girone.A).canonical == "Vicenza"
    wrong = matcher.match("Vicenza", Girone.B)
    assert wrong.canonical is None and wrong.girone is Girone.B
    assert matcher.match("Juve Next Gen", "B").canonical == "Juventus Next Gen"
    assert matcher.match("Juve Next Gen", Girone.A).canonical is None
    assert matcher.match("Juventus U23", Girone.C).canonical is None


def test_threshold_respected(registry: TeamRegistry) -> None:
    strict = TeamMatcher(registry, threshold=100)
    assert strict.match("Juve Next Gen").canonical is None
    assert strict.match("Juventus U23").canonical == "Juventus Next Gen"  # alias path unaffected
    lenient = TeamMatcher(registry, threshold=70)
    assert lenient.match("Virtus Entella").canonical == "Virtus Verona"


def test_ambiguity_returns_none_with_candidates() -> None:
    # "Sporting Xyz" shares only the token "sporting" with both candidates -> identical scores.
    reg = TeamRegistry.from_teams("test", {"A": ["Sporting Alpha", "Sporting Omega", "Union Gamma"]})
    ambiguous = TeamMatcher(reg, threshold=50, ambiguity_margin=3).match("Sporting Xyz")
    assert ambiguous.canonical is None
    assert ambiguous.method == "none"
    assert ambiguous.score >= 50
    names = [name for name, _ in ambiguous.candidates[:2]]
    assert set(names) == {"Sporting Alpha", "Sporting Omega"}
    assert abs(ambiguous.candidates[0][1] - ambiguous.candidates[1][1]) < 3

    decisive = TeamMatcher(reg, threshold=50, ambiguity_margin=0).match("Sporting Xyz")
    assert decisive.canonical in {"Sporting Alpha", "Sporting Omega"}
    assert decisive.method == "fuzzy"
    assert TeamMatcher(reg, threshold=50, ambiguity_margin=3).match("Sporting Alpha").method == "exact"


def test_invalid_configuration(registry: TeamRegistry) -> None:
    with pytest.raises(ValueError):
        TeamMatcher(registry, threshold=101)
    with pytest.raises(ValueError):
        TeamMatcher(registry, ambiguity_margin=-1)
    with pytest.raises(ValueError):
        TeamMatcher(registry, scorer="does_not_exist")
    assert TeamMatcher(registry, scorer="ratio").match("Vicenza").canonical == "Vicenza"


def test_memoisation(matcher: TeamMatcher) -> None:
    first = matcher.match("Juve Next Gen")
    assert matcher.cache_size == 1
    again = matcher.match("Juve Next Gen")
    assert again is first
    other_spelling = matcher.match("JUVE NEXT GEN")
    assert matcher.cache_size == 1
    assert other_spelling.canonical == first.canonical and other_spelling.query == "JUVE NEXT GEN"
    matcher.match("Juve Next Gen", Girone.B)
    assert matcher.cache_size == 2
    matcher.clear_cache()
    assert matcher.cache_size == 0


# ----------------------------------------------------------------- bulk helpers
def test_map_names(matcher: TeamMatcher) -> None:
    out = matcher.map_names(["L.R. Vicenza", "Juve Next Gen", "Virtus Entella", "L.R. Vicenza"])
    assert out == {"L.R. Vicenza": "Vicenza", "Juve Next Gen": "Juventus Next Gen", "Virtus Entella": None}
    assert matcher.map_names(["Vicenza"], girone=Girone.B) == {"Vicenza": None}


def test_map_frame_reports_unmapped_counts(matcher: TeamMatcher) -> None:
    df = pd.DataFrame(
        {
            "home_team": ["L.R. Vicenza", "Virtus Entella", "Juventus U23", None],
            "away_team": ["Pro Vercelli", "Virtus Entella", "Sestri Levante", "Novara"],
            "girone": ["A", "A", "B", "A"],
            "home_goals": [1, 2, 0, 3],
        }
    )
    snapshot = df.copy()
    out, unmapped = matcher.map_frame(df)
    pd.testing.assert_frame_equal(df, snapshot)  # input untouched
    assert out["home_team"].tolist()[:3] == ["Vicenza", "Virtus Entella", "Juventus Next Gen"]
    assert pd.isna(out["home_team"].iloc[3])
    assert out["away_team"].tolist() == ["Pro Vercelli", "Virtus Entella", "Sestri Levante", "Novara"]
    assert unmapped == {"Virtus Entella": 2, "Sestri Levante": 1}
    assert str(out["home_team"].dtype) == "string"
    assert out["home_goals"].tolist() == [1, 2, 0, 3]


def test_map_frame_girone_restriction_and_missing_column(matcher: TeamMatcher) -> None:
    df = pd.DataFrame({"home_team": ["Vicenza"], "away_team": ["Juventus U23"], "girone": ["B"]})
    out, unmapped = matcher.map_frame(df)
    assert out["home_team"].tolist() == ["Vicenza"]  # raw kept: not a girone B team
    assert out["away_team"].tolist() == ["Juventus Next Gen"]
    assert unmapped == {"Vicenza": 1}

    no_girone = pd.DataFrame({"home_team": ["Vicenza"], "away_team": ["Juventus U23"]})
    out2, unmapped2 = matcher.map_frame(no_girone)
    assert out2["home_team"].tolist() == ["Vicenza"] and unmapped2 == {}

    with pytest.raises(ValueError):
        matcher.map_frame(pd.DataFrame({"home_team": ["Vicenza"]}))


# -------------------------------------------------------------- difflib fallback
def test_difflib_fallback_path(monkeypatch: pytest.MonkeyPatch, registry: TeamRegistry) -> None:
    calls: list[str] = []
    original = tm.difflib_extract

    def spy(query: str, choices, *, limit: int = 12, scorer=tm.difflib_score):
        calls.append(query)
        return original(query, choices, limit=limit, scorer=scorer)

    monkeypatch.setattr(tm, "DIFFLIB_FALLBACK", True)
    monkeypatch.setattr(tm, "difflib_extract", spy)
    fallback = TeamMatcher(registry)
    result = fallback.match("Juve Next Gen")
    assert calls == ["juve next gen"]
    assert result.canonical == "Juventus Next Gen"
    assert result.method == "fuzzy"
    assert fallback.match("Virtus Entella").canonical is None
    assert fallback.match("Juventus U23").method == "alias"  # non-fuzzy steps unaffected
    assert len(calls) == 2


def test_difflib_scoring_helpers() -> None:
    assert tm.difflib_score("vicenza", "vicenza") == 100
    assert 0 <= tm.difflib_score("vicenza", "catania") < 100
    ranked = tm.difflib_extract("pergolettese", ["pergolettese", "pro patria", "catania"], limit=2)
    assert ranked[0] == ("pergolettese", 100) and len(ranked) == 2


# ------------------------------------------------------- token-subset guard (review)
@pytest.mark.parametrize(
    "name",
    ["Hellas Verona", "Inter Milan", "Internazionale", "Inter", "Juventus", "Atalanta", "Milano", "Union", "Team", "Pesaro", "Vis", "U23"],
)
def test_token_subset_is_not_fuzzy_evidence(matcher: TeamMatcher, name: str) -> None:
    """Parent clubs / bare qualifiers must not collapse onto the U23 or namesake Serie C sides."""
    result = matcher.match(name)
    assert result.canonical is None and result.method == "none", (name, result)
    assert result.score < matcher.threshold


def test_token_subset_guard_holds_with_girone_restriction(matcher: TeamMatcher) -> None:
    assert matcher.match("Juventus", Girone.B).canonical is None
    assert matcher.match("Inter Milan", Girone.A).canonical is None
    assert matcher.match("Hellas Verona", "A").canonical is None
    # candidates still show what was considered, with the confirmed (low) score
    considered = matcher.match("Juventus", Girone.B)
    assert considered.candidates[0][0] == "Juventus Next Gen" and considered.candidates[0][1] < matcher.threshold


def test_fuzzy_score_is_capped_below_100(matcher: TeamMatcher) -> None:
    reordered = matcher.match("Next Gen Juventus")  # token_set/token_sort both say 100: still only fuzzy evidence
    assert reordered.canonical == "Juventus Next Gen" and reordered.method == "fuzzy"
    assert reordered.score == tm.MAX_FUZZY_SCORE == 99
    assert matcher.match("Juventus Next Gen").score == 100 and matcher.match("Juventus U23").score == 100
    # genuine typos / abbreviations keep their scores
    assert matcher.match("Juve Next Gen").score == 87 and matcher.match("Pergoletese").score == 96


def test_confirmation_helpers() -> None:
    assert tm.token_coverage("juventus", "juventus next gen") == pytest.approx(1 / 3)
    assert tm.token_coverage("hellas verona", "verona") == pytest.approx(0.5)
    assert tm.token_coverage("hellas verona", "virtus verona") == 1.0  # overlap, not containment
    assert tm.token_coverage("vicenza", "vicenza") == 1.0 and tm.token_coverage("", "x") == 1.0
    assert tm.whole_string_score("juve next gen", "juventus next gen") == 87
    assert tm.whole_string_score("gen next juventus", "juventus next gen") == 100
    assert tm.whole_string_score("juventus", "juventus next gen") < 70
    assert tm.confirm_score("internazionale", "internazionale u23") == round(88 * 0.5)
    assert tm.confirm_score("inter milan", "inter milan u23") < 60
    assert TeamMatcher.fuzzy_score("juventus", "juventus next gen", 100) < 30
    assert TeamMatcher.fuzzy_score("gen next juventus", "juventus next gen", 100) == tm.MAX_FUZZY_SCORE


def test_token_subset_guard_on_difflib_fallback(monkeypatch: pytest.MonkeyPatch, registry: TeamRegistry) -> None:
    monkeypatch.setattr(tm, "DIFFLIB_FALLBACK", True)
    fallback = TeamMatcher(registry)
    assert fallback.match("Internazionale").canonical is None  # difflib ratio alone would be 88
    assert fallback.match("Juventus").canonical is None
    assert fallback.match("Juve Next Gen").canonical == "Juventus Next Gen"
    assert tm.whole_string_score("gen next juventus", "juventus next gen") == 100  # token-sorted difflib path


# ------------------------------------------------------- current vs historical teams
def test_default_season_is_the_current_perimeter() -> None:
    assert DEFAULT_SEASON == CURRENT_SEASON
    plain = TeamMatcher(load_registry())
    assert len(plain.current_names) == 60 and plain.historical_names == frozenset()
    assert plain.match("Vicenza").canonical is None  # relegated: unknown without history
    assert plain.match("Juventus U23").canonical == "Juventus Next Gen"  # alias kept in the 2026/27 file


def test_historical_names_are_derived_from_the_registry(history_matcher: TeamMatcher) -> None:
    assert len(history_matcher.current_names) == 60
    assert {"Ternana", "Vicenza", "Pro Patria", "Triestina", "Rimini", "Virtus Verona", "Benevento"} <= history_matcher.historical_names
    assert history_matcher.historical_names.isdisjoint(history_matcher.current_names)
    assert history_matcher.is_historical("Ternana") and not history_matcher.is_current("Ternana")
    assert history_matcher.is_current("Perugia") and not history_matcher.is_historical("Perugia")
    assert not history_matcher.is_current("SPAL") and not history_matcher.is_historical("SPAL")
    # A registry without the ``historical_names`` attribute still works: targets of the
    # alias table that are not current teams are historical.
    synthetic = TeamRegistry(
        season="test",
        teams_by_girone={Girone.A: ("Alpha", "Beta")},
        aliases={"alpha": "Alpha", "beta": "Beta", "gamma": "Gamma", "gamma old": "Gamma"},
    )
    matcher = TeamMatcher(synthetic)
    assert matcher.historical_names == frozenset({"Gamma"}) and matcher.current_names == frozenset({"Alpha", "Beta"})
    assert matcher.match("Gamma Old").canonical == "Gamma" and matcher.match("gamma").method == "exact"


def test_unrestricted_lookup_resolves_historical_teams(history_matcher: TeamMatcher) -> None:
    result = history_matcher.match("Ternana Calcio")  # "Calcio" is a dropped suffix: exact hit on the old canonical
    assert result.canonical == "Ternana" and result.method == "exact" and result.score == 100
    assert history_matcher.match("Ternana").method == "exact"
    via_alias = history_matcher.match("Lane Rossi Vicenza")
    assert via_alias.canonical == "Vicenza" and via_alias.method == "alias" and via_alias.score == 100
    assert history_matcher.match("L.R. Vicenza").canonical == "Vicenza"
    assert history_matcher.match("Virtus Vecomp Verona").canonical == "Virtus Verona"
    typo = history_matcher.match("Ternanna")
    assert typo.canonical == "Ternana" and typo.method == "fuzzy" and typo.score < 100


def test_restricted_lookup_only_returns_current_teams(history_matcher: TeamMatcher) -> None:
    # Ternana played in girone B until 2025/26 but is not in the 2026/27 perimeter.
    assert history_matcher.match("Ternana Calcio", Girone.B).canonical is None
    assert history_matcher.match("Ternana", "B").canonical is None
    assert history_matcher.match("Vicenza", Girone.A).canonical is None
    # Current teams that moved girone follow the CURRENT registry.
    assert history_matcher.match("Juventus U23", Girone.A).canonical == "Juventus Next Gen"
    assert history_matcher.match("Juventus U23", Girone.B).canonical is None  # played in B in 2025/26
    assert history_matcher.match("Juventus U23").canonical == "Juventus Next Gen"
    assert history_matcher.match("Inter U23", Girone.C).canonical == "Inter U23"
    assert history_matcher.match("Inter U23", Girone.A).canonical is None
    assert history_matcher.match("Carpi FC", Girone.A).canonical == "Carpi" and history_matcher.match("Carpi FC", "B").canonical is None
    assert history_matcher.match("Latina Calcio", Girone.B).canonical == "Latina"


def test_historical_aliases_never_steal_current_spellings(history_matcher: TeamMatcher) -> None:
    # Spellings shared by both dictionaries stay with the current team.
    assert history_matcher.match("Brescia").canonical == "Union Brescia"
    assert history_matcher.match("Samb", Girone.B).canonical == "Sambenedettese"
    assert history_matcher.match("Altamura").canonical == "Team Altamura"
    assert history_matcher.match("Juventus").canonical is None  # token-subset guard still holds


def test_map_frame_with_girone_column_keeps_historical_names_raw(history_matcher: TeamMatcher) -> None:
    df = pd.DataFrame(
        {
            "home_team": ["Ternana Calcio", "Juventus U23"],
            "away_team": ["Perugia", "Ternana Calcio"],
            "girone": ["B", "A"],
        }
    )
    out, unmapped = history_matcher.map_frame(df)
    assert out["home_team"].tolist() == ["Ternana Calcio", "Juventus Next Gen"]
    assert out["away_team"].tolist() == ["Perugia", "Ternana Calcio"]
    assert unmapped == {"Ternana Calcio": 2}
    unrestricted, unmapped_all = history_matcher.map_frame(df.drop(columns="girone"))
    assert unrestricted["home_team"].tolist() == ["Ternana", "Juventus Next Gen"] and unmapped_all == {}
