"""Tests for :mod:`src.ingestion.team_matcher` (pure, offline)."""
from __future__ import annotations

import pandas as pd
import pytest

from src.common.schemas import Girone
from src.common.teams import TeamRegistry, load_registry
from src.ingestion import team_matcher as tm
from src.ingestion.team_matcher import MatchResult, TeamMatcher


@pytest.fixture(scope="module")
def registry() -> TeamRegistry:
    return load_registry()


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
