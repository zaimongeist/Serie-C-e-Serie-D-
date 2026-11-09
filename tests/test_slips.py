"""Tests for ``src.engine.slips`` (the three matchday slips: X, GG and mixed)."""
from __future__ import annotations

import math
from collections.abc import Mapping
from datetime import UTC, date, datetime, timedelta
from typing import Any

import numpy as np
import pandas as pd
import pytest

from src.common.schemas import Fixture, Girone, Market, MatchPrediction, OddsQuote
from src.engine.picks import (
    BTTS_NOTE_BOTH_ATTACKS,
    BTTS_NOTE_HIGH_XG,
    DRAW_NOTE_CLOSE_RATES,
    DRAW_NOTE_FEW_GOALS,
    Matchday,
)
from src.engine.slips import (
    DEFAULT_MIXED_MARKETS,
    DEFAULT_N_GG,
    DEFAULT_N_MIXED,
    DEFAULT_N_X,
    FRAME_MARKET_SUFFIX,
    FRAME_SELECTION_SUFFIX,
    INDEPENDENCE_NOTE,
    MIXED_MAX_LEGS,
    MIXED_MIN_LEGS,
    PREDICTIONS_FRAME_COLUMNS,
    SLIPS_COLUMNS,
    SLIPS_SUMMARY_COLUMNS,
    STAKE_NOT_ADVISED,
    TYPICAL_LEG_MARGIN,
    Slip,
    SlipLeg,
    build_slips,
    format_stake,
    make_slip_builder,
    predictions_frame,
    slips_frame,
    slips_from_frame,
    slips_summary,
    suggested_stake,
)
from src.engine.value import ValueOpportunity
from src.simulation.generator import dixon_coles_matrix

#: Sunday 8 November 2026, 11:00 in Rome.
NOW = datetime(2026, 11, 8, 10, 0, tzinfo=UTC)
MATRIX = dixon_coles_matrix(1.3, 1.1, -0.05)


# ----------------------------------------------------------------------------- helpers
def probabilities(
    *,
    p1: float = 0.40,
    px: float = 0.28,
    p2: float | None = None,
    gg: float = 0.50,
    o15: float = 0.72,
    o25: float = 0.45,
    o35: float = 0.22,
) -> dict[Market, dict[str, float]]:
    """A complete probability mapping; ``p2`` defaults to the 1X2 remainder."""
    away = 1.0 - p1 - px if p2 is None else p2
    return {
        Market.MATCH_ODDS: {"1": p1, "X": px, "2": away},
        Market.OVER_UNDER_15: {"Over 1.5": o15, "Under 1.5": 1.0 - o15},
        Market.OVER_UNDER_25: {"Over 2.5": o25, "Under 2.5": 1.0 - o25},
        Market.OVER_UNDER_35: {"Over 3.5": o35, "Under 3.5": 1.0 - o35},
        Market.BTTS: {"GG": gg, "NG": 1.0 - gg},
    }


def prediction(
    match_id: str,
    girone: str = "A",
    *,
    hours: float | None = 4.0,
    day: date | None = None,
    status: str = "scheduled",
    lam: float = 1.3,
    mu: float = 1.1,
    **probs: float,
) -> MatchPrediction:
    """A prediction kicking off ``hours`` after :data:`NOW` (``hours=None``: only the day, ``day`` or today)."""
    kickoff = None if hours is None else NOW + timedelta(hours=hours)
    when = day if day is not None else (kickoff.date() if kickoff is not None else NOW.date())
    fixture = Fixture(match_id, when, 13, girone, f"{match_id}-casa", f"{match_id}-ospite", kickoff=kickoff, status=status)
    p = probabilities(**probs)
    fair = {m: {s: 1.0 / v for s, v in sels.items()} for m, sels in p.items()}
    return MatchPrediction(fixture=fixture, home_rate=lam, away_rate=mu, rho=-0.05, matrix=MATRIX, probabilities=p, fair_odds=fair)


def opportunity(pred: MatchPrediction, market: Market, selection: str, odds: float, bookmaker: str = "BookA", margin: float = 0.05) -> ValueOpportunity:
    """A value-engine row for one selection; the de-vigged probability is ``1 / odds / (1 + margin)``."""
    p = pred.probabilities[market][selection]
    return ValueOpportunity(
        match_id=pred.match_id,
        date=pred.fixture.date,
        girone=pred.fixture.girone,
        home_team=pred.fixture.home_team,
        away_team=pred.fixture.away_team,
        market=market,
        selection=selection,
        bookmaker=bookmaker,
        book_odds=odds,
        fair_odds=1.0 / p,
        p_model=p,
        p_implied_raw=1.0 / odds,
        p_implied_devig=1.0 / odds / (1.0 + margin),
        ev=p * odds - 1.0,
        edge=0.0,
        kelly_full=0.0,
        kelly_fraction=0.125,
        stake_units=0.0,
        value=False,
    )


def by_kind(slips: list[Slip]) -> dict[str, Slip]:
    return {s.kind: s for s in slips}


def ids(slip: Slip) -> list[str]:
    return [leg.match_id for leg in slip.legs]


def leg(match_id: str, p: float, *, odds: float | None = None, margin: float | None = None, selection: str = "X", market: Market = Market.MATCH_ODDS) -> SlipLeg:
    return SlipLeg(
        match_id=match_id,
        date=NOW.date(),
        kickoff=NOW + timedelta(hours=4),
        competition=Girone.A,
        section="Serie C",
        home_team=f"{match_id}-casa",
        away_team=f"{match_id}-ospite",
        market=market,
        selection=selection,
        p_model=p,
        fair_odds=1.0 / p,
        book_odds=odds,
        bookmaker=None if odds is None else "BookA",
        ev=None if odds is None else p * odds - 1.0,
        book_margin=margin,
    )


# ----------------------------------------------------------------------------- structure and defaults
def test_three_slips_in_order_with_default_sizes() -> None:
    preds = [prediction(f"m{i:02d}", px=0.20 + i * 0.005, gg=0.40 + i * 0.01) for i in range(14)]
    slips = build_slips(preds, now=NOW)
    assert [s.kind for s in slips] == ["X", "GG", "MISTA"]
    assert [s.n_legs for s in slips] == [DEFAULT_N_X, DEFAULT_N_GG, DEFAULT_N_MIXED] == [5, 10, 6]
    assert [s.n_requested for s in slips] == [5, 10, 6]
    assert all(s.is_complete and s.coverage_note == "" for s in slips)
    assert slips[0].title == "Schedina X · pareggi"
    assert slips[1].title == "Schedina GG · goal"
    assert slips[2].title == "Schedina mista · 1 / 2 / GG / Over 2.5"
    assert DEFAULT_MIXED_MARKETS == (("1X2", "1"), ("1X2", "2"), ("GG/NG", "GG"), ("OU2.5", "Over 2.5"))
    for slip in slips:
        assert len(set(ids(slip))) == slip.n_legs


def test_x_slip_ranks_by_draw_probability() -> None:
    draws = {"a": 0.26, "b": 0.33, "c": 0.29, "d": 0.31, "e": 0.24, "f": 0.35, "g": 0.27}
    preds = [prediction(mid, px=px) for mid, px in draws.items()]
    slip = build_slips(preds, now=NOW)[0]
    assert ids(slip) == ["f", "b", "d", "c", "g"]
    assert [leg.selection for leg in slip.legs] == ["X"] * 5
    assert all(leg.market is Market.MATCH_ODDS for leg in slip.legs)
    assert [leg.p_model for leg in slip.legs] == sorted((leg.p_model for leg in slip.legs), reverse=True)


def test_gg_slip_ranks_by_gg_probability_and_shrinks_when_few_matches() -> None:
    gg = {"a": 0.52, "b": 0.61, "c": 0.47, "d": 0.58}
    slip = build_slips([prediction(mid, gg=p) for mid, p in gg.items()], now=NOW)[1]
    assert ids(slip) == ["b", "d", "a", "c"]
    assert slip.n_legs == 4 and slip.n_requested == 10
    assert not slip.is_complete
    assert slip.coverage_note == "Solo 4 partite idonee su 10 richieste."
    assert all(leg.market is Market.BTTS and leg.selection == "GG" for leg in slip.legs)


def test_ties_break_on_book_edge_then_kickoff_then_match_id() -> None:
    preds = {
        "b": prediction("b", px=0.30, hours=5),
        "a": prediction("a", px=0.30, hours=5),
        "c": prediction("c", px=0.30, hours=2),
        "d": prediction("d", px=0.30, hours=8),
        "e": prediction("e", px=0.30, hours=1),
    }
    opps = [
        opportunity(preds["d"], Market.MATCH_ODDS, "X", 3.6),  # EV +8%
        opportunity(preds["e"], Market.MATCH_ODDS, "X", 3.2),  # EV -4%, still quoted
    ]
    slip = build_slips(list(preds.values()), now=NOW, opportunities=opps)[0]
    assert ids(slip) == ["d", "e", "c", "a", "b"]
    assert slip.legs[0].ev == pytest.approx(0.30 * 3.6 - 1.0)
    assert slip.legs[1].ev == pytest.approx(0.30 * 3.2 - 1.0)
    assert slip.legs[2].ev is None


def test_ties_in_the_mixed_slip_follow_the_same_rule() -> None:
    preds = [
        prediction("z", p1=0.60, px=0.20, hours=3),
        prediction("y", p1=0.60, px=0.20, hours=3),
        prediction("x", p1=0.60, px=0.20, hours=6),
        prediction("w", p1=0.60, px=0.20, hours=9),
    ]
    opps = [opportunity(preds[3], Market.MATCH_ODDS, "1", 1.80)]  # EV +8%
    mixed = build_slips(preds, now=NOW, n_mixed=3)[2]
    assert ids(mixed) == ["y", "z", "x"]
    mixed = build_slips(preds, now=NOW, n_mixed=3, opportunities=opps)[2]
    assert ids(mixed) == ["w", "y", "z"]
    assert [leg.selection for leg in mixed.legs] == ["1", "1", "1"]


# ----------------------------------------------------------------------------- exclusions
def test_kicked_off_and_in_play_fixtures_are_excluded() -> None:
    preds = [
        prediction("at_now", px=0.40, hours=0),
        prediction("in_play", px=0.39, hours=-1 / 60),
        prediction("played_yesterday", px=0.38, hours=-24),
        prediction("day_only_yesterday", px=0.37, hours=None, day=NOW.date() - timedelta(days=1)),
        prediction("day_only_today", px=0.30, hours=None),
        prediction("in_one_minute", px=0.29, hours=1 / 60),
    ]
    slips = build_slips(preds, now=NOW)
    for slip in slips:
        assert set(ids(slip)) <= {"day_only_today", "in_one_minute"}
    assert ids(slips[0]) == ["day_only_today", "in_one_minute"]
    # a naive ``now`` is read as UTC
    assert ids(build_slips(preds, now=NOW.replace(tzinfo=None))[0]) == ["day_only_today", "in_one_minute"]


def test_now_none_skips_the_time_check_but_not_postponements() -> None:
    preds = [prediction("past", px=0.40, hours=-30), prediction("rinviata", px=0.45, hours=-30, status="postponed"), prediction("next", px=0.30)]
    assert ids(build_slips(preds, now=None)[0]) == ["past", "next"]


def test_postponed_fixtures_are_excluded() -> None:
    preds = [prediction("rinviata", px=0.45, gg=0.70, o25=0.75, status="postponed"), prediction("ok", px=0.30)]
    for slip in build_slips(preds, now=NOW):
        assert "rinviata" not in ids(slip)


def test_duplicated_match_keeps_its_first_prediction() -> None:
    preds = [prediction("dup", px=0.31), prediction("other", px=0.30), prediction("dup", px=0.45)]
    slip = build_slips(preds, now=NOW)[0]
    assert ids(slip) == ["dup", "other"]
    assert slip.legs[0].p_model == pytest.approx(0.31)


def test_slip_rejects_duplicated_legs_and_unknown_kinds() -> None:
    with pytest.raises(ValueError, match="only once per slip"):
        Slip(kind="X", title="t", legs=(leg("a", 0.3), leg("a", 0.3)))
    with pytest.raises(ValueError, match="slip kind"):
        Slip(kind="1X", title="t", legs=())  # type: ignore[arg-type]


# ----------------------------------------------------------------------------- mixed slip
def test_mixed_slip_takes_the_best_sign_of_each_match_then_the_most_probable() -> None:
    preds = [
        prediction("home", p1=0.55, px=0.25, gg=0.40, o25=0.40),
        prediction("away", p1=0.15, px=0.25, p2=0.60, gg=0.45, o25=0.48),
        prediction("goals", p1=0.40, px=0.25, gg=0.65, o25=0.60),
        prediction("over", p1=0.45, px=0.20, gg=0.62, o25=0.70, o15=0.85),
        prediction("draw", p1=0.30, px=0.50, p2=0.20, gg=0.35, o25=0.30),
    ]
    mixed = build_slips(preds, now=NOW, n_mixed=4)[2]
    assert [(leg.match_id, leg.market, leg.selection) for leg in mixed.legs] == [
        ("over", Market.OVER_UNDER_25, "Over 2.5"),
        ("goals", Market.BTTS, "GG"),
        ("away", Market.MATCH_ODDS, "2"),
        ("home", Market.MATCH_ODDS, "1"),
    ]
    # X is not a sign of the mixed slip: the "draw" match plays its best other sign (GG 35%) and ranks last
    mixed = build_slips(preds, now=NOW, n_mixed=5)[2]
    assert (mixed.legs[-1].match_id, mixed.legs[-1].selection) == ("draw", "GG")
    assert mixed.legs[-1].p_model == pytest.approx(0.35)


def test_mixed_sign_tie_goes_to_the_sign_listed_first() -> None:
    preds = [prediction(f"m{i}", p1=0.45, px=0.25, gg=0.45, o25=0.40) for i in range(3)]
    mixed = build_slips(preds, now=NOW, n_mixed=3)[2]
    assert {leg.selection for leg in mixed.legs} == {"1"}
    reordered = (("GG/NG", "GG"), ("1X2", "1"), ("1X2", "2"), ("OU2.5", "Over 2.5"))
    mixed = build_slips(preds, now=NOW, n_mixed=3, mixed_markets=reordered)[2]
    assert {leg.selection for leg in mixed.legs} == {"GG"}
    assert mixed.title == "Schedina mista · GG / 1 / 2 / Over 2.5"


def test_mixed_leg_note_names_the_runner_up_sign() -> None:
    preds = [prediction("m", p1=0.30, px=0.25, gg=0.66, o25=0.58)]
    leg_ = build_slips(preds, now=NOW, n_mixed=3)[2].legs[0]
    assert leg_.selection == "GG"
    assert leg_.note.endswith("alternativa: Over 2.5 58.0% (8.0 punti in meno)")


def test_mixed_runner_up_note_never_reads_as_more_likely_than_the_chosen_sign() -> None:
    """Regression: GG 63.9 was shown next to 'alternativa: Over 2.5 64%' (0-decimal note vs 1-decimal column)."""
    preds = [prediction("m", p1=0.30, px=0.25, gg=0.639, o25=0.636)]
    mixed = build_slips(preds, now=NOW, n_mixed=3)[2]
    leg_ = mixed.legs[0]
    assert (leg_.selection, leg_.p_model) == ("GG", pytest.approx(0.639))
    assert leg_.note.endswith("alternativa: Over 2.5 63.6% (0.3 punti in meno)")
    assert "64%" not in leg_.note
    row = slips_frame([mixed]).iloc[0]
    shown_alternative = float(row["Nota"].split("alternativa: Over 2.5 ")[1].split("%")[0])
    assert shown_alternative <= row["P. Modello (%)"] == 63.9


def test_mixed_runner_up_note_for_ties_and_sub_decimal_gaps() -> None:
    tie = build_slips([prediction("t", p1=0.45, px=0.25, gg=0.45, o25=0.40)], now=NOW, n_mixed=3)[2].legs[0]
    assert tie.selection == "1"
    assert tie.note.endswith("alternativa: GG 45.0% (stessa probabilità)")
    close = build_slips([prediction("c", p1=0.30, px=0.25, gg=0.6390, o25=0.6389)], now=NOW, n_mixed=3)[2].legs[0]
    assert close.selection == "GG"
    assert close.note.endswith("alternativa: Over 2.5 63.9% (meno di 0.1 punti in meno)")


def test_custom_mixed_markets_and_their_validation() -> None:
    preds = [prediction("m1", px=0.34, o15=0.80), prediction("m2", px=0.36, o15=0.30)]
    mixed = build_slips(preds, now=NOW, n_mixed=3, mixed_markets=((Market.MATCH_ODDS, "X"), ("OU1.5", "Over 1.5")))[2]
    assert [(leg.match_id, leg.selection) for leg in mixed.legs] == [("m1", "Over 1.5"), ("m2", "X")]
    # market names and values are both accepted, duplicates collapse
    mixed = build_slips(preds, now=NOW, n_mixed=3, mixed_markets=(("BTTS", "GG"), ("GG/NG", "GG")))[2]
    assert mixed.title == "Schedina mista · GG"
    with pytest.raises(ValueError, match="not a selection"):
        build_slips(preds, now=NOW, mixed_markets=(("1X2", "GG"),))
    with pytest.raises(ValueError, match="unknown market"):
        build_slips(preds, now=NOW, mixed_markets=(("Corner", "Over 9.5"),))
    with pytest.raises(ValueError, match="at least one"):
        build_slips(preds, now=NOW, mixed_markets=())
    with pytest.raises(ValueError, match="pairs"):
        build_slips(preds, now=NOW, mixed_markets=("1",))  # type: ignore[arg-type]


def test_slip_lengths_are_validated() -> None:
    preds = [prediction("m", px=0.3)]
    for bad in (MIXED_MIN_LEGS - 1, MIXED_MAX_LEGS + 1):
        with pytest.raises(ValueError, match="n_mixed"):
            build_slips(preds, now=NOW, n_mixed=bad)
    with pytest.raises(ValueError, match="n_x"):
        build_slips(preds, now=NOW, n_x=-1)
    with pytest.raises(TypeError, match="n_gg"):
        build_slips(preds, now=NOW, n_gg=True)
    with pytest.raises(TypeError, match="n_x"):
        build_slips(preds, now=NOW, n_x=2.5)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="min_probability"):
        build_slips(preds, now=NOW, min_probability=1.5)
    slips = build_slips(preds, now=NOW, n_x=0, n_mixed=MIXED_MAX_LEGS)
    assert slips[0].is_empty and slips[0].is_complete
    assert slips[2].n_requested == MIXED_MAX_LEGS


# ----------------------------------------------------------------------------- filters
def test_min_probability_drops_weak_legs() -> None:
    preds = [prediction("a", px=0.33, gg=0.62), prediction("b", px=0.27, gg=0.55), prediction("c", px=0.31, gg=0.45)]
    slips = by_kind(build_slips(preds, now=NOW, min_probability=0.30))
    assert ids(slips["X"]) == ["a", "c"]
    assert ids(slips["GG"]) == ["a", "b", "c"]
    assert all(leg.p_model >= 0.30 for s in slips.values() for leg in s.legs)
    slips = by_kind(build_slips(preds, now=NOW, min_probability=0.50))
    assert slips["X"].is_empty
    assert ids(slips["GG"]) == ["a", "b"]
    assert [(leg.match_id, leg.selection) for leg in slips["MISTA"].legs] == [("a", "GG"), ("b", "GG")]


def test_only_quoted_keeps_priced_legs_and_picks_the_best_quoted_sign() -> None:
    a = prediction("a", p1=0.30, px=0.33, gg=0.62, o25=0.55)
    b = prediction("b", p1=0.35, px=0.30, gg=0.50, o25=0.40)
    c = prediction("c", p1=0.50, px=0.29, gg=0.48, o25=0.42)
    opps = [
        opportunity(b, Market.MATCH_ODDS, "X", 3.3),
        opportunity(a, Market.OVER_UNDER_25, "Over 2.5", 1.9),  # GG (62%) is a's best sign, but unquoted
        opportunity(c, Market.MATCH_ODDS, "1", 2.1),
    ]
    slips = by_kind(build_slips([a, b, c], now=NOW, opportunities=opps, only_quoted=True, n_mixed=3))
    assert ids(slips["X"]) == ["b"]
    assert slips["GG"].is_empty
    assert [(leg.match_id, leg.selection) for leg in slips["MISTA"].legs] == [("a", "Over 2.5"), ("c", "1")]
    assert all(s.is_quoted for s in (slips["X"], slips["MISTA"]))
    # without the filter a's best sign is GG and the slip is no longer fully quoted
    mixed = build_slips([a, b, c], now=NOW, opportunities=opps, n_mixed=3)[2]
    assert (mixed.legs[0].match_id, mixed.legs[0].selection) == ("a", "GG")
    assert mixed.book_odds is None


def test_sections_restrict_the_matches() -> None:
    preds = [
        prediction("c1", "A", px=0.30),
        prediction("c2", "C", px=0.31),
        prediction("b1", "SB", px=0.32),
        prediction("d1", "DA", px=0.33),
        prediction("d2", "DI", px=0.34),
    ]
    assert ids(build_slips(preds, now=NOW, sections=["Serie D"])[0]) == ["d2", "d1"]
    assert ids(build_slips(preds, now=NOW, sections=["serie c", " Serie  B "])[0]) == ["b1", "c2", "c1"]
    assert ids(build_slips(preds, now=NOW, sections="Serie B")[0]) == ["b1"]
    assert all(s.is_empty for s in build_slips(preds, now=NOW, sections=[]))
    assert {leg.section for leg in build_slips(preds, now=NOW)[0].legs} == {"Serie C", "Serie B", "Serie D"}


def test_matchday_restricts_to_its_matches_still_to_play() -> None:
    preds = [prediction("in", px=0.30), prediction("out", px=0.40)]
    matchday = Matchday(key="2026-11-08", label="Giornata", start=NOW.date(), end=NOW.date(), rounds={Girone.A: 13}, fixture_ids=frozenset({"in"}))
    assert ids(build_slips(preds, now=NOW, matchday=matchday)[0]) == ["in"]


def test_no_eligible_match_gives_three_empty_slips() -> None:
    slips = build_slips([], now=NOW)
    assert [s.kind for s in slips] == ["X", "GG", "MISTA"]
    for slip in slips:
        assert slip.is_empty and not slip.is_complete
        assert math.isnan(slip.probability) and math.isnan(slip.fair_odds)
        assert slip.book_odds is None and slip.ev is None
        assert slip.expected_winning_legs == 0.0
        assert slip.margin_note == ""
        assert slip.coverage_note == "Nessuna partita idonea."
        assert suggested_stake(slip, bankroll_units=100) == 0.0


# ----------------------------------------------------------------------------- maths and prices
def test_slip_probability_fair_odds_and_expected_legs() -> None:
    slip = Slip(kind="X", title="t", legs=(leg("a", 0.30), leg("b", 0.25), leg("c", 0.32)), n_requested=3)
    assert slip.probability == pytest.approx(0.30 * 0.25 * 0.32)
    assert slip.fair_odds == pytest.approx(1.0 / (0.30 * 0.25 * 0.32))
    assert slip.expected_winning_legs == pytest.approx(0.87)
    assert slip.book_odds is None and slip.ev is None and not slip.is_quoted
    built = build_slips([prediction("a", px=0.30), prediction("b", px=0.25)], now=NOW)[0]
    assert built.probability == pytest.approx(0.075)
    for leg_ in built.legs:
        assert leg_.fair_odds == pytest.approx(1.0 / leg_.p_model)
    assert "indipendenza" in INDEPENDENCE_NOTE


def test_book_odds_only_when_every_leg_is_quoted() -> None:
    a, b = prediction("a", px=0.32), prediction("b", px=0.30)
    partial = build_slips([a, b], now=NOW, opportunities=[opportunity(a, Market.MATCH_ODDS, "X", 3.4)])[0]
    assert partial.legs[0].book_odds == 3.4 and partial.legs[1].book_odds is None
    assert partial.book_odds is None and partial.ev is None and not partial.is_quoted
    full = build_slips([a, b], now=NOW, opportunities=[opportunity(a, Market.MATCH_ODDS, "X", 3.4), opportunity(b, Market.MATCH_ODDS, "X", 3.1)])[0]
    assert full.book_odds == pytest.approx(3.4 * 3.1)
    assert full.ev == pytest.approx(0.32 * 0.30 * 3.4 * 3.1 - 1.0)
    assert [leg_.ev for leg_ in full.legs] == [pytest.approx(0.32 * 3.4 - 1.0), pytest.approx(0.30 * 3.1 - 1.0)]


def test_best_price_is_the_highest_across_bookmakers() -> None:
    a = prediction("a", px=0.32)
    opps = [opportunity(a, Market.MATCH_ODDS, "X", 3.2, "BookA"), opportunity(a, Market.MATCH_ODDS, "X", 3.4, "BookB")]
    quote = OddsQuote("a", "BookC", Market.MATCH_ODDS, {"1": 2.4, "X": 3.5, "2": 3.0})
    legs = build_slips([a], now=NOW, opportunities=opps, quotes=[quote])[0].legs
    assert (legs[0].book_odds, legs[0].bookmaker) == (3.5, "BookC")
    assert legs[0].book_margin == pytest.approx(1 / 2.4 + 1 / 3.5 + 1 / 3.0 - 1.0)
    legs = build_slips([a], now=NOW, opportunities=opps)[0].legs
    assert (legs[0].book_odds, legs[0].bookmaker) == (3.4, "BookB")
    assert legs[0].book_margin == pytest.approx(0.05)
    # equal prices: the bookmaker first in alphabetical order, whatever the input order
    tie = [opportunity(a, Market.MATCH_ODDS, "X", 3.4, "Zeta"), opportunity(a, Market.MATCH_ODDS, "X", 3.4, "Alfa")]
    assert build_slips([a], now=NOW, opportunities=tie)[0].legs[0].bookmaker == "Alfa"


def test_margin_note_compounds_the_bookmaker_margin() -> None:
    known = Slip(kind="GG", title="t", legs=tuple(leg(f"m{i}", 0.6, odds=1.6, margin=0.05, selection="GG", market=Market.BTTS) for i in range(4)))
    assert known.compound_margin == pytest.approx(1.0 - (1.0 / 1.05) ** 4)
    assert "margine composto è 17.7%" in known.margin_note
    assert "in media 5.0% per partita" in known.margin_note
    unknown = Slip(kind="X", title="t", legs=tuple(leg(f"m{i}", 0.3) for i in range(5)))
    assert unknown.compound_margin is None
    expected = 1.0 - (1.0 / (1.0 + TYPICAL_LEG_MARGIN)) ** 5
    assert f"{expected * 100:.1f}%" in unknown.margin_note == unknown.margin_note
    assert "Le multiple moltiplicano il margine del bookmaker" in unknown.margin_note


# ----------------------------------------------------------------------------- adjustment (recalibration)
def test_adjust_changes_ranking_probabilities_and_fair_odds() -> None:
    calls: list[tuple[str | None, set[Market]]] = []

    def adjust(probs: Mapping[Market, Mapping[str, float]], *, family: str | None = None) -> dict[Any, dict[str, float]]:
        calls.append((family, set(probs)))
        out = {m.value: dict(s) for m, s in probs.items()}  # keys as market values are accepted
        if family == "Serie D":
            out["GG/NG"] = {"GG": 0.90, "NG": 0.10}
        return out

    preds = [prediction("c", "A", gg=0.70), prediction("d", "DA", gg=0.50)]
    gg = build_slips(preds, now=NOW, adjust=adjust)[1]
    assert ids(gg) == ["d", "c"]
    assert gg.legs[0].p_model == pytest.approx(0.90)
    assert gg.legs[0].fair_odds == pytest.approx(1 / 0.90)
    assert sorted(f for f, _ in calls) == ["Serie C", "Serie D"]
    assert all(markets == set(Market) for _, markets in calls)


def test_adjust_returning_invalid_probabilities_raises() -> None:
    def broken(probs: Mapping[Market, Mapping[str, float]], *, family: str | None = None) -> dict[Market, dict[str, float]]:
        return {Market.MATCH_ODDS: {"1": 1.4, "X": -0.2, "2": -0.2}}

    with pytest.raises(ValueError, match="probability must be in"):
        build_slips([prediction("a")], now=NOW, adjust=broken)


def test_identity_recalibration_leaves_the_slips_unchanged() -> None:
    calibration = pytest.importorskip("src.engine.calibration")
    preds = [prediction(f"m{i}", "DA" if i % 2 else "A", px=0.24 + i * 0.01, gg=0.42 + i * 0.02, o25=0.40 + i * 0.015) for i in range(8)]
    model = calibration.RecalibrationModel.identity()
    assert build_slips(preds, now=NOW, adjust=model.apply) == build_slips(preds, now=NOW)


# ----------------------------------------------------------------------------- stake
def test_suggested_stake_is_capped_fractional_kelly() -> None:
    slip = Slip(kind="X", title="t", legs=(leg("a", 0.6, odds=2.0), leg("b", 0.6, odds=2.0)))
    p, odds = 0.36, 4.0
    full_kelly = (p * (odds - 1.0) - (1.0 - p)) / (odds - 1.0)
    assert slip.ev == pytest.approx(p * odds - 1.0)
    assert suggested_stake(slip, bankroll_units=100) == pytest.approx(1.0)  # 1.83 capped at 1 unit
    assert suggested_stake(slip, bankroll_units=100, cap_units=5.0) == pytest.approx(round(full_kelly * 0.125 * 100, 2))
    assert suggested_stake(slip, bankroll_units=100, fraction=0.25, cap_units=10.0) == pytest.approx(round(full_kelly * 0.25 * 100, 2))
    assert suggested_stake(slip, bankroll_units=100, cap_units=5.0, floor_units=2.0) == 0.0
    losing = Slip(kind="X", title="t", legs=(leg("a", 0.3, odds=3.0), leg("b", 0.3, odds=3.0)))
    assert losing.ev is not None and losing.ev < 0
    assert suggested_stake(losing, bankroll_units=100) == 0.0
    unquoted = Slip(kind="X", title="t", legs=(leg("a", 0.6, odds=2.0), leg("b", 0.6)))
    assert suggested_stake(unquoted, bankroll_units=100) == 0.0
    with pytest.raises(ValueError):
        suggested_stake(slip, bankroll_units=100, fraction=1.5)
    with pytest.raises(ValueError):
        suggested_stake(slip, bankroll_units=0)


def test_format_stake() -> None:
    assert format_stake(0.0) == STAKE_NOT_ADVISED == "non consigliata"
    assert format_stake(float("nan")) == STAKE_NOT_ADVISED
    assert format_stake(0.25) == "0.25 unità"


# ----------------------------------------------------------------------------- frames for display
def test_slips_frame_columns_and_values() -> None:
    a, b = prediction("a", px=0.32, lam=1.0, mu=1.05), prediction("b", px=0.30, hours=None)
    slips = build_slips([a, b], now=NOW, opportunities=[opportunity(a, Market.MATCH_ODDS, "X", 3.4, "BookB")])
    frame = slips_frame(slips)
    assert tuple(frame.columns) == SLIPS_COLUMNS
    assert len(frame) == sum(s.n_legs for s in slips)
    first = frame.iloc[0]
    assert first["Schedina"] == "Schedina X · pareggi"
    assert first["#"] == 1 and frame.iloc[1]["#"] == 2
    assert first["Competizione"] == "Girone A"
    assert first["Data e ora"] == "dom 08/11 15:00"  # 14:00 UTC
    assert frame.iloc[1]["Data e ora"] == "dom 08/11"
    assert first["Partita"] == "a-casa - a-ospite"
    assert first["Esito"] == "X"
    assert first["P. Modello (%)"] == 32.0
    assert first["Quota Fair"] == round(1 / 0.32, 2)
    assert first["Quota Book"] == 3.4 and first["Bookmaker"] == "BookB"
    assert first["Nota"] == f"{DRAW_NOTE_CLOSE_RATES}, {DRAW_NOTE_FEW_GOALS}"
    assert math.isnan(frame.iloc[1]["Quota Book"]) and pd.isna(frame.iloc[1]["Bookmaker"])
    empty = slips_frame([])
    assert tuple(empty.columns) == SLIPS_COLUMNS and empty.empty


def test_slips_summary_columns_and_values() -> None:
    winning = Slip(kind="X", title="Schedina X · pareggi", legs=(leg("a", 0.6, odds=2.0), leg("b", 0.6, odds=2.0)), n_requested=2)
    plain = Slip(kind="GG", title="Schedina GG · goal", legs=(leg("c", 0.55, selection="GG", market=Market.BTTS),), n_requested=10)
    summary = slips_summary([winning, plain])
    assert tuple(summary.columns) == SLIPS_SUMMARY_COLUMNS
    row = summary.iloc[0]
    assert row["Schedina"] == "Schedina X · pareggi"
    assert row["Partite"] == 2
    assert row["Probabilità (%)"] == 36.0
    assert row["Quota fair"] == round(1 / 0.36, 2)
    assert row["Quota book"] == 4.0
    assert row["EV (%)"] == 44.0
    assert row["Partite vincenti attese"] == 1.2
    assert row["Stake suggerito"] == "1.00 unità"
    other = summary.iloc[1]
    assert math.isnan(other["Quota book"]) and math.isnan(other["EV (%)"])
    assert other["Stake suggerito"] == STAKE_NOT_ADVISED
    assert slips_summary([winning], cap_units=5.0).iloc[0]["Stake suggerito"] == "1.83 unità"
    assert tuple(slips_summary([]).columns) == SLIPS_SUMMARY_COLUMNS


def test_leg_validation_and_aliases() -> None:
    with pytest.raises(ValueError, match="not a selection"):
        leg("a", 0.3, selection="GG")
    with pytest.raises(ValueError, match="p_model"):
        leg("a", 1.2)
    with pytest.raises(ValueError, match="book_odds"):
        leg("a", 0.3, odds=1.0)
    item = leg("a", 0.3, odds=3.5)
    assert item.girone is Girone.A and item.label == "a-casa - a-ospite" and item.is_quoted
    assert item.when == "dom 08/11 15:00"


# ----------------------------------------------------------------------------- frame-based builder
def mixed_predictions() -> list[MatchPrediction]:
    """A matchday with every case: kick-off times, a day-only fixture, a postponed and a kicked-off match, three sections."""
    rng = np.random.default_rng(7)
    preds: list[MatchPrediction] = []
    gironi = ["A", "B", "C", "SB", "DA", "DB", "DC"]
    for i in range(24):
        p1 = float(rng.uniform(0.2, 0.6))
        px = float(rng.uniform(0.2, 0.34))
        gg = float(rng.uniform(0.35, 0.7))
        o25 = float(rng.uniform(0.35, 0.7))
        hours: float | None = float(rng.integers(1, 30))
        if i == 3:
            hours = None
        if i == 5:
            hours = -2.0
        preds.append(
            prediction(
                f"m{i:02d}",
                gironi[i % len(gironi)],
                hours=hours,
                status="postponed" if i == 7 else "scheduled",
                lam=float(rng.uniform(0.8, 2.0)),
                mu=float(rng.uniform(0.7, 1.7)),
                p1=p1,
                px=px,
                gg=gg,
                o25=o25,
                o15=min(0.95, o25 + 0.2),
                o35=o25 * 0.5,
            )
        )
    preds[10] = prediction("m02", "A", px=0.99 * 0.34)  # a duplicate of m02: the first one wins in both builders
    preds[12] = prediction("m12", "DA", px=0.30, gg=0.55)
    preds[13] = prediction("m13", "DB", px=0.30, gg=0.55)  # exact ties
    return preds


def mixed_opportunities(preds: list[MatchPrediction]) -> list[ValueOpportunity]:
    out: list[ValueOpportunity] = []
    for i, p in enumerate(preds):
        if i % 3 == 0:
            out.append(opportunity(p, Market.MATCH_ODDS, "X", 3.3 + i * 0.01, "BookA"))
            out.append(opportunity(p, Market.BTTS, "GG", 1.85, "BookB"))
        if i % 4 == 0:
            out.append(opportunity(p, Market.OVER_UNDER_25, "Over 2.5", 1.95, "BookC", margin=0.06))
    return out


@pytest.mark.parametrize(
    "options",
    [
        {},
        {"n_x": 4, "n_gg": 7, "n_mixed": 8, "min_probability": 0.3},
        {"only_quoted": True, "n_mixed": 3},
        {"sections": ["Serie D", "Serie B"]},
        {"mixed_markets": (("GG/NG", "GG"), ("OU2.5", "Over 2.5"), ("OU1.5", "Over 1.5"))},
    ],
)
def test_frame_builder_equals_prediction_builder(options: dict[str, Any]) -> None:
    preds = mixed_predictions()
    opps = mixed_opportunities(preds)
    quotes = [OddsQuote(preds[1].match_id, "BookQ", Market.MATCH_ODDS, {"1": 2.0, "X": 3.9, "2": 3.6})]

    def adjust(probs: Mapping[Market, Mapping[str, float]], *, family: str | None = None) -> dict[Market, dict[str, float]]:
        shift = 0.02 if family == "Serie C" else -0.01
        out = {m: dict(s) for m, s in probs.items()}
        gg = min(max(out[Market.BTTS]["GG"] + shift, 0.0), 1.0)
        out[Market.BTTS] = {"GG": gg, "NG": 1.0 - gg}
        return out

    expected = build_slips(preds, now=NOW, opportunities=opps, quotes=quotes, adjust=adjust, **options)
    frame = predictions_frame(preds)
    got = slips_from_frame(frame, now=NOW, opportunities=opps, quotes=quotes, adjust=adjust, **options)
    assert got == expected
    assert any(not s.is_empty for s in got)
    # the frame keeps the duplicate, the postponed and the kicked-off rows: the builder excludes them
    assert {"m05", "m07"}.isdisjoint({leg_.match_id for s in got for leg_ in s.legs})


def test_predictions_frame_columns_and_dtypes() -> None:
    preds = [prediction("a", "DA", hours=3), prediction("b", hours=None)]
    frame = predictions_frame(preds)
    assert tuple(frame.columns) == PREDICTIONS_FRAME_COLUMNS
    assert str(frame["date"].dtype) == "datetime64[ns]"
    assert str(frame["kickoff"].dtype) == "datetime64[ns, UTC]"
    assert frame.loc[0, "kickoff"] == pd.Timestamp(NOW + timedelta(hours=3))
    assert pd.isna(frame.loc[1, "kickoff"])
    assert list(frame["section"]) == ["Serie D", "Serie C"]
    assert list(frame["family"]) == ["Serie D", "Serie C"]
    assert frame.loc[0, "p_O25"] == pytest.approx(0.45)
    assert tuple(predictions_frame([]).columns) == PREDICTIONS_FRAME_COLUMNS


def backtest_style_frame() -> pd.DataFrame:
    """The pre-match columns the backtest hands to a slip builder: day-only dates, no kick-off, no status."""
    rows = []
    for i, (girone, px, gg, p1, o25) in enumerate(
        [("A", 0.31, 0.52, 0.40, 0.48), ("SB", 0.29, 0.61, 0.38, 0.55), ("DA", 0.33, 0.47, 0.42, 0.41), ("DB", 0.27, 0.58, 0.30, 0.62)]
    ):
        g = Girone.parse(girone)
        rows.append(
            {
                "match_id": f"bt{i}",
                "date": pd.Timestamp(2026, 10, 25),
                "season": "2026/27",
                "matchday": "2026-10-23",
                "girone": girone,
                "family": g.family,
                "section": g.section,
                "round": 10,
                "home_team": f"H{i}",
                "away_team": f"A{i}",
                "lam": 1.0 + i * 0.3,
                "mu": 0.95 + i * 0.2,
                "rho": -0.05,
                "p_1": p1,
                "p_X": px,
                "p_2": 1.0 - p1 - px,
                "p_O15": 0.75,
                "p_U15": 0.25,
                "p_O25": o25,
                "p_U25": 1.0 - o25,
                "p_O35": 0.2,
                "p_U35": 0.8,
                "p_GG": gg,
                "p_NG": 1.0 - gg,
            }
        )
    return pd.DataFrame(rows)


def test_frame_builder_on_a_backtest_style_frame() -> None:
    frame = backtest_style_frame()
    slips = slips_from_frame(frame)  # the frame alone: every row is pre-match
    x, gg, mixed = slips
    assert ids(x) == ["bt2", "bt0", "bt1", "bt3"]
    assert ids(gg) == ["bt1", "bt3", "bt0", "bt2"]
    assert [(leg_.match_id, leg_.selection) for leg_ in mixed.legs] == [("bt3", "Over 2.5"), ("bt1", "GG"), ("bt0", "GG"), ("bt2", "GG")]
    assert all(leg_.kickoff is None and leg_.date == date(2026, 10, 25) for leg_ in x.legs)
    assert x.legs[0].competition is Girone.DA and x.legs[0].section == "Serie D"
    assert BTTS_NOTE_HIGH_XG in gg.legs[1].note and BTTS_NOTE_BOTH_ATTACKS in gg.legs[1].note
    # with a reference instant after the matches every row has kicked off
    assert all(s.is_empty for s in slips_from_frame(frame, now=NOW))


def test_frame_price_columns_merge_with_opportunities() -> None:
    frame = backtest_style_frame().assign(odds_X=[3.2, np.nan, 3.0, 3.6], bookmaker_X=["BookF", None, "BookF", "BookF"], overround_1X2=[0.04, np.nan, 0.05, 0.06])
    x = slips_from_frame(frame)[0]
    priced = {leg_.match_id: (leg_.book_odds, leg_.bookmaker, leg_.book_margin) for leg_ in x.legs}
    assert priced["bt0"] == (3.2, "BookF", pytest.approx(0.04))
    assert priced["bt1"] == (None, None, None)
    pred = prediction("bt2", "DA", px=0.33)
    better = opportunity(pred, Market.MATCH_ODDS, "X", 3.25, "BookO")
    x = slips_from_frame(frame, opportunities=[better])[0]
    assert {leg_.match_id: leg_.bookmaker for leg_ in x.legs}["bt2"] == "BookO"


def test_frame_builder_validation() -> None:
    frame = backtest_style_frame()
    with pytest.raises(ValueError, match=r"missing columns \['p_GG'\]"):
        slips_from_frame(frame.drop(columns=["p_GG"]))
    # not needed when the slips asking for it are empty and the mixed slip does not use it
    slips = slips_from_frame(frame.drop(columns=["p_GG"]), n_gg=0, mixed_markets=(("1X2", "1"), ("1X2", "2")))
    assert slips[1].is_empty and not slips[0].is_empty
    unknown = pd.concat([frame, frame.iloc[[0]].assign(match_id="zz", girone="Atlantide")], ignore_index=True)
    assert "zz" not in {leg_.match_id for s in slips_from_frame(unknown) for leg_ in s.legs}
    assert all(s.is_empty for s in slips_from_frame(frame.iloc[0:0]))
    with pytest.raises(ValueError, match="probability must be in"):
        slips_from_frame(frame.assign(p_X=1.7))
    # any spelling of 'postponed' excludes the row, any other status reads as scheduled
    statuses = frame.assign(status=["Postponed", "complete", None, "scheduled"])
    assert set(ids(slips_from_frame(statuses)[0])) == {"bt1", "bt2", "bt3"}


def test_make_slip_builder_binds_and_validates_options() -> None:
    with pytest.raises(ValueError, match="n_mixed"):
        make_slip_builder(n_mixed=1)
    builder = make_slip_builder(n_x=2, n_mixed=3, sections=(s for s in ["Serie D"]))
    frame = backtest_style_frame()
    first, second = builder(frame), builder(frame)
    assert first == second == slips_from_frame(frame, n_x=2, n_mixed=3, sections=["Serie D"])
    assert ids(first[0]) == ["bt2", "bt3"]


def test_frame_suffixes_match_the_backtest() -> None:
    backtest = pytest.importorskip("src.engine.backtest")
    assert FRAME_SELECTION_SUFFIX == backtest.SELECTION_SUFFIX
    assert FRAME_MARKET_SUFFIX == backtest._MARKET_SUFFIX


def test_backtest_track_record_replays_the_slips() -> None:
    backtest = pytest.importorskip("src.engine.backtest")
    frame = backtest_style_frame().assign(
        block="2026-10-23",
        home_goals=[1, 2, 0, 3],
        away_goals=[1, 1, 0, 2],
        res_1X2=["X", "1", "X", "1"],
        gg=[1, 1, 0, 1],
        o15=[1, 1, 0, 1],
        o25=[0, 1, 0, 1],
        o35=[0, 0, 0, 1],
    )
    record = backtest.slips_track_record(frame, slip_builder=slips_from_frame)
    assert list(record.slips["slip"]) == ["X", "GG", "MISTA"]
    assert list(record.slips["n_legs"]) == [4, 4, 4]
    by_slip = record.slips.set_index("slip")
    assert by_slip.loc["X", "legs_won"] == 2 and not by_slip.loc["X", "won"]
    assert by_slip.loc["X", "p_slip"] == pytest.approx(0.31 * 0.29 * 0.33 * 0.27)
    # mixed legs: bt3 Over 2.5 (won), bt1 GG (won), bt0 GG (won), bt2 GG (lost: 0-0)
    assert by_slip.loc["MISTA", "legs_won"] == 3
