"""Tests for ``src.engine.picks`` (matchdays, picks of the matchday) and ``src.dashboard.legend`` (glossary, README sync)."""
from __future__ import annotations

import re
from collections.abc import Sequence
from datetime import date
from itertools import pairwise
from pathlib import Path

import numpy as np
import pytest

from src.common.schemas import Fixture, Girone, Market, MatchPrediction, OddsQuote
from src.dashboard.legend import (
    LEGEND,
    LEGEND_TITLE,
    POOR_EXAMPLE_FLAGS,
    SYNTHETIC_REASON_NO_QUOTES,
    SYNTHETIC_REASON_NO_VALUE,
    LegendEntry,
    LegendSection,
    WorkedExample,
    legend_markdown,
    legend_terms,
    pick_example_row,
    worked_example_markdown,
)
from src.engine.model import DixonColesConfig, DixonColesEngine, dixon_coles_pmf
from src.engine.picks import (
    BTTS_NOTE_HIGH_XG,
    DRAW_NOTE_CLOSE_RATES,
    PICKS_COLUMNS,
    Matchday,
    Pick,
    btts_picks,
    draw_picks,
    fixtures_of,
    format_date_span,
    matchdays,
    next_matchday,
    picks_frame,
    rank_picks,
    week_block,
)
from src.engine.value import FLAG_LONGSHOT, FLAG_SUSPICIOUS_EDGE, FLAG_VALUE, KellyConfig, ValueEngine, ValueOpportunity
from src.simulation.generator import dixon_coles_matrix, market_probabilities, simulate_league

REPO_ROOT = Path(__file__).resolve().parents[1]


# ----------------------------------------------------------------------------- helpers
def fx(match_id: str, when: date, round_no: int, girone: str, home: str, away: str) -> Fixture:
    return Fixture(match_id, when, round_no, girone, home, away)


# November-December 2026 calendar used by the matchday tests (2026-11-06 is a Friday)
FRI6, SAT7, SUN8, MON9 = date(2026, 11, 6), date(2026, 11, 7), date(2026, 11, 8), date(2026, 11, 9)
TUE10, WED11, FRI13, SAT14, SUN15, MON16 = date(2026, 11, 10), date(2026, 11, 11), date(2026, 11, 13), date(2026, 11, 14), date(2026, 11, 15), date(2026, 11, 16)
TUE17, WED18, SAT21, SUN22, SAT28, SUN29 = date(2026, 11, 17), date(2026, 11, 18), date(2026, 11, 21), date(2026, 11, 22), date(2026, 11, 28), date(2026, 11, 29)
WED_DEC2, SAT_DEC5, WED_DEC23 = date(2026, 12, 2), date(2026, 12, 5), date(2026, 12, 23)


def round_fixtures(girone: str, round_no: int, dates: Sequence[date]) -> list[Fixture]:
    """One fixture per entry of ``dates`` (ten dates = a full round of a 20-team competition), ids ``<girone><round>_<i>``."""
    return [
        fx(f"{girone}{round_no}_{i}", when, round_no, girone, f"{girone}{round_no}-H{i}", f"{girone}{round_no}-A{i}")
        for i, when in enumerate(dates)
    ]


def synthetic_fixtures() -> list[Fixture]:
    """Four 20-team competitions on a realistic calendar: weekend rounds, a Serie B midweek round, a recovery and a straggler.

    Girone A: round 14 on Sat 7 / Sun 8 / Mon 9 Nov, 15 on Sun 15, 16 on Sun 22, plus one postponed round-12 match on
    Wed 23 Dec; Girone B: round 14 on Sun 8 with one match postponed to Wed 2 Dec, round 15 on Sun 15; Girone C: only
    round 14 (Sat 7 / Sun 8); Serie B: round 13 on Fri 6 / Sat 7, round 14 midweek on Tue 10 / Wed 11.
    """
    return [
        *round_fixtures("A", 14, [SAT7] * 3 + [SUN8] * 6 + [MON9]),
        *round_fixtures("A", 15, [SUN15] * 10),
        *round_fixtures("A", 16, [SUN22] * 10),
        fx("A12x", WED_DEC23, 12, "A", "Alpha", "Gamma"),
        *round_fixtures("B", 14, [SUN8] * 9 + [WED_DEC2]),
        *round_fixtures("B", 15, [SUN15] * 10),
        *round_fixtures("C", 14, [SAT7] * 4 + [SUN8] * 6),
        *round_fixtures("SB", 13, [FRI6] + [SAT7] * 9),
        *round_fixtures("SB", 14, [TUE10] * 4 + [WED11] * 6),
    ]


def make_prediction(fixture: Fixture, lam: float, mu: float, *, p_x: float | None = None, p_gg: float | None = None, fair_x: float | None = None) -> MatchPrediction:
    """A prediction from the rates (Dixon-Coles matrix), optionally overriding P(X)/P(GG) and the fair X odds for tie tests."""
    matrix = dixon_coles_matrix(lam, mu, -0.05)
    probs = market_probabilities(matrix)
    if p_x is not None:
        rest = 1.0 - p_x
        probs[Market.MATCH_ODDS] = {"1": rest * 0.6, "X": p_x, "2": rest * 0.4}
    if p_gg is not None:
        probs[Market.BTTS] = {"GG": p_gg, "NG": 1.0 - p_gg}
    fair = {m: {s: 1.0 / p for s, p in sels.items()} for m, sels in probs.items()}
    if fair_x is not None:
        fair[Market.MATCH_ODDS]["X"] = fair_x
    return MatchPrediction(fixture=fixture, home_rate=lam, away_rate=mu, rho=-0.05, matrix=matrix, probabilities=probs, fair_odds=fair)


def opportunity(pred: MatchPrediction, market: Market, selection: str, book_odds: float, bookmaker: str, flags: tuple[str, ...] = ()) -> ValueOpportunity:
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
        book_odds=book_odds,
        fair_odds=1.0 / p,
        p_model=p,
        p_implied_raw=1.0 / book_odds,
        p_implied_devig=1.0 / book_odds / 1.05,
        ev=p * book_odds - 1.0,
        edge=p - 1.0 / book_odds / 1.05,
        kelly_full=0.0,
        kelly_fraction=0.125,
        stake_units=0.0,
        value=p * book_odds - 1.0 >= 0.035,
        flags=flags,
    )


# ----------------------------------------------------------------------------- matchdays
def test_week_block_splits_the_week_into_weekend_and_midweek() -> None:
    for day in (FRI6, SAT7, SUN8, MON9):
        assert week_block(day) == (FRI6, MON9), day
    for day in (TUE10, WED11, date(2026, 11, 12)):
        assert week_block(day) == (TUE10, date(2026, 11, 12)), day
    assert week_block(FRI13) == (FRI13, MON16) and week_block(WED_DEC23) == (date(2026, 12, 22), date(2026, 12, 24))


def test_matchdays_group_the_rounds_of_the_same_week_block() -> None:
    days = matchdays(synthetic_fixtures())
    assert [m.key for m in days] == ["2026-11-06", "2026-11-10", "2026-11-15", "2026-11-22"]
    first, midweek, third, fourth = days
    # weekend block Fri 6 - Mon 9: A14, B14 (without its straggler), C14 and SB13 (Friday anticipo opens the span)
    assert first.label == "Giornata del 6-9 nov 2026 · Girone A g.14, Girone B g.14, Girone C g.14, Serie B g.13"
    assert first.rounds == {Girone.A: 14, Girone.B: 14, Girone.C: 14, Girone.SB: 13}
    assert first.n_fixtures == 39 and "B14_9" not in first.fixture_ids and (first.start, first.end) == (FRI6, MON9)
    assert first.competitions == (Girone.A, Girone.B, Girone.C, Girone.SB)
    # midweek block Tue 10 - Thu 12: only Serie B plays, the Serie C gironi are simply absent
    assert midweek.label == "Giornata del 10-11 nov 2026 · Serie B g.14" and midweek.rounds == {Girone.SB: 14} and midweek.n_fixtures == 10
    # weekend block Fri 13 - Mon 16: A15 and B15 (Girone C has no round left, Serie B neither)
    assert third.rounds == {Girone.A: 15, Girone.B: 15} and third.n_fixtures == 20 and third.label == "Giornata del 15 nov 2026 · Girone A g.15, Girone B g.15"
    # weekend block Fri 20 - Mon 23: A16, plus the leftovers attached to the nearest matchday by date: the Girone B
    # straggler of round 14 (Wed 2 Dec) and the round-12 recovery (Wed 23 Dec) extend its end
    assert fourth.rounds == {Girone.A: 16} and fourth.fixture_ids == frozenset({f"A16_{i}" for i in range(10)} | {"A12x", "B14_9"})
    assert (fourth.start, fourth.end) == (SUN22, WED_DEC23) and fourth.key == "2026-11-22"
    assert fourth.label == "Giornata del 22 nov-23 dic 2026 · Girone A g.16"
    # every fixture lands in exactly one matchday
    all_ids = [i for m in days for i in m.fixture_ids]
    assert sorted(all_ids) == sorted(f.match_id for f in synthetic_fixtures())
    # deterministic: same input, same result (also from a differently ordered input)
    assert matchdays(list(reversed(synthetic_fixtures()))) == days


def test_matchdays_as_of_and_max_days() -> None:
    fixtures = synthetic_fixtures()
    days_rounds_first = {Girone.A: 14, Girone.B: 14, Girone.C: 14, Girone.SB: 13}
    later = matchdays(fixtures, as_of=TUE10)
    assert [m.key for m in later] == ["2026-11-10", "2026-11-15", "2026-11-22"]
    assert "A14_0" not in later[0].fixture_ids and "SB13_0" not in later[0].fixture_ids
    # the Girone B straggler is now the only pending match of round 14: a recovery, attached to the nearest matchday
    assert "B14_9" in later[2].fixture_ids and later[2].rounds == {Girone.A: 16}
    assert matchdays(fixtures, as_of=date(2027, 1, 1)) == []
    # a wide tolerance keeps the straggler inside its round (24 days from the round's anchor); the first matchday then
    # spans to 2 Dec and becomes the nearest one for the 23 Dec recovery too
    wide = matchdays(fixtures, max_days=30)
    assert "B14_9" in wide[0].fixture_ids and "A12x" in wide[0].fixture_ids and wide[0].end == WED_DEC23
    assert wide[0].rounds == days_rounds_first and [m.n_fixtures for m in wide] == [41, 10, 20, 10]
    with pytest.raises(ValueError, match="max_days"):
        matchdays(fixtures, max_days=-1)
    assert matchdays([]) == [] and next_matchday([]) is None


def test_midweek_recovery_before_the_weekend_round_is_not_a_regular_round() -> None:
    """A postponed match replayed midweek *before* the competition's next weekend round (the common Serie C case)."""
    # as of Tue 10 Nov: Girone A has one round-10 recovery on Wed 11, then rounds 15 (Sun 15) and 16 (Sun 22)
    fixtures = [
        fx("A10rec", WED11, 10, "A", "Alpha", "Beta"),
        *round_fixtures("A", 15, [SUN15] * 10),
        *round_fixtures("A", 16, [SUN22] * 10),
        *round_fixtures("B", 15, [SUN15] * 10),
        *round_fixtures("B", 16, [SUN22] * 10),
        *round_fixtures("C", 15, [SUN15] * 10),
        *round_fixtures("C", 16, [SUN22] * 10),
        *round_fixtures("SB", 14, [SAT14] * 10),
        *round_fixtures("SB", 15, [SAT21] * 10),
    ]
    days = matchdays(fixtures)
    assert [m.key for m in days] == ["2026-11-14", "2026-11-21"]
    first, second = days
    assert first.rounds == {Girone.A: 15, Girone.B: 15, Girone.C: 15, Girone.SB: 14}, "round 15 is Girone A's regular round, not the recovery"
    assert "A10rec" in first.fixture_ids and {f"A15_{i}" for i in range(10)} <= first.fixture_ids and first.n_fixtures == 41
    assert first.start == WED11 and first.label == "Giornata dell'11-15 nov 2026 · Girone A g.15, Girone B g.15, Girone C g.15, Serie B g.14"
    assert second.rounds == {Girone.A: 16, Girone.B: 16, Girone.C: 16, Girone.SB: 15} and second.n_fixtures == 40
    assert next_matchday(fixtures) == first and len(fixtures_of(first, fixtures)) == 41
    # two recoveries of different old rounds on the same midweek: both attached, no extra matchday and no '#2' keys
    two = [
        fx("C9rec", WED11, 9, "C", "C1", "C2"),
        fx("C11rec", WED11, 11, "C", "C3", "C4"),
        *round_fixtures("C", 15, [SUN15] * 10),
        *round_fixtures("C", 16, [SUN22] * 10),
        *round_fixtures("SB", 14, [SAT14] * 10),
        *round_fixtures("SB", 15, [SAT21] * 10),
    ]
    days = matchdays(two)
    assert [m.key for m in days] == ["2026-11-14", "2026-11-21"] and {"C9rec", "C11rec"} <= days[0].fixture_ids
    assert days[0].rounds == {Girone.C: 15, Girone.SB: 14} and days[0].n_fixtures == 22 and days[1].n_fixtures == 20
    # a recovery played *after* the next round goes to the nearest matchday too (here the Sat 21 / Sun 22 one)
    after = [fx("A10rec", WED18, 10, "A", "Alpha", "Beta"), *round_fixtures("A", 15, [SUN15] * 10), *round_fixtures("A", 16, [SUN22] * 10)]
    days = matchdays(after)
    assert [dict(m.rounds) for m in days] == [{Girone.A: 15}, {Girone.A: 16}] and "A10rec" in days[0].fixture_ids and days[0].end == WED18
    # an anticipated match (one round-16 game on Fri 13, before round 15) does not turn round 15 into recoveries
    anticipated = [*round_fixtures("A", 15, [SUN15] * 10), *round_fixtures("A", 16, [FRI13] + [SUN22] * 9)]
    days = matchdays(anticipated)
    assert [dict(m.rounds) for m in days] == [{Girone.A: 15}, {Girone.A: 16}]
    assert "A16_0" in days[0].fixture_ids and days[0].start == FRI13 and days[0].n_fixtures == 11 and days[1].n_fixtures == 9


def test_rounds_pair_by_calendar_block_not_by_index() -> None:
    """Serie B skips 14 Nov (international break) while Serie C plays through: no drift of the pairing afterwards."""
    fixtures = [
        *round_fixtures("A", 14, [SUN8] * 10),
        *round_fixtures("A", 15, [SUN15] * 10),
        *round_fixtures("A", 16, [SUN22] * 10),
        *round_fixtures("A", 17, [SUN29] * 10),
        *round_fixtures("SB", 12, [SAT7] * 10),
        *round_fixtures("SB", 13, [SAT21] * 10),
        *round_fixtures("SB", 14, [SAT28] * 10),
        *round_fixtures("SB", 15, [SAT_DEC5] * 10),
    ]
    days = matchdays(fixtures)
    assert [(m.key, dict(m.rounds)) for m in days] == [
        ("2026-11-07", {Girone.A: 14, Girone.SB: 12}),
        ("2026-11-15", {Girone.A: 15}),
        ("2026-11-21", {Girone.A: 16, Girone.SB: 13}),
        ("2026-11-28", {Girone.A: 17, Girone.SB: 14}),
        ("2026-12-05", {Girone.SB: 15}),
    ]
    assert days[1].label == "Giornata del 15 nov 2026 · Girone A g.15" and days[2].label == "Giornata del 21-22 nov 2026 · Girone A g.16, Serie B g.13"
    assert all((m.end - m.start).days <= 1 for m in days), "a matchday is one weekend, never a 7-day span"
    # a midweek Serie B round pairs with the midweek Serie C round (Tue 17 / Wed 18), not with the weekend one
    midweek = [*round_fixtures("A", 15, [SUN15] * 10), *round_fixtures("A", 16, [WED18] * 10), *round_fixtures("SB", 13, [TUE17] * 10)]
    days = matchdays(midweek)
    assert [dict(m.rounds) for m in days] == [{Girone.A: 15}, {Girone.A: 16, Girone.SB: 13}]
    assert days[1].label == "Giornata del 17-18 nov 2026 · Girone A g.16, Serie B g.13" and days[1].key == "2026-11-17"
    # a competition with no fixtures at all is simply absent everywhere
    alone = matchdays([f for f in fixtures if f.girone is Girone.A])
    assert [dict(m.rounds) for m in alone] == [{Girone.A: 14}, {Girone.A: 15}, {Girone.A: 16}, {Girone.A: 17}]


def test_a_round_in_progress_stays_regular_while_half_of_it_is_pending() -> None:
    # Sunday morning: Saturday's 4 matches of Girone A round 15 are played, 6 remain (Sun 15 x5, Mon 16) -> still round 15
    morning = [
        *round_fixtures("A", 15, [SUN15] * 5 + [MON16]),
        *round_fixtures("A", 16, [SUN22] * 10),
        *round_fixtures("B", 15, [SUN15] * 10),
        *round_fixtures("B", 16, [SUN22] * 10),
    ]
    days = matchdays(morning, as_of=SUN15)
    assert [dict(m.rounds) for m in days] == [{Girone.A: 15, Girone.B: 15}, {Girone.A: 16, Girone.B: 16}] and days[0].n_fixtures == 16
    # Sunday evening: 3 matches left (fewer than half the round) -> attached to the nearest matchday, the same weekend
    evening = [*round_fixtures("A", 15, [SUN15] * 2 + [MON16]), *morning[6:]]
    days = matchdays(evening, as_of=SUN15)
    assert [dict(m.rounds) for m in days] == [{Girone.B: 15}, {Girone.A: 16, Girone.B: 16}]
    assert {"A15_0", "A15_1", "A15_2"} <= days[0].fixture_ids and days[0].n_fixtures == 13 and days[0].end == MON16


def test_next_matchday_and_fixtures_of() -> None:
    fixtures = synthetic_fixtures()
    nxt = next_matchday(fixtures)
    assert nxt is not None and nxt == matchdays(fixtures)[0]
    ordered = fixtures_of(nxt, fixtures)
    assert len(ordered) == 39 == nxt.n_fixtures
    # by date, then competition order, then label: the Serie B Friday anticipo, then Saturday (A, C, SB), Sunday, Monday
    assert ordered[0].match_id == "SB13_0" and [f.match_id for f in ordered[1:4]] == ["A14_0", "A14_1", "A14_2"]
    assert [f.girone for f in ordered[4:8]] == [Girone.C] * 4 and [f.girone for f in ordered[8:17]] == [Girone.SB] * 9
    assert ordered[-1].match_id == "A14_9" and [f.date for f in ordered] == sorted(f.date for f in ordered)
    assert nxt.contains("A14_0") and nxt.contains(fixtures[0]) and not nxt.contains("A15_0")


def test_matchday_dataclass_and_date_spans() -> None:
    m = Matchday("2026-11-08", "x", date(2026, 11, 8), date(2026, 11, 9), {"A": 14}, frozenset({"a", "b"}))
    assert m.rounds == {Girone.A: 14} and m.n_fixtures == 2 and hash(m) == hash("2026-11-08")
    assert {m: 1}[m] == 1 and m.rounds_text() == "Girone A g.14"
    with pytest.raises(ValueError):
        Matchday("k", "x", date(2026, 11, 9), date(2026, 11, 8), {}, frozenset())
    with pytest.raises(ValueError):
        Matchday("", "x", date(2026, 11, 8), date(2026, 11, 8), {}, frozenset())
    assert format_date_span(date(2026, 11, 8), date(2026, 11, 8)) == "dell'8 nov 2026"
    assert format_date_span(date(2026, 11, 7), date(2026, 11, 9)) == "del 7-9 nov 2026"
    assert format_date_span(date(2026, 10, 30), date(2026, 11, 2)) == "del 30 ott-2 nov 2026"
    assert format_date_span(date(2026, 12, 30), date(2027, 1, 3)) == "del 30 dic 2026-3 gen 2027"
    assert format_date_span(date(2026, 11, 1), date(2026, 11, 3)) == "dell'1-3 nov 2026"
    with pytest.raises(ValueError):
        format_date_span(date(2026, 11, 9), date(2026, 11, 8))
    # the key is the ISO start date; two rounds of one competition in the same block (a double-header) open two
    # matchdays with distinct keys, the lower round first
    same_day = [
        fx("A1", date(2026, 11, 8), 10, "A", "a", "b"),
        fx("A2", date(2026, 11, 8), 11, "A", "b", "a"),
        fx("SB1", date(2026, 11, 8), 10, "SB", "s", "t"),
    ]
    days = matchdays(same_day)
    assert [m.key for m in days] == ["2026-11-08", "2026-11-08#2"]
    assert days[0].fixture_ids == {"A1", "SB1"} and days[1].fixture_ids == {"A2"}


# ----------------------------------------------------------------------------- rank_picks
def test_rank_picks_ordering_tie_breaks_and_limits() -> None:
    f1 = fx("M1", date(2026, 11, 8), 14, "B", "Bologna", "Carpi")
    f2 = fx("M2", date(2026, 11, 8), 14, "A", "Alpha", "Beta")
    f3 = fx("M3", date(2026, 11, 7), 13, "SB", "Spezia", "Pisa")
    f4 = fx("M4", date(2026, 11, 8), 14, "C", "Crotone", "Bari")
    p1 = make_prediction(f1, 1.3, 1.2, p_x=0.30, fair_x=3.40)  # same P(X) as M2 but worse fair odds -> after M2
    p2 = make_prediction(f2, 1.3, 1.2, p_x=0.30, fair_x=3.30)
    p3 = make_prediction(f3, 1.1, 1.0, p_x=0.33)
    p4 = make_prediction(f4, 2.0, 0.6, p_x=0.20)
    picks = rank_picks([p1, p2, p3, p4], market=Market.MATCH_ODDS, selection="X", n=10)
    assert [p.match_id for p in picks] == ["M3", "M2", "M1", "M4"]
    assert [p.rank for p in picks] == [1, 2, 3, 4]
    assert picks[0].p_model == pytest.approx(0.33) and picks[0].fair_odds == pytest.approx(1 / 0.33)
    assert picks[1].fair_odds == 3.30 and picks[2].fair_odds == 3.40
    # same P(X) and same fair odds: the earlier date wins, then the competition order, then the label
    p5 = make_prediction(fx("M5", date(2026, 11, 9), 14, "A", "Zeta", "Eta"), 1.3, 1.2, p_x=0.30, fair_x=3.30)
    p6 = make_prediction(fx("M6", date(2026, 11, 8), 14, "SB", "Avellino", "Bari"), 1.3, 1.2, p_x=0.30, fair_x=3.30)
    tied = rank_picks([p5, p2, p6], market=Market.MATCH_ODDS, selection="X", n=3)
    assert [p.match_id for p in tied] == ["M2", "M6", "M5"]
    # n limits, n above the available keeps them all, n <= 0 gives nothing, empty input gives nothing
    assert [p.match_id for p in rank_picks([p1, p2, p3, p4], market=Market.MATCH_ODDS, selection="X", n=2)] == ["M3", "M2"]
    assert len(rank_picks([p1, p2], market=Market.MATCH_ODDS, selection="X", n=99)) == 2
    assert rank_picks([p1, p2], market=Market.MATCH_ODDS, selection="X", n=0) == []
    assert rank_picks([], market=Market.MATCH_ODDS, selection="X", n=5) == []
    # min_probability filter and the matchday filter
    assert [p.match_id for p in rank_picks([p1, p2, p3, p4], market=Market.MATCH_ODDS, selection="X", n=10, min_probability=0.3)] == ["M3", "M2", "M1"]
    day = Matchday("2026-11-08", "x", date(2026, 11, 8), date(2026, 11, 8), {"A": 14}, frozenset({"M2", "M4"}))
    assert [p.match_id for p in rank_picks([p1, p2, p3, p4], market=Market.MATCH_ODDS, selection="X", n=10, matchday=day)] == ["M2", "M4"]
    with pytest.raises(ValueError, match="not a selection"):
        rank_picks([p1], market=Market.MATCH_ODDS, selection="GG", n=1)
    # notes: close rates for the draw, high xG for both teams to score
    assert DRAW_NOTE_CLOSE_RATES in picks[0].note  # 1.1 vs 1.0
    gg = btts_picks([make_prediction(f4, 2.0, 1.4)], n=1)
    assert gg[0].selection == "GG" and gg[0].market is Market.BTTS and BTTS_NOTE_HIGH_XG in gg[0].note
    assert rank_picks([p4], market=Market.OVER_UNDER_25, selection="Over 2.5", n=1)[0].note == ""


def test_rank_picks_attaches_the_best_book_price() -> None:
    f1 = fx("M1", date(2026, 11, 8), 14, "A", "Alpha", "Beta")
    f2 = fx("M2", date(2026, 11, 8), 14, "B", "B1", "B2")
    p1 = make_prediction(f1, 1.2, 1.1)
    p2 = make_prediction(f2, 1.4, 1.3)
    opps = [
        opportunity(p1, Market.MATCH_ODDS, "X", 3.10, "BookA"),
        opportunity(p1, Market.MATCH_ODDS, "X", 3.35, "BookB", flags=(FLAG_VALUE,)),  # best price
        opportunity(p1, Market.MATCH_ODDS, "1", 2.50, "BookB"),  # another selection: ignored
        opportunity(p1, Market.BTTS, "GG", 1.90, "BookA", flags=(FLAG_VALUE, FLAG_LONGSHOT)),
    ]
    picks = draw_picks([p1, p2], n=5, opportunities=opps)
    by_id = {p.match_id: p for p in picks}
    assert by_id["M1"].book_odds == 3.35 and by_id["M1"].bookmaker == "BookB" and by_id["M1"].flags == (FLAG_VALUE,)
    assert by_id["M1"].ev == pytest.approx(p1.probabilities[Market.MATCH_ODDS]["X"] * 3.35 - 1.0)
    assert by_id["M2"].book_odds is None and by_id["M2"].bookmaker is None and by_id["M2"].ev is None and by_id["M2"].flags == ()
    gg = btts_picks([p1, p2], n=12, opportunities=opps)
    assert {p.match_id: p.book_odds for p in gg} == {"M1": 1.90, "M2": None}
    assert {p.match_id: p.flags for p in gg}["M1"] == (FLAG_VALUE, FLAG_LONGSHOT)
    # a Pick validates its inputs
    with pytest.raises(ValueError):
        Pick(1, "x", date(2026, 1, 1), "A", "h", "a", Market.MATCH_ODDS, "GG", 0.3, 3.3)
    with pytest.raises(ValueError):
        Pick(0, "x", date(2026, 1, 1), "A", "h", "a", Market.MATCH_ODDS, "X", 0.3, 3.3)


@pytest.fixture(scope="module")
def simulated_predictions() -> tuple[list[MatchPrediction], list[ValueOpportunity]]:
    """Predictions and value rows of the round after 2026-11-06 for the real 2026/27 perimeter (Serie B included)."""
    league = simulate_league(seed=5, as_of=date(2026, 11, 6))
    assert {f.girone for f in league.fixtures} == set(Girone) and len(league.fixtures) == 40
    engine = DixonColesEngine(DixonColesConfig.with_gaussian_prior(xi=0.0))
    engine.fit(league.matches, registry=league.registry)
    predictions = [engine.predict(f) for f in league.fixtures]
    opportunities = ValueEngine(kelly=KellyConfig()).evaluate(predictions, league.odds, matches=league.matches)
    return predictions, opportunities


def test_draw_and_btts_picks_sizes_and_mixed_competitions(simulated_predictions) -> None:
    predictions, opportunities = simulated_predictions
    draws = draw_picks(predictions, opportunities=opportunities)
    btts = btts_picks(predictions, opportunities=opportunities)
    assert len(draws) == 5 and len(btts) == 12
    assert all(p.market is Market.MATCH_ODDS and p.selection == "X" for p in draws)
    assert all(p.market is Market.BTTS and p.selection == "GG" for p in btts)
    assert [p.rank for p in draws] == [1, 2, 3, 4, 5] and [p.rank for p in btts] == list(range(1, 13))
    assert all(a.p_model >= b.p_model for a, b in pairwise(draws))
    assert all(a.p_model >= b.p_model for a, b in pairwise(btts))
    # chosen across the competitions, not within one
    assert len({p.girone for p in draws}) >= 2 and len({p.girone for p in btts}) >= 3
    assert Girone.SB in {p.girone for p in btts} | {p.girone for p in draws}
    # the synthetic bookmaker quotes every fixture, so every pick carries a price, a bookmaker and an EV
    assert all(p.book_odds is not None and p.bookmaker == "SimBook" and p.ev is not None for p in draws + btts)
    # the ranking is by probability, not by value: at least one pick is not a VALUE selection
    assert any(FLAG_VALUE not in p.flags for p in btts)
    # the top draws are the 5 highest P(X) of the whole round
    top_x = sorted((p.probabilities[Market.MATCH_ODDS]["X"] for p in predictions), reverse=True)[:5]
    assert [p.p_model for p in draws] == pytest.approx(top_x)
    # a matchday restricts the candidates
    day = next_matchday([p.fixture for p in predictions])
    assert day is not None and day.n_fixtures == 40 and day.rounds == {g: 12 for g in Girone}
    assert draw_picks(predictions, matchday=day) == draw_picks(predictions)
    half = Matchday("x", "x", day.start, day.end, {"A": 12}, frozenset(p.match_id for p in predictions if p.fixture.girone == Girone.A))
    only_a = btts_picks(predictions, n=12, matchday=half)
    assert len(only_a) == 10 and {p.girone for p in only_a} == {Girone.A}


def test_picks_frame_columns_and_types(simulated_predictions) -> None:
    predictions, opportunities = simulated_predictions
    frame = picks_frame(btts_picks(predictions, opportunities=opportunities))
    assert list(frame.columns) == list(PICKS_COLUMNS) == [
        "#", "Competizione", "Data", "Partita", "Esito", "P. Modello (%)", "Quota Fair", "Quota Book", "Bookmaker", "EV (%)", "Flag", "Nota",
    ]
    assert len(frame) == 12 and frame["#"].tolist() == list(range(1, 13))
    assert str(frame["Data"].dtype).startswith("datetime64") and frame["Esito"].eq("GG").all()
    assert frame["P. Modello (%)"].between(0.0, 100.0).all() and np.all(np.diff(frame["P. Modello (%)"].to_numpy()) <= 1e-9)
    assert (frame["Quota Fair"] > 1.0).all() and frame["Quota Book"].notna().all() and frame["Bookmaker"].eq("SimBook").all()
    assert set(frame["Competizione"]) <= {g.short for g in Girone}
    for column in ("Competizione", "Partita", "Esito", "Bookmaker", "Flag", "Nota"):
        assert str(frame[column].dtype) == "string", column
    empty = picks_frame([])
    assert list(empty.columns) == list(PICKS_COLUMNS) and empty.empty
    assert str(empty["Data"].dtype).startswith("datetime64") and str(empty["#"].dtype) == "int64"
    # unquoted picks: NaN price/EV, <NA> bookmaker
    bare = picks_frame(draw_picks(predictions, n=3))
    assert bare["Quota Book"].isna().all() and bare["EV (%)"].isna().all() and bare["Bookmaker"].isna().all() and bare["Flag"].eq("").all()


# ----------------------------------------------------------------------------- legend
def test_legend_covers_the_terms_a_bettor_needs() -> None:
    terms = legend_terms()
    text = legend_markdown()
    assert text.startswith(f"## {LEGEND_TITLE}\n") and text.endswith("\n")
    assert len(LEGEND) == 9 and all(isinstance(s, LegendSection) and all(isinstance(e, LegendEntry) for e in s.entries) for s in LEGEND)
    titles = [s.title for s in LEGEND]
    assert titles[0].startswith("Parametri del modello") and titles[-1] == "Regole di buon senso"
    for needle in (
        "α (alfa)", "β (beta)", "γ (gamma)", "ρ (rho)", "ξ (xi)", "Prior gaussiana", "λ (lambda)", "μ (mu)",
        "Matrice dei punteggi", "Risultato più probabile", "Quota fair", "Quota book", "de-vig", "EV (%)", "Edge",
        "Soglia EV", "Kelly pieno", "Kelly frazionario", "Stake (unità)", "Limite per puntata", "Stake 0 unità",
        "VALUE", "REGRESSION_TRAP", "TRAP:<squadra>", "LONGSHOT", "SUSPICIOUS_EDGE", "Le 5 partite più probabili da X",
        "Le 12 partite più probabili da GG", "Dati simulati vs reali", "Storico di 3 stagioni", "Copertura minima",
        "Prime giornate", "Bankroll fisso", "Mai inseguire le perdite", "Campione minimo", "quote di chiusura",
    ):
        assert any(needle in term for term in terms), needle
    # the rules and numbers quoted match the engine
    assert "GA ≤ 1" in terms["REGRESSION_TRAP"] and "xGA ≥ 3.5" in terms["REGRESSION_TRAP"] and "ultime 3 gare" in terms["REGRESSION_TRAP"]
    assert "sopra 6" in terms["LONGSHOT"] and "25%" in terms["SUSPICIOUS_EDGE"]
    assert "f* = (p·b − q) / b" in text and "EV = P_modello × Quota book − 1" in text
    assert "0.30" in terms["Prior gaussiana · Stima MAP"] and "15 partite" in terms["Prior gaussiana · Stima MAP"]
    assert "3 unità" in terms["Limite per puntata"] and "0,1 unità" in terms["Stake 0 unità"]
    assert "exp(γ)" in terms["γ (gamma) · Vantaggio casa"] and "35%" in terms["α (alfa) · Forza offensiva"]
    rules = next(s for s in LEGEND if s.title == "Regole di buon senso")
    assert 3 <= len(rules.entries) <= 4
    # Markdown structure: one ### heading per section, one bullet per entry, no empty line inside a list
    assert text.count("\n### ") == len(LEGEND) and text.count("\n- **") == len(terms)
    with pytest.raises(ValueError):
        LegendEntry("x", " ")
    with pytest.raises(ValueError):
        LegendSection("x", ())


def test_readme_legend_section_is_in_sync() -> None:
    readme = (REPO_ROOT / "README.md").read_text(encoding="utf-8")
    match = re.search(r"<!-- legend:start -->\n(.*?)<!-- legend:end -->", readme, flags=re.DOTALL)
    assert match is not None, "README.md must hold the legend between <!-- legend:start --> and <!-- legend:end -->"
    assert match.group(1) == legend_markdown(), (
        "README legend out of date: regenerate it with "
        "python -c \"from src.dashboard.legend import legend_markdown; print(legend_markdown())\""
    )
    assert "## Legenda" in readme and readme.count("<!-- legend:start -->") == 1


def test_worked_example_reproduces_the_value_engine_numbers(simulated_predictions) -> None:
    _predictions, opportunities = simulated_predictions
    best = max(opportunities, key=lambda o: o.ev)
    example = WorkedExample.from_opportunity(best, bankroll_units=100.0)
    assert example.ev == pytest.approx(best.ev) and example.edge == pytest.approx(best.edge, abs=1e-9)
    assert example.fair_odds == pytest.approx(best.fair_odds) and example.kelly_full == pytest.approx(best.kelly_full)
    assert example.stake == pytest.approx(best.stake_units)
    text = worked_example_markdown(example)
    assert best.home_team in text and f"{best.book_odds:.2f}" in text and f"**{best.ev:+.1%}**" in text and "Kelly pieno" in text
    assert "Esempio costruito" not in text
    synthetic = WorkedExample.synthetic()
    text_s = worked_example_markdown(synthetic)
    assert synthetic.ev == pytest.approx(0.08) and synthetic.kelly_full == pytest.approx(0.0470588, abs=1e-6) and synthetic.stake == 0.6
    assert "Esempio costruito" in text_s and "**2.50**" in text_s and "**+8.0%**" in text_s and "**0.6 unità**" in text_s and "1/8" in text_s
    # a negative-EV example says so
    losing = WorkedExample("A", "B", "Girone A", "1X2", "X", "Book", 2.50, 0.30, 0.40, 0.38)
    text_l = worked_example_markdown(losing)
    assert "niente valore" in text_l and "non si punta" in text_l and losing.stake == 0.0 and "nessuno" in text_l
    with pytest.raises(ValueError):
        WorkedExample("A", "B", "G", "1X2", "1", "Book", 1.0, 0.4, 0.5, 0.5)
    with pytest.raises(ValueError):
        WorkedExample("A", "B", "G", "1X2", "1", "Book", 2.0, 1.0, 0.5, 0.5)


def test_legend_claims_match_the_engine() -> None:
    """Regressions of the review: the legend must state what the code does, with the right sign and the right scope."""
    terms = legend_terms()
    # rho: a negative rho RAISES 0-0 and 1-1 (tau(0,0) = 1 - lam*mu*rho > 1) and lowers 1-0 / 0-1
    lam, mu = 1.3, 1.1
    corrected, independent = dixon_coles_pmf(lam, mu, -0.1), dixon_coles_pmf(lam, mu, 0.0)
    assert corrected[0, 0] > independent[0, 0] and corrected[1, 1] > independent[1, 1]
    assert corrected[1, 0] < independent[1, 0] and corrected[0, 1] < independent[0, 1]
    assert np.trace(corrected) > np.trace(independent)
    rho_text = terms["ρ (rho) · Correlazione bassi punteggi"]
    assert "più 0-0 e 1-1" in rho_text and "meno 1-0 / 0-1" in rho_text and "meno 0-0" not in rho_text
    # the prior shrinks alpha to 0 and beta to its competition mean (the intercept), not beta to 0
    prior = terms["Prior gaussiana · Stima MAP"]
    assert "α verso 0 e β verso la media del suo girone" in prior and "α e β verso 0" not in prior
    # the bookmaker column comes from the CSV's bookmaker column, never from the file name
    book = terms["Quota book"]
    assert "colonna bookmaker del CSV" in book and "nome del file" not in book
    # the 3-unit cap is 3% only with the standard 100-unit bankroll (KellyConfig.max_stake_units is fixed)
    cap = terms["Limite per puntata"]
    assert "100 unità" in cap and "(3% del bankroll)" not in cap
    assert KellyConfig(bankroll_units=300.0).max_stake_units == 3.0 and stake_units_cap_demo() == 3.0
    # VALUE does not imply a positive stake: a long shot with EV above threshold can round to 0 units
    value_text = terms["VALUE"]
    assert "può arrotondare a 0" in value_text and "maggiore di zero" not in value_text
    pred = make_prediction(fx("M1", date(2026, 11, 8), 14, "A", "H", "A"), 1.0, 1.0, p_x=0.08)
    rows = ValueEngine().evaluate([pred], [OddsQuote("M1", "Book", Market.MATCH_ODDS, {"1": 2.0, "X": 13.0, "2": 3.0})])
    row = next(r for r in rows if r.selection == "X")
    assert row.value and FLAG_VALUE in row.flags and FLAG_LONGSHOT in row.flags and row.stake_units == 0.0 and row.ev > 0.035
    # the matchday entry describes the calendar-block pairing, not a k-th-round rule
    giornata = terms["Giornata"]
    assert "venerdì-lunedì" in giornata and "martedì-giovedì" in giornata and "k-esima" not in giornata
    assert "giornata più vicina" in giornata
    # the README carries the same text
    readme = (REPO_ROOT / "README.md").read_text(encoding="utf-8")
    assert "più 0-0 e 1-1" in readme and "β verso la media del suo girone" in readme and "colonna bookmaker del CSV" in readme


def stake_units_cap_demo() -> float:
    """``stake_units`` at the cap for a 300-unit bankroll: 3 units, which is 1% of that bankroll, not 3%."""
    from src.engine.value import stake_units

    return stake_units(0.5, 3.0, KellyConfig(bankroll_units=300.0))


def test_synthetic_example_states_why_it_is_synthetic() -> None:
    default = worked_example_markdown(WorkedExample.synthetic())
    assert f"Esempio costruito: {SYNTHETIC_REASON_NO_VALUE};" in default and "nessuna quota nei dati correnti" not in default
    no_quotes = worked_example_markdown(WorkedExample.synthetic(reason=SYNTHETIC_REASON_NO_QUOTES))
    assert f"Esempio costruito: {SYNTHETIC_REASON_NO_QUOTES};" in no_quotes
    custom = worked_example_markdown(WorkedExample.synthetic(reason="nessuna selezione di valore (Girone A, soglia EV 15,0%)"))
    assert "Esempio costruito: nessuna selezione di valore (Girone A, soglia EV 15,0%);" in custom
    assert SYNTHETIC_REASON_NO_QUOTES != SYNTHETIC_REASON_NO_VALUE
    with pytest.raises(ValueError, match="reason"):
        WorkedExample.synthetic(reason=" ")
    # a real example never carries the synthetic header, whatever its reason field says
    real = WorkedExample("A", "B", "Girone A", "1X2", "1", "Book", 2.70, 0.40, 0.37, 0.35)
    assert "Esempio costruito" not in worked_example_markdown(real) and real.synthetic_reason == SYNTHETIC_REASON_NO_VALUE


def test_pick_example_row_prefers_an_ordinary_value_row() -> None:
    f1 = fx("M1", date(2026, 11, 8), 14, "A", "Alpha", "Beta")
    f2 = fx("M2", date(2026, 11, 8), 14, "B", "B1", "B2")
    p1, p2 = make_prediction(f1, 1.2, 1.1), make_prediction(f2, 1.4, 1.3)
    suspicious = opportunity(p1, Market.MATCH_ODDS, "2", 10.3, "Book", flags=(FLAG_VALUE, FLAG_LONGSHOT, FLAG_SUSPICIOUS_EDGE))
    longshot = opportunity(p1, Market.MATCH_ODDS, "X", 6.5, "Book", flags=(FLAG_VALUE, FLAG_LONGSHOT))
    ordinary = opportunity(p2, Market.BTTS, "GG", 1.95, "Book", flags=(FLAG_VALUE,))
    plain = opportunity(p2, Market.MATCH_ODDS, "1", 2.1, "Book")
    assert POOR_EXAMPLE_FLAGS == {FLAG_LONGSHOT, FLAG_SUSPICIOUS_EDGE}
    assert pick_example_row([suspicious, longshot, ordinary, plain]) is ordinary  # the first row without the poor flags
    assert pick_example_row([ordinary, suspicious]) is ordinary and pick_example_row([plain]) is plain
    assert pick_example_row([suspicious, longshot]) is suspicious  # every row is flagged: the first one, as shown
    assert pick_example_row([]) is None
    assert pick_example_row(()) is None
