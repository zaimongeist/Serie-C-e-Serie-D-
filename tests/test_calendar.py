"""Tests for the official-round rule of ``src.engine.calendar`` on synthetic data reproducing the real 2026/27 cases."""
from __future__ import annotations

from collections.abc import Sequence
from datetime import UTC, date, datetime, timedelta

import pandas as pd
import pytest

from src.common.schemas import Fixture, Girone
from src.engine.calendar import (
    IN_PLAY_TEXT,
    ITALIAN_WEEKDAYS,
    KICKOFF_GRACE,
    RESULT_PENDING_WINDOW,
    STATUS_IN_PLAY,
    STATUS_PLAYED,
    STATUS_POSTPONED,
    STATUS_TO_PLAY,
    CompetitionCalendar,
    RoundMatch,
    RoundView,
    current_round,
    format_day,
    format_kickoff,
    kicked_off,
    recoveries,
    rome_midnight,
    round_view,
    rounds_available,
    to_rome,
    upcoming_rounds,
    week_block,
)


# ----------------------------------------------------------------------------- helpers
def utc(day: date, hour: int, minute: int = 0) -> datetime:
    """A kick-off in UTC (November 2026 is CET = UTC+1: 13:30 UTC is 14:30 in Rome)."""
    return datetime(day.year, day.month, day.day, hour, minute, tzinfo=UTC)


def fixture(mid: str, kickoff: datetime, rnd: int, girone: str, home: str, away: str, status: str = "scheduled") -> Fixture:
    return Fixture(mid, kickoff.date(), rnd, girone, home, away, kickoff=kickoff, status=status)


def played_frame(rows: Sequence[tuple[str, date, int, str, str, str, int, int]]) -> pd.DataFrame:
    """A canonical played-match frame from ``(match_id, date, round, girone, home, away, hg, ag)`` rows."""
    return pd.DataFrame(
        [
            {"match_id": mid, "date": pd.Timestamp(day), "round": rnd, "girone": g, "home_team": h, "away_team": a, "home_goals": hg, "away_goals": ag}
            for mid, day, rnd, g, h, a, hg, ag in rows
        ]
    )


FRI6, SAT7, SUN8, MON9 = date(2026, 11, 6), date(2026, 11, 7), date(2026, 11, 8), date(2026, 11, 9)
WED11, SAT14, SUN15 = date(2026, 11, 11), date(2026, 11, 14), date(2026, 11, 15)
SAT31, SUN1 = date(2026, 10, 31), date(2026, 11, 1)
#: Saturday 7 November 2026, 10:00 in Rome: "today" of the diagnosis.
NOW = datetime(2026, 11, 7, 9, 0, tzinfo=UTC)


def serie_c_a_round_13() -> tuple[pd.DataFrame, list[Fixture]]:
    """Serie C Girone A: round 13 with two matches played on Friday 06/11, eight on 07-08/11; round 14 the next weekend."""
    teams = [f"A{i:02d}" for i in range(20)]
    played = played_frame([("A13-0", FRI6, 13, "A", teams[0], teams[1], 2, 1), ("A13-1", FRI6, 13, "A", teams[2], teams[3], 0, 0)])
    fixtures = [
        fixture(f"A13-{k}", utc(SAT7 if k < 6 else SUN8, 13, 30), 13, "A", teams[2 * k], teams[2 * k + 1]) for k in range(2, 10)
    ]
    fixtures += [fixture(f"A14-{k}", utc(SUN15, 13, 30), 14, "A", teams[2 * k + 1], teams[2 * k]) for k in range(10)]
    return played, fixtures


def serie_d_g_rounds_10_11() -> tuple[pd.DataFrame, list[Fixture]]:
    """Serie D Girone G: round 10 with 8 of 9 matches played on 31/10-01/11 and the recovery Trastevere-Anzio on Wed 11/11."""
    teams = ["Trastevere", "Anzio", *[f"G{i:02d}" for i in range(16)]]
    rows = [(f"DG10-{k}", SAT31 if k < 4 else SUN1, 10, "DG", teams[2 * k + 2], teams[2 * k + 3], 1, k % 3) for k in range(8)]
    played = played_frame(rows)
    fixtures = [fixture("DG10-rec", utc(WED11, 13, 30), 10, "DG", "Trastevere", "Anzio")]
    fixtures += [fixture(f"DG11-{k}", utc(SUN8, 13, 30), 11, "DG", teams[2 * k], teams[(2 * k + 3) % 18]) for k in range(9)]
    return played, fixtures


# ----------------------------------------------------------------------------- formatting
def test_kickoff_is_shown_in_rome_with_italian_weekdays() -> None:
    assert ITALIAN_WEEKDAYS == ("lun", "mar", "mer", "gio", "ven", "sab", "dom")
    assert format_kickoff(utc(SAT7, 13, 30)) == "sab 07/11 14:30"  # CET: UTC+1
    assert format_kickoff(datetime(2026, 11, 7, 13, 30)) == "sab 07/11 14:30", "a naive kick-off is UTC"
    assert format_kickoff(None, FRI6) == "ven 06/11" == format_day(FRI6) and format_kickoff(None) == ""
    # last Sunday of October 2026 is the switch from CEST (UTC+2) to CET (UTC+1) at 03:00
    assert format_kickoff(datetime(2026, 10, 24, 13, 0, tzinfo=UTC)) == "sab 24/10 15:00"
    assert format_kickoff(datetime(2026, 10, 25, 13, 0, tzinfo=UTC)) == "dom 25/10 14:00"
    assert format_kickoff(datetime(2026, 10, 25, 0, 30, tzinfo=UTC)) == "dom 25/10 02:30"  # still CEST before 01:00 UTC
    assert format_kickoff(datetime(2026, 10, 25, 1, 30, tzinfo=UTC)) == "dom 25/10 02:30"  # the repeated hour, now CET
    assert to_rome(datetime(2026, 10, 25, 1, 30, tzinfo=UTC)).utcoffset() == timedelta(hours=1)
    assert to_rome(datetime(2026, 10, 24, 1, 30, tzinfo=UTC)).utcoffset() == timedelta(hours=2)
    assert rome_midnight(SAT7) == datetime(2026, 11, 6, 23, 0, tzinfo=UTC)
    # a late kick-off in UTC belongs to the next day in Rome: the row's day follows Rome
    late = RoundMatch.from_fixture(fixture("L", datetime(2026, 11, 7, 23, 30, tzinfo=UTC), 13, "A", "x", "y"))
    assert late.date == SUN8 and late.kickoff_text == "dom 08/11 00:30"
    assert week_block(SUN8) == (FRI6, MON9) and week_block(WED11) == (date(2026, 11, 10), date(2026, 11, 12))


def test_round_match_validates_status_and_scores() -> None:
    m = RoundMatch("1", "A", 13, FRI6, None, "H", "O", STATUS_PLAYED, 2, 1)
    assert m.played and m.score == "2-1" and m.status_text == "giocata 2-1" and m.kickoff_text == "ven 06/11" and m.label == "H - O"
    todo = RoundMatch("2", "A", 13, SAT7, utc(SAT7, 13, 30), "H", "O", STATUS_TO_PLAY)
    assert todo.to_play and todo.score == "" and todo.status_text == "da giocare" and todo.kickoff_rome.hour == 14
    with pytest.raises(ValueError):
        RoundMatch("3", "A", 13, FRI6, None, "H", "O", "boh")
    with pytest.raises(ValueError):
        RoundMatch("4", "A", 13, FRI6, None, "H", "O", STATUS_PLAYED)
    with pytest.raises(ValueError):
        RoundMatch("5", "A", 13, FRI6, None, "H", "O", STATUS_TO_PLAY, 1, 0)


# ----------------------------------------------------------------------------- the current round
def test_friday_played_matches_stay_in_the_current_round() -> None:
    """Serie C A round 13: two matches played on Friday, eight to play -> round 13 is current with all ten matches."""
    played, fixtures = serie_c_a_round_13()
    view = current_round(Girone.A, played, fixtures, now=NOW)
    assert isinstance(view, RoundView) and view.round == 13 and view.n_matches == 10
    assert (view.n_played, view.n_to_play, view.n_postponed) == (2, 8, 0)
    assert [m.status_text for m in view.matches[:2]] == ["giocata 2-1", "giocata 0-0"], "the Friday matches come first, with their score"
    assert [m.kickoff_text for m in view.matches[2:4]] == ["sab 07/11 14:30", "sab 07/11 14:30"] and view.matches[-1].kickoff_text == "dom 08/11 14:30"
    assert view.fixture_ids == {f"A13-{k}" for k in range(2, 10)} and view.match_ids == {f"A13-{k}" for k in range(10)}
    assert (view.start, view.end, view.anchor) == (FRI6, SUN8, SAT7) and view.label == "Girone A g.13"
    # Sunday evening after the 14:30 kick-offs: nothing of round 13 is left to play -> round 14 is current
    assert current_round(Girone.A, played, fixtures, now=utc(SUN8, 18)).round == 14
    # ... and while the last Sunday match is being played (kick-off less than 3 hours ago) round 13 is still current
    assert current_round(Girone.A, played, fixtures, now=utc(SUN8, 13, 30) + KICKOFF_GRACE - timedelta(minutes=5)).round == 13


def test_round_in_progress_stays_current_while_its_remaining_matches_are_in_the_same_weekend() -> None:
    """Sunday lunchtime: six of ten played, four left on Sunday / Monday -> still the current round (fewer than half pending)."""
    teams = [f"B{i:02d}" for i in range(20)]
    rows = [(f"B15-{k}", SAT14 if k < 4 else SUN15, 15, "B", teams[2 * k], teams[2 * k + 1], 1, 1) for k in range(6)]
    played = played_frame(rows)
    fixtures = [fixture(f"B15-{k}", utc(SUN15, 17) if k < 9 else utc(date(2026, 11, 16), 19, 30), 15, "B", teams[2 * k], teams[2 * k + 1]) for k in range(6, 10)]
    fixtures += [fixture(f"B16-{k}", utc(date(2026, 11, 22), 13, 30), 16, "B", teams[2 * k + 1], teams[2 * k]) for k in range(10)]
    noon = utc(SUN15, 12)
    view = current_round(Girone.B, played, fixtures, now=noon)
    assert view.round == 15 and view.n_played == 6 and view.n_to_play == 4 and view.in_progress()
    # Sunday night: only the Monday posticipo is left, still the round being played
    assert current_round(Girone.B, played, fixtures, now=utc(SUN15, 22)).round == 15
    # once the Monday match has kicked off more than 3 hours ago the next round is current
    assert current_round(Girone.B, played, fixtures, now=utc(date(2026, 11, 16), 23, 30)).round == 16


def test_midweek_recovery_of_a_played_round_is_not_the_current_round() -> None:
    """Serie D G round 10: 8 played on 31/10-01/11 + Trastevere-Anzio on Wed 11/11 -> round 11 (Sunday 08/11) is current."""
    played, fixtures = serie_d_g_rounds_10_11()
    view = current_round(Girone.DG, played, fixtures, now=NOW)
    assert view is not None and view.round == 11 and view.n_matches == 9 and view.n_played == 0
    assert [v.round for v in upcoming_rounds(Girone.DG, played, fixtures, now=NOW)] == [11]
    ten = round_view(Girone.DG, 10, played, fixtures)
    assert (ten.n_played, ten.n_to_play) == (8, 1) and not ten.in_progress()
    rec = recoveries(Girone.DG, fixtures, played, NOW)
    assert [m.match_id for m in rec] == ["DG10-rec"]
    assert rec[0].label == "Trastevere - Anzio" and rec[0].kickoff_text == "mer 11/11 14:30" and rec[0].original_date in (SAT31, SUN1)
    assert rec[0].original_date_text in ("sab 31/10", "dom 01/11")
    # after round 11 is played (Monday), the recovery still never becomes the current round: round 12 does
    later = [*fixtures, *(fixture(f"DG12-{k}", utc(SUN15, 13, 30), 12, "DG", f"G{k:02d}", f"G{k + 1:02d}") for k in range(0, 16, 2))]
    later.append(fixture("DG12-8", utc(SUN15, 13, 30), 12, "DG", "Trastevere", "Anzio"))
    eleven_played = played_frame([(f.match_id, SUN8, 11, "DG", f.home_team, f.away_team, 1, 0) for f in fixtures if f.round == 11])
    after = pd.concat([played, eleven_played], ignore_index=True)
    assert current_round(Girone.DG, after, later, now=utc(MON9, 12)).round == 12


def test_postponed_matches_are_listed_and_never_current() -> None:
    """Serie D A round 1 Chisola-Borgosesia (30/09, incomplete with a past kick-off) is postponed: listed, never a round."""
    teams = ["Chisola", "Borgosesia", *[f"DA{i:02d}" for i in range(16)]]
    rows = [(f"DA1-{k}", date(2026, 9, 6), 1, "DA", teams[2 * k + 2], teams[2 * k + 3], 2, 0) for k in range(8)]
    played = played_frame(rows)
    postponed = fixture("DA1-p", datetime(2026, 9, 30, 13, 0, tzinfo=UTC), 1, "DA", "Chisola", "Borgosesia", status="postponed")
    nxt = [fixture(f"DA11-{k}", utc(SUN8, 13, 30), 11, "DA", teams[2 * k], teams[2 * k + 1]) for k in range(9)]
    fixtures = [postponed, *nxt]
    assert current_round(Girone.DA, played, fixtures, now=NOW).round == 11
    one = round_view(Girone.DA, 1, played, fixtures)
    assert one.n_matches == 9 and one.n_postponed == 1 and one.postponed[0].status == STATUS_POSTPONED == one.postponed[0].status_text == "rinviata"
    assert one.start == one.end == date(2026, 9, 6), "a postponed match never stretches its round"
    rec = recoveries(Girone.DA, fixtures, played, NOW)
    assert len(rec) == 1 and rec[0].postponed and rec[0].original_date == date(2026, 9, 30) and rec[0].kickoff_text == "mer 30/09 15:00"
    # a postponed match of the current round is shown in the round (rinviata) and listed too, without blocking the round
    in_round = fixture("DA11-p", datetime(2026, 11, 4, 13, 0, tzinfo=UTC), 11, "DA", "X1", "X2", status="postponed")
    view = current_round(Girone.DA, played, [*fixtures, in_round], now=NOW)
    assert view.round == 11 and view.n_postponed == 1 and "DA11-p" not in view.fixture_ids
    assert {m.match_id for m in recoveries(Girone.DA, [*fixtures, in_round], played, NOW)} == {"DA1-p", "DA11-p"}
    # only postponed fixtures left: no current round at all
    assert current_round(Girone.DA, played, [postponed], now=NOW) is None


def test_competitions_with_different_round_numbers_and_the_three_hour_grace() -> None:
    """Serie B at round 11 (one match played on Friday) while Serie C A is at round 13: each competition its own round."""
    played_a, fixtures_a = serie_c_a_round_13()
    teams = [f"SB{i:02d}" for i in range(20)]
    played_sb = played_frame([("SB11-0", FRI6, 11, "SB", teams[0], teams[1], 3, 1)])
    fixtures_sb = [fixture(f"SB11-{k}", utc(SAT7, 14), 11, "SB", teams[2 * k], teams[2 * k + 1]) for k in range(1, 10)]
    played = pd.concat([played_a, played_sb], ignore_index=True)
    fixtures = [*fixtures_a, *fixtures_sb]
    assert current_round(Girone.SB, played, fixtures, now=NOW).round == 11 and current_round(Girone.A, played, fixtures, now=NOW).round == 13
    assert current_round(Girone.SB, played, fixtures, now=NOW).n_played == 1
    # a fixture that kicked off 2 hours ago keeps its round a candidate, 4 hours ago it does not
    lone = [fixture("C1", utc(SAT7, 13), 20, "C", "x", "y")]
    assert current_round(Girone.C, None, lone, now=utc(SAT7, 15)).round == 20
    assert current_round(Girone.C, None, lone, now=utc(SAT7, 17)) is None
    # now=None: no time condition, every scheduled fixture is upcoming
    assert current_round(Girone.C, None, lone, now=None).round == 20


def test_rounds_available_round_view_and_deduplication() -> None:
    played, fixtures = serie_c_a_round_13()
    assert rounds_available(Girone.A, played, fixtures) == [13, 14] and rounds_available(Girone.B, played, fixtures) == []
    empty = round_view(Girone.A, 30, played, fixtures)
    assert empty.n_matches == 0 and empty.anchor is None and empty.start is None and not empty.is_complete and empty.fixture_ids == frozenset()
    # a stale fixture list that still holds a match already played: the played row wins, never two rows
    stale = [fixture("A13-0", utc(FRI6, 19, 30), 13, "A", "A00", "A01"), *fixtures]
    view = round_view(Girone.A, 13, played, stale)
    assert view.n_matches == 10 and next(m for m in view.matches if m.match_id == "A13-0").played
    cal = CompetitionCalendar.build("A", played, stale)
    assert cal.expected_size == 10 and cal.rounds() == [13, 14] and cal.current(NOW).round == 13
    # the round size never drops below the typical round: 3 pending matches of a round without played rows are a recovery
    partial = [fixture(f"A15-{k}", utc(SUN15, 13, 30), 15, "A", f"P{k}", f"Q{k}") for k in range(3)]
    assert [v.round for v in upcoming_rounds(Girone.A, played, [*fixtures, *partial], now=NOW)] == [13, 14]
    assert {m.match_id for m in recoveries(Girone.A, [*fixtures, *partial], played, NOW)} == {"A15-0", "A15-1", "A15-2"}
    # hashing a view by (competition, round)
    assert hash(view) == hash(round_view(Girone.A, 13, played, fixtures)) and {view: 1}[view] == 1


def test_stragglers_far_from_their_round_are_recoveries() -> None:
    """A match of a regular round rescheduled more than 7 days away is listed with the recoveries, the round stays regular."""
    played, fixtures = serie_c_a_round_13()
    moved = [f if f.match_id != "A14-9" else fixture("A14-9", utc(date(2026, 12, 2), 19), 14, "A", f.home_team, f.away_team) for f in fixtures]
    rec = recoveries(Girone.A, moved, played, NOW)
    assert [m.match_id for m in rec] == ["A14-9"] and rec[0].original_date == SUN15
    view = round_view(Girone.A, 14, played, moved)
    assert [m.match_id for m in view.stragglers(7)] == ["A14-9"] and view.stragglers(30) == []
    assert rome_midnight(date(2026, 12, 2)).tzinfo is not None


# ----------------------------------------------------------------------------- review fixes: in play, postponements, recoveries
def test_many_postponed_matches_and_a_midweek_recovery_do_not_make_the_round_current() -> None:
    """Regression: postponed matches counted as 'not yet played', so a winter Serie D round with 4 played, 4 postponed and
    1 match re-dated to Wednesday 16/12 was the current round on Tuesday 15/12 and pushed the full round 15 (9 matches on
    Sunday 20/12) to the next matchday."""
    teams = [f"D{i:02d}" for i in range(18)]
    played = played_frame([(f"P{k}", date(2026, 12, 13), 14, "DA", teams[2 * k], teams[2 * k + 1], 1, 1) for k in range(4)])
    fixtures = [fixture(f"R{k}", utc(date(2026, 12, 13), 13, 30), 14, "DA", teams[2 * k], teams[2 * k + 1], status="postponed") for k in range(4, 8)]
    fixtures.append(fixture("R8", utc(date(2026, 12, 16), 13, 30), 14, "DA", teams[16], teams[17]))
    fixtures += [fixture(f"N{k}", utc(date(2026, 12, 20), 13, 30), 15, "DA", teams[2 * k + 1], teams[2 * k]) for k in range(9)]
    tuesday = datetime(2026, 12, 15, 18, 0, tzinfo=UTC)  # more than 48 hours after the Sunday kick-offs: postponed for good
    view = current_round("DA", played, fixtures, now=tuesday)
    assert view is not None and view.round == 15 and view.n_to_play == 9
    assert [v.round for v in upcoming_rounds("DA", played, fixtures, now=tuesday)] == [15]
    fourteen = round_view("DA", 14, played, fixtures, now=tuesday)
    assert (fourteen.n_played, fourteen.n_postponed, fourteen.n_to_play) == (4, 4, 1)
    assert not CompetitionCalendar.build("DA", played, fixtures).is_regular(fourteen, tuesday - KICKOFF_GRACE)
    rec = recoveries("DA", fixtures, played, tuesday)
    assert [m.match_id for m in rec] == ["R4", "R5", "R6", "R7", "R8"]
    assert [m.original_date for m in rec] == [date(2026, 12, 13)] * 5, "postponed: own day; R8 moved from Sunday's block to Wednesday"
    # on Tuesday morning (less than 48 hours) the four are still waiting for their result: same current round, R8 alone listed
    morning = datetime(2026, 12, 15, 9, 0, tzinfo=UTC)
    assert current_round("DA", played, fixtures, now=morning).round == 15
    assert round_view("DA", 14, played, fixtures, now=morning).n_in_play == 4 and [m.match_id for m in recoveries("DA", fixtures, played, morning)] == ["R8"]
    # the same round with its remaining match still due on the Monday of its weekend is in progress, hence current
    monday = [f if f.match_id != "R8" else fixture("R8", utc(date(2026, 12, 14), 19, 30), 14, "DA", teams[16], teams[17]) for f in fixtures]
    assert current_round("DA", played, monday, now=datetime(2026, 12, 14, 9, 0, tzinfo=UTC)).round == 14


def test_kicked_off_matches_stay_in_their_round_in_play_never_recoveries_never_picked() -> None:
    """Regression: data downloaded Sunday 12:00 UTC, page at 17:00 UTC: the four Sunday 14:30 matches were listed as
    recoveries with a wrong original date and a prediction; with the page at 16:00 the kicked-off matches stayed in the
    picks pool."""
    played_a, fixtures = serie_c_a_round_13()
    saturday = played_frame([(f"A13-{k}", SAT7, 13, "A", f"A{2 * k:02d}", f"A{2 * k + 1:02d}", 1, 0) for k in range(2, 6)])
    played = pd.concat([played_a, saturday], ignore_index=True)
    sunday_ids = {f"A13-{k}" for k in range(6, 10)}
    # Sunday 14:00 UTC (15:00 Rome): the four 14:30 matches are being played -> round 13 is current, nothing of it to pick
    during = utc(SUN8, 14)
    view = current_round(Girone.A, played, fixtures, now=during)
    assert view.round == 13 and (view.n_played, view.n_in_play, view.n_to_play) == (6, 4, 0)
    assert {m.match_id for m in view.in_play} == sunday_ids and view.fixture_ids == frozenset()
    assert {m.status_text for m in view.in_play} == {"in corso / risultato in attesa"} and all(m.status == STATUS_IN_PLAY for m in view.in_play)
    # 17:00 UTC: the grace is over, the round is no longer current and its matches wait for their result in their round
    evening = utc(SUN8, 17)
    assert current_round(Girone.A, played, fixtures, now=evening).round == 14
    assert recoveries(Girone.A, fixtures, played, evening) == [], "a match that kicked off is never a recovery"
    thirteen = round_view(Girone.A, 13, played, fixtures, now=evening)
    assert thirteen.n_in_play == 4 and thirteen.n_to_play == 0 and not thirteen.is_complete
    # without a reference instant the stored statuses are kept
    assert round_view(Girone.A, 13, played, fixtures).n_to_play == 4
    # Saturday 16:00 Rome with the data of the morning: the 14:30 matches have kicked off, out of fixture_ids
    played_fri, _ = serie_c_a_round_13()
    afternoon = utc(SAT7, 15)
    view = current_round(Girone.A, played_fri, fixtures, now=afternoon)
    started = {f"A13-{k}" for k in range(2, 6)}
    assert view.round == 13 and {m.match_id for m in view.in_play} == started and view.fixture_ids == sunday_ids
    assert all(kicked_off(f, afternoon) for f in fixtures if f.match_id in started) and not kicked_off(fixtures[-1], afternoon)
    assert not kicked_off(fixtures[0], None)


def test_serie_d_round_whose_matches_are_all_in_play_stays_current() -> None:
    """Regression: Serie D A on Sunday 15:00 Rome, all nine matches of round 11 kicked off at 14:30: round 11 vanished and
    round 12 (next week) became current, so the matchday picks were for the wrong round."""
    teams = [f"D{i:02d}" for i in range(18)]
    played = played_frame([(f"DA10-{k}", SUN1, 10, "DA", teams[2 * k], teams[2 * k + 1], 1, 0) for k in range(9)])
    fixtures = [fixture(f"DA11-{k}", utc(SUN8, 13, 30), 11, "DA", teams[2 * k + 1], teams[2 * k]) for k in range(9)]
    fixtures += [fixture(f"DA12-{k}", utc(SUN15, 13, 30), 12, "DA", teams[2 * k], teams[(2 * k + 3) % 18]) for k in range(9)]
    view = current_round("DA", played, fixtures, now=utc(SUN8, 14))
    assert view.round == 11 and view.n_in_play == 9 and view.fixture_ids == frozenset()
    assert rounds_available("DA", played, fixtures) == [10, 11, 12]
    # after the grace the next round is current and round 11 waits for its results (never a recovery)
    assert current_round("DA", played, fixtures, now=utc(SUN8, 19)).round == 12
    assert recoveries("DA", fixtures, played, utc(SUN8, 19)) == []


def test_a_postponed_match_is_a_pending_result_for_48_hours() -> None:
    """FootyStats marks a match postponed as soon as its result is 3 hours late: within 48 hours it is 'risultato in attesa'."""
    late = fixture("L1", utc(SUN8, 13, 30), 11, "DA", "X", "Y", status="postponed")
    assert RESULT_PENDING_WINDOW == timedelta(hours=48)
    row = RoundMatch.from_fixture(late)
    assert row.postponed and row.at(None) is row
    assert row.at(utc(SUN8, 20)).status == STATUS_IN_PLAY and row.at(utc(SUN8, 20)).status_text == IN_PLAY_TEXT
    assert row.at(utc(date(2026, 11, 10), 13, 31)).postponed, "two days later without a result: postponed"
    assert recoveries("DA", [late], None, utc(SUN8, 20)) == [] and len(recoveries("DA", [late], None, utc(date(2026, 11, 11), 9))) == 1
    # a scheduled match without a kick-off time: to play on its day, in play (result pending) from the next day
    no_time = Fixture("N1", SAT7, 13, "A", "H", "O")
    assert RoundMatch.from_fixture(no_time).at(utc(SAT7, 20)).to_play and RoundMatch.from_fixture(no_time).at(utc(SUN8, 9)).in_play
    assert kicked_off(no_time, utc(SUN8, 9)) and not kicked_off(no_time, utc(SAT7, 20))
    # the round of a kick-off-less fixture stops being a candidate at midnight of the next day (no three-hour grace)
    assert current_round("A", None, [no_time], now=rome_midnight(SUN8)) is None


def test_original_date_only_for_a_match_really_moved() -> None:
    """A pending match of a non-regular round on its round's own weekend keeps no original date; a moved one does."""
    played, fixtures = serie_c_a_round_13()
    partial = [fixture(f"A15-{k}", utc(SUN15, 13, 30), 15, "A", f"P{k}", f"Q{k}") for k in range(3)]
    rec = recoveries(Girone.A, [*fixtures, *partial], played, NOW)
    assert {m.match_id for m in rec} == {"A15-0", "A15-1", "A15-2"} and all(m.original_date is None for m in rec)
    from src.engine.tables import recoveries_table

    table = recoveries_table(rec)
    assert table["Data originale"].tolist() == ["dom 15/11"] * 3 and table["Stato"].tolist() == ["da giocare"] * 3
    # a round in progress keeps its in-block match even with a far recovery pending; the recovery is a straggler
    teams = [f"T{i:02d}" for i in range(20)]
    rows = played_frame([(f"T13-{k}", SAT7, 13, "B", teams[2 * k], teams[2 * k + 1], 0, 0) for k in range(7)])
    pending = [fixture("T13-7", utc(SUN8, 13, 30), 13, "B", teams[14], teams[15]), fixture("T13-8", utc(date(2026, 11, 18), 19), 13, "B", teams[16], teams[17])]
    pending.append(fixture("T13-9", utc(SAT7, 13, 30), 13, "B", teams[18], teams[19]))
    after_saturday = utc(SAT7, 20)
    view = current_round(Girone.B, rows, pending, now=after_saturday)
    assert view.round == 13 and view.in_progress(after_saturday - KICKOFF_GRACE) and view.fixture_ids == {"T13-7", "T13-8"}
    assert [m.match_id for m in recoveries(Girone.B, pending, rows, after_saturday)] == ["T13-8"]
