"""Official rounds of the calendar: the current round of a competition, any round, recoveries and postponements.

This module is pure (no ``streamlit``): it reads the canonical played-match
frame of the **current season** (:data:`src.common.schemas.MATCH_COLUMNS`) and
the fixture list (:class:`~src.common.schemas.Fixture`, with the official round
``round`` = FootyStats ``game_week``, the UTC ``kickoff`` and the ``status``
``'scheduled'`` / ``'postponed'``) and answers the questions the dashboard and
the console ask about the calendar:

* :func:`current_round` -- the round of a competition that is being played or
  is about to be played, with **every** match of it: the ones already played
  (with their score), the ones in play or waiting for their result, the ones
  still to play (with their kick-off) and the postponed ones;
* :func:`round_view` / :func:`rounds_available` -- any round of a competition;
* :func:`upcoming_rounds` -- the current round and the regular rounds after it;
* :func:`recoveries` -- the postponed matches and the isolated recoveries that
  are *not* part of the current round (listed apart, they never move a matchday).

Status of a match at the reference instant
------------------------------------------
Every view is taken at a reference instant ``now`` (aware; ``None`` = no time
condition, the fixture list already is the upcoming schedule, as in the demo):

* ``'giocata'`` (:data:`STATUS_PLAYED`) -- a row of the played-match frame,
  with its score;
* ``'da giocare'`` (:data:`STATUS_TO_PLAY`) -- a scheduled fixture whose
  kick-off is still ahead of ``now`` (without a kick-off time: dated today or
  later in Rome). Only these matches are predicted and priced as pre-match
  bets (:attr:`RoundView.fixture_ids`);
* ``'in corso'`` (:data:`STATUS_IN_PLAY`, shown as ``'in corso / risultato in
  attesa'``) -- a scheduled fixture whose kick-off has passed but whose result
  is not in the data yet (being played, or FootyStats has not published the
  score, or the local data predates the end of the match), and a fixture marked
  postponed whose kick-off passed less than :data:`RESULT_PENDING_WINDOW` (48
  hours) ago: FootyStats leaves a finished match ``incomplete`` for a while,
  so only a match still without a result two days later is a real postponement.
  Such a match stays in its own round, is never a recovery and is never picked;
* ``'rinviata'`` (:data:`STATUS_POSTPONED`) -- postponed: original kick-off
  passed more than 48 hours ago without a result.

The rule of the current round
-----------------------------
For one competition and a reference instant ``now``:

1. a round is a **candidate** when at least one of its fixtures is *scheduled*
   (not postponed) with a kick-off at or after ``now - 3h``
   (:data:`KICKOFF_GRACE`: a match that kicked off less than three hours ago is
   still being played). A fixture without a kick-off time (older CSVs, the
   simulator) counts from its date: it is upcoming while its day is not over;
2. a candidate is **regular** when at least half of its matches still to be
   played are pending -- the *pending* matches are the candidate-making ones
   of point 1 (to play, or kicked off less than three hours ago); the matches
   of a round are its played rows plus its fixtures, never fewer than the
   competition's typical round size (the largest round seen, so a round whose
   played rows are missing from the data still counts its full size), minus
   its **postponed** matches, which have no date and are no longer part of the
   round's own weekend -- or when it is a round **in progress**: some of its
   matches are played (or have kicked off) and some pending match falls in the
   same block of the football week (:func:`week_block`, weekend Friday-Monday
   or midweek Tuesday-Thursday) as the bulk of the round
   (:attr:`RoundView.anchor`, the median date). Without the second clause a
   Serie C round played Saturday-Monday would stop being "current" on Sunday
   lunchtime, when fewer than half of its ten matches are left;
3. the **current round** is the smallest regular candidate round. A round whose
   only pending match is a midweek recovery (eight of nine matches played the
   weekend before, one rescheduled ten days later; or, in a Serie D winter,
   four played, four postponed and one re-dated to a Wednesday) is neither
   half pending nor in progress, so the next round is the current one and the
   recovery is listed by :func:`recoveries`. A round with two matches played
   on Friday and eight to play over the weekend is the current round and is
   shown with all ten matches, the two Friday ones with their score.

Kick-off times are stored in UTC and shown in ``Europe/Rome``
(:func:`format_kickoff`: ``'sab 07/11 14:30'``, Italian weekday abbreviations;
``'sab 07/11'`` when only the date is known), so the switch between CEST and
CET on the last Sunday of October is handled by the time-zone database.
"""
from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from datetime import date, datetime, time, timedelta, timezone
from typing import Any, Final
from zoneinfo import ZoneInfo

import pandas as pd

from src.common.logging_config import get_logger
from src.common.schemas import Fixture, Girone

__all__ = [
    "DEFAULT_MAX_DAYS",
    "FIXTURE_POSTPONED",
    "FIXTURE_SCHEDULED",
    "IN_PLAY_TEXT",
    "ITALIAN_WEEKDAYS",
    "KICKOFF_GRACE",
    "MATCH_STATUSES",
    "RESULT_PENDING_WINDOW",
    "ROME",
    "STATUS_IN_PLAY",
    "STATUS_PLAYED",
    "STATUS_POSTPONED",
    "STATUS_TO_PLAY",
    "CompetitionCalendar",
    "RoundMatch",
    "RoundView",
    "current_round",
    "format_day",
    "format_kickoff",
    "kicked_off",
    "recoveries",
    "rome_midnight",
    "round_view",
    "rounds_available",
    "to_rome",
    "upcoming_rounds",
    "week_block",
]

log = get_logger(__name__)

#: Time zone of every date and time shown to the user.
ROME: Final[ZoneInfo] = ZoneInfo("Europe/Rome")
#: Italian weekday abbreviations, Monday first (``date.weekday()`` order).
ITALIAN_WEEKDAYS: Final[tuple[str, ...]] = ("lun", "mar", "mer", "gio", "ven", "sab", "dom")
#: Status of a match in a :class:`RoundView` (the ``Stato`` column of the matchday table).
STATUS_PLAYED: Final[str] = "giocata"
STATUS_TO_PLAY: Final[str] = "da giocare"
STATUS_POSTPONED: Final[str] = "rinviata"
#: A scheduled match whose kick-off has passed without a result in the data (in play, or the result not published yet).
STATUS_IN_PLAY: Final[str] = "in corso"
#: The ``Stato`` cell of a :data:`STATUS_IN_PLAY` match.
IN_PLAY_TEXT: Final[str] = "in corso / risultato in attesa"
MATCH_STATUSES: Final[tuple[str, ...]] = (STATUS_PLAYED, STATUS_TO_PLAY, STATUS_IN_PLAY, STATUS_POSTPONED)
#: :attr:`Fixture.status` values of the fixture list.
FIXTURE_SCHEDULED: Final[str] = "scheduled"
FIXTURE_POSTPONED: Final[str] = "postponed"
#: A scheduled match that kicked off less than this long ago is still being played (it keeps its round a candidate).
KICKOFF_GRACE: Final[timedelta] = timedelta(hours=3)
#: A match marked postponed whose kick-off passed less than this long ago is read as "result pending", not as postponed:
#: FootyStats leaves a finished match ``incomplete`` until it publishes the score, which in Serie D can take many hours.
RESULT_PENDING_WINDOW: Final[timedelta] = timedelta(hours=48)
#: A pending match dated more than this many days from its round's anchor is a straggler (see :func:`recoveries`).
DEFAULT_MAX_DAYS: Final[int] = 7

#: ``date.weekday()`` values (Monday = 0) of the weekend block of a football week: Friday, Saturday, Sunday, Monday.
_WEEKEND_WEEKDAYS: Final[frozenset[int]] = frozenset({4, 5, 6, 0})


# =============================================================================
# Time helpers
# =============================================================================
def to_rome(moment: datetime) -> datetime:
    """``moment`` in ``Europe/Rome``; a naive value is read as UTC (the storage convention of the kick-offs)."""
    if moment.tzinfo is None:
        moment = moment.replace(tzinfo=timezone.utc)
    return moment.astimezone(ROME)


def rome_midnight(day: date) -> datetime:
    """The first instant of ``day`` in ``Europe/Rome`` (aware), e.g. the reference instant of an ``as_of`` date."""
    return datetime.combine(day, time(0, 0), tzinfo=ROME)


def _aware(moment: datetime | None) -> datetime | None:
    """``moment`` as an aware UTC instant (naive = UTC), ``None`` unchanged."""
    if moment is None:
        return None
    if moment.tzinfo is None:
        return moment.replace(tzinfo=timezone.utc)
    return moment.astimezone(timezone.utc)


def format_day(day: date) -> str:
    """``'sab 07/11'``: Italian weekday abbreviation, day and month."""
    return f"{ITALIAN_WEEKDAYS[day.weekday()]} {day:%d/%m}"


def format_kickoff(kickoff: datetime | None, day: date | None = None) -> str:
    """The kick-off as shown in tables: ``'sab 07/11 14:30'`` in ``Europe/Rome``.

    Without a kick-off time (older fixture CSVs, played matches, the simulator)
    the date alone is shown, ``'sab 07/11'``; ``''`` when neither is known. A
    naive ``kickoff`` is read as UTC.
    """
    if kickoff is not None:
        local = to_rome(kickoff)
        return f"{ITALIAN_WEEKDAYS[local.weekday()]} {local:%d/%m %H:%M}"
    if day is not None:
        return format_day(day)
    return ""


def week_block(day: date) -> tuple[date, date]:
    """``(first day, last day)`` of the block of the football week that holds ``day``.

    A football week has two blocks: the **weekend** block, Friday to Monday
    (the regular rounds: Serie B from Friday, Serie C from Saturday, posticipi
    on Monday), and the **midweek** block, Tuesday to Thursday (turni
    infrasettimanali and recoveries).
    """
    weekday = day.weekday()
    if weekday in _WEEKEND_WEEKDAYS:
        start = day - timedelta(days=(weekday - 4) % 7)  # Friday 0, Saturday 1, Sunday 2, Monday 3 days back
        return start, start + timedelta(days=3)
    start = day - timedelta(days=weekday - 1)  # Tuesday 0, Wednesday 1, Thursday 2 days back
    return start, start + timedelta(days=2)


def _kicked_off(kickoff: datetime | None, day: date, now: datetime) -> bool:
    """Whether a match has started at ``now`` (aware): kick-off at or before ``now``; without a time, dated before ``now``'s day in Rome."""
    if kickoff is not None:
        return kickoff <= now
    return day < to_rome(now).date()


def kicked_off(fixture: Fixture, now: datetime | None) -> bool:
    """Whether ``fixture`` has kicked off at the reference instant ``now`` (always ``False`` for ``now=None``).

    The kick-off decides when it is known (a naive one is UTC); without it the
    fixture has started once its day in Rome is over. The pre-match views (the
    picks, the value panel, the predictions export) leave such a fixture out: a
    price computed before the match is no bet once the match is under way.
    """
    if now is None:
        return False
    kickoff = _aware(getattr(fixture, "kickoff", None))
    day = to_rome(kickoff).date() if kickoff is not None else fixture.date
    return _kicked_off(kickoff, day, _aware(now))


def _fixture_status(fixture: Fixture) -> str:
    """:data:`FIXTURE_POSTPONED` or :data:`FIXTURE_SCHEDULED` (the default for a fixture without a status)."""
    status = str(getattr(fixture, "status", FIXTURE_SCHEDULED) or FIXTURE_SCHEDULED).strip().lower()
    return FIXTURE_POSTPONED if status == FIXTURE_POSTPONED else FIXTURE_SCHEDULED


def _lower_median(days: Sequence[date]) -> date:
    """Lower median of ``days`` (non-empty): the day most of a round is played, insensitive to a lone straggler."""
    ordered = sorted(days)
    return ordered[(len(ordered) - 1) // 2]


# =============================================================================
# Rows and rounds
# =============================================================================
@dataclass(frozen=True)
class RoundMatch:
    """One match of a round as the calendar shows it.

    ``status`` is :data:`STATUS_PLAYED` (``home_goals`` / ``away_goals`` set),
    :data:`STATUS_TO_PLAY`, :data:`STATUS_IN_PLAY` (kicked off, no result in the
    data yet: see :meth:`at`) or :data:`STATUS_POSTPONED`. ``kickoff`` is the
    aware UTC kick-off when the source gives one (``None`` for played rows of
    the canonical frame, which store the day only); ``date`` is the day in
    ``Europe/Rome`` when the kick-off is known, the stored day otherwise.
    ``original_date`` is set on the rows :func:`recoveries` returns, only for a
    match that really left its slot: the day it was originally due (its own day
    for a postponed match, the day the rest of its round was played for a match
    rescheduled outside its round's block of the football week); ``None``
    otherwise.
    """

    match_id: str
    competition: Girone
    round: int
    date: date
    kickoff: datetime | None
    home_team: str
    away_team: str
    status: str
    home_goals: int | None = None
    away_goals: int | None = None
    original_date: date | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "competition", Girone.parse(self.competition))
        object.__setattr__(self, "round", int(self.round))
        object.__setattr__(self, "kickoff", _aware(self.kickoff))
        if self.status not in MATCH_STATUSES:
            raise ValueError(f"status must be one of {MATCH_STATUSES}, got {self.status!r}")
        if self.status == STATUS_PLAYED:
            if self.home_goals is None or self.away_goals is None:
                raise ValueError(f"played match {self.match_id} needs both scores")
            object.__setattr__(self, "home_goals", int(self.home_goals))
            object.__setattr__(self, "away_goals", int(self.away_goals))
        elif self.home_goals is not None or self.away_goals is not None:
            raise ValueError(f"match {self.match_id} is {self.status!r}: it cannot carry a score")

    @classmethod
    def from_fixture(cls, fixture: Fixture) -> "RoundMatch":
        """The row of a fixture: :data:`STATUS_POSTPONED` for a postponed one, :data:`STATUS_TO_PLAY` otherwise.

        ``date`` is the kick-off day in ``Europe/Rome`` when the fixture has a
        kick-off (the fixture's own ``date`` is the UTC day), the fixture's
        ``date`` otherwise.
        """
        kickoff = _aware(getattr(fixture, "kickoff", None))
        day = to_rome(kickoff).date() if kickoff is not None else fixture.date
        status = STATUS_POSTPONED if _fixture_status(fixture) == FIXTURE_POSTPONED else STATUS_TO_PLAY
        return cls(
            match_id=str(fixture.match_id),
            competition=fixture.girone,
            round=int(fixture.round),
            date=day,
            kickoff=kickoff,
            home_team=str(fixture.home_team),
            away_team=str(fixture.away_team),
            status=status,
        )

    @property
    def label(self) -> str:
        """``'Home - Away'``."""
        return f"{self.home_team} - {self.away_team}"

    @property
    def played(self) -> bool:
        return self.status == STATUS_PLAYED

    @property
    def to_play(self) -> bool:
        """Scheduled and not yet kicked off (at the instant of the view): the only matches predicted and picked."""
        return self.status == STATUS_TO_PLAY

    @property
    def in_play(self) -> bool:
        """Kicked off without a result in the data (in play, or the result not published yet)."""
        return self.status == STATUS_IN_PLAY

    @property
    def postponed(self) -> bool:
        return self.status == STATUS_POSTPONED

    @property
    def score(self) -> str:
        """``'2-1'`` for a played match, ``''`` otherwise."""
        return f"{self.home_goals}-{self.away_goals}" if self.played else ""

    @property
    def status_text(self) -> str:
        """The ``Stato`` cell: ``'giocata 2-1'``, ``'da giocare'``, ``'in corso / risultato in attesa'`` or ``'rinviata'``."""
        if self.played:
            return f"{STATUS_PLAYED} {self.score}"
        return IN_PLAY_TEXT if self.in_play else self.status

    @property
    def kickoff_rome(self) -> datetime | None:
        """The kick-off in ``Europe/Rome`` (``None`` without a time)."""
        return None if self.kickoff is None else to_rome(self.kickoff)

    @property
    def kickoff_text(self) -> str:
        """``'sab 07/11 14:30'`` (``'sab 07/11'`` without a time), see :func:`format_kickoff`."""
        return format_kickoff(self.kickoff, self.date)

    @property
    def original_date_text(self) -> str:
        """``'dom 01/11'`` for :attr:`original_date`, ``''`` when unset."""
        return "" if self.original_date is None else format_day(self.original_date)

    def sort_key(self) -> tuple[date, datetime, str, str]:
        """Chronological order: day in Rome, kick-off (start of the day without one), label, ``match_id``."""
        moment = self.kickoff if self.kickoff is not None else rome_midnight(self.date).astimezone(timezone.utc)
        return (self.date, moment, self.label, self.match_id)

    def at(self, now: datetime | None) -> "RoundMatch":
        """This match as it stands at the reference instant ``now`` (``self`` when nothing changes; ``None`` = no time condition).

        * a match still :data:`STATUS_TO_PLAY` whose kick-off is at or before
          ``now`` (without a time: dated before ``now``'s day in Rome) becomes
          :data:`STATUS_IN_PLAY`: it is being played, or its result is not in
          the data yet;
        * a :data:`STATUS_POSTPONED` match whose kick-off passed less than
          :data:`RESULT_PENDING_WINDOW` ago becomes :data:`STATUS_IN_PLAY` too:
          the data source marks a match postponed as soon as its result is a
          few hours late, a real postponement is a match still without a result
          two days later.
        """
        if now is None or self.status in (STATUS_PLAYED, STATUS_IN_PLAY):
            return self
        moment = _aware(now)
        if self.status == STATUS_TO_PLAY and _kicked_off(self.kickoff, self.date, moment):
            return replace(self, status=STATUS_IN_PLAY)
        if self.status == STATUS_POSTPONED and self.kickoff is not None and moment - RESULT_PENDING_WINDOW < self.kickoff <= moment:
            return replace(self, status=STATUS_IN_PLAY)
        return self

    def upcoming(self, threshold: datetime | None) -> bool:
        """Whether this scheduled match kicks off at or after ``threshold``: still to play, or kicked off after ``threshold``.

        ``threshold`` is ``now -`` :data:`KICKOFF_GRACE` for a view taken at
        ``now``, so a match being played keeps its round a candidate. ``None``
        drops the time condition: every match still to play is upcoming.
        Without a kick-off time the grace cannot apply: the match is upcoming
        while its day (in Rome) is not before the day of ``now`` (``threshold``
        plus the grace). Played and postponed matches are never upcoming.
        """
        if threshold is None:
            return self.to_play
        if not (self.to_play or self.in_play):
            return False
        if self.kickoff is not None:
            return self.kickoff >= threshold
        return self.date >= to_rome(threshold + KICKOFF_GRACE).date()


@dataclass(frozen=True)
class RoundView:
    """Every match of one round of one competition, in chronological order.

    ``matches`` holds the played rows (with their score), the matches in play or
    waiting for their result, the matches still to play and the postponed ones
    (:class:`RoundMatch`, as they stand at the instant the view was taken, see
    :meth:`CompetitionCalendar.view`). The counts, the date span and the
    identifiers are derived from them; the hash is ``(competition, round)``.
    """

    competition: Girone
    round: int
    matches: list[RoundMatch] = field(default_factory=list)

    def __post_init__(self) -> None:
        object.__setattr__(self, "competition", Girone.parse(self.competition))
        object.__setattr__(self, "round", int(self.round))
        object.__setattr__(self, "matches", sorted(self.matches, key=RoundMatch.sort_key))

    def __hash__(self) -> int:
        return hash((self.competition.value, self.round))

    @property
    def label(self) -> str:
        """``'Girone A g.13'``."""
        return f"{self.competition.short} g.{self.round}"

    @property
    def played(self) -> list[RoundMatch]:
        return [m for m in self.matches if m.played]

    @property
    def to_play(self) -> list[RoundMatch]:
        return [m for m in self.matches if m.to_play]

    @property
    def in_play(self) -> list[RoundMatch]:
        """Matches kicked off without a result in the data (``'in corso / risultato in attesa'``)."""
        return [m for m in self.matches if m.in_play]

    @property
    def postponed(self) -> list[RoundMatch]:
        return [m for m in self.matches if m.postponed]

    @property
    def n_matches(self) -> int:
        return len(self.matches)

    @property
    def n_played(self) -> int:
        return sum(1 for m in self.matches if m.played)

    @property
    def n_to_play(self) -> int:
        return sum(1 for m in self.matches if m.to_play)

    @property
    def n_in_play(self) -> int:
        return sum(1 for m in self.matches if m.in_play)

    @property
    def n_postponed(self) -> int:
        return sum(1 for m in self.matches if m.postponed)

    @property
    def started(self) -> bool:
        """Some match of the round is played or has kicked off."""
        return any(m.played or m.in_play for m in self.matches)

    def pending(self, threshold: datetime | None = None) -> list[RoundMatch]:
        """The matches that keep the round going at ``threshold`` (:meth:`RoundMatch.upcoming`): to play, or kicked off after it."""
        return [m for m in self.matches if m.upcoming(threshold)]

    @property
    def match_ids(self) -> frozenset[str]:
        """``match_id`` of every match of the round (played, to play, postponed)."""
        return frozenset(m.match_id for m in self.matches)

    @property
    def fixture_ids(self) -> frozenset[str]:
        """``match_id`` of the matches still to play (not kicked off at the instant of the view): the ones predicted and picked."""
        return frozenset(m.match_id for m in self.matches if m.to_play)

    @property
    def anchor(self) -> date | None:
        """Lower median day of the non-postponed matches (the day most of the round is played); ``None`` for an empty round."""
        days = [m.date for m in self.matches if not m.postponed] or [m.date for m in self.matches]
        return _lower_median(days) if days else None

    @property
    def start(self) -> date | None:
        """First day of the non-postponed matches (``None`` for an empty round)."""
        days = [m.date for m in self.matches if not m.postponed] or [m.date for m in self.matches]
        return min(days) if days else None

    @property
    def end(self) -> date | None:
        """Last day of the non-postponed matches (``None`` for an empty round)."""
        days = [m.date for m in self.matches if not m.postponed] or [m.date for m in self.matches]
        return max(days) if days else None

    @property
    def is_complete(self) -> bool:
        """Every match played (nothing to play, nothing postponed)."""
        return bool(self.matches) and all(m.played for m in self.matches)

    def stragglers(self, max_days: int = DEFAULT_MAX_DAYS) -> list[RoundMatch]:
        """Matches still to play dated more than ``max_days`` days from :attr:`anchor` (rescheduled far from their round)."""
        anchor = self.anchor
        if anchor is None:
            return []
        return [m for m in self.matches if m.to_play and abs((m.date - anchor).days) > int(max_days)]

    def in_progress(self, threshold: datetime | None = None) -> bool:
        """A round being played: some match is played (or has kicked off) and a pending match falls in the block of the football week of :attr:`anchor`.

        ``threshold`` selects the pending matches (:meth:`pending`). A pending
        match far from the bulk of the round (a midweek recovery) does not keep
        the round in progress by itself, and does not stop it either while
        another pending match is still due in the round's own weekend. ``False``
        when nothing has started yet (a lone recovery would otherwise be "in
        progress" around itself), when nothing is pending or when the round is
        empty.
        """
        pending = self.pending(threshold)
        anchor = self.anchor
        if not pending or anchor is None or not self.started:
            return False
        first, last = week_block(anchor)
        return any(first <= m.date <= last for m in pending)


# =============================================================================
# One competition's calendar
# =============================================================================
def _as_day(value: Any) -> date:
    if isinstance(value, pd.Timestamp):
        return value.to_pydatetime().date()
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    return date.fromisoformat(str(value).strip()[:10])


def _played_rows(competition: Girone, played: pd.DataFrame | None) -> list[RoundMatch]:
    """The played matches of ``competition`` in ``played`` (a canonical match frame; ``None``/empty = none)."""
    if played is None or len(played) == 0:
        return []
    frame = played.loc[(played["girone"].astype(str) == competition.value).to_numpy(dtype=bool)]
    rows: list[RoundMatch] = []
    for row in frame[["match_id", "date", "round", "home_team", "away_team", "home_goals", "away_goals"]].itertuples(index=False):
        rows.append(
            RoundMatch(
                match_id=str(row.match_id),
                competition=competition,
                round=int(row.round),
                date=_as_day(row.date),
                kickoff=None,
                home_team=str(row.home_team),
                away_team=str(row.away_team),
                status=STATUS_PLAYED,
                home_goals=int(row.home_goals),
                away_goals=int(row.away_goals),
            )
        )
    return rows


@dataclass(frozen=True)
class CompetitionCalendar:
    """The rounds of one competition, built once from its played matches and its fixtures.

    ``by_round`` maps every round number to its matches (a fixture whose
    ``match_id`` is also a played row is dropped: the played row wins, a stale
    fixture list never shows a match twice); ``expected_size`` is the typical
    round size (the largest round seen), the floor of the size of any round in
    the half-pending rule. Build it with :meth:`build`; the module functions
    (:func:`current_round`, :func:`round_view`, ...) are thin wrappers.
    """

    competition: Girone
    by_round: Mapping[int, tuple[RoundMatch, ...]]
    expected_size: int

    @classmethod
    def build(cls, competition: Girone | str, played: pd.DataFrame | None, fixtures: Iterable[Fixture]) -> "CompetitionCalendar":
        """The calendar of ``competition`` from the current-season ``played`` frame and the fixture list (other competitions ignored)."""
        comp = Girone.parse(competition)
        rows = _played_rows(comp, played)
        played_ids = {r.match_id for r in rows}
        seen: set[str] = set()
        for fx in fixtures:
            if Girone.parse(fx.girone) != comp:
                continue
            mid = str(fx.match_id)
            if mid in played_ids or mid in seen:
                continue
            seen.add(mid)
            rows.append(RoundMatch.from_fixture(fx))
        grouped: dict[int, list[RoundMatch]] = defaultdict(list)
        for r in rows:
            grouped[r.round].append(r)
        by_round = {number: tuple(sorted(group, key=RoundMatch.sort_key)) for number, group in sorted(grouped.items())}
        expected = max((len(group) for group in by_round.values()), default=0)
        return cls(competition=comp, by_round=by_round, expected_size=expected)

    def rounds(self) -> list[int]:
        """Every round with at least one match (played, to play or postponed), ascending."""
        return list(self.by_round)

    def view(self, round_number: int, now: datetime | None = None) -> RoundView:
        """The :class:`RoundView` of ``round_number`` at the reference instant ``now`` (without matches when the round is unknown).

        Every match is taken as it stands at ``now`` (:meth:`RoundMatch.at`): a
        fixture that has kicked off is ``'in corso / risultato in attesa'``, not
        ``'da giocare'``. ``None`` keeps the stored statuses.
        """
        matches = [m.at(now) for m in self.by_round.get(int(round_number), ())]
        return RoundView(self.competition, int(round_number), matches)

    def is_candidate(self, round_number: int, threshold: datetime | None) -> bool:
        """At least one scheduled match of the round kicks off at or after ``threshold`` (``None`` = any match still to play)."""
        now = None if threshold is None else _aware(threshold) + KICKOFF_GRACE
        return bool(self.view(round_number, now).pending(threshold))

    def is_regular(self, view: RoundView, threshold: datetime | None = None) -> bool:
        """At least half of the round's matches with a date are pending at ``threshold``, or the round is in progress (see the module docstring).

        The postponed matches are left out of the count: they have no date and
        no longer belong to the round's own weekend, so a round with four
        played, four postponed and one match re-dated to a Wednesday is not
        "half pending" because of its postponements.
        """
        pending = len(view.pending(threshold))
        total = max(view.n_matches, self.expected_size) - view.n_postponed
        return (total > 0 and 2 * pending >= total) or view.in_progress(threshold)

    def upcoming(self, now: datetime | None) -> list[RoundView]:
        """The current round first, then every later regular candidate round, ascending; ``[]`` when none.

        ``now`` is the reference instant (aware; a naive value is read as UTC);
        ``None`` drops the time condition (every scheduled match counts as
        upcoming), the right call for a fixture list that already is the
        upcoming schedule (the simulator). The views are taken at ``now``.
        """
        reference = None if now is None else _aware(now)
        threshold = None if reference is None else reference - KICKOFF_GRACE
        out: list[RoundView] = []
        for number in self.by_round:
            view = self.view(number, reference)
            if not view.pending(threshold):
                continue
            if self.is_regular(view, threshold):
                out.append(view)
        return out

    def current(self, now: datetime | None) -> RoundView | None:
        """The current round (the smallest regular candidate round), ``None`` when no round is pending."""
        ahead = self.upcoming(now)
        return ahead[0] if ahead else None

    def recoveries(self, now: datetime | None, *, max_days: int = DEFAULT_MAX_DAYS) -> list[RoundMatch]:
        """Matches pending outside the regular rounds, chronological, as they stand at ``now``.

        * every postponed match (any round, also of the current one);
        * every match still to play -- not kicked off at ``now`` -- of a round
          that is not regular (the isolated recovery of a round already played)
          or that lies before the current round;
        * the stragglers of the regular rounds (:meth:`RoundView.stragglers`:
          rescheduled more than ``max_days`` days from their round).

        A match that has kicked off without a result in the data is never a
        recovery: it stays in its own round as ``'in corso / risultato in
        attesa'``. ``original_date`` is the match's own day for a postponed
        match; for a scheduled one it is the median day of the round's played
        (or kicked-off) matches -- its anchor when none -- only when the match
        falls outside that day's block of the football week (it really was
        moved), ``None`` otherwise.
        """
        reference = None if now is None else _aware(now)
        regular = {v.round: v for v in self.upcoming(reference)}
        out: list[RoundMatch] = []
        for number in self.by_round:
            view = regular.get(number) or self.view(number, reference)
            held_days = [m.date for m in view.matches if m.played or m.in_play]
            round_day = _lower_median(held_days) if held_days else view.anchor
            stragglers = {m.match_id for m in view.stragglers(max_days)} if number in regular else set()
            for m in view.matches:
                if m.postponed:
                    out.append(_with_original(m, m.date))
                elif m.to_play and (number not in regular or m.match_id in stragglers):
                    out.append(_with_original(m, _moved_from(m.date, round_day)))
        out.sort(key=RoundMatch.sort_key)
        return out


def _moved_from(day: date, round_day: date | None) -> date | None:
    """``round_day`` when ``day`` lies outside its block of the football week (the match was moved), ``None`` otherwise."""
    if round_day is None:
        return None
    first, last = week_block(round_day)
    return None if first <= day <= last else round_day


def _with_original(match: RoundMatch, original: date | None) -> RoundMatch:
    return replace(match, original_date=original)


# =============================================================================
# Public functions
# =============================================================================
def current_round(
    competition: Girone | str,
    played: pd.DataFrame | None,
    fixtures: Iterable[Fixture],
    *,
    now: datetime | None,
) -> RoundView | None:
    """The current round of ``competition`` with every one of its matches (see the module docstring for the rule).

    ``played`` is the canonical frame of the **current season** (other
    competitions are ignored), ``fixtures`` the fixture list, ``now`` the
    reference instant (aware; ``None`` = no time condition): the matches are
    shown as they stand at ``now``. ``None`` when the competition has no
    pending round.
    """
    view = CompetitionCalendar.build(competition, played, fixtures).current(now)
    log.debug(
        "current round",
        extra={"competition": Girone.parse(competition).value, "round": None if view is None else view.round, "now": None if now is None else str(now)},
    )
    return view


def upcoming_rounds(
    competition: Girone | str,
    played: pd.DataFrame | None,
    fixtures: Iterable[Fixture],
    *,
    now: datetime | None,
) -> list[RoundView]:
    """The current round of ``competition`` followed by every later regular round with matches to play."""
    return CompetitionCalendar.build(competition, played, fixtures).upcoming(now)


def round_view(
    competition: Girone | str,
    round_number: int,
    played: pd.DataFrame | None,
    fixtures: Iterable[Fixture],
    *,
    now: datetime | None = None,
) -> RoundView:
    """Every match of round ``round_number`` of ``competition`` as it stands at ``now`` (an empty view for a round without matches).

    ``now=None`` keeps the stored statuses (a scheduled fixture stays
    ``'da giocare'`` whatever its kick-off).
    """
    return CompetitionCalendar.build(competition, played, fixtures).view(round_number, now)


def rounds_available(competition: Girone | str, played: pd.DataFrame | None, fixtures: Iterable[Fixture]) -> list[int]:
    """The rounds of ``competition`` that have at least one match (played, to play or postponed), ascending."""
    return CompetitionCalendar.build(competition, played, fixtures).rounds()


def recoveries(
    competition: Girone | str,
    fixtures: Iterable[Fixture],
    played: pd.DataFrame | None,
    now: datetime | None,
    *,
    max_days: int = DEFAULT_MAX_DAYS,
) -> list[RoundMatch]:
    """Postponed matches and isolated recoveries of ``competition`` outside its regular rounds (see :meth:`CompetitionCalendar.recoveries`)."""
    return CompetitionCalendar.build(competition, played, fixtures).recoveries(now, max_days=max_days)
