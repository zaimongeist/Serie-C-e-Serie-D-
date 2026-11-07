"""Matchdays and "picks of the matchday": the most probable draws and both-teams-to-score matches.

This module is pure (no ``streamlit``): it only depends on the shared contracts
of :mod:`src.common.schemas` and on :class:`~src.engine.value.ValueOpportunity`,
so the console pipeline, the dashboard data layer and the tests share one
implementation.

Matchdays
---------
With a real provider the fixture list holds **every** not-yet-played match of
the season (hundreds of fixtures across the four competitions, each with its
round number and date), so "the next round" is not one list but one round per
competition, and those rounds are neither numbered alike (Serie B may be at
round 13 while the Serie C gironi are at round 14) nor played every week
(Serie B stops for the international breaks, Serie C plays through and adds
midweek rounds). :func:`matchdays` therefore groups the fixtures **by
calendar**, not by round index:

1. per competition, the fixtures are grouped by round number and every round
   is **anchored** on the median date of its fixtures, the day most of the
   round is played: a lone postponed or anticipated match cannot move it;
2. a round with fewer fixtures than **half the competition's typical round
   size** (the largest round in the list: 10 for a 20-team competition, so
   fewer than 5) is a **recovery** group -- postponed matches still carrying
   their original round number, whatever their date -- and never counts as a
   regular round; inside a regular round the fixtures dated more than
   ``max_days`` from the anchor are stragglers and are treated the same way;
3. the regular rounds of every competition are placed in the **block** of the
   football week that holds their anchor, the weekend block (Friday to Monday)
   or the midweek block (Tuesday to Thursday), see :func:`week_block`; the
   rounds of different competitions in the same block form one matchday, so
   the Serie B round of Saturday 14 and the Serie C rounds of Sunday 15 are
   one "giornata" while the Serie B round of Saturday 21 (after a break) goes
   with the Serie C rounds of Sunday 22. A competition that does not play in a
   block is simply absent from that matchday; when a competition has two
   rounds in one block (a double-header) the second opens a further matchday
   in the same block;
4. recoveries and stragglers are attached to the **nearest** matchday by date
   (the earlier one on a tie), extending its date span, so no fixture is lost.

A round being played right now (only its last fixtures still pending) stays a
regular round while at least half of it is pending; its last few matches are
attached, like recoveries, to the nearest matchday. The result is
deterministic for a given fixture list: same input, same matchdays, same keys
and labels. The label spells out the date span and the round of every
competition involved, e.g. ``'Giornata del 7-9 nov 2026 · Girone A g.14,
Girone B g.14, Girone C g.14, Serie B g.13'``.

Picks
-----
:func:`rank_picks` ranks the predictions of one matchday (or of every fixture)
by the **model probability** of one selection -- :func:`draw_picks` for ``X`` of
the 1X2 market, :func:`btts_picks` for ``GG`` -- and attaches the best bookmaker
price of that selection, with its EV and flags, when the value engine evaluated
it. The ranking is by probability only: "most probable" is not "worth the
price", which is what the fair odds, the book odds and the EV columns are for.
:func:`picks_frame` renders the picks as a ``pandas`` frame with Italian labels.
"""
from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import date, timedelta
from typing import Final

import pandas as pd

from src.common.logging_config import get_logger
from src.common.schemas import SELECTIONS, Fixture, Girone, Market, MatchPrediction
from src.engine.value import ValueOpportunity

__all__ = [
    "BTTS_NOTE_BOTH_ATTACKS",
    "BTTS_NOTE_HIGH_XG",
    "DEFAULT_N_BTTS",
    "DEFAULT_N_DRAWS",
    "DRAW_NOTE_CLOSE_RATES",
    "DRAW_NOTE_FEW_GOALS",
    "ITALIAN_MONTHS",
    "PICKS_COLUMNS",
    "Matchday",
    "Pick",
    "btts_picks",
    "draw_picks",
    "fixtures_of",
    "format_date_span",
    "in_gironi",
    "matchdays",
    "next_matchday",
    "picks_frame",
    "rank_picks",
    "week_block",
]

log = get_logger(__name__)

#: Default size of the draw list (``X``) and of the both-teams-to-score list (``GG``).
DEFAULT_N_DRAWS: Final[int] = 5
DEFAULT_N_BTTS: Final[int] = 12

#: Italian month abbreviations used by :func:`format_date_span` (index 1 = January).
ITALIAN_MONTHS: Final[tuple[str, ...]] = ("", "gen", "feb", "mar", "apr", "mag", "giu", "lug", "ago", "set", "ott", "nov", "dic")

#: Italian labels of :func:`picks_frame`, in order.
PICKS_COLUMNS: Final[tuple[str, ...]] = (
    "#",
    "Competizione",
    "Data",
    "Partita",
    "Esito",
    "P. Modello (%)",
    "Quota Fair",
    "Quota Book",
    "Bookmaker",
    "EV (%)",
    "Flag",
    "Nota",
)
_PICKS_DTYPES: Final[dict[str, str]] = {
    "#": "int64",
    "Competizione": "string",
    "Data": "datetime64[ns]",
    "Partita": "string",
    "Esito": "string",
    "P. Modello (%)": "float64",
    "Quota Fair": "float64",
    "Quota Book": "float64",
    "Bookmaker": "string",
    "EV (%)": "float64",
    "Flag": "string",
    "Nota": "string",
}

#: Short Italian hints written in :attr:`Pick.note` (model-derived, never about the price).
DRAW_NOTE_CLOSE_RATES: Final[str] = "lambda e mu vicini"
DRAW_NOTE_FEW_GOALS: Final[str] = "pochi gol attesi"
BTTS_NOTE_HIGH_XG: Final[str] = "xG attese alte"
BTTS_NOTE_BOTH_ATTACKS: Final[str] = "entrambi gli attacchi sopra 1.2 gol attesi"
#: Thresholds behind the hints: |lambda - mu| for "close rates", lambda + mu for "few goals" / "high xG",
#: min(lambda, mu) for "both attacks".
_DRAW_CLOSE_RATES_GAP: Final[float] = 0.15
_DRAW_FEW_GOALS_TOTAL: Final[float] = 2.2
_BTTS_HIGH_XG_TOTAL: Final[float] = 3.0
_BTTS_BOTH_ATTACKS_MIN: Final[float] = 1.2


# =============================================================================
# Matchdays
# =============================================================================
def _girone_order(g: Girone) -> int:
    """Position of a competition in configuration order (``config/competitions.toml``)."""
    return list(Girone).index(Girone.parse(g))


def in_gironi(girone: Girone | str, gironi: Iterable[Girone | str] | None) -> bool:
    """Whether ``girone`` belongs to the restriction ``gironi`` (``None`` means no restriction).

    The shared competition filter of :func:`matchdays`, :func:`rank_picks` and
    the tables: the dashboard restricts every view to the competitions of the
    selected *section* (Serie B and Serie C, or Serie D) by passing their
    :class:`Girone` members (or codes) here. An empty iterable restricts to
    nothing, so a section without competitions yields empty views, never a
    silent fallback to the whole perimeter.
    """
    if gironi is None:
        return True
    wanted = {Girone.parse(g) for g in gironi}
    return Girone.parse(girone) in wanted


def _del(day: int) -> str:
    """Italian article before a day number: ``dell'`` for 1, 8 and 11 (vowel sound), ``del `` otherwise."""
    return "dell'" if day in (1, 8, 11) else "del "


def format_date_span(start: date, end: date) -> str:
    """Italian date span, ``'del 7-9 nov 2026'``, ``"dell'8 nov 2026"``, ``'del 30 ott-2 nov 2026'``.

    Months are abbreviated (:data:`ITALIAN_MONTHS`); the year is written once
    unless the span crosses a year (``'del 30 dic 2026-3 gen 2027'``).
    ``end`` before ``start`` raises :class:`ValueError`.
    """
    if end < start:
        raise ValueError(f"end {end.isoformat()} precedes start {start.isoformat()}")
    article = _del(start.day)
    if start == end:
        return f"{article}{start.day} {ITALIAN_MONTHS[start.month]} {start.year}"
    if (start.year, start.month) == (end.year, end.month):
        return f"{article}{start.day}-{end.day} {ITALIAN_MONTHS[start.month]} {start.year}"
    if start.year == end.year:
        return f"{article}{start.day} {ITALIAN_MONTHS[start.month]}-{end.day} {ITALIAN_MONTHS[end.month]} {start.year}"
    return (
        f"{article}{start.day} {ITALIAN_MONTHS[start.month]} {start.year}-"
        f"{end.day} {ITALIAN_MONTHS[end.month]} {end.year}"
    )


@dataclass(frozen=True)
class Matchday:
    """One "giornata" across the competitions: the rounds anchored in the same block of the football week.

    See :func:`matchdays` for the rule (weekend block Friday-Monday, midweek
    block Tuesday-Thursday). ``key`` is the ISO date of the earliest regular
    fixture (``'2026-11-08'``; a ``#k`` suffix disambiguates the rare case of
    two matchdays starting the same day), ``label`` the Italian description
    shown in selectors and consoles, ``start``/``end`` the date span of every
    fixture it holds (recoveries included), ``rounds`` the regular round number
    of each competition involved (configuration order), ``fixture_ids`` the
    ``match_id`` of every fixture (regular and recoveries) and ``n_fixtures``
    their number. Equality compares every field; the hash is the key's.
    """

    key: str
    label: str
    start: date
    end: date
    rounds: dict[Girone, int]
    fixture_ids: frozenset[str]
    n_fixtures: int = field(init=False)

    def __post_init__(self) -> None:
        if not self.key:
            raise ValueError("Matchday.key cannot be empty")
        if self.end < self.start:
            raise ValueError(f"Matchday {self.key}: end {self.end.isoformat()} precedes start {self.start.isoformat()}")
        object.__setattr__(self, "rounds", {Girone.parse(g): int(r) for g, r in self.rounds.items()})
        object.__setattr__(self, "fixture_ids", frozenset(str(i) for i in self.fixture_ids))
        object.__setattr__(self, "n_fixtures", len(self.fixture_ids))

    def __hash__(self) -> int:
        return hash(self.key)

    def contains(self, fixture: Fixture | MatchPrediction | str) -> bool:
        """Whether a fixture, a prediction or a ``match_id`` belongs to this matchday."""
        if isinstance(fixture, MatchPrediction):
            return fixture.fixture.match_id in self.fixture_ids
        if isinstance(fixture, Fixture):
            return fixture.match_id in self.fixture_ids
        return str(fixture) in self.fixture_ids

    @property
    def competitions(self) -> tuple[Girone, ...]:
        """Competitions with a regular round in this matchday, configuration order."""
        return tuple(self.rounds)

    def rounds_text(self) -> str:
        """``'Girone A g.14, Girone B g.14, Girone C g.14, Serie B g.13'``."""
        return ", ".join(f"{g.short} g.{r}" for g, r in self.rounds.items())


def _label(start: date, end: date, rounds: Mapping[Girone, int]) -> str:
    rounds_text = ", ".join(f"{g.short} g.{r}" for g, r in rounds.items())
    return f"Giornata {format_date_span(start, end)} · {rounds_text}"


#: ``date.weekday()`` values (Monday = 0) of the weekend block of a football week: Friday, Saturday, Sunday, Monday.
_WEEKEND_WEEKDAYS: Final[frozenset[int]] = frozenset({4, 5, 6, 0})


def week_block(day: date) -> tuple[date, date]:
    """``(first day, last day)`` of the block of the football week that holds ``day``.

    A football week has two blocks: the **weekend** block, Friday to Monday
    (the regular rounds: Serie B from Friday, Serie C from Saturday, posticipi
    on Monday), and the **midweek** block, Tuesday to Thursday (turni
    infrasettimanali and holiday rounds). Rounds of different competitions
    anchored in the same block form one matchday, whatever their round numbers.
    """
    weekday = day.weekday()
    if weekday in _WEEKEND_WEEKDAYS:
        start = day - timedelta(days=(weekday - 4) % 7)  # Friday 0, Saturday 1, Sunday 2, Monday 3 days back
        return start, start + timedelta(days=3)
    start = day - timedelta(days=weekday - 1)  # Tuesday 0, Wednesday 1, Thursday 2 days back
    return start, start + timedelta(days=2)


def _fixture_key(fx: Fixture) -> tuple[date, int, str, str]:
    """Total order of fixtures: date, competition order, label, ``match_id``."""
    return (fx.date, _girone_order(fx.girone), fx.label, fx.match_id)


def _median_date(fixtures: Sequence[Fixture]) -> date:
    """Lower median of the fixture dates: the day most of a round is played, insensitive to a lone straggler."""
    dates = sorted(f.date for f in fixtures)
    return dates[(len(dates) - 1) // 2]


@dataclass(frozen=True)
class _Round:
    """One regular round of one competition: its anchor date and the fixtures that stay with it."""

    girone: Girone
    number: int
    anchor: date
    fixtures: tuple[Fixture, ...]


def _split_rounds(girone: Girone, fixtures: Sequence[Fixture], max_days: int) -> tuple[list[_Round], list[Fixture]]:
    """One competition's fixtures -> its regular rounds (anchor order) and its leftovers (recoveries, stragglers).

    The typical round size is the largest round in the list; a round with
    fewer fixtures than half of it is a recovery group (every fixture is a
    leftover); inside a regular round the fixtures farther than ``max_days``
    from the anchor (the median date) are stragglers. A regular round always
    keeps at least the fixture played on the anchor date.
    """
    by_round: dict[int, list[Fixture]] = defaultdict(list)
    for fx in fixtures:
        by_round[int(fx.round)].append(fx)
    typical = max(len(group) for group in by_round.values())
    rounds: list[_Round] = []
    leftovers: list[Fixture] = []
    for number, group in sorted(by_round.items()):
        if 2 * len(group) < typical:
            leftovers.extend(group)  # postponed matches still numbered with their original round
            continue
        anchor = _median_date(group)
        members = [f for f in group if abs((f.date - anchor).days) <= max_days]
        leftovers.extend(f for f in group if abs((f.date - anchor).days) > max_days)
        rounds.append(_Round(girone, number, anchor, tuple(sorted(members, key=_fixture_key))))
    rounds.sort(key=lambda r: (r.anchor, r.number))
    return rounds, leftovers


@dataclass
class _Draft:
    """A matchday under construction: the regular rounds of one block plus the leftovers attached to it."""

    block: date
    rounds: dict[Girone, int] = field(default_factory=dict)
    regular: list[Fixture] = field(default_factory=list)
    attached: list[Fixture] = field(default_factory=list)

    @property
    def regular_start(self) -> date:
        return min(f.date for f in self.regular)

    @property
    def regular_end(self) -> date:
        return max(f.date for f in self.regular)

    def distance(self, day: date) -> int:
        """Days between ``day`` and the span of the regular fixtures (0 inside it)."""
        if day < self.regular_start:
            return (self.regular_start - day).days
        if day > self.regular_end:
            return (day - self.regular_end).days
        return 0

    def fixtures(self) -> list[Fixture]:
        return self.regular + self.attached


def matchdays(
    fixtures: Iterable[Fixture],
    *,
    as_of: date | None = None,
    max_days: int = 7,
    gironi: Iterable[Girone | str] | None = None,
) -> list[Matchday]:
    """Group the not-yet-played fixtures into :class:`Matchday` objects (see the module docstring for the rule).

    ``as_of`` drops the fixtures dated before it (already played or stale);
    ``None`` keeps every fixture given, which is the right call when the list is
    already the upcoming schedule. ``max_days`` (>= 0) is how far from its
    round's anchor date a fixture may be and still belong to the round; farther
    away it is a straggler attached to the nearest matchday. ``gironi`` keeps
    only the fixtures of those competitions (:func:`in_gironi`; ``None`` = every
    competition): the matchdays of one section of the dashboard (Serie D alone)
    are grouped from its own fixtures, so their keys, labels and rounds never
    mention another section. The result is ordered by football-week block (the
    next matchday first) and empty for no fixtures.
    """
    if int(max_days) < 0:
        raise ValueError(f"max_days must be >= 0, got {max_days}")
    wanted = None if gironi is None else {Girone.parse(g) for g in gironi}
    pending = [fx for fx in fixtures if (as_of is None or fx.date >= as_of) and (wanted is None or Girone.parse(fx.girone) in wanted)]
    if not pending:
        return []
    by_competition: dict[Girone, list[Fixture]] = defaultdict(list)
    for fx in pending:
        by_competition[Girone.parse(fx.girone)].append(fx)
    rounds: list[_Round] = []
    leftovers: list[Fixture] = []
    for g in sorted(by_competition, key=_girone_order):
        comp_rounds, comp_leftovers = _split_rounds(g, by_competition[g], int(max_days))
        rounds.extend(comp_rounds)
        leftovers.extend(comp_leftovers)

    # regular rounds -> one draft per block; a second round of the same competition in a block opens another draft
    by_block: dict[date, list[_Round]] = defaultdict(list)
    for r in rounds:
        by_block[week_block(r.anchor)[0]].append(r)
    drafts: list[_Draft] = []
    for block in sorted(by_block):
        block_drafts: list[_Draft] = []
        for r in sorted(by_block[block], key=lambda r: (r.anchor, r.number, _girone_order(r.girone))):
            target = next((d for d in block_drafts if r.girone not in d.rounds), None)
            if target is None:
                target = _Draft(block)
                block_drafts.append(target)
            target.rounds[r.girone] = r.number
            target.regular.extend(r.fixtures)
        drafts.extend(block_drafts)

    # leftovers -> the nearest draft by date (the earlier one on a tie)
    for fx in sorted(leftovers, key=_fixture_key):
        nearest = min(range(len(drafts)), key=lambda i: (drafts[i].distance(fx.date), i))
        drafts[nearest].attached.append(fx)

    out: list[Matchday] = []
    seen_keys: dict[str, int] = {}
    for draft in drafts:
        held = draft.fixtures()
        start, end = min(f.date for f in held), max(f.date for f in held)
        base = draft.regular_start.isoformat()
        seen_keys[base] = seen_keys.get(base, 0) + 1
        key = base if seen_keys[base] == 1 else f"{base}#{seen_keys[base]}"
        ordered_rounds = {g: draft.rounds[g] for g in sorted(draft.rounds, key=_girone_order)}
        out.append(
            Matchday(
                key=key,
                label=_label(start, end, ordered_rounds),
                start=start,
                end=end,
                rounds=ordered_rounds,
                fixture_ids=frozenset(f.match_id for f in held),
            )
        )
    log.debug(
        "matchdays grouped",
        extra={
            "n_fixtures": len(pending),
            "n_matchdays": len(out),
            "n_leftovers": len(leftovers),
            "first": out[0].key if out else None,
            "competitions": [g.value for g in by_competition],
        },
    )
    return out


def next_matchday(
    fixtures: Iterable[Fixture],
    *,
    as_of: date | None = None,
    gironi: Iterable[Girone | str] | None = None,
) -> Matchday | None:
    """The first matchday of :func:`matchdays` (the next block with a regular round), ``None`` without fixtures."""
    groups = matchdays(fixtures, as_of=as_of, gironi=gironi)
    return groups[0] if groups else None


def fixtures_of(matchday: Matchday, fixtures: Iterable[Fixture]) -> list[Fixture]:
    """The fixtures of ``matchday``, sorted by date, competition order and label."""
    return sorted(
        (fx for fx in fixtures if fx.match_id in matchday.fixture_ids),
        key=lambda fx: (fx.date, _girone_order(fx.girone), fx.label, fx.match_id),
    )


# =============================================================================
# Picks
# =============================================================================
@dataclass(frozen=True)
class Pick:
    """One ranked selection of the matchday: the model's view plus the best bookmaker price, when quoted.

    ``p_model``/``fair_odds`` come from the prediction; ``book_odds``,
    ``bookmaker``, ``ev`` and ``flags`` from the best-priced
    :class:`~src.engine.value.ValueOpportunity` of the same selection (``None``
    / empty when no bookmaker quotes it). ``note`` is ``''`` or a short Italian
    hint derived from the expected goals (never from the price).
    """

    rank: int
    match_id: str
    date: date
    girone: Girone
    home_team: str
    away_team: str
    market: Market
    selection: str
    p_model: float
    fair_odds: float
    book_odds: float | None = None
    bookmaker: str | None = None
    ev: float | None = None
    flags: tuple[str, ...] = ()
    note: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(self, "girone", Girone.parse(self.girone))
        object.__setattr__(self, "market", Market(self.market))
        object.__setattr__(self, "flags", tuple(str(f) for f in self.flags))
        if self.selection not in SELECTIONS[self.market]:
            raise ValueError(f"{self.selection!r} is not a selection of {self.market.value}")
        if not 0.0 <= self.p_model <= 1.0:
            raise ValueError(f"p_model must be in [0, 1], got {self.p_model}")
        if self.rank < 1:
            raise ValueError(f"rank must be >= 1, got {self.rank}")

    @property
    def label(self) -> str:
        """``'Home - Away'``."""
        return f"{self.home_team} - {self.away_team}"


def _note_for(prediction: MatchPrediction, market: Market, selection: str) -> str:
    """Italian hint for a draw or a both-teams-to-score pick, ``''`` for every other selection."""
    lam, mu = prediction.home_rate, prediction.away_rate
    hints: list[str] = []
    if market is Market.MATCH_ODDS and selection == "X":
        if abs(lam - mu) <= _DRAW_CLOSE_RATES_GAP:
            hints.append(DRAW_NOTE_CLOSE_RATES)
        if lam + mu <= _DRAW_FEW_GOALS_TOTAL:
            hints.append(DRAW_NOTE_FEW_GOALS)
    elif market is Market.BTTS and selection == "GG":
        if lam + mu >= _BTTS_HIGH_XG_TOTAL:
            hints.append(BTTS_NOTE_HIGH_XG)
        if min(lam, mu) >= _BTTS_BOTH_ATTACKS_MIN:
            hints.append(BTTS_NOTE_BOTH_ATTACKS)
    return ", ".join(hints)


def _best_prices(opportunities: Iterable[ValueOpportunity], market: Market, selection: str) -> dict[str, ValueOpportunity]:
    """Best-priced (highest ``book_odds``) opportunity per ``match_id`` for one selection."""
    best: dict[str, ValueOpportunity] = {}
    for o in opportunities:
        if o.market is not market or o.selection != selection:
            continue
        current = best.get(o.match_id)
        if current is None or o.book_odds > current.book_odds or (o.book_odds == current.book_odds and o.bookmaker < current.bookmaker):
            best[o.match_id] = o
    return best


def rank_picks(
    predictions: Iterable[MatchPrediction],
    *,
    market: Market,
    selection: str,
    n: int,
    matchday: Matchday | None = None,
    opportunities: Iterable[ValueOpportunity] = (),
    min_probability: float = 0.0,
    gironi: Iterable[Girone | str] | None = None,
) -> list[Pick]:
    """The ``n`` predictions with the highest model probability of ``selection`` in ``market``.

    Only the fixtures of ``matchday`` are considered (every prediction when
    ``None``), only those of the competitions in ``gironi`` when given (the
    picks of one dashboard section are ranked within that section, see
    :func:`in_gironi`), and those with ``p_model < min_probability`` are
    dropped. The order is probability descending, then fair odds ascending,
    then date, competition order, label and ``match_id`` (a total order, so
    ties resolve the same way every time). When ``opportunities`` quote the
    selection, the best (highest) book price is attached with its bookmaker, EV
    and flags. ``n <= 0`` yields an empty list; ``n`` above the available
    predictions yields them all.
    """
    market = Market(market)
    if selection not in SELECTIONS[market]:
        raise ValueError(f"{selection!r} is not a selection of {market.value} (expected one of {SELECTIONS[market]})")
    if n <= 0:
        return []
    wanted = None if gironi is None else {Girone.parse(g) for g in gironi}
    candidates = [
        p
        for p in predictions
        if (matchday is None or matchday.contains(p))
        and (wanted is None or Girone.parse(p.fixture.girone) in wanted)
        and float(p.probabilities[market][selection]) >= float(min_probability)
    ]
    candidates.sort(
        key=lambda p: (
            -float(p.probabilities[market][selection]),
            float(p.fair_odds[market][selection]),
            p.fixture.date,
            _girone_order(p.fixture.girone),
            p.fixture.label,
            p.fixture.match_id,
        )
    )
    best = _best_prices(opportunities, market, selection)
    picks: list[Pick] = []
    for rank, p in enumerate(candidates[: int(n)], start=1):
        fx = p.fixture
        quote = best.get(fx.match_id)
        picks.append(
            Pick(
                rank=rank,
                match_id=fx.match_id,
                date=fx.date,
                girone=fx.girone,
                home_team=fx.home_team,
                away_team=fx.away_team,
                market=market,
                selection=selection,
                p_model=float(p.probabilities[market][selection]),
                fair_odds=float(p.fair_odds[market][selection]),
                book_odds=None if quote is None else float(quote.book_odds),
                bookmaker=None if quote is None else quote.bookmaker,
                ev=None if quote is None else float(quote.ev),
                flags=() if quote is None else tuple(quote.flags),
                note=_note_for(p, market, selection),
            )
        )
    log.debug(
        "picks ranked",
        extra={
            "market": market.value,
            "selection": selection,
            "n_requested": int(n),
            "n_candidates": len(candidates),
            "n_picks": len(picks),
            "matchday": None if matchday is None else matchday.key,
            "gironi": None if wanted is None else sorted(g.value for g in wanted),
            "n_quoted": sum(1 for p in picks if p.book_odds is not None),
        },
    )
    return picks


def draw_picks(
    predictions: Iterable[MatchPrediction],
    *,
    n: int = DEFAULT_N_DRAWS,
    matchday: Matchday | None = None,
    opportunities: Iterable[ValueOpportunity] = (),
    gironi: Iterable[Girone | str] | None = None,
) -> list[Pick]:
    """The ``n`` most probable draws (``X`` of the 1X2 market), see :func:`rank_picks` (``gironi`` = one section's competitions)."""
    return rank_picks(predictions, market=Market.MATCH_ODDS, selection="X", n=n, matchday=matchday, opportunities=opportunities, gironi=gironi)


def btts_picks(
    predictions: Iterable[MatchPrediction],
    *,
    n: int = DEFAULT_N_BTTS,
    matchday: Matchday | None = None,
    opportunities: Iterable[ValueOpportunity] = (),
    gironi: Iterable[Girone | str] | None = None,
) -> list[Pick]:
    """The ``n`` most probable both-teams-to-score matches (``GG``), see :func:`rank_picks` (``gironi`` = one section's competitions)."""
    return rank_picks(predictions, market=Market.BTTS, selection="GG", n=n, matchday=matchday, opportunities=opportunities, gironi=gironi)


def picks_frame(picks: Iterable[Pick]) -> pd.DataFrame:
    """Italian table of the picks, columns exactly :data:`PICKS_COLUMNS` (an empty frame keeps them).

    Probabilities and EV in percent (1 and 2 decimals), odds with 2 decimals;
    ``Quota Book``/``EV (%)`` are ``NaN`` and ``Bookmaker`` is ``<NA>`` for a
    selection no bookmaker quotes; ``Flag`` joins the flags with ``' | '``.
    """
    rows = [
        {
            "#": int(p.rank),
            "Competizione": p.girone.short,
            "Data": pd.Timestamp(p.date),
            "Partita": p.label,
            "Esito": p.selection,
            "P. Modello (%)": round(p.p_model * 100.0, 1),
            "Quota Fair": round(p.fair_odds, 2),
            "Quota Book": float("nan") if p.book_odds is None else round(p.book_odds, 2),
            "Bookmaker": pd.NA if p.bookmaker is None else p.bookmaker,
            "EV (%)": float("nan") if p.ev is None else round(p.ev * 100.0, 2),
            "Flag": " | ".join(p.flags),
            "Nota": p.note,
        }
        for p in picks
    ]
    if not rows:
        empty = pd.DataFrame({c: pd.Series(dtype=t) for c, t in _PICKS_DTYPES.items()})
        return empty[list(PICKS_COLUMNS)]
    frame = pd.DataFrame(rows, columns=list(PICKS_COLUMNS))
    frame = frame.assign(Data=pd.to_datetime(frame["Data"]).astype("datetime64[ns]"))
    for column, dtype in _PICKS_DTYPES.items():
        if dtype in ("string", "float64", "int64"):
            frame[column] = frame[column].astype(dtype)
    return frame.reset_index(drop=True)
