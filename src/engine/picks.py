"""Matchdays and "picks of the matchday": the most probable draws and both-teams-to-score matches.

This module is pure (no ``streamlit``): it only depends on the shared contracts
of :mod:`src.common.schemas`, on the official-round rule of
:mod:`src.engine.calendar` and on :class:`~src.engine.value.ValueOpportunity`,
so the console pipeline, the dashboard data layer and the tests share one
implementation.

Matchdays
---------
With a real provider the fixture list holds **every** not-yet-played match of
the season (hundreds of fixtures across the competitions, each with its
official round, FootyStats ``game_week``, and its kick-off). A "giornata" is
built from the **official rounds**, never from dates alone:

1. per competition, :func:`src.engine.calendar.upcoming_rounds` gives its
   **current round** (the smallest round with a match still to play in which at
   least half of the matches are not yet played, or which is being played in
   the current block of the football week) and the regular rounds after it;
2. the **first matchday** is the union of the current round of every
   competition, whatever their numbers (Serie B may be at round 11 while the
   Serie C gironi are at round 13) and whatever their dates: a round played
   from Friday to Monday stays in it with its Friday matches already played;
3. the **later matchdays** are the following rounds, grouped by the block of
   the football week of their anchor day (:func:`week_block`: weekend
   Friday-Monday or midweek Tuesday-Thursday), so the Serie B round of Saturday
   21 goes with the Serie C rounds of Sunday 22; a competition that has two
   rounds in one block (a double-header) opens a further matchday;
4. **recoveries never move a matchday**: a postponed match, the isolated
   recovery of a round already played (one match on a Wednesday, the other
   eight played the weekend before) and a match rescheduled more than
   ``max_days`` days from its round are not part of any matchday: they are
   listed apart (:func:`src.engine.calendar.recoveries`, the dashboard's
   *Recuperi e rinvii*) and never stretch a matchday's dates or label.

Each :class:`Matchday` carries the :class:`~src.engine.calendar.RoundView` of
every round it holds (:attr:`Matchday.round_views`: every match of the round as
it stands at the reference instant, played ones with their score, kicked-off
ones as ``'in corso / risultato in attesa'``), its ``fixture_ids`` are the
matches **still to play at that instant** (kick-off ahead of ``now``: the only
ones the picks rank and the value panel prices, a match already under way is
never offered as a pre-match bet) and its span is the span of its rounds.
The result is deterministic for a given input; the label spells out the date
span and the round of every competition involved, e.g. ``'Giornata del 6-9 nov
2026 · Girone A g.13, Girone B g.13, Girone C g.13, Serie B g.11'``.

Picks
-----
:func:`rank_picks` ranks the predictions of one matchday (or of every fixture)
by the **model probability** of one selection -- :func:`draw_picks` for ``X`` of
the 1X2 market, :func:`btts_picks` for ``GG`` -- and attaches the best bookmaker
price of that selection, with its EV and flags, when the value engine evaluated
it. Only the matches still to play of the matchday are ranked (not the ones
that have kicked off at the matchday's reference instant). The ranking is by
probability only: "most probable" is not "worth the price", which is what the
fair odds, the book odds and the EV columns are for. :func:`picks_frame` renders
the picks as a ``pandas`` frame with Italian labels.
"""
from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import date, datetime
from typing import Final

import pandas as pd

from src.common.logging_config import get_logger
from src.common.schemas import SELECTIONS, Fixture, Girone, Market, MatchPrediction
from src.engine.calendar import DEFAULT_MAX_DAYS, CompetitionCalendar, RoundView, rome_midnight, week_block
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
    """One "giornata" across the competitions: one official round per competition (see :func:`matchdays`).

    ``key`` is the ISO date of the first day of the matchday (``'2026-11-06'``;
    a ``#k`` suffix disambiguates the rare case of two matchdays starting the
    same day), ``label`` the Italian description shown in selectors and
    consoles, ``start``/``end`` the span of its rounds (played matches included,
    postponed matches and far-away recoveries excluded), ``rounds`` the official
    round of each competition involved (configuration order), ``fixture_ids``
    the ``match_id`` of the matches **still to play** (not kicked off at the
    reference instant the matchday was built at) and ``n_fixtures`` their
    number. ``round_views`` holds the :class:`~src.engine.calendar.RoundView` of
    each round (every match of it, played ones with their score; empty for a
    matchday built by hand). Equality compares every field but ``round_views``;
    the hash is the key's.
    """

    key: str
    label: str
    start: date
    end: date
    rounds: dict[Girone, int]
    fixture_ids: frozenset[str]
    n_fixtures: int = field(init=False)
    round_views: tuple[RoundView, ...] = field(default=(), compare=False)

    def __post_init__(self) -> None:
        if not self.key:
            raise ValueError("Matchday.key cannot be empty")
        if self.end < self.start:
            raise ValueError(f"Matchday {self.key}: end {self.end.isoformat()} precedes start {self.start.isoformat()}")
        object.__setattr__(self, "rounds", {Girone.parse(g): int(r) for g, r in self.rounds.items()})
        object.__setattr__(self, "fixture_ids", frozenset(str(i) for i in self.fixture_ids))
        object.__setattr__(self, "n_fixtures", len(self.fixture_ids))
        object.__setattr__(self, "round_views", tuple(self.round_views))

    def __hash__(self) -> int:
        return hash(self.key)

    def contains(self, fixture: Fixture | MatchPrediction | str) -> bool:
        """Whether a fixture, a prediction or a ``match_id`` is one of the matches still to play of this matchday."""
        if isinstance(fixture, MatchPrediction):
            return fixture.fixture.match_id in self.fixture_ids
        if isinstance(fixture, Fixture):
            return fixture.match_id in self.fixture_ids
        return str(fixture) in self.fixture_ids

    @property
    def competitions(self) -> tuple[Girone, ...]:
        """Competitions with a round in this matchday, configuration order."""
        return tuple(self.rounds)

    @property
    def n_played(self) -> int:
        """Matches of the matchday's rounds already played (``0`` without round views)."""
        return sum(v.n_played for v in self.round_views)

    @property
    def n_in_play(self) -> int:
        """Matches of the matchday's rounds that have kicked off without a result in the data (``0`` without round views)."""
        return sum(v.n_in_play for v in self.round_views)

    @property
    def n_matches(self) -> int:
        """Every match of the matchday's rounds (played, to play, postponed); ``n_fixtures`` without round views."""
        return sum(v.n_matches for v in self.round_views) if self.round_views else self.n_fixtures

    def view(self, girone: Girone | str) -> RoundView | None:
        """The :class:`~src.engine.calendar.RoundView` of one competition's round (``None`` when absent)."""
        wanted = Girone.parse(girone)
        return next((v for v in self.round_views if v.competition == wanted), None)

    def rounds_text(self) -> str:
        """``'Girone A g.14, Girone B g.14, Girone C g.14, Serie B g.13'``."""
        return ", ".join(f"{g.short} g.{r}" for g, r in self.rounds.items())


def _label(start: date, end: date, rounds: Mapping[Girone, int]) -> str:
    rounds_text = ", ".join(f"{g.short} g.{r}" for g, r in rounds.items())
    return f"Giornata {format_date_span(start, end)} · {rounds_text}"


def _regular_days(view: RoundView, max_days: int) -> list[date]:
    """Days of a round that make its span: every match but the postponed ones and the far-away stragglers."""
    far = {m.match_id for m in view.stragglers(max_days)}
    days = [m.date for m in view.matches if not m.postponed and m.match_id not in far]
    if not days and view.anchor is not None:
        days = [view.anchor]
    return days


def _build_matchday(views: Sequence[RoundView], max_days: int, seen_keys: dict[str, int]) -> Matchday:
    ordered = sorted(views, key=lambda v: _girone_order(v.competition))
    days = [d for v in ordered for d in _regular_days(v, max_days)]
    start, end = min(days), max(days)
    rounds = {v.competition: v.round for v in ordered}
    far = {m.match_id for v in ordered for m in v.stragglers(max_days)}
    ids = frozenset(m.match_id for v in ordered for m in v.to_play if m.match_id not in far)
    base = start.isoformat()
    seen_keys[base] = seen_keys.get(base, 0) + 1
    key = base if seen_keys[base] == 1 else f"{base}#{seen_keys[base]}"
    return Matchday(key=key, label=_label(start, end, rounds), start=start, end=end, rounds=rounds, fixture_ids=ids, round_views=tuple(ordered))


def matchdays(
    fixtures: Iterable[Fixture],
    *,
    as_of: date | None = None,
    max_days: int = DEFAULT_MAX_DAYS,
    gironi: Iterable[Girone | str] | None = None,
    played: pd.DataFrame | None = None,
    now: datetime | None = None,
) -> list[Matchday]:
    """Group the calendar into :class:`Matchday` objects (see the module docstring for the rule).

    ``played`` is the canonical frame of the **current season**'s played
    matches: with it every round view holds its played matches too and the
    half-pending rule counts them (without it only the fixtures are seen and a
    round's size is the largest round of the list). ``now`` is the reference
    instant of the official-round rule (aware; ``None`` drops the time
    condition, the right call when the list already is the upcoming schedule):
    the round views are taken at it, so a fixture that has kicked off is in its
    round as ``'in corso / risultato in attesa'`` and out of ``fixture_ids``;
    ``as_of`` drops the fixtures dated before it and, without ``now``, sets the
    reference instant to its midnight in ``Europe/Rome``. ``max_days`` (>= 0)
    is how far from its round's anchor day a match still to play may be and
    still belong to the matchday; farther away it is a straggler, listed with
    the recoveries. ``gironi`` keeps only those competitions (:func:`in_gironi`;
    ``None`` = every competition): the matchdays of one section of the
    dashboard are built from its own competitions, so their keys, labels and
    rounds never mention another section. The current matchday comes first;
    the result is empty when no competition has a round to play.
    """
    if int(max_days) < 0:
        raise ValueError(f"max_days must be >= 0, got {max_days}")
    max_days = int(max_days)
    wanted = None if gironi is None else {Girone.parse(g) for g in gironi}
    pending = [fx for fx in fixtures if (as_of is None or fx.date >= as_of) and (wanted is None or Girone.parse(fx.girone) in wanted)]
    if not pending:
        return []
    reference = now if now is not None else (rome_midnight(as_of) if as_of is not None else None)
    by_competition: dict[Girone, list[Fixture]] = defaultdict(list)
    for fx in pending:
        by_competition[Girone.parse(fx.girone)].append(fx)
    sequences: dict[Girone, list[RoundView]] = {}
    for g in sorted(by_competition, key=_girone_order):
        ahead = CompetitionCalendar.build(g, played, by_competition[g]).upcoming(reference)
        if ahead:
            sequences[g] = ahead
    if not sequences:
        return []

    seen_keys: dict[str, int] = {}
    out = [_build_matchday([seq[0] for seq in sequences.values()], max_days, seen_keys)]
    # the following rounds -> one group per block of the football week; a second round of the same competition in a
    # block (a double-header) opens another group of that block
    by_block: dict[date, list[RoundView]] = defaultdict(list)
    for seq in sequences.values():
        for view in seq[1:]:
            by_block[week_block(view.anchor)[0]].append(view)
    for block in sorted(by_block):
        groups: list[list[RoundView]] = []
        for view in sorted(by_block[block], key=lambda v: (v.anchor, v.round, _girone_order(v.competition))):
            target = next((grp for grp in groups if all(v.competition != view.competition for v in grp)), None)
            if target is None:
                target = []
                groups.append(target)
            target.append(view)
        out.extend(_build_matchday(grp, max_days, seen_keys) for grp in groups)
    log.debug(
        "matchdays grouped",
        extra={
            "n_fixtures": len(pending),
            "n_matchdays": len(out),
            "first": out[0].key,
            "current_rounds": {g.value: seq[0].round for g, seq in sequences.items()},
            "now": None if reference is None else str(reference),
        },
    )
    return out


def next_matchday(
    fixtures: Iterable[Fixture],
    *,
    as_of: date | None = None,
    gironi: Iterable[Girone | str] | None = None,
    played: pd.DataFrame | None = None,
    now: datetime | None = None,
) -> Matchday | None:
    """The first matchday of :func:`matchdays` (the current round of every competition), ``None`` without a round to play."""
    groups = matchdays(fixtures, as_of=as_of, gironi=gironi, played=played, now=now)
    return groups[0] if groups else None


def fixtures_of(matchday: Matchday, fixtures: Iterable[Fixture]) -> list[Fixture]:
    """The fixtures still to play of ``matchday``, sorted by date, competition order and label."""
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
