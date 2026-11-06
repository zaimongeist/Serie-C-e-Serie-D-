"""Shared data contracts for the Serie C quant pipeline.

Every module (ingestion, engine, value, dashboard) exchanges data through the
types defined here. The canonical interchange format for match history is a
``pandas.DataFrame`` whose columns are exactly :data:`MATCH_COLUMNS`; the
helpers below validate and coerce such frames so that downstream code can rely
on dtypes and column order.
"""
from __future__ import annotations

import math
import re
import warnings
from dataclasses import asdict, dataclass, field, fields
from datetime import date, datetime, timedelta
from enum import Enum
from typing import Any, Final, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd

from src.common.competitions import COMPETITIONS, Competition, find_competition
from src.common.logging_config import get_logger

__all__ = [
    "CompetitionBase",
    "Girone",
    "build_girone_enum",
    "MATCH_COLUMNS",
    "MATCH_DTYPES",
    "REQUIRED_MATCH_COLUMNS",
    "OPTIONAL_MATCH_COLUMNS",
    "SEASON_START_MONTH",
    "MatchRecord",
    "Fixture",
    "OddsQuote",
    "MatchPrediction",
    "Market",
    "SELECTIONS",
    "ValidationReport",
    "matches_to_frame",
    "season_bounds",
    "season_of",
    "validate_matches_df",
    "validate_matches_report",
    "empty_matches_frame",
]

log = get_logger(__name__)

#: A football season runs from July to June: a match played in July 2026 or
#: later (up to June 2027) belongs to the season labelled ``'2026/27'``.
SEASON_START_MONTH: Final[int] = 7

_SEASON_LABEL_RE: Final[re.Pattern[str]] = re.compile(r"^\s*(\d{4})\s*[/_\-]\s*(\d{2}|\d{4})\s*$")


def season_of(d: date | datetime | pd.Timestamp | str) -> str:
    """Season label of a match date, ``'2026/27'`` for July 2026 - June 2027.

    The 16-column match frame deliberately has no ``season`` column: the season is
    a function of the date, so every module that needs a label (per-season
    summaries, "current season" filters, history ranges) derives it here and
    agrees with the others. ISO strings and ``pandas`` timestamps are accepted;
    a missing value (``NaT``/``None``) raises :class:`ValueError`.
    """
    if d is None or d is pd.NaT:
        raise ValueError("season_of: the date is missing")
    if isinstance(d, str):
        d = date.fromisoformat(d.strip()[:10])
    elif isinstance(d, pd.Timestamp):
        if pd.isna(d):
            raise ValueError("season_of: the date is missing")
        d = d.date()
    elif isinstance(d, datetime):
        d = d.date()
    elif not isinstance(d, date):
        raise TypeError(f"season_of expects a date, datetime, Timestamp or ISO string, got {type(d).__name__}")
    start_year = d.year if d.month >= SEASON_START_MONTH else d.year - 1
    return f"{start_year}/{(start_year + 1) % 100:02d}"


def season_bounds(label: str) -> tuple[date, date]:
    """First and last calendar day of a season label (``'2026/27'`` -> 1 July 2026, 30 June 2027).

    ``'2026_27'``, ``'2026-27'`` and ``'2026/2027'`` are accepted as well, so the
    file stems of ``config/teams_<season>.json`` work directly. A label whose
    second year is not the first plus one raises :class:`ValueError`.
    """
    match = _SEASON_LABEL_RE.match(str(label))
    if match is None:
        raise ValueError(f"invalid season label {label!r}: expected 'YYYY/YY' such as '2026/27'")
    start_year = int(match.group(1))
    second = match.group(2)
    end_year = int(second) if len(second) == 4 else (start_year // 100) * 100 + int(second)
    if len(second) == 2 and end_year < start_year:  # '1999/00'
        end_year += 100
    if end_year != start_year + 1:
        raise ValueError(f"invalid season label {label!r}: the second year must follow the first")
    first = date(start_year, SEASON_START_MONTH, 1)
    last = date(end_year, SEASON_START_MONTH, 1) - timedelta(days=1)
    return first, last


class CompetitionBase(str, Enum):
    """Behaviour of the dynamic :data:`Girone` enum; the base itself has no members.

    :func:`build_girone_enum` derives a concrete enum from a tuple of
    :class:`~src.common.competitions.Competition` records: one member per
    competition, named and valued by its code (``Girone.A.value == 'A'``,
    ``Girone.SB.value == 'SB'``) in configuration order. Members are plain
    strings too (``Girone.A == 'A'``), so frames, JSON payloads and CSV columns
    keep storing the code. The records the enum was built from are reachable
    through :meth:`competitions` and, per member, :attr:`competition`,
    :attr:`label`, :attr:`short`, :attr:`family`, :attr:`tier`.
    """

    @classmethod
    def competitions(cls) -> tuple[Competition, ...]:
        """The competition records this enum was built from (empty on the base class)."""
        return tuple(getattr(cls, "_competitions", ()))

    @classmethod
    def parse(cls, value: "str | CompetitionBase") -> "CompetitionBase":
        """Member for any accepted spelling of a competition.

        The code (``'A'``, ``'SB'``), the labels (``'Girone A'``, ``'Serie B'``,
        ``'serie_b'``), the FootyStats name (``'Italy Serie B'``) and a code
        wrapped in filler words (``'Serie C Girone A'``, ``'GIRONE_A'``) are all
        accepted, case-insensitively; ``'B'`` is always the competition coded
        ``B`` (Serie C Girone B). A member of another enum built by
        :func:`build_girone_enum` is matched by its value. ``ValueError`` for an
        unknown spelling.
        """
        if isinstance(value, cls):
            return value
        if isinstance(value, CompetitionBase):
            value = value.value
        try:
            comp = find_competition(value, cls.competitions())
        except ValueError as exc:
            raise ValueError(f"Unknown girone: {value!r} (known: {[m.value for m in cls]})") from exc
        return cls(comp.code)

    @property
    def competition(self) -> Competition:
        """The :class:`~src.common.competitions.Competition` record of this member."""
        by_code: Mapping[str, Competition] | None = getattr(type(self), "_by_code", None)
        if by_code is None or self.value not in by_code:  # pragma: no cover - only a hand-built enum lacks the table
            raise LookupError(f"no competition record for {self!r}; build the enum with build_girone_enum")
        return by_code[self.value]

    @property
    def label(self) -> str:
        """Full public name, ``'Serie C Girone A'`` / ``'Serie B'``."""
        return self.competition.label

    @property
    def short(self) -> str:
        """Compact name for tables and selectors, ``'Girone A'`` / ``'Serie B'``."""
        return self.competition.short

    @property
    def family(self) -> str:
        """League the competition belongs to (``'Serie C'`` for the three gironi)."""
        return self.competition.family

    @property
    def tier(self) -> int:
        """Level in the national pyramid (2 = Serie B, 3 = Serie C)."""
        return self.competition.tier

    @property
    def country(self) -> str:
        return self.competition.country

    @property
    def footystats_name(self) -> str:
        """``name`` of the competition in the FootyStats ``/league-list`` endpoint."""
        return self.competition.footystats_name

    def __str__(self) -> str:  # pragma: no cover - trivial
        return self.value


def build_girone_enum(
    competitions: Iterable[Competition],
    *,
    name: str = "Girone",
    module: str = __name__,
    qualname: str | None = None,
) -> type[CompetitionBase]:
    """Create a :class:`CompetitionBase` enum with one member per competition, in the given order.

    Member names and values are the competition codes. ``module``/``qualname``
    make the class picklable (they default to this module and ``name``). The
    global :data:`Girone` is ``build_girone_enum(COMPETITIONS)``; tests build
    further enums from custom tuples without touching it. ``ValueError`` for an
    empty iterable or duplicate codes.
    """
    comps = tuple(competitions)
    if not comps:
        raise ValueError("build_girone_enum needs at least one competition")
    codes = [c.code for c in comps]
    if len(set(codes)) != len(codes):
        raise ValueError(f"duplicate competition codes: {codes}")
    enum_cls = CompetitionBase(name, {c.code: c.code for c in comps}, module=module, qualname=qualname or name)
    # Attached after creation on purpose: inside an Enum class body every non-descriptor
    # name (underscore-prefixed included) would become a member.
    enum_cls._competitions = comps  # type: ignore[attr-defined]
    enum_cls._by_code = {c.code: c for c in comps}  # type: ignore[attr-defined]
    enum_cls.__doc__ = (
        f"Competitions of the perimeter, from config/competitions.toml: {', '.join(codes)}. "
        "The value of a member is its code; see CompetitionBase for the helpers."
    )
    return enum_cls


#: The competitions of this installation (``config/competitions.toml``): ``Girone.A``, ``Girone.B``,
#: ``Girone.C`` (Serie C) and ``Girone.SB`` (Serie B) by default. The value is the competition code.
Girone = build_girone_enum(COMPETITIONS)


#: Column order of the canonical match frame (identical to the spec).
MATCH_COLUMNS: Final[tuple[str, ...]] = (
    "match_id",
    "date",
    "round",
    "girone",
    "home_team",
    "away_team",
    "home_goals",
    "away_goals",
    "home_shots",
    "away_shots",
    "home_shots_target",
    "away_shots_target",
    "home_corners",
    "away_corners",
    "home_xg",
    "away_xg",
)

REQUIRED_MATCH_COLUMNS: Final[tuple[str, ...]] = (
    "match_id",
    "date",
    "round",
    "girone",
    "home_team",
    "away_team",
    "home_goals",
    "away_goals",
)

OPTIONAL_MATCH_COLUMNS: Final[tuple[str, ...]] = tuple(
    c for c in MATCH_COLUMNS if c not in REQUIRED_MATCH_COLUMNS
)

#: Target pandas dtypes. Optional stats are nullable floats (NaN when missing).
MATCH_DTYPES: Final[dict[str, str]] = {
    "match_id": "string",
    "date": "datetime64[ns]",
    "round": "int64",
    "girone": "string",
    "home_team": "string",
    "away_team": "string",
    "home_goals": "int64",
    "away_goals": "int64",
    "home_shots": "float64",
    "away_shots": "float64",
    "home_shots_target": "float64",
    "away_shots_target": "float64",
    "home_corners": "float64",
    "away_corners": "float64",
    "home_xg": "float64",
    "away_xg": "float64",
}


def _opt_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return None if math.isnan(out) else out


@dataclass(frozen=True, slots=True)
class MatchRecord:
    """One played match, normalised. Field order == :data:`MATCH_COLUMNS`."""

    match_id: str
    date: date
    round: int
    girone: Girone
    home_team: str
    away_team: str
    home_goals: int
    away_goals: int
    home_shots: float | None = None
    away_shots: float | None = None
    home_shots_target: float | None = None
    away_shots_target: float | None = None
    home_corners: float | None = None
    away_corners: float | None = None
    home_xg: float | None = None
    away_xg: float | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "girone", Girone.parse(self.girone))
        if isinstance(self.date, datetime):
            object.__setattr__(self, "date", self.date.date())
        elif isinstance(self.date, str):
            object.__setattr__(self, "date", date.fromisoformat(self.date[:10]))
        if self.home_goals < 0 or self.away_goals < 0:
            raise ValueError("goals cannot be negative")
        if self.home_team == self.away_team:
            raise ValueError("home_team and away_team must differ")
        for name in OPTIONAL_MATCH_COLUMNS:
            object.__setattr__(self, name, _opt_float(getattr(self, name)))

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["girone"] = self.girone.value
        d["date"] = pd.Timestamp(self.date)
        return d

    @classmethod
    def from_mapping(cls, m: Mapping[str, Any]) -> "MatchRecord":
        known = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in m.items() if k in known})


@dataclass(frozen=True, slots=True)
class Fixture:
    """An upcoming (not yet played) match."""

    match_id: str
    date: date
    round: int
    girone: Girone
    home_team: str
    away_team: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "girone", Girone.parse(self.girone))
        if isinstance(self.date, datetime):
            object.__setattr__(self, "date", self.date.date())
        elif isinstance(self.date, str):
            object.__setattr__(self, "date", date.fromisoformat(self.date[:10]))

    @property
    def label(self) -> str:
        return f"{self.home_team} - {self.away_team}"


class Market(str, Enum):
    """Supported betting markets."""

    MATCH_ODDS = "1X2"
    OVER_UNDER_15 = "OU1.5"
    OVER_UNDER_25 = "OU2.5"
    OVER_UNDER_35 = "OU3.5"
    BTTS = "GG/NG"


#: Selections for each market, in canonical order. The probability dict
#: produced by the engine and the odds dict produced by the bookmaker feed both
#: use these keys.
SELECTIONS: Final[dict[Market, tuple[str, ...]]] = {
    Market.MATCH_ODDS: ("1", "X", "2"),
    Market.OVER_UNDER_15: ("Over 1.5", "Under 1.5"),
    Market.OVER_UNDER_25: ("Over 2.5", "Under 2.5"),
    Market.OVER_UNDER_35: ("Over 3.5", "Under 3.5"),
    Market.BTTS: ("GG", "NG"),
}


@dataclass(frozen=True, slots=True)
class OddsQuote:
    """Decimal odds quoted by one bookmaker for one market of one fixture."""

    match_id: str
    bookmaker: str
    market: Market
    odds: dict[str, float] = field(default_factory=dict)  # selection -> decimal odds
    timestamp: datetime | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "market", Market(self.market))
        expected = set(SELECTIONS[self.market])
        got = set(self.odds)
        if got != expected:
            raise ValueError(
                f"odds for {self.market.value} must have selections {sorted(expected)}, got {sorted(got)}"
            )
        for sel, o in self.odds.items():
            if not (o > 1.0):
                raise ValueError(f"decimal odds must be > 1.0 ({sel}={o})")


@dataclass(frozen=True, slots=True)
class MatchPrediction:
    """Model output for one fixture: the interface between the engine
    (``src/engine/model.py``), the value module (``src/engine/value.py``) and the
    dashboard. ``probabilities``/``fair_odds`` are keyed exactly like
    :data:`SELECTIONS`; ``matrix[i, j]`` is ``P(home goals = i, away goals = j)``."""

    fixture: Fixture
    home_rate: float  # lambda
    away_rate: float  # mu
    rho: float
    matrix: np.ndarray
    probabilities: dict[Market, dict[str, float]]
    fair_odds: dict[Market, dict[str, float]]

    def __post_init__(self) -> None:
        if self.matrix.ndim != 2 or self.matrix.shape[0] != self.matrix.shape[1]:
            raise ValueError("matrix must be square")
        if not np.isclose(self.matrix.sum(), 1.0, atol=1e-6):
            raise ValueError("matrix must sum to 1")

    @property
    def match_id(self) -> str:
        return self.fixture.match_id

    @property
    def expected_goals(self) -> float:
        return self.home_rate + self.away_rate

    def most_likely_score(self) -> tuple[int, int, float]:
        i, j = np.unravel_index(int(np.argmax(self.matrix)), self.matrix.shape)
        return int(i), int(j), float(self.matrix[i, j])


def empty_matches_frame() -> pd.DataFrame:
    df = pd.DataFrame({c: pd.Series(dtype=t) for c, t in MATCH_DTYPES.items()})
    return df[list(MATCH_COLUMNS)]


def matches_to_frame(records: Iterable[MatchRecord]) -> pd.DataFrame:
    rows = [r.to_dict() for r in records]
    if not rows:
        return empty_matches_frame()
    return validate_matches_df(pd.DataFrame(rows))


#: How many offending ``match_id`` values are quoted in log lines and errors.
_MAX_IDS_IN_MESSAGE: Final[int] = 10


@dataclass(frozen=True)
class ValidationReport:
    """What :func:`validate_matches_df` dropped, so callers can show it.

    ``missing_required`` lists the rows (by ``match_id``, or ``row <label>`` when
    the id itself is missing) dropped because a required cell was empty or could
    not be parsed; ``missing_by_column`` counts the offending cells per column
    (``{"date": 19}`` pinpoints a column-wide date problem at once).
    ``duplicates`` lists the ``match_id`` values that occurred more than once;
    only their last occurrence survives. The report is also attached to the
    returned frame as ``frame.attrs["validation"]`` (see :meth:`to_dict`).
    """

    n_input: int
    n_output: int
    missing_required: tuple[str, ...] = ()
    missing_by_column: dict[str, int] = field(default_factory=dict)
    duplicates: tuple[str, ...] = ()
    n_duplicates_dropped: int = 0

    @property
    def n_dropped(self) -> int:
        return len(self.missing_required) + self.n_duplicates_dropped

    def to_dict(self) -> dict[str, Any]:
        return {
            "n_input": self.n_input,
            "n_output": self.n_output,
            "n_dropped": self.n_dropped,
            "missing_required": list(self.missing_required),
            "missing_by_column": dict(self.missing_by_column),
            "duplicates": list(self.duplicates),
            "n_duplicates_dropped": self.n_duplicates_dropped,
        }

    def describe(self) -> str | None:
        """One Italian sentence for console/dashboard notes, ``None`` when nothing was dropped."""
        if not self.n_dropped:
            return None
        parts: list[str] = []
        if self.missing_required:
            cols = ", ".join(f"{c}: {n}" for c, n in self.missing_by_column.items())
            parts.append(
                f"{len(self.missing_required)} righe con valori obbligatori mancanti o non validi ({cols}; "
                f"match_id: {_preview(self.missing_required)})"
            )
        if self.duplicates:
            parts.append(
                f"{self.n_duplicates_dropped} righe duplicate per match_id, tenuta l'ultima "
                f"(match_id: {_preview(self.duplicates)})"
            )
        return f"{self.n_dropped} partite scartate su {self.n_input} in validazione: " + "; ".join(parts) + "."


def _preview(ids: Sequence[str]) -> str:
    shown = ", ".join(ids[:_MAX_IDS_IN_MESSAGE])
    return shown + (", ..." if len(ids) > _MAX_IDS_IN_MESSAGE else "")


def _blank_to_na(values: pd.Series) -> pd.Series:
    """Coerce to the nullable ``string`` dtype, stripping whitespace; blanks become ``<NA>``."""
    text = values.astype("string").str.strip()
    return text.mask(text == "", pd.NA)


def _parse_date_cell(text: str) -> pd.Timestamp:
    """Parse one non-ISO date cell (``24/08/2025``, ``24.08.2025 18:00``, ...).

    Day-first precedence, since the data is Italian; the time zone (if any) is
    dropped keeping the local wall-clock day. Unparseable text yields ``NaT``.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            ts = pd.Timestamp(pd.to_datetime(text, dayfirst=True))
        except (ValueError, TypeError, OverflowError):
            return pd.NaT
    if ts is pd.NaT:
        return pd.NaT
    if ts.tzinfo is not None:
        ts = ts.tz_localize(None)
    return ts.normalize()


def _coerce_dates(values: pd.Series) -> pd.Series:
    """Parse the ``date`` column into naive, midnight-normalised ``datetime64[ns]``.

    ISO-8601 cells are parsed in one vectorised pass with an explicit format;
    whatever is left is parsed cell by cell with :func:`_parse_date_cell`. With
    pandas' default format inference one ``24/08/2025`` row in a hand-maintained
    CSV decided the format for the whole column and turned every ISO date into
    ``NaT``; here each cell stands on its own and only genuinely unparseable
    cells (``TBD``) become ``NaT``.
    """
    if pd.api.types.is_datetime64_any_dtype(values):
        parsed = values
    else:
        text = _blank_to_na(values)
        try:
            parsed = pd.to_datetime(text, errors="coerce", format="ISO8601")
        except ValueError:  # mixed time zones within the column: parse every cell on its own
            parsed = pd.Series(pd.NaT, index=values.index, dtype="datetime64[ns]")
        if parsed.dt.tz is not None:
            parsed = parsed.dt.tz_localize(None)
        parsed = parsed.astype("datetime64[ns]")
        rest = parsed.isna() & text.notna().fillna(False).astype(bool)
        if rest.any():
            parsed = parsed.copy()
            parsed.loc[rest] = pd.to_datetime(text[rest].map(_parse_date_cell), errors="coerce")
    if parsed.dt.tz is not None:
        parsed = parsed.dt.tz_localize(None)
    return parsed.dt.normalize().astype("datetime64[ns]")


def _parse_girone_or_none(value: Any) -> str | None:
    try:
        return Girone.parse(value).value
    except ValueError:
        return None


def validate_matches_df(df: pd.DataFrame, *, strict: bool = False) -> pd.DataFrame:
    """Return a copy of ``df`` coerced to the canonical schema.

    - Required columns must be present (``ValueError`` otherwise).
    - Missing optional columns are added as NaN.
    - Every required column is coerced *before* rows are judged: dates are parsed
      per cell (ISO first, then day-first fallback), ``round`` and the goals are
      parsed as numbers, blank ids/team names count as missing.
    - Rows with a required value missing or unparseable (an empty goal cell, a
      ``'pp'`` sentinel, a ``'TBD'`` date) are dropped with a logged warning that
      names the columns and the ``match_id`` values (or raise when ``strict``).
    - An unknown ``girone`` (not a competition code or label of
      ``config/competitions.toml``) raises ``ValueError`` naming the rows; the
      column is stored as the code (``'Serie B'`` -> ``'SB'``).
    - Duplicate ``match_id`` keeps the last occurrence (logged).

    The drop counts are attached as ``frame.attrs["validation"]``; use
    :func:`validate_matches_report` to get them as a :class:`ValidationReport`.
    """
    frame, _ = validate_matches_report(df, strict=strict)
    return frame


def validate_matches_report(df: pd.DataFrame, *, strict: bool = False) -> tuple[pd.DataFrame, ValidationReport]:
    """Like :func:`validate_matches_df`, also returning the :class:`ValidationReport`."""
    missing = [c for c in REQUIRED_MATCH_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"matches frame is missing required columns: {missing}")
    out = df.copy()
    for c in OPTIONAL_MATCH_COLUMNS:
        if c not in out.columns:
            out[c] = float("nan")
    extra = [c for c in out.columns if c not in MATCH_COLUMNS]
    out = out.drop(columns=extra)
    n_input = len(out)

    # 1. coerce every required column first, so that an unparseable cell shows up
    #    as missing in the row filter below instead of crashing a later astype.
    out["match_id"] = _blank_to_na(out["match_id"])
    out["date"] = _coerce_dates(out["date"])
    out["round"] = pd.to_numeric(out["round"], errors="coerce")
    out["girone"] = _blank_to_na(out["girone"])
    out["home_team"] = _blank_to_na(out["home_team"])
    out["away_team"] = _blank_to_na(out["away_team"])
    out["home_goals"] = pd.to_numeric(out["home_goals"], errors="coerce")
    out["away_goals"] = pd.to_numeric(out["away_goals"], errors="coerce")

    # 2. rows with a required value missing/unparseable
    na = out[list(REQUIRED_MATCH_COLUMNS)].isna()
    bad = na.any(axis=1)
    dropped_ids: tuple[str, ...] = ()
    by_column: dict[str, int] = {}
    if bad.any():
        by_column = {c: int(n) for c, n in na.sum().items() if n}
        labels = out.loc[bad, "match_id"]
        dropped_ids = tuple(
            str(mid) if not pd.isna(mid) else f"row {idx}" for idx, mid in zip(labels.index, labels)
        )
        detail = f"columns: {by_column}; match_id: {_preview(dropped_ids)}"
        if strict:
            raise ValueError(f"{int(bad.sum())} rows have missing required values ({detail})")
        log.warning(
            "rows dropped: missing required values",
            extra={"n_dropped": int(bad.sum()), "n_input": n_input, "by_column": by_column, "match_ids": list(dropped_ids[:_MAX_IDS_IN_MESSAGE])},
        )
        out = out.loc[~bad]

    # 3. value coercions that can only run on complete rows
    girone = out["girone"].map(_parse_girone_or_none)
    unknown = girone.isna()
    if unknown.any():
        ids = out.loc[unknown, "match_id"].astype(str).tolist()
        values = sorted(set(out.loc[unknown, "girone"].astype(str)))
        raise ValueError(
            f"{int(unknown.sum())} rows have an unknown girone {values[:_MAX_IDS_IN_MESSAGE]} "
            f"(expected one of {[g.value for g in Girone]}; match_id: {_preview(ids)})"
        )
    out["girone"] = girone
    out["round"] = out["round"].astype("int64")
    out["home_goals"] = out["home_goals"].astype("int64")
    out["away_goals"] = out["away_goals"].astype("int64")
    for c in OPTIONAL_MATCH_COLUMNS:
        out[c] = pd.to_numeric(out[c], errors="coerce").astype("float64")

    # 4. duplicate ids: the last occurrence wins (an update overrides the earlier line)
    duplicated = out["match_id"].duplicated(keep="last")
    dup_ids: tuple[str, ...] = ()
    if duplicated.any():
        dup_ids = tuple(dict.fromkeys(out.loc[duplicated, "match_id"].astype(str)))
        log.warning(
            "duplicate match_id: last occurrence kept",
            extra={"n_dropped": int(duplicated.sum()), "n_ids": len(dup_ids), "match_ids": list(dup_ids[:_MAX_IDS_IN_MESSAGE])},
        )
        out = out.loc[~duplicated]

    out = out.sort_values(["date", "girone", "match_id"], kind="stable").reset_index(drop=True)
    for c, t in MATCH_DTYPES.items():
        if t == "string":
            out[c] = out[c].astype("string")
    out = out[list(MATCH_COLUMNS)]
    report = ValidationReport(
        n_input=n_input,
        n_output=len(out),
        missing_required=dropped_ids,
        missing_by_column=by_column,
        duplicates=dup_ids,
        n_duplicates_dropped=int(duplicated.sum()),
    )
    out.attrs["validation"] = report.to_dict()
    return out, report
