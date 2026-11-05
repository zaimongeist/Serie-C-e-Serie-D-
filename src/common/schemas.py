"""Shared data contracts for the Serie C quant pipeline.

Every module (ingestion, engine, value, dashboard) exchanges data through the
types defined here. The canonical interchange format for match history is a
``pandas.DataFrame`` whose columns are exactly :data:`MATCH_COLUMNS`; the
helpers below validate and coerce such frames so that downstream code can rely
on dtypes and column order.
"""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field, fields
from datetime import date, datetime
from enum import Enum
from typing import Any, Final, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd

__all__ = [
    "Girone",
    "MATCH_COLUMNS",
    "MATCH_DTYPES",
    "REQUIRED_MATCH_COLUMNS",
    "OPTIONAL_MATCH_COLUMNS",
    "MatchRecord",
    "Fixture",
    "OddsQuote",
    "MatchPrediction",
    "Market",
    "SELECTIONS",
    "matches_to_frame",
    "validate_matches_df",
    "empty_matches_frame",
]


class Girone(str, Enum):
    """The three Serie C groups. The value is the public letter."""

    A = "A"
    B = "B"
    C = "C"

    @classmethod
    def parse(cls, value: "str | Girone") -> "Girone":
        if isinstance(value, Girone):
            return value
        text = str(value).strip().upper()
        # accept 'A', 'Girone A', 'GIRONE_A', 'serie c girone a'
        text = text.replace("GIRONE", "").replace("SERIE C", "").replace("_", "").strip()
        try:
            return cls(text)
        except ValueError as exc:  # pragma: no cover - defensive
            raise ValueError(f"Unknown girone: {value!r}") from exc

    def __str__(self) -> str:  # pragma: no cover - trivial
        return self.value


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


def validate_matches_df(df: pd.DataFrame, *, strict: bool = False) -> pd.DataFrame:
    """Return a copy of ``df`` coerced to the canonical schema.

    - Required columns must be present (``ValueError`` otherwise).
    - Missing optional columns are added as NaN.
    - ``girone`` is normalised to ``'A' | 'B' | 'C'``.
    - Rows with missing required values are dropped (or raise when ``strict``).
    - Duplicate ``match_id`` keeps the last occurrence.
    """
    missing = [c for c in REQUIRED_MATCH_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"matches frame is missing required columns: {missing}")
    out = df.copy()
    for c in OPTIONAL_MATCH_COLUMNS:
        if c not in out.columns:
            out[c] = float("nan")
    extra = [c for c in out.columns if c not in MATCH_COLUMNS]
    out = out.drop(columns=extra)

    bad = out[list(REQUIRED_MATCH_COLUMNS)].isna().any(axis=1)
    if bad.any():
        if strict:
            raise ValueError(f"{int(bad.sum())} rows have missing required values")
        out = out.loc[~bad]

    out["match_id"] = out["match_id"].astype(str)
    out["date"] = pd.to_datetime(out["date"], errors="coerce").dt.tz_localize(None).dt.normalize().astype("datetime64[ns]")
    out["round"] = pd.to_numeric(out["round"], errors="coerce").fillna(0).astype("int64")
    out["girone"] = out["girone"].map(lambda g: Girone.parse(g).value)
    out["home_team"] = out["home_team"].astype(str).str.strip()
    out["away_team"] = out["away_team"].astype(str).str.strip()
    out["home_goals"] = pd.to_numeric(out["home_goals"], errors="coerce").astype("int64")
    out["away_goals"] = pd.to_numeric(out["away_goals"], errors="coerce").astype("int64")
    for c in OPTIONAL_MATCH_COLUMNS:
        out[c] = pd.to_numeric(out[c], errors="coerce").astype("float64")
    out = out.drop_duplicates(subset="match_id", keep="last")
    out = out.sort_values(["date", "girone", "match_id"], kind="stable").reset_index(drop=True)
    for c, t in MATCH_DTYPES.items():
        if t == "string":
            out[c] = out[c].astype("string")
    return out[list(MATCH_COLUMNS)]
