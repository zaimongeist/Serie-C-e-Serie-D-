"""Tests for ``src.common.schemas.validate_matches_df`` (review findings on hand-maintained CSVs).

The canonical frame is often a CSV edited by hand: one row in a different date
format, a ``'pp'`` sentinel for a postponed match, an empty goal cell or a
corrected line pasted twice. Validation must coerce first and judge after, and
must say what it drops.
"""
from __future__ import annotations

import logging
from datetime import date
from pathlib import Path

import pandas as pd
import pytest

import main as pipeline
from src.common.schemas import (
    MATCH_COLUMNS,
    MATCH_DTYPES,
    Girone,
    MatchRecord,
    ValidationReport,
    empty_matches_frame,
    matches_to_frame,
    validate_matches_df,
    validate_matches_report,
)
from src.simulation.generator import simulate_mini_dataset

MISSING_MSG = "rows dropped: missing required values"
DUPLICATE_MSG = "duplicate match_id: last occurrence kept"


@pytest.fixture(scope="module")
def mini() -> pd.DataFrame:
    return simulate_mini_dataset(seed=42, n_matches=20).matches


@pytest.fixture
def csv_like(mini: pd.DataFrame) -> pd.DataFrame:
    """The mini dataset as it comes back from ``pd.read_csv(dtype=str)``: every cell a string."""
    out = mini.copy()
    out["date"] = out["date"].dt.strftime("%Y-%m-%d")
    return out.astype(object).astype(str).replace("nan", None)


def _records(caplog: pytest.LogCaptureFixture, message: str) -> list[logging.LogRecord]:
    return [r for r in caplog.records if r.getMessage() == message]


# ----------------------------------------------------------------------- finding 1: dates
def test_one_non_iso_date_does_not_poison_the_whole_column(csv_like: pd.DataFrame, caplog: pytest.LogCaptureFixture) -> None:
    """pandas inferred the format from the first cell: one ``24/08/2025`` row among
    ISO dates used to turn the other 19 dates into NaT, and the fit then ran on 1 match."""
    mixed = csv_like.copy()
    mixed.loc[mixed.index[3], "date"] = "24/08/2025"
    with caplog.at_level(logging.WARNING):
        out = validate_matches_df(mixed)
    assert len(out) == 20
    assert out["date"].notna().all()
    assert out["date"].dtype == "datetime64[ns]"
    assert (out["date"] == pd.Timestamp("2025-08-24")).sum() >= 1
    assert out.attrs["validation"]["n_dropped"] == 0
    assert not _records(caplog, MISSING_MSG)


def test_dates_keep_the_local_day_whatever_the_cell_shape() -> None:
    """ISO with/without time zone (mixed within one column), ``date`` objects,
    Timestamps and day-first Italian formats all land on the same calendar day."""
    cells = [
        "2025-08-24",
        "2025-08-24T00:30:00+02:00",  # wall-clock day, not the UTC day
        "2025-08-24 21:00:00",
        " 24/08/2025 ",
        "24.08.2025 18:00",
        date(2025, 8, 24),
        pd.Timestamp("2025-08-24 15:00"),
    ]
    n = len(cells)
    frame = pd.DataFrame(
        {
            "match_id": [f"M{i}" for i in range(n)],
            "date": pd.Series(cells, dtype=object),
            "round": [1] * n,
            "girone": ["A"] * n,
            "home_team": [f"H{i}" for i in range(n)],
            "away_team": [f"A{i}" for i in range(n)],
            "home_goals": [1] * n,
            "away_goals": [0] * n,
        }
    )
    out = validate_matches_df(frame)
    assert len(out) == n
    assert (out["date"] == pd.Timestamp("2025-08-24")).all()
    assert out["date"].dt.tz is None

    # an already-parsed, tz-aware column is accepted as well
    aware = frame.assign(date=pd.to_datetime(["2025-08-24T23:30:00+02:00"] * n))
    assert (validate_matches_df(aware)["date"] == pd.Timestamp("2025-08-24")).all()


def test_unparseable_date_drops_only_that_row_and_says_so(csv_like: pd.DataFrame, caplog: pytest.LogCaptureFixture) -> None:
    tbd = csv_like.copy()
    victim = str(tbd.loc[tbd.index[0], "match_id"])
    tbd.loc[tbd.index[0], "date"] = "TBD"
    with caplog.at_level(logging.WARNING):
        out, report = validate_matches_report(tbd)
    assert len(out) == 19 and victim not in set(out["match_id"])
    assert report == ValidationReport(n_input=20, n_output=19, missing_required=(victim,), missing_by_column={"date": 1})
    [record] = _records(caplog, MISSING_MSG)
    assert record.n_dropped == 1 and record.by_column == {"date": 1} and record.match_ids == [victim]


# ------------------------------------------------------------- finding 3: non-numeric goal cells
def test_text_goal_sentinel_is_dropped_with_its_match_id_instead_of_crashing(csv_like: pd.DataFrame, caplog: pytest.LogCaptureFixture) -> None:
    """``'pp'`` (postponed) survived the isna filter and blew up in ``astype('int64')``
    with an IntCastingNaNError that named no row."""
    pp = csv_like.copy()
    victims = [str(pp.loc[pp.index[5], "match_id"]), str(pp.loc[pp.index[7], "match_id"])]
    pp.loc[pp.index[5], "home_goals"] = "pp"
    pp.loc[pp.index[7], "away_goals"] = ""  # empty cell in a postponed round
    with caplog.at_level(logging.WARNING):
        out, report = validate_matches_report(pp)
    assert len(out) == 18 and not set(victims) & set(out["match_id"])
    assert out["home_goals"].dtype == "int64" and out["away_goals"].dtype == "int64"
    assert report.n_dropped == 2 and sorted(report.missing_required) == sorted(victims)
    assert report.missing_by_column == {"home_goals": 1, "away_goals": 1}
    [record] = _records(caplog, MISSING_MSG)
    assert record.n_dropped == 2 and record.n_input == 20 and sorted(record.match_ids) == sorted(victims)
    assert out.attrs["validation"]["missing_required"] == list(report.missing_required)
    note = report.describe()
    assert note is not None and "2 partite scartate su 20" in note and victims[0] in note and "home_goals: 1" in note

    with pytest.raises(ValueError, match="2 rows have missing required values") as exc:
        validate_matches_df(pp, strict=True)
    assert victims[0] in str(exc.value) and "home_goals" in str(exc.value)


def test_non_numeric_round_counts_as_missing(csv_like: pd.DataFrame, caplog: pytest.LogCaptureFixture) -> None:
    bad = csv_like.copy()
    bad.loc[bad.index[2], "round"] = "1a giornata"
    with caplog.at_level(logging.WARNING):
        out = validate_matches_df(bad)
    assert len(out) == 19 and out["round"].dtype == "int64"
    assert _records(caplog, MISSING_MSG)[0].by_column == {"round": 1}


# --------------------------------------------------------------- finding 2: silent drops
def test_duplicate_match_id_keeps_the_last_line_and_logs_it(mini: pd.DataFrame, caplog: pytest.LogCaptureFixture) -> None:
    corrected = mini.head(1).assign(home_goals=9)
    frame = pd.concat([mini.head(4), corrected, corrected], ignore_index=True)
    with caplog.at_level(logging.WARNING):
        out, report = validate_matches_report(frame)
    victim = str(mini.loc[mini.index[0], "match_id"])
    assert len(out) == 4
    assert int(out.loc[out["match_id"] == victim, "home_goals"].iloc[0]) == 9
    assert report.duplicates == (victim,) and report.n_duplicates_dropped == 2 and report.n_dropped == 2
    [record] = _records(caplog, DUPLICATE_MSG)
    assert record.n_dropped == 2 and record.match_ids == [victim]
    assert "2 righe duplicate" in (report.describe() or "")


def test_blank_id_and_team_cells_count_as_missing(csv_like: pd.DataFrame, caplog: pytest.LogCaptureFixture) -> None:
    blank = csv_like.copy()
    blank.loc[blank.index[0], "match_id"] = "   "
    blank.loc[blank.index[1], "home_team"] = ""
    second = str(blank.loc[blank.index[1], "match_id"])
    with caplog.at_level(logging.WARNING):
        out, report = validate_matches_report(blank)
    assert len(out) == 18
    assert report.missing_by_column == {"match_id": 1, "home_team": 1}
    assert report.missing_required == (f"row {blank.index[0]}", second)  # a row without an id is still named


def test_unknown_girone_raises_naming_the_rows(mini: pd.DataFrame) -> None:
    bad = mini.head(3).copy()
    bad["girone"] = bad["girone"].astype(object)
    bad.loc[bad.index[1], "girone"] = "D"
    with pytest.raises(ValueError, match="1 rows have an unknown girone") as exc:
        validate_matches_df(bad)
    assert str(bad.loc[bad.index[1], "match_id"]) in str(exc.value) and "'D'" in str(exc.value)
    # the lenient spellings still work
    assert validate_matches_df(bad.head(1).assign(girone="Girone A"))["girone"].tolist() == ["A"]


def test_clean_frames_validate_silently_and_unchanged(mini: pd.DataFrame, caplog: pytest.LogCaptureFixture) -> None:
    with caplog.at_level(logging.WARNING):
        out = validate_matches_df(mini)
        empty = validate_matches_df(empty_matches_frame())
    pd.testing.assert_frame_equal(out, mini)
    assert list(out.columns) == list(MATCH_COLUMNS)
    assert out.dtypes.astype(str).to_dict() == MATCH_DTYPES
    assert out.attrs["validation"] == {
        "n_input": 20,
        "n_output": 20,
        "n_dropped": 0,
        "missing_required": [],
        "missing_by_column": {},
        "duplicates": [],
        "n_duplicates_dropped": 0,
    }
    assert empty.empty and list(empty.columns) == list(MATCH_COLUMNS)
    assert not [r for r in caplog.records if r.name == "src.common.schemas"]
    records = [MatchRecord("R1", date(2025, 8, 23), 1, Girone.A, "Lecco", "Novara", 2, 0)]
    frame = matches_to_frame(records)
    assert frame.attrs["validation"]["n_dropped"] == 0 and frame["date"].iloc[0] == pd.Timestamp("2025-08-23")


# ----------------------------------------------------------- end to end through main.py
def test_pipeline_match_counts_agree_on_a_hand_edited_csv(tmp_path: Path, mini: pd.DataFrame, capsys: pytest.CaptureFixture[str]) -> None:
    """``python main.py --processed mixed.csv`` printed 'partite: 1' / n=1 with exit 0
    and nothing on stderr; the dataset summary and the fit must see the same rows."""
    from src.ingestion.harvester import save_processed

    clean = tmp_path / "clean.csv"
    save_processed(mini, clean)
    raw = pd.read_csv(clean, dtype=str)

    mixed = raw.copy()
    mixed.loc[3, "date"] = "24/08/2025"
    mixed_path = tmp_path / "mixed.csv"
    mixed.to_csv(mixed_path, index=False)
    out = pipeline.main(["--processed", str(mixed_path), "--no-export", "--log-level", "WARNING"])
    console, err = capsys.readouterr()
    assert len(out.matches) == 20 == out.params.n_matches
    assert out.matches["date"].notna().all()
    assert "partite: 20" in console
    assert MISSING_MSG not in err

    postponed = raw.copy()
    postponed.loc[5, "home_goals"] = "pp"
    postponed.loc[7, "date"] = "TBD"
    victims = [postponed.loc[5, "match_id"], postponed.loc[7, "match_id"]]
    pp_path = tmp_path / "pp.csv"
    postponed.to_csv(pp_path, index=False)
    out = pipeline.main(["--processed", str(pp_path), "--no-export", "--log-level", "WARNING"])
    console, err = capsys.readouterr()
    assert len(out.matches) == 18 == out.params.n_matches
    assert "partite: 18" in console
    assert MISSING_MSG in err and all(v in err for v in victims) and "'home_goals': 1" in err and "'date': 1" in err
