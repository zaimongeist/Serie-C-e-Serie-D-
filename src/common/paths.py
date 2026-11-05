"""Project directory layout (single source of truth for file locations)."""
from __future__ import annotations

from pathlib import Path

ROOT: Path = Path(__file__).resolve().parents[2]
CONFIG_DIR: Path = ROOT / "config"
DATA_DIR: Path = ROOT / "data"
RAW_DIR: Path = DATA_DIR / "raw"
PROCESSED_DIR: Path = DATA_DIR / "processed"
ODDS_DIR: Path = DATA_DIR / "odds"


def ensure_dirs() -> None:
    for p in (RAW_DIR, PROCESSED_DIR, ODDS_DIR):
        p.mkdir(parents=True, exist_ok=True)
