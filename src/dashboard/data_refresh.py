"""Runtime download of the FootyStats data with the user's own key (no data in the repository).

FootyStats data is licensed to the account that downloads it: publishing the
CSVs in a public repository would redistribute it. The dashboard (and
``main.py --footystats``) therefore fetch the data **at runtime** with the
user's key and keep it **only locally** (``data/processed`` and ``data/odds``,
git-ignored):

* :func:`find_key` -- the key from the environment (``FOOTYSTATS_API_KEY``),
  then the ``.env`` file of the repository, then ``st.secrets`` when Streamlit
  is installed and a secrets file exists (Streamlit Cloud: *App settings ->
  Secrets*, ``FOOTYSTATS_API_KEY = "..."``). The key is never printed, logged,
  written or put in an exception: :class:`KeyLocation` hides it from ``repr``;
* :func:`data_age` / :func:`needs_refresh` -- how old the local data is (the
  stamp :data:`STAMP_FILE` written by :func:`refresh`, unless ``matches.csv``
  was rewritten after it -- by the launchers' ``python -m
  src.ingestion.footystats`` -- in which case the file's modification time)
  and whether it is older than :data:`DEFAULT_MAX_AGE` (6 hours) or missing;
  :func:`stale_warning` / :func:`kept_data_warnings` are the Italian warnings
  the dashboard shows on every run while the data is old or a competition still
  carries the data of an earlier download;
* :func:`refresh` -- one FootyStats harvest of every competition (the
  provider's registry synchronisation included, so the team dictionary follows
  FootyStats), the matches, fixtures and odds CSVs saved
  locally (atomically: a failed run never leaves half a file) and a
  :class:`RefreshResult` with the counts, the registry differences, the error
  of every competition that could not be downloaded (a league not chosen on the
  account, a network failure: the others are still saved, and the previous
  data of a failed competition is kept -- also when only one season of it
  failed: a failed current season keeps the previous current-season matches,
  fixtures and odds of that competition) and the timestamp. It never raises on
  the failure of a single competition. The pre-match quotes FootyStats keeps on
  the **completed** matches go to the odds history
  (``odds_dir/history/odds_history.csv``, :func:`odds_history_file`), merged
  with the history already there: the walk-forward backtest compares the
  model with the market on it. It sits in a sub-folder because every ``*.csv``
  directly in ``odds_dir`` is read as a list of current prices.

The walk-forward backtest of the *Affidabilità modello* tab is persisted next
to the data folder (:func:`save_backtest` / :func:`load_backtest`:
``data/backtest.parquet`` and ``data/backtest.json``, git-ignored), keyed on
the data version and the model settings, so a rerun or another session reads
it back instead of refitting every past matchday; a refresh of the data
changes the key and the next run recomputes it. The matchday-by-matchday
out-of-sample recalibration of that backtest (:func:`save_rolling` /
:func:`load_rolling`: ``data/backtest_rolling.parquet`` and
``data/backtest_rolling.json``, same key) is persisted the same way, so a
restart does not refit it either. :func:`load_odds_history_frame`
reads the odds history for the comparison with the market (an empty frame when
there is none).

Pure of ``streamlit`` except for the optional ``st.secrets`` lookup, so the
console and the tests use it without a Streamlit runtime; tests pass a fake
``provider_factory`` and never touch the network.
"""
from __future__ import annotations

import asyncio
import importlib.util
import inspect
import json
import os
import tempfile
import time
from collections.abc import AsyncIterator, Callable, Iterable, Mapping, Sequence
from contextlib import AbstractAsyncContextManager, asynccontextmanager
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any, Final, Protocol

import pandas as pd

from src.common.logging_config import get_logger
from src.common.paths import CONFIG_DIR, ODDS_DIR, PROCESSED_DIR, RAW_DIR
from src.common.schemas import Fixture, Girone, OddsQuote, empty_matches_frame, season_of
from src.common.teams import DEFAULT_SEASON, load_registry, season_file_stem
from src.engine.calendar import ROME
from src.ingestion.footystats import DEFAULT_COMPETITIONS, DEFAULT_ENV_FILE, find_api_key, mask_key
from src.engine.backtest import BacktestResult
from src.engine.calibration import RecalibrationModel, RollingRecalibration
from src.ingestion.harvester import (
    load_fixtures,
    load_odds_history,
    load_processed,
    odds_history_path,
    save_fixtures,
    save_odds,
    save_odds_history,
    save_processed,
)

__all__ = [
    "BACKTEST_FILE",
    "BACKTEST_META_FILE",
    "ROLLING_FILE",
    "ROLLING_META_FILE",
    "DEFAULT_MAX_AGE",
    "KEY_ENV",
    "ODDS_FILE",
    "PROGRESS_INTERVAL",
    "STAMP_FILE",
    "HarvestProvider",
    "KeyLocation",
    "ProgressCallback",
    "ProviderFactory",
    "RefreshResult",
    "age_text",
    "backtest_paths",
    "data_age",
    "find_key",
    "kept_data_warnings",
    "last_refresh",
    "load_backtest",
    "load_odds_history_frame",
    "load_rolling",
    "locate_key",
    "needs_refresh",
    "odds_history_file",
    "refresh",
    "refresh_label",
    "rolling_paths",
    "save_backtest",
    "save_rolling",
    "stale_warning",
    "supported_competitions",
]

log = get_logger(__name__)

#: Environment variable (and ``.env`` / Streamlit secret name) of the FootyStats key.
KEY_ENV: Final[str] = "FOOTYSTATS_API_KEY"
#: Data older than this is downloaded again when the dashboard starts.
DEFAULT_MAX_AGE: Final[timedelta] = timedelta(hours=6)
#: JSON stamp written next to ``matches.csv`` by :func:`refresh` (timestamp, counts, errors, registry differences).
STAMP_FILE: Final[str] = "refresh_stamp.json"
#: Odds file written by :func:`refresh` in the odds directory.
ODDS_FILE: Final[str] = "footystats.csv"
#: Seasons downloaded per competition (the current one and the two before it).
DEFAULT_SEASONS: Final[int] = 3
def odds_history_file(odds_dir: Path = ODDS_DIR) -> Path:
    """The odds history :func:`refresh` writes for ``odds_dir``: ``odds_dir/history/odds_history.csv``.

    A sub-folder, never a ``*.csv`` directly in ``odds_dir`` (that folder is read
    as current prices by the dashboard); git-ignored like all of ``data/odds``.
    """
    return odds_history_path(odds_dir)


#: Seconds between two progress reports while the provider downloads.
PROGRESS_INTERVAL: float = 0.5

#: ``progress(fraction in [0, 1], Italian message)`` called while :func:`refresh` runs.
ProgressCallback = Callable[[float, str], None]


class HarvestProvider(Protocol):
    """What :func:`refresh` needs from a provider: :meth:`src.ingestion.footystats.FootyStatsProvider.harvest`."""

    async def harvest(self, codes: Iterable[str], *, seasons: int = 3, with_odds: bool = True, sync_registry: bool = True) -> Any:
        """Harvest ``codes`` (dictionary synchronised); returns a :class:`~src.ingestion.footystats.FootyStatsHarvest`-like result."""


#: ``factory(key, season=..., config_dir=...)`` -> async context manager yielding a :class:`HarvestProvider`.
ProviderFactory = Callable[..., AbstractAsyncContextManager[HarvestProvider]]


# =============================================================================
# Key
# =============================================================================
@dataclass(frozen=True)
class KeyLocation:
    """A FootyStats key and where it was found (``'ambiente'``, ``'.env'``, ``'Streamlit secrets'``).

    The value is excluded from ``repr`` and from equality output, so logging or
    printing a location never leaks the key.
    """

    value: str = field(repr=False)
    source: str


def _streamlit_secret(name: str) -> str | None:
    """``st.secrets[name]`` when Streamlit is installed and a secrets file defines it, else ``None`` (never raises)."""
    if importlib.util.find_spec("streamlit") is None:
        return None
    try:
        import streamlit as st

        secrets = st.secrets
        loader = getattr(secrets, "load_if_toml_exists", None)
        if callable(loader) and not loader():
            return None
        if name not in secrets:
            return None
        value = secrets[name]
    except Exception as exc:  # noqa: BLE001 - any secrets problem means "no key from secrets"
        log.debug("streamlit secrets unavailable", extra={"error_type": type(exc).__name__})
        return None
    text = str(value).strip() if value is not None else ""
    return text or None


def locate_key(
    *,
    environ: Mapping[str, str] | None = None,
    env_file: Path | None = DEFAULT_ENV_FILE,
    secrets: Mapping[str, Any] | None = None,
    use_streamlit_secrets: bool = True,
) -> KeyLocation | None:
    """The key and its source: environment, then ``env_file``, then the secrets; ``None`` when no key is configured.

    ``secrets`` replaces ``st.secrets`` (tests); with ``use_streamlit_secrets``
    off and no ``secrets`` mapping the secrets are not consulted at all.
    """
    source = os.environ if environ is None else environ
    from_env = str(source.get(KEY_ENV, "") or "").strip()
    if from_env:
        return KeyLocation(from_env, "ambiente")
    from_file = find_api_key(None, key_env=KEY_ENV, environ={}, env_file=env_file)
    if from_file:
        return KeyLocation(from_file, ".env")
    if secrets is not None:
        value = str(secrets.get(KEY_ENV, "") or "").strip()
        return KeyLocation(value, "Streamlit secrets") if value else None
    if use_streamlit_secrets:
        value = _streamlit_secret(KEY_ENV)
        if value:
            return KeyLocation(value, "Streamlit secrets")
    return None


def find_key(
    *,
    environ: Mapping[str, str] | None = None,
    env_file: Path | None = DEFAULT_ENV_FILE,
    secrets: Mapping[str, Any] | None = None,
    use_streamlit_secrets: bool = True,
) -> str | None:
    """The FootyStats key (environment, ``.env``, Streamlit secrets), ``None`` when none is configured. Never printed or logged."""
    found = locate_key(environ=environ, env_file=env_file, secrets=secrets, use_streamlit_secrets=use_streamlit_secrets)
    return None if found is None else found.value


# =============================================================================
# Age of the local data
# =============================================================================
def _stamp_path(processed_dir: Path) -> Path:
    return Path(processed_dir) / STAMP_FILE


def last_refresh(processed_dir: Path = PROCESSED_DIR) -> dict[str, Any] | None:
    """The content of the stamp written by the last successful :func:`refresh` (``None`` when absent or unreadable)."""
    path = _stamp_path(processed_dir)
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return payload if isinstance(payload, dict) else None


#: Tolerance (seconds) between the modification time of ``matches.csv`` recorded in the stamp and the one on disk.
_MTIME_TOLERANCE: Final[float] = 1.0


def _refreshed_at(processed_dir: Path) -> datetime | None:
    """When the local data was written.

    The stamp's ``refreshed_at`` describes the files :func:`refresh` wrote; the
    stamp also records the modification time of ``matches.csv`` right after the
    write (``matches_mtime``). When ``matches.csv`` is newer than that, someone
    else rewrote it after the stamp -- the launchers run ``python -m
    src.ingestion.footystats``, which writes the CSVs but no stamp -- and its
    modification time is the answer; the same for a stamp without
    ``matches_mtime`` (written by an earlier release) older than the file.
    Without a readable stamp the file time counts; ``None`` without either.
    """
    matches = Path(processed_dir) / "matches.csv"
    try:
        file_mtime: float | None = matches.stat().st_mtime
    except OSError:
        file_mtime = None
    file_time = None if file_mtime is None else datetime.fromtimestamp(file_mtime, tz=UTC)
    stamp = last_refresh(processed_dir)
    stamped: datetime | None = None
    if stamp is not None:
        try:
            stamped = datetime.fromisoformat(str(stamp.get("refreshed_at", "")))
        except ValueError:
            stamped = None
        if stamped is not None and stamped.tzinfo is None:
            stamped = stamped.replace(tzinfo=UTC)
    if stamped is None or file_time is None:
        return stamped if stamped is not None else file_time
    recorded = stamp.get("matches_mtime") if stamp is not None else None
    if isinstance(recorded, (int, float)) and not isinstance(recorded, bool):
        if file_mtime is not None and file_mtime > float(recorded) + _MTIME_TOLERANCE:
            log.debug("matches.csv rewritten after the refresh stamp: its modification time counts", extra={"path": str(matches)})
            return file_time
        return stamped
    return max(stamped, file_time)


def data_age(processed_dir: Path = PROCESSED_DIR, *, now: datetime | None = None) -> timedelta | None:
    """Age of the local data (``None`` when ``matches.csv`` does not exist); a negative age (clock skew) reads as zero."""
    if not (Path(processed_dir) / "matches.csv").is_file():
        return None
    refreshed = _refreshed_at(processed_dir)
    if refreshed is None:
        return None
    current = now if now is not None else datetime.now(UTC)
    if current.tzinfo is None:
        current = current.replace(tzinfo=UTC)
    return max(current - refreshed, timedelta(0))


def needs_refresh(processed_dir: Path = PROCESSED_DIR, *, max_age: timedelta = DEFAULT_MAX_AGE, now: datetime | None = None) -> bool:
    """Whether the local data is missing or older than ``max_age``."""
    age = data_age(processed_dir, now=now)
    return age is None or age > max_age


def refresh_label(processed_dir: Path = PROCESSED_DIR) -> str | None:
    """``'Dati FootyStats aggiornati alle 14:05 del 07/11'`` (Europe/Rome) for the local data, ``None`` without data."""
    if not (Path(processed_dir) / "matches.csv").is_file():
        return None
    refreshed = _refreshed_at(processed_dir)
    if refreshed is None:
        return None
    local = refreshed.astimezone(ROME)
    return f"Dati FootyStats aggiornati alle {local:%H:%M} del {local:%d/%m}"


def age_text(age: timedelta) -> str:
    """``'45 minuti'``, ``'7 ore'``, ``'2 giorni'``: an age in Italian (minutes below one hour, hours below two days)."""
    minutes = max(int(age.total_seconds() // 60), 0)
    if minutes < 60:
        return "1 minuto" if minutes == 1 else f"{minutes} minuti"
    hours = minutes // 60
    if hours < 48:
        return "1 ora" if hours == 1 else f"{hours} ore"
    return f"{hours // 24} giorni"


def stale_warning(
    processed_dir: Path = PROCESSED_DIR,
    *,
    key_available: bool,
    last_error: str | None = None,
    max_age: timedelta = DEFAULT_MAX_AGE,
    now: datetime | None = None,
) -> str | None:
    """The Italian warning for data older than ``max_age`` (``None`` for fresh data or no data).

    ``'Dati vecchi di 2 giorni (ultimo aggiornamento FootyStats alle 13:46 del
    05/11/2026): chiave FootyStats mancante, ...'``: the age, the full date of
    the last download and why it was not renewed -- no key, the error of the
    last attempt (``last_error``, already masked) or an attempt still to come.
    The dashboard shows it on every run while the data stays old, so calendars
    and results days old are never shown as current.
    """
    age = data_age(processed_dir, now=now)
    if age is None or age <= max_age:
        return None
    refreshed = _refreshed_at(processed_dir)
    when = "" if refreshed is None else f" (ultimo aggiornamento FootyStats alle {refreshed.astimezone(ROME):%H:%M del %d/%m/%Y})"
    if not key_available:
        reason = f"chiave FootyStats mancante ({KEY_ENV} in .env, nell'ambiente o nei secrets di Streamlit), i dati non si aggiornano"
    elif last_error:
        reason = f"aggiornamento non riuscito: {last_error}"
    else:
        reason = "aggiornamento non ancora eseguito"
    return (
        f"Dati vecchi di {age_text(age)}{when}: {reason}. Calendari, risultati e quote possono non essere attuali: "
        "le partite senza risultato nei dati compaiono «in corso / risultato in attesa»."
    )


def kept_data_warnings(processed_dir: Path = PROCESSED_DIR) -> list[str]:
    """Italian lines for the competitions that still carry the data of an earlier download (from the stamp of the last refresh).

    A competition (or one season of it) that could not be downloaded keeps its
    previous rows (:func:`refresh`): its calendar and results may be behind the
    others until the next successful download. ``[]`` without a stamp or when
    every competition was renewed.
    """
    stamp = last_refresh(processed_dir)
    if stamp is None:
        return []
    kept = stamp.get("kept") or {}
    lines: list[str] = []
    if isinstance(kept, Mapping):
        for code, info in kept.items():
            if not isinstance(info, Mapping):
                continue
            seasons = ", ".join(f"stagione {x}" for x in info.get("stagioni", []) or [])
            scope = _competition_label(str(code)) + (f", {seasons}" if seasons else "")
            lines.append(
                f"{scope}: non aggiornata all'ultimo download ({info.get('motivo', 'download non riuscito')}): restano i dati del "
                f"download precedente ({int(info.get('partite', 0))} partite, {int(info.get('in_calendario', 0))} in calendario)."
            )
    return lines


def supported_competitions() -> tuple[str, ...]:
    """Codes of the configured competitions FootyStats serves (``config/competitions.toml``), configuration order."""
    return tuple(DEFAULT_COMPETITIONS)


# =============================================================================
# Refresh
# =============================================================================
@dataclass
class RefreshResult:
    """Outcome of :func:`refresh`.

    ``competitions`` are the codes requested; ``per_competition`` maps each
    code that delivered data to ``{"partite": n, "in_calendario": n, "quote": n}``;
    ``errors`` maps each code that delivered nothing to an Italian reason
    (masked: never the key), ``not_chosen`` the codes whose league the account
    has not chosen to the league name, ``season_failures`` the single seasons
    (``'<code>:<season>'``) that failed for a competition that still delivered
    the others; ``kept`` maps each code whose previous rows were kept (a
    competition that failed, or one of its seasons) to ``{"stagioni": [...],
    "partite": n, "in_calendario": n, "motivo": reason}``; ``registry_sync`` holds the lines that
    describe how the team dictionaries changed to follow FootyStats.
    ``written`` says whether the CSVs were saved (``False`` when nothing at all
    could be downloaded: the previous files are then left untouched) and
    ``error`` is the reason of a run that failed as a whole (no key, every
    competition failed). Timestamps are aware UTC. ``n_odds_history`` counts the
    quotes (one per match and market) of completed matches written to the odds
    history by this run (0 when the provider delivered none: the history file
    is then left as it was).
    """

    started_at: datetime
    finished_at: datetime
    competitions: tuple[str, ...]
    processed_dir: Path
    odds_dir: Path
    n_matches: int = 0
    n_fixtures: int = 0
    n_odds: int = 0
    per_competition: dict[str, dict[str, int]] = field(default_factory=dict)
    errors: dict[str, str] = field(default_factory=dict)
    not_chosen: dict[str, str] = field(default_factory=dict)
    season_failures: dict[str, str] = field(default_factory=dict)
    registry_sync: tuple[str, ...] = ()
    written: bool = False
    error: str | None = None
    kept: dict[str, dict[str, Any]] = field(default_factory=dict)
    n_odds_history: int = 0

    @property
    def ok(self) -> bool:
        """Saved and every competition delivered every season."""
        return self.written and not self.errors and not self.season_failures and self.error is None

    @property
    def partial(self) -> bool:
        """Saved, but some competition (or one season of it) could not be downloaded."""
        return self.written and bool(self.errors or self.season_failures)

    @property
    def seconds(self) -> float:
        return max((self.finished_at - self.started_at).total_seconds(), 0.0)

    @property
    def timestamp_text(self) -> str:
        """``'14:05 del 07/11'`` in ``Europe/Rome``."""
        local = self.finished_at.astimezone(ROME)
        return f"{local:%H:%M} del {local:%d/%m}"

    def warnings(self) -> list[str]:
        """Italian lines for ``st.warning``: the global error, then one line per competition skipped or failed."""
        lines: list[str] = []
        if self.error:
            lines.append(self.error)
        for code, reason in self.errors.items():
            label = _competition_label(code)
            if code in self.not_chosen:
                lines.append(
                    f"{label}: saltata, la lega «{self.not_chosen[code]}» non è tra quelle scelte sull'account FootyStats "
                    "(aggiungila su footystats.org, Choose leagues)."
                )
            else:
                lines.append(f"{label}: non scaricata ({reason}).")
            if code in self.kept:
                lines[-1] += f" {self._kept_text(code)}"
        for name, reason in self.season_failures.items():
            code, _, season = name.partition(":")
            if code not in self.errors:
                line = f"{_competition_label(code)}: stagione {season} non scaricata ({reason})."
                if season in (self.kept.get(code) or {}).get("stagioni", []):
                    line += f" {self._kept_text(code)}"
                lines.append(line)
        return lines

    def _kept_text(self, code: str) -> str:
        info = self.kept.get(code) or {}
        return (
            f"Tenuti i dati del download precedente ({int(info.get('partite', 0))} partite, "
            f"{int(info.get('in_calendario', 0))} in calendario)."
        )

    def summary_text(self) -> str:
        """One Italian line for the sidebar and the console."""
        if not self.written:
            return f"Aggiornamento FootyStats non riuscito: {self.error or 'nessun dato scaricato'}"
        done = len(self.per_competition)
        text = (
            f"Dati FootyStats aggiornati alle {self.timestamp_text}: {self.n_matches} partite giocate, "
            f"{self.n_fixtures} in calendario, {self.n_odds} quote ({done}/{len(self.competitions)} competizioni)"
        )
        if self.errors:
            text += f"; {len(self.errors)} competizioni non scaricate"
        return text + "."

    def to_stamp(self) -> dict[str, Any]:
        """The JSON stamp saved next to ``matches.csv`` (no key, no data: counts and reasons only)."""
        return {
            "refreshed_at": self.finished_at.isoformat(),
            "started_at": self.started_at.isoformat(),
            "competitions": list(self.competitions),
            "n_matches": self.n_matches,
            "n_fixtures": self.n_fixtures,
            "n_odds": self.n_odds,
            "n_odds_history": self.n_odds_history,
            "per_competition": self.per_competition,
            "errors": self.errors,
            "not_chosen": self.not_chosen,
            "season_failures": self.season_failures,
            "registry_sync": list(self.registry_sync),
            "kept": self.kept,
        }


def _competition_label(code: str) -> str:
    try:
        return Girone.parse(code).label
    except ValueError:
        return str(code)


@asynccontextmanager
async def _footystats_provider(key: str, *, season: str = DEFAULT_SEASON, config_dir: Path = CONFIG_DIR) -> AsyncIterator[HarvestProvider]:
    """The real provider, wired exactly like the CLI ``python -m src.ingestion.footystats`` (raw cache, team mapping, registry)."""
    from src.ingestion import footystats as fs
    from src.ingestion.harvester import Harvester, RawCache, load_endpoints
    from src.ingestion.team_matcher import TeamMatcher

    endpoints = load_endpoints()
    registry = load_registry(season_file_stem(season), config_dir)
    matcher = TeamMatcher(Harvester.registry_with_history(registry, config_dir))
    cache = RawCache(RAW_DIR, ttl=endpoints.http.cache_ttl, stable_ttl=endpoints.http.history_cache_ttl)
    client = fs.FootyStatsClient(api_key=key, endpoints=endpoints.footystats, http=endpoints.http, cache=cache)
    mapper = fs.FootyStatsTeamMapper(registry, matcher, mapping_path=fs.DEFAULT_MAPPING_PATH)
    provider = fs.FootyStatsProvider(
        client,
        mapper,
        endpoints.footystats,
        config_dir=Path(config_dir),
        registry_path=Path(config_dir) / f"teams_{season_file_stem(season)}.json",
    )
    async with client:
        yield provider


def _sync_lines(report: Any) -> list[str]:
    """Human lines of a registry-sync report (``describe()``, ``lines()`` or ``str``), ``[]`` for ``None`` / no change."""
    if report is None:
        return []
    for name in ("lines", "describe"):
        method = getattr(report, name, None)
        if callable(method):
            value = method()
            text = "\n".join(str(v) for v in value) if isinstance(value, (list, tuple)) else str(value)
            return [line.strip() for line in text.splitlines() if line.strip()]
    text = str(report).strip()
    return [text] if text else []


def _harvest_kwargs(provider: HarvestProvider, seasons: int) -> dict[str, Any]:
    """Keyword arguments of the provider's ``harvest``: seasons, odds and the dictionary sync (``sync_registry``, else the older ``auto_init_registry``)."""
    kwargs: dict[str, Any] = {"seasons": int(seasons), "with_odds": True}
    try:
        parameters = inspect.signature(provider.harvest).parameters
    except (TypeError, ValueError):
        parameters = {}
    if "sync_registry" in parameters:
        kwargs["sync_registry"] = True
    else:
        kwargs["auto_init_registry"] = True
    return kwargs


def _requests_made(provider: HarvestProvider) -> int | None:
    """HTTP requests the provider's client has made so far (``None`` when it does not say)."""
    client = getattr(provider, "client", None)
    stats = getattr(client, "stats", None)
    if isinstance(stats, Mapping):
        value = stats.get("requests")
        return int(value) if isinstance(value, (int, float)) else None
    return None


def _atomic_write(path: Path, writer: Callable[[Path], object]) -> None:
    """Run ``writer(tmp)`` on a temporary file next to ``path``, then replace ``path`` with it."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    handle, tmp_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent))
    os.close(handle)
    tmp = Path(tmp_name)
    try:
        writer(tmp)
        os.replace(tmp, path)
    finally:
        if tmp.exists():
            tmp.unlink()


def _read_odds_file(path: Path) -> pd.DataFrame:
    """The rows of an odds CSV (``match_id,bookmaker,market,selection,odds``), empty when absent or unreadable."""
    try:
        return pd.read_csv(path, dtype={"match_id": "string", "bookmaker": "string", "market": "string", "selection": "string"})
    except (OSError, pd.errors.EmptyDataError, pd.errors.ParserError, UnicodeDecodeError):
        return pd.DataFrame(columns=["match_id", "bookmaker", "market", "selection", "odds"])


def _kept_from_previous(
    processed_dir: Path,
    odds_path: Path,
    failed: set[str],
    failed_seasons: Mapping[str, set[str]] | None = None,
    current_failed: set[str] | None = None,
) -> tuple[pd.DataFrame | None, list[Fixture], pd.DataFrame]:
    """The previous rows of what failed this time, so a failure never erases data.

    * ``failed`` -- competitions that delivered nothing: every previous match,
      fixture and quote of theirs is kept;
    * ``failed_seasons`` -- ``{code: {season label}}`` of the single seasons
      that failed for a competition that delivered the others: the previous
      matches of those seasons are kept;
    * ``current_failed`` -- competitions whose **current** season failed: the
      fixtures come from the current season only, so their previous fixtures
      and quotes are kept too (else the competition would lose its current
      round, its picks and its prices until the next successful download).
    """
    seasons = {code: set(labels) for code, labels in (failed_seasons or {}).items() if labels}
    calendar_codes = set(failed) | set(current_failed or ())
    if not failed and not seasons and not calendar_codes:
        return None, [], pd.DataFrame()
    matches_path = Path(processed_dir) / "matches.csv"
    fixtures_path = Path(processed_dir) / "fixtures.csv"
    old_matches: pd.DataFrame | None = None
    old_fixtures: list[Fixture] = []
    try:
        if matches_path.is_file():
            frame = load_processed(matches_path)
            codes = frame["girone"].astype(str)
            keep = codes.isin(failed)
            if seasons and len(frame):
                labels = frame["date"].map(season_of)
                for code, wanted in seasons.items():
                    keep = keep | (codes.eq(code) & labels.isin(wanted))
            old_matches = frame.loc[keep.to_numpy(dtype=bool)]
        if fixtures_path.is_file():
            old_fixtures = [f for f in load_fixtures(fixtures_path) if f.girone.value in calendar_codes]
    except (OSError, ValueError, TypeError, pd.errors.ParserError) as exc:
        log.warning("previous data unreadable, not kept", extra={"error": str(exc)})
        return None, [], pd.DataFrame()
    ids = {f.match_id for f in old_fixtures}
    odds = _read_odds_file(odds_path)
    old_odds = odds.loc[odds["match_id"].astype(str).isin(ids).to_numpy(dtype=bool)] if len(odds) else odds
    return old_matches, old_fixtures, old_odds


@dataclass
class _Harvested:
    """What one harvest delivered, split per competition."""

    frames: list[pd.DataFrame] = field(default_factory=list)
    fixtures: list[Fixture] = field(default_factory=list)
    odds: list[OddsQuote] = field(default_factory=list)
    #: Pre-match quotes of the completed matches of the competitions that delivered (the odds history).
    odds_history: list[OddsQuote] = field(default_factory=list)
    per_competition: dict[str, dict[str, int]] = field(default_factory=dict)
    errors: dict[str, str] = field(default_factory=dict)
    season_failures: dict[str, str] = field(default_factory=dict)
    not_chosen: dict[str, str] = field(default_factory=dict)
    sync: list[str] = field(default_factory=list)
    #: ``{code: {season label}}`` of the seasons that failed for a competition that delivered others.
    failed_seasons: dict[str, set[str]] = field(default_factory=dict)
    #: Competitions whose current (newest) season failed while an older one was delivered.
    current_failed: set[str] = field(default_factory=set)


async def _run(
    codes: Sequence[str],
    *,
    key: str,
    seasons: int,
    season: str,
    config_dir: Path,
    factory: ProviderFactory,
    progress: ProgressCallback,
) -> _Harvested:
    """One provider harvest for every competition (the dictionary sync needs them together), split per competition.

    The provider isolates the competitions itself (a league not chosen, a
    failed season are recorded, the others delivered); while it runs a ticker
    reports progress from the number of requests made. A failed season is
    mapped back to its competition (``failed_seasons``) and, when it is the
    newest season the provider listed for it (``seasons``; the registry season
    ``season`` when the result does not say), to ``current_failed``.
    """
    out = _Harvested()
    async with factory(key, season=season, config_dir=config_dir) as provider:
        kwargs = _harvest_kwargs(provider, seasons)
        expected = max(len(codes) * (2 * int(seasons) + 2), 1)
        progress(0.02, f"Scarico {len(codes)} competizioni da FootyStats…")
        task = asyncio.ensure_future(provider.harvest(list(codes), **kwargs))
        started = time.perf_counter()
        while not task.done():
            await asyncio.wait({task}, timeout=PROGRESS_INTERVAL)
            if task.done():
                break
            made = _requests_made(provider)
            elapsed = time.perf_counter() - started
            if made is not None:
                progress(min(0.9, 0.02 + 0.88 * made / expected), f"Scarico da FootyStats… {made} richieste")
            else:
                progress(min(0.9, 0.02 + elapsed / 300.0), f"Scarico da FootyStats… {int(elapsed)} s")
        result = task.result()
    for name, league in (getattr(result, "not_chosen", None) or {}).items():
        out.not_chosen[str(name)] = str(league)
    out.sync.extend(_sync_lines(getattr(result, "registry_sync", None)))
    sync_error = getattr(result, "registry_sync_error", None)
    if sync_error:
        out.sync.append(mask_key(f"Dizionario squadre non sincronizzato: {sync_error}", key))
    matches = getattr(result, "matches", None)
    if matches is None:
        matches = pd.DataFrame(columns=["girone"])
    fixtures = list(getattr(result, "fixtures", None) or [])
    odds = list(getattr(result, "odds", None) or [])
    history = list(getattr(result, "odds_history", None) or [])
    failures = dict(getattr(result, "failures", None) or {})
    problem_for = getattr(result, "problem_for", None)
    listed = getattr(result, "seasons", None)
    listed = listed if isinstance(listed, Mapping) else {}
    registry_label = season_file_stem(season).replace("_", "/")
    for code in codes:
        mine = matches.loc[matches["girone"].astype(str).eq(code).to_numpy(dtype=bool)] if len(matches) else matches.iloc[0:0]
        got_fixtures = [f for f in fixtures if f.girone.value == code]
        ids = {f.match_id for f in got_fixtures}
        got_odds = [q for q in odds if q.match_id in ids]
        for name, message in failures.items():
            failed_code, _, failed_label = str(name).partition(":")
            if failed_code == code:
                out.season_failures[str(name)] = mask_key(str(message), key)
                if failed_label:
                    out.failed_seasons.setdefault(code, set()).add(failed_label)
        labels = [str(x) for x in (listed.get(code) or [])]
        newest = labels[0] if labels else registry_label
        if newest in out.failed_seasons.get(code, set()):
            out.current_failed.add(code)
        if not len(mine) and not got_fixtures:
            reason = problem_for(code) if callable(problem_for) else None
            out.errors[code] = mask_key(reason or "nessuna partita restituita da FootyStats", key)
            log.warning("competition delivered nothing", extra={"competition": code, "reason": out.errors[code]})
            continue
        out.frames.append(mine)
        out.fixtures.extend(got_fixtures)
        out.odds.extend(got_odds)
        if history and len(mine):
            played = set(mine["match_id"].astype(str))
            out.odds_history.extend(q for q in history if q.match_id in played)
        out.per_competition[code] = {"partite": int(len(mine)), "in_calendario": len(got_fixtures), "quote": len(got_odds)}
        log.info("competition refreshed", extra={"competition": code, **out.per_competition[code]})
        if code in out.current_failed:
            log.warning(
                "current season of a competition failed: its previous current-season data is kept",
                extra={"competition": code, "season": newest, "reason": out.season_failures.get(f"{code}:{newest}")},
            )
    return out


def _kept_summary(harvested: _Harvested, old_matches: pd.DataFrame | None, old_fixtures: Sequence[Fixture]) -> dict[str, dict[str, Any]]:
    """``{code: {"stagioni", "partite", "in_calendario", "motivo"}}`` of the competitions whose previous rows were kept."""
    out: dict[str, dict[str, Any]] = {}
    counts = old_matches["girone"].astype(str).value_counts().to_dict() if old_matches is not None and len(old_matches) else {}
    n_fixtures: dict[str, int] = {}
    for f in old_fixtures:
        n_fixtures[f.girone.value] = n_fixtures.get(f.girone.value, 0) + 1
    for code in sorted(set(counts) | set(n_fixtures), key=lambda c: (c not in harvested.errors, c)):
        if code in harvested.errors:
            seasons: list[str] = []
            reason = harvested.errors[code]
        else:
            seasons = sorted(harvested.failed_seasons.get(code, set()), reverse=True)
            reason = "; ".join(harvested.season_failures.get(f"{code}:{label}", "") for label in seasons if harvested.season_failures.get(f"{code}:{label}"))
        out[code] = {
            "stagioni": seasons,
            "partite": int(counts.get(code, 0)),
            "in_calendario": int(n_fixtures.get(code, 0)),
            "motivo": reason or "download non riuscito",
        }
    return out


def refresh(
    processed_dir: Path = PROCESSED_DIR,
    odds_dir: Path = ODDS_DIR,
    *,
    competitions: Sequence[str] | None = None,
    seasons: int = DEFAULT_SEASONS,
    progress: ProgressCallback | None = None,
    key: str | None = None,
    provider_factory: ProviderFactory | None = None,
    season: str = DEFAULT_SEASON,
    config_dir: Path = CONFIG_DIR,
    now: Callable[[], datetime] | None = None,
) -> RefreshResult:
    """Download the FootyStats data of ``competitions`` (every supported one by default) and save it locally.

    One provider harvest for all the competitions (``seasons`` seasons each,
    odds included; the team dictionary is synchronised with FootyStats for all
    of them together, so a club moved between gironi is a move, not a removal
    plus an addition). The provider isolates the competitions: one that
    delivers nothing (league not chosen, network failure) is recorded in
    :attr:`RefreshResult.errors`, the others are still saved, and the previous
    rows of a failed competition are kept; when only one season of a
    competition failed (:attr:`RefreshResult.season_failures`) its previous rows
    of that season are kept, and for the current season also its previous
    fixtures and odds (:attr:`RefreshResult.kept` says what was kept). The stamp
    records the modification time of the ``matches.csv`` it describes
    (``matches_mtime``, see :func:`data_age`). The
    files written are ``processed_dir/matches.csv``, ``processed_dir/fixtures.csv``,
    ``odds_dir/footystats.csv`` (:data:`ODDS_FILE`), the odds history
    ``odds_dir/history/odds_history.csv`` (:func:`odds_history_file`, only when
    the provider delivered quotes of completed matches; merged with the rows
    already there, see :func:`src.ingestion.harvester.save_odds_history`) and the stamp
    ``processed_dir/refresh_stamp.json`` (:data:`STAMP_FILE`), each replaced
    atomically. Nothing is written when no competition delivered anything.
    ``key`` defaults to :func:`find_key`; without a key the result carries the
    error and nothing is downloaded. ``progress(fraction, message)`` is called
    while the download runs (from the number of requests made) and at the end. ``provider_factory`` replaces the
    real FootyStats provider (tests); ``now`` the clock. Never raises for a
    download problem: the key never appears in the result, the logs or the files.
    """
    clock = now if now is not None else (lambda: datetime.now(UTC))
    report = progress if progress is not None else (lambda _fraction, _message: None)
    started = clock()
    codes = tuple(dict.fromkeys(str(c).strip().upper() for c in (competitions if competitions is not None else supported_competitions())))
    result = RefreshResult(started_at=started, finished_at=started, competitions=codes, processed_dir=Path(processed_dir), odds_dir=Path(odds_dir))
    secret = key if key is not None and str(key).strip() else find_key()
    if not secret:
        result.error = f"chiave FootyStats non configurata ({KEY_ENV} in .env, nell'ambiente o nei secrets di Streamlit)"
        result.finished_at = clock()
        log.warning("refresh skipped: no FootyStats key", extra={"key_env": KEY_ENV})
        return result
    factory = provider_factory if provider_factory is not None else _footystats_provider
    log.info("FootyStats refresh started", extra={"competitions": list(codes), "seasons": int(seasons)})
    try:
        harvested = asyncio.run(
            _run(codes, key=secret, seasons=seasons, season=season, config_dir=Path(config_dir), factory=factory, progress=report)
        )
    except Exception as exc:  # noqa: BLE001 - setup or a total failure (endpoints, dictionary, network): report, do not raise
        result.error = mask_key(f"aggiornamento non riuscito: {type(exc).__name__}: {exc}", secret)
        result.finished_at = clock()
        log.error("FootyStats refresh failed", extra={"error": result.error})
        report(1.0, "Aggiornamento non riuscito.")
        return result
    result.per_competition, result.errors, result.not_chosen = harvested.per_competition, harvested.errors, harvested.not_chosen
    result.season_failures = harvested.season_failures
    result.registry_sync = tuple(dict.fromkeys(harvested.sync))
    fixtures, odds = harvested.fixtures, harvested.odds
    if not harvested.per_competition:
        result.error = "nessuna competizione scaricata: i dati locali non sono stati modificati"
        result.finished_at = clock()
        log.warning("FootyStats refresh delivered nothing", extra={"errors": harvested.errors})
        report(1.0, "Nessun dato scaricato.")
        return result

    odds_path = Path(odds_dir) / ODDS_FILE
    delivered_failed = {code: labels for code, labels in harvested.failed_seasons.items() if code not in harvested.errors}
    old_matches, old_fixtures, old_odds = _kept_from_previous(
        Path(processed_dir), odds_path, set(harvested.errors), delivered_failed, harvested.current_failed - set(harvested.errors)
    )
    parts = [f for f in harvested.frames if len(f)]
    new_ids = {str(i) for f in parts for i in f["match_id"]}
    fixture_ids = {f.match_id for f in fixtures}
    if old_matches is not None and len(old_matches):
        old_matches = old_matches.loc[~old_matches["match_id"].astype(str).isin(new_ids).to_numpy(dtype=bool)]
        parts.append(old_matches)
    old_fixtures = [f for f in old_fixtures if f.match_id not in fixture_ids and f.match_id not in new_ids]
    kept_ids = {f.match_id for f in old_fixtures}
    if len(old_odds):
        old_odds = old_odds.loc[old_odds["match_id"].astype(str).isin(kept_ids).to_numpy(dtype=bool)]
    result.kept = _kept_summary(harvested, old_matches, old_fixtures)
    matches = pd.concat(parts, ignore_index=True) if parts else empty_matches_frame()
    all_fixtures = [*fixtures, *old_fixtures]
    report(0.95, "Salvo i dati in locale…")
    _atomic_write(Path(processed_dir) / "matches.csv", lambda tmp: save_processed(matches, tmp))
    _atomic_write(Path(processed_dir) / "fixtures.csv", lambda tmp: save_fixtures(all_fixtures, tmp))

    def write_odds(tmp: Path) -> None:
        save_odds(odds, tmp)
        if len(old_odds):
            pd.concat([pd.read_csv(tmp, dtype="string"), old_odds.astype("string")], ignore_index=True).to_csv(tmp, index=False)

    _atomic_write(odds_path, write_odds)
    if harvested.odds_history:
        try:
            save_odds_history(harvested.odds_history, odds_history_file(Path(odds_dir)))
            result.n_odds_history = len(harvested.odds_history)
        except (OSError, ValueError) as exc:  # the history is an extra: never lose the refresh for it
            log.warning("odds history not written", extra={"path": str(odds_history_file(Path(odds_dir))), "error": str(exc)})
    result.n_matches = int(len(matches))
    result.n_fixtures = len(all_fixtures)
    kept_quotes = old_odds[["match_id", "bookmaker", "market"]].drop_duplicates() if len(old_odds) else old_odds
    result.n_odds = len(odds) + int(len(kept_quotes))
    result.written = True
    result.finished_at = clock()
    stamp = result.to_stamp()
    try:
        stamp["matches_mtime"] = (Path(processed_dir) / "matches.csv").stat().st_mtime
    except OSError:
        pass
    _atomic_write(_stamp_path(Path(processed_dir)), lambda tmp: tmp.write_text(json.dumps(stamp, ensure_ascii=False, indent=1), encoding="utf-8"))
    report(1.0, "Dati aggiornati.")
    log.info(
        "FootyStats refresh complete",
        extra={
            "n_matches": result.n_matches,
            "n_fixtures": result.n_fixtures,
            "n_odds": result.n_odds,
            "n_odds_history": result.n_odds_history,
            "competitions_ok": list(result.per_competition),
            "competitions_failed": list(result.errors),
            "competitions_kept": list(result.kept),
            "seconds": round(result.seconds, 1),
        },
    )
    return result


# =============================================================================
# Walk-forward backtest cache (data/backtest.parquet + data/backtest.json, git-ignored)
# =============================================================================
#: The persisted backtest frame and its metadata (cache key, run figures, recalibration), next to the data folder.
BACKTEST_FILE: Final[str] = "backtest.parquet"
BACKTEST_META_FILE: Final[str] = "backtest.json"
_BACKTEST_SCHEMA: Final[int] = 1


def backtest_paths(processed_dir: Path = PROCESSED_DIR) -> tuple[Path, Path]:
    """``(frame, metadata)`` paths of the persisted backtest: ``data/backtest.parquet`` and ``data/backtest.json`` for ``data/processed``."""
    base = Path(processed_dir).parent
    return base / BACKTEST_FILE, base / BACKTEST_META_FILE


def save_backtest(result: BacktestResult, recalibration: RecalibrationModel, *, key: str, processed_dir: Path = PROCESSED_DIR) -> Path | None:
    """Persist a walk-forward backtest with its recalibration under ``key``; ``None`` (logged) when the files cannot be written.

    Aggregates and out-of-sample probabilities of the user's own data, local
    only (the paths are git-ignored, never committed). Written atomically.
    """
    frame_path, meta_path = backtest_paths(processed_dir)
    meta = {
        "schema": _BACKTEST_SCHEMA,
        "key": str(key),
        "written_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "seasons": list(result.seasons),
        "block": result.block,
        "n_blocks": int(result.n_blocks),
        "n_skipped_blocks": int(result.n_skipped_blocks),
        "skipped_matches": {str(k): int(v) for k, v in result.skipped_matches.items()},
        "seconds": float(result.seconds),
        "fit_seconds": float(result.fit_seconds),
        "n_not_converged": int(result.n_not_converged),
        "min_train_matches": int(result.min_train_matches),
        "n_fits": int(result.n_fits),
        "recalibration": recalibration.to_dict(),
    }
    try:
        frame_path.parent.mkdir(parents=True, exist_ok=True)
        _atomic_write(frame_path, lambda tmp: result.frame.to_parquet(tmp, index=False))
        _atomic_write(meta_path, lambda tmp: tmp.write_text(json.dumps(meta, ensure_ascii=False, indent=1), encoding="utf-8"))
    except (OSError, ValueError, ImportError) as exc:
        log.warning("backtest not persisted", extra={"path": str(frame_path), "error": str(exc)})
        return None
    log.info("backtest persisted", extra={"path": str(frame_path), "n_matches": result.n_matches, "key": key})
    return frame_path


def _enabled_rows(frame: pd.DataFrame) -> pd.DataFrame:
    """The rows of a persisted backtest frame whose competition is still enabled (a ``Girone`` member), logged when some are not."""
    if frame.empty or "girone" not in frame.columns:
        return frame
    mask = frame["girone"].astype(str).isin({g.value for g in Girone}).to_numpy(dtype=bool)
    if mask.all():
        return frame
    log.warning(
        "persisted backtest rows of competitions not enabled skipped",
        extra={"codes": sorted(set(frame.loc[~mask, "girone"].astype(str))), "rows": int((~mask).sum())},
    )
    return frame.loc[mask].reset_index(drop=True)


def load_backtest(processed_dir: Path = PROCESSED_DIR, *, key: str) -> tuple[BacktestResult, RecalibrationModel] | None:
    """The backtest persisted under ``key`` (``None`` when missing, unreadable or computed on other data / settings).

    ``key`` carries the competition configuration (see
    :func:`src.dashboard.state.reliability_key`), so a league switched off or
    moved since misses the cache; rows of a competition that is no longer
    enabled are dropped anyway (as the CSV loaders do), never handed to the views.
    """
    frame_path, meta_path = backtest_paths(processed_dir)
    if not (frame_path.is_file() and meta_path.is_file()):
        return None
    try:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        if int(meta.get("schema", 0)) != _BACKTEST_SCHEMA or str(meta.get("key")) != str(key):
            return None
        frame = _enabled_rows(pd.read_parquet(frame_path))
        result = BacktestResult(
            frame=frame,
            seasons=tuple(str(s) for s in meta.get("seasons", [])),
            block=str(meta.get("block", "week")),
            n_blocks=int(meta.get("n_blocks", 0)),
            n_skipped_blocks=int(meta.get("n_skipped_blocks", 0)),
            skipped_matches={str(k): int(v) for k, v in (meta.get("skipped_matches") or {}).items()},
            seconds=float(meta.get("seconds", 0.0)),
            fit_seconds=float(meta.get("fit_seconds", 0.0)),
            n_not_converged=int(meta.get("n_not_converged", 0)),
            min_train_matches=int(meta.get("min_train_matches", 0)),
            n_fits=int(meta.get("n_fits", 0)),
        )
        model = RecalibrationModel.from_dict(meta.get("recalibration") or {})
    except (OSError, ValueError, KeyError, TypeError, ImportError) as exc:
        log.warning("persisted backtest unreadable, recomputing", extra={"path": str(frame_path), "error": str(exc)})
        return None
    log.info("backtest read from disk", extra={"path": str(frame_path), "n_matches": result.n_matches, "key": key})
    return result, model


#: The persisted matchday-by-matchday recalibration of the backtest (its probability columns and steps), same key.
ROLLING_FILE: Final[str] = "backtest_rolling.parquet"
ROLLING_META_FILE: Final[str] = "backtest_rolling.json"
_ROLLING_SCHEMA: Final[int] = 1
_ROLLING_STEP_COLUMNS: Final[tuple[str, ...]] = ("matchday", "n_history", "enabled")


def rolling_paths(processed_dir: Path = PROCESSED_DIR) -> tuple[Path, Path]:
    """``(frame, metadata)`` paths of the persisted rolling recalibration, next to :func:`backtest_paths`."""
    base = Path(processed_dir).parent
    return base / ROLLING_FILE, base / ROLLING_META_FILE


def save_rolling(rolling: RollingRecalibration, *, key: str, processed_dir: Path = PROCESSED_DIR) -> Path | None:
    """Persist the out-of-sample recalibrated probabilities of a backtest (``match_id`` + ``p_*`` columns and the steps) under ``key``.

    The key is the backtest's (:func:`save_backtest`): read back only together
    with the backtest it was computed on. ``None`` (logged) when the files
    cannot be written; local only, git-ignored.
    """
    frame_path, meta_path = rolling_paths(processed_dir)
    columns = ["match_id", *[c for c in rolling.frame.columns if str(c).startswith("p_")]]
    meta = {
        "schema": _ROLLING_SCHEMA,
        "key": str(key),
        "written_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "n_matches": int(len(rolling.frame)),
        "steps": [{c: (int(v) if c == "n_history" else str(v)) for c, v in row.items()} for row in rolling.steps[list(_ROLLING_STEP_COLUMNS)].to_dict(orient="records")],
    }
    try:
        frame_path.parent.mkdir(parents=True, exist_ok=True)
        probabilities = rolling.frame[columns].assign(match_id=rolling.frame["match_id"].astype(str))
        _atomic_write(frame_path, lambda tmp: probabilities.to_parquet(tmp, index=False))
        _atomic_write(meta_path, lambda tmp: tmp.write_text(json.dumps(meta, ensure_ascii=False, indent=1), encoding="utf-8"))
    except (OSError, ValueError, KeyError, ImportError) as exc:
        log.warning("rolling recalibration not persisted", extra={"path": str(frame_path), "error": str(exc)})
        return None
    log.info("rolling recalibration persisted", extra={"path": str(frame_path), "n_matches": int(len(rolling.frame)), "key": key})
    return frame_path


def load_rolling(processed_dir: Path = PROCESSED_DIR, *, key: str, frame: pd.DataFrame) -> RollingRecalibration | None:
    """The rolling recalibration persisted under ``key``, rebuilt on ``frame`` (the backtest frame it was computed on).

    ``None`` when missing, unreadable, saved under another key or not covering
    exactly the matches of ``frame`` (then it is recomputed): the probability
    columns are copied onto ``frame`` match by match, every other column is
    ``frame``'s own.
    """
    frame_path, meta_path = rolling_paths(processed_dir)
    if not (frame_path.is_file() and meta_path.is_file()):
        return None
    try:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        if int(meta.get("schema", 0)) != _ROLLING_SCHEMA or str(meta.get("key")) != str(key):
            return None
        probabilities = pd.read_parquet(frame_path)
        ids = frame["match_id"].astype(str)
        saved = probabilities["match_id"].astype(str)
        if len(probabilities) != len(frame) or saved.duplicated().any() or set(saved) != set(ids):
            return None
        aligned = probabilities.assign(match_id=saved).set_index("match_id").reindex(ids.to_numpy())
        out = frame.copy()
        for column in aligned.columns:
            if column in out.columns:
                out[column] = aligned[column].to_numpy(dtype=float)
        steps = pd.DataFrame(meta.get("steps") or [], columns=list(_ROLLING_STEP_COLUMNS))
        steps["n_history"] = pd.to_numeric(steps["n_history"], errors="coerce").fillna(0).astype(int)
        steps["matchday"] = steps["matchday"].astype(str)
        steps["enabled"] = steps["enabled"].fillna("").astype(str)
    except (OSError, ValueError, KeyError, TypeError, ImportError) as exc:
        log.warning("persisted rolling recalibration unreadable, recomputing", extra={"path": str(frame_path), "error": str(exc)})
        return None
    log.info("rolling recalibration read from disk", extra={"path": str(frame_path), "n_matches": int(len(out)), "key": key})
    return RollingRecalibration(out, steps)


def load_odds_history_frame(odds_dir: Path = ODDS_DIR) -> pd.DataFrame:
    """The odds history of the completed matches (:func:`odds_history_file`), an empty frame when missing or unreadable."""
    path = odds_history_file(Path(odds_dir))
    if not path.is_file():
        return pd.DataFrame()
    try:
        return load_odds_history(path)
    except (OSError, ValueError, pd.errors.ParserError) as exc:
        log.warning("odds history unreadable", extra={"path": str(path), "error": str(exc)})
        return pd.DataFrame()
