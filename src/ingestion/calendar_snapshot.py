"""Calendar snapshot and consistency check against FootyStats, and the round view of the product.

Two read-only modes, used locally and by the ``verifica-calendari`` GitHub
workflow (which never writes FootyStats data to the public repository: it only
prints Markdown to the run log):

**Snapshot** (default) -- downloads, for every configured competition, the
*current* season's team list and fixture list from FootyStats and checks them:

* every competition is among the account's chosen leagues;
* each team plays at most once per round (``game_week``) and every round has
  ``n_teams / 2`` matches;
* postponed matches (``incomplete`` with a kick-off more than three hours in
  the past) are listed instead of silently disappearing;
* the current round is chosen with **the same rule as the product**
  (:func:`src.engine.calendar.current_round`) and listed with every match:
  kick-off in ``Europe/Rome``, status and score of the matches already played;
* the team list of FootyStats is compared with the dictionary
  ``config/teams_<season>.json`` through a *dry run* of the registry sync
  (:func:`src.ingestion.footystats.apply_registry_sync` with ``write=False``):
  the clubs the next harvest would add, remove or move.

**Product view** (``--product-view``) -- reads the CSVs the product itself wrote
(``python -m src.ingestion.footystats --out ... --fixtures-out ...``) and prints,
per competition, the current round exactly as the dashboard shows it (same
helper, canonical team names) plus the postponed matches and recoveries
(:func:`src.engine.calendar.recoveries`).

**Reliability** (``--reliability``) -- reads the product's matches CSV (every
season) and, optionally, its odds history, runs the walk-forward backtest of
the dashboard's *Affidabilità modello* tab (:func:`src.engine.backtest.walk_forward`:
every past week block predicted by a model fitted only on the matches before
it) and prints **aggregated metrics only** (:func:`reliability_markdown`): per
section and market the number of matches, log loss and Brier of the model, of
the league base rates and of the de-vigged bookmaker, the calibration bins with
their counts, the hit rates of the top-N lists and of the three slips with
their Wilson intervals, the aggregate ROI and the recalibration decision. No
match, no team, no price is printed: the repository and its CI logs are public.

Only public calendar facts are handled by the first two modes (rounds, kick-off
times, teams, final scores): no odds and no match statistics. The API key is
read like the harvester does (``--key``, ``FOOTYSTATS_API_KEY``, ``.env``) and is
never printed.

Usage::

    python -m src.ingestion.calendar_snapshot --summary snapshot/riepilogo.md
    python -m src.ingestion.calendar_snapshot --product-view --matches ci/matches.csv --fixtures ci/fixtures.csv
    python -m src.ingestion.calendar_snapshot --reliability --matches ci/matches.csv --odds-history ci/history/odds_history.csv
"""
from __future__ import annotations

import argparse
import asyncio
import json
import sys
from collections import Counter, defaultdict
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from src.common.competitions import COMPETITIONS, Competition
from src.common.logging_config import configure_logging, get_logger
from src.common.paths import CONFIG_DIR
from src.common.schemas import Fixture, Girone, MatchRecord, matches_to_frame, season_of
from src.common.teams import DEFAULT_SEASON, season_file_stem
from src.engine.calendar import ROME, RoundMatch, RoundView, current_round, format_kickoff, recoveries
from src.ingestion.footystats import (
    MAX_PAGES,
    FootyStatsClient,
    MissingApiKeyError,
    ParseReport,
    RegistrySyncReport,
    TeamInfo,
    apply_registry_sync,
    parse_match,
    resolve_api_key,
)
from src.ingestion.harvester import HarvestError, HttpConfig, load_endpoints, load_fixtures, load_processed

__all__ = [
    "MATCH_FIELDS",
    "CompetitionCheck",
    "check_calendar",
    "compare_with_dictionary",
    "kickoff_rome",
    "main",
    "product_view_markdown",
    "round_lines",
    "season_key",
    "snapshot",
    "summary_markdown",
]

log = get_logger(__name__)

#: Match fields copied into the snapshot (public calendar facts only: no odds, no statistics).
MATCH_FIELDS: tuple[str, ...] = (
    "id",
    "game_week",
    "revised_game_week",
    "status",
    "date_unix",
    "homeID",
    "awayID",
    "home_name",
    "away_name",
    "homeGoalCount",
    "awayGoalCount",
)


def season_key(label: str) -> str:
    """``'2026_27'`` / ``'2026/27'`` -> ``'2026/27'``."""
    return str(label).strip().replace("_", "/")


def kickoff_rome(epoch: Any) -> datetime | None:
    """A FootyStats ``date_unix`` as an aware ``Europe/Rome`` datetime (``None`` when unreadable or not positive)."""
    try:
        seconds = int(epoch)
    except (TypeError, ValueError):
        return None
    if seconds <= 0:
        return None
    try:
        return datetime.fromtimestamp(seconds, tz=timezone.utc).astimezone(ROME)
    except (OverflowError, OSError, ValueError):
        return None


def round_lines(view: RoundView | None) -> list[str]:
    """One line per match of ``view``: ``'sab 07/11 14:30  Lecco - Trento  da giocare'`` (``[]`` for ``None``)."""
    if view is None:
        return []
    return [f"{m.kickoff_text}  {m.label}  {m.status_text}" for m in view.matches]


def _recovery_lines(rows: Iterable[RoundMatch]) -> list[str]:
    out: list[str] = []
    for m in rows:
        origin = f", in origine {m.original_date_text}" if m.original_date_text else ""
        out.append(f"g.{m.round} {m.kickoff_text}  {m.label}  {m.status_text}{origin}")
    return out


@dataclass
class CompetitionCheck:
    """The current season of one competition as FootyStats publishes it, with the checks of :func:`check_calendar`."""

    code: str
    label: str
    league: str
    season_id: int | None
    chosen: bool | None = None
    error: str | None = None
    teams: list[dict[str, Any]] = field(default_factory=list)
    matches: list[dict[str, Any]] = field(default_factory=list)
    issues: list[str] = field(default_factory=list)
    postponed: list[str] = field(default_factory=list)
    registry_missing: list[str] = field(default_factory=list)  # FootyStats clubs the sync would add (or move here)
    registry_extra: list[str] = field(default_factory=list)  # dictionary clubs the sync would remove
    next_round: int | None = None
    next_round_lines: list[str] = field(default_factory=list)
    recovery_lines: list[str] = field(default_factory=list)
    status_counts: dict[str, int] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "code": self.code,
            "label": self.label,
            "league": self.league,
            "season_id": self.season_id,
            "chosen": self.chosen,
            "error": self.error,
            "teams": self.teams,
            "matches": self.matches,
            "issues": self.issues,
            "postponed": self.postponed,
            "registry_missing": self.registry_missing,
            "registry_extra": self.registry_extra,
            "next_round": self.next_round,
            "next_round_lines": self.next_round_lines,
            "recovery_lines": self.recovery_lines,
            "status_counts": self.status_counts,
        }


def _parsed_calendar(check: CompetitionCheck, *, now: datetime) -> tuple[pd.DataFrame, list[Fixture], ParseReport]:
    """The raw rows of ``check`` through the product's own parser: played frame, fixtures (scheduled and postponed)."""
    names = {int(t["id"]): str(t.get("cleanName") or t.get("name") or "") for t in check.teams if t.get("id") is not None}
    report = ParseReport()
    records: list[MatchRecord] = []
    fixtures: list[Fixture] = []
    for row in check.matches:
        parsed = parse_match(row, check.code, names, now=now, report=report)
        if isinstance(parsed, MatchRecord):
            records.append(parsed)
        elif isinstance(parsed, Fixture):
            fixtures.append(parsed)
    return matches_to_frame(records), fixtures, report


def check_calendar(check: CompetitionCheck, *, now: datetime) -> None:
    """Fill ``issues``, ``postponed``, ``next_round`` (+ lines), ``recovery_lines`` and ``status_counts`` from the raw rows.

    The current round and the recoveries come from :mod:`src.engine.calendar`
    applied to the output of :func:`~src.ingestion.footystats.parse_match`, so
    the snapshot and the dashboard can never disagree on which round is current.
    """
    now = now if now.tzinfo is not None else now.replace(tzinfo=timezone.utc)
    names = {int(t["id"]): t.get("cleanName") or t.get("name") for t in check.teams}
    n_teams = len(names)
    check.status_counts = dict(Counter(str(m.get("status")) for m in check.matches))
    by_round: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for m in check.matches:
        if str(m.get("status")) in ("suspended", "canceled"):
            continue
        by_round[int(m.get("game_week") or 0)].append(m)
    expected = n_teams // 2 if n_teams else None
    for rnd in sorted(by_round):
        rows = by_round[rnd]
        seen: Counter[int] = Counter()
        for m in rows:
            seen[int(m.get("homeID") or 0)] += 1
            seen[int(m.get("awayID") or 0)] += 1
        doubles = [str(names.get(t, t)) for t, c in seen.items() if c > 1]
        if doubles:
            check.issues.append(f"giornata {rnd}: squadre presenti più volte: {', '.join(sorted(doubles))}")
        if expected and len(rows) != expected:
            check.issues.append(f"giornata {rnd}: {len(rows)} partite invece di {expected}")
        unknown = [t for t in seen if t not in names]
        if unknown:
            check.issues.append(f"giornata {rnd}: id squadra non presenti in league-teams: {unknown}")
    for m in check.matches:
        if kickoff_rome(m.get("date_unix")) is None:
            check.issues.append(f"partita {m.get('id')} senza data")
    played, fixtures, _ = _parsed_calendar(check, now=now)
    for fx in sorted((f for f in fixtures if f.status == "postponed"), key=lambda f: (f.kickoff or datetime.min.replace(tzinfo=timezone.utc), f.match_id)):
        check.postponed.append(f"g.{fx.round} {format_kickoff(fx.kickoff, fx.date)} {fx.home_team} - {fx.away_team} (data passata, non giocata)")
    view = current_round(check.code, played, fixtures, now=now)
    check.next_round = None if view is None else view.round
    check.next_round_lines = round_lines(view)
    # postponed matches are listed in ``postponed``; here the rescheduled recoveries outside the current round
    check.recovery_lines = _recovery_lines(m for m in recoveries(check.code, fixtures, played, now) if not m.postponed)


def compare_with_dictionary(
    checks: Sequence[CompetitionCheck],
    *,
    season: str,
    registry_path: Path,
    mapping_path: Path | None = None,
) -> RegistrySyncReport | None:
    """What the next harvest's registry sync would change, per competition (a dry run: no file is written).

    Fills ``registry_missing`` (clubs FootyStats lists that the sync would add,
    or move here from another competition) and ``registry_extra`` (dictionary
    clubs it would remove) of every check that has a team list, and returns the
    dry-run :class:`~src.ingestion.footystats.RegistrySyncReport` (``None`` when
    the dictionary file does not exist). Identity follows the sync: FootyStats
    id, exact/alias spellings, never fuzzy -- ``'AC Palermo'`` (Serie D) is not
    reported as Serie B's ``'Palermo'``.
    """
    path = Path(registry_path)
    if not path.exists():
        return None
    listings = {c.code: [TeamInfo.from_payload(t) for t in c.teams] for c in checks if c.teams and c.error is None}
    listings = {code: [t for t in teams if t is not None] for code, teams in listings.items()}
    if not listings:
        return None
    kwargs: dict[str, Any] = {} if mapping_path is None else {"mapping_path": Path(mapping_path)}
    report = apply_registry_sync(listings, season=season_key(season), registry_path=path, write=False, **kwargs)
    for check in checks:
        comp = report.competitions.get(check.code)
        if comp is None:
            continue
        check.registry_missing = [*comp.added, *(name for name, _ in comp.moved)]
        check.registry_extra = list(comp.removed)
    return report


async def snapshot(
    competitions: Sequence[Competition],
    *,
    season: str,
    api_key: str,
    now: datetime | None = None,
    http: HttpConfig | None = None,
    transport: Any | None = None,
) -> list[CompetitionCheck]:
    """Download and check the current season of every competition (``transport``: an injected ``httpx`` transport for tests)."""
    endpoints = load_endpoints()
    now = now or datetime.now(tz=ROME)
    label = season_key(season)
    checks: list[CompetitionCheck] = []
    codes = [c.code for c in competitions]
    async with FootyStatsClient(
        api_key=api_key, endpoints=endpoints.footystats, http=http or HttpConfig(max_concurrency=2), transport=transport
    ) as client:
        try:
            resolution = await client.season_resolution(codes, seasons=MAX_PAGES)
        except HarvestError as exc:
            log.warning("league-list unavailable and no configured season ids", extra={"error": client.mask(exc)})
            resolution = None
        for comp in competitions:
            refs = resolution.seasons.get(comp.code, []) if resolution is not None else []
            ref = next((r for r in refs if r.label == label), None)
            season_id = ref.season_id if ref is not None else comp.season_ids.get(label)
            check = CompetitionCheck(code=comp.code, label=comp.label, league=comp.footystats_name, season_id=season_id)
            if resolution is not None and not resolution.from_configured_ids:
                check.chosen = comp.code not in resolution.not_chosen
            if check.chosen is False:
                check.error = "lega non scelta nell'account FootyStats (Choose leagues)"
                checks.append(check)
                continue
            if season_id is None:
                check.error = f"stagione {label} non elencata da FootyStats e nessun season_id configurato"
                checks.append(check)
                continue
            try:
                teams = await client.league_teams(season_id)
                rows = await client.league_matches(season_id)
            except HarvestError as exc:
                check.error = client.mask(str(exc))
                checks.append(check)
                continue
            check.teams = [{"id": t.id, "name": t.name, "cleanName": t.clean_name, "english_name": t.english_name} for t in teams]
            check.matches = [{k: row.get(k) for k in MATCH_FIELDS} for row in rows]
            check_calendar(check, now=now)
            checks.append(check)
            log.info(
                "competition checked",
                extra={"competition": comp.code, "teams": len(check.teams), "matches": len(check.matches), "issues": len(check.issues), "round": check.next_round},
            )
    return checks


def summary_markdown(checks: Iterable[CompetitionCheck], *, season: str, now: datetime, sync: RegistrySyncReport | None = None) -> str:
    """The Markdown report of the snapshot (one table, then one section per competition)."""
    checks = list(checks)
    lines = [f"# Verifica calendari FootyStats · stagione {season_key(season)} · {now.astimezone(ROME):%d/%m/%Y %H:%M}", ""]
    lines.append("| Competizione | Lega FootyStats | Scelta | Squadre | Partite | Giocate | Da giocare | Rinviate | Giornata corrente | Anomalie | Diff. dizionario |")
    lines.append("|---|---|---|---|---|---|---|---|---|---|---|")
    for c in checks:
        played = c.status_counts.get("complete", 0)
        todo = max(c.status_counts.get("incomplete", 0) - len(c.postponed), 0)
        chosen = "?" if c.chosen is None else ("sì" if c.chosen else "NO")
        diff = f"+{len(c.registry_missing)}/-{len(c.registry_extra)}"
        if c.error:
            lines.append(f"| {c.label} | {c.league} | {chosen} | – | – | – | – | – | – | ERRORE: {c.error} | – |")
            continue
        lines.append(
            f"| {c.label} | {c.league} | {chosen} | {len(c.teams)} | {len(c.matches)} | {played} | {todo} | {len(c.postponed)} | {c.next_round or '-'} | {len(c.issues)} | {diff} |"
        )
    if sync is not None:
        lines += ["", "## Dizionario squadre (prova a secco della sincronizzazione)", "", "```", sync.describe(), "```"]
    for c in checks:
        if c.error:
            continue
        lines += ["", f"## {c.label}"]
        if c.next_round_lines:
            lines.append(f"**Giornata corrente (g.{c.next_round}) come la mostra la dashboard, orari Europe/Rome:**")
            lines += [f"- {x}" for x in c.next_round_lines]
        if c.recovery_lines:
            lines.append("**Recuperi fuori dalla giornata corrente:**")
            lines += [f"- {x}" for x in c.recovery_lines]
        if c.postponed:
            lines.append("**Partite rinviate (data passata, non giocate):**")
            lines += [f"- {x}" for x in c.postponed]
        if c.issues:
            lines.append("**Anomalie calendario:**")
            lines += [f"- {x}" for x in c.issues[:30]]
        if c.registry_missing:
            lines.append("**Squadre FootyStats che la sincronizzazione aggiungerebbe:** " + ", ".join(sorted(c.registry_missing)))
        if c.registry_extra:
            lines.append("**Squadre del dizionario che la sincronizzazione rimuoverebbe:** " + ", ".join(c.registry_extra))
    return "\n".join(lines) + "\n"


def product_view_markdown(
    matches: pd.DataFrame,
    fixtures: Sequence[Fixture],
    *,
    season: str,
    now: datetime,
    competitions: Iterable[Competition] = COMPETITIONS,
) -> str:
    """The current round of every competition as the product shows it, from the product's own CSVs.

    ``matches`` is the canonical match frame (every season: only the season
    ``season`` is used), ``fixtures`` the fixture list (scheduled and postponed).
    Per competition: the current round chosen by
    :func:`src.engine.calendar.current_round` with every match (kick-off in
    ``Europe/Rome``, status, score of the played ones), then the postponed
    matches and recoveries of :func:`src.engine.calendar.recoveries`. No odds,
    no statistics.
    """
    label = season_key(season)
    if len(matches):
        seasons = matches["date"].map(season_of)
        played = matches.loc[(seasons == label).to_numpy(dtype=bool)]
    else:
        played = matches
    now = now if now.tzinfo is not None else now.replace(tzinfo=timezone.utc)
    lines = [f"# Giornate come le mostra il prodotto · stagione {label} · {now.astimezone(ROME):%d/%m/%Y %H:%M} (Europe/Rome)", ""]
    for comp in competitions:
        try:
            Girone.parse(comp.code)
        except ValueError:
            continue
        mine_fixtures = [f for f in fixtures if f.girone.value == comp.code]
        n_played = int((played["girone"].astype(str) == comp.code).sum()) if len(played) else 0
        lines.append(f"## {comp.label}")
        if not mine_fixtures and not n_played:
            lines += ["Nessun dato per questa competizione (lega non scelta, raccolta non riuscita o competizione saltata).", ""]
            continue
        view = current_round(comp.code, played, mine_fixtures, now=now)
        if view is None:
            lines.append(f"Nessuna giornata in programma ({n_played} partite giocate nella stagione).")
        else:
            lines.append(f"**Giornata {view.round}**: {view.n_played} giocate, {view.n_to_play} da giocare, {view.n_postponed} rinviate")
            lines += [f"- {x}" for x in round_lines(view)]
        extra = _recovery_lines(recoveries(comp.code, mine_fixtures, played, now))
        if extra:
            lines.append("**Recuperi e rinvii fuori dalla giornata:**")
            lines += [f"- {x}" for x in extra]
        lines.append("")
    return "\n".join(lines) + "\n"


#: Seasons up to the current one evaluated by the CI backtest (the dashboard's default).
RELIABILITY_SEASONS: int = 2


def _fmt(value: Any, digits: int = 4) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "-"
    return "-" if number != number else f"{number:.{digits}f}"  # NaN check without numpy


def _pct_text(value: Any, digits: int = 1) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "-"
    return "-" if number != number else f"{number * 100:.{digits}f}%"


def _md_table(header: Sequence[str], rows: Iterable[Sequence[Any]]) -> list[str]:
    lines = ["| " + " | ".join(header) + " |", "|" + "|".join("---" for _ in header) + "|"]
    lines += ["| " + " | ".join(str(c) for c in row) + " |" for row in rows]
    return lines


#: Minimum number of quoted matches/bets behind any odds-derived aggregate printed in the public CI log:
#: below it, a market score or an ROI could reveal the FootyStats price of a single match.
PUBLIC_MIN_QUOTED: int = 30


def _public(value: str, n_quoted: Any) -> str:
    """``value`` when it rests on at least :data:`PUBLIC_MIN_QUOTED` quoted items, else a placeholder."""
    try:
        enough = int(n_quoted) >= PUBLIC_MIN_QUOTED
    except (TypeError, ValueError):
        enough = False
    return value if enough else f"n<{PUBLIC_MIN_QUOTED}"


def reliability_markdown(result: Any, recalibration: Any, odds_history: pd.DataFrame | None = None) -> str:
    """Aggregated walk-forward metrics as Markdown, safe for a public CI log (no match, no team, no price).

    ``result`` is a :class:`src.engine.backtest.BacktestResult`, ``recalibration``
    the :class:`src.engine.calibration.RecalibrationModel` fitted on it,
    ``odds_history`` the long odds frame of the completed matches (only its
    de-vigged aggregates are scored). Sections: per section and market the
    scores of model, base rates and bookmaker; calibration bins (counts and
    frequencies) of the reference selection of each market over every section;
    hit rates of the top-5 X / top-12 GG lists, of their top-k multiples and of
    the three slips with their Wilson intervals and the aggregate ROI; the
    recalibration decision.
    """
    from src.engine.backtest import ALL_SECTIONS, DEFAULT_MARKETS, market_comparison, market_spec, metrics, picks_track_record, reliability, slips_track_record
    from src.engine.slips import make_slip_builder

    frame = result.frame
    odds = odds_history if odds_history is not None and len(odds_history) else None
    lines = ["# Affidabilità del modello (backtest walk-forward, solo metriche aggregate)", "", result.summary(), ""]
    if frame.empty:
        lines.append("Nessuna partita valutabile: niente metriche.")
        return "\n".join(lines) + "\n"
    lines.append(
        f"Partite con quote pre-partita nello storico: {int(frame['match_id'].isin(set(odds['match_id'].astype(str))).sum()) if odds is not None else 0}."
    )
    lines.append("")
    model_scores = metrics(frame, by=("section",))
    overall = metrics(frame, by=())
    cmp_sec = market_comparison(frame, odds, by=("section",)) if odds is not None else pd.DataFrame()
    cmp_all = market_comparison(frame, odds, by=()) if odds is not None else pd.DataFrame()
    lines.append("## Punteggi per sezione e mercato")
    # the market is scored only on the quoted matches: the model's scores on those same matches sit next to it
    # ("LL modello" / "Brier" cover every evaluated match and are never comparable with the market's)
    header = [
        "Sezione",
        "Mercato",
        "n",
        "LL modello",
        "LL base",
        "Brier",
        "Accuratezza",
        "n con quote",
        "LL modello (con quote)",
        "LL mercato",
        "Brier modello (con quote)",
        "Brier mercato",
        "Skill vs mercato",
    ]
    rows = []
    for scope, scores, cmp in ((ALL_SECTIONS, overall, cmp_all), *((s, model_scores.loc[model_scores["section"] == s], cmp_sec.loc[cmp_sec["section"] == s] if len(cmp_sec) else cmp_sec) for s in dict.fromkeys(model_scores["section"].astype(str)))):
        by_market = {str(r["market"]): r for r in cmp.to_dict(orient="records")} if len(cmp) else {}
        for r in scores.to_dict(orient="records"):
            c = by_market.get(str(r["market"]))
            rows.append(
                [
                    scope,
                    r["market"],
                    int(r["n"]),
                    _fmt(r["log_loss"]),
                    _fmt(r["log_loss_naive"]),
                    _fmt(r["brier"]),
                    _pct_text(r["accuracy"]),
                    int(c["n"]) if c is not None else 0,
                    _public(_fmt(c["log_loss_model"]), c["n"]) if c is not None else "-",
                    _public(_fmt(c["log_loss_market"]), c["n"]) if c is not None else "-",
                    _public(_fmt(c["brier_model"]), c["n"]) if c is not None else "-",
                    _public(_fmt(c["brier_market"]), c["n"]) if c is not None else "-",
                    _public(_pct_text(c["skill"], 2), c["n"]) if c is not None else "-",
                ]
            )
    lines += _md_table(header, rows)
    lines.append("")
    lines.append(
        "Il mercato è misurato solo sulle partite con quote: confrontalo con «LL modello (con quote)» e «Brier modello "
        "(con quote)», calcolate sulle stesse partite, non con «LL modello» (tutte le partite valutate)."
        f" Nel log pubblico ogni valore ricavato dalle quote è mostrato solo con almeno {PUBLIC_MIN_QUOTED} partite quotate."
    )
    lines.append("")
    lines.append("## Taratura (tutte le sezioni): probabilità prevista vs frequenza osservata per fascia")
    for market in DEFAULT_MARKETS:
        spec = market_spec(market)
        selections = spec.selections if spec.is_binary is False else spec.selections[:1]
        for selection in selections:
            table = reliability(frame, market, selection, bins=10)
            lines.append(f"**{spec.key} · {selection}**")
            lines += _md_table(
                ["Fascia", "n", "Prevista", "Osservata", "IC 95%"],
                (
                    [f"{r['bin_low']:.1f}-{r['bin_high']:.1f}", int(r["n"]), _pct_text(r["mean_pred"]), _pct_text(r["observed"]), f"{_pct_text(r['wilson_low'])} - {_pct_text(r['wilson_high'])}"]
                    for r in table.to_dict(orient="records")
                ),
            )
            lines.append("")
    picks = picks_track_record(frame, odds_history=odds, by_section=True)
    lines.append("## Liste di giornata (top 5 X, top 12 GG)")
    lines += _md_table(
        ["Ambito", "Lista", "Giornate", "Selezioni", "Hit rate", "IC 95%", "P media", "Con quota", "ROI", "IC 95% ROI"],
        (
            [r["scope"], r["list"], int(r["n_matchdays"]), int(r["n_picks"]), _pct_text(r["hit_rate"]), f"{_pct_text(r['wilson_low'])} - {_pct_text(r['wilson_high'])}", _pct_text(r["mean_p"]), int(r["n_with_odds"]), _public(_pct_text(r["roi"]), r["n_with_odds"]), _public(f"{_pct_text(r['roi_low'])} - {_pct_text(r['roi_high'])}", r["n_with_odds"])]
            for r in picks.summary.to_dict(orient="records")
        ),
    )
    lines.append("")
    lines.append("## Multiple top-k delle liste (le prime k partite di ogni giornata, tutte vincenti)")
    lines += _md_table(
        ["Ambito", "Multipla", "Giornate", "Vinte", "Attese", "Win rate", "IC 95%", "P media", "Con quota", "ROI", "IC 95% ROI"],
        (
            [r["scope"], f"Top {int(r['k'])} {r['list']}", int(r["n_matchdays"]), int(r["wins"]), _fmt(r["expected_wins"], 2), _pct_text(r["win_rate"]), f"{_pct_text(r['wilson_low'])} - {_pct_text(r['wilson_high'])}", _pct_text(r["mean_p"], 2), int(r["n_with_odds"]), _public(_pct_text(r["roi"]), r["n_with_odds"]), _public(f"{_pct_text(r['roi_low'])} - {_pct_text(r['roi_high'])}", r["n_with_odds"])]
            for r in picks.accumulators.to_dict(orient="records")
        ),
    )
    lines.append("")
    slips = slips_track_record(frame, odds_history=odds, slip_builder=make_slip_builder())
    lines.append("## Schedine (5 X, 10 GG, mista 1 / 2 / GG / Over 2.5 da 6)")
    lines += _md_table(
        ["Schedina", "Giornate", "Vinte", "Attese", "Win rate", "IC 95%", "Gambe vinte", "P media gamba", "Con quota", "ROI", "IC 95% ROI"],
        (
            [r["slip"], int(r["n_slips"]), int(r["wins"]), _fmt(r["expected_wins"], 2), _pct_text(r["win_rate"]), f"{_pct_text(r['wilson_low'])} - {_pct_text(r['wilson_high'])}", _pct_text(r["leg_hit_rate"]), _pct_text(r["mean_leg_p"]), int(r["n_with_odds"]), _public(_pct_text(r["roi"]), r["n_with_odds"]), _public(f"{_pct_text(r['roi_low'])} - {_pct_text(r['roi_high'])}", r["n_with_odds"])]
            for r in slips.summary.to_dict(orient="records")
        ),
    )
    lines.append("")
    lines.append("## Ricalibrazione")
    lines.append(recalibration.describe())
    summary = recalibration.summary_frame()
    lines += _md_table(
        ["Mercato", "Applicata", "n verifica", "LL prima", "LL dopo", "Miglioramento"],
        ([r["market"], "sì" if r["enabled"] else "no", int(r["n_eval"]), _fmt(r["log_loss_before"]), _fmt(r["log_loss_after"]), _pct_text(r["gain"], 2)] for r in summary.to_dict(orient="records")),
    )
    return "\n".join(lines) + "\n"


def run_reliability(matches_path: Path, *, season: str, odds_history_path: Path | None = None, config_dir: Path = CONFIG_DIR, n_seasons: int = RELIABILITY_SEASONS) -> str:
    """Walk-forward backtest of the product's matches CSV and its aggregated Markdown (see :func:`reliability_markdown`)."""
    from src.common.teams import load_registry
    from src.engine.backtest import walk_forward
    from src.engine.calibration import RecalibrationModel, fit_recalibration
    from src.ingestion.harvester import load_odds_history

    matches = load_processed(matches_path)
    try:
        registry = load_registry(season, config_dir=config_dir)
    except (FileNotFoundError, ValueError, TypeError):
        registry = None
    current = season_key(season)
    present = set(matches["date"].map(season_of)) if len(matches) else set()
    seasons = sorted(s for s in present | {current} if s <= current)[-max(int(n_seasons), 1):]
    result = walk_forward(matches, registry=registry, seasons=seasons)
    model = fit_recalibration(result.frame) if not result.empty else RecalibrationModel.identity()
    odds = load_odds_history(odds_history_path) if odds_history_path is not None and Path(odds_history_path).is_file() else None
    return reliability_markdown(result, model, odds)


def _parse_now(value: str | None) -> datetime:
    if not value:
        return datetime.now(tz=ROME)
    moment = datetime.fromisoformat(value.strip())
    return moment if moment.tzinfo is not None else moment.replace(tzinfo=ROME)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m src.ingestion.calendar_snapshot",
        description="Snapshot e verifica dei calendari FootyStats, o vista delle giornate del prodotto (solo calendario: niente quote, niente statistiche)",
    )
    parser.add_argument("--season", default=DEFAULT_SEASON, help="stagione del dizionario squadre (default: %(default)s)")
    parser.add_argument("--competitions", nargs="*", default=None, help="codici (default: tutte le competizioni configurate)")
    parser.add_argument("--out", type=Path, default=None, help="JSON del calendario scaricato (facoltativo: nessun file senza questa opzione)")
    parser.add_argument("--summary", type=Path, default=None, help="riepilogo Markdown (facoltativo; viene sempre stampato)")
    parser.add_argument("--config-dir", type=Path, default=CONFIG_DIR, help="cartella dei dizionari squadre (default: %(default)s)")
    parser.add_argument("--mapping", type=Path, default=None, help="mappa id FootyStats per la prova della sincronizzazione (default: quella del prodotto)")
    parser.add_argument("--key", default=None, help="chiave FootyStats (default: FOOTYSTATS_API_KEY o .env); mai stampata")
    parser.add_argument("--now", default=None, help="istante di riferimento ISO 8601 (default: adesso, Europe/Rome)")
    parser.add_argument("--product-view", action="store_true", help="stampa la giornata corrente di ogni competizione dai CSV del prodotto (--matches, --fixtures)")
    parser.add_argument("--matches", type=Path, default=None, help="CSV partite scritto da python -m src.ingestion.footystats --out")
    parser.add_argument("--fixtures", type=Path, default=None, help="CSV fixture scritto da python -m src.ingestion.footystats --fixtures-out")
    parser.add_argument(
        "--reliability",
        action="store_true",
        help="backtest walk-forward sui CSV del prodotto (--matches, --odds-history facoltativo): stampa solo metriche aggregate",
    )
    parser.add_argument("--odds-history", type=Path, default=None, help="storico quote delle partite giocate (history/odds_history.csv della raccolta)")
    parser.add_argument("--log-level", default="INFO")
    return parser


def _selected(codes: Sequence[str] | None) -> list[Competition]:
    wanted = {str(c).strip().upper() for c in codes} if codes else None
    return [c for c in COMPETITIONS if wanted is None or c.code in wanted]


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entry point; exit codes: 0 ok, 1 a competition could not be checked (or a CSV is missing), 2 missing key."""
    parser = _build_parser()
    args = parser.parse_args(argv)
    configure_logging(args.log_level, force=True)
    now = _parse_now(args.now)
    comps = _selected(args.competitions)
    if args.reliability:
        if args.matches is None:
            parser.error("--reliability needs --matches")
        if not args.matches.exists():
            print(f"affidabilità: file mancante ({args.matches}): la raccolta FootyStats non l'ha scritto", file=sys.stderr)
            return 1
        text = run_reliability(args.matches, season=args.season, odds_history_path=args.odds_history, config_dir=args.config_dir)
        if args.summary is not None:
            args.summary.parent.mkdir(parents=True, exist_ok=True)
            args.summary.write_text(text, encoding="utf-8")
        print(text)
        return 0
    if args.product_view:
        if args.matches is None or args.fixtures is None:
            parser.error("--product-view needs --matches and --fixtures")
        if not args.matches.exists() or not args.fixtures.exists():
            print(f"vista prodotto: file mancante ({args.matches} / {args.fixtures}): la raccolta FootyStats non li ha scritti", file=sys.stderr)
            return 1
        text = product_view_markdown(load_processed(args.matches), load_fixtures(args.fixtures), season=args.season, now=now, competitions=comps)
        if args.summary is not None:
            args.summary.parent.mkdir(parents=True, exist_ok=True)
            args.summary.write_text(text, encoding="utf-8")
        print(text)
        return 0
    try:
        api_key = resolve_api_key(args.key)
    except MissingApiKeyError as exc:
        print(exc.instructions, file=sys.stderr)
        return 2
    checks = asyncio.run(snapshot(comps, season=args.season, api_key=api_key, now=now))
    registry_path = Path(args.config_dir) / f"teams_{season_file_stem(args.season)}.json"
    try:
        sync = compare_with_dictionary(checks, season=args.season, registry_path=registry_path, mapping_path=args.mapping)
    except ValueError as exc:
        log.warning("dictionary comparison skipped", extra={"path": str(registry_path), "error": str(exc)})
        sync = None
    text = summary_markdown(checks, season=args.season, now=now, sync=sync)
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        payload = {"season": season_key(args.season), "generated_at": now.isoformat(), "competitions": [c.to_dict() for c in checks]}
        args.out.write_text(json.dumps(payload, ensure_ascii=False, indent=1), encoding="utf-8")
    if args.summary is not None:
        args.summary.parent.mkdir(parents=True, exist_ok=True)
        args.summary.write_text(text, encoding="utf-8")
    print(text)
    return 0 if all(c.error is None for c in checks) else 1


if __name__ == "__main__":  # pragma: no cover - manual entry point
    sys.exit(main())
