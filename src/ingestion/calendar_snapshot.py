"""Calendar snapshot and consistency check against FootyStats.

Downloads, for every configured competition, the *current* season's team list
and fixture list from FootyStats and checks them:

* every competition is among the account's chosen leagues;
* each team plays at most once per round (``game_week``) and every round has
  ``n_teams / 2`` matches;
* postponed matches (``incomplete`` with a kick-off in the past) are listed
  instead of silently disappearing;
* the team list of FootyStats is compared with the dictionary
  ``config/teams_<season>.json`` (teams missing on either side).

Only public calendar facts are written (rounds, kick-off times, teams, final
scores): no odds and no match statistics. Used locally and by the
``verifica-calendari`` GitHub workflow, which keeps the output as a 1-day
artifact so that FootyStats data is never committed to the (public) repository.

Usage::

    python -m src.ingestion.calendar_snapshot --out snapshot/calendario.json --summary snapshot/riepilogo.md
"""
from __future__ import annotations

import argparse
import asyncio
import json
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence
from zoneinfo import ZoneInfo

from src.common.competitions import COMPETITIONS, Competition
from src.common.logging_config import configure_logging, get_logger
from src.common.teams import DEFAULT_SEASON, TeamRegistry, load_registry, normalize_name
from src.ingestion.footystats import FootyStatsClient, FootyStatsError, MissingApiKeyError, resolve_api_key
from src.ingestion.harvester import HarvestError, HttpConfig, load_endpoints

log = get_logger(__name__)
ROME = ZoneInfo("Europe/Rome")

#: Match fields copied into the snapshot (public calendar facts only).
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
    return label.replace("_", "/")


def kickoff_rome(epoch: Any) -> datetime | None:
    try:
        return datetime.fromtimestamp(int(epoch), tz=timezone.utc).astimezone(ROME)
    except (TypeError, ValueError, OverflowError, OSError):
        return None


@dataclass
class CompetitionCheck:
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
    registry_missing: list[str] = field(default_factory=list)  # in FootyStats, not in the dictionary
    registry_extra: list[str] = field(default_factory=list)  # in the dictionary, not in FootyStats
    next_round: int | None = None
    next_round_lines: list[str] = field(default_factory=list)
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
            "status_counts": self.status_counts,
        }


def check_calendar(check: CompetitionCheck, *, now: datetime) -> None:
    """Fill ``issues``, ``postponed``, ``next_round`` and ``status_counts`` from ``matches``/``teams``."""
    names = {int(t["id"]): t["cleanName"] or t["name"] for t in check.teams}
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
        doubles = [names.get(t, str(t)) for t, c in seen.items() if c > 1]
        if doubles:
            check.issues.append(f"giornata {rnd}: squadre presenti più volte: {', '.join(sorted(doubles))}")
        if expected and len(rows) != expected:
            check.issues.append(f"giornata {rnd}: {len(rows)} partite invece di {expected}")
        unknown = [t for t in seen if t not in names]
        if unknown:
            check.issues.append(f"giornata {rnd}: id squadra non presenti in league-teams: {unknown}")
    upcoming_rounds = []
    for m in check.matches:
        ko = kickoff_rome(m.get("date_unix"))
        if ko is None:
            check.issues.append(f"partita {m.get('id')} senza data")
            continue
        status = str(m.get("status"))
        if status == "incomplete" and ko < now:
            check.postponed.append(
                f"g.{m.get('game_week')} {ko:%d/%m %H:%M} {m.get('home_name')} - {m.get('away_name')} (data passata, non giocata)"
            )
        if status == "incomplete" and ko >= now:
            upcoming_rounds.append(int(m.get("game_week") or 0))
    if upcoming_rounds:
        check.next_round = min(upcoming_rounds)
        rows = sorted(by_round.get(check.next_round, []), key=lambda m: int(m.get("date_unix") or 0))
        for m in rows:
            ko = kickoff_rome(m.get("date_unix"))
            when = f"{ko:%a %d/%m %H:%M}" if ko else "?"
            score = ""
            if str(m.get("status")) == "complete":
                score = f"  {m.get('homeGoalCount')}-{m.get('awayGoalCount')}"
            check.next_round_lines.append(f"{when}  {m.get('home_name')} - {m.get('away_name')}{score}")


def compare_with_registry(check: CompetitionCheck, registry: TeamRegistry | None) -> None:
    """Teams of FootyStats vs the dictionary for the same competition (exact or alias match)."""
    if registry is None:
        return
    try:
        dictionary = set(registry.teams(check.code))
    except (KeyError, ValueError):
        dictionary = set()
    table = registry.alias_table()
    matched: set[str] = set()
    for team in check.teams:
        canonical = None
        for spelling in (team.get("cleanName"), team.get("name"), team.get("english_name")):
            if spelling:
                hit = table.get(normalize_name(str(spelling)))
                if hit in dictionary:
                    canonical = hit
                    break
        if canonical is None:
            check.registry_missing.append(str(team.get("cleanName") or team.get("name")))
        else:
            matched.add(canonical)
    check.registry_extra = sorted(dictionary - matched)


async def snapshot(
    competitions: Sequence[Competition],
    *,
    season: str,
    api_key: str,
    registry: TeamRegistry | None,
    now: datetime | None = None,
) -> list[CompetitionCheck]:
    """Download and check the current season of every competition."""
    endpoints = load_endpoints()
    now = now or datetime.now(tz=ROME)
    label = season_key(season)
    checks: list[CompetitionCheck] = []
    async with FootyStatsClient(api_key=api_key, endpoints=endpoints.footystats, http=HttpConfig(max_concurrency=2)) as client:
        try:
            chosen_names = {league.name for league in await client.league_list(chosen_only=True)}
        except (HarvestError, FootyStatsError) as exc:
            log.warning("league-list unavailable", extra={"error": client.mask(exc)})
            chosen_names = None
        for comp in competitions:
            season_id = comp.season_ids.get(label)
            check = CompetitionCheck(code=comp.code, label=comp.label, league=comp.footystats_name, season_id=season_id)
            check.chosen = None if chosen_names is None else comp.footystats_name in chosen_names
            if season_id is None:
                check.error = f"nessun season_id configurato per {label}"
                checks.append(check)
                continue
            try:
                teams = await client.league_teams(season_id)
                rows = await client.league_matches(season_id)
            except (HarvestError, FootyStatsError) as exc:
                check.error = client.mask(str(exc))
                checks.append(check)
                continue
            check.teams = [
                {"id": t.id, "name": t.name, "cleanName": t.clean_name, "english_name": t.english_name} for t in teams
            ]
            check.matches = [{k: row.get(k) for k in MATCH_FIELDS} for row in rows]
            check_calendar(check, now=now)
            compare_with_registry(check, registry)
            checks.append(check)
            log.info(
                "competition checked",
                extra={"competition": comp.code, "teams": len(check.teams), "matches": len(check.matches), "issues": len(check.issues)},
            )
    return checks


def summary_markdown(checks: Iterable[CompetitionCheck], *, season: str, now: datetime) -> str:
    lines = [f"# Verifica calendari FootyStats · stagione {season_key(season)} · {now:%d/%m/%Y %H:%M}", ""]
    lines.append("| Competizione | Lega FootyStats | Scelta | Squadre | Partite | Giocate | Da giocare | Rinviate | Prossima g. | Anomalie | Diff. dizionario |")
    lines.append("|---|---|---|---|---|---|---|---|---|---|---|")
    for c in checks:
        played = c.status_counts.get("complete", 0)
        todo = c.status_counts.get("incomplete", 0) - len(c.postponed)
        chosen = "?" if c.chosen is None else ("sì" if c.chosen else "NO")
        diff = f"+{len(c.registry_missing)}/-{len(c.registry_extra)}"
        if c.error:
            lines.append(f"| {c.label} | {c.league} | {chosen} | – | – | – | – | – | – | ERRORE: {c.error} | – |")
            continue
        lines.append(
            f"| {c.label} | {c.league} | {chosen} | {len(c.teams)} | {len(c.matches)} | {played} | {todo} | {len(c.postponed)} | {c.next_round or '-'} | {len(c.issues)} | {diff} |"
        )
    for c in checks:
        if c.error:
            continue
        lines += ["", f"## {c.label}"]
        if c.next_round_lines:
            lines.append(f"**Prossima giornata (g.{c.next_round}), orari Europe/Rome:**")
            lines += [f"- {x}" for x in c.next_round_lines]
        if c.postponed:
            lines.append("**Partite rinviate / non ancora giocate con data passata:**")
            lines += [f"- {x}" for x in c.postponed]
        if c.issues:
            lines.append("**Anomalie calendario:**")
            lines += [f"- {x}" for x in c.issues[:30]]
        if c.registry_missing:
            lines.append("**Squadre FootyStats assenti dal dizionario:** " + ", ".join(sorted(c.registry_missing)))
        if c.registry_extra:
            lines.append("**Squadre del dizionario assenti in FootyStats:** " + ", ".join(c.registry_extra))
    return "\n".join(lines) + "\n"


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Snapshot e verifica dei calendari FootyStats (solo calendario, niente quote)")
    parser.add_argument("--season", default=DEFAULT_SEASON)
    parser.add_argument("--competitions", nargs="*", default=None, help="codici (default: tutte le competizioni configurate)")
    parser.add_argument("--out", type=Path, default=Path("snapshot/calendario.json"))
    parser.add_argument("--summary", type=Path, default=Path("snapshot/riepilogo.md"))
    parser.add_argument("--key", default=None)
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args(argv)
    configure_logging(args.log_level, force=True)
    try:
        api_key = resolve_api_key(args.key)
    except MissingApiKeyError as exc:
        print(exc.instructions(), file=sys.stderr)
        return 2
    comps = [c for c in COMPETITIONS if not args.competitions or c.code in set(args.competitions)]
    try:
        registry = load_registry(args.season)
    except FileNotFoundError:
        registry = None
    now = datetime.now(tz=ROME)
    checks = asyncio.run(snapshot(comps, season=args.season, api_key=api_key, registry=registry, now=now))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps({"season": season_key(args.season), "generated_at": now.isoformat(), "competitions": [c.to_dict() for c in checks]}, ensure_ascii=False, indent=1),
        encoding="utf-8",
    )
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    args.summary.write_text(summary_markdown(checks, season=args.season, now=now), encoding="utf-8")
    print(args.summary.read_text(encoding="utf-8"))
    return 0 if all(c.error is None for c in checks) else 1


if __name__ == "__main__":
    sys.exit(main())
