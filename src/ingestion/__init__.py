"""Ingestion layer: async harvesters for SofaScore / FotMob and team-name matching.

Public names are re-exported lazily (PEP 562) so that importing the package is
cheap and ``python -m src.ingestion.harvester`` does not import the harvester
module twice.
"""
from __future__ import annotations

from importlib import import_module
from typing import Any

_EXPORTS: dict[str, str] = {
    "DEFAULT_USER_AGENTS": "src.ingestion.harvester",
    "AsyncHarvesterClient": "src.ingestion.harvester",
    "ClientStats": "src.ingestion.harvester",
    "EndpointsConfig": "src.ingestion.harvester",
    "FotMobEndpoints": "src.ingestion.harvester",
    "FotMobProvider": "src.ingestion.harvester",
    "HarvestError": "src.ingestion.harvester",
    "HarvestResult": "src.ingestion.harvester",
    "Harvester": "src.ingestion.harvester",
    "HttpConfig": "src.ingestion.harvester",
    "NotFoundError": "src.ingestion.harvester",
    "RateLimitedError": "src.ingestion.harvester",
    "RawCache": "src.ingestion.harvester",
    "SeasonHarvest": "src.ingestion.harvester",
    "SeasonInfo": "src.ingestion.harvester",
    "SofaScoreEndpoints": "src.ingestion.harvester",
    "SofaScoreProvider": "src.ingestion.harvester",
    "load_endpoints": "src.ingestion.harvester",
    "load_fixtures": "src.ingestion.harvester",
    "load_processed": "src.ingestion.harvester",
    "parse_retry_after": "src.ingestion.harvester",
    "parse_stat_value": "src.ingestion.harvester",
    "run_concurrently": "src.ingestion.harvester",
    "save_fixtures": "src.ingestion.harvester",
    "save_processed": "src.ingestion.harvester",
    "season_label_from_year": "src.ingestion.harvester",
    "DIFFLIB_FALLBACK": "src.ingestion.team_matcher",
    "MatchResult": "src.ingestion.team_matcher",
    "TeamMatcher": "src.ingestion.team_matcher",
}

__all__ = sorted(_EXPORTS)


def __getattr__(name: str) -> Any:
    try:
        module_name = _EXPORTS[name]
    except KeyError:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from None
    value = getattr(import_module(module_name), name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(_EXPORTS))
