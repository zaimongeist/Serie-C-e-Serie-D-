"""The FootyStats registry sync: the current season's team dictionary equals ``/league-teams``.

Offline: the listings are built from the real 2026/27 compositions read on FootyStats on 2026-11-07 (the
``cleanName`` spellings) or are small synthetic files; the async entry point runs on ``httpx.MockTransport``.
Every test works on a temporary copy of the dictionaries, and an autouse guard checks that the repository's
``config/*.json`` and ``data/`` are never written.
"""
from __future__ import annotations

import hashlib
import json
import logging
import random
import shutil
from collections.abc import Callable
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import httpx
import pytest

from src.common.paths import CONFIG_DIR, DATA_DIR, ROOT
from src.common.schemas import Girone
from src.common.teams import TeamRegistry, dumps_registry_payload, load_registry, normalize_name
from src.ingestion import footystats as fs
from src.ingestion.harvester import Harvester, HttpConfig, load_endpoints
from src.ingestion.team_matcher import TeamMatcher

KEY = "registry-sync-test-key-0123456789"
HOST = "api.football-data-api.com"
NOW = datetime(2026, 11, 7, 12, 0, tzinfo=timezone.utc)

#: FootyStats /league-teams of 2026/27 (cleanName), read live through the verifica-calendari workflow on 2026-11-07.
FOOTYSTATS_2026_27: dict[str, tuple[str, ...]] = {
    "A": ("Union Brescia", "Arzignano Valchiampo", "Alcione", "Lumezzane", "Pro Vercelli", "Dolomiti Bellunesi", "Pergolettese", "Renate", "Trento", "AlbinoLeffe", "Giana Erminio", "Cittadella", "Lecco", "Juventus II", "Desenzano Calvina", "Ospitaletto", "Folgore Caratese", "Athletic Carpi", "Treviso", "Novara"),
    "B": ("Atalanta II", "Reggiana", "Ostia Mare Lidocalcio", "Spezia", "Vado", "Ravenna", "Forlì", "Gubbio", "Grosseto", "Pescara", "Sassari Torres", "Latina", "Città di Campobasso", "Livorno", "Sambenedettese", "Nuova Monterosi", "Pianese", "Perugia", "Pineto", "Vis Pesaro"),
    "C": ("Inter II", "Salernitana", "Casertana", "Virtus Casarano", "Crotone", "SSC Giugliano", "Scafatese", "Audace Cerignola", "Savoia", "Potenza Calcio", "Monopoli", "Cosenza", "Catania", "Sorrento Calcio", "Foggia", "Barletta", "Team Altamura", "Cavese", "Picerno", "Bari 1908"),
    "SB": ("Ascoli", "Palermo", "Cesena", "Catanzaro", "Juve Stabia", "Arezzo", "Calcio Padova", "Empoli", "Pisa", "Südtirol", "Mantova", "Hellas Verona", "Benevento", "Avellino", "Virtus Entella", "Modena", "Carrarese", "Sampdoria", "Vicenza", "Cremonese"),
    "DA": ("AS Biellese 1902", "Asti", "Borgosesia", "Bra", "Celle Varazze", "Chisola", "Derthona", "Fezzanese", "Gozzano", "Imperia", "Lascaris", "Ligorna", "Millesimo", "Saluzzo", "Sanremo", "Sestri Levante", "US Alessandria", "Valenzana"),
    "DB": ("Brusaporto", "Calcio Pavonese", "Caldiero Terme", "Chievo", "Club Milano", "Fiorenzuola", "Leon", "Milan II", "Nibbiano & Valtidone", "Piacenza", "Pro Palazzolo", "Real Calepina", "Rovato Vertova", "Scanzorosciate", "Tritium", "Villa Valle", "Virtus Ciserano Bergamo", "Virtus Verona"),
    "DC": ("Altavilla", "Bassano Virtus", "Brian Lignano", "Calvi Noale", "Campodarsego", "Cjarlins Muzane", "Clodiense", "Conegliano", "Este", "Lavarian Mortean Esperia", "Legnago Salus", "Luparense", "Mestre", "Obermais", "Sandonà", "Schio", "Triestina", "Virtus Bolzano"),
    "DD": ("Arconatese", "Casatese", "Castellanzese", "Cittadella Vis Modena", "Città di Varese", "Coriano", "Correggese", "Crema", "Lentigione", "Oltrepò", "Pavia", "Pistoiese", "Pontedera", "Pro Patria", "Pro Sesto", "Sant'Angelo", "Solbiatese 1911", "Varesina"),
    "DE": ("Flaminia", "Gavorrano", "Ghivizzano Borgo Mozzano", "Grassina", "Lucchese", "Mezzolara", "Montevarchi Calcio", "Prato", "Progresso", "Robur Siena", "Rondinella Marzocco", "San Donato Tavarnelle", "Sasso Marconi", "Scandicci", "Seravezza", "Tau Altopascio", "Ternana", "Terranuova Traiana"),
    "DF": ("ACD Maceratese", "Ancona 1905", "Angelana", "Atletico Ascoli", "Foligno", "Fossombrone", "L'Aquila", "Montecchio Gallo", "Pietralunghese", "Real Giulianova Ssd", "Recanatese", "San Nicolò", "Santegidiese SSD", "Sporting Trestina", "Teramo", "Termoli Calcio", "Vigor Senigallia", "Virtus Lanciano"),
    "DG": ("Afragolese", "Albalonga", "Anzio", "Aranova", "Atletico Lodigiani", "Budoni", "Certosa V. Campagnano", "Città di Anagni", "Gelbison", "Latte Dolce", "Monastir", "Ossese", "Paganese", "Sarnese", "Sarrabus Ogliastra", "Trastevere", "US Venafro", "Unipomezia"),
    "DH": ("Aversa Normanna", "Bisceglie", "Brindisi", "Ebolitana", "Fidelis Andria", "Francavilla", "Gladiator", "Gravina", "Ischia", "Manfredonia", "Martina Franca", "Melfi", "Nardò", "Nocerina", "Real Forio", "Turris Neapolis", "USD Palmese", "Virtus Francavilla"),
    "DI": ("AC Palermo", "Avola", "CastrumFavara", "Digiesse PraiaTortora", "Enna", "Gela", "Igea Virtus", "Licata", "Milazzo", "Modica", "Nissa", "Ragusa", "Reggina", "Sambiase", "Siracusa", "Trapani 1905", "Vibonese", "Vigor Lamezia"),
}

#: FootyStats spelling -> the canonical name the shipped dictionary keeps (the club is the same).
OUR_NAMES = {
    "Juventus II": "Juventus Next Gen",
    "Atalanta II": "Atalanta U23",
    "Inter II": "Inter U23",
    "Città di Campobasso": "Campobasso",
    "Ostia Mare Lidocalcio": "Ostiamare",
    "Sassari Torres": "Torres",
    "SSC Giugliano": "Giugliano",
    "Virtus Casarano": "Casarano",
    "Athletic Carpi": "Carpi",
    "Alcione": "Alcione Milano",
    "Desenzano Calvina": "Desenzano",
    "Potenza Calcio": "Potenza",
    "Sorrento Calcio": "Sorrento",
    "Bari 1908": "Bari",
    "Calcio Padova": "Padova",
    "AC Palermo": "AC Palermo Serie D",
    "Trapani 1905": "Trapani",
}


def _team_ids() -> dict[tuple[str, str], int]:
    ids: dict[tuple[str, str], int] = {}
    for index, (code, names) in enumerate(FOOTYSTATS_2026_27.items()):
        for position, name in enumerate(names):
            ids[(code, name)] = 10_000 + 100 * index + position
    return ids


TEAM_IDS = _team_ids()


def real_listings(**overrides: dict[str, str]) -> dict[str, list[fs.TeamInfo]]:
    """The 2026/27 listings as TeamInfo (name = cleanName unless ``overrides[code][clean] = name``)."""
    out: dict[str, list[fs.TeamInfo]] = {}
    for code, names in FOOTYSTATS_2026_27.items():
        extra = overrides.get(code, {})
        out[code] = [fs.TeamInfo(TEAM_IDS[(code, n)], extra.get(n, n), n) for n in names]
    return out


def _repository_state() -> dict[str, str]:
    paths = [*sorted(CONFIG_DIR.glob("*.json")), *sorted(p for p in DATA_DIR.rglob("*") if p.is_file())]
    return {str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest() for path in paths}


@pytest.fixture(autouse=True)
def _repository_untouched() -> Any:
    before = _repository_state()
    yield
    assert _repository_state() == before, "a test wrote config/*.json or data/ of the repository"


@pytest.fixture
def config_copy(tmp_path: Path) -> Path:
    directory = tmp_path / "config"
    directory.mkdir()
    for name in ("teams_2026_27.json", "teams_2025_26.json"):
        shutil.copy(CONFIG_DIR / name, directory / name)
    return directory


def write_dictionary(path: Path, gironi: dict[str, dict[str, list[str]]], season: str = "2026/27") -> Path:
    path.write_text(dumps_registry_payload({"season": season, "gironi": gironi}), encoding="utf-8")
    return path


# ============================================================================ the shipped dictionary
def test_shipped_dictionary_has_242_clubs_in_13_competitions_and_footystats_spellings_resolve() -> None:
    registry = load_registry()
    assert registry.n_teams == 242 and registry.gironi == tuple(Girone) and len(registry.gironi) == 13 and registry.missing_competitions() == ()
    for code, names in FOOTYSTATS_2026_27.items():
        assert len(registry.teams(code)) == len(names) == (18 if code.startswith("D") else 20), code
        for name in names:
            canonical = OUR_NAMES.get(name, name)
            assert registry.girone_of(canonical) is Girone(code), (code, name)
            if name != "AC Palermo":  # normalises like Serie B 'Palermo': only the FootyStats id tells them apart
                assert registry.exact(name) == canonical, (code, name)
    assert registry.exact("Juventus II") == "Juventus Next Gen" and registry.girone_of("Juventus Next Gen") is Girone.A
    assert "Nuova Monterosi" in registry.teams("B") and "Guidonia Montecelio" not in registry.canonical_names
    assert registry.exact("Guidonia") is None and normalize_name("AC Palermo") == normalize_name("Palermo")
    assert registry.girone_of("Palermo") is Girone.SB and registry.girone_of("AC Palermo Serie D") is Girone.DI
    # the previous season knows the B teams under the FootyStats spellings too, and Guidonia is history now
    old = load_registry("2025_26")
    assert old.exact("Juventus II") == "Juventus Next Gen" and old.exact("Atalanta II") == "Atalanta U23" and old.exact("Inter II") == "Inter U23"
    merged = load_registry(with_history=True)
    assert "Guidonia Montecelio" in merged.historical_names and merged.exact("Guidonia") == "Guidonia Montecelio"
    # the file is written in the compact one-club-per-line layout the sync uses
    text = (CONFIG_DIR / "teams_2026_27.json").read_text(encoding="utf-8")
    assert dumps_registry_payload(json.loads(text)) == text


def test_sync_of_the_shipped_dictionary_with_the_real_lists_changes_nothing(config_copy: Path, tmp_path: Path) -> None:
    path = config_copy / "teams_2026_27.json"
    before = path.read_bytes()
    report = fs.apply_registry_sync(real_listings(), season="2026/27", registry_path=path, mapping_path=tmp_path / "map.json")
    assert not report.changed and not report.written and report.n_changes == 0 and report.total == 242 and path.read_bytes() == before
    assert list(report.competitions) == [g.value for g in Girone] and all(not c.changed for c in report.competitions.values())
    ids = report.assignments
    assert ids["DI"][TEAM_IDS[("DI", "AC Palermo")]] == "AC Palermo Serie D" and ids["SB"][TEAM_IDS[("SB", "Palermo")]] == "Palermo"
    assert ids["A"][TEAM_IDS[("A", "Juventus II")]] == "Juventus Next Gen" and ids["DI"][TEAM_IDS[("DI", "Trapani 1905")]] == "Trapani"
    assert report.skipped_aliases == {"AC Palermo Serie D": ("AC Palermo",)}, "the colliding spelling is reported, never written"
    assert report.describe().startswith("Dizionario squadre 2026/27 già allineato a FootyStats") and "nessuna variazione" in report.describe()
    # the id mapping now identifies every club (method 'sync'), and a second run is a no-op for it too
    mapping = json.loads((tmp_path / "map.json").read_text(encoding="utf-8"))["teams"]
    assert len(mapping) == 242 and {entry["method"] for entry in mapping.values()} == {fs.SYNC_METHOD}
    again = fs.apply_registry_sync(real_listings(), season="2026/27", registry_path=path, mapping_path=tmp_path / "map.json")
    assert not again.changed and not again.mapping_changed and again.assignments == report.assignments


def test_sync_from_the_former_dictionary_reaches_the_real_composition(tmp_path: Path) -> None:
    """The dictionary shipped before this change had Guidonia Montecelio in girone B and an empty Serie D."""
    shipped = json.loads((CONFIG_DIR / "teams_2026_27.json").read_text(encoding="utf-8"))
    former = {code: dict(members) for code, members in shipped["gironi"].items()}
    former["B"] = {**{k: v for k, v in former["B"].items() if k != "Nuova Monterosi"}, "Guidonia Montecelio": ["Guidonia"]}
    for code in [c for c in former if c.startswith("D")]:
        former[code] = {}
    path = write_dictionary(tmp_path / "teams_2026_27.json", former)
    report = fs.apply_registry_sync(real_listings(), season="2026/27", registry_path=path, mapping_path=tmp_path / "map.json")
    assert report.changed and report.written and report.total == 242
    assert report.competitions["B"].added == ("Nuova Monterosi",) and report.competitions["B"].removed == ("Guidonia Montecelio",)
    assert all(len(report.competitions[code].added) == 18 for code in FOOTYSTATS_2026_27 if code.startswith("D"))
    assert "AC Palermo Serie D" in report.competitions["DI"].added and "Palermo" in TeamRegistry.from_json(path).teams("SB")
    synced = TeamRegistry.from_json(path)
    assert set(synced.canonical_names) == set(load_registry().canonical_names) - {"Trapani"} | {"Trapani 1905"}
    assert "Serie C Girone B (20 squadre): aggiunte Nuova Monterosi; rimosse Guidonia Montecelio" in report.describe()


# ============================================================================ sync semantics on small dictionaries
def test_sync_adds_removes_moves_and_reports_in_italian(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    path = write_dictionary(
        tmp_path / "teams_2026_27.json",
        {"A": {"Alfa": ["Alfa FC"], "Beta": [], "Gamma": ["Gamma 1920"]}, "B": {"Delta": [], "Epsilon": []}, "DA": {"Zeta": []}},
    )
    listings = {
        "A": [fs.TeamInfo(1, "Alfa Calcio", "Alfa"), fs.TeamInfo(2, "Beta", "Beta"), fs.TeamInfo(3, "Omega", "Omega", "Omega United")],
        "B": [fs.TeamInfo(4, "Delta", "Delta"), fs.TeamInfo(5, "Epsilon", "Epsilon"), fs.TeamInfo(6, "Gamma", "Gamma")],
    }
    with caplog.at_level(logging.INFO):
        report = fs.apply_registry_sync(listings, season="2026_27", registry_path=path, mapping_path=tmp_path / "map.json")
    a, b = report.competitions["A"], report.competitions["B"]
    assert a.added == ("Omega",) and a.removed == () and a.moved == () and a.total == 3
    assert b.moved == (("Gamma", "A"),) and b.added == () and b.removed == () and b.total == 3
    assert report.moved == {"Gamma": ("A", "B")} and report.added == {"A": ("Omega",)} and report.removed == {}
    assert report.n_changes == 2 and report.changed and report.written and report.total == 7 and "DA" not in report.competitions
    saved = json.loads(path.read_text(encoding="utf-8"))["gironi"]
    assert saved["A"] == {"Alfa": ["Alfa FC"], "Beta": [], "Omega": ["Omega United"]}, "existing aliases are preserved, survivors keep their order"
    assert saved["B"]["Gamma"] == ["Gamma 1920"] and saved["DA"] == {"Zeta": []}, "a moved club keeps its aliases; other competitions are untouched"
    text = report.describe()
    assert text.startswith(f"Dizionario squadre 2026/27 sincronizzato con FootyStats ({path}): 2 variazioni, 7 squadre in totale.")
    assert "Serie C Girone A (3 squadre): aggiunte Omega" in text and "Serie C Girone B (3 squadre): spostate qui Gamma (da Serie C Girone A)" in text
    assert "Nuovi alias per 1 squadre: Omega <- Omega United" in text
    record = next(r for r in caplog.records if r.getMessage() == "team dictionary synchronised with FootyStats")
    assert record.added == {"A": ["Omega"]} and record.moved == {"Gamma": ["A", "B"]} and record.changed and record.total == 7
    # FootyStats drops a club: it is removed from the competition (it stays in the older season files as history)
    shrunk = {"A": listings["A"][:2], "B": listings["B"]}
    second = fs.apply_registry_sync(shrunk, season="2026/27", registry_path=path, mapping_path=tmp_path / "map.json")
    assert second.competitions["A"].removed == ("Omega",) and second.removed == {"A": ("Omega",)} and "rimosse Omega" in second.describe()
    assert TeamRegistry.from_json(path).teams("A") == ("Alfa", "Beta")


def test_identity_by_footystats_id_is_persisted_and_follows_renames(tmp_path: Path) -> None:
    path = write_dictionary(tmp_path / "teams_2026_27.json", {"DA": {}, "A": {"Juventus Next Gen": ["Juventus II"]}})
    mapping = tmp_path / "data" / "footystats_teams.json"
    first = fs.apply_registry_sync(
        {"DA": [fs.TeamInfo(7, "Borgo FC", "Borgo")], "A": [fs.TeamInfo(8, "Juventus II", "Juventus II")]},
        season="2026/27", registry_path=path, mapping_path=mapping,
    )  # fmt: skip
    assert first.competitions["DA"].added == ("Borgo",) and mapping.exists()
    entries = json.loads(mapping.read_text(encoding="utf-8"))["teams"]
    assert entries["7"] == {"canonical": "Borgo", "clean_name": "Borgo", "competition": "DA", "seasons": ["2026/27"], "method": fs.SYNC_METHOD}
    assert fs.load_team_mapping(mapping)[8]["canonical"] == "Juventus Next Gen"
    # FootyStats renames both clubs: the id keeps the identity; a canonical that was the cleanName follows the rename
    # (the old spelling becomes an alias), a canonical chosen by hand stays and gains the new spelling as an alias
    second = fs.apply_registry_sync(
        {"DA": [fs.TeamInfo(7, "Polisportiva Borgo Nuovo", "Borgo Nuovo")], "A": [fs.TeamInfo(8, "Juventus Under 23", "Juventus Under 23")]},
        season="2026/27", registry_path=path, mapping_path=mapping,
    )  # fmt: skip
    assert second.competitions["DA"].renamed == (("Borgo", "Borgo Nuovo"),) and second.renamed == {"Borgo": "Borgo Nuovo"}
    assert second.competitions["DA"].added == () and second.competitions["DA"].removed == ()
    assert second.competitions["A"].renamed == () and second.competitions["A"].added == ()
    saved = json.loads(path.read_text(encoding="utf-8"))["gironi"]
    assert saved["DA"] == {"Borgo Nuovo": ["Borgo", "Polisportiva Borgo Nuovo"]} and saved["A"] == {"Juventus Next Gen": ["Juventus II", "Juventus Under 23"]}
    assert "rinominate Borgo -> Borgo Nuovo" in second.describe() and second.assignments["DA"] == {7: "Borgo Nuovo"}
    # an id whose dictionary entry was removed gets its old canonical back when it returns
    fs.apply_registry_sync({"DA": [fs.TeamInfo(9, "Altro", "Altro")]}, season="2026/27", registry_path=path, mapping_path=mapping)
    back = fs.apply_registry_sync({"DA": [fs.TeamInfo(7, "Borgo Nuovissimo", "Borgo Nuovissimo")]}, season="2026/27", registry_path=path, mapping_path=mapping)
    assert back.competitions["DA"].added == ("Borgo Nuovo",) and back.competitions["DA"].removed == ("Altro",)


def test_sync_never_matches_by_resemblance(tmp_path: Path) -> None:
    """'Trestina' is not 'Triestina' and 'Juventus' is not 'Juventus Next Gen': without an exact or alias hit a club is new."""
    path = write_dictionary(tmp_path / "teams_2026_27.json", {"DF": {"Triestina": []}, "A": {"Juventus Next Gen": []}})
    report = fs.apply_registry_sync(
        {"DF": [fs.TeamInfo(1, "Sporting Trestina", "Trestina")], "A": [fs.TeamInfo(2, "Juventus", "Juventus")]},
        season="2026/27", registry_path=path, mapping_path=tmp_path / "map.json",
    )  # fmt: skip
    assert report.competitions["DF"].added == ("Trestina",) and report.competitions["DF"].removed == ("Triestina",)
    assert report.competitions["A"].added == ("Juventus",) and report.competitions["A"].removed == ("Juventus Next Gen",)


def test_collision_naming_and_partial_sync_never_moves_a_club_it_cannot_see(tmp_path: Path) -> None:
    gironi = {"SB": {"Palermo": ["Palermo FC"]}, "DI": {}, "DA": {}}
    # a full view (Serie B listed too): Palermo stays in Serie B, the Serie D club is named after the collision rule
    path = write_dictionary(tmp_path / "teams_2026_27.json", gironi)
    full = fs.apply_registry_sync(
        {"SB": [fs.TeamInfo(1, "Palermo FC", "Palermo")], "DI": [fs.TeamInfo(2, "AC Palermo", "AC Palermo")]},
        season="2026/27", registry_path=path, mapping_path=tmp_path / "map.json",
    )  # fmt: skip
    assert full.competitions["DI"].added == ("AC Palermo Serie D",) and full.competitions["SB"].total == 1 and not full.moved
    assert full.skipped_aliases == {"AC Palermo Serie D": ("AC Palermo",)} and full.conflicts == {}
    # FootyStats `name` free: it wins over the suffixed names
    named = fs.apply_registry_sync(
        {"SB": [fs.TeamInfo(1, "Palermo FC", "Palermo")], "DA": [fs.TeamInfo(3, "ACD Palermo Dilettanti", "AC Palermo")]},
        season="2026/27", registry_path=write_dictionary(tmp_path / "named.json", gironi), mapping_path=tmp_path / "map-named.json",
    )  # fmt: skip
    assert named.competitions["DA"].added == ("ACD Palermo Dilettanti",)
    # two Serie D clubs colliding with a Serie B club and with each other: family suffix first, then the girone label
    clash = fs.apply_registry_sync(
        {"SB": [fs.TeamInfo(3, "Sporting", "Sporting")], "DA": [fs.TeamInfo(4, "Sporting", "Sporting")], "DI": [fs.TeamInfo(5, "Sporting", "Sporting")]},
        season="2026/27", registry_path=write_dictionary(tmp_path / "clash.json", {"SB": {"Sporting": []}, "DA": {}, "DI": {}}), mapping_path=tmp_path / "map-clash.json",
    )  # fmt: skip
    assert clash.competitions["SB"].n_changes == 0 and clash.competitions["DA"].added == ("Sporting Serie D",)
    assert clash.competitions["DI"].added == ("Sporting Serie D Girone I",)
    # the collision name identifies the club on the next run, even under a fresh id (rule 3)
    rerun = fs.apply_registry_sync(
        {"DA": [fs.TeamInfo(40, "Sporting", "Sporting")]}, season="2026/27", registry_path=tmp_path / "clash.json", mapping_path=tmp_path / "map-fresh.json"
    )
    assert rerun.n_changes == 0 and rerun.assignments["DA"] == {40: "Sporting Serie D"}
    # a partial view (Serie D only, fresh mapping): the move out of Serie B is refused, the club is added as new
    partial_path = write_dictionary(tmp_path / "partial.json", gironi)
    partial = fs.apply_registry_sync(
        {"DI": [fs.TeamInfo(2, "AC Palermo", "AC Palermo")]}, season="2026/27", registry_path=partial_path, mapping_path=tmp_path / "map-partial.json"
    )
    assert partial.competitions["DI"].added == ("AC Palermo Serie D",) and not partial.moved
    assert TeamRegistry.from_json(partial_path).girone_of("Palermo") is Girone.SB
    assert "AC Palermo" in partial.conflicts and "non sincronizzata" in partial.conflicts["AC Palermo"] and "Da controllare: AC Palermo" in partial.describe()
    # ... and once the club is in the dictionary, the next partial run identifies it by the collision name
    again = fs.apply_registry_sync(
        {"DI": [fs.TeamInfo(99, "AC Palermo", "AC Palermo")]}, season="2026/27", registry_path=partial_path, mapping_path=tmp_path / "map-other.json"
    )
    assert again.n_changes == 0 and again.assignments["DI"] == {99: "AC Palermo Serie D"}


def test_spellings_become_aliases_only_when_no_other_club_answers_to_them(tmp_path: Path) -> None:
    path = write_dictionary(tmp_path / "teams_2026_27.json", {"A": {"Alfa": ["Vecchio Alias"], "Beta": ["I Leoni"]}})
    report = fs.apply_registry_sync(
        {"A": [fs.TeamInfo(1, "Alfa Sporting", "Alfa", "Alfa City", ("I Leoni", "Alfa Nick")), fs.TeamInfo(2, "Beta", "Beta")]},
        season="2026/27", registry_path=path, mapping_path=tmp_path / "map.json",
    )  # fmt: skip
    saved = json.loads(path.read_text(encoding="utf-8"))["gironi"]["A"]
    assert saved["Alfa"] == ["Vecchio Alias", "Alfa Sporting", "Alfa City", "Alfa Nick"] and saved["Beta"] == ["I Leoni"]
    assert report.new_aliases == {"Alfa": ("Alfa Sporting", "Alfa City", "Alfa Nick")} and report.skipped_aliases == {"Alfa": ("I Leoni",)}
    assert report.changed and report.n_changes == 0 and "composizione invariata, alias aggiornati" in report.describe()


def test_skips_dry_run_atomic_write_and_refusals(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    path = write_dictionary(tmp_path / "teams_2026_27.json", {"A": {"Alfa": []}, "B": {"Beta": []}})
    before = path.read_bytes()
    # an empty listing never empties a competition; a code without a listing or unknown to the schema is skipped
    report = fs.apply_registry_sync({"A": [], "ZZ": [fs.TeamInfo(1, "X", "X")]}, season="2026/27", registry_path=path, mapping_path=tmp_path / "map.json", competitions=["A", "B", "ZZ"])
    assert report.competitions == {} and not report.changed and path.read_bytes() == before
    assert set(report.skipped) == {"A", "B", "ZZ"} and "lasciata invariata" in report.skipped["A"] and "nessuna lista" in report.skipped["B"]
    assert "Serie C Girone A: non sincronizzata" in report.describe()
    # raw /league-teams entries (data already downloaded) work as they are; a dry run writes nothing
    dry = fs.apply_registry_sync({"A": [{"id": 5, "name": "Gamma", "cleanName": "Gamma"}]}, season="2026/27", registry_path=path, mapping_path=tmp_path / "map.json", write=False)
    assert dry.changed and not dry.written and dry.competitions["A"].added == ("Gamma",) and path.read_bytes() == before and not (tmp_path / "map.json").exists()
    # a failure while replacing the file leaves the old file and no temporary file behind
    def boom(src: str, dst: str) -> None:
        raise OSError("disk full")

    with monkeypatch.context() as patched:
        patched.setattr(fs.os, "replace", boom)
        with pytest.raises(OSError, match="disk full"):
            fs.apply_registry_sync({"A": [fs.TeamInfo(5, "Gamma", "Gamma")]}, season="2026/27", registry_path=path, mapping_path=tmp_path / "map.json")
    assert path.read_bytes() == before and sorted(p.name for p in tmp_path.iterdir()) == ["teams_2026_27.json"]
    # another season's file is refused, and so is a dictionary that would not load
    with pytest.raises(ValueError, match="holds season '2026/27'"):
        fs.apply_registry_sync({"A": [fs.TeamInfo(5, "Gamma", "Gamma")]}, season="2027/28", registry_path=path, mapping_path=tmp_path / "map.json")
    (tmp_path / "broken.json").write_text("[]", encoding="utf-8")
    with pytest.raises(ValueError, match="not a team dictionary"):
        fs.apply_registry_sync({"A": [fs.TeamInfo(5, "Gamma", "Gamma")]}, season="2026/27", registry_path=tmp_path / "broken.json", mapping_path=tmp_path / "map.json")
    # a missing file is created from the listings
    created = fs.apply_registry_sync({"SB": [fs.TeamInfo(6, "Zeta", "Zeta")]}, season="2027/28", registry_path=tmp_path / "teams_2027_28.json", mapping_path=tmp_path / "map.json")
    assert created.created and created.written and TeamRegistry.from_json(tmp_path / "teams_2027_28.json").teams("SB") == ("Zeta",)
    assert json.loads((tmp_path / "teams_2027_28.json").read_text(encoding="utf-8"))["competition"] == "Serie B" and "File creato." in created.describe()


def test_mapping_lives_under_data_and_the_old_location_is_migrated_once(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    assert fs.DEFAULT_MAPPING_PATH == DATA_DIR / "footystats_teams.json" and fs.LEGACY_MAPPING_PATH == CONFIG_DIR / "footystats_teams.json"
    new, legacy = tmp_path / "data" / "footystats_teams.json", tmp_path / "config" / "footystats_teams.json"
    legacy.parent.mkdir()
    legacy.write_text(json.dumps({"teams": {"42": {"canonical": "Lecco", "clean_name": "Lecco", "competition": "A", "seasons": ["2025/26"], "method": "exact"}}}), encoding="utf-8")
    monkeypatch.setattr(fs, "DEFAULT_MAPPING_PATH", new)
    monkeypatch.setattr(fs, "LEGACY_MAPPING_PATH", legacy)
    assert fs.load_team_mapping(new)[42]["canonical"] == "Lecco", "the default path falls back to the old file while it does not exist"
    assert fs.load_team_mapping(tmp_path / "elsewhere.json") == {}, "any other path never reads the old file"
    mapper = fs.FootyStatsTeamMapper(TeamRegistry.from_teams("2026/27", {"A": ("Lecco", "Trento")}), mapping_path=new)
    assert mapper.persisted[42]["canonical"] == "Lecco"
    mapper.save()
    assert new.exists() and json.loads(new.read_text(encoding="utf-8"))["teams"]["42"]["canonical"] == "Lecco"
    assert "git-ignored" in new.read_text(encoding="utf-8")


def test_mapper_trusts_sync_entries_over_spelling_hits_in_every_season(tmp_path: Path) -> None:
    """'AC Palermo' (Serie D) normalises like 'Palermo' (Serie B): only the id assigned by the sync tells them apart."""
    registry = TeamRegistry.from_mapping("2026/27", {"SB": {"Palermo": []}, "DI": {"AC Palermo Serie D": []}})
    mapping = tmp_path / "map.json"
    mapping.write_text(
        json.dumps({"teams": {"2": {"canonical": "AC Palermo Serie D", "clean_name": "AC Palermo", "competition": "DI", "seasons": ["2026/27"], "method": fs.SYNC_METHOD}}}),
        encoding="utf-8",
    )
    mapper = fs.FootyStatsTeamMapper(registry, TeamMatcher(registry), mapping_path=mapping)
    ac = fs.TeamInfo(2, "AC Palermo", "AC Palermo")
    resolved = mapper.resolve(ac, "DI", "2025/26", current=False)
    assert (resolved.canonical, resolved.method, resolved.current) == ("AC Palermo Serie D", fs.SYNC_METHOD, True)
    assert mapper.resolve(fs.TeamInfo(1, "Palermo FC", "Palermo"), "SB", "2025/26", current=False).canonical == "Palermo"
    mapper.save()
    saved = json.loads(mapping.read_text(encoding="utf-8"))["teams"]["2"]
    assert saved["method"] == fs.SYNC_METHOD and saved["seasons"] == ["2026/27", "2025/26"] and saved["competition"] == "DI"
    # without the sync entry the spelling would have won (the trap the id avoids)
    plain = fs.FootyStatsTeamMapper(registry, TeamMatcher(registry), mapping_path=tmp_path / "none.json")
    assert plain.resolve(ac, "DI", "2025/26", current=False).canonical == "Palermo"


# ============================================================================ the async entry point and the harvest
def _envelope(rows: list[dict[str, Any]]) -> httpx.Response:
    return httpx.Response(200, json={"success": True, "pager": {"current_page": 1, "max_page": 1}, "data": rows})


def footystats_router(chosen: tuple[str, ...] = ("A", "SB", "DI")) -> Callable[[httpx.Request], httpx.Response]:
    """A mock account holding ``chosen``; ``/league-teams`` of 2026/27 serves the real lists, matches: two per competition."""
    season_ids = {code: Girone(code).competition.season_ids for code in FOOTYSTATS_2026_27}
    by_id = {sid: code for code, ids in season_ids.items() for label, sid in ids.items() if label == "2026/27"}

    def router(request: httpx.Request) -> httpx.Response:
        assert request.url.host == HOST and request.url.params.get("key") == KEY
        if request.url.path == "/league-list":
            return _envelope([
                {"name": Girone(code).footystats_name, "country": "Italy", "season": [{"id": season_ids[code]["2026/27"], "year": 20262027}]}
                for code in chosen
            ])  # fmt: skip
        code = by_id.get(int(request.url.params.get("season_id", "0")))
        if code is None or code not in chosen:
            return httpx.Response(200, json={"success": False, "message": "League is not chosen by the user."})
        names = FOOTYSTATS_2026_27[code]
        if request.url.path == "/league-teams":
            return _envelope([{"id": TEAM_IDS[(code, n)], "name": n, "cleanName": n} for n in names])
        kickoff = int(datetime(2026, 11, 1, 14, 0, tzinfo=timezone.utc).timestamp())
        rows = [
            {"id": TEAM_IDS[(code, names[0])] * 10, "status": "complete", "date_unix": kickoff, "game_week": 10, "homeID": TEAM_IDS[(code, names[0])], "awayID": TEAM_IDS[(code, names[1])], "home_name": names[0], "away_name": names[1], "homeGoalCount": 1, "awayGoalCount": 0},
            {"id": TEAM_IDS[(code, names[2])] * 10, "status": "incomplete", "date_unix": kickoff + 14 * 86400, "game_week": 12, "homeID": TEAM_IDS[(code, names[2])], "awayID": TEAM_IDS[(code, names[0])], "home_name": names[2], "away_name": names[0], "homeGoalCount": 0, "awayGoalCount": 0},
        ]  # fmt: skip
        return _envelope(rows)

    return router


def make_client(router: Callable[[httpx.Request], httpx.Response]) -> fs.FootyStatsClient:
    http = HttpConfig(timeout_seconds=5.0, max_retries=1, backoff_base_seconds=0.1, backoff_max_seconds=0.2, max_concurrency=2, min_delay_seconds=0.0)
    return fs.FootyStatsClient(api_key=KEY, endpoints=load_endpoints().footystats, http=http, transport=httpx.MockTransport(router), rng=random.Random(1))


@pytest.mark.asyncio
async def test_sync_registry_fetches_the_listings_with_a_client_or_takes_them_as_data(config_copy: Path, tmp_path: Path) -> None:
    path = config_copy / "teams_2026_27.json"
    async with make_client(footystats_router(chosen=("SB", "DI"))) as client:
        report = await fs.sync_registry(client, season="2026_27", registry_path=path, mapping_path=tmp_path / "map.json", competitions=["SB", "DI", "DA"])
    assert list(report.competitions) == ["SB", "DI"] and report.n_changes == 0 and not report.changed
    assert "non scelta" in report.skipped["DA"] and "Serie D Girone A: non sincronizzata (lega 'Italy Serie D Group A' non scelta" in report.describe()
    assert report.assignments["DI"][TEAM_IDS[("DI", "AC Palermo")]] == "AC Palermo Serie D"
    assert KEY not in report.describe() and KEY not in (tmp_path / "map.json").read_text(encoding="utf-8")
    data = await fs.sync_registry({"SB": real_listings()["SB"]}, season="2026/27", registry_path=path, mapping_path=tmp_path / "map.json")
    assert list(data.competitions) == ["SB"] and data.n_changes == 0


@pytest.mark.asyncio
async def test_harvest_runs_the_sync_first_and_names_every_season_by_id(config_copy: Path, tmp_path: Path) -> None:
    registry = TeamRegistry.from_json(config_copy / "teams_2026_27.json")
    matcher = TeamMatcher(Harvester.registry_with_history(registry, config_copy))
    mapper = fs.FootyStatsTeamMapper(registry, matcher, mapping_path=tmp_path / "map.json")
    async with make_client(footystats_router()) as client:
        provider = fs.FootyStatsProvider(client, mapper, now=lambda: NOW, config_dir=config_copy)
        result = await provider.harvest(["A", "SB", "DI"], seasons=1, with_odds=False)
    assert result.registry_sync is not None and not result.registry_sync.changed and result.registry_path is None
    di = result.matches.loc[result.matches["girone"] == "DI"]
    assert di["home_team"].tolist() == ["AC Palermo Serie D"], "the Serie D club is never mistaken for Serie B's Palermo"
    a = result.matches.loc[result.matches["girone"] == "A"]
    assert a["home_team"].tolist() == ["Union Brescia"] and a["away_team"].tolist() == ["Arzignano Valchiampo"]
    assert {f.away_team for f in result.fixtures if f.girone is Girone.DI} == {"AC Palermo Serie D"}
    assert result.unmapped_teams == {} and "registry_sync=unchanged" in result.summary()
    mapping = json.loads((tmp_path / "map.json").read_text(encoding="utf-8"))["teams"]
    assert mapping[str(TEAM_IDS[("DI", "AC Palermo")])]["method"] == fs.SYNC_METHOD


@pytest.mark.asyncio
async def test_harvest_never_synchronises_a_registry_that_was_not_read_from_its_file(config_copy: Path, tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    before = (config_copy / "teams_2026_27.json").read_bytes()
    in_memory = TeamRegistry.from_teams("2026/27", {"A": FOOTYSTATS_2026_27["A"][:4]})
    mapper = fs.FootyStatsTeamMapper(in_memory, TeamMatcher(in_memory), mapping_path=tmp_path / "map.json")
    with caplog.at_level(logging.WARNING):
        async with make_client(footystats_router()) as client:
            provider = fs.FootyStatsProvider(client, mapper, now=lambda: NOW, config_dir=config_copy)
            result = await provider.harvest(["A"], seasons=1, with_odds=False)
    assert result.registry_sync is None and result.registry_sync_error and "non è stato letto" in result.registry_sync_error
    assert (config_copy / "teams_2026_27.json").read_bytes() == before and "registry_sync=error" in result.summary()
    assert any(r.getMessage() == "registry sync skipped: the provider's registry was not loaded from its dictionary file" for r in caplog.records)


def test_report_serialisation_and_log_fields(tmp_path: Path) -> None:
    path = write_dictionary(tmp_path / "teams_2026_27.json", {"A": {"Alfa": []}})
    report = fs.apply_registry_sync({"A": [fs.TeamInfo(1, "Beta", "Beta")]}, season="2026/27", registry_path=path, mapping_path=tmp_path / "m.json")
    payload = report.to_dict()
    assert payload["competitions"]["A"] == {"code": "A", "label": "Serie C Girone A", "total": 1, "added": ["Beta"], "removed": ["Alfa"], "moved": [], "renamed": []}
    assert payload["n_changes"] == 2 and payload["changed"] and json.dumps(payload)
    extra = report.log_extra()
    assert extra["added"] == {"A": ["Beta"]} and extra["removed"] == {"A": ["Alfa"]} and extra["file_created"] is False
    assert isinstance(report.competitions["A"], fs.CompetitionSync) and report.competitions["A"].changed and report.competitions["A"].n_changes == 2


# ============================================================================ one club, one competition (AC Palermo / Palermo)
@pytest.mark.asyncio
async def test_harvest_without_the_sync_never_merges_ac_palermo_with_palermo(config_copy: Path, tmp_path: Path) -> None:
    """Regression: with ``sync_registry=False`` (``--no-sync``) the Serie D 'AC Palermo' became Serie B's 'Palermo'."""
    registry = TeamRegistry.from_json(config_copy / "teams_2026_27.json")
    matcher = TeamMatcher(Harvester.registry_with_history(registry, config_copy))
    mapper = fs.FootyStatsTeamMapper(registry, matcher, mapping_path=tmp_path / "map.json")
    async with make_client(footystats_router(chosen=("SB", "DI"))) as client:
        provider = fs.FootyStatsProvider(client, mapper, now=lambda: NOW, config_dir=config_copy)
        result = await provider.harvest(["SB", "DI"], seasons=1, with_odds=False, sync_registry=False)
    assert result.registry_sync is None
    di = result.matches.loc[result.matches["girone"] == "DI"]
    sb = result.matches.loc[result.matches["girone"] == "SB"]
    assert di["home_team"].tolist() == ["AC Palermo Serie D"] and sb["away_team"].tolist() == ["Palermo"]
    di_teams = set(di["home_team"]) | set(di["away_team"]) | {t for f in result.fixtures if f.girone is Girone.DI for t in (f.home_team, f.away_team)}
    sb_teams = set(sb["home_team"]) | set(sb["away_team"]) | {t for f in result.fixtures if f.girone is Girone.SB for t in (f.home_team, f.away_team)}
    assert not di_teams & sb_teams, "no team plays in two leagues"


def test_mapper_refuses_a_current_club_of_another_competition_in_every_lookup(tmp_path: Path) -> None:
    ac = fs.TeamInfo(2, "AC Palermo", "AC Palermo")
    # the dictionary knows the Serie D club: it is found under the sync's collision name
    known = TeamRegistry.from_mapping("2026/27", {"SB": {"Palermo": []}, "DI": {"AC Palermo Serie D": [], "Avola": []}})
    mapper = fs.FootyStatsTeamMapper(known, TeamMatcher(known), mapping_path=tmp_path / "a.json")
    resolved = mapper.resolve(ac, "DI", "2026/27")
    assert (resolved.canonical, resolved.method, resolved.current) == ("AC Palermo Serie D", "exact", True) and mapper.unmapped_teams == {}
    # Serie D not in the dictionary yet: the club gets the sync's name, reported unmapped, never 'Palermo'
    bare = TeamRegistry.from_mapping("2026/27", {"SB": {"Palermo": []}, "DI": {}})
    first = fs.FootyStatsTeamMapper(bare, TeamMatcher(bare), mapping_path=tmp_path / "b.json")
    assert first.resolve(ac, "DI", "2026/27").canonical == "AC Palermo Serie D" and first.unmapped_teams == {"AC Palermo Serie D": 1}
    # a mapping file left by an earlier (buggy) run pointing the id at Serie B's club is not trusted either
    dirty = tmp_path / "dirty.json"
    dirty.write_text(json.dumps({"teams": {"2": {"canonical": "Palermo", "clean_name": "AC Palermo", "competition": "DI", "seasons": ["2026/27"], "method": "exact"}}}), encoding="utf-8")
    again = fs.FootyStatsTeamMapper(known, TeamMatcher(known), mapping_path=dirty)
    assert again.resolve(ac, "DI", "2026/27").canonical == "AC Palermo Serie D"
    again.save()
    assert json.loads(dirty.read_text(encoding="utf-8"))["teams"]["2"]["canonical"] == "AC Palermo Serie D"
    # Serie B's own club still resolves to itself, and a canonical another id owns by sync is refused in past seasons too
    assert again.resolve(fs.TeamInfo(1, "Palermo FC", "Palermo"), "SB", "2026/27").canonical == "Palermo"
    owned = tmp_path / "owned.json"
    owned.write_text(json.dumps({"teams": {"1": {"canonical": "Palermo", "clean_name": "Palermo", "competition": "SB", "seasons": ["2026/27"], "method": fs.SYNC_METHOD}}}), encoding="utf-8")
    past = fs.FootyStatsTeamMapper(known, TeamMatcher(known), mapping_path=owned)
    assert past.resolve(ac, "DI", "2025/26", current=False).canonical == "AC Palermo Serie D"


def test_init_registry_of_a_new_season_keeps_both_palermo_clubs_and_promotions(config_copy: Path) -> None:
    """Regression: ``--init-registry --season 2027_28`` dropped the Serie D club ('conflict Palermo ... left unchanged')."""
    registry, matcher = fs._load_perimeter("2027_28", config_copy, for_init=True)
    assert registry.season_label == "2026/27"  # the newest dictionary supplies the names
    teams = {
        "SB": [fs.TeamInfo(1, "Palermo FC", "Palermo"), fs.TeamInfo(3, "Union Brescia", "Union Brescia")],  # promoted from girone A
        "DI": [fs.TeamInfo(2, "AC Palermo", "AC Palermo"), fs.TeamInfo(4, "Avola", "Avola")],
    }
    target = config_copy / "teams_2027_28.json"
    diff = fs.write_registry_update(target, "2027_28", teams, matcher=matcher)
    saved = json.loads(target.read_text(encoding="utf-8"))["gironi"]
    assert set(saved["SB"]) == {"Palermo", "Union Brescia"} and set(saved["DI"]) == {"AC Palermo Serie D", "Avola"}
    assert "AC Palermo" in diff.conflicts and "'AC Palermo Serie D', not merged" in diff.conflicts["AC Palermo"] and "Palermo" not in diff.conflicts
    assert "conflict AC Palermo" in diff.describe()
    loaded = TeamRegistry.from_json(target)
    assert loaded.girone_of("AC Palermo Serie D") is Girone.DI and loaded.girone_of("Palermo") is Girone.SB
