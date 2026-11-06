# Serie-C-e-Serie-D-
Tool di analisi statistiche su partite della Serie C e Serie D italiane di calcio. focus su risultati, gol, Serie storiche, tattiche e notizie/aggiornamenti


# Calcio Quant Lab: Serie C (e Serie D) Analytics Engine

[![Python](https://img.shields.io/badge/Python-3.12%2B-blue.svg)](https://www.python.org/)
[![Architecture](https://img.shields.io/badge/Architecture-Modular%20Pipeline-orange.svg)]()

Infrastruttura software per l'acquisizione automatizzata di telemetria calcistica, modellazione statistica bivariata e individuazione sistematica di disallineamenti di prezzo (+EV) nei mercati del betting sportivo per le leghe minori italiane.

---

## 📌 Visione e Obiettivi

Il progetto supera l'analisi euristica e le fallacie di regressione empiriche implementando:
1. **Automazione E2E dell'Ingestion Dati:** Scraping asincrono e reverse-engineering di payload JSON da portali sportivi (SofaScore, FotMob) per estrarre tiri, angoli, tiri nello specchio e $xG$.
2. **Motore Quantitativo Bivariato:** Risoluzione del modello Dixon-Coles con correzione per punteggi bassi ($\rho$), stima dei parametri di forza latente ($\alpha, \beta, \gamma$) isolati per cluster/girone, e time-decay esponenziale.
3. **Market Mispricing Detector:** Rimozione automatica dell'aggio del bookmaker (de-vigging), calcolo dell'Expected Value e sizing del bankroll via Criterio di Kelly Frazionario.

---

## 🏗️ Architettura del Sistema

```
Serie-C-e-Serie-D-/
├── main.py                        # Pipeline end-to-end dimostrativa (console)
├── .env.example                   # Modello del file .env con la chiave FootyStats (il .env vero non si committa)
├── requirements.txt               # Dipendenze pinnate
├── pyproject.toml / pytest.ini    # Configurazione ruff e pytest (il progetto si esegue dalla root, non si installa)
├── config/
│   ├── competitions.toml          # Le competizioni del perimetro (Serie C A/B/C + Serie B), in ordine di visualizzazione
│   ├── teams_2026_27.json         # Dizionario canonico squadre + alias per competizione (A/B/C/SB) della stagione corrente = PERIMETRO (default)
│   ├── teams_2025_26.json         # Stagione precedente: solo fonte di alias storici (squadre uscite dal perimetro, movimenti B<->C)
│   ├── footystats_teams.json      # Mappa id squadra FootyStats -> nome canonico (generata e aggiornata dall'harvester)
│   └── endpoints.toml             # Endpoint FootyStats (API ufficiale) e SofaScore/FotMob, retry, cache, rate-limit
├── data/
│   ├── raw/                       # Cache JSON grezzi (chiave = sha1 URL, TTL)
│   ├── processed/                 # matches.csv / fixtures.csv normalizzati
│   └── odds/                      # Snapshot quote bookmaker (CSV)
├── src/
│   ├── common/                    # Contratti condivisi
│   │   ├── competitions.py        # Competition + load_competitions(): legge config/competitions.toml (fallback integrato)
│   │   ├── schemas.py             # Girone (enum DINAMICO dalle competizioni), MatchRecord, Fixture, OddsQuote, MatchPrediction, Market
│   │   ├── teams.py               # TeamRegistry (qualsiasi numero di competizioni) + normalizzazione nomi
│   │   ├── logging_config.py      # Logging strutturato (JSON lines)
│   │   └── paths.py
│   ├── ingestion/                 # MODULO 1
│   │   ├── footystats.py          # Provider FootyStats (API ufficiale): storico 3 stagioni, fixture, quote pre-match, CLI
│   │   ├── harvester.py           # Client httpx asincrono, cache, backoff, provider SofaScore/FotMob, facade con fallback
│   │   └── team_matcher.py        # Fuzzy matching nomi squadre -> canonico
│   ├── engine/                    # MODULI 2-3
│   │   ├── model.py               # DixonColesEngine (MLE SLSQP/L-BFGS-B, time-decay, rho, gamma per girone)
│   │   └── value.py               # De-vigging, EV, Kelly frazionario, REGRESSION_TRAP
│   ├── dashboard/                 # MODULO 4
│   │   ├── state.py               # Data layer (pure, testabile)
│   │   └── app.py                 # Interfaccia Streamlit + heatmap Plotly + export CSV
│   └── simulation/
│       └── generator.py           # Lega sintetica con parametri noti (test di recupero, demo offline)
└── tests/                         # Unit test, validazione numerica, test dashboard (AppTest)
```

## 🖱️ Avvio con un doppio clic (senza terminale)

1. Installa Python 3.12 o superiore da <https://www.python.org/downloads/> (su Windows spunta "Add python.exe to PATH").
2. Scarica il progetto: su GitHub pulsante **Code → Download ZIP**, poi estrai la cartella.
3. Fai doppio clic su `avvia.bat` (Windows) o `avvia.command` (macOS; la prima volta: tasto destro → Apri).

Lo script crea l'ambiente Python, installa le dipendenze, raccoglie i dati reali delle ultime tre stagioni
e apre la dashboard nel browser. Se trova la chiave FootyStats (variabile `FOOTYSTATS_API_KEY` oppure un file
`.env` accanto allo script, copia di `.env.example`) usa l'API ufficiale FootyStats (Serie C A/B/C + Serie B,
quote pre-match incluse: pochi secondi); altrimenti SofaScore (solo da connessione residenziale, la prima volta
serve qualche decina di minuti, poi è veloce grazie alla cache).
Per aprire solo la dashboard senza aggiornare i dati: `avvia.bat --solo-dashboard`.
Le quote dei bookmaker vanno in `data/odds/*.csv` (formato più sotto).
Per pubblicare i CSV raccolti sul repository (così li vede anche la dashboard in cloud):
`pubblica_dati.bat` / `pubblica_dati.command` (serve git e una copia clonata, non lo ZIP).

## ⏱️ Aggiornamento automatico dopo ogni giornata (GitHub Actions)

Il workflow `.github/workflows/aggiorna-dati.yml` gira lunedì e giovedì alle 06:30 UTC (o a mano da
*Actions → Aggiorna dati Serie C e Serie B → Run workflow*), raccoglie le ultime tre stagioni e committa
`data/processed/matches.csv`, `fixtures.csv`, `data/odds/footystats.csv` e `config/footystats_teams.json`
solo se sono cambiati. Il riepilogo del job mostra le partite per competizione e stagione.

- **Con FootyStats (consigliato, funziona sui runner GitHub)**: salva la chiave API come *segreto* di repository
  `FOOTYSTATS_API_KEY` (Settings → Secrets and variables → Actions → New repository secret). Il job esegue
  `python -m src.ingestion.footystats --competitions A B C SB --seasons 3 --odds-out data/odds/footystats.csv`;
  la chiave non compare mai nei log né nei file. Vedi la sezione *Dati reali con FootyStats*.
- **Senza segreto** il job usa SofaScore. **Sui runner GitHub** SofaScore risponde spesso `403 Forbidden` agli IP
  dei datacenter: in quel caso il job fallisce e il riepilogo lo spiega. Vale la pena tenerlo attivo (a volte
  passa), ma non contarci.
- **Dal tuo PC, in automatico**: registra un *self-hosted runner* (Settings → Actions → Runners → New self-hosted
  runner, segui i comandi mostrati) e imposta la variabile di repository `HARVEST_RUNNER = self-hosted`
  (Settings → Secrets and variables → Actions → Variables). Lo stesso job partirà sul tuo PC, con il tuo IP,
  quando è acceso e il runner è in esecuzione. È la via affidabile.
- **A mano**: `avvia.bat` / `avvia.command` raccolgono i dati, poi `pubblica_dati.*` li carica sul repository.

I CSV in `data/processed/` e `data/odds/` sono versionati; la cache grezza `data/raw/` no.

## 🚀 Quick Start

```bash
# Python 3.12+ (richiesto dai pin di numpy/scipy in requirements.txt)
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# 1) demo completa su mini-dataset simulato (20 partite 2026/27 + 2 stagioni storiche sintetiche) con output a console
python main.py

# 2) stagione simulata 2026/27 al calendario di oggi (--as-of per un'altra data, --rounds 30 per 30 giornate fisse)
#    + verifica del recupero dei parametri veri (--history 0 = senza storico)
python main.py --mode full
#    opzioni comuni: --season 2026_27 (perimetro = config/teams_<stagione>.json), --history 0|1|2 (stagioni sintetiche
#    precedenti quella del dizionario: 2024/25-2025/26 per il 2026/27); il mini è un'istantanea fissa di inizio stagione

# 3) raccolta dati reali: FootyStats (API ufficiale, chiave in .env o FOOTYSTATS_API_KEY; Serie C A/B/C + Serie B,
#    ultime 3 stagioni, quote pre-match in data/odds/footystats.csv) -> data/processed/
python -m src.ingestion.footystats --competitions A B C SB --seasons 3
#    oppure SofaScore (senza chiave, solo dal proprio PC: vedi "Ingestion") -> data/processed/
python -m src.ingestion.harvester --gironi A B C --seasons 3 --out data/processed/matches.csv

# 4) pipeline sui dati reali con quote bookmaker in data/odds/*.csv
python main.py --processed data/processed/matches.csv --fixtures data/processed/fixtures.csv --odds data/odds

# 5) dashboard interattiva (usa data/processed se presente, altrimenti la lega simulata 2026/27 al calendario di oggi)
streamlit run src/dashboard/app.py
#    variabili utili: SERIEC_FORCE_SIMULATE=1  SERIEC_SEASON=2026_27  SERIEC_HISTORY=2  SERIEC_AS_OF=2026-11-06
#                     SERIEC_SIM_ROUNDS=15 (giornate fisse invece del calendario)  SERIEC_SEED=7

# test
python -m pytest -q
```

Formato CSV quote (`data/odds/*.csv`, è anche quello che FootyStats scrive in `data/odds/footystats.csv`): `match_id,bookmaker,market,selection,odds`, una riga per esito,
con `market` in `1X2 | OU1.5 | OU2.5 | OU3.5 | GG/NG` e `selection` in `1,X,2 | Over 2.5,Under 2.5 | GG,NG` ecc.

## 🏆 Competizioni (`config/competitions.toml`)

Le competizioni del perimetro **non sono cablate nel codice**: `config/competitions.toml` è un array di tabelle
`[[competition]]`, una per competizione, nell'ordine in cui compaiono nelle tabelle e nei selettori:

```toml
[[competition]]
code = "SB"                          # identificatore stabile: chiave di teams_<stagione>.json, colonna girone del CSV, valore dell'enum
label = "Serie B"                    # nome pubblico completo
short = "Serie B"                    # etichetta compatta (tabelle, selettore della dashboard)
country = "Italy"
footystats_name = "Italy Serie B"    # campo `name` di /league-list su FootyStats
sofascore_tournament_id = 53         # id SofaScore (legacy); 0 se sconosciuto
tier = 2                             # livello della piramide (2 = Serie B, 3 = Serie C)
family = "Serie B"                   # lega di appartenenza: i tre gironi condividono "Serie C"
```

Da questo file nasce, **all'import**, l'enum `src.common.schemas.Girone` (`Girone.A`, `Girone.B`, `Girone.C`,
`Girone.SB`; `Girone.parse()` accetta `'A'`, `'Girone A'`, `'SB'`, `'Serie B'`, `'serie_b'`, `'Italy Serie B'`,
senza distinzione di maiuscole; `'B'` è sempre il girone B di Serie C) con le proprietà `.label`, `.short`,
`.family`, `.tier`, `.competition`. Se il file manca il codice usa le definizioni integrate (A/B/C/SB), così gli
import non falliscono mai. Il **perimetro** è il dizionario della stagione corrente (`config/teams_2026_27.json`,
chiavi = codici delle competizioni): `Serie C 2026/27 Gironi A/B/C + Serie B (80 squadre)` — la frase è derivata
dalla configurazione (`TeamRegistry.perimeter_label`) e compare identica nel banner della console e della
dashboard. Il modello stima un fattore campo e un vincolo Σα = 0 per competizione; simulatore, console e
dashboard iterano l'enum, mai una lista A/B/C.

**Per aggiungere un campionato** (es. Serie D girone A, o un'altra nazione):

1. aggiungi una tabella `[[competition]]` a `config/competitions.toml` con un `code` nuovo (maiuscole/cifre, es. `DA`);
2. aggiungi le squadre di quella competizione, con gli alias, sotto la stessa chiave in `config/teams_<stagione>.json`
   (una squadra ha **un solo nome canonico** anche se cambia divisione: `Pescara` in Serie B 2025/26 e nel girone B
   2026/27; il dizionario della stagione corrente decide la sua competizione, le stagioni passate restano storico);
3. inserisci il `footystats_name` esatto restituito da `/league-list` e aggiungi la lega al tuo account su
   footystats.org (l'API mostra solo le leghe scelte), poi raccogli i dati con l'harvester FootyStats.

Nient'altro va toccato: enum, dizionario, motore, selettori e banner seguono il file. Per controllare una
composizione (es. la Serie B 2026/27) usa `python -m src.ingestion.footystats --init-registry`: **aggiunge** al
dizionario le squadre e gli alias mancanti secondo `/league-teams` di FootyStats e stampa il diff, comprese le squadre
del file che FootyStats **non elenca più** (*no longer listed by FootyStats*). Il comando non cancella mai squadre o
alias: un club di troppo (retrocesso, escluso, inserito nel girone sbagliato) va rimosso a mano dal file.

## 🔑 Dati reali con FootyStats (API ufficiale, consigliata)

FootyStats (<https://footystats.org/api/>) è un'API documentata e autenticata con chiave: risponde anche dai
datacenter (GitHub Actions, cloud), copre i tre gironi di Serie C e la Serie B con risultati, tiri, tiri nello
specchio, angoli, xG (dove disponibile) e **quote pre-match** (1X2, Over/Under 1.5-2.5-3.5, Gol/NoGol) che
alimentano il motore di valore. È la sorgente di default appena la chiave è disponibile; SofaScore resta il
fallback (sezione successiva).

1. **Chiave**: dal tuo account su footystats.org → *API*. Non committarla mai: va in una variabile d'ambiente
   `FOOTYSTATS_API_KEY` oppure in un file `.env` nella root del repository (copia `.env.example`;
   `.env` è in `.gitignore`), oppure passata con `--key`. Il codice la maschera ovunque (`***`): non finisce
   in log, cache (`data/raw/`), messaggi d'errore né nei file prodotti.
2. **Leghe dell'account**: su footystats.org aggiungi al tuo account (*Choose leagues*) **Italy Serie C Group A**,
   **Italy Serie C Group B**, **Italy Serie C Group C** e **Italy Serie B**. L'API elenca solo le leghe scelte
   (`/league-list?chosen_leagues_only=true`): se una manca l'harvester si ferma elencando i nomi da aggiungere.
   Verifica con `python -m src.ingestion.footystats --list-leagues`.
3. **Raccolta**:

```bash
# ultime 3 stagioni di Serie C (A/B/C) e Serie B in UN solo matches.csv + fixture della stagione corrente + quote
python -m src.ingestion.footystats --competitions A B C SB --seasons 3 --odds-out data/odds/footystats.csv
# solo Serie B, 2 stagioni, senza quote
python -m src.ingestion.footystats --competitions SB --seasons 2 --no-odds
# leghe e season id del tuo account
python -m src.ingestion.footystats --list-leagues
# crea/aggiorna config/teams_2027_28.json dalle squadre FootyStats: aggiunge squadre/alias mancanti e stampa il diff
# (squadre e alias esistenti non vengono mai rimossi; quelle non più elencate da FootyStats sono segnalate, da togliere a mano)
python -m src.ingestion.footystats --init-registry --season 2027_28 --competitions A B C SB
# poi la pipeline / la dashboard leggono data/processed e data/odds
python main.py --processed data/processed/matches.csv --fixtures data/processed/fixtures.csv --odds data/odds
```

- Le competizioni sono i codici di `config/competitions.toml` (`A`, `B`, `C`, `SB`); il nome FootyStats di
  ciascuna (`footystats_name`, es. `Italy Serie B`) è lì, con un fallback in `[footystats.league_names]` di
  `config/endpoints.toml`. Per aggiungere un campionato: una tabella `[[competition]]`, le squadre nel
  dizionario di stagione (`--init-registry` le aggiunge con gli alias FootyStats), la lega nell'account FootyStats
  (le istruzioni stampate quando manca la chiave elencano i `footystats_name` di tutte le competizioni configurate).
- Storico: per ogni competizione vengono scaricate le ultime `--seasons` stagioni elencate dall'API (season id
  risolti a runtime da `/league-list`; `[footystats.season_ids]` è solo un fallback se la lista non risponde).
  Tutte le squadre che vi hanno partecipato restano nello storico, anche quelle poi retrocesse, promosse o
  fallite: mantengono la loro grafia canonica (*historical teams*) e servono a valutare le avversarie, ma non
  entrano in classifica/previsioni. Le partite `suspended`/`canceled` sono scartate; le `incomplete` future della
  stagione corrente diventano `fixtures.csv`, le loro quote `data/odds/footystats.csv`
  (`match_id,bookmaker,market,selection,odds`, bookmaker `FootyStats`).
- Statistiche mancanti: FootyStats usa `0`/`-1` per i dati non raccolti; tiri a 0 per entrambe le squadre o xG a 0
  per entrambe vengono letti come assenti (`NaN`), così il modello non li confonde con zeri veri.
- Nomi squadra: `config/footystats_teams.json` memorizza la mappa *id FootyStats → nome canonico* e viene
  aggiornato a ogni esecuzione (gli id battono il fuzzy matching; un hit esatto/alias del dizionario batte
  entrambi). Correggi lì il campo `canonical` se una squadra è mappata male. Le squadre che nessun dizionario
  conosce mantengono il `cleanName` FootyStats e sono elencate come *unmapped teams*.
- Limiti: circa **1800 richieste/ora** sui piani standard; un run completo (4 competizioni × 3 stagioni) ne usa
  ~25. Il 429 viene ritentato con backoff; la cache in `data/raw/` evita di riscaricare le stagioni concluse.
- Exit code della CLI: `0` ok, `2` chiave mancante (con le istruzioni), `1` raccolta fallita o parziale (il CSV
  esistente non viene sovrascritto senza `--allow-partial`).
- La facade `src.ingestion.harvester.Harvester` prova i provider in ordine (`footystats`, poi `sofascore`
  quando una chiave è disponibile; solo `sofascore` altrimenti): `python -m src.ingestion.harvester --odds-out ...`
  usa quindi FootyStats e ripiega su SofaScore competizione per competizione in caso di errore.

## 📥 Ingestion: raccolta dati reali (SofaScore, fallback)

Il **perimetro** è il dizionario della stagione corrente (`config/teams_2026_27.json`: le 60 squadre dei tre
gironi ufficiali Lega Pro più le 20 della Serie B, 80 in tutto): solo queste squadre vengono classificate, previste
e prezzate. Le stagioni precedenti servono **unicamente come base storica** per stimare meglio la forza delle
squadre correnti. L'harvester SofaScore copre i soli gironi di Serie C; la Serie B arriva da FootyStats.

```bash
# dal proprio PC (connessione residenziale, vedi la nota 403): ultime 3 stagioni (default) in UN solo CSV
python -m src.ingestion.harvester --gironi A B C --seasons 3
# solo la stagione corrente
python -m src.ingestion.harvester --seasons 1
# perimetro esplicito (config/teams_<stagione>.json) oppure una singola stagione SofaScore
python -m src.ingestion.harvester --season 2026_27 --season-id 77031
# poi copiare i CSV dove gira la pipeline / la dashboard
scp data/processed/matches.csv data/processed/fixtures.csv utente@server:/percorso/Serie-C-e-Serie-D-/data/processed/
```

- `--seasons N` raccoglie le ultime N stagioni di ogni girone (la corrente + N−1 storiche, default 3) e scrive
  **un solo** `data/processed/matches.csv` con lo schema fisso a 16 colonne: non esiste una colonna `season`,
  la stagione di una riga è `season_of(date)` (luglio–giugno, es. `2026/27`, in `src/common/schemas.py`).
  `fixtures.csv` contiene solo le partite non ancora giocate della stagione corrente.
- Nomi squadra: il dizionario corrente viene esteso con gli alias dei dizionari precedenti
  (`TeamRegistry.with_historical_aliases`). Una squadra che gioca ancora mantiene il nome corrente qualunque
  fosse il suo girone allora (`Juventus U23` 2025/26 → `Juventus Next Gen`, oggi nel girone A); una squadra
  uscita dalla Serie C mantiene la vecchia grafia canonica (`Ternana Calcio` → `Ternana`) ed è elencata fra
  le *historical teams*: resta nella verosimiglianza come avversaria ma non entra in classifica, previsioni
  e opportunità di valore. Le grafie che nessun dizionario conosce restano grezze (*unmapped teams*).
- La console riporta sempre il perimetro, le partite per stagione, quante partite storiche sono state usate e
  le squadre storiche (`HarvestResult.registry_season`, `seasons`, `n_matches_by_season`, `historical_teams`).
- Una stagione storica che fallisce viene registrata in `failures` (es. `B:sofascore:2024/25`) e il CSV non
  viene sovrascritto senza `--allow-partial`; un harvest vuoto non sovrascrive mai dati esistenti.
- Cache dei JSON grezzi (`data/raw/`): i payload della stagione corrente (risultati e statistiche cambiano
  ancora) scadono dopo `[http] cache_ttl_hours` (6 h, `--cache-ttl-hours`); quelli delle stagioni **concluse**
  (giornate, eventi e statistiche sono immutabili) dopo `[http] history_cache_ttl_hours` (8760 h = 365 giorni,
  `--history-cache-ttl-hours`). Così un run interrotto riprende senza riscaricare ciò che è già riuscito e un
  `--seasons 3` ripetuto a distanza di giorni riscarica solo la stagione corrente (le due stagioni storiche
  costano ~2500 richieste per girone con le statistiche). `--history-cache-ttl-hours 0` forza il riscaricamento
  dello storico; una stagione indicata con `--season-id` è trattata come corrente (TTL breve).
- Errori di avvio distinti: un file `--endpoints` mancante o malformato viene segnalato come tale
  (`endpoints file not found` / `endpoints file invalid`), un dizionario `--season` inesistente come
  `team dictionary not found`; in entrambi i casi exit code 1 senza traceback.

> **Nota 403 (stato novembre 2026).** `api.sofascore.com` risponde HTTP 403 alle richieste provenienti da IP di
> datacenter/cloud (CI, notebook cloud, VPS, container remoti), con qualunque header: l'harvester va eseguito da
> una **connessione residenziale** (il proprio PC) e i file `data/processed/*.csv` vanno poi copiati (scp, commit,
> cartella condivisa) dove girano `main.py --processed ...` o la dashboard. Il 403 non viene ritentato perché
> non è transitorio. FotMob è **sperimentale**: dal 2026 `/api/allLeagues` restituisce una pagina HTML 404 e
> l'API richiede l'header firmato `x-mas` (`--x-mas` / env `FOTMOB_X_MAS`); il provider resta disponibile con
> `--providers sofascore fotmob` ma non è un fallback funzionante senza firma. Dettagli in `config/endpoints.toml`.

## 🧮 Modello

- **Tassi**: λ = exp(α_casa − β_ospite + γ_girone), μ = exp(α_ospite − β_casa); β alto = difesa solida.
- **Dipendenza bassi punteggi**: τ(x,y,λ,μ,ρ) di Dixon–Coles su 0-0, 1-0, 0-1, 1-1.
- **Ponderazione temporale**: w(t) = exp(−ξ·t), t in giorni dalla partita più recente (ξ default 0.0018).
- **Identificabilità**: Σα = 0 per ciascuna competizione (vincolo di uguaglianza SLSQP, o riparametrizzazione con L-BFGS-B);
  un fattore campo γ per competizione (Serie B compresa). Le divisioni non si incontrano nella stessa stagione, quindi
  ogni blocco è identificato dal proprio vincolo e dal proprio livello di gol; la stima **congiunta** fra Serie B e
  Serie C è legata dai *movers*: una squadra promossa o retrocessa conserva un solo nome canonico, il dizionario
  corrente decide il suo gruppo e le sue partite nell'altra divisione restano nella verosimiglianza (Pescara porta in
  girone B la forza mostrata contro avversarie di Serie B). α resta centrato per competizione: confronta le squadre
  dentro la stessa competizione. Se una competizione del dizionario non ha ancora partite nei dati (Serie B non
  raccolta), le sue squadre restano correnti ma, per il solo vincolo, entrano nel gruppo della competizione in cui
  hanno giocato (le promosse non vengono centrate a zero da sole) e nessun γ viene stimato per essa: il fit lo
  segnala e le sue partite in programma sono saltate finché non la si raccoglie.
- **Gradiente analitico** verificato numericamente nei test.
- **Prior gaussiana (stima MAP)** per campioni piccoli: `DixonColesConfig.with_gaussian_prior()` traduce deviazioni
  standard a priori (α/β 0.30, γ 0.25 ± 0.15, ρ 0.10) in penalità ridge; la ridge su β è centrata per girone così il
  livello medio dei gol non viene distorto. `main.py` (`--prior auto|on|off`) la attiva finché il girone meno coperto
  ha in media meno di 15 partite per squadra (con 20 squadre per girone: fino alla 14ª giornata inclusa); una soglia
  sul totale delle partite non basta, perché 100 gare sulle 80 squadre del perimetro sono poco più di 1 a testa e la
  MLE pura porta γ e ρ sui limiti dell'ottimizzatore (nel simulatore succede ancora in 1 fit su 10 a 12 gare per
  squadra, mai da 15). La copertura è contata **per competizione** (ogni girone di Serie C e la Serie B a sé) sulle
  sole partite della stagione corrente fra squadre del perimetro; una competizione del dizionario **senza alcuna
  partita nei dati** (es. Serie B con il solo harvester SofaScore, che non la copre) non conta come copertura zero ma
  viene ignorata dalla regola, e il banner lo dichiara (`Serie B senza partite nei dati (ignorata nel calcolo della
  copertura)`); una competizione con storico ma senza partite della stagione in corso resta a copertura zero.
  Dopo la stima la console segnala i parametri finiti sui limiti dell'ottimizzatore.
- **Mercati**: 1X2, Over/Under 1.5-2.5-3.5, Goal/No Goal dalla matrice congiunta 11×11.
- **Valore**: de-vigging proporzionale (anche power e Shin), EV = P·quota − 1, Kelly frazionario (1/8 o 1/4),
  filtro `REGRESSION_TRAP` (GA ≤ 1 nelle ultime 3 gare ma xGA ≥ 3.5 cumulato), calcolato sulle sole partite
  della stagione corrente.
- **Perimetro e storico (stagioni)**: il modello può essere stimato sulle ultime 3 stagioni (2024/25, 2025/26,
  2026/27: la stagione di una partita è `season_of(date)`, luglio–giugno), ma il *perimetro* è il dizionario della
  stagione corrente passato a `DixonColesEngine.fit(matches, registry=...)`: le sue 80 squadre sono le uniche
  classificate (`ranking()`), previste e prezzate. Le squadre presenti solo nello storico (retrocesse, promosse,
  escluse: Ternana, Rimini, Vicenza, Arezzo, …) restano nella verosimiglianza come avversarie — così le forze delle
  squadre correnti sono stimate contro l'opposizione giusta — ma sono `FittedParams.historical_teams`:
  fuori da classifica, previsioni e pannello valore (`ranking(include_historical=True)` le mostra). Allo stesso
  modo una partita in programma datata in un'altra stagione (un `fixtures.csv` rimasto dalla stagione scorsa) non
  viene mai prevista: `season_of(data)` deve coincidere con la stagione del dizionario, altrimenti la partita è
  elencata fra le saltate («partita di una stagione precedente (2025/26), fuori dal perimetro 2026/27»). Il girone di
  una squadra corrente è quello del dizionario corrente, non quello in cui giocava in passato (Inter U23 A→C,
  Atalanta U23 C→B, Juventus Next Gen B→A, Latina C→B, Carpi B→A); il fattore campo γ di una partita storica resta
  quello del girone in cui fu giocata. Il vincolo Σα = 0 vale per *gruppo di vincolo*: il girone del dizionario per
  le squadre correnti, il girone in cui ha giocato più partite per una squadra storica. La prior gaussiana
  (`--prior auto`) conta la copertura sulle sole partite della stagione corrente fra squadre correnti: tre stagioni
  di storico non rendono più lunga la stagione in corso.

> Il dizionario squadre di default è `config/teams_2026_27.json` (`src.common.teams.DEFAULT_SEASON = "2026_27"`,
> composizione ufficiale Lega Pro dei tre gironi più la Serie B, chiave `SB`); `teams_2025_26.json` resta come fonte di alias storici
> (`TeamRegistry.with_historical_aliases`, `load_registry(with_history=True)`): `Juventus U23` → `Juventus Next Gen`,
> `Ternana Calcio` → `Ternana` (grafia storica, squadra non corrente). Per una nuova stagione basta aggiungere
> `config/teams_<stagione>.json` con la stessa struttura (`season`, `gironi` → squadra → alias), aggiornare
> `DEFAULT_SEASON` oppure passare `--season`/`SERIEC_SEASON`; `list_seasons()` elenca i file disponibili.
> Gli endpoint SofaScore/FotMob non sono documentati e possono cambiare: sono tutti centralizzati in
> `config/endpoints.toml`.

## 🖥️ Dashboard e demo a console

- **Perimetro sempre dichiarato**: badge della sidebar e sottotitolo della pagina riportano
  `Perimetro: Serie C 2026/27 Gironi A/B/C + Serie B (80 squadre) · N partite stagione corrente · M partite storiche
  (2024/25-2025/26)`; la console di `main.py` apre con `Perimetro: Serie C 2026/27 Gironi A/B/C + Serie B (80 squadre) ·
  stagione corrente: N partite · storico: M partite 2024/25-2025/26` e stampa il riepilogo per stagione (partite,
  squadre attuali e storiche). La frase è derivata da `config/competitions.toml` e dal dizionario, mai cablata.
- **Selettore Competizione**: `Tutti` + l'etichetta breve di ogni competizione configurata (`Girone A`, `Girone B`,
  `Girone C`, `Serie B`); filtra classifica, partite e opportunità.
- **Classifica Forza**: solo le 80 squadre correnti, colonna *Competizione* = competizione del dizionario 2026/27
  (`Girone A`, ..., `Serie B`); stesso nome di colonna nel pannello valore e negli alert.
- **Quote**: ogni CSV in `data/odds/` è un bookmaker come gli altri, compreso quello scritto dall'harvester FootyStats
  (`bookmaker = FootyStats`, quote pre-match 1X2 / Over-Under / GG-NG): la colonna *Bookmaker* del pannello valore dice
  da dove viene ogni quota.
- **Match & Heatmap / Opportunità di Valore**: solo partite in programma della stagione corrente fra squadre
  correnti; le partite datate in un'altra stagione (fixtures stantii dell'anno scorso), con una squadra storica o
  sconosciuta allo storico sono elencate come saltate in *Dettagli pipeline* (e il tab Match lo segnala quando non
  resta nulla da mostrare).
- **Fallback simulato** (nessun `data/processed/matches.csv`, o `SERIEC_FORCE_SIMULATE=1`): lega della stagione del
  dizionario sul suo calendario reale (2026/27: via dal 23/08/2026, una giornata a settimana di domenica; con
  `SERIEC_SEASON=2025_26` il calendario parte dal 24/08/2025, così le partite «correnti» sono sempre della stagione
  del dizionario): le giornate con data precedente a `SERIEC_AS_OF` risultano giocate, la prima in data uguale o
  successiva è il turno da prevedere. Senza `SERIEC_AS_OF` la data è *oggi, risolta a ogni esecuzione*: la posizione
  nel calendario avanza da sola mentre il server resta acceso (fa parte della chiave di cache). `SERIEC_HISTORY`
  (0-2, default 2) stagioni sintetiche *precedenti* quella del dizionario (2024/25-2025/26 per il 2026/27) con
  composizione in parte diversa (squadre storiche reali, Inter U23 nel girone A, Atalanta U23 nel C, Juventus Next
  Gen nel B) alimentano lo storico; se i nomi storici reali non bastano (12 dei 20 giocano la Serie C 2025/26) il
  generatore completa con le squadre storiche del dizionario e, in ultima istanza, con club sintetici `Storica 01`, `Storica 02`, ...,
  mai con un errore. `SERIEC_SIM_ROUNDS=N` fissa invece le giornate giocate. Se una combinazione di variabili è
  comunque rifiutata dalla pipeline, la pagina mostra l'errore e ripiega sulla configurazione predefinita.
- **Console (`main.py`)**: `--mode full` segue il calendario come la dashboard (`--as-of AAAA-MM-GG`, default oggi:
  giornate precedenti giocate, prima giornata uguale o successiva da prevedere; prima del via si simula comunque una
  giornata, a stagione finita si tiene l'ultima come turno da prevedere) oppure `--rounds N` fissa le giornate
  giocate (es. `--rounds 30` per il report di recupero dei parametri); `--mode mini` è un'istantanea fissa di inizio
  stagione (prime 20 partite = giornate 1-3, 4ª giornata da prevedere) e lo dichiara in testa all'output. Opzioni
  comuni: `--history 0|1|2`, `--season`. Ogni output apre con il perimetro (`Perimetro: Serie C 2026/27 Gironi A/B/C +
  Serie B (80 squadre) · stagione corrente: N partite · storico: M partite 2024/25-2025/26`, derivato da
  `config/competitions.toml`) e la posizione nel calendario.

# ⚙️ Stack Tecnologico

Solo ciò che è in `requirements.txt` e nel codice:

- **Core & Runtime:** Python 3.12+ (i pin di `numpy`/`scipy` non pubblicano wheel per 3.11)
- **Ingestion & Networking:** `httpx` (client asincrono con rotazione User-Agent, backoff esponenziale su 429/5xx e cache locale dei raw); `thefuzz` + `rapidfuzz` per il matching fuzzy dei nomi squadra, con fallback a `difflib`
- **Calcolo Scientifico:** `numpy`, `scipy` (`optimize.minimize`, SLSQP / L-BFGS-B), `pandas`
- **Dashboard & Visualizzazione:** `Streamlit` + `Plotly`
- **Test:** `pytest`, `pytest-asyncio` (HTTP simulato con `httpx.MockTransport`)

La raccolta dati si lancia a mano (`python -m src.ingestion.harvester ...`): non è prevista alcuna automazione schedulata.

---

## 📊 Pipeline Operativa

```mermaid
graph TD
    A[SofaScore / FotMob Endpoints] -->|Scraping Asincrono| B[Data Sanitizer & Storage]
    B --> C[Master Dataset Serie C A-B-C + Serie B]
    C --> D[Dixon-Coles Optimizer]
    D --> E[Matrice Bivariata di Probabilità]
    F[Palinsesto Quote Bookmaker] --> G[De-Vigging Module]
    E --> H[EV Engine & Kelly Sizing]
    G --> H
    H --> I[Dashboard & Alert Report]