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
│   ├── competitions.toml          # Le 13 competizioni del perimetro (Serie C A/B/C, Serie B, Serie D A-I) + season id FootyStats, in ordine di visualizzazione
│   ├── teams_2026_27.json         # Dizionario canonico squadre + alias delle 13 competizioni (242 squadre) = PERIMETRO (default), allineato a FootyStats a ogni raccolta
│   ├── teams_2025_26.json         # Stagione precedente: solo fonte di alias storici (squadre uscite dal perimetro, movimenti B<->C)
│   └── endpoints.toml             # Endpoint FootyStats (API ufficiale) e SofaScore/FotMob, retry, cache, rate-limit
├── data/                          # SOLO LOCALE (escluso da git): dati FootyStats scaricati con la tua chiave
│   ├── raw/                       # Cache JSON grezzi (chiave = sha1 URL, TTL)
│   ├── processed/                 # matches.csv / fixtures.csv normalizzati (fixture con orario UTC e stato scheduled/postponed)
│   ├── odds/                      # Quote pre-match (CSV)
│   └── footystats_teams.json      # Mappa id squadra FootyStats -> nome canonico (creata alla prima raccolta)
├── src/
│   ├── common/                    # Contratti condivisi
│   │   ├── competitions.py        # Competition + load_competitions(): legge config/competitions.toml (fallback integrato)
│   │   ├── schemas.py             # Girone (enum DINAMICO dalle competizioni), MatchRecord, Fixture, OddsQuote, MatchPrediction, Market
│   │   ├── teams.py               # TeamRegistry (qualsiasi numero di competizioni) + normalizzazione nomi
│   │   ├── logging_config.py      # Logging strutturato (JSON lines)
│   │   └── paths.py
│   ├── ingestion/                 # MODULO 1
│   │   ├── footystats.py          # Provider FootyStats (API ufficiale): sincronizzazione dizionario, storico 3 stagioni, fixture con orari e rinvii, quote, CLI
│   │   ├── calendar_snapshot.py   # Verifica dei calendari FootyStats e vista delle giornate del prodotto (solo calendario)
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

Lo script crea l'ambiente Python, installa le dipendenze, **chiede la chiave FootyStats** se non la trova (variabile
`FOOTYSTATS_API_KEY` oppure file `.env` accanto allo script, copia di `.env.example`; la chiave resta solo sul tuo
computer), scarica i dati reali con `python -m src.ingestion.footystats` (Serie C A/B/C, Serie B, Serie D A–I, ultime
tre stagioni, orari, partite rinviate, quote pre-match) **allineando prima il dizionario squadre a FootyStats** e apre
la dashboard nel browser. Senza chiave non ci sono dati reali da mostrare: la dashboard non presenta mai una lega
simulata come se fosse reale. Per aprire solo la dashboard senza aggiornare i dati: `avvia.bat --solo-dashboard`.

## 🔒 Dati FootyStats: solo in locale, mai nel repository

Il repository è **pubblico** e i dati FootyStats (squadre, calendari, risultati, statistiche, quote) sono concessi in
licenza al titolare della chiave: pubblicarli su GitHub significherebbe ridistribuirli. Per questo:

- i dati si scaricano **a runtime con la tua chiave** (`avvia.*`, `python -m src.ingestion.footystats`, l'aggiornamento
  dalla dashboard) e restano **solo sul disco locale**: `.gitignore` esclude `data/processed/*`, `data/odds/*`,
  `data/raw/*`, `data/*.json` (la mappa id → nome) e `.env`;
- non esistono più il workflow che committava i CSV (`aggiorna-dati.yml`) né gli script `pubblica_dati.*`: c'era anche
  un motivo tecnico, i workflow pianificati girano solo sul branch principale, quindi la raccolta automatica non era mai
  partita e la dashboard ripiegava in silenzio sulla lega simulata;
- il workflow `.github/workflows/verifica-calendari.yml` (a ogni push sul branch di lavoro che tocca l'ingestion o il
  calendario, oppure *Run workflow*) usa il secret `FOOTYSTATS_API_KEY` **senza scrivere nulla nel repository e senza
  artifact**: stampa nel log solo fatti di calendario (squadre, giornate, orari, stato, risultati; mai quote né
  statistiche). Passo 1: istantanea `python -m src.ingestion.calendar_snapshot` (anomalie, rinvii, giornata corrente,
  prova a secco della sincronizzazione del dizionario). Passo 2: il percorso del prodotto,
  `python -m src.ingestion.footystats --seasons 3 --no-odds --out ci/matches.csv --fixtures-out ci/fixtures.csv
  --odds-out ci/odds.csv` in una cartella temporanea fuori da `data/` (il dizionario viene sincronizzato nella copia di lavoro, mai
  committato), poi `--product-view` stampa la giornata corrente di ogni competizione **come la mostra la dashboard**
  (stessa regola di `src/engine/calendar.py`), con i recuperi e le partite rinviate.

### Dashboard su Streamlit Community Cloud

Non caricare né `.env` né CSV: nelle impostazioni dell'app, **Secrets**, aggiungi una riga
`FOOTYSTATS_API_KEY = "la-tua-chiave"`. L'app scarica i dati da FootyStats a runtime sul proprio disco (temporaneo) con
quella chiave; il repository resta privo di dati.

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

# 3) raccolta dati reali da FootyStats con la TUA chiave (.env o FOOTYSTATS_API_KEY): tutte le 13 competizioni, ultime
#    3 stagioni, fixture con orario e stato (rinviate incluse), quote pre-match -> data/ (solo locale, escluso da git);
#    prima allinea config/teams_2026_27.json a /league-teams (squadre aggiunte/spostate/rimosse, report in console)
python -m src.ingestion.footystats --seasons 3
#    solo l'allineamento del dizionario squadre, senza scaricare partite
python -m src.ingestion.footystats --sync-registry
#    fallback manuale per i soli gironi di Serie C: SofaScore (senza chiave, solo dal proprio PC: vedi "Ingestion")
python -m src.ingestion.harvester --gironi A B C --seasons 3 --providers sofascore

# 4) pipeline sui dati reali con quote bookmaker in data/odds/*.csv (--section all | "Serie C" | serie-b | d | c,d ...
#    sceglie le sezioni stampate); con i dati reali stampa anche AFFIDABILITÀ MODELLO (backtest walk-forward,
#    --no-backtest per saltarla) e SCHEDINE (le 3 schedine della prossima giornata, --no-slips per saltarla)
python main.py --processed data/processed/matches.csv --fixtures data/processed/fixtures.csv --odds data/odds
python main.py --processed data/processed/matches.csv --odds data/odds --section serie-b --slip-mixed 8
python main.py --mode full --rounds 6 --history 0 --serie-d-demo --section d   # la sezione Serie D su dati sintetici
python main.py --mode full --history 1 --backtest                              # demo con la sezione AFFIDABILITÀ MODELLO

# 5) dashboard interattiva sui dati reali di data/processed (scaricati con la tua chiave); senza dati mostra la pagina
#    iniziale che spiega come configurare la chiave, MAI una lega simulata al loro posto
streamlit run src/dashboard/app.py
#    demo esplicita su dati inventati (banner rosso su ogni pagina, mai per scommettere): SERIEC_DEMO=1
#    (SERIEC_FORCE_SIMULATE=1 è l'alias deprecato); altre variabili: SERIEC_SEASON=2026_27  SERIEC_HISTORY=2
#    SERIEC_AS_OF=2026-11-06  SERIEC_SIM_ROUNDS=15 (solo demo: giornate fisse invece del calendario)  SERIEC_SEED=7
#    SERIEC_BACKTEST=auto|manual|off (backtest della tab Affidabilità modello: all'avvio, su richiesta o mai)

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
family = "Serie B"                   # lega di appartenenza: i tre gironi condividono "Serie C", i nove di Serie D "Serie D"
[competition.season_ids]             # opzionale: season id FootyStats per stagione = fallback usato SOLO se /league-list
"2024/25" = 12621                    # non risponde (una lega non scelta nell'account viene saltata: l'API la rifiuta)
"2025/26" = 15632
"2026/27" = 17404
```

Da questo file nasce, **all'import**, l'enum `src.common.schemas.Girone` (13 membri in ordine di file: `Girone.A`,
`Girone.B`, `Girone.C`, `Girone.SB`, `Girone.DA` … `Girone.DI`; `Girone.parse()` accetta `'A'`, `'Girone A'`, `'SB'`,
`'Serie B'`, `'serie_b'`, `'Italy Serie B'`, `'DA'`, `'D-A'`, `'Serie D Girone A'`, `'Italy Serie D Group A'`, senza
distinzione di maiuscole; `'A'`/`'B'`/`'C'` sono sempre i gironi di Serie C e `'D'` da solo è ambiguo e viene
rifiutato) con le proprietà `.label`, `.short`, `.family`, `.tier`, `.competition`. Se il file manca il codice usa le
definizioni integrate (le stesse 13), così gli import non falliscono mai. Le competizioni sono raggruppate in
**famiglie** (`families()` → `Serie C`, `Serie B`, `Serie D`; `describe_competitions(family=...)`,
`competitions_of_family`, `family_label`): il modello stima **una sola volta** tutto il perimetro, l'interfaccia è
divisa per famiglia. Il **perimetro** è il dizionario della stagione corrente (`config/teams_2026_27.json`, chiavi =
codici delle competizioni, 242 squadre): `Serie C 2026/27 Gironi A/B/C + Serie B + Serie D Gironi A-I (242 squadre)` — la frase è derivata dalla
configurazione (`TeamRegistry.perimeter_label`, che chiamata con una famiglia la restringe:
`registry.perimeter_label('Serie D')` → `Serie D 2026/27 Gironi A-I (162 squadre)`), conta solo le competizioni che
hanno squadre (`registry.gironi`; quelle vuote sono in `registry.missing_competitions()`) e compare identica nel
banner della console e della dashboard. Il modello stima un fattore campo e un vincolo Σα = 0 per competizione;
simulatore, console e dashboard iterano l'enum, mai una lista A/B/C.

| codice | etichetta | breve | famiglia | tier | FootyStats (`footystats_name`) | SofaScore |
|---|---|---|---|---|---|---|
| `A` | Serie C Girone A | Girone A | Serie C | 3 | Italy Serie C Group A | 141 |
| `B` | Serie C Girone B | Girone B | Serie C | 3 | Italy Serie C Group B | 142 |
| `C` | Serie C Girone C | Girone C | Serie C | 3 | Italy Serie C Group C | 143 |
| `SB` | Serie B | Serie B | Serie B | 2 | Italy Serie B | 53 |
| `DA` | Serie D Girone A | D-A | Serie D | 4 | Italy Serie D Group A | – |
| `DB` | Serie D Girone B | D-B | Serie D | 4 | Italy Serie D Group B | – |
| `DC` | Serie D Girone C | D-C | Serie D | 4 | Italy Serie D Group C | – |
| `DD` | Serie D Girone D | D-D | Serie D | 4 | Italy Serie D Group D | – |
| `DE` | Serie D Girone E | D-E | Serie D | 4 | Italy Serie D Group E | – |
| `DF` | Serie D Girone F | D-F | Serie D | 4 | Italy Serie D Group F | – |
| `DG` | Serie D Girone G | D-G | Serie D | 4 | Italy Serie D Group G | – |
| `DH` | Serie D Girone H | D-H | Serie D | 4 | Italy Serie D Group H | – |
| `DI` | Serie D Girone I | D-I | Serie D | 4 | Italy Serie D Group I | – |

Ogni riga porta anche i season id FootyStats 2024/25–2026/27 (`[competition.season_ids]`, verificati il 6 novembre
2026); `–` = `sofascore_tournament_id = 0`, la competizione non è servita da SofaScore
(`Competition.supports_provider('sofascore')` è falso).

### Sezioni, gruppi di stima e campionati da abilitare (Serie A, Europa)

Ogni `[[competition]]` ha due chiavi facoltative: `enabled = true|false` (default `true`) e `section = "..."` (default:
la `family`). Le **sezioni** sono le pagine della dashboard e della console, nell'ordine in cui compaiono nel file:
con il file distribuito `Serie C` (gironi A, B, C), `Serie B` e `Serie D` (gironi A-I), ognuna con la propria pagina
(la vecchia pagina unica «Serie B e Serie C» è divisa in due; il nome resta accettato come alias, per esempio
`main.py --section bc`). Il **gruppo di stima** è il `country`: le competizioni dello stesso paese sono stimate
**insieme** da un solo modello Dixon-Coles (Serie B, C e D sono collegate dalle squadre promosse e retrocesse), paesi
diversi **separatamente** (`src/engine/fitgroups.py`, `fit_by_group`; ottimizzatore L-BFGS-B di default, circa 40
volte più veloce di SLSQP a parità di ottimo; SLSQP resta selezionabile in *Opzioni del modello* e con `--method SLSQP`).

Il file contiene già, **disabilitati**, Serie A (`SA`, primo nel file: abilitata diventa la prima sezione ed entra nel
gruppo `Italy`), Premier League (`EPL`), La Liga (`LL`), Bundesliga (`BL`), Ligue 1 (`L1`), Eredivisie (`ERE`) e
Primeira Liga (`PL1`). Una competizione disabilitata non esiste per il resto del codice (niente enum, raccolta, sezione,
gruppo di stima; la sua chiave nel dizionario squadre viene ignorata con una riga di log). **Per abilitarne una**:

1. metti `enabled = true` nella sua tabella di `config/competitions.toml`;
2. scegli la lega sull'account footystats.org (*Choose leagues*: l'API rifiuta i dati di una lega non scelta e i piani
   limitano il numero di leghe);
3. premi **Aggiorna dati ora** nella dashboard (o lancia `python -m src.ingestion.footystats`): la sincronizzazione del
   dizionario riempie le sue squadre in `config/teams_<stagione>.json` da `/league-teams`, poi ne vengono scaricate
   partite, calendario e quote. Al riavvio compare la nuova sezione (un campionato estero ha il suo gruppo di stima).

`enabled = false` lo spegne di nuovo, in modo reversibile. Per un campionato non elencato aggiungi una tabella con le
stesse chiavi e segui i passi 2 e 3.

### Serie D (Gironi A–I)

La Serie D 2026/27 ha **nove gironi** (`DA` … `DI`, 18 squadre ciascuno, 34 giornate) ed è una **sezione esclusiva**
dell'interfaccia (selettore *Sezione* della dashboard, `--section d` a console), ma viene stimata insieme a Serie B e
Serie C: le squadre promosse e retrocesse collegano le divisioni, il fattore campo per competizione e il vincolo
Σα = 0 per girone tengono i gironi confrontabili. Come arrivano i dati:

- **Solo FootyStats**: SofaScore non copre la Serie D. Le nove leghe *Italy Serie D Group A* … *Italy Serie D Group I*
  **devono** essere fra le leghe scelte del tuo account footystats.org (*Choose leagues*): l'API rifiuta i dati di una
  lega non scelta (`League is not chosen by the user`, verificato il 7 novembre 2026), qualunque season id si passi,
  quindi i season id di `config/competitions.toml` non possono sostituirla. Una competizione la cui lega non è scelta
  viene **saltata** con un avviso che nomina la lega da aggiungere (riga `skipped DA: …` in console,
  `FootyStatsHarvest.not_chosen`), senza fermare né invalidare la raccolta di Serie B e Serie C, che vengono scritte
  normalmente (exit code 0). I piani FootyStats limitano il numero di leghe scelte: se non puoi aggiungerle tutte,
  restringi `--competitions` (o l'input `competitions` del workflow) alle leghe che hai.
- **Liste squadre da FootyStats (sincronizzazione del dizionario)**: `config/teams_2026_27.json` contiene le 162
  squadre dei nove gironi così come le elenca `/league-teams` (nome canonico = `cleanName` FootyStats; unica eccezione
  `AC Palermo Serie D`, perché `AC Palermo` si normalizza come il `Palermo` di Serie B, un altro club). Ogni raccolta
  FootyStats (`python -m src.ingestion.footystats`, `avvia.*`, l'aggiornamento della dashboard) **sincronizza** il
  dizionario della stagione corrente con `/league-teams`, per tutte le competizioni richieste (vedi *Dati reali con
  FootyStats*); `--no-sync` la disattiva.
- **Statistiche più povere**: su FootyStats la Serie D ha spesso tiri e xG non raccolti (`0`/`-1`, letti come `NaN`)
  e quote pre-match per meno partite. Tutto degrada senza errori: λ e μ si stimano dai gol, senza xG il filtro
  REGRESSION_TRAP non scatta, il pannello valore è in parte vuoto.
- **Simulazione**: `simulate_serie_d_registry(seed, n_teams=18)` costruisce un dizionario sintetico di Serie D
  (`Squadra DA 01` …) per test e demo; `simulate_league`/`simulate_history` simulano solo le competizioni che hanno
  squadre nel dizionario: con il dizionario 2026/27 allineato a FootyStats (Serie D compresa) la lega simulata di
  default copre tutte le 13 competizioni (242 squadre).

**Per aggiungere un campionato** (es. un'altra nazione):

1. aggiungi una tabella `[[competition]]` a `config/competitions.toml` con un `code` nuovo (maiuscole/cifre, es. `DA`);
2. aggiungi la chiave della competizione in `config/teams_<stagione>.json` (anche un oggetto vuoto `{}`): la prima
   raccolta FootyStats la riempie da `/league-teams` (sincronizzazione del dizionario). Una squadra ha **un solo nome
   canonico** anche se cambia divisione: `Pescara` in Serie B 2025/26 e nel girone B 2026/27; il dizionario della
   stagione corrente decide la sua competizione, le stagioni passate restano storico;
3. inserisci il `footystats_name` esatto restituito da `/league-list` e aggiungi la lega al tuo account su
   footystats.org (l'API mostra solo le leghe scelte), poi raccogli i dati con l'harvester FootyStats.

Nient'altro va toccato: enum, dizionario, motore, selettori e banner seguono il file. Per una stagione nuova
(`config/teams_2027_28.json` ancora assente) `python -m src.ingestion.footystats --init-registry --season 2027_28`
crea il file dalle squadre FootyStats (aggiunge squadre e alias, non rimuove nulla); da lì in poi ogni raccolta con
`--season 2027_28` lo mantiene allineato.

## 🔑 Dati reali con FootyStats (API ufficiale, unica fonte di verità)

FootyStats (<https://footystats.org/api/>) è un'API documentata e autenticata con chiave: copre i tre gironi di Serie
C, la Serie B e i nove gironi di Serie D con squadre, calendario ufficiale (`game_week` = giornata, orario
`date_unix`), risultati, tiri, tiri nello specchio, angoli, xG (dove disponibile) e **quote pre-match** (1X2,
Over/Under 1.5-2.5-3.5, Gol/NoGol). È **l'unica fonte di verità** per squadre e calendari: il dizionario squadre e le
giornate mostrate seguono FootyStats, mai una lista scritta a mano o una lega simulata. I dati restano **solo in
locale** (vedi *Dati FootyStats: solo in locale*).

1. **Chiave**: dal tuo account su footystats.org → *API*. Non committarla mai: va in una variabile d'ambiente
   `FOOTYSTATS_API_KEY` oppure in un file `.env` nella root del repository (copia `.env.example`;
   `.env` è in `.gitignore`), nei *Secrets* dell'app su Streamlit Cloud, oppure passata con `--key`. Il codice la
   maschera ovunque (`***`): non finisce in log, cache (`data/raw/`), messaggi d'errore né nei file prodotti.
2. **Leghe dell'account**: su footystats.org aggiungi al tuo account (*Choose leagues*) **Italy Serie C Group A**,
   **Italy Serie C Group B**, **Italy Serie C Group C**, **Italy Serie B** e le nove leghe **Italy Serie D Group A** …
   **Italy Serie D Group I**. **Ogni competizione da raccogliere deve essere fra le leghe scelte**: l'API elenca solo
   quelle (`/league-list?chosen_leagues_only=true`) e rifiuta `/league-teams` e `/league-matches` di una lega non
   scelta (`League is not chosen by the user`), quindi una competizione la cui lega manca viene saltata con un avviso
   che nomina la lega da aggiungere (`skipped DA: …`) mentre le altre vengono scritte normalmente. I piani FootyStats
   limitano il numero di leghe scelte: se non puoi aggiungerle tutte, restringi `--competitions` (es. `A B C SB`). I
   season id di `config/competitions.toml` (`[competition.season_ids]`) sono un fallback usato solo se `/league-list`
   non risponde. Verifica con `python -m src.ingestion.footystats --list-leagues`.
3. **Raccolta**:

```bash
# ultime 3 stagioni di TUTTE le 13 competizioni (Serie C A/B/C, Serie B, Serie D A-I) in UN solo matches.csv + fixture
# della stagione corrente (orario, stato, rinviate) + quote; prima allinea config/teams_2026_27.json a FootyStats
python -m src.ingestion.footystats --seasons 3
# solo Serie C e Serie B
python -m src.ingestion.footystats --competitions A B C SB --seasons 3
# solo Serie B, 2 stagioni, senza quote, senza toccare il dizionario (--no-odds non riscrive data/odds/footystats.csv:
# toglie solo le quote delle partite rinviate o già iniziate, prezzi non più validi)
python -m src.ingestion.footystats --competitions SB --seasons 2 --no-odds --no-sync
# solo l'allineamento del dizionario squadre (nessuna partita scaricata), con il report
python -m src.ingestion.footystats --sync-registry
# leghe e season id del tuo account
python -m src.ingestion.footystats --list-leagues
# crea config/teams_2027_28.json per una stagione nuova (aggiunge squadre/alias, non rimuove nulla)
python -m src.ingestion.footystats --init-registry --season 2027_28 --competitions A B C SB
# controllo dei calendari (solo calendario: niente quote né statistiche), e la giornata corrente come la mostra il prodotto
python -m src.ingestion.calendar_snapshot --summary snapshot/riepilogo.md
python -m src.ingestion.calendar_snapshot --product-view --matches data/processed/matches.csv --fixtures data/processed/fixtures.csv
```

- **Sincronizzazione del dizionario squadre** (`sync_registry` / `apply_registry_sync` in
  `src/ingestion/footystats.py`, eseguita da ogni raccolta prima di leggere le partite): per ogni competizione
  richiesta il dizionario della stagione corrente diventa **uguale** a `/league-teams`. L'identità di un club si
  decide, in quest'ordine e **mai per somiglianza (fuzzy)**: id squadra FootyStats già associato da una
  sincronizzazione precedente (`data/footystats_teams.json`); grafia esatta o alias (`cleanName`, `name`,
  `english_name`) di un club della stessa competizione; il nome che la regola delle collisioni gli darebbe; grafia
  esatta o alias di un club di un'altra competizione, che viene **spostato** (solo se anche quella competizione è
  sincronizzata nella stessa esecuzione). I club nuovi prendono il `cleanName` (se la grafia è già di un altro club:
  `name`, poi `cleanName + famiglia`, es. `AC Palermo Serie D`, poi `cleanName + girone`); quelli che FootyStats non
  elenca più per la competizione vengono **rimossi** (restano nello storico tramite i file delle stagioni precedenti);
  un nome canonico uguale al vecchio `cleanName` segue una ridenominazione FootyStats (il vecchio resta alias), i nomi
  scelti a mano (`Juventus Next Gen` ← `Juventus II`, `Atalanta U23` ← `Atalanta II`, `Inter U23` ← `Inter II`,
  `Campobasso` ← `Città di Campobasso`, …) non cambiano mai. Le grafie FootyStats diventano alias quando nessun altro club
  le usa; gli alias esistenti restano. Il file viene scritto in modo atomico solo se cambia, e mai se non si
  caricherebbe. Il report (`RegistrySyncReport.describe()`) è stampato dalla CLI: aggiunte, rimosse, spostate,
  rinominate per competizione e totale squadre. Una competizione senza lista (lega non scelta, errore) o con lista
  vuota resta invariata.
- **Un club, una competizione**: anche senza la sincronizzazione (`--no-sync`, una lista non scaricata, un id non
  ancora associato) una squadra della stagione corrente non viene mai fusa con un club che il dizionario mette in
  un'altra competizione: `AC Palermo` (Serie D, girone I) resta `AC Palermo Serie D` e non diventa il `Palermo` di
  Serie B, così nessuna squadra gioca in due campionati. Lo stesso vale per `--init-registry` di una stagione nuova:
  due club FootyStats che rispondono allo stesso nome restano due club (il secondo prende il nome della regola delle
  collisioni, segnalato come `conflict`), mentre promozioni e retrocessioni mantengono il nome. Uno spostamento vero
  fra competizioni nella stagione in corso lo applica solo la sincronizzazione, che riconosce i club per id.
- **Calendario fedele**: `game_week` è la giornata ufficiale e le giornate si giocano da venerdì a lunedì, con
  recuperi infrasettimanali. Le partite `incomplete` con orario futuro sono fixture `scheduled` con il loro orario
  (`Fixture.kickoff`, UTC; `kickoff_local` in Europe/Rome); quelle con orario passato da **più di 3 ore** sono fixture
  `postponed` (rinviate) con data e orario originali, non più scartate, così una giornata non sembra mai più corta di
  quello che è; quelle iniziate da meno di 3 ore sono **in corso** e restano fixture `scheduled` con il loro orario
  (già passato), così la giornata le elenca ancora e resta quella corrente fino al risultato, che arriva con la
  raccolta successiva. Le partite giocate della stagione corrente (anche quelle della giornata in corso, es. i due
  anticipi del venerdì) sono in `matches.csv`. `fixtures.csv` ha le colonne
  `match_id,date,round,girone,home_team,away_team,kickoff,status` (`kickoff` ISO 8601 UTC, `status`
  `scheduled|postponed`); i file del vecchio formato a 6 colonne si leggono ancora. Le quote si raccolgono solo per le
  fixture `scheduled` non ancora iniziate.
- **Dati freschi anche dalla cache**: lo stato di una partita `incomplete` si giudica all'istante in cui FootyStats
  l'ha servita (`fetched_at` della cache `data/raw/`), mai a un orologio successivo: una partita ancora da giocare in
  un payload di cinque ore fa non diventa «rinviata». Il calendario della stagione in corso (`/league-matches`) si
  riusa dalla cache al massimo per `live_matches_cache_minutes` minuti (`[footystats]` di `config/endpoints.toml`,
  default 10), qualunque sia `[http] cache_ttl_hours`: nei giorni di partita un aggiornamento mostra i risultati
  appena arrivati.
- Le competizioni sono i codici di `config/competitions.toml` (`A`, `B`, `C`, `SB`, `DA` … `DI`; senza
  `--competitions` si raccolgono tutte quelle che il provider supporta, 13); il nome FootyStats di ciascuna
  (`footystats_name`, es. `Italy Serie B`) è lì, con un fallback in `[footystats.league_names]` di
  `config/endpoints.toml`.
- Storico: per ogni competizione vengono scaricate le ultime `--seasons` stagioni elencate dall'API (season id
  risolti a runtime da `/league-list`; `[competition.season_ids]` di `competitions.toml`, poi
  `[footystats.season_ids]` di `endpoints.toml`, sono il fallback solo se la lista non risponde; una lega che la lista
  non elenca non è scelta nell'account e viene saltata, vedi sopra).
  Tutte le squadre che vi hanno partecipato restano nello storico, anche quelle poi retrocesse, promosse o
  fallite: mantengono la loro grafia canonica (*historical teams*) e servono a valutare le avversarie, ma non
  entrano in classifica/previsioni. Le partite `suspended`/`canceled` sono scartate.
- Statistiche mancanti: FootyStats usa `0`/`-1` per i dati non raccolti; tiri a 0 per entrambe le squadre o xG a 0
  per entrambe vengono letti come assenti (`NaN`), così il modello non li confonde con zeri veri.
- Nomi squadra: `data/footystats_teams.json` (locale, escluso da git, creato alla prima raccolta; un vecchio
  `config/footystats_teams.json` viene letto una volta per migrarlo) memorizza la mappa *id FootyStats → nome
  canonico*. Le voci scritte dalla sincronizzazione (`"method": "sync"`) identificano il club e battono ogni ricerca per
  grafia, anche nelle stagioni passate (gli id FootyStats sono stabili); per le altre un hit esatto/alias del
  dizionario batte l'id memorizzato, che batte il fuzzy. Correggi lì il campo `canonical` se una squadra è mappata male.
- Limiti: circa **1800 richieste/ora** sui piani standard; un run completo (13 competizioni × 3 stagioni) ne usa
  ~80 (4 competizioni: ~25). Il 429 viene ritentato con backoff; la cache in `data/raw/` evita di riscaricare le stagioni concluse.
- Exit code della CLI: `0` ok, `2` chiave mancante (con le istruzioni), `1` raccolta fallita o parziale (il CSV
  esistente non viene sovrascritto senza `--allow-partial`).
- La facade `src.ingestion.harvester.Harvester` prova i provider in ordine (`footystats`, poi `sofascore`
  quando una chiave è disponibile; solo `sofascore` altrimenti) e adotta il dizionario sincronizzato da FootyStats.

## 📥 Ingestion: raccolta dati reali (SofaScore, fallback)

SofaScore è un **fallback manuale** per i soli gironi di Serie C (nessun calendario con orari e rinvii, nessuna Serie B
né Serie D): squadre e calendari di riferimento restano quelli di FootyStats. Il **perimetro** è il dizionario della
stagione corrente (`config/teams_2026_27.json`: le 60 squadre dei tre gironi ufficiali Lega Pro, le 20 della Serie B e
le 162 dei nove gironi di Serie D, 242 in tutto, allineate a FootyStats): solo queste squadre vengono classificate,
previste e prezzate. Le stagioni precedenti servono
**unicamente come base storica** per stimare meglio la forza delle squadre correnti. L'harvester SofaScore copre i
soli gironi di Serie C; Serie B e Serie D arrivano da FootyStats (la Serie D non ha id SofaScore:
`sofascore_tournament_id = 0`, quindi `Harvester.supported_gironi('sofascore')` la esclude).

```bash
# dal proprio PC (connessione residenziale, vedi la nota 403): ultime 3 stagioni (default) in UN solo CSV
python -m src.ingestion.harvester --gironi A B C --seasons 3
# solo la stagione corrente
python -m src.ingestion.harvester --seasons 1
# perimetro esplicito (config/teams_<stagione>.json) oppure una singola stagione SofaScore
python -m src.ingestion.harvester --season 2026_27 --season-id 77031
# i CSV restano in data/processed/ (solo locale, escluso da git)
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
> una **connessione residenziale** (il proprio PC), e `main.py --processed ...` o la dashboard vanno eseguiti sullo
> stesso computer: i CSV non si committano mai (il repository è pubblico). Il 403 non viene ritentato perché
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

- **Solo dati reali, mai numeri inventati in silenzio**: la dashboard (`streamlit run src/dashboard/app.py`) lavora
  esclusivamente sui dati FootyStats scaricati **a runtime con la tua chiave** (`src/dashboard/data_refresh.py`) e
  salvati **solo in locale** (`data/processed`, `data/odds`, esclusi da git: i dati FootyStats sono in licenza al tuo
  account e non vanno pubblicati nel repository). All'avvio, se la chiave è configurata e i dati mancano o hanno più di
  **6 ore**, li riscarica (una volta ogni 6 ore per server, con barra di avanzamento): partite delle ultime 3 stagioni,
  calendario con orari e rinvii, quote pre-match, e allinea il dizionario squadre a FootyStats. La sidebar mostra
  `Dati FootyStats aggiornati alle HH:MM del gg/mm` e il pulsante **Aggiorna dati ora**; una competizione che non si
  può scaricare (lega non scelta sull'account, errore di rete) è segnalata con `st.warning` e tiene i dati precedenti,
  le altre si aggiornano; lo stesso vale per una sola stagione: se fallisce la stagione in corso di una competizione
  (es. HTTP 503) restano le sue partite, il calendario e le quote del download precedente. Finché i dati restano più
  vecchi di 6 ore (chiave mancante, aggiornamento non riuscito in una qualunque sessione) ogni esecuzione mostra in cima
  `Dati vecchi di 2 giorni (ultimo aggiornamento FootyStats alle 13:46 del 05/11/2026): …` con il motivo, e un avviso
  per ogni competizione ferma ai dati di un download precedente. Se i CSV sono stati riscritti dopo l'ultimo
  aggiornamento della dashboard (per esempio da `avvia.bat` / `avvia.command`, che lanciano
  `python -m src.ingestion.footystats`) conta l'ora del file `matches.csv`, così la dashboard non riscarica tutto
  subito dopo il launcher. Le differenze del dizionario squadre (squadre aggiunte, spostate, rinominate) sono in
  *Dettagli pipeline*.
- **Pagina iniziale senza dati**: senza `data/processed/matches.csv` la dashboard non mostra alcun tab, ma una pagina
  che dice cosa manca, come fornire la chiave e — se la chiave c'è — offre il pulsante **Scarica i dati da FootyStats
  adesso**. La chiave si legge, nell'ordine, da `FOOTYSTATS_API_KEY` nell'ambiente, dal file `.env` (copia di
  `.env.example`) e dai secrets di Streamlit; non viene mai stampata, registrata nei log o scritta nei file.
- **Streamlit Community Cloud**: nessun dato nel repository; in *App settings → Secrets* incolla
  `FOOTYSTATS_API_KEY = "la_tua_chiave"`: al primo avvio l'app scarica i dati sul server (e li riscarica ogni 6 ore o
  con *Aggiorna dati ora*). Le leghe delle competizioni vanno scelte sull'account footystats.org (*Choose leagues*).
- **Demo esplicita** (`SERIEC_DEMO=1`; `SERIEC_FORCE_SIMULATE=1` resta come alias deprecato): lega simulata della
  stagione del dizionario sul suo calendario (squadre vere, **calendari, risultati e quote inventati**). Ogni pagina e
  ogni tab apre con l'avviso rosso `DATI SIMULATI: squadre, calendari e risultati inventati. Non usare per
  scommettere.` e il badge della sidebar dice **Demo**; ogni CSV scaricato dalla demo (tabella giornata, risultati,
  selezioni, report valore, previsioni) si chiama `DEMO_…` e ha come prima colonna `AVVISO` =
  `DATI SIMULATI - NON USARE PER SCOMMETTERE`. Le giornate con data precedente a `SERIEC_AS_OF` (default oggi,
  risolto a ogni esecuzione) risultano giocate, la prima successiva è il turno da prevedere; `SERIEC_SIM_ROUNDS=N`
  fissa le giornate giocate, `SERIEC_HISTORY` (0-2, default 2) aggiunge stagioni sintetiche precedenti, `SERIEC_SEED`
  il seme. Con il dizionario reale completo (Serie B, Serie C e Serie D, 242 squadre) la prima stima della demo
  richiede qualche secondo (più il backtest della tab Affidabilità modello); poi resta in cache. Le cartelle dei dati reali si cambiano con `SERIEC_PROCESSED_DIR` /
  `SERIEC_ODDS_DIR`.
- **Perimetro sempre dichiarato**: badge della sidebar e sottotitolo della pagina riportano il perimetro derivato da
  `config/competitions.toml` e dal dizionario, mai cablato (`Perimetro: Serie C 2026/27 Gironi A/B/C + Serie B (80
  squadre) · N partite stagione corrente · M partite storiche (2024/25-2025/26)` per l'intero perimetro, ristretto alla sezione scelta nella dashboard); la
  console di `main.py` apre con la stessa frase e stampa il riepilogo per stagione (partite, squadre attuali e storiche).
- **Sezioni**: la sidebar apre con il selettore **Sezione**: una pagina per campionato, dalle sezioni di
  `config/competitions.toml` (`Serie C` di default, `Serie B`, `Serie D`, e ogni campionato abilitato: vedi «Sezioni,
  gruppi di stima e campionati da abilitare»). La stima Dixon-Coles è **una per paese** (Serie B, C e D insieme: le
  squadre promosse e retrocesse collegano le divisioni, il fattore campo e il vincolo Σα = 0 per competizione tengono i
  gironi confrontabili), ma tutto ciò che si vede è della sezione scelta: titoli dei tab (`Classifica di forza · Serie
  D`), selettore Competizione (`Tutti` + `D-A` … `D-I`), tabella giornata, risultati, pannello valore, alert
  REGRESSION_TRAP, selezioni (le 5 X / 12 GG sono calcolate **dentro** la sezione), export e la frase del perimetro
  (`Perimetro: Serie D 2026/27 Gironi A-I (162 squadre) · …`). Le giornate di ogni sezione sono costruite dalle sue
  giornate ufficiali; la giornata, la partita e i filtri scelti sono per sezione (`st.session_state["matchday_d"]`,
  `fixture_b`, …). Una sezione configurata ma senza dati (Serie D prima della prima raccolta, un campionato appena
  abilitato) mostra in ogni tab un solo avviso con lo stato e come scaricarla. Le statistiche di Serie D su FootyStats
  sono più povere e tutto degrada senza errori: xG e tiri spesso mancano (colonne `-`, λ e μ stimati dai gol), senza
  xG il filtro REGRESSION_TRAP non scatta, le quote coprono meno partite (pannello valore in parte vuoto, con una nota).
- **Affidabilità modello** (tab, `src/engine/backtest.py`, `src/engine/calibration.py`): quanto le probabilità del
  modello hanno «preso» i risultati, misurato **fuori campione**: un **backtest walk-forward** prevede ogni giornata
  passata (blocco della settimana) con un modello stimato **solo** sulle partite giocate prima (stesso decadimento,
  stessa prior, storico incluso, L-BFGS-B ripartendo dalla stima della settimana prima); le probabilità della stima di
  produzione, che ha già visto quei risultati, non sono mai usate per giudicare. Di default stagione corrente e
  precedente (selettore *Stagioni valutate*), per la sezione della sidebar o *Tutte le sezioni*: riquadri (partite
  valutate, log loss modello contro le frequenze di base, *Modello vs mercato* con le due log loss sulle **stesse**
  partite con quote, Brier, accuratezza), tabella per mercato (1X2, GG/NG, Over/Under 1.5/2.5/3.5) con le frequenze di
  base del campionato e le probabilità **senza margine** del bookmaker (quote pre-partita delle partite giocate,
  `data/odds/history/odds_history.csv`). Il mercato si misura solo sulle *Partite con quote*: accanto alle sue colonne
  ci sono *Log loss modello (partite con quote)* e *Brier modello (partite con quote)*, gli unici numeri del modello
  confrontabili con il bookmaker (la copertura delle quote cambia da campionato a campionato); lo stesso vale per
  l'export, la console e il riepilogo CI. Poi il grafico di affidabilità (probabilità prevista contro frequenza osservata
  con le barre di Wilson al 95%, selettore dell'esito), la tabella per competizione, il track record delle liste top 5
  X / top 12 GG, delle **multiple top-k** (le prime 2-5 partite della lista di ogni giornata giocate insieme, con il
  dettaglio giornata per giornata), delle tre schedine e delle singole di valore (hit rate con intervallo di Wilson, ROI
  alle quote reali con intervallo al 95%, numero di giornate) e la spiegazione del perché poche vincite non sono una
  prova. Il backtest gira una volta per versione dei dati (barra di avanzamento), resta in memoria ed è salvato in
  `data/backtest.parquet` / `data/backtest.json` (esclusi da git), insieme alla ricalibrazione giornata per giornata
  (`data/backtest_rolling.*`, calcolata subito dopo il backtest e riletta al riavvio); la chiave della cache contiene la
  versione della logica e la configurazione delle competizioni, quindi spegnere o spostare un campionato in
  `config/competitions.toml` lo fa ricalcolare (le righe di un campionato disattivato non arrivano mai alle viste).
  `SERIEC_BACKTEST=manual` lo calcola solo con il pulsante, `off` mai. Scarica i punteggi aggregati in CSV.
- **Ricalibrazione validata** («imparare dal differenziale»): sullo stesso backtest una piccola correzione per mercato
  (Platt su GG/NG e Over/Under, temperatura multinomiale sull'1X2, penalità verso l'identità; parametri propri per
  famiglia solo con abbastanza partite) è stimata sulle giornate più vecchie e giudicata sulle più recenti; si applica
  **solo** ai mercati in cui la log loss fuori campione migliora almeno dello 0,5%. Il tab dice *Calibrazione
  applicata: sì/no* con il miglioramento per mercato; l'interruttore **Usa probabilità ricalibrate** (acceso di default
  solo se un mercato ha superato la verifica) la applica a Giornata (con la probabilità originale accanto, colonne
  `orig.`), Opportunità di Valore, Selezioni e Schedine, ed è reversibile. Facoltativa e lenta, l'ottimizzazione di ξ e
  della forza della prior (9 combinazioni, log loss walk-forward) sceglie la combinazione sulle giornate più vecchie
  della stagione e la conferma sul 30% più recente: il secondo interruttore che la applica si accende solo se lì batte la
  configurazione standard di almeno lo 0,5%, e con la combinazione applicata la tab avvisa che le giornate della scelta
  non sono più del tutto fuori campione.
- **Schedine** (tab): per la giornata scelta (segue quella degli altri tab) e le sezioni scelte (di default tutte quelle
  con dati), fra le partite non ancora iniziate, tre schedine: **X** (5 partite con la X più probabile), **GG** (10
  partite con il GG più probabile) e **mista** (per ogni partita il segno più probabile fra 1, 2, GG e Over 2.5, poi le
  6 partite — da 3 a 10 — in cui quel segno è più probabile). Lunghezze, probabilità minima per partita e «solo partite
  quotate» sono regolabili. Per ogni schedina: partite, probabilità (prodotto, ipotesi di indipendenza dichiarata),
  quota fair, quota book (prodotto delle migliori quote, solo se tutte quotate), EV, partite vincenti attese, stake
  suggerito (Kelly frazionario sulla schedina con tetto di 1 unità, «non consigliata» con EV ≤ 0) e l'avviso che le
  multiple moltiplicano il margine del bookmaker; CSV con le tre schedine e, sotto, come sarebbero andate le stesse
  schedine nelle giornate passate del backtest.
- **Selettore Competizione**: `Tutti` + l'etichetta breve di ogni competizione configurata della sezione (`Girone A`,
  `Girone B`, `Girone C`, `Serie B`; `D-A` … `D-I` in Serie D); filtra classifica, partite e opportunità.
- **Classifica Forza**: solo le squadre correnti del dizionario, colonna *Competizione* = competizione del dizionario
  2026/27 (`Girone A`, ..., `Serie B`); stesso nome di colonna nel pannello valore e negli alert.
- **Quote**: ogni CSV in `data/odds/` è un bookmaker come gli altri, compreso quello scritto dall'aggiornamento
  FootyStats (`bookmaker = FootyStats`, quote pre-match 1X2 / Over-Under / GG-NG): la colonna *Bookmaker* del pannello
  valore dice da dove viene ogni quota.
- **Giornata corrente: la regola** (`src/engine/calendar.py`): la giornata è quella **ufficiale** (`game_week` di
  FootyStats), competizione per competizione. Sono candidate le giornate con almeno una partita in programma (non
  rinviata) con calcio d'inizio non più vecchio di 3 ore (una partita iniziata da meno di 3 ore tiene la giornata
  candidata); la **giornata corrente** è la più bassa candidata in cui almeno **metà** delle partite con una data non è
  ancora giocata (le rinviate senza nuova data non contano), oppure che è in corso: qualche partita giocata o iniziata e
  una partita rimasta nello stesso blocco della settimana (venerdì-lunedì o martedì-giovedì, `week_block`) della
  maggior parte del turno — un turno in corso resta corrente fino al posticipo del lunedì. Esempi reali: Serie C
  girone A, 13ª giornata con due partite giocate venerdì 06/11 e otto il 07-08/11 → è la giornata corrente e si vede con
  tutte e dieci le partite, le due del venerdì con il risultato; Serie D girone G, 10ª giornata con otto partite giocate
  il 31/10-01/11 e il recupero Trastevere-Anzio mercoledì 11/11 → non è la giornata corrente (lo è l'11ª del fine
  settimana) e il recupero finisce in **Recuperi e rinvii**; lo stesso per un turno invernale di Serie D con quattro
  partite giocate, quattro rinviate e una spostata a mercoledì.
- **Stato di una partita, all'istante della pagina**: `giocata 2-1` (nei dati con il risultato), `da giocare` (calcio
  d'inizio ancora da venire: le sole partite previste, nelle selezioni e nel pannello valore), `in corso / risultato in
  attesa` (calcio d'inizio passato ma risultato non ancora nei dati: partita in corso, risultato non ancora pubblicato
  da FootyStats o dati locali di prima della fine; resta nel suo turno, senza previsione né quote: nessuna giocata
  pre-partita su una partita iniziata) e `rinviata` (nessun risultato **48 ore** dopo il calcio d'inizio, come
  Chisola-Borgosesia della 1ª di Serie D girone A, 30/09: resta nella sua giornata con stato `rinviata` e nell'elenco
  dei rinvii, senza previsione).
- **Giornata** (tab, `src/engine/tables.py`, `matchday_table`): **ogni partita** delle giornate scelte in una tabella:
  colonna *Calcio d'inizio* in ora italiana (`sab 07/11 14:30`, conversione da UTC con ora legale/solare; i file senza
  orario mostrano solo il giorno), colonna *Stato* (`giocata 2-1` con le colonne del modello vuote, `da giocare` con la
  previsione, `in corso / risultato in attesa` e `rinviata` senza previsione), poi forza (α/β), forma delle ultime 5 gare (`V-P-N-V-V`, punti, gol fatti/subiti), xG delle
  ultime 3, λ, μ, totale atteso, risultato più probabile, probabilità 1X2 / Over 1.5-2.5-3.5 / GG, quote fair, miglior
  quota book per esito, EV massimo (`X @3.40`) e l'unione dei flag. Senza competizione nella sidebar la tabella è la
  giornata scelta (di default la giornata corrente di ogni competizione); con una competizione compare il menù
  **Giornata n.** con tutte le sue giornate ufficiali (passate e future, `(corrente)` accanto a quella in corso).
  L'espansore **Recuperi e rinvii** elenca le partite rinviate e i recuperi ancora da giocare fuori dalla giornata
  corrente con nuova data, data originale (solo per una partita spostata davvero) e probabilità 1X2 / GG del modello;
  una partita già iniziata senza risultato non è mai un recupero. Cliccando una partita da giocare si apre sotto il dettaglio —
  metriche, matrice dei punteggi, mercati — con il menù *Partita* come alternativa sincronizzata; «Scarica tabella
  giornata (CSV)» esporta la tabella completa (più `match_id`).
- **Risultati** (`results_table`): una riga per partita giocata della stagione scelta (default quella corrente; il
  selettore *Stagione* apre lo storico) e della competizione della sidebar, filtrabile per squadra (il menù *Squadra*
  offre le squadre che hanno giocato quella stagione in quella competizione: nelle stagioni passate anche quelle non
  più nel perimetro, mai una squadra senza partite): risultato, esito 1X2, tiri / tiri in porta / angoli casa-ospiti
  (`12/7`), xG di entrambe, più λ, μ, **P esito** (la probabilità che il modello assegna all'esito avvenuto: lettura a
  posteriori, il modello è stimato anche su quella partita) e **Sorpresa** («sorpresa» sotto il 20%, «atteso» sopra il
  50%). Colonne ordinabili dall'intestazione, «Scarica risultati (CSV)».
- **Previsioni e valore**: solo partite in programma della stagione corrente fra squadre correnti; le partite datate in
  un'altra stagione (fixtures stantii dell'anno scorso), con una squadra storica o sconosciuta allo storico sono
  elencate come saltate in *Dettagli pipeline*.
- **Giornate** (`src/engine/picks.py`, `matchdays`): la **prima giornata** è l'unione della giornata corrente di ogni
  competizione, qualunque sia il numero (Serie B all'11ª, gironi di Serie C alla 13ª) e con le partite già giocate; le
  **successive** raggruppano i turni seguenti giocati nello stesso blocco della settimana calcistica (così la Serie B in
  sosta per le nazionali manca da quella giornata senza spostare gli accoppiamenti seguenti, e due turni della stessa
  competizione nello stesso blocco aprono due giornate). L'etichetta riporta date e numeri ufficiali (`Giornata del 6-9
  nov 2026 · Girone A g.13, Girone B g.13, Girone C g.13, Serie B g.11`). **I recuperi non spostano mai una giornata**:
  rinvii, recuperi di turni già giocati e partite riprogrammate a più di 7 giorni dal loro turno sono elencati a parte
  (Recuperi e rinvii). La tabella del tab *Giornata*, il pannello valore (con la casella «Mostra tutte le giornate») e
  le selezioni usano di default la giornata corrente; la scelta nel tab *Selezioni di giornata* vale per tutti e tre i
  tab (`st.session_state["matchday"]`). Gli export CSV restano completi.
- **Selezioni di giornata** (tab omonimo e sezione `8) SELEZIONI DI GIORNATA` della console): per la giornata scelta,
  fra le sole partite **ancora da giocare** (calcio d'inizio non ancora passato: una partita iniziata esce dalle liste,
  dal pannello valore e dall'export delle previsioni), le **5 partite più probabili da X** e le **12 partite più probabili da GG**
  fra le competizioni della sezione (i tre gironi di Serie C insieme, la Serie B, oppure i nove gironi di Serie D insieme),
  ordinate solo per probabilità del modello, con quota fair, miglior quota book, bookmaker, EV e flag quando una
  quota esiste, più una nota («lambda e mu vicini», «pochi gol attesi», «xG attese alte», «entrambi gli attacchi sopra
  1.2 gol attesi»). Le dimensioni si cambiano con i controlli *N risultati X* / *N partite GG* (console: `--n-draws`,
  `--n-btts`); un solo CSV scarica entrambe le liste (console: `<export>_selezioni.csv` accanto al report valore).
- **Console (`main.py`)**: `python main.py --footystats` scarica prima i dati reali con la chiave (`.env` o ambiente;
  restano in `data/processed` / `data/odds`), stampa l'esito (competizioni saltate, differenze del dizionario) e lavora
  su quei dati; `--processed <matches.csv> --fixtures <fixtures.csv> --odds <cartella>` usa CSV già raccolti; per i dati
  reali la giornata corrente è valutata adesso (o a mezzanotte di `--as-of AAAA-MM-GG`). **Senza** una delle due
  opzioni la console gira su una lega **simulata** e lo dichiara: la **prima e l'ultima riga** dell'output sono
  `DATI SIMULATI: squadre, calendari e risultati inventati. Non usare per scommettere.` e i CSV di `--export` si
  chiamano `DEMO_<nome>` (`reports/DEMO_value_report.csv`, `DEMO_value_report_selezioni.csv`, …) con la prima colonna
  `AVVISO` = `DATI SIMULATI - NON USARE PER SCOMMETTERE`. `--mode full` simulato segue
  il calendario (`--as-of`, default oggi) oppure `--rounds N` fissa le giornate giocate (es. `--rounds 30` per il
  report di recupero dei parametri); `--mode mini` è un'istantanea fissa di inizio stagione (prime 20 partite = giornate
  1-3, 4ª giornata da prevedere). Opzioni comuni: `--history 0|1|2`, `--season`. La sezione `4) PREVISIONI PROSSIMO
  TURNO` stampa la giornata corrente come **tabella compatta** (competizione, data, calcio d'inizio, squadre, stato, λ,
  μ, P1/PX/P2, O2.5, GG, Fair X, Book X, EV max, Flag) con il conteggio di partite giocate / da giocare / in corso /
  rinviate (la sezione 5 esclude le quote delle partite già iniziate); con
  `--export` scrive anche la tabella completa in `<export>_giornata.csv` e i risultati giocati della stagione corrente
  (con P esito e Sorpresa) in `<export>_risultati.csv`, accanto al report valore e alle selezioni.
  **Sezioni a console**: `--section` (default `all`) accetta `all`, il nome o la sigla di una sezione o un elenco
  separato da virgole (`"Serie C"`, `serie-b`, `b`, `c,d`) e gli alias `bc` (Serie C e Serie B) e `d`. Le fasi 1-2
  (dataset, stima per paese) sono uniche; le fasi 3-8 (ranking per competizione, tabella giornata, valore, alert,
  selezioni 5 X / 12 GG calcolate dentro la sezione) sono stampate **una volta per sezione**, ognuna con il proprio
  banner (`SEZIONE SERIE D -- Perimetro: Serie D 2026/27 Gironi A-I (162 squadre) · …`). Corrono solo le sezioni con
  squadre; con più sezioni i CSV portano la sigla della sezione (`<export>_c_giornata.csv`, `<export>_b_giornata.csv`,
  `<export>_d_selezioni.csv`, …), il report valore in `--export` copre sempre tutto il perimetro. `--serie-d-demo`
  (solo `--mode full` simulato) aggiunge un dizionario sintetico di Serie D (9 gironi × 18 squadre, `Squadra DA 01` …).
  **AFFIDABILITÀ MODELLO** (dopo la stima; di default con i dati reali, `--backtest` / `--no-backtest`,
  `--backtest-seasons 1|2|3`): il backtest walk-forward con i punteggi aggregati per sezione e mercato (il modello
  accanto al mercato sulle stesse partite con quote), il track record di liste, multiple top-k delle liste per sezione
  e schedine e la decisione sulla ricalibrazione, applicata alle probabilità delle fasi seguenti solo se
  migliora fuori campione (`--no-recalibration` per le probabilità originali); CSV `<export>_affidabilita.csv`.
  **SCHEDINE** (alla fine, `--slips` / `--no-slips`, lunghezze `--slip-x 5`, `--slip-gg 10`, `--slip-mixed 6`): le tre
  schedine della prossima giornata delle sezioni stampate; CSV `<export>_schedine.csv`.
- **Legenda** (tab omonimo, ultimo): il glossario qui sotto, generato da `src/dashboard/legend.py`
  (`legend_markdown()`), con un esempio svolto («Come leggere una riga del pannello valore») su una riga del pannello
  valore così come è filtrato in quel momento (giornata scelta o tutte, competizione della sidebar, soglia EV): la
  prima riga senza flag `LONGSHOT` / `SUSPICIOUS_EDGE`, perché una quota lunga o un EV sospetto insegnano poco; senza
  righe l'esempio è costruito e dice perché (nessuna quota, oppure nessuna selezione di valore con i filtri correnti).
  La console rimanda a questa sezione (`Legenda: tab Legenda della dashboard oppure README`).

<!-- Blocco generato da src/dashboard/legend.py (legend_markdown()): non modificare a mano, rigenerare con
     python -c "from src.dashboard.legend import legend_markdown; print(legend_markdown())"
     (tests/test_picks.py verifica che README e codice coincidano). -->
<!-- legend:start -->
## Legenda: come leggere i numeri

Glossario di tutti i numeri mostrati da dashboard, console ed export, scritto per chi li usa per pianificare le scommesse. Le formule sono quelle del codice (`src/engine/model.py`, `src/engine/value.py`, `src/engine/picks.py`, `src/engine/tables.py`).

### Parametri del modello (tab Classifica Forza)

- **α (alfa) · Forza offensiva** — quanto una squadra segna più o meno della media della sua competizione, su scala logaritmica: 0 = attacco medio del girone, +0.3 ≈ 35% di gol in più (exp(0.3) = 1.35), −0.3 ≈ 26% in meno. È centrata per competizione (la somma degli α di un girone è zero): confronta gli α solo dentro la stessa competizione, un +0.2 in Serie B e un +0.2 nel girone C non misurano la stessa cosa.
- **β (beta) · Solidità difensiva** — quanto una squadra fa segnare poco gli avversari: più alto = subisce meno. Entra nei gol attesi dell'avversario con il segno meno (λ = exp(α_casa − β_ospite + γ)). A differenza di α, β NON è centrata su zero: la media dei β di una competizione fa da livello base dei gol di quel campionato, quindi il valore assoluto non dice nulla e β = 0 non significa "difesa media". Leggila in confronto: 0.3 di β in più di un'altra squadra della stessa competizione ≈ 26% di gol subiti in meno (exp(−0.3) = 0.74); sopra la media dei β del girone = difesa migliore della media, sotto = peggiore.
- **γ (gamma) · Vantaggio casa** — fattore campo stimato per ogni competizione (uno per girone, uno per la Serie B), non per squadra: i gol attesi della squadra di casa sono moltiplicati per exp(γ). γ = 0.25 ⇒ ×1.28, cioè circa il 28% di gol in più di quanti ne farebbe in campo neutro. Valori tipici 0.15-0.35.
- **Indice Attacco / Indice Difesa** — exp(α) ed exp(β): gli stessi parametri in scala moltiplicativa. Indice Attacco 1.00 = attacco medio del girone, 1.20 = +20%, 0.85 = −15%. L'Indice Difesa non ha 1.00 come media (β non è centrata): si legge per rapporto, se una squadra ha indice difesa 1.20 volte quello di un'altra della stessa competizione, a parità di avversario subisce circa 1/1.20 ≈ 17% di gol in meno.
- **ρ (rho) · Correlazione bassi punteggi** — correzione di Dixon-Coles ai risultati 0-0, 1-0, 0-1 e 1-1. Con ρ negativo (tipico, tra −0.05 e −0.15) il modello prevede più 0-0 e 1-1 (e meno 1-0 / 0-1) di due Poisson indipendenti: i pareggi a basso punteggio salgono, con ρ = −0.10 la X guadagna circa 2-3 punti percentuali. ρ = 0 equivale a gol di casa e ospiti indipendenti. È unico per tutto il perimetro.
- **ξ (xi) · Decadimento temporale** — peso delle partite nel tempo: ogni gara pesa exp(−ξ·giorni trascorsi). Con ξ = 0.0018 una partita di un anno fa vale circa il 52% di una di oggi, una di due anni fa il 27%: lo storico delle tre stagioni aiuta la stima ma conta sempre meno della stagione in corso. ξ = 0 (dati simulati) pesa tutto allo stesso modo.
- **Prior gaussiana · Stima MAP** — a inizio stagione ci sono poche partite e la stima «pura» (massima verosimiglianza, MLE) esagera: una squadra con due vittorie 4-0 sembrerebbe imbattibile. La prior tira i parametri verso valori normali (α verso 0 e β verso la media del suo girone, entrambi con deviazione standard 0.30, γ verso 0.25 ± 0.15, ρ verso 0 ± 0.10) e la stima diventa «MAP» (massimo a posteriori). Si attiva da sola finché la competizione meno coperta ha meno di 15 partite giocate per squadra nella stagione corrente (con 20 squadre per girone: fino alla 14ª giornata inclusa); il badge della sidebar dice se è attiva. Con una stagione intera i dati dominano e MAP ≈ MLE.

### Previsione della partita (tab Giornata)

- **λ (lambda) · Gol attesi casa** — numero medio di gol che il modello si aspetta dalla squadra di casa: λ = exp(α_casa − β_ospite + γ). Un λ di 1.60 non vuol dire «segna 1 o 2 gol», ma che in media su molte partite così ne segnerebbe 1.6: la distribuzione dei singoli punteggi è nella matrice.
- **μ (mu) · Gol attesi ospiti** — lo stesso per la squadra in trasferta: μ = exp(α_ospite − β_casa), senza fattore campo. λ + μ è il totale gol atteso della partita (sopra 2.8 = gara da Over, sotto 2.2 = gara chiusa).
- **Matrice dei punteggi (heatmap)** — probabilità di ogni risultato esatto, gol casa sulle righe e gol ospiti sulle colonne, da 0 a 10 gol per squadra (la vista compatta accorpa la coda in «N+»). Le celle sommano al 100%: ogni mercato è solo una somma di celle (1 = celle sotto la diagonale, X = diagonale, Over 2.5 = celle con somma ≥ 3, GG = celle con riga ≥ 1 e colonna ≥ 1).
- **Risultato più probabile** — la cella più alta della matrice, con la sua probabilità. Di norma vale il 10-14%: è il punteggio singolo più frequente, non «il risultato previsto». Un 1-1 al 12% convive con un 1 (vittoria casa) al 45%.
- **Probabilità dei mercati · 1X2, Over/Under, GG/NG** — per ogni mercato gli esiti sono esaustivi e si escludono a vicenda, quindi sommano al 100% (1 + X + 2, Over 2.5 + Under 2.5, GG + NG). Over 1.5 / 2.5 / 3.5 contano i gol totali (strettamente più di 1, 2, 3); GG = «entrambe le squadre segnano», NG = almeno una resta a zero.

### Tabella giornata e risultati

- **Tabella giornata (tab Giornata)** — una riga per **ogni partita** della giornata ufficiale di ogni competizione (tutte le competizioni della sezione o quella della sidebar): le partite già giocate con il risultato finale, quelle da giocare con orario e previsione del modello (forza α/β, forma recente, gol attesi, risultato più probabile, probabilità e quote fair dei mercati, miglior quota book per esito, EV migliore e flag), quelle già iniziate senza risultato nei dati e quelle rinviate. Tutto il turno si legge in un colpo d'occhio, anche quando una parte è già stata giocata il venerdì; cliccando una partita da giocare (o scegliendola nel menù «Partita») sotto compaiono la matrice dei punteggi e i mercati di quella partita. «Scarica tabella giornata (CSV)» ha le stesse colonne più match_id.
- **Giornata corrente** — la giornata ufficiale (game_week di FootyStats) che si sta giocando o che sta per iniziare, competizione per competizione: la più bassa che ha ancora almeno una partita in programma (calcio d'inizio non più vecchio di 3 ore: una partita iniziata da meno di 3 ore tiene la giornata corrente) e in cui almeno metà delle partite con una data non è ancora giocata (le rinviate senza nuova data non contano), oppure che è in corso: qualche partita giocata o iniziata e una partita rimasta nello stesso blocco della settimana (venerdì-lunedì o martedì-giovedì) della maggior parte del turno. Così un turno con due partite giocate il venerdì e otto nel fine settimana resta la giornata corrente, con tutte e dieci le partite; un turno di cui resta solo un recupero infrasettimanale no (anche se altre partite sono rinviate): la giornata corrente è la successiva e il recupero va in «Recuperi e rinvii».
- **Calcio d'inizio** — giorno e ora della partita in ora italiana, «sab 07/11 14:30» (FootyStats dà l'orario in UTC: il passaggio fra ora legale e solare è già considerato). Le partite giocate e i file senza orario mostrano solo il giorno, «ven 06/11».
- **Stato · giocata / da giocare / in corso / rinviata** — «giocata 2-1» = partita già disputata, con il risultato finale e le colonne del modello vuote (non si prevede il passato); «da giocare» = calcio d'inizio ancora da venire, con la previsione: sono le sole partite delle selezioni e del pannello valore; «in corso / risultato in attesa» = il calcio d'inizio è passato ma il risultato non è ancora nei dati (partita in corso, FootyStats non l'ha ancora pubblicato o i dati locali sono di prima della fine): resta nel suo turno, senza previsione e senza quote, perché una giocata pre-partita su una partita iniziata non ha senso; «rinviata» = nessun risultato 48 ore dopo il calcio d'inizio e recupero non ancora fissato: niente previsione finché non ha una nuova data.
- **Giornata n. (una competizione)** — con una sola competizione scelta nella sidebar compare il menù «Giornata n.»: mostra qualsiasi giornata ufficiale di quella competizione, passata o futura, con tutte le sue partite; «(corrente)» indica la giornata corrente.
- **Recuperi e rinvii** — l'elenco, sotto la tabella, delle partite rinviate e dei recuperi ancora da giocare fissati fuori dalla giornata corrente della loro competizione (per esempio un recupero del mercoledì di un turno già giocato), con la nuova data e, per una partita spostata davvero, la data originale (il giorno in cui si giocava il resto del turno). Non spostano mai la giornata e non entrano nelle selezioni di giornata; per quelle in programma sono indicate le probabilità 1X2 e GG del modello. Una partita già iniziata senza risultato non è un recupero: resta nel suo turno come «in corso / risultato in attesa».
- **Forma (Casa / Ospiti)** — gli esiti delle ultime 5 partite giocate dalla squadra nella stagione corrente, casa e trasferta insieme, dalla più recente alla più vecchia: V vittoria, N pareggio (nullo), P sconfitta. «V-V-N-P-N» = ha vinto le ultime due. Meno di 5 lettere = ha giocato meno di 5 partite; vuota = nessuna partita giocata.
- **Pt (5)** — punti raccolti nelle stesse ultime 5 partite (3 per vittoria, 1 per pareggio): da 0 a 15; il numero fra parentesi nel nome della colonna è la finestra. Non è la classifica: misura il momento di forma, che il modello non usa direttamente (α e β pesano tutta la stagione, più lo storico).
- **GF/GS (5)** — gol fatti / gol subiti nelle ultime 5 partite: «9/4» = 9 segnati e 4 subiti. Confrontalo con λ e μ: una squadra che segna molto più di quanto il modello si aspetti sta sovra-performando (o il modello la sottovaluta), una che subisce poco con tanti xG contro è candidata alla trappola di regressione.
- **xG F/S (3)** — expected goals prodotti / concessi nelle ultime 3 partite: «4.1/2.6» = 4.1 xG creati e 2.6 subiti; «-» quando il dato manca anche in una sola delle tre gare. La finestra di 3 è la stessa del filtro REGRESSION_TRAP: xG subiti alti con pochi gol subiti (GF/GS) = difesa in sovra-rendimento.
- **Tot. att.** — λ + μ, i gol totali attesi della partita secondo il modello: sopra 2.8 gara da Over, sotto 2.2 gara chiusa. È la stessa somma che guida le probabilità Over/Under e GG delle colonne accanto.
- **Ris. prob.** — il risultato esatto più probabile con la sua probabilità, «1-0 (14%)»: la cella più alta della matrice dei punteggi, di norma fra il 10 e il 14%. Non è «il pronostico»: un 1-1 al 13% convive con una vittoria casa al 45%.
- **P1 / PX / P2, O1.5 / O2.5 / O3.5, GG (%) · Fair · Book** — le probabilità del modello per vittoria casa, pareggio, vittoria ospite, Over 1.5 / 2.5 / 3.5 gol totali ed entrambe a segno; Fair 1 / X / 2 / O2.5 / GG le quote fair corrispondenti (1/P); Book 1 / X / 2 / O2.5 / GG la miglior quota fra i bookmaker presenti nei dati per quell'esito (vuota se nessuno lo quota).
- **EV max (%) · Miglior EV** — il valore atteso più alto fra tutte le selezioni quotate della partita (ogni mercato, ogni bookmaker) e la selezione che lo realizza: «X @3.40» = il pareggio alla quota 3.40. È il punto da cui partire per quella partita; il dettaglio (bookmaker, edge, stake) è nel pannello Opportunità di Valore. Vuoti senza quote.
- **Flag (tabella giornata)** — l'unione dei flag di tutte le selezioni quotate della partita (VALUE, LONGSHOT, SUSPICIOUS_EDGE: vedi la sezione Flag) più REGRESSION_TRAP e TRAP:<squadra> quando una delle due squadre è in trappola di regressione, anche senza quote: il segnale sulla difesa vale a prescindere dal prezzo.
- **Tabella risultati (tab Risultati)** — una riga per ogni partita già giocata della stagione e della competizione scelte, filtrabile per squadra (il menù offre chi ha giocato quella stagione in quella competizione: nelle stagioni passate anche le squadre non più nel perimetro, mai una squadra senza partite): risultato, esito 1X2, tiri, tiri in porta e angoli casa/ospiti («12/7», «-» se il dato manca), xG di entrambe, più la lettura del modello (λ, μ, P esito, Sorpresa). Le colonne si ordinano cliccando l'intestazione; «Scarica risultati (CSV)» la esporta.
- **P esito (%)** — la probabilità che il modello assegna all'esito 1X2 effettivamente avvenuto, calcolata con λ e μ di quella partita (la stessa matrice dei punteggi). Attenzione: il modello è stimato su tutte le partite giocate, compresa quella stessa, quindi è una lettura a posteriori, non una previsione fatta prima del calcio d'inizio; serve a vedere quali risultati il modello spiega e quali no.
- **Sorpresa** — «sorpresa» quando P esito è sotto il 20% (un esito che il modello riteneva improbabile), «atteso» sopra il 50% (l'esito che il modello indicava come favorito), vuoto nel mezzo. Molte sorprese di fila per una squadra = qualcosa che il modello non vede (infortuni, cambio allenatore, calendario) oppure semplice varianza: è un invito a guardare, non un segnale di scommessa.

### Quote e valore (tab Opportunità di Valore)

- **Quota fair** — 1 / probabilità del modello: la quota a cui la scommessa sarebbe alla pari nel lungo periodo. P = 40% ⇒ fair 2.50. Se il bookmaker paga più della fair la selezione ha valore, se paga meno no.
- **Quota book** — la quota decimale del bookmaker (colonna Bookmaker = da dove viene: FootyStats, SimBook del simulatore o il valore della colonna bookmaker del CSV in data/odds). Con più book sulla stessa selezione si usa la quota più alta.
- **Probabilità implicita e de-vig** — 1 / quota è quello che il book «dice» di credere, ma le implicite di un mercato sommano a più del 100% (es. 105%): il 5% in eccesso è l'allibramento (vig, margine). Il de-vig lo rimuove riproporzionando le implicite in modo che sommino al 100% (metodo proporzionale di default; power e Shin disponibili). Serve a confrontare il modello con l'opinione del book depurata dal margine.
- **EV (%) · Valore atteso** — EV = P_modello × Quota book − 1. Esempio: 40% × 2.70 − 1 = +8%: per ogni euro puntato ci si aspetta di guadagnare in media 8 centesimi **nel lungo periodo**, su molte scommesse simili; sulla singola partita si vince o si perde. EV negativo = scommessa in perdita attesa anche se vince stavolta.
- **Edge** — P_modello − P_implicita de-viggata: di quanti punti percentuali il modello è più ottimista del bookmaker su quell'esito. Misura il disaccordo; l'EV misura quanto rende. Un edge grande con quota bassa vale poco; un edge piccolo su quota alta può valere molto.
- **Soglia EV (3,5%)** — sotto questa soglia la selezione non è considerata di valore: l'EV calcolato ha un errore (quello del modello) e un margine di sicurezza evita di puntare su differenze che sono solo rumore. Il cursore nella sidebar la cambia; alzarla riduce le puntate ma le rende più selettive.

### Dimensionamento della puntata (colonna Stake)

- **Kelly pieno · f* = (p·b − q) / b** — frazione del bankroll che massimizza la crescita a lungo termine, con p = P_modello, q = 1 − p e b = quota − 1 (vincita netta per 1 euro). Esempio: p = 40%, quota 2.70 ⇒ b = 1.70, f* = (0.40×1.70 − 0.60) / 1.70 = 4.7% del bankroll. Se p·b < q (EV negativo) f* è zero: non si punta.
- **Kelly frazionario (1/8 o 1/4)** — si punta solo una frazione di f*, perché p è una stima: il Kelly pieno con una probabilità sbagliata porta a scommesse troppo grosse e a oscillazioni violente del bankroll. 1/8 (default) è prudente e adatto a un modello nuovo; 1/4 è già aggressivo. La frazione dimezza (o riduce) la varianza molto più di quanto riduca la crescita attesa.
- **Stake (unità)** — la puntata consigliata in unità di bankroll: con bankroll 100 unità, 1 unità = 1% del capitale dedicato alle scommesse. Stake = f* × frazione di Kelly × bankroll, arrotondato a 0,1 unità. Nell'esempio sopra con 1/8 Kelly e 100 unità: 4.7% × 1/8 × 100 = 0,6 unità.
- **Limite per puntata** — lo stake è comunque tappato a 3 unità (il 3% con il bankroll standard di 100 unità; il tappo resta 3 unità anche cambiando il bankroll nella sidebar): oltre, l'eventuale errore del modello costa troppo. Uno stake «3.0» dice che il Kelly avrebbe chiesto di più.
- **Stake 0 unità** — la selezione o ha EV sotto soglia (nessun valore) o la puntata arrotondata scende sotto 0,1 unità: non vale la pena piazzarla. Il pannello valore mostra solo righe sopra soglia, ma l'export CSV completo elenca anche quelle a zero.

### Flag (colonna Flag)

- **VALUE** — EV ≥ soglia: la selezione è di valore; lo stake è di norma > 0 ma può arrotondare a 0 (sotto 0,1 unità) sulle quote molto alte, dove il Kelly frazionario chiede una puntata minuscola. È il flag che si cerca.
- **REGRESSION_TRAP** — la selezione è Over o GG contro una squadra in «trappola di regressione»: nelle ultime 3 gare ha subito al massimo 1 gol (GA ≤ 1) ma gli avversari hanno prodotto almeno 3.5 xG cumulati (xGA ≥ 3.5). La difesa sta rendendo oltre il lecito e il mercato, che guarda i gol subiti, sottoprezza i gol contro di lei. Come usarla: l'Over e il GG contro quella squadra sono il lato giusto; i suoi Under / No Gol sono da evitare. Il flag è informativo: non cambia l'EV calcolato.
- **TRAP:<squadra>** — accompagna REGRESSION_TRAP e dice quale squadra della partita è in trappola (possono essere entrambe).
- **LONGSHOT** — quota book sopra 6: anche con EV positivo è una scommessa che perde la gran parte delle volte, con lunghe serie negative. Il Kelly frazionario la dimensiona già piccola; sapere che lo è aiuta a non demoralizzarsi (o a saltarla, se la varianza non piace).
- **SUSPICIOUS_EDGE** — EV sopra il 25%: un vantaggio così grande sui bookmaker è quasi sempre un errore di dati o una quota stantia (partita rinviata, formazione nota al book e non al modello, quota pre-apertura, nome squadra sbagliato), non una miniera d'oro. Verifica la quota sul sito del book e la data della partita prima di puntare.

### Selezioni di giornata (tab Selezioni di giornata)

- **Giornata** — la giornata corrente di ogni competizione della sezione scelta (i tre gironi di Serie C, la Serie B, i nove gironi di Serie D o un campionato abilitato), messe insieme qualunque sia il loro numero (gironi di Serie C alla 13ª, D-A alla 12ª) e con le partite già giocate incluse; le giornate successive raggruppano i turni seguenti giocati nello stesso blocco della settimana calcistica: fine settimana (venerdì-lunedì) oppure turno infrasettimanale (martedì-giovedì). Una competizione che in quel blocco non gioca (Serie B in sosta per le nazionali) manca dall'etichetta. Le liste X e GG considerano solo le partite ancora da giocare (calcio d'inizio non ancora passato: una partita iniziata esce dalle liste e dal pannello valore); rinvii e recuperi fuori turno non spostano la giornata (sono in «Recuperi e rinvii»). Ogni sezione raggruppa le giornate dal proprio calendario: le 34 giornate di Serie D non spostano quelle di Serie C o di Serie B.
- **Le 5 partite più probabili da X** — fra tutte le partite della giornata della sezione scelta (i tre gironi di Serie C insieme, la Serie B, oppure i nove gironi di Serie D insieme), le cinque con la probabilità di pareggio più alta secondo il modello. Ordinate **solo per probabilità**, non per valore: una X al 33% a quota 2.90 (EV −4%) sta in lista anche se non conviene. Nota «lambda e mu vicini» = squadre equilibrate, «pochi gol attesi» = gara chiusa: entrambe spingono il pareggio.
- **Le 12 partite più probabili da GG** — le dodici partite della giornata, su tutte le competizioni della sezione, con la probabilità più alta che entrambe le squadre segnino. Stesso criterio: solo probabilità del modello. Nota «xG attese alte» = tanti gol attesi in totale, «entrambi gli attacchi sopra 1.2 gol attesi» = nessuna delle due dovrebbe restare a secco.
- **«Più probabile» non è «conveniente»** — la probabilità dice quante volte l'esito accade, la quota dice quanto paga: un GG al 65% pagato 1.45 (fair 1.54) perde soldi nel lungo periodo, un GG al 55% pagato 2.00 (fair 1.82) li guadagna. Le colonne Quota Fair, Quota Book ed EV sono lì per questo: incrocia sempre la lista con la quota e con l'EV prima di puntare, e dà la precedenza alle righe che compaiono anche nel pannello valore.
- **Nessuna quota in lista** — se Quota Book, Bookmaker ed EV sono vuoti nessun bookmaker nei dati quota quella selezione: la lista resta utile per sapere dove guardare, il prezzo va cercato a mano.

### Schedine (tab Schedine)

- **Schedina X** — le 5 partite della giornata (5 di default, regolabile) con la probabilità di pareggio più alta secondo il modello, fra le sezioni scelte e solo fra le partite non ancora iniziate; una partita entra al massimo una volta per schedina. Stesso criterio della lista X: probabilità, non valore.
- **Schedina GG** — le 10 partite (default) con la probabilità più alta che entrambe le squadre segnino.
- **Schedina mista** — per ogni partita il segno più probabile fra 1, 2, GG e Over 2.5 secondo il modello, poi le 6 partite (da 3 a 10) in cui quel segno è più probabile: integra le partite con la probabilità più alta per il loro segno. Con «Solo partite quotate» il segno si sceglie fra quelli che hanno una quota; «P minima» esclude i segni sotto la soglia.
- **Probabilità della schedina** — il prodotto delle probabilità delle singole partite, nell'ipotesi che le partite siano indipendenti (un'ipotesi, non un fatto: due gare dello stesso girone con lo stesso arbitro o lo stesso meteo non lo sono del tutto). Cinque X al 35% fanno 0,35⁵ ≈ 0,5%: una schedina su 190. Quota fair = 1 / probabilità.
- **Quota book ed EV della schedina** — la quota book è il prodotto delle migliori quote delle partite, solo quando tutte sono quotate; EV = probabilità × quota book − 1. Se una partita non ha quota, quota book ed EV restano vuoti.
- **Margine composto** — una multipla moltiplica il margine del bookmaker: con il 5% per partita, 5 partite cedono circa il 22% della vincita equa e 10 partite quasi il 39%. Per questo una schedina lunga ha quasi sempre EV negativo anche quando ogni partita è «probabile»: le singole di valore (tab Opportunità) sono il modo più efficiente di sfruttare un vantaggio.
- **Partite vincenti attese** — la somma delle probabilità delle partite: quante ne indovini in media. Con 10 GG al 62% sono circa 6, e la schedina intera si vince solo se sono 10.
- **Stake suggerito della schedina** — Kelly frazionario sulla schedina intera (la frazione della sidebar), con un tetto di 1 unità; «non consigliata» quando l'EV è negativo o manca una quota. È volutamente piccolo: la varianza di una multipla è enorme.
- **Track record delle schedine** — le stesse tre schedine ricostruite per ogni giornata passata con le probabilità fuori campione del backtest: giornate, schedine vinte contro quelle attese, percentuale di gambe vinte e ROI alle quote reali quando ci sono, con intervallo di confidenza al 95%.

### Affidabilità del modello (tab Affidabilità modello)

- **Backtest walk-forward** — ogni giornata passata (blocco della settimana calcistica) è prevista da un modello stimato solo sulle partite giocate prima del suo calcio d'inizio, con lo stesso decadimento, la stessa prior e lo storico di produzione. Sono le probabilità che il modello avrebbe pubblicato allora: la stima di produzione ha già visto quei risultati e non va mai usata per giudicare la precisione. Di default si valutano la stagione corrente e la precedente.
- **Log loss** — −ln della probabilità data all'esito avvenuto, in media: più bassa è meglio. Un modello che dà sempre 1/3 a 1, X e 2 fa 1,099; le frequenze di base di un campionato intorno a 1,07-1,09; un buon modello e i bookmaker fra 0,98 e 1,03 sull'1X2. Punisce molto le sicurezze sbagliate (un 90% che non esce costa più di dieci 55% sbagliati).
- **Brier** — errore quadratico medio fra probabilità ed esito (1X2 da 0 a 2, mercati a due esiti da 0 a 1): più basso è meglio, meno severo della log loss con gli errori rari.
- **Accuratezza (%)** — quante volte l'esito più probabile secondo il modello è avvenuto. Utile ma ingannevole: sull'1X2 il segno 1 vince spesso anche senza modello; conta che le probabilità siano giuste, non solo il favorito.
- **Base e Skill vs base** — la base sono le frequenze di 1/X/2, GG e Over del campionato nei 365 giorni precedenti (quello che sa chi non ha un modello). Skill vs base = 1 − log loss modello / log loss base: positiva = il modello batte le frequenze di base. È il primo esame da superare.
- **Mercato e Skill vs mercato** — le probabilità del bookmaker senza margine (quote pre-partita delle partite giocate, dallo storico FootyStats), valutate solo sulle «Partite con quote». Il modello va confrontato con il mercato sulle stesse partite: le colonne «Log loss modello (partite con quote)» e «Brier modello (partite con quote)» e il riquadro «Modello vs mercato»; «Log loss modello» copre tutte le partite valutate, anche quelle senza quote (la copertura cambia da campionato a campionato), e non è confrontabile con il mercato. Skill vs mercato = 1 − log loss modello / log loss mercato sulle stesse partite: battere il mercato è raro; un valore negativo di poco è normale e non impedisce di trovare singole di valore, uno molto negativo invita alla prudenza sugli EV.
- **Grafico di affidabilità** — le partite sono raggruppate per probabilità prevista (fasce di 10 punti): in orizzontale la probabilità media della fascia, in verticale quante volte l'esito è avvenuto, con la barra dell'intervallo di Wilson al 95%. Sulla diagonale il modello è tarato; sopra la diagonale è troppo prudente (l'esito accade più spesso), sotto è troppo sicuro. Una fascia con poche partite ha la barra lunga: non trarne conclusioni.
- **Hit rate, IC 95% e ROI** — hit rate = selezioni vinte / selezioni, con l'intervallo di Wilson al 95%; va confrontato con la «P media» del modello (se coincidono, il modello è tarato). ROI = profitto / puntate a puntata fissa sulla miglior quota pre-partita, con un intervallo al 95% (approssimazione normale): finché l'intervallo contiene lo zero il guadagno non si distingue dalla fortuna.
- **Multiple top-k delle liste** — le prime k partite (da 2 a 5) della lista X o GG di ogni giornata passata giocate insieme, per sezione o per tutte le sezioni: è la giocata «le prime 2 GG della Serie C» ripetuta su ogni giornata. Giornate, multiple vinte contro quelle attese (somma dei prodotti delle probabilità, indipendenza ipotizzata), win rate con intervallo di Wilson e ROI alle quote reali; il dettaglio giornata per giornata è in «Liste giornata per giornata».
- **Poche vincite non sono una prova** — una domenica con la schedina vinta (100 € con 5 €) o due GG azzeccati capitano anche senza alcun vantaggio: contano il numero di giornate e di partite valutate e gli intervalli di confidenza, non l'aneddoto.
- **Ricalibrazione validata** — imparare dal differenziale fra previsioni ed esiti: per ogni mercato una piccola correzione (Platt su GG/NG e Over/Under, temperatura multinomiale sull'1X2, con penalità verso «nessuna correzione»; parametri propri per Serie B, C o D solo con abbastanza partite) è stimata sulle giornate più vecchie del backtest e giudicata su quelle più recenti. Si applica solo ai mercati in cui la log loss fuori campione migliora almeno dello 0,5%; altrimenti restano le probabilità originali. «Calibrazione applicata: sì/no, miglioramento log loss X%» lo dice; l'interruttore «Usa probabilità ricalibrate» la accende o la spegne (è reversibile: il modello non viene toccato) e la tabella Giornata mostra accanto la probabilità originale («orig.»).
- **Ottimizzazione di xi e prior** — facoltativa: ripete il backtest della stagione corrente per 9 combinazioni di decadimento (ξ 0,001 / 0,0018 / 0,003) e forza della prior. La combinazione è scelta sulle giornate più vecchie e confermata sul 30% più recente, che la scelta non ha visto: si può applicare con l'interruttore dedicato solo se lì batte la configurazione standard di almeno lo 0,5% (la soglia della ricalibrazione), altrimenti il guadagno è rumore. Quando è applicata, la tab Affidabilità avvisa che le giornate su cui è stata scelta non sono più del tutto fuori campione. Si torna indietro spegnendo l'interruttore.

### Dati e limiti

- **Dati simulati vs reali** — la dashboard mostra solo dati reali FootyStats, scaricati con la tua chiave: il badge della sidebar dice «Dati reali FootyStats». Senza dati non mostra numeri ma una pagina iniziale che spiega cosa manca e come configurare la chiave. La lega simulata (squadre vere, calendari, risultati e quote del bookmaker fittizio SimBook inventati) esiste solo come demo esplicita, avviata con SERIEC_DEMO=1: allora ogni pagina ha in cima l'avviso rosso «DATI SIMULATI: squadre, calendari e risultati inventati. Non usare per scommettere.» e il badge dice «Demo»; ogni CSV scaricato dalla demo si chiama DEMO_... e ha come prima colonna AVVISO = «DATI SIMULATI - NON USARE PER SCOMMETTERE». **Non usare mai la demo per scommettere.**
- **Aggiornamento dei dati FootyStats** — all'avvio, se la chiave è configurata e i dati locali mancano o hanno più di 6 ore, la dashboard li scarica di nuovo da FootyStats (partite delle ultime 3 stagioni, calendario con orari e rinvii, quote) e allinea il dizionario squadre; la sidebar dice «Dati FootyStats aggiornati alle HH:MM del gg/mm» e il pulsante «Aggiorna dati ora» li riscarica subito. Una competizione che non si può scaricare (lega non scelta sull'account, errore di rete, anche solo per la stagione in corso) è segnalata con un avviso e tiene i dati precedenti; le altre si aggiornano. Finché i dati restano più vecchi di 6 ore (chiave mancante, aggiornamento non riuscito) un avviso giallo in cima alla pagina ne dice l'età e il motivo, e un altro avviso resta per ogni competizione ferma ai dati di un download precedente. Le differenze del dizionario squadre sono in «Dettagli pipeline».
- **Chiave FootyStats e dati locali** — la chiave non va mai pubblicata: in locale sta nel file .env (FOOTYSTATS_API_KEY=...), su Streamlit Cloud in App settings → Secrets (FOOTYSTATS_API_KEY = "..."). I dati scaricati restano solo in data/processed e data/odds sul tuo computer o sul server dell'app: non sono nel repository pubblico, perché i dati FootyStats sono in licenza al tuo account.
- **Storico di 3 stagioni** — il modello è stimato sulla stagione in corso più le due precedenti; le partite vecchie contano meno (ξ). Le squadre presenti solo nello storico (retrocesse, promosse, escluse) restano nella stima come avversarie ma non compaiono in classifica, previsioni e valore.
- **Squadre storiche e perimetro** — il perimetro è il dizionario squadre della stagione (config/teams_<stagione>.json), allineato alle liste squadre di FootyStats a ogni aggiornamento: le 80 squadre di Serie C A/B/C e Serie B più le 162 dei nove gironi di Serie D. Una partita in programma con una squadra fuori perimetro o datata in un'altra stagione viene saltata e segnalata in «Dettagli pipeline».
- **Sezioni** — il selettore «Sezione» della sidebar ha una pagina per campionato: Serie C (gironi A, B, C), Serie B, Serie D (gironi A-I) e ogni campionato abilitato in config/competitions.toml. La stima Dixon-Coles è una per paese: Serie B, C e D sono stimate insieme (le squadre promosse e retrocesse collegano le divisioni; fattore campo e Σα = 0 per competizione tengono i gironi confrontabili), un campionato estero a parte. Classifica, giornate, risultati, valore, selezioni ed export mostrano solo la sezione scelta; le schedine e l'affidabilità possono unire più sezioni.
- **Abilitare Serie A o un campionato europeo** — Serie A, Premier League, La Liga, Bundesliga, Ligue 1, Eredivisie e Primeira Liga sono già configurati in config/competitions.toml con enabled = false: per accenderne uno metti enabled = true, sceglilo sull'account footystats.org («Choose leagues», il piano limita il numero di leghe) e premi «Aggiorna dati ora». La raccolta riempie le sue squadre nel dizionario e ne scarica partite, calendario e quote; il campionato diventa una sezione nuova (la Serie A entra nella stima italiana, un campionato estero ha la sua). enabled = false lo spegne di nuovo.
- **Serie D (sezione dedicata)** — nove gironi (A-I) di 18 squadre, 34 giornate ciascuno, codici DA-DI, nella stessa stima di Serie B e Serie C; classifica, giornate, risultati, valore, selezioni ed export mostrano solo la sezione e le liste X / GG sono calcolate dentro la sezione. Le liste squadre dei gironi vengono da FootyStats e si riallineano a ogni aggiornamento dei dati; finché non ci sono partite di Serie D nei dati la sezione mostra solo un avviso con il comando di raccolta (python -m src.ingestion.footystats --competitions DA DB DC DD DE DF DG DH DI --seasons 3). Le statistiche di Serie D su FootyStats sono più povere: xG e tiri spesso mancano (colonne «-», λ e μ restano stimati dai gol), senza xG il filtro REGRESSION_TRAP non può scattare, e le quote coprono meno partite (pannello valore in parte vuoto): leggi gli EV di Serie D con prudenza ancora maggiore.
- **Copertura minima** — sotto le 15 partite giocate per squadra nella competizione meno coperta la prior resta attiva; sotto le 5-6 le stime sono comunque fragili e gli EV vanno letti con sospetto (le quote dei book incorporano già informazioni che il modello non ha ancora visto).
- **Prime giornate** — nelle prime 3-4 giornate il modello si regge quasi solo sullo storico e sulla prior: usalo per orientarti (gol attesi, squadre equilibrate), punta poco o niente sulle sole indicazioni di valore e aspetta che la copertura cresca. Il filtro REGRESSION_TRAP ha bisogno di 3 gare giocate per squadra.
- **Quote e tempi** — le quote cambiano fino al calcio d'inizio: un EV calcolato su quote di ieri può non esistere più oggi. Confronta la quota presa con quella di chiusura (closing line): batterla stabilmente è il segnale più affidabile che il modello vede qualcosa.

### Regole di buon senso

- **Bankroll fisso e separato** — decidi all'inizio quanto dedicare alle scommesse, dividilo in 100 unità e dimensiona tutto in unità; non ricaricare a stagione in corso e non puntare mai più del tappo per selezione.
- **Mai inseguire le perdite** — una serie negativa è normale anche con EV positivo (con quote intorno a 2.70 si perde il 60% delle volte); raddoppiare la puntata per rifarsi distrugge il vantaggio del Kelly frazionario.
- **Campione minimo prima di fidarsi** — qualche decina di scommesse non dice nulla: servono almeno 200-300 selezioni registrate (quota presa, EV, esito) per distinguere la fortuna dal valore. Tieni un registro.
- **Confronta con le quote di chiusura** — se le quote che prendi sono in media più alte di quelle di chiusura il modello anticipa il mercato; se sono più basse stai pagando troppo, qualunque cosa dica l'EV.
<!-- legend:end -->

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