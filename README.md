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
├── requirements.txt               # Dipendenze pinnate
├── pyproject.toml / pytest.ini    # Configurazione ruff e pytest (il progetto si esegue dalla root, non si installa)
├── config/
│   ├── teams_2025_26.json         # Dizionario canonico squadre + alias (per girone)
│   └── endpoints.toml             # Endpoint SofaScore/FotMob, retry, cache, rate-limit
├── data/
│   ├── raw/                       # Cache JSON grezzi (chiave = sha1 URL, TTL)
│   ├── processed/                 # matches.csv / fixtures.csv normalizzati
│   └── odds/                      # Snapshot quote bookmaker (CSV)
├── src/
│   ├── common/                    # Contratti condivisi
│   │   ├── schemas.py             # MatchRecord, Fixture, OddsQuote, MatchPrediction, Market
│   │   ├── teams.py               # TeamRegistry + normalizzazione nomi
│   │   ├── logging_config.py      # Logging strutturato (JSON lines)
│   │   └── paths.py
│   ├── ingestion/                 # MODULO 1
│   │   ├── harvester.py           # Client httpx asincrono, UA rotation, backoff, cache, provider
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

## 🚀 Quick Start

```bash
# Python 3.12+ (richiesto dai pin di numpy/scipy in requirements.txt)
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# 1) demo completa su mini-dataset simulato (20 partite) con output a console
python main.py

# 2) stagione simulata completa + verifica del recupero dei parametri veri
python main.py --mode full

# 3) raccolta dati reali (SofaScore) -> data/processed/
python -m src.ingestion.harvester --gironi A B C --out data/processed/matches.csv

# 4) pipeline sui dati reali con quote bookmaker in data/odds/*.csv
python main.py --processed data/processed/matches.csv --fixtures data/processed/fixtures.csv --odds data/odds

# 5) dashboard interattiva (usa data/processed se presente, altrimenti una lega simulata)
streamlit run src/dashboard/app.py
#    variabili utili: SERIEC_FORCE_SIMULATE=1  SERIEC_SIM_ROUNDS=15  SERIEC_SEED=7

# test
python -m pytest -q
```

Formato CSV quote (`data/odds/*.csv`): `match_id,bookmaker,market,selection,odds`, una riga per esito,
con `market` in `1X2 | OU1.5 | OU2.5 | OU3.5 | GG/NG` e `selection` in `1,X,2 | Over 2.5,Under 2.5 | GG,NG` ecc.

## 🧮 Modello

- **Tassi**: λ = exp(α_casa − β_ospite + γ_girone), μ = exp(α_ospite − β_casa); β alto = difesa solida.
- **Dipendenza bassi punteggi**: τ(x,y,λ,μ,ρ) di Dixon–Coles su 0-0, 1-0, 0-1, 1-1.
- **Ponderazione temporale**: w(t) = exp(−ξ·t), t in giorni dalla partita più recente (ξ default 0.0018).
- **Identificabilità**: Σα = 0 per ciascun girone (vincolo di uguaglianza SLSQP, o riparametrizzazione con L-BFGS-B).
- **Gradiente analitico** verificato numericamente nei test.
- **Prior gaussiana (stima MAP)** per campioni piccoli: `DixonColesConfig.with_gaussian_prior()` traduce deviazioni
  standard a priori (α/β 0.30, γ 0.25 ± 0.15, ρ 0.10) in penalità ridge; la ridge su β è centrata per girone così il
  livello medio dei gol non viene distorto. `main.py` (`--prior auto|on|off`) la attiva finché il girone meno coperto
  ha in media meno di 15 partite per squadra (con 20 squadre per girone: fino alla 14ª giornata inclusa); una soglia
  sul totale delle partite non basta, perché 100 gare su 60 squadre sono meno di 2 a testa e la MLE pura porta γ e ρ
  sui limiti dell'ottimizzatore (nel simulatore succede ancora in 1 fit su 10 a 12 gare per squadra, mai da 15).
  Dopo la stima la console segnala i parametri finiti sui limiti dell'ottimizzatore.
- **Mercati**: 1X2, Over/Under 1.5-2.5-3.5, Goal/No Goal dalla matrice congiunta 11×11.
- **Valore**: de-vigging proporzionale (anche power e Shin), EV = P·quota − 1, Kelly frazionario (1/8 o 1/4),
  filtro `REGRESSION_TRAP` (GA ≤ 1 nelle ultime 3 gare ma xGA ≥ 3.5 cumulato).

> Il dizionario squadre fornito è quello della stagione 2025/26; per una nuova stagione basta aggiungere
> `config/teams_<stagione>.json`. Gli endpoint SofaScore/FotMob non sono documentati e possono cambiare:
> sono tutti centralizzati in `config/endpoints.toml`.

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
    B --> C[Master Dataset Gironi A-B-C]
    C --> D[Dixon-Coles Optimizer]
    D --> E[Matrice Bivariata di Probabilità]
    F[Palinsesto Quote Bookmaker] --> G[De-Vigging Module]
    E --> H[EV Engine & Kelly Sizing]
    G --> H
    H --> I[Dashboard & Alert Report]