# Serie-C-e-Serie-D-
Tool di analisi statistiche su partite della Serie C e Serie D italiane di calcio. focus su risultati, gol, Serie storiche, tattiche e notizie/aggiornamenti


# Calcio Quant Lab: Serie C (e Serie D) Analytics Engine

[![Python](https://img.shields.io/badge/Python-3.11%2B-blue.svg)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
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

Serie-C-Quant/
├── data/
│   ├── raw/                  # JSON grezzi da endpoint
│   ├── processed/            # Master CSV normalizzati per girone
│   └── odds/                 # Snapshot quote bookmaker pre-match
├── src/
│   ├── ingestion/            # Scraper headless & client API REST
│   │   ├── sofascore_client.py
│   │   └── sanitizers.py
│   ├── engine/               # Core matematico
│   │   ├── dixon_coles.py    # Ottimizzatore SLSQP/BFGS
│   │   ├── probability.py    # Convoluzione matriciale 1X2, U/O, GG/NG
│   │   └── xg_proxy.py       # Regressione logistica per match senza xG
│   ├── strategies/           # Moduli +EV e filtri anomalie
│   │   ├── value_finder.py
│   │   └── kelly.py
│   └── dashboard/            # Interfaccia grafica (Streamlit / Web)
│       └── app.py
├── tests/                    # Unit test e validazione numerica
├── .github/workflows/        # Automation cron jobs (aggiornamenti post-turno)
├── requirements.txt
└── README.md

# ⚙️ Stack Tecnologico

- **Core & Runtime:** Python 3.11+
- **Data Scraping & Networking:** `httpx` (asincrono), `requests`, `playwright` (headless fallback)
- **Calcolo Scientifico:** `numpy`, `scipy` (`optimize.minimize`), `pandas`
- **Dashboard & Visualizzazione:** `Streamlit` / `FastAPI` + `Plotly`
- **Automation:** GitHub Actions (job schedulati post-matchday)

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