#!/usr/bin/env bash
# Avvio con un doppio clic (macOS / Linux): prepara Python, aggiorna i dati, apre la dashboard.
# Prerequisito: Python 3.12 o superiore installato (https://www.python.org/downloads/).
set -u
cd "$(dirname "$0")"

PY=""
for c in python3.13 python3.12 python3 python; do
  if command -v "$c" >/dev/null 2>&1 && "$c" -c 'import sys; sys.exit(0 if sys.version_info >= (3, 12) else 1)' 2>/dev/null; then
    PY="$c"; break
  fi
done
if [ -z "$PY" ]; then
  echo "ERRORE: serve Python 3.12 o superiore. Scaricalo da https://www.python.org/downloads/ e riprova."
  read -r -p "Premi Invio per chiudere." _; exit 1
fi

if [ ! -d .venv ]; then
  echo ">> Prima esecuzione: creo l'ambiente Python (una volta sola)..."
  "$PY" -m venv .venv || { echo "ERRORE nella creazione dell'ambiente."; read -r -p "Premi Invio per chiudere." _; exit 1; }
fi
# shellcheck disable=SC1091
source .venv/bin/activate
python -m pip install --quiet --disable-pip-version-check -r requirements.txt || {
  echo "ERRORE nell'installazione delle dipendenze (serve la connessione internet)."; read -r -p "Premi Invio per chiudere." _; exit 1; }

# Sorgente dati: FootyStats (API ufficiale, con quote pre-match) se la chiave è disponibile nella
# variabile d'ambiente FOOTYSTATS_API_KEY o nel file .env accanto a questo script (vedi .env.example),
# altrimenti SofaScore (solo da connessione residenziale). La chiave non viene mai stampata.
# Il controllo è delegato allo stesso parser Python della CLI (variabile d'ambiente o .env, anche con BOM,
# CRLF, "export", spazi iniziali, virgolette e commenti), così lanciatore e CLI non possono essere in disaccordo.
has_footystats_key() {
  python -c 'import sys; from src.ingestion.footystats import find_api_key; from src.ingestion.harvester import load_endpoints; sys.exit(0 if find_api_key(key_env=load_endpoints().footystats.key_env) else 1)' >/dev/null 2>&1
}

if [ "${1:-}" != "--solo-dashboard" ]; then
  if ! has_footystats_key && [ ! -f .env ]; then
    echo ">> Nessuna chiave FootyStats trovata."
    echo "   Incolla qui la chiave API di FootyStats e premi Invio (oppure premi solo Invio per usare SofaScore)."
    read -r -p "   Chiave: " FS_KEY
    FS_KEY="$(printf '%s' "$FS_KEY" | tr -d '[:space:]"'"'"')"
    if [ -n "$FS_KEY" ]; then
      printf 'FOOTYSTATS_API_KEY=%s\n' "$FS_KEY" > .env
      echo "   Chiave salvata nel file .env (resta solo su questo computer)."
    fi
    unset FS_KEY
  fi
  if has_footystats_key; then
    echo ">> Aggiorno i dati reali da FootyStats (Serie C Gironi A/B/C + Serie B, ultime 3 stagioni, quote pre-match)..."
    HARVEST="python -m src.ingestion.footystats --competitions A B C SB --seasons 3 --odds-out data/odds/footystats.csv"
  else
    echo ">> Nessuna chiave FootyStats (FOOTYSTATS_API_KEY o file .env, vedi .env.example): uso SofaScore."
    echo ">> Aggiorno i dati reali (ultime 3 stagioni, Gironi A/B/C). La prima volta serve qualche decina di minuti..."
    HARVEST="python -m src.ingestion.harvester --gironi A B C --seasons 3"
  fi
  if ! $HARVEST; then
    echo "ATTENZIONE: raccolta dati non riuscita (vedi messaggi sopra)."
    if [ -s data/processed/matches.csv ]; then
      echo "Uso i dati reali raccolti in precedenza."
    else
      echo "Nessun dato reale disponibile: la dashboard mostrerà la lega simulata."
    fi
  fi
fi

# Evita la domanda interattiva di benvenuto di Streamlit (richiesta email) al primo avvio.
mkdir -p "$HOME/.streamlit"
[ -f "$HOME/.streamlit/credentials.toml" ] || printf '[general]\nemail = ""\n' > "$HOME/.streamlit/credentials.toml"

echo ">> Apro la dashboard nel browser (chiudi questa finestra per fermarla)."
python -m streamlit run src/dashboard/app.py --server.headless false --browser.gatherUsageStats false
