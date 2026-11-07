#!/usr/bin/env bash
# Avvio con un doppio clic (macOS / Linux): prepara Python, scarica i dati FootyStats con la tua chiave, apre la dashboard.
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

# Sorgente dati: FootyStats (API ufficiale), l'unica fonte dei dati reali: squadre, calendari con orari e
# partite rinviate, risultati e quote pre-match. La chiave si legge dalla variabile d'ambiente
# FOOTYSTATS_API_KEY o dal file .env accanto a questo script (vedi .env.example) e non viene mai stampata.
# I dati restano SOLO su questo computer (data/ è escluso da git): il repository è pubblico e i dati
# FootyStats sono concessi in licenza al titolare della chiave, quindi non vengono mai pubblicati.
# Il controllo è delegato allo stesso parser Python della CLI (variabile d'ambiente o .env, anche con BOM,
# CRLF, "export", spazi iniziali, virgolette e commenti), così lanciatore e CLI non possono essere in disaccordo.
has_footystats_key() {
  python -c 'import sys; from src.ingestion.footystats import find_api_key; from src.ingestion.harvester import load_endpoints; sys.exit(0 if find_api_key(key_env=load_endpoints().footystats.key_env) else 1)' >/dev/null 2>&1
}

if [ "${1:-}" != "--solo-dashboard" ]; then
  if ! has_footystats_key && [ ! -f .env ]; then
    echo ">> Nessuna chiave FootyStats trovata."
    echo "   Incolla qui la chiave API di FootyStats e premi Invio (footystats.org -> il tuo account -> API)."
    read -r -p "   Chiave: " FS_KEY
    FS_KEY="$(printf '%s' "$FS_KEY" | tr -d ' \t\r\n"' | tr -d "'")"
    if [ -n "$FS_KEY" ]; then
      printf 'FOOTYSTATS_API_KEY=%s\n' "$FS_KEY" > .env
      echo "   Chiave salvata nel file .env (resta solo su questo computer)."
    fi
    unset FS_KEY
  fi
  if has_footystats_key; then
    echo ">> Scarico i dati reali da FootyStats (Serie C Gironi A/B/C + Serie B + Serie D Gironi A-I, ultime 3 stagioni,"
    echo "   orari, partite rinviate e quote pre-match) e allineo il dizionario squadre a FootyStats..."
    echo "   Le competizioni la cui lega non è fra quelle scelte nel tuo account FootyStats vengono saltate (le altre sono aggiornate comunque)."
    # Nessun --competitions = tutte le competizioni di config/competitions.toml (13); la sincronizzazione del
    # dizionario config/teams_2026_27.json con /league-teams è inclusa (squadre aggiunte, spostate o rimosse).
    if ! python -m src.ingestion.footystats --seasons 3; then
      echo "ATTENZIONE: raccolta dati non riuscita (vedi messaggi sopra)."
      if [ -s data/processed/matches.csv ]; then
        echo "La dashboard usa i dati reali scaricati in precedenza (controlla la data dell'ultimo aggiornamento)."
      else
        echo "Nessun dato reale disponibile: la dashboard non mostra partite finché la raccolta FootyStats non riesce."
      fi
    fi
  else
    echo ">> Nessuna chiave FootyStats (FOOTYSTATS_API_KEY o file .env, vedi .env.example): nessun dato reale da scaricare."
    echo "   La dashboard si apre ma non mostra partite inventate: inserisci la chiave e riavvia."
  fi
fi

# Evita la domanda interattiva di benvenuto di Streamlit (richiesta email) al primo avvio.
mkdir -p "$HOME/.streamlit"
[ -f "$HOME/.streamlit/credentials.toml" ] || printf '[general]\nemail = ""\n' > "$HOME/.streamlit/credentials.toml"

echo ">> Apro la dashboard nel browser (chiudi questa finestra per fermarla)."
python -m streamlit run src/dashboard/app.py --server.headless false --browser.gatherUsageStats false
