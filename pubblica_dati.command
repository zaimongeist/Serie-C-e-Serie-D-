#!/usr/bin/env bash
# Dopo aver raccolto i dati sul PC (avvia.command), pubblica i CSV sul repository GitHub
# così la dashboard in cloud (Streamlit Community Cloud) li vede. Richiede git configurato.
set -u
cd "$(dirname "$0")"
if ! command -v git >/dev/null 2>&1 || [ ! -d .git ]; then
  echo "Serve git e una copia clonata del repository (non lo ZIP). Vedi README, sezione Quick Start."
  read -r -p "Premi Invio per chiudere." _; exit 1
fi
git add data/processed/matches.csv data/processed/fixtures.csv data/odds/*.csv 2>/dev/null
if git diff --cached --quiet; then echo "Nessuna variazione da pubblicare."; read -r -p "Premi Invio per chiudere." _; exit 0; fi
git commit -m "Aggiorna dati Serie C ($(date +%Y-%m-%d))" && git push
read -r -p "Fatto. Premi Invio per chiudere." _
