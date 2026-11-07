#!/usr/bin/env bash
# Dopo aver raccolto i dati sul PC (avvia.command), pubblica i CSV sul repository GitHub
# così la dashboard in cloud (Streamlit Community Cloud) li vede. Richiede git configurato.
# Oltre ai CSV carica anche i dizionari squadre che la raccolta FootyStats aggiorna in locale
# (config/teams_2026_27.json: liste dei gironi di Serie D create dall'auto-init;
# config/footystats_teams.json: mappa id FootyStats -> nome canonico), altrimenti la dashboard
# in cloud vedrebbe le partite di Serie D ma gironi "non inizializzati".
set -u
cd "$(dirname "$0")"
if ! command -v git >/dev/null 2>&1 || [ ! -d .git ]; then
  echo "Serve git e una copia clonata del repository (non lo ZIP). Vedi README, sezione Quick Start."
  read -r -p "Premi Invio per chiudere." _; exit 1
fi
git add data/processed/matches.csv data/processed/fixtures.csv data/odds/*.csv 2>/dev/null
[ -f config/teams_2026_27.json ] && git add config/teams_2026_27.json
[ -f config/footystats_teams.json ] && git add config/footystats_teams.json
if git diff --cached --quiet; then echo "Nessuna variazione da pubblicare."; read -r -p "Premi Invio per chiudere." _; exit 0; fi
git commit -m "Aggiorna dati Serie C / Serie B / Serie D ($(date +%Y-%m-%d))" && git push
read -r -p "Fatto. Premi Invio per chiudere." _
