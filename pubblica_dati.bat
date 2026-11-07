@echo off
REM Dopo aver raccolto i dati sul PC (avvia.bat), pubblica i CSV sul repository GitHub
REM cosi' la dashboard in cloud (Streamlit Community Cloud) li vede. Richiede git configurato.
REM Oltre ai CSV carica anche i dizionari squadre che la raccolta FootyStats aggiorna in locale
REM (config\teams_2026_27.json: liste dei gironi di Serie D create dall'auto-init;
REM config\footystats_teams.json: mappa id FootyStats -> nome canonico), altrimenti la dashboard
REM in cloud vedrebbe le partite di Serie D ma gironi "non inizializzati".
setlocal
cd /d "%~dp0"
where git >nul 2>&1 || (echo Serve git e una copia clonata del repository ^(non lo ZIP^). Vedi README. & pause & exit /b 1)
if not exist ".git" (echo Questa cartella non e' un repository git ^(hai usato lo ZIP^). Vedi README. & pause & exit /b 1)
git add data\processed\matches.csv data\processed\fixtures.csv data\odds\*.csv 2>nul
if exist "config\teams_2026_27.json" git add config\teams_2026_27.json
if exist "config\footystats_teams.json" git add config\footystats_teams.json
git diff --cached --quiet && (echo Nessuna variazione da pubblicare. & pause & exit /b 0)
for /f %%d in ('powershell -NoProfile -Command "Get-Date -Format yyyy-MM-dd"') do set TODAY=%%d
git commit -m "Aggiorna dati Serie C / Serie B / Serie D (%TODAY%)" && git push
echo Fatto.
pause
