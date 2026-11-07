@echo off
REM Avvio con un doppio clic (Windows): prepara Python, aggiorna i dati, apre la dashboard.
REM Prerequisito: Python 3.12 o superiore installato da https://www.python.org/downloads/
REM (nell'installer spuntare "Add python.exe to PATH").
setlocal
cd /d "%~dp0"

set "PY="
for %%c in (py python) do (
  if not defined PY (
    %%c -3 -c "import sys; sys.exit(0 if sys.version_info >= (3, 12) else 1)" >nul 2>&1 && set "PY=%%c -3"
  )
)
if not defined PY (
  python -c "import sys; sys.exit(0 if sys.version_info >= (3, 12) else 1)" >nul 2>&1 && set "PY=python"
)
if not defined PY (
  echo ERRORE: serve Python 3.12 o superiore. Scaricalo da https://www.python.org/downloads/ e riprova.
  pause
  exit /b 1
)

if not exist ".venv\Scripts\python.exe" (
  echo ^>^> Prima esecuzione: creo l'ambiente Python ^(una volta sola^)...
  %PY% -m venv .venv || (echo ERRORE nella creazione dell'ambiente. & pause & exit /b 1)
)
call ".venv\Scripts\activate.bat"
python -m pip install --quiet --disable-pip-version-check -r requirements.txt || (
  echo ERRORE nell'installazione delle dipendenze ^(serve la connessione internet^). & pause & exit /b 1)

if /I "%~1"=="--solo-dashboard" goto dashboard
REM Sorgente dati: FootyStats (API ufficiale, con quote pre-match) se la chiave e' nella variabile
REM d'ambiente FOOTYSTATS_API_KEY o nel file .env accanto a questo script (vedi .env.example),
REM altrimenti SofaScore (solo da connessione residenziale). La chiave non viene mai stampata.
REM Il controllo e' delegato allo stesso parser Python della CLI (variabile d'ambiente o .env, anche con
REM BOM, CRLF, "export", spazi iniziali, virgolette e commenti), cosi' lanciatore e CLI non possono essere in disaccordo.
set "USE_FOOTYSTATS="
python -c "import sys; from src.ingestion.footystats import find_api_key; from src.ingestion.harvester import load_endpoints; sys.exit(0 if find_api_key(key_env=load_endpoints().footystats.key_env) else 1)" >nul 2>&1 && set "USE_FOOTYSTATS=1"
if not defined USE_FOOTYSTATS if not exist ".env" (
  echo ^>^> Nessuna chiave FootyStats trovata.
  echo    Incolla qui la chiave API di FootyStats e premi Invio ^(oppure premi solo Invio per usare SofaScore^).
  set "FS_KEY="
  set /p "FS_KEY=   Chiave: "
  if defined FS_KEY (
    > ".env" echo FOOTYSTATS_API_KEY=%FS_KEY: =%
    echo    Chiave salvata nel file .env ^(resta solo su questo computer^).
    set "FS_KEY="
    python -c "import sys; from src.ingestion.footystats import find_api_key; from src.ingestion.harvester import load_endpoints; sys.exit(0 if find_api_key(key_env=load_endpoints().footystats.key_env) else 1)" >nul 2>&1 && set "USE_FOOTYSTATS=1"
  )
)
if defined USE_FOOTYSTATS (
  echo ^>^> Aggiorno i dati reali da FootyStats ^(Serie C Gironi A/B/C + Serie B + Serie D Gironi A-I, ultime 3 stagioni, quote pre-match^)...
  echo    Serie D: alla prima esecuzione le liste squadre dei 9 gironi vengono create in config\teams_2026_27.json dall'API.
  REM Nessun --competitions = tutte le competizioni di config\competitions.toml ^(13^).
  python -m src.ingestion.footystats --seasons 3 --odds-out data\odds\footystats.csv
) else (
  echo ^>^> Nessuna chiave FootyStats ^(FOOTYSTATS_API_KEY o file .env, vedi .env.example^): uso SofaScore.
  echo ^>^> Aggiorno i dati reali ^(ultime 3 stagioni, Gironi A/B/C^). La prima volta serve qualche decina di minuti...
  python -m src.ingestion.harvester --gironi A B C --seasons 3
)
if errorlevel 1 (
  echo ATTENZIONE: raccolta dati non riuscita ^(vedi messaggi sopra^).
  if exist "data\processed\matches.csv" (echo Uso i dati reali raccolti in precedenza.) else (echo Nessun dato reale disponibile: la dashboard mostrera' la lega simulata.)
)

:dashboard
REM Evita la domanda interattiva di benvenuto di Streamlit (richiesta email) al primo avvio.
if not exist "%USERPROFILE%\.streamlit" mkdir "%USERPROFILE%\.streamlit"
if not exist "%USERPROFILE%\.streamlit\credentials.toml" (
  echo [general]> "%USERPROFILE%\.streamlit\credentials.toml"
  echo email = "">> "%USERPROFILE%\.streamlit\credentials.toml"
)
echo ^>^> Apro la dashboard nel browser ^(chiudi questa finestra per fermarla^).
python -m streamlit run src\dashboard\app.py --server.headless false --browser.gatherUsageStats false
pause
