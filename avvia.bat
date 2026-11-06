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
echo ^>^> Aggiorno i dati reali ^(ultime 3 stagioni, Gironi A/B/C^). La prima volta serve qualche decina di minuti...
python -m src.ingestion.harvester --gironi A B C --seasons 3
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
