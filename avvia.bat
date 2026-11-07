@echo off
REM Avvio con un doppio clic (Windows): prepara Python, scarica i dati FootyStats con la tua chiave, apre la dashboard.
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
REM Sorgente dati: FootyStats (API ufficiale), l'unica fonte dei dati reali: squadre, calendari con orari e
REM partite rinviate, risultati e quote pre-match. La chiave si legge dalla variabile d'ambiente
REM FOOTYSTATS_API_KEY o dal file .env accanto a questo script (vedi .env.example) e non viene mai stampata.
REM I dati restano SOLO su questo computer (data\ e' escluso da git): il repository e' pubblico e i dati
REM FootyStats sono concessi in licenza al titolare della chiave, quindi non vengono mai pubblicati.
REM Il controllo e' delegato allo stesso parser Python della CLI (variabile d'ambiente o .env, anche con
REM BOM, CRLF, "export", spazi iniziali, virgolette e commenti), cosi' lanciatore e CLI non possono essere in disaccordo.
set "USE_FOOTYSTATS="
python -c "import sys; from src.ingestion.footystats import find_api_key; from src.ingestion.harvester import load_endpoints; sys.exit(0 if find_api_key(key_env=load_endpoints().footystats.key_env) else 1)" >nul 2>&1 && set "USE_FOOTYSTATS=1"
if not defined USE_FOOTYSTATS if not exist ".env" (
  echo ^>^> Nessuna chiave FootyStats trovata.
  echo    Incolla qui la chiave API di FootyStats e premi Invio ^(footystats.org -^> il tuo account -^> API^).
  set "FS_KEY="
  set /p "FS_KEY=   Chiave: "
  if defined FS_KEY (
    > ".env" echo FOOTYSTATS_API_KEY=%FS_KEY: =%
    echo    Chiave salvata nel file .env ^(resta solo su questo computer^).
    set "FS_KEY="
    python -c "import sys; from src.ingestion.footystats import find_api_key; from src.ingestion.harvester import load_endpoints; sys.exit(0 if find_api_key(key_env=load_endpoints().footystats.key_env) else 1)" >nul 2>&1 && set "USE_FOOTYSTATS=1"
  )
)
if not defined USE_FOOTYSTATS (
  echo ^>^> Nessuna chiave FootyStats ^(FOOTYSTATS_API_KEY o file .env, vedi .env.example^): nessun dato reale da scaricare.
  echo    La dashboard si apre ma non mostra partite inventate: inserisci la chiave e riavvia.
  goto dashboard
)
echo ^>^> Scarico i dati reali da FootyStats ^(Serie C Gironi A/B/C + Serie B + Serie D Gironi A-I, ultime 3 stagioni,
echo    orari, partite rinviate e quote pre-match^) e allineo il dizionario squadre a FootyStats...
echo    Le competizioni la cui lega non e' fra quelle scelte nel tuo account FootyStats vengono saltate ^(le altre sono aggiornate comunque^).
REM Nessun --competitions = tutte le competizioni di config\competitions.toml ^(13^); la sincronizzazione del
REM dizionario config\teams_2026_27.json con /league-teams e' inclusa ^(squadre aggiunte, spostate o rimosse^).
python -m src.ingestion.footystats --seasons 3
if errorlevel 1 (
  echo ATTENZIONE: raccolta dati non riuscita ^(vedi messaggi sopra^).
  if exist "data\processed\matches.csv" (echo La dashboard usa i dati reali scaricati in precedenza ^(controlla la data dell'ultimo aggiornamento^).) else (echo Nessun dato reale disponibile: la dashboard non mostra partite finche' la raccolta FootyStats non riesce.)
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
