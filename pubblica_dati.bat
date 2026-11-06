@echo off
REM Dopo aver raccolto i dati sul PC (avvia.bat), pubblica i CSV sul repository GitHub
REM cosi' la dashboard in cloud (Streamlit Community Cloud) li vede. Richiede git configurato.
setlocal
cd /d "%~dp0"
where git >nul 2>&1 || (echo Serve git e una copia clonata del repository ^(non lo ZIP^). Vedi README. & pause & exit /b 1)
if not exist ".git" (echo Questa cartella non e' un repository git ^(hai usato lo ZIP^). Vedi README. & pause & exit /b 1)
git add data\processed\matches.csv data\processed\fixtures.csv data\odds\*.csv 2>nul
git diff --cached --quiet && (echo Nessuna variazione da pubblicare. & pause & exit /b 0)
for /f %%d in ('powershell -NoProfile -Command "Get-Date -Format yyyy-MM-dd"') do set TODAY=%%d
git commit -m "Aggiorna dati Serie C (%TODAY%)" && git push
echo Fatto.
pause
