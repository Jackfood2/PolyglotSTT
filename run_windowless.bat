@echo off
rem Windowless launcher: no console window stays open (a brief one may
rem flash on double-click - use run_windowless.vbs for zero flash).
cd /d "%~dp0"
rem C:-only: D: is BitLocker-locked, drop non-C: entries so the app never touches D:.
for /f "delims=" %%P in ('powershell -NoProfile -Command "$a=$env:PATH -split ';' | Where-Object { $_ -ne '' -and $_ -match '^[Cc]:' -and $_ -notlike '*Graphviz*' } | Select-Object -Unique; [string]::Join(';',$a)"') do set "PATH=%%P"
set "PYTHONPATH="
set "PYTHONHOME="
if not exist "venv\Scripts\pythonw.exe" (
    echo No venv found - run run.bat once first (it installs everything with a window).
    pause
    exit /b 1
)
start "PolyglotSTT" "%~dp0venv\Scripts\pythonw.exe" moonshine_stt.py
