@echo off
rem Windowless launcher: no console window stays open (a brief one may
rem flash on double-click - use run_windowless.vbs for zero flash).
cd /d "%~dp0"
if not exist "venv\Scripts\pythonw.exe" (
    echo No venv found - run run.bat once first (it installs everything with a window).
    pause
    exit /b 1
)
start "PolyglotSTT" "%~dp0venv\Scripts\pythonw.exe" moonshine_stt.py
