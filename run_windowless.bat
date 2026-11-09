@echo off
rem Windowless launcher: no console window ever appears.
rem run_windowless.vbs is flash-free; this .bat hides its own console after
rem the checks so a double-click shows nothing beyond a possible brief flash.
cd /d "%~dp0"
set "POLYGLOTSTT_WINDOWLESS=1"
set "PATH=%~dp0venv\Scripts;%~dp0python;%PATH%"
set "PYTHONPATH="
set "PYTHONHOME="
if not exist "venv\Scripts\pythonw.exe" (
    echo No venv found - run run.bat once first (it installs everything with a window).
    pause
    exit /b 1
)
start "" "%~dp0venv\Scripts\pythonw.exe" moonshine_stt.py %*
exit /b 0
