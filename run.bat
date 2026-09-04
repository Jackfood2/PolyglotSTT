@echo off
setlocal DisableDelayedExpansion
cd /d "%~dp0"
set "PATH=%~dp0python;%PATH%"
rem Portable Python preferred; fall back to system Python 3.11 (source checkouts)
set "PYTHON_EXE="%~dp0python\python.exe""
if exist "%~dp0python\python.exe" goto PY_OK
set "PYTHON_EXE="
where py >nul 2>&1
if not errorlevel 1 (
    py -3.11 -c "import sys; sys.exit(0 if sys.version_info[:2]==(3,11) else 1)" >nul 2>&1
    if not errorlevel 1 set "PYTHON_EXE=py -3.11"
)
if defined PYTHON_EXE goto PY_OK
where python >nul 2>&1
if not errorlevel 1 (
    python -c "import sys; sys.exit(0 if sys.version_info[:2]==(3,11) else 1)" >nul 2>&1
    if not errorlevel 1 set "PYTHON_EXE=python"
)
if not defined PYTHON_EXE (
    echo ERROR: Need portable python\python.exe or system Python 3.11 - see README.md
    pause
    exit /b 1
)
:PY_OK
if not exist "venv\Scripts\python.exe" goto CREATE_VENV
"%~dp0venv\Scripts\python.exe" -c "import sys" >nul 2>&1
if not errorlevel 1 goto CHECK_DEPS
echo Venv moved or broken, recreating for new location...
rmdir /S /Q "venv" >nul 2>&1
:CREATE_VENV
echo Creating venv from portable Python...
%PYTHON_EXE% -m venv venv
if errorlevel 1 (
    echo Failed to create venv.
    pause
    exit /b 1
)
:CHECK_DEPS
"%~dp0venv\Scripts\python.exe" -c "import moonshine_voice" >nul 2>&1
if not errorlevel 1 goto CHECK_CANARY
echo Installing dependencies...
"%~dp0venv\Scripts\python.exe" -m pip install -r requirements.txt >nul 2>&1
if not errorlevel 1 goto CHECK_CANARY
echo Trying offline wheels...
"%~dp0venv\Scripts\python.exe" -m pip install --no-index --find-links="%~dp0wheels" -r requirements.txt
if errorlevel 1 (
    echo Failed to install dependencies. Ensure wheels\ folder is present.
    pause
    exit /b 1
)
:CHECK_CANARY
if not exist "requirements-canary.txt" goto CHECK_WHISPER
"%~dp0venv\Scripts\python.exe" -c "import nemo.collections.asr" >nul 2>&1
if not errorlevel 1 goto CHECK_WHISPER
echo Installing Canary dependencies (torch+nemo, offline)...
"%~dp0venv\Scripts\python.exe" -m pip install --no-index --find-links="%~dp0wheels" -r requirements-canary.txt
if errorlevel 1 echo Warning: Canary deps not installed. Moonshine engine still works.
:CHECK_WHISPER
if not exist "requirements-whisper.txt" goto RUN_APP
"%~dp0venv\Scripts\python.exe" -c "import faster_whisper" >nul 2>&1
if not errorlevel 1 goto RUN_APP
echo Installing Whisper Large v3 dependencies (faster-whisper, offline)...
"%~dp0venv\Scripts\python.exe" -m pip install --no-index --find-links="%~dp0wheels" -r requirements-whisper.txt
if errorlevel 1 echo Warning: Whisper deps not installed. Moonshine engine still works.
:GPU_NOTE
rem Warning only here (never a surprise 2.5GB download on launch).
where nvidia-smi >nul 2>&1
if errorlevel 1 goto RUN_APP
"%~dp0venv\Scripts\python.exe" -c "import torch,sys; sys.exit(0 if torch.cuda.is_available() else 1)" >nul 2>&1
if not errorlevel 1 goto RUN_APP
echo Note: NVIDIA GPU found but torch is CPU-only. Run setup.bat once for GPU support.
:RUN_APP
if not exist "models_cache\download.moonshine.ai" echo Warning: models_cache not found. Will try APPDATA cache.
"%~dp0venv\Scripts\python.exe" moonshine_stt.py %*
if errorlevel 1 (
    echo.
    echo MoonshineSTT exited with an error.
    pause
)
endlocal
