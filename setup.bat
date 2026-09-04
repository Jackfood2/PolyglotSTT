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
if exist "venv\Scripts\python.exe" goto UPGRADE_PIP
echo Creating virtual environment...
%PYTHON_EXE% -m venv venv
if errorlevel 1 (
    echo ERROR: Failed to create venv.
    pause
    exit /b 1
)
echo Venv created.
:UPGRADE_PIP
echo Upgrading pip...
"%~dp0venv\Scripts\python.exe" -m pip install --upgrade pip --quiet
echo Installing dependencies...
"%~dp0venv\Scripts\python.exe" -m pip install -r requirements.txt
if not errorlevel 1 goto INSTALL_CANARY
echo Online install failed, trying offline wheels...
"%~dp0venv\Scripts\python.exe" -m pip install --no-index --find-links="%~dp0wheels" -r requirements.txt
if errorlevel 1 (
    echo ERROR: Failed to install dependencies. Ensure wheels\ folder present.
    pause
    exit /b 1
)
echo Dependencies installed via offline wheels.
:INSTALL_CANARY
if not exist "requirements-canary.txt" goto INSTALL_WHISPER
echo Installing Canary dependencies (torch+nemo, large)...
"%~dp0venv\Scripts\python.exe" -m pip install -r requirements-canary.txt --extra-index-url https://download.pytorch.org/whl/cpu
if not errorlevel 1 goto INSTALL_WHISPER
echo Online Canary install failed, trying offline wheels...
"%~dp0venv\Scripts\python.exe" -m pip install --no-index --find-links="%~dp0wheels" -r requirements-canary.txt
if errorlevel 1 echo Warning: Canary deps failed. Moonshine still works.
:INSTALL_WHISPER
if not exist "requirements-whisper.txt" goto CACHE_WHEELS
echo Installing Whisper Large v3 dependencies (faster-whisper, CPU)...
"%~dp0venv\Scripts\python.exe" -m pip install -r requirements-whisper.txt
if not errorlevel 1 goto CACHE_WHEELS
echo Online Whisper install failed, trying offline wheels...
"%~dp0venv\Scripts\python.exe" -m pip install --no-index --find-links="%~dp0wheels" -r requirements-whisper.txt
if errorlevel 1 echo Warning: Whisper deps failed. Moonshine still works.
goto CACHE_WHEELS
:INSTALL_OK
echo Dependencies installed.
:CACHE_WHEELS
echo Caching wheels for offline use...
if not exist "wheels" mkdir "wheels"
"%~dp0venv\Scripts\python.exe" -m pip download -r requirements.txt -d wheels --quiet
if exist "requirements-canary.txt" "%~dp0venv\Scripts\python.exe" -m pip download -r requirements-canary.txt -d wheels --extra-index-url https://download.pytorch.org/whl/cpu --quiet
if exist "requirements-whisper.txt" "%~dp0venv\Scripts\python.exe" -m pip download -r requirements-whisper.txt -d wheels --quiet
echo Wheels cached.
if not exist "models_cache" mkdir "models_cache"
echo Checking portable model cache...
if exist "models_cache\download.moonshine.ai\model\medium-streaming-en" (
    echo Medium model already cached, skipping.
) else (
    echo Downloading medium-streaming (110MB)...
    "%~dp0venv\Scripts\python.exe" -m moonshine_voice.download --language en --stt --root "%~dp0models_cache"
    if errorlevel 1 (
        echo ERROR: medium download failed
        pause
        exit /b 1
    )
)
if exist "models_cache\download.moonshine.ai\model\tiny-streaming-en" (
    echo Tiny model already cached, skipping.
) else (
    echo Downloading tiny-streaming (45MB)...
    "%~dp0venv\Scripts\python.exe" -m moonshine_voice.download --language en --stt --model-arch 2 --root "%~dp0models_cache"
    if errorlevel 1 echo Warning: tiny download failed, medium is enough.
)
echo.
echo ============================================
echo Setup complete! This folder is now portable.
echo Zip the entire MoonshineSTT folder for OFFLINE use.
echo On offline PC just run run.bat (no internet, no admin needed).
echo ============================================
pause
endlocal
