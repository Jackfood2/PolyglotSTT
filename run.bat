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
rem One fast probe for every group (importlib find_spec ~= milliseconds).
rem Previously one full interpreter + heavy import per group (~16s total:
rem nemo alone ~12s, faster-whisper ~3s, torch ~2s on every launch).
rem Corrupt-but-present installs are left to the app, whose per-engine
rem fallbacks log clearly. If the probe itself fails, assume present.
set "DEPFLAGS=%TEMP%\polyglotstt_deps.txt"
del "%DEPFLAGS%" >nul 2>&1
"%~dp0venv\Scripts\python.exe" -c "import importlib.util as _u; _m={'moonshine_voice':'base','faster_whisper':'whisper','torch':'canary','nemo':'canary'}; open(r'%TEMP%\polyglotstt_deps.txt','w').write(' '.join(_v for _m2,_v in _m.items() if _u.find_spec(_m2) is None))" >nul 2>&1
set "NEED_BASE=" & set "NEED_WHISPER=" & set "NEED_CANARY="
if exist "%DEPFLAGS%" (
    findstr /c:"base" "%DEPFLAGS%" >nul 2>&1
    if not errorlevel 1 set "NEED_BASE=1"
    findstr /c:"whisper" "%DEPFLAGS%" >nul 2>&1
    if not errorlevel 1 set "NEED_WHISPER=1"
    findstr /c:"canary" "%DEPFLAGS%" >nul 2>&1
    if not errorlevel 1 set "NEED_CANARY=1"
    del "%DEPFLAGS%" >nul 2>&1
)
if not defined NEED_BASE goto CHECK_CANARY
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
if not defined NEED_CANARY goto CHECK_WHISPER
if not exist "requirements-canary.txt" goto CHECK_WHISPER
echo Installing Canary dependencies (torch+nemo, large)...
"%~dp0venv\Scripts\python.exe" -m pip install -r requirements-canary.txt --extra-index-url https://download.pytorch.org/whl/cpu
if not errorlevel 1 goto CHECK_WHISPER
echo Online Canary install failed, trying offline wheels...
"%~dp0venv\Scripts\python.exe" -m pip install --no-index --find-links="%~dp0wheels" -r requirements-canary.txt
if errorlevel 1 echo Warning: Canary deps failed. Moonshine still works.
:CHECK_WHISPER
if not defined NEED_WHISPER goto GPU_NOTE
if not exist "requirements-whisper.txt" goto GPU_NOTE
echo Installing Whisper Large v3 dependencies (faster-whisper, CPU)...
"%~dp0venv\Scripts\python.exe" -m pip install -r requirements-whisper.txt
if not errorlevel 1 goto GPU_NOTE
echo Online Whisper install failed, trying offline wheels...
"%~dp0venv\Scripts\python.exe" -m pip install --no-index --find-links="%~dp0wheels" -r requirements-whisper.txt
if errorlevel 1 echo Warning: Whisper deps failed. Moonshine still works.
:GPU_NOTE
rem Warning only here (never a surprise 2.5GB download on launch).
rem The torch import (~2s) runs once: marker lives inside venv, so a
rem recreated venv re-checks automatically. Missing marker + failing
rem check keeps nagging (actionable); a pass silences future launches.
where nvidia-smi >nul 2>&1
if errorlevel 1 goto RUN_APP
if exist "venv\.cuda_ok" goto RUN_APP
"%~dp0venv\Scripts\python.exe" -c "import torch,sys; sys.exit(0 if torch.cuda.is_available() else 1)" >nul 2>&1
if not errorlevel 1 (
    type nul > "venv\.cuda_ok"
    goto RUN_APP
)
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
