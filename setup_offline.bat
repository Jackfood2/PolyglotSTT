@echo off
setlocal DisableDelayedExpansion
cd /d "%~dp0"

rem ============================================================
rem  PolyglotSTT - offline setup for a zipped copy of this folder
rem
rem  Rebuilds venv\ from the bundled python\ runtime and the
rem  wheels\ folder. Downloads nothing, needs no admin rights,
rem  and writes nothing outside this folder.
rem
rem  Usage: double-click, or:  setup_offline.bat [--no-pause]
rem  Models are NOT bundled - copy models_cache\ from the old
rem  machine if you want the heavy engines to work offline.
rem ============================================================

set "NO_PAUSE="
for %%a in (%*) do if /i "%%~a"=="--no-pause" set "NO_PAUSE=1"

rem Keep every interpreter, pip and cache artefact inside this folder.
set "POLY_ROOT=%~dp0"
rem pip 26 resolves every PATH entry while installing, so a locked or
rem vanished drive would abort the whole setup. Use a minimal PATH only.
set "PATH=%~dp0venv\Scripts;%~dp0python;%SystemRoot%\System32;%SystemRoot%;%SystemRoot%\System32\Wbem"
set "PYTHONPATH="
set "PYTHONHOME="
set "PIP_FIND_LINKS="
set "PIP_NO_INPUT=1"
set "PIP_DISABLE_PIP_VERSION_CHECK=1"
set "PIP_NO_INDEX=1"
set "PIP_CACHE_DIR=%~dp0cache\pip"
set "XDG_CACHE_HOME=%~dp0cache"
set "HF_HOME=%~dp0models_cache\huggingface"
set "TORCH_HOME=%~dp0models_cache\torch"
if not exist "%~dp0cache" mkdir "%~dp0cache" >nul 2>&1

echo ============================================
echo  PolyglotSTT offline setup
echo  Folder: %~dp0
echo ============================================
echo.

if not exist "%~dp0python\python.exe" goto NO_PYTHON
"%~dp0python\python.exe" -c "import sys; sys.exit(0 if sys.version_info[:2]==(3,11) else 1)" >nul 2>&1
if errorlevel 1 goto BAD_PYTHON
echo [1/4] Bundled Python 3.11 found.

set "VENV_OK="
if exist "%~dp0venv\Scripts\python.exe" (
    "%~dp0venv\Scripts\python.exe" -c "import sys,os; r=os.path.normcase(os.environ['POLY_ROOT']); s=os.path.normcase(sys.prefix)+os.sep; sys.exit(0 if s.startswith(r) and sys.version_info[:2]==(3,11) else 1)" >nul 2>&1
    if not errorlevel 1 set "VENV_OK=1"
)
if defined VENV_OK goto INSTALL
if exist "%~dp0venv" (
    echo [2/4] Existing venv is missing, broken or was moved - rebuilding...
    rmdir /S /Q "%~dp0venv"
    if exist "%~dp0venv\Scripts\python.exe" goto VENV_LOCKED
) else (
    echo [2/4] Creating venv from bundled Python...
)
"%~dp0python\python.exe" -m venv "%~dp0venv"
if errorlevel 1 goto VENV_FAIL
if not exist "%~dp0venv\Scripts\python.exe" goto VENV_FAIL
echo       venv ready.

:INSTALL
if not exist "%~dp0wheels" goto NO_WHEELS
echo [3/4] Installing dependencies from wheels\ (offline)...
rem wheels\freeze.txt pins the exact versions this package was built with,
rem so a rebuild reproduces the tested environment. Fall back to the
rem requirements files if that pin set cannot be satisfied.
if exist "%~dp0wheels\freeze.txt" (
    "%~dp0venv\Scripts\python.exe" -m pip install --no-index --find-links="%~dp0wheels" --no-warn-script-location --retries 0 --timeout 5 -r "%~dp0wheels\freeze.txt"
    if not errorlevel 1 goto INSTALLED
    echo       Exact-version install failed, using requirements files instead...
)
call :PIP requirements.txt "base"
if errorlevel 1 goto BASE_FAIL
call :PIP requirements-whisper.txt "Whisper"
call :PIP requirements-canary.txt "Canary"
call :PIP requirements-qwen.txt "Qwen3-ASR"
:INSTALLED

echo.
echo [4/4] Verifying imports...
"%~dp0venv\Scripts\python.exe" -c "import importlib.util as u; m=[('moonshine_voice','Moonshine'),('customtkinter','GUI toolkit'),('sounddevice','audio input'),('faster_whisper','Whisper'),('torch','Canary torch'),('nemo','Canary NeMo'),('transcribe_cpp','Qwen3-ASR')]; [print(('  OK    ' if u.find_spec(k) else '  MISS  ')+v) for k,v in m]"
if errorlevel 1 goto VERIFY_FAIL

echo.
echo ============================================
echo  Offline setup complete. Run run.bat to start.
echo  Optional: copy models_cache\ from the old machine
echo  to use the heavy engines without internet.
echo ============================================
if not defined NO_PAUSE pause
exit /b 0

:PIP
if not exist "%~1" exit /b 0
rem Drop pip option lines (--extra-index-url ...) - pip still honours those
rem with --no-index, which would stall on an offline machine.
findstr /v /r /c:"^[-#]" "%~1" > "%~dp0cache\offline-req.tmp"
"%~dp0venv\Scripts\python.exe" -m pip install --no-index --find-links="%~dp0wheels" --no-warn-script-location --retries 0 --timeout 5 -r "%~dp0cache\offline-req.tmp"
if errorlevel 1 (
    echo       Warning: %~2 dependencies did not install cleanly.
    exit /b 1
)
echo       %~2 dependencies installed.
exit /b 0

:NO_PYTHON
echo ERROR: python\python.exe is missing.
echo This folder must ship with the bundled Python 3.11 runtime.
echo Re-copy the project zip; it should contain python\ next to run.bat.
goto FAIL

:BAD_PYTHON
echo ERROR: python\python.exe is not Python 3.11.
echo Replace the python\ folder with the bundled 3.11 runtime.
goto FAIL

:VENV_LOCKED
echo ERROR: could not remove the old venv - close any running
echo PolyglotSTT window and try again.
goto FAIL

:VENV_FAIL
echo ERROR: failed to create venv from python\python.exe.
goto FAIL

:NO_WHEELS
echo ERROR: wheels\ folder is missing - cannot install offline.
echo Copy the wheels\ folder from the machine that ran setup.bat.
goto FAIL

:BASE_FAIL
echo ERROR: base dependencies failed to install from wheels\.
goto FAIL

:VERIFY_FAIL
echo ERROR: dependency verification failed.
goto FAIL

:FAIL
echo.
if not defined NO_PAUSE pause
exit /b 1
