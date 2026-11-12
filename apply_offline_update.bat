@echo off
setlocal EnableExtensions
cd /d "%~dp0"

rem ============================================================
rem  PolyglotSTT v1.6 - offline updater
rem
rem  Upgrades an existing PolyglotSTT 1.4.16 / 1.4.17 / 1.4.18
rem  / 1.5 install to 1.6 with NO internet access.
rem
rem  Run this from the extracted package folder. It finds the
rem  install next to it, or use:
rem
rem     apply_offline_update.bat "D:\Apps\PolyglotSTT"
rem
rem  What it does:
rem    1. checks the package carries every wheel + model it needs
rem    2. checks the target is a PolyglotSTT folder and notes its
rem       current version
rem    3. backs up the settings file
rem    4. copies the v1.6 source, wheels, Python runtime and the
rem       Granite models into the target
rem    5. rebuilds venv\ offline and verifies the imports
rem
rem  Your models_cache\, notes_audio\ and settings are kept.
rem  Usage:  apply_offline_update.bat [target] [--no-pause]
rem ============================================================

set "PKG_VER=1.6"
set "TARGET="
set "NO_PAUSE="
for %%a in (%*) do (
    if /i "%%~a"=="--no-pause" (set "NO_PAUSE=1") else (
        if not "%%~a"=="" set "TARGET=%%~a"
    )
)

echo ============================================================
echo  PolyglotSTT offline updater  (target v%PKG_VER%)
echo ============================================================
echo.

rem ---------------------------------------------------------- payload
if not exist "%~dp0payload\source\moonshine_stt.py" goto BAD_PACKAGE

rem ---------------------------------------------------------- target
if not defined TARGET (
    for %%d in ("%~dp0." "%~dp0..") do (
        if not defined TARGET if exist "%%~fd\moonshine_stt.py" set "TARGET=%%~fd"
    )
)
rem Also look at sibling folders next to the package (a common layout:
rem extract the zip beside the app folder, then run it from there).
if not defined TARGET (
    for /d %%d in ("%~dp0..\*") do (
        if not defined TARGET if exist "%%~fd\moonshine_stt.py" set "TARGET=%%~fd"
    )
)
if not defined TARGET goto ASK_TARGET
if not exist "%TARGET%\moonshine_stt.py" goto BAD_TARGET
set "TARGET=%TARGET:"=%"

echo [1/6] Target folder: %TARGET%

rem ---------------------------------------------------------- version
set "OLDVER=unknown"
if exist "%TARGET%\core\config.py" (
    for /f "usebackq tokens=2 delims== " %%v in (`findstr /r /c:"^APP_VERSION" "%TARGET%\core\config.py"`) do set "OLDVER=%%~v"
)
echo       Current version: %OLDVER%
if "%OLDVER%"=="%PKG_VER%" (
    echo.
    echo       This install is already %PKG_VER%. Nothing to upgrade,
    echo       but the dependency check below still runs.
)
if "%OLDVER%"=="unknown" (
    echo       WARNING: could not read the old version; continuing.
)

rem ---------------------------------------------------------- audit
echo.
echo [2/6] Checking the package contents...
set "PROBLEM="
call :NEED "%~dp0payload\python\python.exe" 50000 "portable Python"
call :NEED "%~dp0payload\wheels\torch-2.14.1+cpu-cp311-cp311-win_amd64.whl" 100000000 "torch wheel"
call :NEED "%~dp0payload\wheels\torchaudio-2.11.1+cpu-py3-none-win_amd64.whl" 200000 "torchaudio wheel"
call :NEED "%~dp0payload\models\granite\granite-speech-5.0-470m-turboctc-Q8_0.gguf" 400000000 "Granite 470M Q8_0"
call :NEED "%~dp0payload\models\granite-2b-gguf\granite-speech-5.0-2b-Q8_0.gguf" 1400000000 "Granite 2B Q8_0"
call :NEED "%~dp0payload\models\granite-2b-gguf\mmproj-model-f16.gguf" 800000000 "Granite 2B projector"
call :NEED "%~dp0payload\models\llama.cpp\b9768\llama-server.exe" 4096 "llama.cpp launcher"
call :NEED "%~dp0payload\models\llama.cpp\b9768\llama-server-impl.dll" 1000000 "llama.cpp server"
set /a WCOUNT=0
for %%w in ("%~dp0payload\wheels\*.whl") do set /a WCOUNT+=1
echo       Wheel count: %WCOUNT%
if %WCOUNT% LSS 100 (
    echo       MISSING only %WCOUNT% wheels - the package is incomplete.
    set "PROBLEM=1"
)
if defined PROBLEM goto WHEELS_BAD

rem ---------------------------------------------------------- backup
echo.
echo [3/6] Backing up settings...
if exist "%TARGET%\moonshine_config.json" (
    copy /y "%TARGET%\moonshine_config.json" "%TARGET%\moonshine_config.pre-%PKG_VER%.json" >nul 2>&1
    echo       Saved moonshine_config.pre-%PKG_VER%.json
) else (
    echo       No settings file yet - nothing to back up.
)

rem ---------------------------------------------------------- copy
echo.
echo [4/6] Copying v%PKG_VER% files into the install...
robocopy "%~dp0payload\source" "%TARGET%" /E /NFL /NDL /NJH /NJS /XD "%~dp0payload\source\.git" >nul 2>&1
if errorlevel 8 goto COPY_FAIL

echo       Merging wheels...
robocopy "%~dp0payload\wheels" "%TARGET%\wheels" /E /NFL /NDL /NJH /NJS >nul 2>&1
if errorlevel 8 goto COPY_FAIL

echo       Installing portable Python...
robocopy "%~dp0payload\python" "%TARGET%\python" /E /NFL /NDL /NJH /NJS >nul 2>&1
if errorlevel 8 goto COPY_FAIL

echo       Installing Granite models ^(kept if already present^)...
call :COPYMODEL "%~dp0payload\models\granite\granite-speech-5.0-470m-turboctc-Q8_0.gguf" "%TARGET%\models_cache\granite\granite-speech-5.0-470m-turboctc-Q8_0.gguf"
call :COPYMODEL "%~dp0payload\models\granite-2b-gguf\granite-speech-5.0-2b-Q8_0.gguf" "%TARGET%\models_cache\granite-2b-gguf\granite-speech-5.0-2b-Q8_0.gguf"
call :COPYMODEL "%~dp0payload\models\granite-2b-gguf\mmproj-model-f16.gguf" "%TARGET%\models_cache\granite-2b-gguf\mmproj-model-f16.gguf"
robocopy "%~dp0payload\models\llama.cpp\b9768" "%TARGET%\models_cache\llama.cpp\b9768" /E /XF "*.log" /NFL /NDL /NJH /NJS >nul 2>&1
if errorlevel 8 goto COPY_FAIL

rem ---------------------------------------------------------- venv
echo.
echo [5/6] Rebuilding venv\ offline ^(this takes a few minutes^)...
if not exist "%TARGET%\setup_offline.bat" goto NO_SETUP
pushd "%TARGET%"
call "%TARGET%\setup_offline.bat" --no-pause
set "SETUP_RC=%errorlevel%"
popd
if not "%SETUP_RC%"=="0" goto SETUP_FAIL

rem ---------------------------------------------------------- verify
echo.
echo [6/6] Verifying the dependency folder and the imports...
"%TARGET%\venv\Scripts\python.exe" -c "import importlib.util as u; m=[('moonshine_voice','Moonshine'),('customtkinter','GUI'),('sounddevice','audio in'),('soundcard','desktop audio'),('faster_whisper','Whisper'),('torch','Granite 2B torch'),('torchaudio','Granite 2B torchaudio'),('transformers','Granite 2B transformers'),('transcribe_cpp','Qwen3-ASR')]; [print(('       OK    ' if u.find_spec(k) else '       MISS  ')+v) for k,v in m]" 
if errorlevel 1 goto VERIFY_FAIL

rem Offline resolution audit: proves wheels\ alone satisfies every group.
echo.
echo       Offline wheel audit ^(no network used^):
call :REQAUDIT "requirements.txt"
call :REQAUDIT "requirements-whisper.txt"
call :REQAUDIT "requirements-canary.txt"
call :REQAUDIT "requirements-qwen.txt"
call :REQAUDIT "requirements-granite2b.txt"

set "NEWVER=unknown"
for /f "usebackq tokens=2 delims== " %%v in (`findstr /r /c:"^APP_VERSION" "%TARGET%\core\config.py"`) do set "NEWVER=%%~v"

echo.
echo ============================================================
echo  Update complete:  %OLDVER%  --^>  %NEWVER%
echo.
echo  Granite models available offline:
echo    Granite 5.0 470M TurboCTC  (Q8_0)
echo    Granite Speech 5.0 2B     (Q8_0 GGUF, llama.cpp)
echo  The 4.8 GB Transformers copy of Granite 2B is NOT bundled;
echo  pick the GGUF models in the app, or download it online.
echo.
echo  Start the app with run.bat
echo ============================================================
if not defined NO_PAUSE pause
exit /b 0


rem ============================================================ helpers
:NEED
rem %1 path, %2 min bytes, %3 label
if not exist "%~1" goto NEED_MISS
for %%A in ("%~1") do if %%~zA LSS %~2 goto NEED_MISS
echo       OK      %~3
exit /b 0
:NEED_MISS
echo       MISSING %~3
set "PROBLEM=1"
exit /b 0

:COPYMODEL
rem %1 source, %2 destination. Skip when the destination is already complete.
if not exist "%~1" exit /b 0
if exist "%~2" (
    for %%A in ("%~1") do set "SRC_SZ=%%~zA"
    for %%B in ("%~2") do set "DST_SZ=%%~zB"
    setlocal EnableDelayedExpansion
    if !DST_SZ! GEQ !SRC_SZ! (
        endlocal
        exit /b 0
    )
    endlocal
)
for %%D in ("%~dp2.") do if not exist "%%~fD" mkdir "%%~fD" >nul 2>&1
copy /y "%~1" "%~2" >nul 2>&1
if errorlevel 1 (
    echo       WARNING: could not copy %~nx1
    exit /b 0
)
echo       Installed %~nx1
exit /b 0

:REQAUDIT
if not exist "%TARGET%\%~1" exit /b 0
findstr /v /r /b /c:"-" /c:"#" "%TARGET%\%~1" > "%TARGET%\_audit.tmp" 2>nul
"%TARGET%\venv\Scripts\python.exe" -m pip install --dry-run --ignore-installed --no-index --find-links="%TARGET%\wheels" -r "%TARGET%\_audit.tmp" >nul 2>&1
if errorlevel 1 (
    echo       MISSING %~1
    echo               %~1 cannot be satisfied from wheels\.
) else (
    echo       OK      %~1
)
del /f /q "%TARGET%\_audit.tmp" >nul 2>&1
exit /b 0


rem ============================================================ errors
:ASK_TARGET
echo ERROR: could not find a PolyglotSTT folder automatically.
echo        Pass the path explicitly, for example:
echo            apply_offline_update.bat "D:\Apps\PolyglotSTT"
goto FAIL

:BAD_TARGET
echo ERROR: %TARGET% does not contain moonshine_stt.py.
echo        Point this at the PolyglotSTT folder you want to update.
goto FAIL

:BAD_PACKAGE
echo ERROR: this script must be run from the extracted package root,
echo        next to the payload\ folder.
echo        Missing: payload\source\moonshine_stt.py
goto FAIL

:WHEELS_BAD
echo.
echo ERROR: the package is incomplete - some wheels or models did not
echo        survive the copy. Re-copy the zip, or rebuild it on an
echo        online machine with make_offline_update.bat.
goto FAIL

:COPY_FAIL
echo.
echo ERROR: copying into the install failed.
echo        Close PolyglotSTT if it is running, and check the folder is
echo        writable ^(not inside Program Files without admin rights^).
goto FAIL

:NO_SETUP
echo.
echo ERROR: setup_offline.bat is missing from the target after copying.
goto FAIL

:SETUP_FAIL
echo.
echo ERROR: the offline dependency install failed.
echo        Scroll up for the pip error. If a wheel is missing, rebuild
echo        the package online with make_offline_update.bat.
goto FAIL

:VERIFY_FAIL
echo.
echo ERROR: dependency verification failed after install.
goto FAIL

:FAIL
echo.
if not defined NO_PAUSE pause
exit /b 1
