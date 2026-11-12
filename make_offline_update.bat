@echo off
setlocal EnableExtensions
cd /d "%~dp0"

rem ============================================================
rem  PolyglotSTT v1.6 - offline update builder
rem
rem  Run this on the ONLINE machine where PolyglotSTT is
rem  installed and working. It produces, next to this folder:
rem
rem     PolyglotSTT-update-v1.6-offline\      (the package)
rem     PolyglotSTT-update-v1.6-offline.zip   (same, zipped)
rem
rem  The package carries the v1.6 source, a portable Python
rem  3.11 runtime, every dependency wheel and the two Granite
rem  models, so apply_offline_update.bat can upgrade a
rem  1.4.16 / 1.4.17 / 1.4.18 / 1.5 install with no internet.
rem
rem  Bundled models (GGUF, llama.cpp):
rem    Granite 5.0 470M TurboCTC  Q8_0
rem    Granite Speech 5.0 2B     Q8_0 + mmproj
rem    llama.cpp b9768 runtime
rem
rem  Usage:  make_offline_update.bat [--no-pause]
rem ============================================================

set "PKG_VER=1.6"
set "PKG_NAME=PolyglotSTT-update-v1.6-offline"
set "OUTDIR=%~dp0..\%PKG_NAME%"
set "ZIP=%~dp0..\%PKG_NAME%.zip"
set "PY=%~dp0venv\Scripts\python.exe"

rem RAR parts are written here. Ask the shell first so a OneDrive-moved
rem Desktop still resolves, then fall back to the usual path.
set "DESKTOP=%USERPROFILE%\Desktop"
for /f "usebackq delims=" %%D in (`powershell -NoProfile -Command "[Environment]::GetFolderPath('Desktop')" 2^>nul`) do if not "%%D"=="" set "DESKTOP=%%D"

set "NO_PAUSE="
set "NO_ARCHIVE="
for %%a in (%*) do (
    if /i "%%~a"=="--no-pause" set "NO_PAUSE=1"
    if /i "%%~a"=="--no-archive" set "NO_ARCHIVE=1"
)

echo ============================================================
echo  PolyglotSTT offline update builder  (target v%PKG_VER%)
echo  Source: %~dp0
echo ============================================================
echo.

if not exist "%PY%" goto NO_VENV

rem ---------------------------------------------------------- 0. version
rem Read via a temp file: "for /f in (`"quoted.exe" ...`)" loses the quotes
rem and fails with "not recognized as an internal or external command".
"%PY%" -c "import core.config as c;print(c.APP_VERSION)" > "%~dp0_pkg_ver.tmp" 2>nul
set "APPV="
if exist "%~dp0_pkg_ver.tmp" set /p APPV=<"%~dp0_pkg_ver.tmp"
del /f /q "%~dp0_pkg_ver.tmp" >nul 2>&1
if not defined APPV goto NO_VERSION
echo [0/5] Source version: %APPV%
if not "%APPV%"=="%PKG_VER%" (
    echo.
    echo ERROR: this folder reports version %APPV%, but this builder
    echo        produces the %PKG_VER% package.
    goto FAIL
)

rem ---------------------------------------------------------- 1. wheels
echo.
echo [1/5] Auditing wheels\ against every requirements file...
echo       ^(a group that cannot resolve offline is re-downloaded^)
set "AUDIT_FAIL="
for %%R in (requirements.txt requirements-whisper.txt requirements-canary.txt requirements-qwen.txt requirements-granite2b.txt) do (
    if exist "%%R" call :AUDIT_ONE "%%R"
)
if defined AUDIT_FAIL goto AUDIT_ERR

rem ---------------------------------------------------------- 2. python
echo.
echo [2/5] Portable Python runtime...
call :MAKE_PYTHON
if errorlevel 1 goto PYTHON_ERR

rem ---------------------------------------------------------- 3. models
echo.
echo [3/5] Checking the models that will be bundled...
call :CHECK_MODELS
if errorlevel 1 goto MODEL_ERR

rem ---------------------------------------------------------- 4. stage
echo.
echo [4/5] Staging the package...
call :STAGE
if errorlevel 1 goto STAGE_ERR

rem ---------------------------------------------------------- 5. archive
echo.
if defined NO_ARCHIVE (
    echo [5/5] Archiving skipped ^(--no-archive^).
    echo   Staged folder: %OUTDIR%
    if not defined NO_PAUSE pause
    exit /b 0
)
echo [5/5] Writing split archives ^(990 MB per volume, to the Desktop^)...
call :MAKE_ARCHIVE
if errorlevel 1 goto ZIP_ERR

echo   Staged folder: %OUTDIR%
echo.
echo ============================================================
echo  Package ready on your Desktop:
echo    %PKG_NAME%.part1.rar ...
echo.
echo  Copy EVERY .partN.rar to the offline machine, open part1 in
echo  WinRAR ^(it finds the rest automatically^), extract it, then run
echo  apply_offline_update.bat there.
echo ============================================================
if not defined NO_PAUSE pause
exit /b 0

:MAKE_ARCHIVE
rem WinRAR multi-volume, 990 MB per part, written to the Desktop.
rem -ep1 drops the leading path so the archive extracts into one folder.
set "RAR_EXE=%ProgramFiles%\WinRAR\Rar.exe"
if not exist "%RAR_EXE%" set "RAR_EXE=%ProgramFiles(x86)%\WinRAR\Rar.exe"
if not exist "%RAR_EXE%" goto MAKE_ZIP
if not exist "%DESKTOP%" mkdir "%DESKTOP%" >nul 2>&1
del /f /q "%DESKTOP%\%PKG_NAME%.part*.rar" >nul 2>&1
pushd "%OUTDIR%"
"%RAR_EXE%" a -r -ep1 -m1 -v990m -y "%DESKTOP%\%PKG_NAME%.rar" payload apply_offline_update.bat > "%OUTDIR%\..\_rar.log" 2>&1
set "RAR_RC=%errorlevel%"
popd
if not "%RAR_RC%"=="0" goto MAKE_ARCHIVE_FAIL
if not exist "%DESKTOP%\%PKG_NAME%.part1.rar" goto MAKE_ZIP
dir /b "%DESKTOP%\%PKG_NAME%.part*.rar" > "%OUTDIR%\..\_rar_parts.tmp" 2>nul
for /f "usebackq tokens=*" %%P in ("%OUTDIR%\..\_rar_parts.tmp") do echo   Part:  %%P
if exist "%OUTDIR%\..\_rar_parts.tmp" del /f /q "%OUTDIR%\..\_rar_parts.tmp" >nul 2>&1
exit /b 0

:MAKE_ARCHIVE_FAIL
echo   RAR failed - see _rar.log next to this folder.
exit /b 1

:MAKE_ZIP
echo   WinRAR not found - writing a single zip next to this folder.
if exist "%ZIP%" del /f /q "%ZIP%"
where tar >nul 2>&1
if errorlevel 1 goto NO_TAR
pushd "%OUTDIR%"
tar -a -c -f "%ZIP%" payload apply_offline_update.bat >nul 2>&1
set "ZIP_RC=%errorlevel%"
popd
if not "%ZIP_RC%"=="0" exit /b 1
if not exist "%ZIP%" exit /b 1
for %%A in ("%ZIP%") do echo   Zip:   %%~nxA  ^(%%~zA bytes^)
exit /b 0


rem ============================================================ helpers
:AUDIT_ONE
rem %1 = requirements file. Recreates missing wheels, then re-checks.
set "REQFILE=%~1"
set "TMPREQ=%~dp0_pkg_req.tmp"
findstr /v /r /b /c:"-" /c:"#" "%REQFILE%" > "%TMPREQ%" 2>nul
if not exist "%TMPREQ%" goto :EOF
if not exist "%~dp0wheels" mkdir "%~dp0wheels" >nul 2>&1

"%PY%" -m pip install --dry-run --ignore-installed --no-index --find-links="%~dp0wheels" -r "%TMPREQ%" >nul 2>&1
if not errorlevel 1 (
    echo       OK      %REQFILE%
    del /f /q "%TMPREQ%" >nul 2>&1
    exit /b 0
)

echo       MISSING %REQFILE% - re-downloading wheels...
"%PY%" -m pip download --only-binary=:all: -r "%TMPREQ%" -d "%~dp0wheels" --extra-index-url https://download.pytorch.org/whl/cpu
if errorlevel 1 goto :AUDIT_ONE_FAIL

"%PY%" -m pip install --dry-run --ignore-installed --no-index --find-links="%~dp0wheels" -r "%TMPREQ%" >nul 2>&1
if errorlevel 1 goto :AUDIT_ONE_FAIL
echo       FIXED   %REQFILE%
del /f /q "%TMPREQ%" >nul 2>&1
exit /b 0

:AUDIT_ONE_FAIL
del /f /q "%TMPREQ%" >nul 2>&1
echo       FAILED  %REQFILE% - cannot be satisfied offline.
set "AUDIT_FAIL=1"
exit /b 1

:MAKE_PYTHON
if exist "%~dp0python\python.exe" (
    "%~dp0python\python.exe" -c "import sys,venv,ensurepip,tkinter;sys.exit(0 if sys.version_info[:2]==(3,11) else 1)" >nul 2>&1
    if not errorlevel 1 (
        echo       Reusing existing python\ runtime.
        exit /b 0
    )
    echo       Existing python\ is unusable - rebuilding it.
    rmdir /s /q "%~dp0python" >nul 2>&1
)
echo       Copying the base Python 3.11 runtime (stdlib + tkinter)...
"%PY%" -c "import sys;print(sys.base_prefix)" > "%~dp0_pkg_base.tmp" 2>nul
set "BASEPY="
if exist "%~dp0_pkg_base.tmp" set /p BASEPY=<"%~dp0_pkg_base.tmp"
del /f /q "%~dp0_pkg_base.tmp" >nul 2>&1
if not exist "%BASEPY%\python.exe" goto MAKE_PYTHON_FAIL
robocopy "%BASEPY%" "%~dp0python" /E /NFL /NDL /NJH /NJS /XD "%BASEPY%\Lib\site-packages" "%BASEPY%\Doc" "%BASEPY%\Tools" "%BASEPY%\include" "%BASEPY%\libs" >nul 2>&1
if errorlevel 8 goto MAKE_PYTHON_FAIL
"%~dp0python\python.exe" -c "import sys,venv,ensurepip,tkinter" >nul 2>&1
if errorlevel 1 goto MAKE_PYTHON_FAIL
echo       Portable Python ready.
exit /b 0

:MAKE_PYTHON_FAIL
echo       ERROR: could not build the portable python\ runtime.
exit /b 1

:CHECK_MODELS
set "M_OK=1"
call :NEED_FILE "models_cache\granite\granite-speech-5.0-470m-turboctc-Q8_0.gguf" 400000000 "Granite 470M Q8_0"
call :NEED_FILE "models_cache\granite-2b-gguf\granite-speech-5.0-2b-Q8_0.gguf" 1400000000 "Granite 2B Q8_0"
call :NEED_FILE "models_cache\granite-2b-gguf\mmproj-model-f16.gguf" 800000000 "Granite 2B projector"
call :NEED_FILE "models_cache\llama.cpp\b9768\llama-server.exe" 4096 "llama.cpp b9768 launcher"
call :NEED_FILE "models_cache\llama.cpp\b9768\llama-server-impl.dll" 1000000 "llama.cpp b9768 server"
if defined M_OK exit /b 0
exit /b 1

:NEED_FILE
rem %1 path, %2 min bytes, %3 label
if not exist "%~1" goto NEED_FILE_MISS
for %%A in ("%~1") do if %%~zA LSS %~2 goto NEED_FILE_MISS
echo       OK      %~3
exit /b 0
:NEED_FILE_MISS
echo       MISSING %~3  ^(%~1^)
echo               Run run.bat online once, pick that model, let it finish.
set "M_OK="
exit /b 0

:STAGE
if exist "%OUTDIR%" rmdir /s /q "%OUTDIR%" >nul 2>&1
if exist "%OUTDIR%" (
    echo       ERROR: could not clear %OUTDIR%
    exit /b 1
)
mkdir "%OUTDIR%\payload\source" >nul 2>&1
mkdir "%OUTDIR%\payload\wheels" >nul 2>&1
mkdir "%OUTDIR%\payload\python" >nul 2>&1
mkdir "%OUTDIR%\payload\models\granite" >nul 2>&1
mkdir "%OUTDIR%\payload\models\granite-2b-gguf" >nul 2>&1
mkdir "%OUTDIR%\payload\models\llama.cpp\b9768" >nul 2>&1

echo       Source files...
robocopy "%~dp0." "%OUTDIR%\payload\source" /E /NFL /NDL /NJH /NJS /XD "%~dp0venv" "%~dp0wheels" "%~dp0models_cache" "%~dp0python" "%~dp0logs" "%~dp0notes_audio" "%~dp0cache" "%~dp0__pycache__" "%~dp0.git" >nul 2>&1
if errorlevel 8 goto STAGE_FAIL

echo       Wheels...
robocopy "%~dp0wheels" "%OUTDIR%\payload\wheels" /E /NFL /NDL /NJH /NJS >nul 2>&1
if errorlevel 8 goto STAGE_FAIL

echo       Portable Python...
robocopy "%~dp0python" "%OUTDIR%\payload\python" /E /NFL /NDL /NJH /NJS >nul 2>&1
if errorlevel 8 goto STAGE_FAIL

echo       Models...
copy /y "%~dp0models_cache\granite\granite-speech-5.0-470m-turboctc-Q8_0.gguf" "%OUTDIR%\payload\models\granite\" >nul 2>&1
copy /y "%~dp0models_cache\granite-2b-gguf\granite-speech-5.0-2b-Q8_0.gguf" "%OUTDIR%\payload\models\granite-2b-gguf\" >nul 2>&1
copy /y "%~dp0models_cache\granite-2b-gguf\mmproj-model-f16.gguf" "%OUTDIR%\payload\models\granite-2b-gguf\" >nul 2>&1
robocopy "%~dp0models_cache\llama.cpp\b9768" "%OUTDIR%\payload\models\llama.cpp\b9768" /E /XF "*.log" /NFL /NDL /NJH /NJS >nul 2>&1
if errorlevel 8 goto STAGE_FAIL

copy /y "%~dp0apply_offline_update.bat" "%OUTDIR%\apply_offline_update.bat" >nul 2>&1
if not exist "%OUTDIR%\apply_offline_update.bat" goto STAGE_FAIL
exit /b 0

:STAGE_FAIL
echo       ERROR: staging failed. Check free disk space.
exit /b 1


rem ============================================================ errors
:NO_VENV
echo ERROR: venv\Scripts\python.exe not found.
echo        Run setup.bat (or run.bat) once on this machine first.
goto FAIL

:NO_VERSION
echo ERROR: could not read APP_VERSION from core\config.py.
goto FAIL

:AUDIT_ERR
echo.
echo ERROR: one or more dependency groups cannot be installed offline.
echo        Re-run this builder with internet access, or run setup.bat
echo        first so that wheels\ can be recreated.
goto FAIL

:PYTHON_ERR
echo.
echo ERROR: the portable Python runtime could not be prepared.
goto FAIL

:MODEL_ERR
echo.
echo ERROR: a model that must be bundled is missing ^(see above^).
echo        Open the app online, download that model, close it, retry.
goto FAIL

:STAGE_ERR
echo.
echo ERROR: could not stage the package.
goto FAIL

:NO_TAR
echo ERROR: tar.exe not found ^(needs Windows 10 1803+^).
echo        The folder %OUTDIR% is complete; zip it manually.
if not defined NO_PAUSE pause
exit /b 1

:ZIP_ERR
echo ERROR: archiving failed. The staged folder %OUTDIR% is complete;
echo        copy that folder itself, or compress it manually.
goto FAIL

:FAIL
echo.
if not defined NO_PAUSE pause
exit /b 1
