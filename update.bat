@echo off
setlocal
cd /d "%~dp0"
where git >nul 2>&1
if errorlevel 1 (
    echo ERROR: git not found in PATH. Install Git for Windows first.
    pause
    exit /b 1
)
echo Updating PolyglotSTT in %CD% ...
for /f "delims=" %%V in ('git log -1 "--format=%%h %%s" 2^>nul') do set "OLD_VER=%%V"
for /f "delims=" %%V in ('git describe --tags --abbrev^=0 2^>nul') do set "OLD_TAG=%%V"
for /f "delims=" %%V in ('git rev-parse HEAD 2^>nul') do set "OLD_SHA=%%V"
if defined OLD_VER echo Before: %OLD_VER%
if defined OLD_TAG echo Tag before: %OLD_TAG%
git fetch origin --tags
if errorlevel 1 (
    echo Fetch failed. Check your internet connection.
    pause
    exit /b 1
)
git pull --ff-only
if errorlevel 1 (
    echo Pull failed. You may have local changes - run "git status" to inspect.
    pause
    exit /b 1
)
rem A new release usually brings new engine packages (Whisper, Canary, Qwen).
rem Refresh only the requirement lists that actually changed in this update,
rem so an update does not reinstall torch/nemo every single time.
set "DEPS_CHANGED="
if defined OLD_SHA (
    for /f "delims=" %%F in ('git diff --name-only %OLD_SHA% HEAD -- "requirements*.txt" 2^>nul') do (
        if not "%%F"=="" call :INSTALL_DEPS "%%F"
    )
)
echo.
echo Current version:
git log --oneline -3
for /f "delims=" %%V in ('git log -1 "--format=%%h %%s" 2^>nul') do set "NEW_VER=%%V"
for /f "delims=" %%V in ('git describe --tags --abbrev^=0 2^>nul') do set "NEW_TAG=%%V"
echo.
if defined OLD_VER if defined NEW_VER echo Updated: %OLD_VER% --^> %NEW_VER%
if defined NEW_TAG echo Tag now: %NEW_TAG%
if defined DEPS_CHANGED echo Dependencies refreshed. New models download on first use.
git status --short --branch
echo.
echo Done. You can close this window and start run.bat.
pause
endlocal
exit /b 0

:INSTALL_DEPS
rem %1 = requirements file that changed between the old and new commit.
if not exist "venv\Scripts\python.exe" goto :EOF
if not exist "%~1" goto :EOF
set "DEPS_CHANGED=1"
echo.
echo Installing %~nx1 ...
if /i "%~nx1"=="requirements-canary.txt" (
    "%~dp0venv\Scripts\python.exe" -m pip install -r "%~1" --extra-index-url https://download.pytorch.org/whl/cpu
) else (
    "%~dp0venv\Scripts\python.exe" -m pip install -r "%~1"
)
if errorlevel 1 (
    echo Warning: could not install %~nx1 - run setup.bat once.
) else (
    echo Installed %~nx1
)
goto :EOF
