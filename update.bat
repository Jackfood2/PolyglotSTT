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
echo.
echo Current version:
git log --oneline -3
for /f "delims=" %%V in ('git log -1 "--format=%%h %%s" 2^>nul') do set "NEW_VER=%%V"
for /f "delims=" %%V in ('git describe --tags --abbrev^=0 2^>nul') do set "NEW_TAG=%%V"
echo.
if defined OLD_VER if defined NEW_VER echo Updated: %OLD_VER% --^> %NEW_VER%
if defined NEW_TAG echo Tag now: %NEW_TAG%
git status --short --branch
echo.
echo Done. You can close this window and start run.bat.
pause
endlocal
