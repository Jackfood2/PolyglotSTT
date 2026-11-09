@echo off
setlocal EnableExtensions
title PolyglotSTT update 1.4.15 - 1.4.16
cd /d "%~dp0"

rem One-shot updater for machines that are still on v1.4.15.
rem It is safe to run twice: when the tree is already at v1.4.16 it exits.
rem This release adds itn.py only - no new Python packages and no model
rem downloads, so nothing needs pip-installing.

where git >nul 2>&1
if errorlevel 1 (
    echo ERROR: git not found in PATH. Install Git for Windows first.
    pause
    exit /b 1
)
if not exist ".git" (
    echo ERROR: run this file from the PolyglotSTT folder.
    pause
    exit /b 1
)

set "WANT=1.4.16"
call :READ_VERSION CUR
echo Current version: %CUR%
echo Target version:  %WANT%
if "%CUR%"=="%WANT%" (
    echo.
    echo Already on v%WANT% - nothing to do.
    goto :DONE
)

echo.
echo Fetching updates from GitHub ...
git fetch origin --tags
if errorlevel 1 (
    echo Fetch failed. Check your internet connection.
    pause
    exit /b 1
)

git pull --ff-only
if errorlevel 1 (
    echo Pull failed. You may have local changes - run "git status" to inspect,
    echo or use update.bat which reports the same error without the version check.
    pause
    exit /b 1
)

call :READ_VERSION CUR
if not "%CUR%"=="%WANT%" (
    echo.
    echo WARNING: pulled, but core\config.py reports version %CUR% instead of %WANT%.
    echo The remote branch may have moved on. Run update.bat to get the newest.
    goto :DONE
)

echo.
echo Updated to v%WANT%. No new Python packages in this release.
echo Starting PolyglotSTT ...

:DONE
echo.
git log --oneline -1
echo.
echo Press any key to close. Start the app with run.bat.
if not "%POLYGLOT_NO_PAUSE%"=="1" pause
endlocal
exit /b 0

:READ_VERSION
rem %1 = variable to receive the version found in core\config.py
set "%~1="
set "RAW="
if not exist "core\config.py" goto :EOF
for /f "tokens=2 delims==" %%V in ('findstr /c:"APP_VERSION = " core\config.py') do set "RAW=%%V"
if not defined RAW goto :EOF
set "RAW=%RAW:"=%"
set "RAW=%RAW: =%"
call set "%~1=%%RAW%%"
goto :EOF
