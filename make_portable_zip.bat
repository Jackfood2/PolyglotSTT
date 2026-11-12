@echo off
setlocal
cd /d "%~dp0"

rem ============================================================
rem  Builds ..\PolyglotSTT-portable.zip for another (offline) PC.
rem
rem  Included: source, python\ (portable 3.11), wheels\ (all deps
rem            + freeze.txt with the exact versions)
rem  Excluded: venv\ (holds absolute paths, so setup_offline.bat
rem            rebuilds it), cache\ (pip download cache only),
rem            models_cache\, logs\
rem
rem  On the target PC: extract anywhere, run setup_offline.bat,
rem  then run.bat.
rem
rem  Note: this lists the top-level items explicitly, because tar's
rem  --exclude also matches nested folders (python\Lib\venv).
rem  Add any new top-level folder to the tar line below.
rem ============================================================

set "OUT=%~dp0..\PolyglotSTT-portable.zip"
echo Building "%OUT%" ...
if exist "%OUT%" del "%OUT%"

tar -a -c -f "%OUT%" .gitignore *.py *.bat *.vbs *.md *.txt *.ico *.png .git core engines services ui python wheels
if errorlevel 1 goto ZIP_FAIL

for %%A in ("%OUT%") do echo Done: %%~nxA  (%%~zA bytes)
echo.
echo Copy the zip to the other PC, extract it anywhere and run
echo setup_offline.bat once, then run.bat.
pause
exit /b 0

:ZIP_FAIL
echo ERROR: could not create the zip. Check free disk space and
echo that the parent folder is writable.
pause
exit /b 1
