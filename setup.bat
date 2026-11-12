@echo off
setlocal DisableDelayedExpansion
cd /d "%~dp0"
rem Put this project's venv and portable Python first, then keep the user's PATH.
set "POLYSTT_USER_PATH=%PATH%"
set "PATH=%~dp0venv\Scripts;%~dp0python;%PATH%"
rem pip 26 resolves every PATH entry while installing, so one locked or
rem vanished drive aborts venv/dep installs. Do that work with a minimal
rem PATH and restore the user's PATH before the GPU probe.
set "POLYSTT_MIN_PATH=%~dp0venv\Scripts;%~dp0python;%SystemRoot%\System32;%SystemRoot%;%SystemRoot%\System32\Wbem"
set "PYTHONPATH="
set "PYTHONHOME="
set "PIP_FIND_LINKS="
rem Keep pip's download cache inside this folder (portable / offline copies).
set "PIP_CACHE_DIR=%~dp0cache\pip"
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
rem From here on only the project's venv/python and System32 are needed.
set "PATH=%POLYSTT_MIN_PATH%"
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
if not exist "requirements-canary.txt" goto INSTALL_GRANITE2B
echo Installing Canary dependencies (torch+nemo, large)...
"%~dp0venv\Scripts\python.exe" -m pip install -r requirements-canary.txt --extra-index-url https://download.pytorch.org/whl/cpu
if not errorlevel 1 goto INSTALL_GRANITE2B
echo Online Canary install failed, trying offline wheels...
"%~dp0venv\Scripts\python.exe" -m pip install --no-index --find-links="%~dp0wheels" -r requirements-canary.txt
if errorlevel 1 echo Warning: Canary deps failed. Moonshine still works.
:INSTALL_GRANITE2B
if not exist "requirements-granite2b.txt" goto INSTALL_WHISPER
echo Installing Granite 2B dependencies (torch+transformers)...
"%~dp0venv\Scripts\python.exe" -m pip install -r requirements-granite2b.txt --extra-index-url https://download.pytorch.org/whl/cpu
if not errorlevel 1 goto INSTALL_WHISPER
echo Online Granite 2B install failed, trying offline wheels...
"%~dp0venv\Scripts\python.exe" -m pip install --no-index --find-links="%~dp0wheels" -r requirements-granite2b.txt
if errorlevel 1 echo Warning: Granite 2B deps failed. Other engines still work.
:INSTALL_WHISPER
if not exist "requirements-whisper.txt" goto INSTALL_QWEN
echo Installing Whisper dependencies (faster-whisper, CPU)...
"%~dp0venv\Scripts\python.exe" -m pip install -r requirements-whisper.txt
if not errorlevel 1 goto INSTALL_QWEN
echo Online Whisper install failed, trying offline wheels...
"%~dp0venv\Scripts\python.exe" -m pip install --no-index --find-links="%~dp0wheels" -r requirements-whisper.txt
if errorlevel 1 echo Warning: Whisper deps failed. Moonshine still works.
:INSTALL_QWEN
if not exist "requirements-qwen.txt" goto GPU_TORCH
echo Installing Qwen3-ASR dependencies (transcribe.cpp, CPU+Vulkan)...
"%~dp0venv\Scripts\python.exe" -m pip install -r requirements-qwen.txt
if not errorlevel 1 goto GPU_TORCH
echo Online Qwen3-ASR install failed, trying offline wheels...
"%~dp0venv\Scripts\python.exe" -m pip install --no-index --find-links="%~dp0wheels" -r requirements-qwen.txt
if errorlevel 1 echo Warning: Qwen3-ASR deps failed. Other engines still work.
goto GPU_TORCH
:GPU_TORCH
rem nvidia-smi lives outside the minimal PATH, so restore the user's PATH.
set "PATH=%~dp0venv\Scripts;%~dp0python;%POLYSTT_USER_PATH%"
rem NVIDIA dGPU present but torch blind to it -> install the CUDA build once.
where nvidia-smi >nul 2>&1
if errorlevel 1 goto CACHE_WHEELS
"%~dp0venv\Scripts\python.exe" -c "import torch,sys; sys.exit(0 if torch.cuda.is_available() else 1)" >nul 2>&1
if not errorlevel 1 goto CACHE_WHEELS
echo NVIDIA GPU found but torch is CPU-only - installing CUDA build (~2.5GB)...
"%~dp0venv\Scripts\python.exe" -m pip install -U torch torchaudio --index-url https://download.pytorch.org/whl/cu126
if errorlevel 1 (
    echo Online CUDA torch failed, trying default index...
    "%~dp0venv\Scripts\python.exe" -m pip install -U torch
)
if errorlevel 1 echo Warning: GPU torch install failed - CPU remains fully working.
goto CACHE_WHEELS
:INSTALL_OK
echo Dependencies installed.
:CACHE_WHEELS
echo Caching wheels for offline use...
if not exist "wheels" mkdir "wheels"
"%~dp0venv\Scripts\python.exe" -m pip download -r requirements.txt -d wheels --quiet
if exist "requirements-canary.txt" "%~dp0venv\Scripts\python.exe" -m pip download -r requirements-canary.txt -d wheels --extra-index-url https://download.pytorch.org/whl/cpu --quiet
if exist "requirements-granite2b.txt" "%~dp0venv\Scripts\python.exe" -m pip download -r requirements-granite2b.txt -d wheels --extra-index-url https://download.pytorch.org/whl/cpu --quiet
if exist "requirements-whisper.txt" "%~dp0venv\Scripts\python.exe" -m pip download -r requirements-whisper.txt -d wheels --quiet
if exist "requirements-qwen.txt" "%~dp0venv\Scripts\python.exe" -m pip download -r requirements-qwen.txt -d wheels --quiet
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
echo Setup complete! Run run.bat to start.
echo Keep wheels\ and models_cache\ with the project for offline reinstalls.
echo No API keys or admin rights are needed after setup.
echo ============================================
pause
endlocal
