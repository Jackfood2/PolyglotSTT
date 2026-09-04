MoonshineSTT - Hold F2 to Transcribe (Moonshine v2)
==============================================

Portable on-device Speech-to-Text. No API keys, no internet needed after setup.

Folder:  C:\Users\Susan\Desktop\MoonshineSTT
Python:  python\python.exe (3.11.9, portable, self-contained)
Venv:    venv\   (all dependencies isolated here)
Model:   %LOCALAPPDATA%\moonshine_voice\Cache\download.moonshine.ai\model\...

Quick Start
-----------
1. Setup (first time only, needs internet):
     Double-click setup.bat
     - Creates venv
     - Installs dependencies
     - Downloads Moonshine v2 English model (~100 MB)
     - Installs VC++ Redist if needed (already done)

2. Run:
     Double-click run.bat
     Or:  venv\Scripts\python.exe moonshine_stt.py

Usage - Live tab
----------------
- Hold F2  -> record (button turns red, meter animates)
- Release F2 -> transcribe automatically
- Text is pasted into the focused window (Ctrl+V simulation)
- If no focus window, text stays in clipboard + history panel
- GUI buttons: RECORD (click to toggle), Clear, Copy
- History shows last transcriptions with timestamps
- Works system-wide while window is open (global F2 hook)
- Engine dropdown: Moonshine v2 (fast, English), Canary-1B (de/en/es/fr), or Whisper Large v3 (JA->EN translate, recommended for Japanese)
- Task/Src menus (for Canary + Whisper): Task=transcribe keeps source language, Task=translate outputs English; Src=source language (ja/zh/ko/auto/...)

Usage - SRT File tab (video/audio -> .srt)
-----------------------------------------
- Drag & drop a video/audio file (mp4/mkv/avi/mov/mp3/wav/...) or Browse
- Output defaults to the SAME folder as the source (same name, .srt);
  pick another folder with Browse... if needed
- Uses the Live tab engine (Moonshine = English, Canary = translate->EN for de/en/es/fr, Whisper Large v3 = JA->EN with native timestamps)
- Whisper Large v3: single native pass with timestamps (no VAD chunking), best for Japanese (slower than Turbo but actually translates)
- CPU slider: auto-detected cores, default 80% (CPU-only, no GPU needed)
- Generate SRT -> live progress bar + % + per-chunk log; Cancel anytime
- Cues are sentence-aware: max 2 lines x 42 chars, max 7s per cue,
  split on . ? ! and CJK punctuation, so the screen never floods
- ffmpeg ships inside the folder (imageio-ffmpeg wheel) - no install needed

Requirements
------------
- Windows 10/11
- Microphone (default input device)
- Python not required system-wide (portable copy included)

Troubleshooting
---------------
- No audio?  Check Windows mic privacy: Settings > Privacy > Microphone
- Model download fails? Check internet, then re-run setup.bat
- F2 not working? Some keyboards/laptops need Fn+F2. App must be running.
- Paste fails in admin/elevated windows? Copy from history and paste manually.
- Antivirus blocks pynput? Allow MoonshineSTT.

Files
-----
moonshine_stt.py   Main app, F2 hook, live queue + SRT file jobs
engine.py          Moonshine v2 wrapper (transcribe_without_streaming)
canary_engine.py   NVIDIA Canary-1B wrapper (local .nemo, transcribe/translate)
whisper_engine.py  Faster-Whisper large-v3 wrapper (transcribe/translate, native timestamps)
srt.py             SRT backend: ffmpeg extract, VAD segments (Moonshine/Canary) or native Whisper pass, cue packing
recorder.py        sounddevice capture (16kHz, int16)
input_sim.py       clipboard + Ctrl+V via SendInput
gui.py             customtkinter dark UI (Live tab + SRT File tab)
requirements.txt   pip deps (incl. tkinterdnd2 drag-drop, imageio-ffmpeg)
requirements-whisper.txt  Whisper deps (faster-whisper, ctranslate2, av, onnxruntime)
setup.bat          One-click setup
run.bat            One-click launch
python\            Portable Python 3.11.9
venv\              Isolated environment
wheels\            Offline reinstall wheels (incl. ffmpeg binary wheel + whisper wheels)
models_cache\      All models (Moonshine + canary-1b.nemo + whisper-models/large-v3) - travels in zip

Credits
-------
Moonshine v2 by Useful Sensors / Moonshine AI (MIT)
GUI: customtkinter
