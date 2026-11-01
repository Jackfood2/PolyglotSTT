PolyglotSTT (formerly MoonshineSTT)
===================================
This is the legacy quick-start file. See README.md for full documentation.

What it is
----------
Offline, on-device speech-to-text for Windows 10/11:
- Live hold-to-talk dictation (F2)
- SRT subtitle generation from video/audio
- Note-taking mic transcription
- Import tab for audio/video files
No API keys, no cloud, no admin rights needed after setup.

Quick start
-----------
1. Install Python 3.11 from python.org if you do not have it.
2. From the project folder, run:
     setup.bat
     run.bat
   setup.bat creates the venv, installs dependencies, and pre-fetches the
   Moonshine English model. Whisper and Canary models download on first use
   into models_cache\.
3. Hold F2 to record; release to transcribe.

Fresh-install defaults
----------------------
- Engine: Moonshine v2 (English)
- Whisper: large-v3-turbo, transcribe mode (fast, ~1.6 GB)
- Translation: switch the Whisper model to large-v3 and set Task to
  Translate. large-v3-turbo is transcription-only by design.

Requirements
------------
- Windows 10/11, microphone (default input device)
- Python 3.11 (setup.bat checks for it)
- Disk space for models; ~16 GB RAM if you use Canary-1B

Troubleshooting and details: README.md
