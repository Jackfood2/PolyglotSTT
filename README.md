# PolyglotSTT

Offline, on-device speech-to-text for Windows: hold-to-talk live dictation
plus video/audio-to-SRT subtitle generation. Three engines, multilingual
transcribe **and translate** (e.g. Japanese audio → English subtitles).
No API keys, no cloud, no admin rights needed once set up.

> Windows 10/11 only. The app window is titled "MoonshineSTT" (original
> project name); the repository is named PolyglotSTT because it now covers
> Moonshine v2, NVIDIA Canary-1B, and Whisper Large v3.

## Engines

| Engine | What it does | Languages |
|---|---|---|
| Moonshine v2 | Fast lightweight English ASR (Tiny → Medium-Streaming) | English |
| Canary-1B | Accurate ASR + translation to English | Source: German, English, Spanish, French (`auto` → English) |
| Whisper Large v3 | Full-size Whisper, native timestamps, word-level alignment | ~100 source languages; `transcribe` keeps source, `translate` outputs English |

**Why not Turbo?** OpenAI trained `large-v3-turbo` on transcription data
only, explicitly excluding translation data — the `<|translate|>` token is
untrained, so Turbo silently *transcribes* instead of translating. Verified
here with two independent conversions (correct prompt, Japanese output
anyway). `large-v3` is therefore the smallest Whisper that translates.
Canary-1B likewise only carries `de/en/es/fr` language tokens — anything
else fails fast with a message pointing at Whisper. The GUI only offers
each engine the languages it can actually do.

## Features

- **Live tab** — hold `F2` to record, release to transcribe; text is typed
  into the focused window (clipboard fallback), with level meter, waveform,
  history, and per-engine Task / source-language selectors.
- **SRT File tab** — drag & drop a video/audio file, pick an output folder,
  generate `.srt` with live progress bar, per-chunk log, and cancel.
- **Accurate timing** — Whisper cues are snapped to measured speech energy,
  anchored to per-word timestamps (including one-word cues), silence
  hallucinations and `[Music]` markers are dropped; sentence-aware packing
  (max 2×42 chars, 7s per cue).

## Requirements

- Windows 10/11, microphone (default input device).
- **Python 3.11** ([python.org](https://www.python.org/downloads/);
  `setup.bat`/`run.bat` prefer a local portable `python\` folder if present,
  otherwise fall back to system Python 3.11).
- Disk for models (see below) and patience on first run (large downloads).
- ~16 GB RAM if you use Canary-1B; Whisper Large v3 runs on CPU
  (slower than Turbo, but it actually translates).

## Install

```bat
git clone https://github.com/jackfood2/PolyglotSTT.git
cd PolyglotSTT
setup.bat      :: creates venv, installs deps, downloads the Moonshine model
run.bat        :: launches the app (rechecks deps each start)
```

`setup.bat` installs online with `pip`, falling back to a local `wheels\`
folder when present, then caches wheels there for fully-offline reinstalls.

## Models to download

Only Moonshine is fetched by `setup.bat`. The heavy engines download
**automatically on first use** into `models_cache\` (offline-first: cached
copies are reused with no network). Prefer to pre-fetch or have a slow
connection — grab them manually:

| Model | Size | Source | Cache location |
|---|---|---|---|
| Moonshine medium-streaming (default) + tiny-streaming | ~155 MB | fetched by `setup.bat` via `moonshine_voice.download` | `models_cache\download.moonshine.ai\` |
| `nvidia/canary-1b` | ~3.9 GB | <https://huggingface.co/nvidia/canary-1b> — or drop a `canary-1b.nemo` file at `models_cache\canary-1b\canary-1b.nemo` (>100 MB is picked up automatically, fully offline) | `models_cache\canary-1b\` (+ `models_cache\huggingface\`, `models_cache\torch\`) |
| `Systran/faster-whisper-large-v3` | ~3 GB | <https://huggingface.co/Systran/faster-whisper-large-v3> (auto-downloaded by Faster-Whisper) | `models_cache\whisper-models\` |

`ffmpeg` ships via the `imageio-ffmpeg` wheel (no system install needed).
Everything the app needs lives inside the project folder (except the
system Python install), so zipping the folder makes a portable offline copy.

## Usage

**Live dictation** — pick an engine; for Canary/Whisper pick Task
(`transcribe` keeps the source language, `translate` outputs English) and
source language. Hold `F2`, speak, release. Typing method / suffix options
are in the Live tab; settings persist in `moonshine_config.json`.

**SRT subtitles** — open the *SRT File* tab, drop a video/audio file
(mp4/mkv/avi/mov/mp3/wav/m4a/…), optionally set an output folder (default:
next to the source), set *Input/Output Language* (enabled for Canary and
Whisper only; `translate` always outputs English), tune CPU threads, then
*Generate SRT*. The engine label (e.g. `Whisper Large v3 (translate
Japanese->English)`) always shows exactly what the job will do.

## Project structure

```text
moonshine_stt.py   main app: F2 hook, live queue, SRT jobs, config
gui.py             customtkinter dark UI (Live + SRT File tabs)
engine.py          Moonshine v2 wrapper
recorder.py        microphone capture (16 kHz)
input_sim.py       clipboard + Ctrl+V insertion
srt.py             SRT backend: ffmpeg extract, VAD, cue packing, timing refine
canary_engine.py   Canary-1B wrapper (NeMo, offline, language-validated)
whisper_engine.py  Faster-Whisper large-v3 wrapper (per-call task/lang overrides)
setup.bat          one-click setup (venv, deps, wheels cache, Moonshine model)
run.bat            one-click launch (dependency self-check)
requirements*.txt  dependency pins (base / Canary / Whisper)
```

## Troubleshooting

- **Canary fails to load / OOM** — it needs ~16 GB RAM; use Moonshine v2
  or Whisper on smaller machines.
- **"No speech detected / transcribed"** — check the mic privacy setting
  and that the file actually contains speech (trailers: dialogue, not just
  music/SFX).
- **F2 does nothing** — some laptops need `Fn+F2`; the app must be running.
- **Paste fails in admin windows** — copy from Transcription History
  manually (UAC focus protection).
- **Japanese + Canary** — unsupported by the model (only de/en/es/fr);
  switch Engine to Whisper Large v3.
- **Model download fails** — check internet, re-run `setup.bat`; Canary
  also accepts a manually placed `canary-1b.nemo`.

## Credits & licenses

- Moonshine v2 by Useful Sensors / Moonshine AI (MIT).
- Whisper large-v3 by OpenAI (MIT weights); Faster-Whisper by SYSTRAN.
- Canary-1B by NVIDIA — check its license on Hugging Face before
  commercial use.
- GUI: CustomTkinter; audio: sounddevice/soundfile/imageio-ffmpeg.
