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

## GPU acceleration (automatic)

CPU is the default and always works. When an NVIDIA dedicated GPU is
present, the app detects it (model, driver, free VRAM) and uses **both**:
inference moves to the GPU while audio prep, VAD, cue packing, ffmpeg and
Moonshine stay on CPU.

- **Whisper** → CUDA `float16` on cards with room, `int8` on smaller ones,
  CPU otherwise. Override in `moonshine_config.json` with
  `"whisper_device": "cpu"` (or `"cuda"` to force it — falls back to CPU
  with a log line when unusable, never crashes).
- **Canary-1B** → CUDA only with ~10 GB+ free VRAM (fp32-hungry; small
  cards stay on CPU instead of OOM-crashing mid-load).
- The SRT log states the chosen device per job, e.g.
  `Transcribing … (translate ja->en, cuda float16, native timestamps)`.
- `setup.bat` installs a CUDA torch build automatically when an NVIDIA
  GPU is found but torch is CPU-only; `run.bat` only warns in that case
  (never a surprise 2.5 GB download on launch). Faster-Whisper's bundled
  CTranslate2 already ships CUDA kernels — no extra install for it.

## Features

- **Live tab** — hold `F2` to record, release to transcribe; text is typed
  into the focused window (clipboard fallback), with level meter, waveform,
  history, and per-engine Task / source-language selectors. The *Model* row
  follows the engine: Moonshine sizes, downloadable Whisper sizes (Tiny
  75MB → Large v3 3GB, fetched once on first pick), or the fixed Canary-1B.
  Recording is politely refused for a moment while a model is swapping.
- **SRT File tab** — queue one or many video/audio files (batch runs one by
  one, failures don't stop the queue), pick an output folder, generate
  `.srt` with live progress bar, per-file status list, per-chunk log,
  cancel, and a remaining-time countdown.
- **Burn SRT into MP4** — hardcode subtitles into a new `.mp4`
  (`<name>.burned.mp4`, any input: mp4/mkv/ts/…) with x264 two-pass sized
  to match the original file (audio copied when possible, CPU threads from
  the slider). Subtitle size is adjustable (12–32) with an instant
  single-frame **Preview** before the full encode. Latin burns use Arial
  with outline; Japanese/Chinese/Korean burns automatically switch to
  MS Gothic (the only CJK setup this renderer draws — verified).
- **Live + jobs at once** — hold-to-talk dictation keeps working while an
  SRT/burn job runs (separate threads; engines serialize inference, so
  everything just shares CPU).
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
| Moonshine (all 6 sizes: tiny → medium-streaming) | ~25–260 MB each | fetched on first pick via the *Model* row (`setup.bat` pre-fetches medium + tiny-streaming) | `models_cache\download.moonshine.ai\` |
| `nvidia/canary-1b` | ~3.9 GB | <https://huggingface.co/nvidia/canary-1b> — or drop a `canary-1b.nemo` file at `models_cache\canary-1b\canary-1b.nemo` (>100 MB is picked up automatically, fully offline) | `models_cache\canary-1b\` (+ `models_cache\huggingface\`, `models_cache\torch\`) |
| `Systran/faster-whisper-*` (tiny/base/small/medium/large/v1/v2/v3) | 75 MB – 3 GB | <https://huggingface.co/Systran/faster-whisper-large-v3> (auto-downloaded by Faster-Whisper) — pick the size in the Live tab *Model* row when the Whisper engine is active; each size downloads once on first selection. No Turbo offered: it cannot translate by training design; no distil/`.en` (English-only) | `models_cache\whisper-models\` |

`ffmpeg` ships via the `imageio-ffmpeg` wheel (no system install needed).
Everything the app needs lives inside the project folder (except the
system Python install), so zipping the folder makes a portable offline copy.

## Usage

**Live dictation** — pick an engine; for Canary/Whisper pick Task
(`transcribe` keeps the source language, `translate` outputs English) and
source language. The *Model* row right underneath always belongs to the
active engine (Moonshine sizes / downloadable Whisper sizes / fixed
Canary), with ✓ = stored locally and ↓ = downloads on first pick; the
*Manage…* button shows every model, its disk size, and lets you delete
downloaded ones individually (or all at once) to save space. Hold `F2`,
speak, release. Typing method / suffix options are in the Live tab;
settings persist in `moonshine_config.json`.

**SRT subtitles** — open the *SRT File* tab, drop video/audio files
(mp4/mkv/avi/mov/ts/mp3/wav/m4a/… — multi-select for batch), optionally set
an output folder (default: next to the source), set *Input/Output
Language* (enabled for Canary and Whisper only; `translate` always outputs
English), tune CPU threads, then *Generate SRT*. The engine label (e.g.
`Whisper Large v3 (translate Japanese->English)`) always shows exactly
what the job will do. The status line counts down remaining time from an
estimate learned per engine and audio length on this machine
(`srt_eta.json`: short/medium/long buckets plus an overall average), then
commits to it — linear to 99% / 0s, holding there if a job overruns — and
records the actual outcome so the next estimate is better. The bar keeps
moving even through Whisper's long single pass and slow Canary chunks.

**Burn-in** — with SRTs generated, *Burn SRT into MP4* hardcodes each
queued video's subtitles into a `.burned.mp4` next to it (same folder or
the chosen output folder). Needs the SRT first or the file is skipped
with a notice. *Burn speed* offers **Match size (2-pass)** — exact
(±1–3%, slow, two full encodes), **Fast (1-pass)** — ~half the time at
~±10% size drift, or **Fastest (ultrafast 1-pass)** — several times
faster but visibly softer (fine for a quick check, not for keeps; CPU
threads come from the slider). Drag *Subtitle size*, hit *Preview Frame*
to see one burned frame instantly (also needs that file's SRT), then run
the full encode. No SRT yet? Preview auto-transcribes just the
*Sample from / Sample len* slice (e.g. 15s @ 0:30) with the current live
engine settings and previews from that instead — a quick look before
committing to a full transcription.

## Project structure

```text
moonshine_stt.py   main app: F2 hook, live queue, SRT jobs, config
gui.py             customtkinter dark UI (Live + SRT File tabs)
engine.py          Moonshine v2 wrapper
recorder.py        microphone capture (16 kHz)
input_sim.py       clipboard + Ctrl+V insertion
srt.py             SRT backend: ffmpeg extract, VAD, word-anchored cue packing,
                 timing refine, batch queue, size-matched burn-in, ETA tracking
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
  also accepts a manually placed `canary-1b.nemo`. Whisper sizes download
  when first selected (watch the log for progress on slow links).
- **Burned CJK subtitles look plainer than Latin ones** — Japanese /
  Chinese / Korean burns use MS Gothic without outline (the only CJK
  setup this renderer draws); Latin burns use outlined Arial.
- **Switching engines feels lighter the second time** — the idle heavy
  engine is unloaded to reclaim gigabytes of RAM; it reloads from cache
  when you switch back.

## Credits & licenses

- Moonshine v2 by Useful Sensors / Moonshine AI (MIT).
- Whisper large-v3 by OpenAI (MIT weights); Faster-Whisper by SYSTRAN.
- Canary-1B by NVIDIA — check its license on Hugging Face before
  commercial use.
- GUI: CustomTkinter; audio: sounddevice/soundfile/imageio-ffmpeg.
