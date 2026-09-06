# PolyglotSTT

Offline, on-device speech-to-text for Windows: hold-to-talk live dictation
plus video/audio-to-SRT subtitle generation. Three engines, multilingual
transcribe **and translate** (e.g. Japanese audio → English subtitles).
No API keys, no cloud, no admin rights needed once set up.

> Windows 10/11 only. The app window is titled "MoonshineSTT" (original
> project name); the repository is named PolyglotSTT because it now covers
> Moonshine v2, NVIDIA Canary-1B, and Whisper.

## Engines

| Engine | What it does | Languages |
|---|---|---|
| Moonshine v2 | Fast lightweight English ASR (Tiny → Medium-Streaming) | English |
| Canary-1B | Accurate ASR + translation to English | Source: German, English, Spanish, French (`auto` → English) |
| Whisper | Full-size Whisper, native timestamps, word-level alignment | ~100 source languages; `transcribe` keeps source, `translate` outputs English |

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

> **Note:** *Compute* (Auto/CPU/GPU) drives **inference only**. Burn
> encoding always uses x264 (CPU) unless an NVENC burn speed is selected.
> The SRT log notes this when Compute=GPU but a CPU burn speed is chosen.

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
  cancel, and a remaining-time countdown. *Boost quiet audio* applies a
  loudnorm pass before transcription for soft/uneven recordings (burns
  always re-encode from the original, so they are unaffected).
- **Per-tab engines** — Live, SRT File, and Note each pick their own
  engine + model. A tab you never explicitly set follows your last pick
  elsewhere; changing a tab while another session runs asks first
  (dual engines = dual RAM, stated upfront), otherwise idle engines are
  unloaded to make room. Mic dictation, an SRT batch, and Note can all run
  at once on different engines.
- **One primary button** — the action slot morphs with the queue: files
  still need SRTs → *Generate SRT*; every queued file already has one →
  *Convert to MP4*. Tick *Burn MP4 automatically after SRT* and one click
  generates the missing SRTs and hardcodes everything burnable (audio-only
  files, SRT-less failures, and already-burned videos are marked skipped,
  never failed — use Convert to MP4 with its overwrite prompts to redo).
  Generate/Convert, Cancel, and Open Folder share a single compact row.
- **Learned-history reset** — the ↺ SRT and ↺ Burn buttons in the Progress
  header wipe the timing/size baselines (per-model ETA, per-speed size
  estimates) back to defaults; they recalibrate on the next run.
- **Dark / light theme** — the ☀/☾ button in the header swaps the whole UI
  instantly (persisted across restarts, no rebuild needed).
- **Finish behavior** — *Shut down PC when done* force-powers off 60s after
  a fully successful job only (nothing on cancel/failure; abort any time
  with `shutdown /a`); otherwise an optional pop-up + window focus fires
  once per finished job. Both switches persist.
- **Burn SRT into MP4** — hardcode subtitles into a new `.mp4`
  (`<name>.burned.mp4`, any input: mp4/mkv/ts/…) with x264 two-pass sized
  to match the original file (audio copied when possible, CPU threads from
  the slider). The *Codec* switch (H.264 / HEVC) applies to the NVENC
  speeds (same GPU speed, ≈30% smaller, needs ~2016+ playback) and to the
  CPU speeds via x265 (≈30–40% smaller, much slower — yes, CPU does H.265).
  Without an NVIDIA GPU the menu offers CPU speeds only. The
  ⓘ button compares every speed × codec (accuracy, speed, playback,
  best-for) without trial burns. *Est. size* under the bitrate control learns from your past
  burns: every successful burn records its actual-vs-predicted ratio per
  speed (`burn_size.json`), so manual-kbps estimates calibrate to your
  content and settings — including unpredictable NVENC modes. With 2+ burns
  at a speed the estimate becomes a target box: type MB and the bitrate
  snaps to the exact kbps (1kbps precision, same for CPU and NVENC); drag
  the slider and the box follows back. The slider stays the coarse control
  and always wins on drag. Before any history the box still works off a
  pure analytic estimate (marked uncalibrated); empty input is ignored,
  never an error. Size-match mode simply shows ≈ source size. Burns also self-correct: with 2+ past burns at a
  speed, a learned systematic overshoot (typical for NVENC 1-pass) is
  compensated by requesting proportionally less, so repeat burns land on
  target instead of running hot forever — the log states the adjustment
  (`overshoot compensation: …`), x264 2-pass measures ≈1.00 and is
  untouched, and undershoot is never compensated. Subtitle size is adjustable (12–32) with an instant
  single-frame **Preview** before the full encode. Latin burns use Arial
  with outline; Japanese/Chinese/Korean burns automatically switch to
  MS Gothic (the only CJK setup this renderer draws — verified).
- **Burn speeds** — **Match size (2-pass x264)** exact ±1–3% size,
  **Fast (1-pass x264)** ~half time at ~±10%, **Fastest (ultrafast 1-pass)**
  several times faster but softer. NVIDIA-only: **Draft / Turbo / Balanced
  (NVENC)** fastest encodes, approximate size. Each menu entry explains
  itself; unavailable NVENC entries revert with a reason.
- **Honest progress bar** — never shows 100% mid-encode; holds at 99%
  through encode, goes to 100% only when the output file is verified
  non-empty. Reports avg fps and flags VFR overshoots.
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
- ~16 GB RAM if you use Canary-1B; Whisper runs on CPU
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
| `Systran/faster-whisper-large-v3` | ~3 GB | <https://huggingface.co/Systran/faster-whisper-large-v3> — auto-downloaded by Faster-Whisper; pick size in the Live tab *Model* row; each size downloads once on first selection. No Turbo offered: it cannot translate by design | `models_cache\whisper-models\` |

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
`Whisper (translate Japanese->English)`) always shows exactly
what the job will do. The status line counts down remaining time from an
estimate learned per model, per burn speed, and per audio length on this
machine (`srt_eta.json`: `whisper:large-v3` and `tiny` track separately,
as do `burn:match` vs `burn:fastest`, each in short/medium/long buckets
plus an overall average; older shared entries are reused as fallback),
then commits to it — linear to 99% / 0s, holding there if a job overruns —
and records the actual outcome so the next estimate is better. Every file
logs its total wall time with a realtime factor (`Total: 3:12 for 45.0s
audio (4.3x realtime)`, `Burn total: ...`), so speed regressions are
visible per job. The bar keeps moving even through Whisper's long single
pass and slow Canary chunks.

**Burn-in** — with SRTs generated, *Burn SRT into MP4* hardcodes each
queued video's subtitles into a `.burned.mp4` next to it (same folder or
the chosen output folder). Needs the SRT first or the file is skipped
with a notice. *Burn speed* offers **Match size (2-pass x264)** — exact
(±1–3%, slow, two full encodes), **Fast (1-pass x264)** — ~half the time
at ~±10% size drift, **Fastest (ultrafast 1-pass)** — several times
faster but visibly softer (fine for a quick check, not for keeps), plus
**Draft / Turbo / Balanced (NVENC)** GPU modes when an NVIDIA card is
detected (fastest encodes, approximate size; each menu entry explains
itself, and unavailable NVENC entries revert with a reason). *Video
bitrate* stays on Auto (size-match) or takes a manual 300–10000 kbps
(the log states the expected total, size-match explicitly off).
*Compute* (Auto/CPU/GPU, greyed without NVIDIA) picks Whisper/Canary
inference device with hot-reload. CPU threads come from
the slider. Drag *Subtitle size*, hit *Preview Frame*
to see one burned frame instantly (also needs that file's SRT), then run
the full encode. No SRT yet? Preview auto-transcribes just the
*Sample from / Sample len* slice (e.g. 15s @ 0:30) with the current live
engine settings and previews from that instead — a quick look before
committing to a full transcription. Existing outputs ask before
overwriting (all / per-file / abort); the app remembers your last tab.

## Project structure

```text
moonshine_stt.py   main app: F2 hook, live queue, SRT jobs, config
gui.py             customtkinter dark UI (Live + SRT File + Note tabs)
engine.py          Moonshine v2 wrapper
note_engine.py     Note-mode chunked recorder + async transcriber
recorder.py        microphone capture (16 kHz)
input_sim.py       clipboard + Ctrl+V insertion
srt.py             SRT backend: ffmpeg extract, VAD, word-anchored cue packing,
                 timing refine, batch queue, size-matched burn-in, ETA tracking
canary_engine.py   Canary-1B wrapper (NeMo, offline, language-validated)
whisper_engine.py  Faster-Whisper large-v3 wrapper (per-call task/lang overrides)
gpu.py             dGPU/VRAM/NVENC detection, device recommendation
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
  switch Engine to Whisper.
- **Model download fails** — check internet, re-run `setup.bat`; Canary
  also accepts a manually placed `canary-1b.nemo`. Whisper sizes download
  when first selected (watch the log for progress on slow links).
- **Burned CJK subtitles look plainer than Latin ones** — Japanese /
  Chinese / Korean burns use MS Gothic without outline (the only CJK
  setup this renderer draws); Latin burns use outlined Arial.
- **Compute=GPU but burn is CPU** — Compute drives inference only; pick
  an NVENC burn speed (Draft/Turbo/Balanced) for GPU burn encoding.
- **Progress bar holds at 99%** — intentional; the bar reports honestly
  during encode and only reaches 100% when the output file is verified.
- **Switching engines feels lighter the second time** — the idle heavy
  engine is unloaded to reclaim gigabytes of RAM; it reloads from cache
  when you switch back.
- **Auto-burn skipped a file** — audio-only files, files whose SRT step
  failed, and videos that already have a burned MP4 are marked skipped,
  never failed or overwritten; use *Convert to MP4* (with its overwrite
  prompts) to refresh an existing burn.
- **PC shut down after a job** — that is the *Shut down PC when done*
  switch (fully successful jobs only); abort a pending one within 60s
  with `shutdown /a` in a terminal, or uncheck the box.
- **Burned MP4 wildly bigger than the target on a phone/VFR file** — some
  containers lie about Duration (seen: 67s header on a 15min file). The
  probe cross-checks size×8÷bitrate and says so in the log (`probe:
  header duration disagreed…`); the burn itself was correct for the set
  bitrate. If one bad run poisoned the learned timing, hit ↺ Burn once
  and reburn to relearn.

## Changelog

### v1.2.17 (latest)

- Window title bar now shows the running version (`MoonshineSTT v1.2.17`), same source as the footer

### v1.2.16

- Reliability pass: engine loads carry a generation token, so a stale load can no longer overwrite the newly selected model; heavy-engine unloads take the inference lock before clearing the model
- Fixed a VAD buffer bug that re-added already-consumed audio (wrong speech regions plus quadratic slowdown on long files)
- Engine renamed from "Whisper Large v3" to "Whisper" everywhere, old configs migrate automatically; Whisper keeps its full language list, Canary stays auto/en/de/es/fr
- Live and Note queues are now truly bounded (overflow reports instead of growing without limit); recorder stop can no longer raise; Note worker pinned to its own session queue
- FFmpeg probes time out after 60s; burn staging uses a validated collision-proof temp name; GPU probe cache refreshes on compute change

### v1.2.15

- Note Save Audio button with location picker; unsaved session audio cleared on close, orphans swept on launch
- Fast launch: single millisecond dependency probe instead of four heavy imports (~5s to window, was ~18s)

### v1.2.14

- Note full-session MP3 archive: parallel WAV during record, background convert on stop
- Note pending auto-start: RECORD arms the wait, capture begins when the engine is ready

### v1.2.13

- Light theme engine/model dropdown text readable (were invisible)
- Tab strip surround and segment canvases follow the theme, border off
- Theme palette collision fixes plus a regression guard

### v1.2.12

- Two-tone tab highlight: pale lavender when selected in light mode, saturated purple in dark mode — readable in both

### v1.2.11

- Light theme: borderless card-blended tab strip (no dark surround)
- Engine/model dropdown text readable in light mode

### v1.2.10

- Per-tab engines: Live, SRT File, and Note each pick engine + model
- Shared-default mirror: untouched tabs follow your last pick
- Dual-engine confirm with stated RAM cost when another session runs
- Note fixes: engine errors surface instead of dropping chunks, honest counter, non-blocking STOP, editable box, append-on-record, mic watchdog, level meter, close save-prompt

### v1.2.9

- Engine readiness guards: recording and queued-clip transcription refuse
  while the engine is not ready instead of failing silently
- Deadlock fixes: engine reload restarts outside the lock; unload waits on
  the inference lock; Note recorder cuts chunks outside the audio lock
- Idle-unload respects queued clips and Note recording; CJK burns get
  outline styling; 16 kHz resampling uses linear interpolation
- Clipboard paste simplified (keeps new text, longer settle delay)

### v1.2.8

- Note-tab engine (`note_engine.py`): chunked recording with adaptive
  silence detection (40–80s chunks cut at natural pauses) + async worker
  — the v1.2.7 Note tab required this module to record at all
- Post-processing lockout fixes: Clear works after a job (progress reset),
  finished queues auto-reset so the same files can be dropped again,
  `set_srt_running`/`srt_done` re-enable every widget individually
- Fixed Save-as-TXT (`pathlib.Path` import was missing in `gui.py`)

### v1.2.7

- Note tab: record/transcribe in 40–80s chunks with timer, save TXT,
  copy-all (uses current Live-tab engine)
- Button-state hardening: per-widget try/except so one failure cannot
  leave Clear/Browse/start stuck disabled

### v1.2.6

- Critical concurrency fixes, language routing, codebase cleanup

### v1.2.5

- HEVC codec switch (H.264 / HEVC): NVENC speeds encode HEVC at the same
  GPU speed (≈30% smaller), CPU speeds switch x264/x265 (≈30–40% smaller,
  slower); per-codec learned history, estimate, and compensation.
  CPU-only boxes list CPU speeds only; the ⓘ button compares everything.
- NVENC quality hardening: explicit VBR rate control plus lookahead and
  adaptive quantization (same size, better bit distribution).
- Light-theme readability fixes (option menus, buttons, tab strip,
  scroll gaps) and a no-scroll tab flow (queue → actions → progress).

### v1.2.4

- Hotfix #2: MB target box is editable from the first burn (analytic
  fallback when uncalibrated, self-sharpening with history); empty input
  no longer logs a false error. Duration probe cross-checks size/bitrate
  against lying container headers (seen: 67s claimed on a 15min VFR file),
  fixing Auto budgets, progress, and learned history for such files.

### v1.2.3

- Hotfix: closed-loop overshoot compensation — burns at speeds with 2+
  learned samples and a measured overshoot now request proportionally less
  bitrate, so NVENC 1-pass (and any hot speed) lands on target instead of
  always oversizing. Logged per burn; x264 2-pass unaffected; the learner
  records against the requested rate so it converges instead of oscillating.

### v1.2.2

- Two-way target size: with 2+ burns at a speed the estimate becomes an
  editable MB box — type a size and the bitrate snaps to the exact kbps
  (1kbps precision, CPU and NVENC alike); drag the slider and the box
  follows back, slider always wins on drag. Typing implies manual mode.

### v1.2.1

- GitHub discoverability: 20 repository topics added (speech-to-text,
  whisper, subtitles, offline, NVENC, …) so the project surfaces in
  topic search alongside the existing description

### v1.2.0

- Dark / light theme (☀/☾ header toggle, instant, persisted) — including
  the tab strip, which is now explicitly themed (it stayed dark before)
- Learned-history reset: ↺ SRT and ↺ Burn buttons in the Progress header
- One-row action buttons (Generate/Convert + Cancel + Open Folder) and a
  tighter layout throughout (same controls, less scrolling)
- Footer + startup log carry the build version, so any screenshot or log
  identifies the exact checkout

### v1.1.1

- Learned burn-size estimates: every successful burn records its
  actual-vs-predicted ratio per speed (`burn_size.json`), and the burn card
  shows a live `Est. size` that follows the kbps slider, speed, and queue —
  including unpredictable NVENC modes. No history for a speed shows
  `-- no historical data --` instead of guessing; size-match shows
  ≈ source size.

### v1.1.0

- Accurate ETAs: history is now keyed per Whisper model size and per burn
  speed (`whisper:tiny` vs `whisper:large-v3`, `burn:match` vs
  `burn:fastest` no longer poison each other); existing shared history is
  reused as fallback, with sane first-run defaults per size/speed
- Per-file total time in the log: `Total: … for … audio (…x realtime)` for
  SRT jobs and `Burn total: …` for burns
- Fixed invalid SRT timestamps at minute boundaries (`00:00:60,000` → total
  millisecond arithmetic in both the writer and the sample-shift path)
- Preview transcriptions now honor the audio-boost toggle, matching what
  Generate will hear
- Starting a new job aborts a pending auto-shutdown instead of letting the
  PC die mid-job
- Drag-dropping files mid-run no longer clobbers the live progress message

### v1.0.9

- Carries the v1.0.8 menu fix: all 6 burn speeds show (verified in this
  build, including label round-trips for every speed id)
- Audio boost toggle: optional loudnorm pass before SRT transcription for
  quiet/uneven recordings (burns re-encode from the original, unaffected)
- One-button SRT/MP4 flow: the action slot morphs Generate SRT <->
  Convert to MP4 with the queue; *Burn MP4 automatically after SRT*
  generates + hardcodes in one run (audio-only / SRT-less / already-burned
  files are marked skipped, never failed or overwritten)
- Finish behavior: *Shut down PC when done* (fully successful jobs only,
  60s delay, `shutdown /a` aborts) or a one-shot pop-up + window focus
- Reliability: Moonshine engine unloads on switch (RAM reclaim for all
  three engines), live progress during model loads, Compute selector
  stores `gpu` consistently (survives restart, menu + burn note agree),
  burn fps stats recovered on fast encodes

### v1.0.8

- Fixed Burn speed menu `NameError` in `gui.py` (imported as `_BSL` but read
  as `BURN_SPEED_LABELS`): menu silently fell back to 3 CPU entries, hiding
  the Draft / Turbo / Balanced NVENC options — all 6 speeds now show

### v1.0.7

- Fixed duplicate output-size check in burn path (srt.py)
- NVENC Turbo burn speed now properly recognized in config validation
- Honest 100%: progress bar only completes when output file verified non-empty
- Speed label updated: "Match size (2-pass x264)" for clarity
- Burn encode logs avg fps and flags VFR overshoots
- Compute=GPU note: SRT log warns when CPU burn is used with GPU inference

### v1.0.6

- Cancel-crash fix: `Event` object shadowed method call → renamed to `_cancel_srt_job`
- Overwrite guards: askyesnocancel all/per-file/abort for SRT and burn outputs
- Tab memory: app remembers last used tab on restart
- Burn kbps: manual bitrate slider (300–10000) with Auto default
- Compute selector: Auto/CPU/GPU with hot-reload and model-safe switching
- NVENC Turbo burn speed added between Draft and Balanced

### v1.0.5

- NVENC draft/balanced burn modes with contextual preset explanations

### v1.0.4

- Burn speed modes: match 2-pass, fast/fastest 1-pass

### v1.0.3

- Working CJK subtitles via MS Gothic minimal style (verified pixel-level)

### v1.0.2

- ETA: length-bucketed baseline, committed linear countdown to 99%

### v1.0.1

- GPU auto-detect (VRAM-gated CUDA/CPU) + live progress pump + GPU setup notes
- Preview: auto-transcribe sample slice when no SRT + start/length controls
- Model manager: red Delete buttons when deletable

### v1.0.0

- Initial release: Moonshine v2 + Canary-1B + Whisper
- Live F2 dictation + SRT batch generation
- Size-matched burn-in with x264 two-pass

## Credits & licenses

- Moonshine v2 by Useful Sensors / Moonshine AI (MIT).
- Whisper large-v3 by OpenAI (MIT weights); Faster-Whisper by SYSTRAN.
- Canary-1B by NVIDIA — check its license on Hugging Face before
  commercial use.
- GUI: CustomTkinter; audio: sounddevice/soundfile/imageio-ffmpeg.
