# PolyglotSTT

Offline, on-device speech-to-text for Windows: hold-to-talk live dictation
plus video/audio-to-SRT subtitle generation. Six engines, multilingual
transcription — and **translation to English** on Whisper, Canary-1B and
Granite Speech 5.0 2B. No API keys, no cloud, no admin rights needed once
set up.

> Windows 10/11 only. The app window is titled "MoonshineSTT" (original
> project name); the repository is named PolyglotSTT because it now covers
> Moonshine v2, NVIDIA Canary-1B, Whisper, Qwen3-ASR and the Granite Speech
> family (470M TurboCTC and 5.0 2B).

## Engines

| Engine | What it does | Languages |
|---|---|---|
| Moonshine v2 | Fast lightweight English ASR (Tiny → Medium-Streaming) | English |
| Canary-1B | Accurate ASR + translation to English | Source: German, English, Spanish, French (`auto` → English) |
| Whisper | Full-size Whisper, native timestamps, word-level alignment | ~100 source languages; `transcribe` keeps source, `translate` outputs English |
| Qwen3-ASR | 1.7B GGUF ASR (Q8_0 → Q4_K_M), 30 languages, transcription only | 30 languages incl. zh/yue/ja/ko/ar/hi/th/vi/tr/… ; keeps the spoken language |
| Granite 5.0 TurboCTC | 470M GGUF CTC ASR (Q8_0 → Q4_K_M), English only, transcription only | English |
| Granite Speech 5.0 2B | 2B speech-LLM, punctuation + keyword biasing, translate to English | en/fr/de/es/pt/ja (`auto` or pinned); English output |

**Turbo is transcription-only.** OpenAI trained `large-v3-turbo` on
transcription data only, explicitly excluding translation data — its
`<|translate|>` token is untrained, so it silently *transcribes* instead of
translating. It is selectable for `transcribe` (about 2x faster than
large-v3 on CPU), but the app blocks it in `translate` mode and keeps
large-v3 for English output. Canary-1B likewise only carries `de/en/es/fr`
language tokens — anything else fails fast with a message pointing at
Whisper. **Qwen3-ASR is transcription-only**: it has no translate task and
emits no timestamps, so SRT runs it over VAD chunks (like Canary) and takes
cue timing from the chunk boundaries. **Granite 5.0 TurboCTC** is the speed
engine: a 470M encoder-only CTC model that runs roughly 9x faster than
Qwen3-ASR here (measured 24x realtime on Vulkan) and costs 700 MB of RAM,
which makes it the best choice for Live dictation. It is English-only,
cannot translate, emits no timestamps, and — being CTC — produces no
punctuation or capitalisation, so prefer Qwen3-ASR or Whisper for subtitles
and notes you intend to read. It also has no vocabulary or prompt input, so
the Dictionary does not apply to it. The GUI only offers each engine the
languages it can actually do.

**Granite Speech 5.0 2B is the accurate multilingual sibling.** It is a full
speech-LLM (conformer encoder + projector + Granite 4.0 1B), so unlike the
470M CTC model it produces punctuation and capitalisation, accepts the
Dictionary as real keyword biasing (up to 2000 terms plus free-text
context), and can translate into English from all six spoken languages. It
has no timestamps, so SRT and Import run it over VAD chunks like
Canary/Qwen3-ASR, and it is CPU-slow for its size: expect roughly 0.9x
realtime on the float32 CPU path. Two backends are offered: the default
Transformers snapshot (4.8 GB, the most accurate, works on CPU and CUDA) and
optional llama.cpp GGUF builds (Q8_0 3.0 GB and Q6_K 2.6 GB on disk, about
3x faster here and much lighter in RAM). The GGUFs are downloaded only when
picked; the Transformers snapshot stays the default.

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
- **Qwen3-ASR** → `transcribe.cpp`/ggml picks the fastest available backend
  automatically (Vulkan on Intel/AMD/NVIDIA iGPUs and GPUs, CPU otherwise).
  Set `"compute": "cpu"` to force CPU; the chosen backend is logged at load
  (`qwen3_asr: using vulkan backend: Vulkan0`).
- **Granite** → same `transcribe.cpp` backend choice as Qwen3-ASR
  (`granite5_ctc: using vulkan backend: Vulkan0`).
- **Granite 2B** → CUDA `bfloat16` when the GPU has room, float32 on CPU
  (consumer CPUs emulate bf16, which measured about 12x slower than float32
  here). The GGUF variants run on CPU through llama.cpp. `GRANITE2B_DTYPE`
  overrides the dtype for experiments.
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
   75MB → Large v3 3GB; Large v3 Turbo 1.6GB, fetched once on first pick),
   the fixed Canary-1B, Qwen3-ASR 1.7B in Q8_0 (2.2GB, default) →
   Q6_K (1.7GB) → Q5_K_M (1.5GB) → Q4_K_M (1.3GB), Granite 5.0
   TurboCTC 470M in Q8_0 (506MB, default) → Q6_K (392MB) → Q5_K_M (336MB)
   → Q4_K_M (279MB), or Granite Speech 5.0 2B (4.8GB Transformers default,
   Q8_0 3.0GB, Q6_K 2.6GB). Whisper generations: v1
   oldest, v2 middle, v3
   newest and best. Large v3 is ~3GB; Large v3 Turbo is ~1.6GB and
   transcription-only. The bare Large entry is gone: the installed
   faster-whisper resolves it to byte-identical files as Large v3, so it was
   pure duplication (saved "large" configs switch to Large v3, no
   re-download).
  Recording is politely refused for a moment while a model is swapping.
- **Dictionary (Whisper / Qwen3-ASR / Granite 2B)** — the Live tab's
  *Dictionary: Edit…* button opens a shared word list plus optional context
  text. Add, edit or delete terms; tick which engines should use them. Whisper
  receives them as `hotwords` + `initial_prompt` (its decoder keeps 223
  tokens), Qwen3-ASR as a priority-ordered `vocabulary` + `prompt`, and
  Granite 2B as the model card's `Keywords: …` instruction (up to 2000
  terms) with the context appended. If the list is too long for an
  enabled engine the dialog asks whether to trim it, keep it (extra terms are
  ignored) or go back and edit; engines that are not installed are greyed out.
  Moonshine v2, Canary-1B and Granite TurboCTC have no equivalent knob, so
  they are not offered.
- **SRT File tab** — queue one or many video/audio files (batch runs one by
  one, failures don't stop the queue), pick an output folder, generate
  `.srt` with live progress bar, per-file status list, per-chunk log,
  cancel, and a remaining-time countdown. *Boost quiet audio* applies a
  loudnorm pass before transcription for soft/uneven recordings (burns
  always re-encode from the original, so they are unaffected).
- **Per-tab engines** — Live, SRT File, Note, and Import each pick their
  own engine + model. A tab you never explicitly set follows your last pick
  elsewhere; changing a tab while another session runs asks first
  (dual engines = dual RAM, stated upfront), otherwise idle engines are
  unloaded to make room. Mic dictation, an SRT batch, Note, and an Import
  job can all run at once on different engines.
- **Note tab** — mic dictation in 40–80s chunks with timer, *Save as TXT*
  (UTF-8), *Copy All*, and session-audio export. File import moved out —
  see the Import tab (mic and import no longer share a box).
- **Microphone picker** — Live and Note both list every capture endpoint the
  app can open (USB mic, headset, laptop array), so a machine with more than
  one input is no longer stuck with whatever Windows defaulted to. The row
  appears only when there is a real choice, the pick persists across
  restarts, and an unplugged mic falls back to the system default instead of
  failing. Both tabs animate a live waveform, and a mic that opens but never
  delivers audio is called out in the status line instead of leaving a flat
  line.
- **Import tab** — drag & drop (or Browse) any audio/video
  (`wav/mp3/m4a/mp4/mkv/avi/mov/…`), *Transcribe File* runs it through
  this tab's own engine (Whisper single-pass, Canary/Qwen/Moonshine
  VAD-chunked)
  with progress + cancel, then appends the result **one sentence per line**
  (Latin + CJK aware, abbreviations kept) into its own editable box.
  *Save as TXT* writes UTF-8, *Copy All* copies to clipboard; quit warns
  while a file still runs or sentences are unsaved.
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
   single-frame **Preview** before the full encode: **Preview Frame** renders one still with the current
   size, using *Sample from/len* only as its test spot (transcribing that short clip when no SRT exists
   yet) — *Generate SRT/MP4* always processes the FULL file, samples never affect it. Latin burns use Arial
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

- Windows 10/11, microphone (default input device) — or no microphone at all:
  Note mode can record desktop/system audio through WASAPI loopback.
- **Python 3.11** ([python.org](https://www.python.org/downloads/);
  `setup.bat`/`run.bat` prefer a local portable `python\` folder if present,
  otherwise fall back to system Python 3.11).
- Disk for models (see below) and patience on first run (large downloads).
- ~16 GB RAM if you use Canary-1B; ~4 GB free for Qwen3-ASR (2.8 GB
  resident for the default Q8_0 model), ~1 GB free for Granite (700 MB
  resident), ~7 GB free for Granite 2B in Transformers form (or ~3 GB for
  its Q8_0/Q6_K GGUF builds). Whisper runs on CPU:
  large-v3-turbo is the fast default for transcription; large-v3 is slower
  but supports translation.

## Install

```bat
git clone https://github.com/jackfood2/PolyglotSTT.git
cd PolyglotSTT
setup.bat      :: creates venv, installs deps, pre-fetches the Moonshine model
run.bat        :: launches the app (rechecks deps each start)
run_windowless.vbs :: same, with no console window at all (needs venv set up)
```

`setup.bat` installs online with `pip`, falling back to a local `wheels\`
folder when present, then caches wheels there for fully-offline reinstalls.
It installs the base, Whisper, Canary, Qwen3-ASR and Granite 2B dependency
groups; the Canary group is large, and setup warns but continues if it cannot
be installed. Qwen3-ASR needs `transcribe-cpp` + `transcribe-cpp-native`
(`requirements-qwen.txt`), whose native wheel bundles the CPU and Vulkan
backends — no CUDA toolkit required. Granite 2B needs `torch` +
`transformers` + `torchaudio` (`requirements-granite2b.txt`, CPU wheels by
default); its GGUF variants need no Python packages beyond
`huggingface_hub`, because they run on a pinned llama.cpp build that the
engine fetches on first use.

**Portable / offline copy:** `make_portable_zip.bat` builds
`..\PolyglotSTT-portable.zip` with the source, the bundled `python\`
runtime, and a complete `wheels\` wheelhouse. Extract the zip anywhere on
another PC and run `setup_offline.bat` once — it rebuilds `venv\` from
those wheels with no network access, because a venv stores absolute paths
and does not survive being moved. `models_cache\` is not included; copy it
across, or let the engines download on first use where there is internet.

**Updating an existing install with no internet:** `make_offline_update.bat`
upgrades a 1.4.16 / 1.4.17 / 1.4.18 / 1.5 install to 1.6. Run it on a machine
that has PolyglotSTT working and has internet: it audits `wheels\` against
every requirements file, re-downloads whatever cannot resolve offline,
rebuilds a portable `python\` runtime, stages the source, wheels and the
Granite 470M / 2B GGUF weights, then writes split WinRAR volumes (990 MB
each) to your Desktop. Carry those volumes to the offline machine, extract
them, and run the bundled script with the target folder:

```bat
apply_offline_update.bat "D:\Apps\PolyglotSTT"
```

It backs up `moonshine_config.json`, keeps `models_cache\`, rebuilds `venv\`
from the bundled wheels only, and re-checks every dependency group with no
network access. These offline packages are **not** hosted on GitHub — they
contain the model weights and are several GB, so they are always built
locally.

Since `pip --no-index --find-links=wheels` can only pick from the wheelhouse,
the bundled wheel set *is* the version pin; there is no separate freeze file
for `setup_offline.bat` to read.

**Fresh-install defaults:** the app starts with Moonshine v2 (English).
When you switch to Whisper, the default is `large-v3-turbo` in
`transcribe` mode — the fastest CPU option. To translate, select
`large-v3` and set Task to `translate` (turbo cannot translate by design).

**Hallucination guard:** every Whisper model runs with the bundled Silero
VAD. Silence is removed before decoding, so empty audio produces no text;
short utterances and real phrases such as “Thank you for watching” are
preserved.

## Models to download

Only Moonshine is fetched by `setup.bat`. The heavy engines download
**automatically on first use** into `models_cache\` (offline-first: cached
copies are reused with no network). Prefer to pre-fetch or have a slow
connection — grab them manually:

| Model | Size | Source | Cache location |
|---|---|---|---|
| Moonshine (all 6 sizes: tiny → medium-streaming) | ~25–260 MB each | fetched on first pick via the *Model* row (`setup.bat` pre-fetches medium + tiny-streaming) | `models_cache\download.moonshine.ai\` |
| `nvidia/canary-1b` | ~3.9 GB | <https://huggingface.co/nvidia/canary-1b> — or drop a `canary-1b.nemo` file at `models_cache\canary-1b\canary-1b.nemo` (>100 MB is picked up automatically, fully offline) | `models_cache\canary-1b\` (+ `models_cache\huggingface\`, `models_cache\torch\`) |
| `Systran/faster-whisper-large-v3` | ~3 GB | <https://huggingface.co/Systran/faster-whisper-large-v3> — auto-downloaded by Faster-Whisper; pick size in the Live tab *Model* row; each size downloads once on first selection | `models_cache\whisper-models\` |
| `mobiuslabsgmbh/faster-whisper-large-v3-turbo` | ~1.6 GB | <https://huggingface.co/mobiuslabsgmbh/faster-whisper-large-v3-turbo> — fast default for `transcribe`; **cannot translate by design**, so the app blocks it in `translate` mode | `models_cache\whisper-models\` |
| `handy-computer/Qwen3-ASR-1.7B-gguf` | 2.2 GB (Q8_0) / 1.7 / 1.5 / 1.3 GB | <https://huggingface.co/handy-computer/Qwen3-ASR-1.7B-gguf> — single `.gguf` files, auto-downloaded on first pick; drop `Qwen3-ASR-1.7B-Q8_0.gguf` (or any quant) at `models_cache\qwen3-asr\` for fully-offline use | `models_cache\qwen3-asr\` |
| `handy-computer/granite-speech-5.0-470m-turboctc-gguf` | 506 MB (Q8_0) / 392 / 336 / 279 MB | <https://huggingface.co/handy-computer/granite-speech-5.0-470m-turboctc-gguf> — single `.gguf` files, auto-downloaded on first pick; drop `granite-speech-5.0-470m-turboctc-Q8_0.gguf` at `models_cache\granite\` for fully-offline use. Apache-2.0, English only. | `models_cache\granite\` |
| `ibm-granite/granite-speech-5.0-2b` | ~4.8 GB (Transformers snapshot) | <https://huggingface.co/ibm-granite/granite-speech-5.0-2b> — the default Granite 2B backend, auto-downloaded on first pick. Apache-2.0, en/fr/de/es/pt/ja, punctuated output, translates to English. | `models_cache\granite-2b\` (+ `models_cache\huggingface\`) |
| `ibm-granite/granite-speech-5.0-2b-GGUF` | 3.0 GB (Q8_0) / 2.6 GB (Q6_K), each incl. the 1.1 GB f16 audio projector | <https://huggingface.co/ibm-granite/granite-speech-5.0-2b-GGUF> — optional llama.cpp builds, downloaded only when picked. The engine also fetches the pinned llama.cpp `b9768` CPU runtime (~17 MB) once. | `models_cache\granite-2b-gguf\` and `models_cache\llama.cpp\b9768\` |

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

**Numbers as digits (Live only)** — two checkboxes sit under the typing
row in the Live tab:

- *Numbers → digits* runs a local inverse-text-normalisation pass over
  the live transcript, so spoken forms become written forms: `three` → `3`,
  `twenty three` → `23`, `three thirty pm` → `3:30 PM`, `five dollars` →
  `$5`, `twenty five percent` → `25%`, `third` → `3rd`, `three point one
  four` → `3.14`. It is conservative about ambiguous words (`one of them`,
  `no one`, `wait a second` and `first of all` are left alone). Pure
  Python, no NeMo/torch dependency, no network; applies only to Live
  dictation, not SRT/Note/Import.
- *Qwen digit prompt* asks Qwen3-ASR itself (via its free-text prompt —
  "native transcription polishing") to emit digits. Both are on by
  default and each can be switched off independently.

**SRT subtitles** — open the *SRT File* tab, drop video/audio files
(mp4/mkv/avi/mov/ts/mp3/wav/m4a/… — multi-select for batch), optionally set
an output folder (default: next to the source), set *Input/Output
Language* (enabled for Canary and Whisper only; `translate` always outputs
English), tune CPU threads, then *Generate SRT*. The engine label (e.g.
`Whisper (translate Japanese->English)`) always shows exactly
what the job will do. The status line counts down remaining time from an
estimate learned per model, per burn speed, and per audio length on this
machine (`srt_eta.json`: `whisper:large-v3`, `whisper:large-v3-turbo`, and `tiny` track separately,
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
moonshine_stt.py   main app: F2 hook, live queue, SRT/Note/Import jobs (imports core.*)
gui.py             app window shell (~850 lines: build, theme toggle, tab glue)
ui/tabs/live.py    Live-tab controls (record/output/history, engine menus)
ui/tabs/srt.py     SRT File tab (queue, burn controls, languages, preview)
ui/tabs/note.py    Note-tab mic dictation + engine row
ui/tabs/import_tab.py  Import-tab file UI + legacy note_file_* aliases
ui/dialogs.py      model manager + burn-speed compare dialogs
ui/choices.py      static menus/labels (engines, models, languages, burn)
core/              app config + version (config.py) - single source of truth
ui/                shared theme palette (theme.py) + widgets (widgets.py)
services/          file-import sentence formatting (note_format.py)
engines/           uniform re-export of every STT backend wrapper
engine.py          Moonshine v2 wrapper
note_engine.py     Note-mode chunked recorder + async transcriber
recorder.py        microphone capture (16 kHz)
input_sim.py       clipboard + Ctrl+V insertion
srt.py             SRT backend: ffmpeg extract, VAD, word-anchored cue packing,
                 timing refine, batch queue, size-matched burn-in, ETA tracking
canary_engine.py   Canary-1B wrapper (NeMo, offline, language-validated)
whisper_engine.py  Faster-Whisper wrapper (large-v3 / turbo, per-call task/lang overrides)
qwen_engine.py     Qwen3-ASR wrapper (transcribe.cpp GGUF, 30 languages, no timestamps)
granite_engine.py  Granite Speech 5.0 TurboCTC wrapper (transcribe.cpp GGUF,
                 English only, fastest, no punctuation/timestamps)
granite2b_engine.py Granite Speech 5.0 2B wrapper (Transformers default, optional
                 Q8_0/Q6_K GGUF on a pinned llama.cpp server, keyword biasing)
gpu.py             dGPU/VRAM/NVENC detection, device recommendation
setup.bat          one-click setup (venv, deps, wheels cache, Moonshine model)
run.bat            one-click launch (dependency self-check)
requirements*.txt  dependency pins (base / Canary / Whisper / Qwen / Granite 2B)
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

### v1.6

New sixth engine: **Granite Speech 5.0 2B** (IBM, Apache-2.0) — the
speech-LLM sibling of the 470M TurboCTC model, added as its own engine kind
(next to `Granite`, not as another quant of it) in the Live, SRT, Note and
Import tabs.

- **Two backends, one engine.** The default is the Hugging Face Transformers
  snapshot (~4.8 GB) at `models_cache\granite-2b`, which runs float32 on CPU
  and bfloat16 on CUDA. Optional **Q8_0** (3.0 GB) and **Q6_K** (2.6 GB) GGUF
  builds run through the pinned llama.cpp `b9768` CPU runtime and are
  downloaded only when picked, so they add to rather than replace the
  Transformers model.
- **Dictionary and prompt actually work.** Terms reach the model as the model
  card's `Keywords: <term>, …` instruction (up to 2000, with the live count in
  the dictionary panel) and free-text context is sent as `Context: …`.
- **Prompt order is load-bearing.** The model follows the last instruction,
  so the wording is `Keywords: …` → `Context: …` → language sentence →
  "proper punctuation and capitalization". Putting the context *after* the
  punctuation request made it return all-lowercase text and lowercase the
  keyword it had just been given. Leading capitals are also restored
  deterministically after decoding, so lowercase-prompted translations can no
  longer start mid-sentence.
- **CPU speed, measured.** Transformers float32 runs at about **1.0x realtime**
  on this machine (0.85–1.0x depending on clip; bfloat16 on CPU was ~3x slower
  because it is emulated). GGUF Q8_0 warms to ~**4.3x realtime** and Q6_K to
  ~**4.8x**, at ~2.9 GB / 2.4 GB resident versus ~6.4 GB for Transformers — the
  GGUFs are labelled faster but slightly less accurate, since their keyword
  biasing is looser ("Polyglot stt" instead of exact "PolyglotSTT").
- **Six languages, English-only output.** en/fr/de/es/pt/ja are auto-detected
  or pinned, and translation targets English only (the model's en→it/zh paths
  are intentionally not exposed). Output is already punctuated and
  capitalised.
- **No timestamps, as before.** SRT derives cue boundaries from VAD chunking
  exactly as it does for Canary-1B, Qwen3-ASR and the 470M Granite, and
  segments longer than ~110 s are split internally at low-energy cuts as the
  model card recommends.
- **New dependency group.** `requirements-granite2b.txt` (torch, torchaudio,
  transformers) keeps the Granite-only install lean instead of pulling NeMo in
  with Canary's group. `setup.bat` now installs matching CUDA `torch` **and**
  `torchaudio` wheels when a GPU is chosen, avoiding the ABI mismatch that a
  torch-only install could cause.

### v1.5

New fifth engine: **Granite Speech 5.0 470M TurboCTC** (IBM, Apache-2.0),
added as a first-class engine in the Live, SRT, Note and Import tabs, and
built on top of the v1.4.18 clipboard-restore fix.

- **Granite is the Live-dictation engine.** It is a 470M encoder-only CTC
  model shipped as a single `.gguf`, so it loads in under a second and runs
  at roughly 24x realtime on Vulkan here — about 9x faster than Qwen3-ASR on
  the same machine, for 700 MB of RAM instead of 2.8 GB. It also replaces
  Qwen3-ASR for long files, where that model's Vulkan path can fail outright.
- **Trade-offs are surfaced, not hidden.** Granite is English-only, cannot
  translate, emits no timestamps, and being CTC it produces no punctuation
  or capitalisation. The GUI pins its language picker to English and disables
  the ones it cannot do, and SRT derives cue timing from VAD chunk boundaries
  exactly as it already does for Canary-1B and Qwen3-ASR. For subtitles and
  notes you intend to read, Qwen3-ASR or Whisper still win on readability.
- **Silence guard.** The CTC head invents short words on non-speech ("the",
  "thank you"). Input below -78 dBFS RMS is now skipped instead of decoded,
  on top of the recorder's existing Silero gate.
- **Model row and manager.** The Live model row lists the four Granite quants
  (Q8_0 506 MB default, Q6_K 392 MB, Q5_K_M 336 MB, Q4_K_M 279 MB) with the
  usual download badges, fetched from
  `handy-computer/granite-speech-5.0-470m-turboctc-gguf` on first pick into
  `models_cache\granite\`. The Model manager lists, totals and deletes them,
  and refuses to delete the one in use.
- **No dictionary.** TurboCTC takes no vocabulary or prompt, so
  `dictionary.build()` reports it as ineligible and the per-engine enable
  flags stay Whisper/Qwen-only.

### v1.4.18

Live dictation no longer leaves the transcript sitting on the clipboard after
it has been pasted into the target field.

- **The previous clipboard comes back after a successful paste.** The app
  snapshots the clipboard text before putting the transcript there, sends
  Ctrl+V, then restores the snapshot once the target has had time to read it.
  If the clipboard was empty, it is cleared again. Your copied text is no
  longer replaced by the last thing you dictated.
- **A copy made during the paste settles untouched.** The restore only runs
  while the clipboard still holds the transcript, so copying something else
  in that moment is never clobbered.
- **The no-target fallback still copies on purpose.** With no external input
  field focused, the transcript goes to the clipboard and the status bar says
  `Copied to clipboard - press Ctrl+V`.
- **Manual Copy buttons are unchanged.** Copy last / Copy All still put text
  on the clipboard on purpose.

Portable / offline build tooling is now part of the source tree:

- `make_portable_zip.bat` builds `..\PolyglotSTT-portable.zip` from the
  source, the bundled `python\` runtime and the `wheels\` wheelhouse (plus
  `wheels\freeze.txt` with the exact tested versions).
- `setup_offline.bat` rebuilds `venv\` on the target PC with no network,
  because a venv stores absolute paths and does not survive being moved.
- `run.bat` and `setup.bat` use a minimal PATH while pip runs, so a locked
  or vanished drive on the user PATH can no longer abort an install. The pip
  cache is kept inside the project folder for portable copies.

### v1.4.17

Digit strings are no longer mistaken for clock times. Saying "nine nine five
zero" used to produce "9:09 5:00"; it now produces `9950`.

- **Digit sequences are read left to right as one number** —
  `nine nine five zero` → `9950`, `five nine five nine` → `5959`,
  `zero five nine` → `059`, `nine nine five oh` → `9950`,
  `double seven` → `77`, `triple five` → `555`, `nine nine five zero
  dollars` → `$9950`, and `9 9 5 0` → `9950` when the model already emits
  spaced digits. The pass runs before the
  clock-time rule and stops that rule fires on purpose: a clock reading
  now needs evidence (a meridian, `o'clock`, a time preposition such as
  `at`/`around`, or a minute part of at least ten).
- **Number cues win** — `pin`, `code`, `order number`, `room`,
  `extension`, `flight`, `zip`, `serial` … put in front of spoken digits
  force a number even when a time reading is possible
  (`room two oh four` → `room 204`).
- **Years** — `in twenty twenty four` → `in 2024`,
  `since nineteen ninety` → `since 1990`, `in nineteen eighty four` →
  `in 1984`. A bare pair such as `seventeen ninety` is still left alone.
- **Small extras** — `three o'clock` → `3:00`, `point five` → `0.5`,
  and `a.m.`/`p.m.` supplied as two tokens are joined to `AM`/`PM`.
- `test_itn.py` is a dependency-free regression check for all of the
  above: `python test_itn.py` (55 cases).

### v1.4.16

Live dictation now writes numbers as digits instead of spelling them out
(Qwen3-ASR especially emits verbatim text such as "three" and "five
dollars"). Two independent checkboxes under the typing row in the Live tab
control it; both are on by default and persist in `moonshine_config.json`.

- **Numbers → digits (ITN)** — a local, dependency-free inverse text
  normalisation pass after each live decode: `three` → `3`, `twenty three`
  → `23`, `three thirty pm` → `3:30 PM`, `five dollars` → `$5`, `twenty
  five percent` → `25%`, `third` → `3rd`, `three point one four` → `3.14`,
  `a hundred and five` → `105`. Ambiguous words are left alone (`one of
  them`, `no one`, `wait a second`, `first of all`, `one third`).
  Live only — SRT, Note and Import output are untouched. No NeMo/torch
  dependency and no network access.
- **Qwen digit prompt** — appends a "write numbers as digits" instruction
  to Qwen3-ASR's free-text prompt (native transcription polishing), so the
  model itself leans towards digits. Ignored by the other engines.

### v1.4.14

A USB mic that "cannot catch the sound" was usually the wrong Windows
host API, and picking the right microphone is now automatic.

- **The app now chooses the microphone for you.** An external USB mic
  wins whenever it is plugged in and enabled in Windows; with none
  listed, capture falls back to the system default, i.e. the laptop's
  own microphone. Unplug or disable the USB mic and the built-in one
  takes over within a few seconds, with the USB mic's name remembered so
  it is re-selected the moment it returns. It never switches mid
  recording. Opt out with `"auto_input": false` in
  `moonshine_config.json`.
- **Host API order is now MME → WASAPI → DirectSound.** Measured on a
  laptop where every DirectSound capture endpoint returned digital
  zeros, or the same mic 45 dB quieter than MME (peak -76 dBFS against
  -28 dBFS). The picker and name resolution now lead with MME, which
  resamples inside Windows and therefore always accepts the 16 kHz the
  engines want; WASAPI refuses anything but the endpoint's native rate
  and is only a second try. Labels still use the untruncated name.
- **A stream that opens is no longer assumed to work.** Opening a dead
  or disabled endpoint succeeds and then delivers nothing. Each open
  candidate must now hand over a non-silent block within 2 s, otherwise
  it is closed at once and the same microphone's twin on another host
  API is tried before the system default. The twin that worked is
  remembered, so the next F2 press does not pay for the probe again.
- **New `mic_check.py` diagnostic.** It opens every capture endpoint
  under every host API, reads a second of audio with a hard timeout and
  reports OK / TOO QUIET / SILENT / NO DATA, so a "mic error" can be
  pinned on the app, on Windows, or on the hardware in one run:
  `venv\Scripts\python.exe mic_check.py`.

### v1.4.13

Two reasons a quiet mic looked dead: the meter was on a linear scale, and
a saved mic could silently open on a different host API.

- **Level meter and waveform are now dBFS.** The level was `rms × 5`, so
  speech at 0.02 RMS drew a 0.7-pixel ripple — indistinguishable from a
  dead microphone. It now maps -60 dBFS…0 dBFS onto the 0…1 the widgets
  draw, so the same speech reaches ~15 px and a whisper still moves the
  bar. Measured with the app's own recorder on a real voice: peak level
  0.409, 14.7 px of 36 (was 0.7 px). Silence still reads 0 and the idle
  waveform stays a flat line.
- **The picked endpoint is opened on the host API you picked.** A saved
  device used to be resolved by name first, and one mic is listed twice
  (DirectSound *and* MME), so a saved DirectSound pick could reopen as
  the MME twin — truncated name, extra latency, different level. The
  saved index now wins while it still names the saved endpoint; the name
  is followed only when the index is gone or now points at another
  device, which is still how a replugged mic is found on its new index.
  When the fallback does run it prefers DirectSound, matching the picker.

### v1.4.12

A mic that is unplugged, replugged or reset mid-session now recovers on
its own, and SRT jobs stop dying over one bad timestamp.

- **Mic hot-plug recovery.** Holding F2 after a USB mic was unplugged or
  re-enumerated used to fail with *"Error opening InputStream … [-9999]"*
  until the app was restarted, because MME and DirectSound only read the
  endpoint list once, at start-up. The saved endpoint is now tracked **by
  name** as well as by index, the list is re-scanned when the device set
  changes, and a capture that dies mid-session (unplugged, suspended by
  USB selective suspend, or held exclusively by another app) is dropped
  and re-opened — the audio already captured is kept and still transcribed.
  Every capture stream now goes through one registry, so a re-scan can
  never terminate PortAudio under a live stream (the state that used to
  wedge the host API for the rest of the process).
- **Note follows the same rule.** A replugged mic keeps its selection even
  though Windows gave it a new index, and an index that Windows handed to a
  *different* device no longer silently records the wrong microphone.
- **Fixed: mic picker never refreshed after a mic error.** The refresh ran
  on the hotkey thread and touched Tk widgets from there, which raises
  *"main thread is not in main loop"* into a bare `except: pass`. Off-thread
  updates are now queued to the GUI thread.
- **Fixed: silent-mic reopen loop.** A muted or exclusively-held endpoint
  was closed and re-opened every second forever; the retry now backs off to
  one attempt per 5 s and says *"Mic open but silent — replug it or pick
  another mic"*.
- **Fixed: one bad segment killed a whole SRT batch.** A segment with a
  zero-length or out-of-audio timestamp (quantised decoder output) raised
  and aborted the file — after the audio had already been transcribed.
  Zero-length segments now get a minimal duration and unusable ones are
  skipped, so the rest of the batch still lands.
- **Fixed: Whisper turbo could be asked to translate.** *Large v3 Turbo* has
  no translate task; a language pair stored before a model switch made the
  job fail from inside the engine after decoding. The SRT job now logs and
  falls back to transcription instead.
- **Fixed: SRT task followed the wrong engine.** The task handed to a batch
  was read from the *Live* tab's engine rather than the SRT tab's, so a
  Whisper batch could run with Canary's task (or vice versa).
- **Smaller fixes.** Canary no longer passes `"auto"` as a language code to
  NeMo (it is translated to *unset*), Whisper/Canary language-pair coercion
  handles *source*/*same* output without a known input language, the Note
  WAV writer stops on a stop flag instead of a sentinel so a stalled writer
  cannot hold the next session's handle, and Import files are converted
  through the system temp directory instead of the source folder (read-only
  network shares no longer fail).

### v1.4.11

Microphone choice in Live and Note, a waveform for Note, and two
silent-failure fixes.

- **Microphone picker (Live + Note).** Both tabs now list every capture
  endpoint the app can open — USB mic, headset, laptop array — instead of
  quietly using whatever Windows picked. Only DirectSound and MME endpoints
  are offered, because those are the only host APIs that resample to the
  16 kHz the engines expect (WASAPI refuses anything but the endpoint's
  native rate, WDM-KS fails on most drivers). The row stays hidden on
  single-mic machines, the choice persists across restarts
  (`input_device` / `note_mic_id`), and a mic that disappears falls back to
  the system default instead of wedging the recorder.
- **Silent-mic warning.** If a recording opens but never receives a single
  audio block, the status line now says *"Mic open but silent - pick
  another mic"* after 2.5 s instead of showing a flat line indefinitely.
- **Note waveform.** The Note tab has the same animated waveform and
  full-width level meter as Live, so you can see whether the mic is hearing
  you while a chunk records.
- **Fixed: auto-shutdown never ran.** *Shut down PC when done* was a
  `@staticmethod` that still declared `self`, so the call raised
  `TypeError` and a bare `except: pass` swallowed it — the log claimed the
  shutdown was scheduled while nothing was. It now checks the exit code and
  only reports success when `shutdown.exe` accepted the request, otherwise
  it prints the command to run by hand.
- **Fixed: invisible mic dropdown in light theme.** The Live mic menu
  inherited hardcoded colours that the light-theme remap turned into
  near-black text on a near-black background; it now uses the same
  appearance-aware colours as every other menu.
- **Fixed: cut-off waveform and level bar.** The idle line and the meter
  fill were drawn once at the widget's construction width, so they stopped
  short of the box edge; both now track the real canvas width on resize.

### v1.4.10

Menus now follow what each engine and model can actually do, instead of
offering choices that fail at decode time.

- **Live Task menu** lists only supported tasks: Moonshine v2 and Qwen3-ASR
  show a single disabled `transcribe`; Canary-1B and Whisper show both. A
  stored task that the selected model cannot perform falls back to
  `transcribe` automatically and is saved.
- **Whisper large-v3-turbo** cannot translate. Picking it while Task is
  Translate asks before switching, then leaves the Task menu showing a single
  disabled `transcribe` immediately.
- **SRT output language** is disabled (and English removed) when the selected
  Whisper model cannot translate, with the hint *"Whisper large-v3-turbo
  cannot translate - pick large-v3 in the Live model list, or choose the
  spoken language as output."* Qwen3-ASR keeps its disabled "Same as input".
- **Import** coerces `translate` to `transcribe` for non-translating Whisper
  models instead of failing the batch.
- New `capabilities.py` holds the single engine/model capability table that
  all of the above consult.

### v1.4.9.2

Bug-fix release for SRT cue timing: subtitles that lingered over silence and
sentences that were split into one-word cues.

- **Cues now end shortly after the speech, not at the next sound.** Whisper
  reports each word's end as the *next* word's start, and translate-mode
  segments cover the silence after the speech as well — one segment could be
  `"Don't do it."` stretched over 19.5 seconds. The cue builder now estimates
  how long the text actually takes to say (per-character speaking rate) and
  clamps both segment spans and word ends to it, so a 10-second word followed
  by silence becomes a 1-second cue.
- **Sentences stay in one cue.** Because the inflated spans exceeded the
  seven-second cue limit, the splitter fell back to emitting one word per cue
  (`Don't` / `do` / `it.`). With the spans corrected the same sentence now
  produces a single cue.
- **0.5 s readability tail.** A cue stays on screen for half a second after the
  last word instead of holding until the next cue begins, then disappears.
  `_finalize_cues` still prevents overlap and the tail is clamped to the media
  length.

Note: `Whisper large-v3-turbo` cannot translate. A `ja → en` SRT job needs
`large-v3` (3 GB); with turbo the job stops with "use large-v3 or switch to
transcribe".

### v1.4.9.1

Bug-fix release for SRT jobs whose output language differs from the input.

- **SRT no longer fails with "transcription preserves the source language".**
  The SRT tab has no task control — it inherits `whisper_task` from the Live
  tab, which ships as `transcribe`. So any job with a real language pair
  (Japanese → English, German → English, …) was rejected before a single cue
  was written, including 2.5-hour files that had already spent minutes on audio
  preparation. The job now derives the task from the language pair you actually
  picked: output English means translate, output equal to the input means
  transcribe. Canary-1B had the same dead end and got the same fix.
- **Clearer language hints in the SRT tab.** The hint under the pickers now says
  what the choice does ("output English to translate, pick the spoken language
  to transcribe as-is") instead of talking about a task menu that the tab does
  not have.

### v1.4.9

Bug-fix release for the Live tab's Qwen3-ASR model picker.

- **The Live model dropdown now lists the Qwen quants.** Picking Qwen3-ASR as
  the Live engine left the model row showing Moonshine's list (Tiny, Base,
  Small, Medium Streaming) under a "Model (Qwen3-ASR)" title, and picking
  anything did nothing. The row now shows the four Qwen3-ASR GGUF quants with
  the configured one preselected, and a pick switches the loaded model in
  place.
- **Model badges work again in every dropdown.** The downloaded / not-downloaded
  markers (✓ and ↓) were empty for all engines because the status lookup used
  the wrong key. Whisper, Canary-1B, Qwen3-ASR and Moonshine rows now show the
  real state of the cache.
- **A Live Qwen model pick is persisted.** The chosen quant is written to
  `qwen_model`, so it survives a restart instead of reverting to Q8_0.

### v1.4.8

Bug-fix release for the Qwen3-ASR and Dictionary work in v1.4.7.

- **Qwen3-ASR SRT jobs no longer fail.** The output-language picker offered
  translations the model cannot do, so a default run (input Japanese, output
  English) raised an error immediately. The picker is now a single disabled
  *Same as input* row, and an old config that still asks for a translation is
  coerced to the source language with a log note instead of failing.
- **Qwen-only languages now stick.** Cantonese, Malay, Swedish, Danish,
  Finnish, Polish, Czech, Filipino, Persian, Greek, Romanian, Hungarian and
  Macedonian were silently reset to auto-detect in the SRT and Note tabs
  because the handlers only knew Whisper's language list.
- **Import tab uses Qwen's own settings.** Chunks were being transcribed with
  the Whisper task and source language, so a Whisper source of e.g. Japanese
  forced Japanese on Qwen. Qwen failures also stopped being appended to the
  transcript as text.
- **Dictionary token budget is accurate.** The Whisper prompt could overshoot
  the 223-token decoder budget (the status still showed a green check), and one
  over-long term discarded every later term that would still have fit. Context
  is now trimmed word by word until the estimate fits, and the status flags an
  overshoot.
- Language pickers no longer get re-enabled behind the engine's back when the
  SRT tab repaints; Moonshine's SRT output picker stays disabled (it has no
  language setting at all).
- Model manager, Import progress and the Import model-mismatch guard now name
  and check the Qwen model instead of the Whisper one.

### v1.4.7

- New engine: **Qwen3-ASR 1.7B** (GGUF via `transcribe.cpp`). 30 languages
  including Mandarin, Cantonese, Japanese, Korean, Arabic, Hindi, Thai,
  Vietnamese, Turkish — spoken language preserved, no translation. Four
  quantisations (Q8_0 2.2 GB default → Q4_K_M 1.3 GB), picked per tab like
  every other engine.
- Runs on the ggml runtime: Vulkan is chosen automatically on capable GPUs
  and iGPUs, CPU otherwise (`"compute": "cpu"` forces CPU).
- SRT/Import use the VAD-chunked path (Qwen3-ASR emits no timestamps);
  Live dictation and Note work the same as Canary.
- Model manager lists, downloads and deletes the Qwen `.gguf` files;
  models live in `models_cache\qwen3-asr\` so the folder stays portable.
- New dependency group: `requirements-qwen.txt`.
- New **Dictionary** (Live tab → *Dictionary: Edit…*): a shared word/phrase
  list plus optional context text, addable, editable and deletable, with
  per-engine checkboxes. Whisper uses it as `hotwords` + `initial_prompt`
  (223-token decoder budget), Qwen3-ASR as `vocabulary` + `prompt`. Over-limit
  lists prompt to trim, keep (ignored terms) or go back; engines that are not
  installed are disabled, and Moonshine/Canary are excluded (no such setting).
  The list is saved in the config and re-applied to every live engine.

### v1.4.6

- Note tab: new **Audio** source picker. Choose the default microphone, any
  connected input (laptop array, USB mic, headset...) by name, or
  **No microphone**; tick **Desktop audio** to capture system sound through
  WASAPI loopback. Mic + desktop are mixed into one track, so meetings,
  videos and calls can be transcribed with or without a microphone.
- The source list refreshes when the Note tab opens and again just before
  recording, so a hot-plugged USB microphone appears without a restart.
  Both choices persist as `note_mic_id` / `note_desktop_audio`.
- New dependency: `soundcard` (WASAPI loopback). Recording falls back to the
  normal sounddevice path for the default microphone.
- The mic watchdog now reports "No audio input" for desktop-only sessions and
  "Audio silent" when the selected source produces only digital silence.
- Tab strip: segment borders removed so the strip no longer shows a halo.

### v1.4.5

- Responsive layout: every scroll area re-fits its cards on startup, on window
  resize and on tab change, so nothing is clipped at the default 480 px width
  or the 440 px minimum. Live, Note, SRT File and Import cards use stretchy
  grids instead of fixed single rows, and option menus no longer grow to their
  longest value.
- Note tab: new **Language** picker (Auto-detect plus 18 spoken languages) for
  Whisper, disabled for the other engines. The choice is saved as
  `note_whisper_lang` and Note transcribes with it (Auto = detect).
- Note tab: new **Unload** button beside **Free memory** to drop the Note
  engine manually; it refuses while a recording or transcription is running.
- Startup guard for 125% DPI: if CustomTkinter applies the initial geometry
  before the monitor scale is known, the window size is re-applied in physical
  pixels so the scaled widgets fit.

### v1.4.4

- Whisper hallucination guard: Silero VAD is enabled for every Whisper model
  and task, so silence is removed before decoding and empty audio returns no
  text. Short utterances and real canned phrases are preserved.
- Fresh-install defaults: Whisper starts in `transcribe` mode with
  `large-v3-turbo`; Moonshine v2 remains the initial engine. Select
  `large-v3` + `translate` for translation.
- Portable launch scripts: removed the machine-specific drive-letter PATH filter;
  setup/run scripts prepend the project venv and portable Python while
  keeping the user's PATH.

### v1.4.3

- Note + Whisper: RECORD no longer waits forever after the model finishes loading — readiness and pre-flight now resolve the shared Whisper engine instead of a per-tab cache slot (Import had the same stale-cache wait)
- Note engine safety: pressing RECORD while the Note preload is still running no longer unloads and restarts that same load; the target engine is excluded from cache swaps by identity
- Recording pipeline: session-WAV archiving moved off the sounddevice audio callback to a bounded writer thread; UI updates from audio/transcriber threads now go through a GUI queue instead of calling Tk directly; input-overflow status is captured
- Crash safety: unfinished `note_*.wav` files are moved to `notes_audio/recovered/` on startup instead of deleted; MP3 export writes a `.part` file and renames it atomically
- No lost tail: RECORD is blocked while the previous chunk is still transcoding; close waits briefly for MP3 export and never deletes a WAV under ffmpeg
- Status/close hardening: 15-minute Note engine load deadline with retry hint; SRT jobs are confirmed and cancelled on close; single-instance mutex; rotating app/crash logs in `%LOCALAPPDATA%\PolyglotSTT\logs`
- Note tab: new **Open folder** button opens the `notes_audio` archive in Explorer
- SRT: heartbeat progress output, safer default worker count, improved fallback timing when word timestamps are missing

### v1.4.2

- RAM safeguards: Whisper checks free RAM before loading (`whisper_engine._check_ram_budget`, `gpu.whisper_resident_mb`), refuses with a message instead of `mkl_malloc` mid-session crash; threads trim under pressure; dual-engine confirm dialog agrees on cost
- Free memory: Live `Unload` + Note `Free memory` buttons (`gui.live_unload_btn`, `unload_tab_engines` / `unload_other_tab_engines`); idle tab engines auto-swap out before heavy loads; one universal Whisper model picker for all tabs
- Note typing fix: transcribed chunks insert at a dedicated `transcript_end` mark, never interleaved into typed words
- Launch fix: `run.bat` / `setup.bat` / `run_windowless.bat` prepend the project venv/portable Python to PATH and clear `PYTHONPATH/HOME` (replaces the old machine-specific drive-letter filter)

### v1.4.1

- Subtitle timing: a batch no longer fails when only a minority of segments
  lack matching word timestamps (common in Whisper translate mode) — the
  matched majority keeps exact timing, the rest is estimated; the hard fail
  now only triggers on systemic loss (>50% unmatched)
- SRT language menus follow the SRT tab's own engine (they used to track
  the Live engine, so they stayed greyed out with SRT=Whisper + Live=
  Moonshine); lists narrow per engine with safe fallbacks

### v1.4.0

- Import tab: the Note tab's *Import Audio / Video File* card is now a
  standalone 4th tab with its own engine + model row, progress, and
  sentence box (Save TXT / Copy All / Clear); Note keeps mic recording
  only. Old `note_file_*` GUI/app method names still work as aliases
- Modular packages: `core/` (config + version), `ui/` (theme + widgets),
  `services/` (sentence formatting), `engines/` (uniform re-exports) —
  top-level modules re-export the same names, so existing imports work
- GUI split: `gui.py` (5,100 lines) is now a thin shell (~850 lines:
  window build, theme toggle, tab glue) plus `ui/tabs/` mixins
  (`live`/`srt`/`note`/`import_tab`), `ui/dialogs.py`, and
  `ui/choices.py` — every method moved verbatim (152/152 pre-existing
  bodies byte-identical), old `from gui import …` names still resolve
- Fixed along the way: `SEG_SELECTED`/`SEG_SELECTED_HOVER` had no module
  definition, so the initial tab-strip theming always `NameError`d into a
  silent `except: pass` (only repairs on toggle worked) — now defined up
  front from the dark theme
- Engine-switch safety: answering No no longer applies the change anyway;
  a tab's own engine can't be retargeted while it records/processes/loads;
  apply revalidates the plan (no TOCTOU); Import no longer skipped in the
  conflict scan; matching-but-loading Whisper instances are kept, not
  unloaded and recreated; Live cost checks read the live singletons
- Completion honesty: shutdown/pop-up gating uses an explicit per-stage
  success flag instead of `"all " in text` (a failed auto-burn after good
  SRTs no longer triggers shutdown); `_shutdown_pc`/`_abort_shutdown`
  actually reference the imported subprocess module; model-switch flags
  clear on every path (headless included); model manager blocks models
  referenced by any tab, not just Live
- Import Cancel actually cancels: the cancel Events no longer shadow the cancel
  methods (registrations received an Event, so Cancel was dead); duplicate
  starts can't clear a running job's busy flags anymore
- Canary lifecycle: unload cancels in-flight loads, stale loaders can't corrupt
  newer state, loads serialize, inference holds the model under lock, option
  callbacks fire outside the lock, success-callback exceptions aren't mistaken
  for load failures, stereo downmixes (empty audio is silence, not an error),
  and the no-args transcribe retry only triggers on real signature mismatches
- Pipeline fixes: Import jobs use unique temp WAVs (no fixed-name
  collisions), run on the Import tab's own engine cache slot, and report
  to the Import tab (status no longer lands on Note); tab-switch
  adopt-default fixed (it compared display names to ids, so it never
  fired); close path cancels Import jobs and warns on unsaved sentences

### v1.3.0

- Note file import: drag & drop (or Browse) audio/video (wav, mp3, mp4, mkv, …) into the Note tab — transcribes with the Note tab's engine (Whisper single-pass, Canary/Moonshine VAD-chunked) with progress + cancel, then appends one sentence per line (Latin + CJK, abbreviations kept)
- Note export stays UTF-8 TXT + clipboard: imported sentences land in the same editable box, so Save as TXT (UTF-8) and Copy All work for mic and file notes alike; quit warns while a file import still runs and the tab is remembered across restarts

### v1.2.23

- Note mic callback no longer calls out while holding the recorder lock — a slow level-meter update can never stall the audio stream and drop live speech
- Removed dead code the audit proved unreferenced (unused config helper, write-only unload counter)
- Investigated, no change: overwrite already replaces in place for SRT and burns (reserve lists only dedupe within a batch, e.g. same clip queued twice) — verified prompt to file-write path end to end

### v1.2.22

- Note Stop now drains honestly: the tail audio up to the click becomes a final chunk, the status reads "Finishing last N chunks" until the drain lands, and dropped (queue-full) chunks no longer inflate the counter
- Quitting mid-Note warns first: still recording or still transcribing blocks exit with a "lose it?" prompt, before the unsaved-note question

### v1.2.21

- Windowless launch: `run_windowless.vbs` starts the app with no console window at all (`run_windowless.bat` if you prefer bat); output goes to `logs\app.log` since pythonw has no console
- Footer switch "Windowless next launch": tick it and the next `run.bat` start goes windowless by itself (stored in config, so it survives restarts)

### v1.2.20

- Live Src menu spells languages in full (Japanese, not ja) for both Whisper and Canary; saved codes restore to their names on launch
- Every Live-row change (engine, task, language, model) already saved instantly — stale Japanese fallbacks corrected to auto-detect throughout

### v1.2.19

- Critical Canary fix: every Canary load died with `NameError` before publishing the model, so the engine could never become ready — loads now complete and report ready correctly
- Stale-load guards completed for Whisper too (success and failure paths discard results from superseded loads instead of clobbering the new model)

- Removed the bare Whisper Large option: the installed faster-whisper resolves it to the exact same files as Large v3, so the double entry (and double size readout in Model storage) is gone; old configs move over with no re-download
- Engine/model selection now sits at the top of every tab (Live, SRT File, Note) in the same position

### v1.2.18

- Whisper model menu now tells Large apart: bare Large is the same v1 weights as Large v1 (oldest), v2 middle, v3 newest and best
- Burn card states it outright: Sample from/len only pick the Preview Frame test spot, Generate SRT/MP4 always runs the full file
- Taller drop box so the two-line hint is never clipped

### v1.2.17

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
- Whisper large-v3 and large-v3-turbo by OpenAI (MIT weights); Faster-Whisper by SYSTRAN.
- Canary-1B by NVIDIA — check its license on Hugging Face before
  commercial use.
- Qwen3-ASR by Alibaba Qwen (Apache-2.0); GGUF conversion by
  `handy-computer`, runtime by `transcribe.cpp` (MIT).
- GUI: CustomTkinter; audio: sounddevice/soundfile/imageio-ffmpeg.
