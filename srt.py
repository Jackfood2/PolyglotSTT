"""SRT file generation backend - portable, CPU-only, all engines.

Pipeline: video/audio -> ffmpeg (16kHz mono wav) ->
  Moonshine/Canary: energy VAD segments -> per-chunk transcription ->
  Whisper Large v3: single native pass with timestamps (no VAD needed) ->
  Whisper timings snapped to measured speech energy (native boundaries are
  approximate - often early / spanning silence) ->
sentence-aware cue packing -> .srt next to the source file by default.
"""

import json
import os
import re
import shutil
import subprocess
import threading
import time
import unicodedata
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np


# ---------------- ETA tracking (history baseline, confident countdown) ----------------
# Aggregated baseline: per-MODEL and per-LENGTH totals persist in srt_eta.json
# so repeat jobs predict from YOUR machine's measured speed, not guesses.
# Keys are specific - whisper:tiny and whisper:large-v3 differ ~10x, and one
# burn speed (ultrafast 1-pass) vs another (medium 2-pass) differs just as
# much, so lumping them "per engine" poisoned every estimate. Families fall
# back: a new key with no samples borrows its family's aggregate (and the
# pre-split "whisper"/"burn" history keeps working as that fallback).
# Length matters because short clips are dominated by fixed overhead (model
# load, ffmpeg extract, VAD) while long files converge on the true rate -
# one global factor would mislead both. Buckets: s<60s, m<600s, l=rest.
# The countdown commits to the estimate (linear to 99% / 0s, no hedging);
# the ACTUAL outcome is recorded at completion, so the next estimate in
# that bucket is better. Old flat-format files migrate automatically.
ETA_PATH = Path(__file__).parent / "srt_eta.json"
ETA_DEFAULTS = {"whisper": 1.0, "canary": 3.0, "moonshine": 0.3, "burn": 1.5,
                # First-run guesses until a sample lands (learning corrects).
                "whisper:tiny": 0.08, "whisper:base": 0.12,
                "whisper:small": 0.2, "whisper:medium": 0.4,
                "whisper:large": 1.0, "whisper:large-v1": 1.0,
                "whisper:large-v2": 1.0, "whisper:large-v3": 1.0,
                "burn:match": 1.5, "burn:fast": 0.7, "burn:fastest": 0.35,
                "burn:nvenc_draft": 0.25, "burn:nvenc_turbo": 0.2,
                "burn:nvenc_balanced": 0.4}
ETA_BUCKETS = (("s", 60.0), ("m", 600.0), ("l", float("inf")))
_ETA_LOCK = threading.Lock()


def _eta_bucket(audio_s) -> str:
    try:
        v = float(audio_s)
    except Exception:
        return "m"
    for name, limit in ETA_BUCKETS:
        if v < limit:
            return name
    return "l"


def _load_eta_stats() -> Dict:
    try:
        with open(str(ETA_PATH), "r", encoding="utf-8") as f:
            d = json.load(f)
            return d if isinstance(d, dict) else {}
    except Exception:
        return {}


def _bucket_entry(stats: Dict, key: str, bucket: Optional[str]):
    """(entry_or_None, fell_back) - prefers the length bucket, then the
    engine-wide aggregate, then legacy flat entries."""
    try:
        node = stats.get(key) or {}
        if not isinstance(node, dict):
            return None, True
        if bucket and isinstance(node.get(bucket), dict):
            e = node[bucket]
            if float(e.get("audio") or 0) > 1.0 and float(e.get("proc") or 0) > 0:
                return e, False
        all_e = node.get("all")
        if isinstance(all_e, dict) and float(all_e.get("audio") or 0) > 1.0:
            return all_e, True
        # Legacy flat format: {"n":..,"audio":..,"proc":..} directly on key.
        if float(node.get("audio") or 0) > 1.0:
            return node, True
    except Exception:
        pass
    return None, True


def _eta_factor(key: str, duration=None) -> float:
    """proc/audio rate for a key, with family fallback.

    whisper:large-v3 borrows the legacy "whisper" aggregate until it has
    its own samples (same for burn:{speed} <- "burn"), so splitting keys
    never discards the history users already accumulated."""
    try:
        bucket = _eta_bucket(duration) if duration else None
        stats = _load_eta_stats()
        entry, _fb = _bucket_entry(stats, key, bucket)
        if entry is None and ":" in (key or ""):
            entry, _fb = _bucket_entry(stats, (key or "").split(":")[0],
                                       bucket)
        if entry:
            a = float(entry.get("audio") or 0)
            p = float(entry.get("proc") or 0)
            if a > 1.0 and p > 0:
                return min(20.0, max(0.05, p / a))
    except Exception:
        pass
    try:
        return float(ETA_DEFAULTS.get(key or "",
                                      ETA_DEFAULTS.get((key or "").split(":")[0],
                                                       1.0)))
    except Exception:
        return 1.0


def _eta_safe_key(part: str, fallback: str = "") -> str:
    """Sanitize one key segment (model ids come from config/downloads)."""
    try:
        s = str(part or fallback or "").strip().lower()
    except Exception:
        s = str(fallback or "")
    s = re.sub(r"[^a-z0-9_+-]", "", s) or str(fallback or "")
    return s[:40] or str(fallback or "")


def clear_eta_history(which: str = "all") -> int:
    """Delete learned ETA baselines so estimates restart from defaults.

    which: "srt" (whisper/canary/moonshine keys), "burn" ("burn"/"burn:*"
    keys), or "all". The burn-size learner is separate (see
    clear_burn_size_history). Returns the number of keys removed. Never
    raises (a missing/corrupt file counts as 0)."""
    removed = 0
    with _ETA_LOCK:
        try:
            stats = _load_eta_stats()
            if not stats:
                return 0
            if (which or "all") == "all":
                removed = len(stats)
                stats = {}
            else:
                want_burn = (which == "burn")
                for k in list(stats):
                    kb = (k == "burn" or str(k).startswith("burn:"))
                    if kb == want_burn:
                        try:
                            del stats[k]
                            removed += 1
                        except Exception:
                            pass
            tmp = str(ETA_PATH) + ".tmp"
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump(stats, f, indent=1)
            os.replace(tmp, str(ETA_PATH))
        except Exception:
            pass
    return removed


# ---------------- Burn-size history (learned output estimates) ----------------
# GPU (NVENC) and 1-pass sizes are hard to predict analytically: rate-control
# overshoot varies by content and preset. So every successful burn records
# its actual/analytic-raw ratio under its speed id, and the estimate
# multiplies the analytic raw size by the learned mean for that speed.
# Manual-kbps mode with no history for the speed reports "no data" instead
# of guessing. Size-match (auto bitrate) needs no history at all: the
# target IS the source size.
BURN_SIZE_PATH = Path(__file__).parent / "burn_size.json"
_BURN_LOCK = threading.Lock()


def record_burn_sample(speed_id, video_kbps, audio_bps, duration_s,
                       actual_bytes, codec="h264"):
    """Fold one successful burn into per-speed size history. Never raises."""
    try:
        vkbps = float(video_kbps)
        abps = max(0.0, float(audio_bps or 0))
        dur = float(duration_s)
        act = float(actual_bytes)
    except Exception:
        return
    if not (vkbps > 0 and dur > 1.0 and act > 0):
        return
    raw = (vkbps * 1000.0 + abps) / 8.0 * dur
    if raw <= 0:
        return
    ratio = act / raw
    if not (0.2 < ratio < 5.0):
        return  # aborted/partial/corrupt write - must not poison history
    key = _burn_history_key(speed_id, codec)
    with _BURN_LOCK:
        try:
            try:
                with open(str(BURN_SIZE_PATH), "r", encoding="utf-8") as f:
                    stats = json.load(f)
                if not isinstance(stats, dict):
                    stats = {}
            except Exception:
                stats = {}
            node = stats.get(key)
            if not isinstance(node, dict):
                node = {"n": 0, "sum": 0.0}
            try:
                node = {"n": int(node.get("n", 0)) + 1,
                        "sum": float(node.get("sum", 0.0)) + ratio}
            except Exception:
                node = {"n": 1, "sum": ratio}
            stats[key] = node
            tmp = str(BURN_SIZE_PATH) + ".tmp"
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump(stats, f, indent=1)
            os.replace(tmp, str(BURN_SIZE_PATH))
        except Exception:
            pass


def burn_size_fudge(speed_id, codec="h264"):
    """(mean actual/analytic-raw ratio or None, sample count) for a
    speed+codec."""
    try:
        with open(str(BURN_SIZE_PATH), "r", encoding="utf-8") as f:
            stats = json.load(f)
        node = (stats or {}).get(_burn_history_key(speed_id, codec))
        if isinstance(node, dict):
            n = int(node.get("n", 0) or 0)
            s = float(node.get("sum", 0.0) or 0.0)
            if n >= 1 and s > 0:
                return s / n, n
    except Exception:
        pass
    return None, 0


def clear_burn_size_history() -> int:
    """Delete the learned burn-size ratios (est-size recalibrates from
    scratch). Returns the number of speed entries removed. Never raises."""
    removed = 0
    with _BURN_LOCK:
        try:
            try:
                with open(str(BURN_SIZE_PATH), "r", encoding="utf-8") as f:
                    stats = json.load(f)
                if isinstance(stats, dict):
                    removed = len(stats)
            except Exception:
                removed = 0
            try:
                if BURN_SIZE_PATH.exists():
                    BURN_SIZE_PATH.unlink()
            except Exception:
                pass
        except Exception:
            pass
    return removed


def estimate_burn_batch(entries, speed_id, vbr_auto, vbr_kbps,
                        codec="h264"):
    """Estimate total burn output for queued files.

    entries: [{duration (s), audio_bps, src_bytes}]. vbr_auto=True means
    size-match (target is the source size - no history needed).
    Returns {"mode": "manual"/"auto"/"none", "bytes": int|None,
             "basis": n_history_samples, "files": n_counted,
             "calibrated": bool}. Manual mode always estimates: learned
    fudge with history, pure analytic (fudge 1.0) without - "calibrated"
    tells the caller which one it got."""
    try:
        files = [e for e in (entries or [])
                 if float((e or {}).get("duration") or 0) > 1.0]
    except Exception:
        files = []
    if not files:
        return {"mode": "none", "bytes": None, "basis": 0, "files": 0,
                "calibrated": False}
    if vbr_auto:
        total = 0
        for e in files:
            try:
                total += max(0, int(e.get("src_bytes") or 0))
            except Exception:
                pass
        return {"mode": "auto", "bytes": total or None,
                "basis": 0, "files": len(files), "calibrated": True}
    try:
        vkbps = max(100, int(vbr_kbps or 0))
    except Exception:
        return {"mode": "none", "bytes": None, "basis": 0, "files": 0,
                "calibrated": False}
    raw = 0.0
    for e in files:
        try:
            abps = max(0.0, float(e.get("audio_bps") or 0))
            raw += (vkbps * 1000.0 + abps) / 8.0 * float(e.get("duration"))
        except Exception:
            continue
    fudge, n = burn_size_fudge(speed_id, codec)
    calibrated = fudge is not None and n >= 1 and fudge > 0
    if not calibrated:
        fudge = 1.0
    return {"mode": "manual", "bytes": int(raw * fudge), "basis": n,
            "files": len(files), "calibrated": calibrated}


def solve_burn_kbps(entries, speed_id, target_mb, codec="h264"):
    """Invert the size model: manual video kbps (1kbps precision) expected
    to produce target_mb for these entries.

    Uses the learned fudge when history exists, else a pure analytic
    inversion (fudge 1.0) so the target box works before any burn - the
    caller reports calibrated-vs-analytic from the sample count. Returns
    an int, or None when unsolvable: bad target, no usable durations, or
    a target at/below the audio floor (even 1kbps video would overshoot).
    The caller clamps to its own slider range. Never raises."""
    try:
        target = float(target_mb)
    except Exception:
        return None
    if not (target > 0):
        return None
    try:
        files = [e for e in (entries or [])
                 if float((e or {}).get("duration") or 0) > 1.0]
    except Exception:
        return None
    if not files:
        return None
    fudge, n = burn_size_fudge(speed_id, codec)
    if fudge is None or not (fudge > 0):
        # No history yet: pure analytic inversion (fudge 1.0). The box
        # stays usable from the first burn; samples sharpen it later.
        # (The *encode-side* compensation still wants 2+ samples before
        # it trusts the ratio.)
        fudge = 1.0
    try:
        total_dur = sum(float(e.get("duration")) for e in files)
        audio_part = sum(max(0.0, float(e.get("audio_bps") or 0))
                         * float(e.get("duration")) for e in files) / 8.0
    except Exception:
        return None
    if not (total_dur > 0):
        return None
    vkbps = ((target * 1e6 / fudge) - audio_part) * 8.0 / (1000.0 * total_dur)
    if not (vkbps > 0) or vkbps != vkbps:  # <=0 or NaN (audio floor / inf)
        return None
    try:
        return max(1, int(round(vkbps)))
    except Exception:
        return None


def record_eta_sample(key: str, audio_s: float, proc_s: float):
    """Fold one successful job into the aggregated baseline (atomic write),
    updating both its length bucket and the engine-wide aggregate."""
    try:
        audio_s, proc_s = float(audio_s), float(proc_s)
    except Exception:
        return
    if not (audio_s >= 1.0 and proc_s >= 0.0) or not key:
        return
    bucket = _eta_bucket(audio_s)
    with _ETA_LOCK:
        try:
            stats = _load_eta_stats()
            node = stats.get(key)
            if not isinstance(node, dict):
                node = {}
            for slot in (bucket, "all"):
                e = node.get(slot)
                if not isinstance(e, dict):
                    e = {"n": 0, "audio": 0.0, "proc": 0.0}
                try:
                    e = {"n": int(e.get("n", 0)) + 1,
                         "audio": float(e.get("audio", 0.0)) + audio_s,
                         "proc": float(e.get("proc", 0.0)) + proc_s}
                except Exception:
                    continue
                node[slot] = e
            stats[key] = node
            tmp = str(ETA_PATH) + ".tmp"
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump(stats, f, indent=1)
            os.replace(tmp, str(ETA_PATH))
        except Exception:
            pass


def format_eta(seconds) -> str:
    try:
        s = max(0, int(round(float(seconds))))
    except Exception:
        return ""
    if s < 60:
        return f"{s}s"
    if s < 3600:
        return f"{s // 60}:{s % 60:02d}"
    return f"{s // 3600}h{(s % 3600) // 60:02d}m"


class ProgressPump:
    """Keeps bar/%/ETA alive across ONE blocking call (Whisper native pass,
    a slow Canary chunk) that otherwise freezes progress for minutes.

    Committed linear countdown: advances toward base+99% of span over the
    estimated seconds and HOLDS at 99% if the estimate is exceeded (never
    backward, never past it - the real completion sets the true value).
    A daemon thread ticks; stop() is idempotent; the pump never raises.
    """

    def __init__(self, prog, base: float, span: float, total_s: float,
                 label: str = ""):
        self._prog = prog
        self._base = max(0.0, min(1.0, float(base)))
        self._span = max(0.0, float(span))
        try:
            self._total = max(1.0, float(total_s))
        except Exception:
            self._total = 60.0
        self._label = label or ""
        self._stop = threading.Event()
        self._thread = None
        self._t0 = time.monotonic()

    def start(self):
        if self._thread is not None:
            return self
        try:
            self._thread = threading.Thread(target=self._tick, daemon=True)
            self._thread.start()
        except Exception:
            self._thread = None
        return self

    def _tick(self):
        try:
            while not self._stop.wait(0.5):
                try:
                    el = max(0.0, time.monotonic() - self._t0)
                    frac = self._base + self._span * min(el / self._total, 0.99)
                    self._prog(min(self._base + self._span * 0.99, frac),
                               f"{self._label} ({int(el)}s)".strip())
                except Exception:
                    pass
        except Exception:
            pass

    def stop(self):
        try:
            self._stop.set()
        except Exception:
            pass
        try:
            if self._thread is not None:
                self._thread.join(timeout=2)
        except Exception:
            pass
        self._thread = None

    def __enter__(self):
        return self.start()

    def __exit__(self, *exc):
        self.stop()
        return False


def _eta_key_for(engine_kind, arch=0) -> str:
    ek = engine_kind or ""
    if "Whisper" in ek:
        return "whisper"
    if "Canary" in ek:
        return "canary"
    try:
        return f"moonshine:{int(arch)}"
    except Exception:
        return "moonshine"


class EtaTracker:
    """Remaining-time estimator: committed countdown from the bucketed
    baseline. No mid-run hedging - the estimate is fixed once audio length
    is known, counts linearly to 0, and the ACTUAL outcome is recorded for
    the next aggregate."""

    def __init__(self, key: str):
        self.key = key or "job"
        self.t0 = time.monotonic()
        try:
            self.factor = float(_eta_factor(self.key))
        except Exception:
            self.factor = 1.0
        self.dur = None
        self.total = None

    def set_duration(self, seconds):
        try:
            d = float(seconds)
            self.dur = d if d > 0 else None
        except Exception:
            self.dur = None
        if self.dur:
            # Resolve the LENGTH-CLASS factor now (audio length just arrived).
            try:
                self.factor = float(_eta_factor(self.key, self.dur))
            except Exception:
                pass
            try:
                self.total = max(1.0, self.dur * max(0.05, self.factor))
            except Exception:
                self.total = None
        else:
            self.total = None

    def elapsed(self) -> float:
        try:
            return max(0.0, time.monotonic() - self.t0)
        except Exception:
            return 0.0

    def remaining(self, progress=None):
        el = self.elapsed()
        if self.total:
            return max(0.0, self.total - el)
        try:
            p = float(progress)
        except Exception:
            return None
        if p is not None and p > 0.05:
            return max(0.0, el / max(p, 1e-6) - el)
        return None

    def suffix(self, progress) -> str:
        try:
            if float(progress) >= 0.999:
                return ""
            r = self.remaining(progress)
        except Exception:
            return ""
        if r is None:
            return ""
        return f" \u00b7 ETA {format_eta(r)}"


def _dlen(s: str) -> int:
    """Display width: CJK full-width chars count 2 (a 42-col line fits
    ~21 kanji, not 42). Without this, JA cues overflow the video frame."""
    w = 0
    for c in s:
        w += 2 if unicodedata.east_asian_width(c) in ("W", "F") else 1
    return w


def _width_cut(s: str, limit: int) -> Tuple[str, str]:
    """Split s into (fit, rest) where display width of fit <= limit."""
    w = 0
    for i, c in enumerate(s):
        w += 2 if unicodedata.east_asian_width(c) in ("W", "F") else 1
        if w > limit:
            return s[:i], s[i:]
    return s, ""

PORTABLE_ROOT = Path(__file__).parent

# SRT cue limits - prevents flooding the video screen
MAX_LINE_CHARS = 42
MAX_LINES_PER_CUE = 2
MAX_CUE_SECONDS = 7.0
MIN_CUE_SECONDS = 0.6  # short/one-word cues vanish sooner instead of lingering
MIN_GAP_SECONDS = 0.08

SUPPORTED_EXTS = (".mp4", ".mkv", ".avi", ".mov", ".wmv", ".flv", ".webm",
                  ".mp3", ".wav", ".m4a", ".aac", ".ogg", ".flac", ".wma")

# Unambiguous audio-only files: nothing to burn a subtitle image into.
# Ambiguous containers (.webm/.ogg can carry video) are NOT listed here -
# they go through probing and fail with a clear message when audioless.
AUDIO_ONLY_EXTS = frozenset({".mp3", ".wav", ".m4a", ".aac",
                             ".flac", ".wma", ".opus"})


def is_audio_only_path(path) -> bool:
    """True for unambiguous audio-only files (auto-burn skips them with a
    status mark instead of a burn failure). Never raises."""
    try:
        return str(Path(path).suffix or "").lower() in AUDIO_ONLY_EXTS
    except Exception:
        return False


def cpu_count() -> int:
    try:
        return max(1, os.cpu_count() or 1)
    except Exception:
        return 4


def default_workers() -> int:
    """80% of max cores, at least 1 (CPU processing only)."""
    return max(1, int(round(cpu_count() * 0.8)))


def configure_cpu(n: int):
    """Apply CPU thread budget to BLAS libs + torch (no CUDA anywhere)."""
    try:
        n = max(1, int(n))
    except Exception:
        n = default_workers()
    os.environ["OMP_NUM_THREADS"] = str(n)
    os.environ["MKL_NUM_THREADS"] = str(n)
    os.environ["OPENBLAS_NUM_THREADS"] = str(n)
    try:
        import torch
        torch.set_num_threads(n)
        try:
            torch.set_num_interop_threads(min(n, 4))
        except Exception:
            pass
    except Exception:
        pass
    return n


def get_ffmpeg_exe() -> Optional[str]:
    """Portable-first ffmpeg resolution (no admin, no PATH needed)."""
    # 1) Optional manual override inside the folder
    for cand in (PORTABLE_ROOT / "ffmpeg" / "bin" / "ffmpeg.exe",
                 PORTABLE_ROOT / "ffmpeg" / "ffmpeg.exe"):
        try:
            if cand.exists() and cand.stat().st_size > 1_000_000:
                return str(cand)
        except Exception:
            pass
    # 2) Bundled binary from imageio-ffmpeg wheel (lives inside venv/)
    try:
        import imageio_ffmpeg
        exe = imageio_ffmpeg.get_ffmpeg_exe()
        if exe and Path(exe).exists():
            return exe
    except Exception:
        pass
    # 3) System PATH fallback
    found = shutil.which("ffmpeg")
    if found:
        return found
    return None


def format_ts(seconds: float) -> str:
    """SRT timestamp. Total-millisecond arithmetic: the old split-and-carry
    version emitted 00:00:60,000 at exact minute boundaries (59.9999s)."""
    try:
        total_ms = max(0, int(round(float(seconds) * 1000)))
    except Exception:
        total_ms = 0
    h, rem = divmod(total_ms, 3600000)
    m, rem = divmod(rem, 60000)
    s, ms = divmod(rem, 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def extract_audio(src: Path, dst_wav: Path, ffmpeg: str, cancel_event=None,
                  normalize_audio: bool = False):
    """Extract 16kHz mono wav. Raises on failure. Polls cancel_event (Popen,
    not run(): a 2h video extract would otherwise ignore Cancel for minutes).

    normalize_audio: single-pass EBU R128 loudnorm (I=-16 LUFS) for quiet /
    uneven recordings - raises soft speech before transcription. The burned
    MP4 is unaffected (it re-encodes from the original source, never this
    wav). Costs one extra audio filter pass, no temp stats file."""
    cmd = [ffmpeg, "-y", "-v", "error", "-i", str(src)]
    if normalize_audio:
        cmd += ["-af", "loudnorm=I=-16:TP=-1.5:LRA=11"]
    cmd += ["-ac", "1", "-ar", "16000", "-vn", str(dst_wav)]
    try:
        proc = subprocess.Popen(cmd, stdout=subprocess.DEVNULL,
                                stderr=subprocess.PIPE)
    except FileNotFoundError:
        raise RuntimeError(f"ffmpeg not executable: {ffmpeg}")
    try:
        while proc.poll() is None:
            if cancel_event is not None and cancel_event.is_set():
                try:
                    proc.terminate()
                except Exception:
                    pass
                try:
                    proc.wait(timeout=5)
                except Exception:
                    try:
                        proc.kill()
                    except Exception:
                        pass
                raise InterruptedError("cancelled")
            try:
                proc.wait(timeout=0.2)
            except subprocess.TimeoutExpired:
                pass
        _, err = proc.communicate(timeout=30)
    except InterruptedError:
        raise
    except Exception as e:
        try:
            proc.kill()
        except Exception:
            pass
        raise RuntimeError(f"ffmpeg extract failed: {e}")
    if proc.returncode != 0 or not dst_wav.exists():
        msg = ""
        try:
            msg = (err.decode("utf-8", "ignore") if isinstance(err, bytes) else (err or ""))[:300]
        except Exception:
            pass
        raise RuntimeError(f"ffmpeg extract failed: {msg}")


def load_wav_16k(wav_path: Path) -> Tuple[np.ndarray, int]:
    import soundfile as sf
    audio, sr = sf.read(str(wav_path), dtype="float32", always_2d=False)
    audio = np.asarray(audio, dtype=np.float32).flatten()
    if sr != 16000:
        # Simple resample fallback (ffmpeg already outputs 16k, this is just safety)
        import math
        ratio = 16000 / float(sr)
        idx = (np.arange(int(len(audio) * ratio)) / ratio).astype(np.int64)
        idx = np.clip(idx, 0, len(audio) - 1)
        audio = audio[idx]
        sr = 16000
    return audio, sr


def vad_segments(audio: np.ndarray, sr: int = 16000,
                 frame_ms: int = 30, min_silence_s: float = 0.5,
                 max_chunk_s: float = 20.0,
                 target_chunk_s: float = 12.0) -> List[Tuple[float, float]]:
    """Energy VAD -> speech segments, merged toward target length (better for translation context)."""
    if len(audio) == 0:
        return []
    frame_len = max(1, int(sr * frame_ms / 1000))
    n_frames = int(np.ceil(len(audio) / frame_len))
    if n_frames == 0:
        return []
    padded = np.pad(audio, (0, n_frames * frame_len - len(audio)))
    frames = padded.reshape(n_frames, frame_len)
    rms = np.sqrt(np.mean(frames ** 2, axis=1) + 1e-12)
    noise = float(np.percentile(rms, 10)) if len(rms) else 0.0
    thresh = max(0.004, noise * 2.5)
    active = rms > thresh
    if not bool(active.any()):
        return [(0.0, len(audio) / sr)]

    # Raw speech spans separated by >= min_silence_s
    min_sil_frames = max(1, int(min_silence_s * 1000 / frame_ms))
    spans = []
    in_speech = False
    start = 0
    sil = 0
    for i, a in enumerate(active):
        if a:
            if not in_speech:
                in_speech = True
                start = i
            sil = 0
        else:
            if in_speech:
                sil += 1
                if sil >= min_sil_frames:
                    spans.append((start, i - sil + 1))
                    in_speech = False
                    sil = 0
    if in_speech:
        spans.append((start, n_frames))

    # Merge short spans toward target length; hard-split overlong ones
    merged: List[Tuple[float, float]] = []
    cur_s, cur_e = None, None
    max_frames = int(max_chunk_s * 1000 / frame_ms)
    target_frames = int(target_chunk_s * 1000 / frame_ms)
    for s, e in spans:
        # Hard-split span longer than max
        while e - s > max_frames:
            merged.append((s * frame_len / sr, (s + max_frames) * frame_len / sr))
            s += max_frames
        if cur_s is None:
            cur_s, cur_e = s, e
        elif (e - cur_s) <= target_frames:
            cur_e = e
        else:
            merged.append((cur_s * frame_len / sr, cur_e * frame_len / sr))
            cur_s, cur_e = s, e
    if cur_s is not None:
        merged.append((cur_s * frame_len / sr, cur_e * frame_len / sr))
    total = len(audio) / sr
    return [(max(0.0, s), min(total, e)) for s, e in merged if e > s]


def speech_activity(wav_path, frame_ms: int = 30):
    """Frame-level speech activity for a wav file, streamed from disk.

    Returns (active_bool_array, sample_rate). Same energy rule as
    vad_segments (noise-floor percentile), but never holds the full PCM in
    RAM - only one float per frame (~1MB for 2h). Used to snap Whisper's
    approximate native timestamps onto real speech.
    """
    import soundfile as sf
    mean_sq: list = []
    sr = 16000
    with sf.SoundFile(str(wav_path)) as f:
        try:
            sr = int(f.samplerate or 16000)
        except Exception:
            sr = 16000
        frame_len = max(1, int(sr * frame_ms / 1000))
        buf = np.zeros(0, dtype=np.float32)
        while True:
            try:
                block = f.read(sr * 30, dtype="float32", always_2d=False)
            except Exception:
                break
            if block is None or len(block) == 0:
                break
            arr = np.asarray(block, dtype=np.float32).flatten()
            buf = np.concatenate([buf, arr]) if buf.size else arr
            n_full = len(buf) // frame_len
            if n_full:
                frames = buf[:n_full * frame_len].reshape(n_full, frame_len)
                mean_sq.extend((frames.astype(np.float64) ** 2).mean(axis=1).tolist())
                buf = buf[n_full * frame_len:]
        if buf.size:
            pad = np.pad(buf.astype(np.float64), (0, frame_len - buf.size))
            mean_sq.append(float((pad ** 2).mean()))
    rms = np.sqrt(np.array(mean_sq, dtype=np.float64) + 1e-12)
    if len(rms) == 0:
        return np.zeros(0, dtype=bool), sr
    noise = float(np.percentile(rms, 10)) if len(rms) else 0.0
    thresh = max(0.004, noise * 2.5)
    return (rms > thresh), sr


# Exact-match music/SFX markers Whisper emits over non-speech (trailers!).
# Conservative: only bare markers are dropped, never real sentences.
HALLUCINATION_MARKERS = frozenset({
    "[music]", "*music*", "(music)", "music",
    "[musique]", "[musica]", "[musik]",
    "[applause]", "*applause*", "(applause)",
    "[laughter]", "*laughter*",
    "♪", "♫", "♪♪", "[♪]", "(♪)",
})

REFINE_SEARCH_S = 1.0    # look this far outside each boundary for speech
REFINE_PRE_PAD_S = 0.15  # keep a breath before word onsets (no clipping)
REFINE_POST_PAD_S = 0.30  # natural hang time after offsets
REFINE_MIN_DUR_S = 0.4


def refine_segments_with_vad(segments: List[Tuple[float, float, str]],
                             activity, frame_ms: int = 30, words=None):
    """Snap Whisper-native (start, end) onto measured speech energy.

    Whisper timestamps routinely start early (even 0.000) and span silence.
    Segments with no speech activity in [start-1s, end+1s] are dropped
    (hallucinations over silence/music intros), as are bare music markers.

    If per-word stamps are supplied, each segment is additionally anchored
    to its first/last word (the only exact timing for short/one-word cues),
    with the energy window kept as guardrails so alignment glitches (e.g. a
    0.00 first-word stamp) cannot push a cue into silence.
    Returns (refined_list, stats_dict).
    """
    stats = {"snapped": 0, "dropped": 0, "markers": 0, "anchored": 0}
    if activity is None or len(activity) == 0:
        return list(segments), stats
    try:
        if not bool(activity.any()):
            # Degenerate input (e.g. fully-loud file with no quiet frames:
            # the noise-floor estimate then sits at speech level and nothing
            # passes). Keep everything rather than deleting the transcript.
            return list(segments), stats
    except Exception:
        return list(segments), stats
    words = list(words or [])
    fps = 1000.0 / float(frame_ms)
    n = len(activity)
    total = n / fps
    out: List[Tuple[float, float, str]] = []
    for s, e, t in segments:
        txt = (t or "").strip()
        if not txt:
            stats["dropped"] += 1
            continue
        if txt.lower() in HALLUCINATION_MARKERS:
            stats["markers"] += 1
            continue
        lo = max(0.0, s - REFINE_SEARCH_S)
        hi = min(total, e + REFINE_SEARCH_S)
        i_lo = max(0, int(lo * fps))
        i_s = max(0, int(s * fps))
        i_e = min(n, int(e * fps) + 1)
        i_hi = min(n, int(hi * fps) + 1)
        # Start: first speech at/after s (never drag the start forward into
        # a LATER speech burst); fall back to speech just before s.
        f0 = -1
        if i_hi > i_s:
            idx = np.flatnonzero(activity[i_s:i_hi])
            if idx.size:
                f0 = i_s + int(idx[0])
        if f0 < 0 and i_s > i_lo:
            idx = np.flatnonzero(activity[i_lo:i_s])
            if idx.size:
                f0 = i_lo + int(idx[-1])
        # End: last speech at/before e (never drag the end back over a gap
        # into an EARLIER burst); fall back to speech just after e.
        f1 = -1
        if i_e > i_lo:
            idx = np.flatnonzero(activity[i_lo:i_e])
            if idx.size:
                f1 = i_lo + int(idx[-1])
        if f1 < 0 and i_hi > i_e:
            idx = np.flatnonzero(activity[i_e:i_hi])
            if idx.size:
                f1 = i_e + int(idx[0])
        if f0 < 0 or f1 < 0 or f1 < f0:
            stats["dropped"] += 1
            continue
        ns = max(0.0, f0 / fps - REFINE_PRE_PAD_S)
        ne = min(total, (f1 + 1) / fps + REFINE_POST_PAD_S)
        if ne - ns < REFINE_MIN_DUR_S:
            ne = min(total, ns + REFINE_MIN_DUR_S)
        if words:
            sel = select_seg_words(s, e, words)
            if sel:
                w0 = min(ws for ws, _we, _wt in sel) - 0.10
                w1 = max(we for _ws, we, _wt in sel) + 0.15
                ns2 = max(0.0, w0, ns - 0.25)
                ne2 = min(w1, ne + 0.25)
                if ne2 - ns2 >= 0.3:
                    if abs(ns2 - ns) > 0.03 or abs(ne2 - ne) > 0.03:
                        stats["snapped"] += 1
                    ns, ne = ns2, ne2
                    stats["anchored"] += 1
                    out.append((ns, ne, txt))
                    continue
        if abs(ns - s) > 0.05 or abs(ne - e) > 0.05:
            stats["snapped"] += 1
        out.append((ns, ne, txt))
    return out, stats


_SENT_END = re.compile(r"(.+?[\.!?…。！？;；]+[\"'」』\)\]]*\s*|.+?$)", re.DOTALL)


def split_sentences(text: str) -> List[str]:
    """Split on EN + CJK sentence punctuation, keeping delimiters."""
    text = re.sub(r"\s+", " ", (text or "")).strip()
    if not text:
        return []
    parts = []
    for m in _SENT_END.finditer(text):
        s = m.group(1).strip()
        if s:
            parts.append(s)
    # CJK often has no spaces: further split very long runs at 。！？ if missed
    out = []
    for p in parts:
        if len(p) > MAX_LINE_CHARS * MAX_LINES_PER_CUE * 2:
            chunks = re.split(r"(?<=[。！？])", p)
            out.extend(c.strip() for c in chunks if c.strip())
        else:
            out.append(p)
    return out or ([text] if text else [])


def _wrap_lines(text: str) -> List[str]:
    """Wrap to <=2 lines of <=42 display columns, prefer word/CJK boundaries."""
    words = text.split(" ") if " " in text else list(text)
    lines: List[str] = []
    cur = ""
    joiner = " " if " " in text else ""
    for w in words:
        trial = (cur + joiner + w).strip() if cur else w
        if _dlen(trial) <= MAX_LINE_CHARS:
            cur = trial
        else:
            if cur:
                lines.append(cur)
            cur = w
            while _dlen(cur) > MAX_LINE_CHARS:  # single long token: hard cut
                fit, cur = _width_cut(cur, MAX_LINE_CHARS)
                if not fit:  # safety: never emit empty line / loop forever
                    fit, cur = cur[:1], cur[1:]
                lines.append(fit)
    if cur:
        lines.append(cur)
    return lines[:MAX_LINES_PER_CUE] if len(lines) > MAX_LINES_PER_CUE else lines


def select_seg_words(s: float, e: float, words) -> list:
    """Word stamps (ws,we,wt) relevant to span [s,e] (with 0.5s margin)."""
    if not words:
        return []
    lo, hi = s - 0.5, e + 0.5
    sel = [(ws, we, wt) for (ws, we, wt) in (words or [])
           if ws < hi and we > lo]
    sel.sort(key=lambda x: (x[0], x[1]))
    return sel


def _ns_spans(strs) -> List[Tuple[int, int]]:
    """Non-space char spans (joiner-independent: 'a b' and 'ab' align)."""
    spans = []
    pos = 0
    for x in strs:
        n = sum(1 for c in x if not c.isspace())
        spans.append((pos, pos + n))
        pos += n
    return spans


def map_sentences_to_words(sents: List[str], words) -> list:
    """Map each sentence to (first_word_start, last_word_end) via non-space
    character overlap with the word stamps; None when uncovered."""
    if not sents or not words:
        return [None] * len(sents)
    s_spans = _ns_spans(sents)
    w_spans = _ns_spans([wt for (_ws, _we, wt) in words])
    out = []
    for (a, b) in s_spans:
        hits = [(ws, we) for (ws, we, _wt), (c, d) in zip(words, w_spans)
                if c < b and d > a]
        if hits:
            out.append((min(x for x, _y in hits), max(y for _x, y in hits)))
        else:
            out.append(None)
    return out


def build_cues(segments: List[Tuple[float, float, str]],
               seg_words=None) -> List[Tuple[float, float, str]]:
    """Pack (start, end, text) into SRT cues: sentence-aware, no screen flooding.

    seg_words (optional): per-segment word-stamp lists aligned by segment
    index (as returned for refined segments). When a cue's sentences all
    have word coverage, the cue starts on its own first word instead of a
    proportional time share - this fixes short/late sentences appearing
    1-2s early. Falls back to proportional timing otherwise.
    """
    cues: List[Tuple[float, float, str]] = []
    for seg_idx, (seg_start, seg_end, text) in enumerate(segments):
        sents = split_sentences(text)
        if not sents:
            continue
        seg_dur = max(0.1, seg_end - seg_start)
        segw: list = []
        if seg_words is not None and seg_idx < len(seg_words or []):
            try:
                segw = list(seg_words[seg_idx] or [])
            except Exception:
                segw = []
        smap = map_sentences_to_words(sents, segw)
        total_w = sum(_dlen(s) for s in sents) or 1
        # Greedy pack sentences into cues (budgets in display columns)
        cur_sents: List[str] = []
        cur_spans: list = []
        cur_w = 0
        cur_t0 = seg_start

        def flush():
            nonlocal cur_sents, cur_spans, cur_w, cur_t0
            if not cur_sents:
                return
            joined = " ".join(cur_sents) if any(" " in s for s in cur_sents) else "".join(cur_sents)
            lines = _wrap_lines(joined)
            wpresent = [sp for sp in cur_spans if sp is not None]
            have_words = (len(wpresent) == len(cur_sents) and bool(wpresent))
            # If still overflowing (very long sentence), split by width across cues
            if len(lines) >= MAX_LINES_PER_CUE and _dlen(joined) > MAX_LINE_CHARS * MAX_LINES_PER_CUE:
                # Emit in chunks of 84 columns with proportional time
                chunk_size = MAX_LINE_CHARS * MAX_LINES_PER_CUE
                if have_words:
                    # Slice the sentences' own word-time span, not the segment share
                    b0 = max(seg_start, min(a for a, _b in wpresent))
                    b1 = min(seg_end, max(b for _a, b in wpresent))
                    span = max(0.3, b1 - b0)
                    t = b0
                else:
                    span = seg_dur * (cur_w / total_w)
                    t = cur_t0
                dur_share = span
                rest = joined
                while rest:
                    part, rest = _width_cut(rest, chunk_size)
                    if not part:
                        part, rest = rest[:1], rest[1:]
                    frac = _dlen(part) / max(1, cur_w)
                    d = max(MIN_CUE_SECONDS, min(MAX_CUE_SECONDS, dur_share * frac))
                    cues.append((t, min(seg_end, t + d), "\n".join(_wrap_lines(part))))
                    t += d
                cur_t0 = t
            else:
                if have_words:
                    # Each cue starts on its own first word: a sentence spoken
                    # late in a long segment no longer inherits an early
                    # proportional share (the classic 1-2s-early subtitle).
                    bw0 = max(seg_start, min(a for a, _b in wpresent) - 0.06)
                    bw1 = min(seg_end, max(b for _a, b in wpresent) + 0.10)
                    if bw1 - bw0 >= 0.3:
                        cues.append((bw0, bw1, "\n".join(lines)))
                        cur_t0 = min(seg_end, bw1)
                        cur_sents, cur_spans, cur_w = [], [], 0
                        return
                dur_share = seg_dur * (cur_w / total_w)
                d = max(MIN_CUE_SECONDS, min(MAX_CUE_SECONDS, dur_share))
                cues.append((cur_t0, min(seg_end, cur_t0 + d), "\n".join(lines)))
                cur_t0 = min(seg_end, cur_t0 + d)
            cur_sents, cur_spans, cur_w = [], [], 0

        for s, sp in zip(sents, smap):
            trial = cur_w + _dlen(s)
            if cur_sents and (trial > MAX_LINE_CHARS * MAX_LINES_PER_CUE
                              or len(cur_sents) >= 2):
                flush()
            cur_sents.append(s)
            cur_spans.append(sp)
            cur_w += _dlen(s)
        flush()

    # Enforce min gap + monotonic times
    fixed: List[Tuple[float, float, str]] = []
    prev_end = -1.0
    for s, e, t in cues:
        if not t.strip():
            continue
        s = max(s, prev_end + MIN_GAP_SECONDS if prev_end >= 0 else s)
        if e <= s:
            e = s + MIN_CUE_SECONDS
        fixed.append((s, e, t))
        prev_end = e
    return fixed


def write_srt(cues: List[Tuple[float, float, str]], path: Path):
    with open(path, "w", encoding="utf-8") as f:
        for i, (s, e, t) in enumerate(cues, 1):
            f.write(f"{i}\n{format_ts(s)} --> {format_ts(e)}\n{t.strip()}\n\n")


def default_out_path(src: Path, out_dir: Optional[str]) -> Path:
    if out_dir and str(out_dir).strip():
        d = Path(str(out_dir).strip())
        d.mkdir(parents=True, exist_ok=True)
        return d / (src.stem + ".srt")
    return src.parent / (src.stem + ".srt")


def _reserve_names(pairs, base_fn) -> dict:
    """{index: Path} with same-stem batch siblings suffixed " (2)"/" (3)".
    Only batch-internal collisions rename - pre-existing files still
    overwrite on re-runs (desired), so the filesystem is NOT consulted."""
    seen = set()
    out = {}
    for i, (src_path, out_dir) in enumerate(pairs or []):
        try:
            base = base_fn(Path(src_path), out_dir)
        except Exception:
            continue
        cand, n = base, 2
        while cand.name.lower() in seen:
            cand = base.with_name(f"{base.stem} ({n}){base.suffix}")
            n += 1
        seen.add(cand.name.lower())
        out[i] = cand
    return out


def reserve_batch_names(pairs) -> dict:
    """Disambiguated .srt outputs for a batch (see _reserve_names)."""
    return _reserve_names(pairs, default_out_path)


def reserve_burn_names(pairs) -> dict:
    """Disambiguated .burned.mp4 outputs for a batch (see _reserve_names)."""
    return _reserve_names(pairs, default_burn_path)


def transcribe_chunk_moonshine(transcriber, audio_f32: np.ndarray, sr: int,
                               chunk_start: float) -> List[Tuple[float, float, str]]:
    """Returns [(abs_start, abs_end, text)] using Moonshine line timestamps."""
    audio_list = np.clip(audio_f32.astype(np.float32), -1.0, 1.0).tolist()
    transcript = transcriber.transcribe_without_streaming(audio_list, sample_rate=sr)
    out = []
    if transcript and transcript.lines:
        for line in transcript.lines:
            txt = (line.text or "").strip()
            if not txt:
                continue
            try:
                ls = float(getattr(line, "start_time", 0.0) or 0.0)
                ld = float(getattr(line, "duration", 0.0) or 0.0)
            except Exception:
                ls, ld = 0.0, 0.0
            if ld <= 0:
                ld = max(0.8, len(txt) * 0.06)  # ~16 chars/sec fallback
            out.append((chunk_start + max(0.0, ls), chunk_start + max(0.0, ls) + ld, txt))
    return out


def run_srt_job(src_path: str, out_dir: str, engine_kind: str,
                moonshine_arch: int, canary_task: str, canary_src: str,
                cpu_workers: int, get_moonshine_transcriber: Callable,
                get_canary_engine: Callable,
                progress_cb: Optional[Callable[[float, str], None]] = None,
                log_cb: Optional[Callable[[str], None]] = None,
                cancel_event: Optional[threading.Event] = None,
                 whisper_task: str = "translate", whisper_src: str = "ja",
                 get_whisper_engine: Optional[Callable] = None,
                 srt_input_lang: str = "auto",
                 srt_output_lang: str = "en",
                 out_path: Optional[str] = None,
                 normalize_audio: bool = False) -> str:
    """Blocking SRT job. progress_cb(fraction 0..1, message). Returns output .srt path."""
    eta = EtaTracker(_eta_key_for(engine_kind, moonshine_arch))

    def prog(f, m):
        if progress_cb:
            try:
                progress_cb(max(0.0, min(1.0, f)), str(m) + eta.suffix(f))
            except Exception:
                pass

    def log(m):
        if log_cb:
            try:
                log_cb(m)
            except Exception:
                pass

    def cancelled() -> bool:
        return bool(cancel_event is not None and cancel_event.is_set())

    src = Path(src_path)
    if not src.exists():
        raise FileNotFoundError(f"Input not found: {src}")
    if src.suffix.lower() not in SUPPORTED_EXTS:
        raise ValueError(f"Unsupported type {src.suffix}. Supported: {', '.join(SUPPORTED_EXTS)}")

    workers = configure_cpu(cpu_workers)
    log(f"CPU: {cpu_count()} cores detected, using {workers} threads "
        f"for audio prep (inference device is on the Transcribing line)")
    # Normalize language codes (GUI already validates, this is belt-and-braces
    # for direct API callers). translate tasks always output English.
    srt_input_lang = (srt_input_lang or "auto").strip().lower()
    srt_output_lang = (srt_output_lang or "en").strip().lower()
    log(f"Language: input={srt_input_lang}, output={srt_output_lang}")

    def _wait_for_model(eng, name: str, loading_msg: str, load_pct: float):
        """Shared wait loop for heavy engines. Raises on failure/timeout/cancel."""
        import time as _t
        if eng.is_ready:
            return
        prog(load_pct, loading_msg)
        log(f"Loading {name} model...")
        eng.load()
        waited = 0
        # Pump the bar through the (possibly minutes-long) load: without it
        # a first-time Canary/Whisper download sits frozen at load_pct.
        with ProgressPump(prog, load_pct, 0.04, 300.0,
                          loading_msg):
            while not eng.is_ready and waited < 900:
                if cancelled():
                    raise InterruptedError("cancelled")
                _t.sleep(1)
                waited += 1
                try:
                    loading = bool(getattr(eng, "_loading", True))
                except Exception:
                    loading = True
                if not loading and not eng.is_ready:
                    err = getattr(eng, "_last_error", None) or "unknown error"
                    raise RuntimeError(f"{name} failed to load: {err}")
                # The pump carries the bar (never backward); the log carries
                # the elapsed time, sparingly.
                if waited % 30 == 0:
                    log(f"  still loading {name} ({waited}s)...")
        if not eng.is_ready:
            raise RuntimeError(f"{name} timed out loading.")

    ffmpeg = get_ffmpeg_exe()
    if not ffmpeg:
        raise RuntimeError("ffmpeg not found. It ships in wheels\\ (imageio-ffmpeg) - run setup.bat once.")
    log(f"ffmpeg: {ffmpeg}")

    out_path = Path(out_path) if out_path else default_out_path(src, out_dir)
    try:
        out_path.parent.mkdir(parents=True, exist_ok=True)
    except Exception:
        pass
    log(f"Output: {out_path} (default: same folder as source)")
    tmp_wav = src.parent / (src.stem + ".srt_tmp16k.wav")
    if out_path.resolve() == tmp_wav.resolve():
        tmp_wav = tmp_wav.with_name(src.stem + "_audio16k.wav")

    try:
        # Stage 1: extract (0-10%)
        if cancelled():
            raise InterruptedError("cancelled")
        prog(0.02, "Extracting audio (ffmpeg 16kHz mono)...")
        log("Extracting audio..." + (" [normalize: loudnorm -16 LUFS]"
                                     if normalize_audio else ""))
        extract_audio(src, tmp_wav, ffmpeg, cancel_event,
                      normalize_audio=normalize_audio)
        if cancelled():
            raise InterruptedError("cancelled")
        import soundfile as sf
        info = sf.info(str(tmp_wav))
        duration = float(info.frames) / float(info.samplerate or 16000)
        log(f"Audio: {duration:.1f}s, {info.samplerate}Hz")
        eta.set_duration(duration)
        prog(0.10, f"Audio ready ({duration:.0f}s)")

        # Stage 2: speech segments (10-15%).
        # Whisper native pass needs no VAD (it timestamps the whole file in
        # one go), so defer the full-RAM load + VAD until the fallback path.
        # A 2h file is ~460MB float32 (+ a padded copy in VAD) - skipping it
        # keeps peak RAM ~1GB lower alongside the 1.5-3.9GB models.
        audio = None
        sr = 16000
        spans: List[Tuple[float, float]] = []
        is_whisper = (engine_kind == "Whisper Large v3")
        if not is_whisper:
            audio, sr = load_wav_16k(tmp_wav)
            prog(0.12, "Detecting speech segments...")
            spans = vad_segments(audio, sr)
            if not spans:
                raise RuntimeError("No speech detected in file.")
            log(f"Speech segments: {len(spans)}")
            prog(0.15, f"{len(spans)} speech segments")

        def _ensure_vad():
            """Lazy load + VAD for the Whisper chunk-fallback path."""
            nonlocal audio, sr, spans
            if spans:
                return
            prog(0.12, "Detecting speech segments...")
            audio, sr = load_wav_16k(tmp_wav)
            spans = vad_segments(audio, sr)
            if not spans:
                raise RuntimeError("No speech detected in file.")
            log(f"Speech segments: {len(spans)}")
            prog(0.15, f"{len(spans)} speech segments")

        # Stage 3: transcribe (15-90%)
        segments: List[Tuple[float, float, str]] = []
        # Word stamps from the Whisper native pass ([]) for chunk paths).
        # Carried into cue packing so each cue starts on its own first word.
        _words_ready: list = []
        # First "[Engine Error: ...]" seen in any chunk. Surfaced at the end
        # instead of the misleading generic "No speech transcribed."
        first_engine_error: Optional[str] = None
        if engine_kind == "Canary-1B":
            eng = get_canary_engine()
            # Per-call langs (no shared-engine mutation -> live keeps working
            # with its own settings while this job runs).
            eff_src = srt_input_lang if srt_input_lang != "auto" else (canary_src or "ja")
            eff_tgt = "en" if canary_task == "translate" else (srt_output_lang or "en")
            _wait_for_model(eng, "Canary-1B",
                            "Loading Canary-1B (3.8GB, slow first time)...", 0.16)
            # Fail fast: e.g. Japanese is not in this model's tokenizer
            # (supports de/en/es/fr only) - churning all chunks through
            # NeMo just to explode each one helps nobody.
            _sup = getattr(eng, "supported_source_langs", None)
            _checker = getattr(eng, "_check_lang_support", None)
            if _sup and callable(_checker):
                _err = _checker(_sup, canary_task, eff_src, eff_tgt)
                if _err:
                    raise RuntimeError(_err)
            log(f"Transcribing {len(spans)} chunks with Canary-1B ({canary_task} {eff_src}->{eff_tgt}, {getattr(eng, 'device_info', 'cpu')})...")
            import soundfile as _sf
            import tempfile as _tf
            _n_spans = max(1, len(spans))
            for i, (s, e) in enumerate(spans):
                if cancelled():
                    raise InterruptedError("cancelled")
                # Announce BEFORE the blocking call: a single Canary chunk can
                # take 30s+ on CPU, and without this the bar/log look frozen.
                _slice_base = 0.15 + 0.75 * i / _n_spans
                _slice_span = 0.75 / _n_spans
                prog(_slice_base,
                     f"Canary chunk {i + 1}/{len(spans)} ({s:.0f}s, {e - s:.1f}s audio)...")
                log(f"  chunk {i + 1}/{len(spans)}: transcribing {e - s:.1f}s audio...")
                s_i, e_i = int(s * sr), int(e * sr)
                chunk = audio[s_i:e_i]
                with _tf.NamedTemporaryFile(suffix=".wav", delete=False) as tf:
                    tmp_c = tf.name
                got_text = False
                # Pump ticks the bar through the blocking inference below.
                with ProgressPump(prog, _slice_base, _slice_span,
                                  max(3.0, (e - s) * max(0.3, eta.factor)),
                                  f"Canary chunk {i + 1}/{len(spans)}"):
                    try:
                        _sf.write(tmp_c, chunk, samplerate=sr)
                        text = eng.transcribe_file(tmp_c, task=canary_task,
                                                   source_lang=eff_src,
                                                   target_lang=eff_tgt)
                        if text and text.strip() and not text.startswith("["):
                            segments.append((s, e, text.strip()))
                            got_text = True
                        elif text and text.startswith("[") and first_engine_error is None:
                            first_engine_error = text.strip()
                    finally:
                        try:
                            os.unlink(tmp_c)
                        except Exception:
                            pass
                prog(0.15 + 0.75 * (i + 1) / len(spans),
                     f"Canary chunk {i + 1}/{len(spans)} done")
                if got_text:
                    log(f"  chunk {i + 1}/{len(spans)}: {segments[-1][2][:60]}...")
                else:
                    log(f"  chunk {i + 1}/{len(spans)}: (silence)")
        elif engine_kind == "Whisper Large v3":
            if get_whisper_engine is None:
                raise RuntimeError("Whisper engine not wired. Update moonshine_stt.py.")
            eng = get_whisper_engine()
            # Per-call langs (no shared-engine mutation -> live keeps working
            # with its own settings while this job runs). "auto" passes
            # through so Whisper auto-detects; translate always outputs en.
            eff_wsrc = srt_input_lang or "auto"
            eff_wtgt = "en" if whisper_task == "translate" else (srt_output_lang or "en")
            _wait_for_model(eng, f"Whisper {getattr(eng, 'model_id', 'Large v3')}",
                            f"Loading Whisper {getattr(eng, 'model_id', 'Large v3')} "
                            f"(slow first time - downloads when missing)...", 0.16)
            # Per-model ETA: tiny and large-v3 differ ~10x - sharing one
            # "whisper" key poisoned both estimates. The factor re-resolves
            # here (family fallback keeps old history useful until this
            # size records its own samples).
            try:
                eta.key = "whisper:" + _eta_safe_key(
                    getattr(eng, "model_id", ""), "large-v3")
                eta.factor = float(_eta_factor(eta.key, eta.dur))
            except Exception:
                pass
            _wmodel = getattr(eng, 'model_id', 'Large v3')
            _wdev = getattr(eng, 'device_info', 'cpu')
            log(f"Transcribing full audio with Whisper {_wmodel} ({whisper_task} {eff_wsrc}->{eff_wtgt}, {_wdev}, native timestamps)...")
            prog(0.20, "Whisper transcribing (single pass)...")
            if cancelled():
                raise InterruptedError("cancelled")
            # Pump keeps bar/%/ETA alive through the one long blocking call
            # (previously frozen at 20% until the whole file finished).
            _pump = ProgressPump(prog, 0.20, 0.70,
                                 max(10.0, duration * max(0.2, eta.factor)),
                                 "Whisper transcribing").start()
            try:
                # Prefer word stamps (exact placement incl. one-word cues);
                # older engines without transcribe_file_words fall back to
                # plain segments (refine() then uses VAD timing only).
                _wm = getattr(eng, "transcribe_file_words", None)
                if callable(_wm):
                    native, _words = _wm(str(tmp_wav), task=whisper_task,
                                         source_lang=eff_wsrc)
                else:
                    native = eng.transcribe_file_segments(
                        str(tmp_wav), task=whisper_task, source_lang=eff_wsrc)
                    _words = []
            except Exception as ex:
                log(f"  native pass failed, falling back to chunks: {ex}")
                native = []
            finally:
                _pump.stop()
            refined: List[Tuple[float, float, str]] = []
            if native:
                # Whisper boundaries are approximate (often early / spanning
                # silence): snap them onto measured speech energy and drop
                # pure-silence hallucinations before cue packing.
                try:
                    _act, _asr = speech_activity(str(tmp_wav))
                    native, _ref = refine_segments_with_vad(native, _act,
                                                            words=_words)
                    log(f"  timing refined: {len(native)} kept "
                        f"({_ref['snapped']} snapped, {_ref['anchored']} word-anchored, "
                        f"{_ref['dropped']} silence-dropped, "
                        f"{_ref['markers']} music-markers, "
                        f"{len(_words)} word stamps)")
                except Exception as ex:
                    log(f"  timing refine skipped: {ex}")
                if native:
                    refined = native
                    segments.extend(native)
                    _words_ready = list(_words or [])
                    prog(0.90, f"Whisper done ({len(native)} segments)")
                    log(f"  native segments: {len(native)}")
            if not refined:
                # Fallback: VAD chunks (same as Canary path). VAD computed
                # lazily here so the common native path never pays for it.
                log("  native pass empty, trying VAD chunks...")
                if cancelled():
                    raise InterruptedError("cancelled")
                _ensure_vad()
                import soundfile as _sf
                import tempfile as _tf
                for i, (s, e) in enumerate(spans):
                    if cancelled():
                        raise InterruptedError("cancelled")
                    prog(0.15 + 0.75 * i / len(spans),
                         f"Whisper chunk {i + 1}/{len(spans)} ({s:.0f}s, {e - s:.1f}s audio)...")
                    log(f"  chunk {i + 1}/{len(spans)}: transcribing {e - s:.1f}s audio...")
                    s_i, e_i = int(s * sr), int(e * sr)
                    chunk = audio[s_i:e_i]
                    with _tf.NamedTemporaryFile(suffix=".wav", delete=False) as tf:
                        tmp_c = tf.name
                    try:
                        _sf.write(tmp_c, chunk, samplerate=sr)
                        text = eng.transcribe_file(tmp_c, task=whisper_task,
                                                   source_lang=eff_wsrc)
                        if text and text.strip() and not text.startswith("["):
                            segments.append((s, e, text.strip()))
                        elif text and text.startswith("[") and first_engine_error is None:
                            first_engine_error = text.strip()
                    finally:
                        try:
                            os.unlink(tmp_c)
                        except Exception:
                            pass
                    prog(0.15 + 0.75 * (i + 1) / len(spans),
                         f"Whisper chunk {i + 1}/{len(spans)} done")
        else:
            # Moonshine: dedicated per-job Transcriber (does not disturb live engine)
            prog(0.16, "Loading Moonshine model for file...")
            transcriber, arch_name = get_moonshine_transcriber()
            log(f"Transcribing {len(spans)} chunks with Moonshine {arch_name}...")
            try:
                for i, (s, e) in enumerate(spans):
                    if cancelled():
                        raise InterruptedError("cancelled")
                    prog(0.15 + 0.75 * i / len(spans),
                         f"Moonshine chunk {i + 1}/{len(spans)} ({s:.0f}s, {e - s:.1f}s audio)...")
                    s_i, e_i = int(s * sr), int(e * sr)
                    chunk = audio[s_i:e_i]
                    try:
                        lines = transcribe_chunk_moonshine(transcriber, chunk, sr, s)
                        segments.extend(lines)
                    except Exception as ex:
                        log(f"  chunk {i + 1} error: {ex}")
                        if first_engine_error is None:
                            first_engine_error = "[Moonshine Error: %s]" % ex
                    prog(0.15 + 0.75 * (i + 1) / len(spans),
                         f"Moonshine chunk {i + 1}/{len(spans)} done")
            finally:
                # Always release native resources, notably on Cancel (the old
                # code skipped close() on early exit, leaking per job).
                try:
                    transcriber.close()
                except Exception:
                    pass

        if cancelled():
            raise InterruptedError("cancelled")
        if not segments:
            if first_engine_error:
                raise RuntimeError(first_engine_error.strip("[] "))
            raise RuntimeError("No speech transcribed.")
        # Release the full-file PCM before cue packing (up to ~500MB for
        # long files) - segments/cues are tiny by comparison.
        audio = None

        # Stage 4: cue packing + write (90-100%)
        prog(0.92, "Breaking sentences into cues...")
        seg_words = None
        if _words_ready:
            try:
                seg_words = [select_seg_words(s, e, _words_ready)
                             for (s, e, _t) in segments]
            except Exception:
                seg_words = None
        cues = build_cues(segments, seg_words)
        log(f"Cues: {len(cues)} (max {MAX_LINES_PER_CUE}x{MAX_LINE_CHARS} chars, {MAX_CUE_SECONDS:.0f}s)")
        write_srt(cues, out_path)
        prog(1.0, f"Done: {out_path.name} ({len(cues)} cues)")
        log(f"Wrote {out_path}")
        try:
            _el = eta.elapsed()
            log(f"Total: {format_eta(_el)} for {duration:.1f}s audio "
                f"({duration / max(0.1, _el):.1f}x realtime)")
        except Exception:
            pass
        try:
            record_eta_sample(eta.key, duration, eta.elapsed())
        except Exception:
            pass
        return str(out_path)
    finally:
        try:
            if tmp_wav.exists():
                tmp_wav.unlink()
        except Exception:
            pass


def _batch_short(path: str, limit: int = 40) -> str:
    """Basename truncated for progress/log prefixes."""
    try:
        name = Path(path).name or str(path)
    except Exception:
        name = str(path)
    return name if len(name) <= limit else name[:limit - 1] + "…"


def run_srt_batch(src_paths, run_one: Callable,
                  progress_cb: Optional[Callable[[float, str], None]] = None,
                  log_cb: Optional[Callable[[str], None]] = None,
                  cancel_event: Optional[threading.Event] = None,
                  file_cb: Optional[Callable] = None):
    """Run SRT jobs sequentially over a file queue (one worker thread).

    run_one(path, progress_cb, log_cb) -> output .srt path (may raise;
    InterruptedError on cancel). Per-file progress/log lines are prefixed
    with "[i/N name]". file_cb(kind, path, info) reports "start"/"done"/
    "skip" for queue-UI updates. A failed file is recorded and the batch
    continues; cancel stops after the current file. Never raises for
    per-file errors - returns (results, cancelled) where results holds
    (path, ok, out_or_error) tuples in queue order.
    """
    paths = [str(p) for p in (src_paths or []) if str(p or "").strip()]
    total = len(paths)
    results: list = []

    def _cancelled() -> bool:
        return bool(cancel_event is not None and cancel_event.is_set())

    def _emit(kind, path, info):
        if file_cb:
            try:
                file_cb(kind, path, info or {})
            except Exception:
                pass

    for i, path in enumerate(paths):
        tag = f"[{i + 1}/{total} {_batch_short(path)}]"
        if _cancelled():
            for j in range(i, total):
                q = paths[j]
                results.append((q, False, "Skipped (cancelled)"))
                _emit("skip", q, {"index": j, "total": total,
                                  "reason": "cancelled"})
            try:
                if log_cb:
                    log_cb(f"{tag} batch cancelled")
            except Exception:
                pass
            return results, True
        _emit("start", path, {"index": i, "total": total})
        try:
            if progress_cb:
                progress_cb(0.0, f"{tag} starting...")
        except Exception:
            pass

        def _prog(f, m, _tag=tag):
            if progress_cb:
                try:
                    progress_cb(max(0.0, min(1.0, f)), f"{_tag} {m}")
                except Exception:
                    pass

        def _log(m, _tag=tag):
            if log_cb:
                try:
                    log_cb(f"{_tag} {m}")
                except Exception:
                    pass

        try:
            out = run_one(path, _prog, _log)
            results.append((path, True, out))
            _emit("done", path, {"index": i, "total": total, "ok": True,
                                 "out": out})
        except InterruptedError:
            results.append((path, False, "Cancelled by user"))
            _emit("done", path, {"index": i, "total": total, "ok": False,
                                 "error": "Cancelled by user"})
            for j in range(i + 1, total):
                q = paths[j]
                results.append((q, False, "Skipped (cancelled)"))
                _emit("skip", q, {"index": j, "total": total,
                                  "reason": "cancelled"})
            return results, True
        except Exception as e:
            err = str(e) or type(e).__name__
            results.append((path, False, err))
            _emit("done", path, {"index": i, "total": total, "ok": False,
                                 "error": err})
            try:
                if log_cb:
                    log_cb(f"{tag} FAILED: {err} (continuing batch)")
            except Exception:
                pass
    return results, False


# ---------------- Subtitle burn-in (hardcode into a new MP4) ----------------
# Audio codecs the MP4 container accepts for stream-copy; anything else is
# re-encoded to AAC at (about) its original bitrate so total size still lands.
MP4_OK_AUDIO = {"aac", "ac3", "mp3", "opus"}
BURN_OVERHEAD = 0.015  # reserve for moov/moov atom differences between containers
BURN_SUFFIX = ".burned.mp4"


def default_burn_path(src: Path, out_dir: Optional[str]) -> Path:
    if out_dir and str(out_dir).strip():
        d = Path(str(out_dir).strip())
        d.mkdir(parents=True, exist_ok=True)
        return d / (src.stem + BURN_SUFFIX)
    return src.parent / (src.stem + BURN_SUFFIX)


# Burn speed modes: id -> dict(encoder/preset/passes). x264 two-pass is the
# only combo promising ±1-3% size; 1-pass ABR drifts (~±10%); ultrafast is
# visibly softer at the same size. NVENC (h264_nvenc) is 3-6x faster than
# x264 but coarser at hitting sizes even in multipass VBR - offered as
# explicit Draft/Balanced choices (greyed out without NVIDIA), never silent.
BURN_SPEEDS = {
    "match": {"encoder": "cpu", "preset": "medium", "passes": 2,
              "label": "Match size (2-pass x264)"},
    "fast": {"encoder": "cpu", "preset": "veryfast", "passes": 1,
             "label": "Fast (1-pass x264)"},
    "fastest": {"encoder": "cpu", "preset": "ultrafast", "passes": 1,
                "label": "Fastest (ultrafast 1-pass)"},
    "nvenc_draft": {"encoder": "nvenc", "preset": "p1", "passes": 1,
                    "tune": "hq",
                    "label": "Draft (NVENC fast 1-pass)"},
    "nvenc_turbo": {"encoder": "nvenc", "preset": "p1", "passes": 1,
                    "tune": "ull",
                    "label": "Turbo (NVENC ultra-fast 1-pass)"},
    "nvenc_balanced": {"encoder": "nvenc", "preset": "p4", "passes": 2,
                       "tune": "hq",
                       "label": "Balanced (NVENC 2-pass)"},
}
BURN_SPEED_LABELS = {k: v["label"] for k, v in BURN_SPEEDS.items()}
BURN_SPEED_IDS = {v: k for k, v in BURN_SPEED_LABELS.items()}

# Video codec axis (orthogonal to speed). HEVC applies to the NVENC
# speeds only - the CPU speeds are x264 by definition (x265 lives behind
# a future "CPU codec" switch, not this one). ~30% smaller than H.264 at
# equal quality on typical content; needs ~2016+ playback hardware.
BURN_CODECS = {"h264": "H.264", "hevc": "HEVC (H.265)"}


def normalize_burn_codec(codec) -> str:
    """'hevc' or 'h264' (anything else -> h264). Never raises."""
    try:
        return "hevc" if str(codec or "").strip().lower() == "hevc" else "h264"
    except Exception:
        return "h264"


def _burn_history_key(speed_id, codec="h264") -> str:
    """Size-history key (burn_size.json). HEVC speeds learn separately
    (different efficiency!) - any hevc encode, CPU or NVENC, gets its own
    +hevc key; H.264 keeps the long-standing bare speed id so all existing
    history keeps working."""
    sid = _burn_speed_id(speed_id)
    if normalize_burn_codec(codec) == "hevc":
        return f"{sid}+hevc"
    return sid


def _burn_eta_key(speed_id, codec="h264") -> str:
    """ETA-history key (srt_eta.json). Same split as the size history, but
    under the long-standing "burn:" namespace (family fallback and the
    clear-burn filter both key off that prefix)."""
    sid = _burn_speed_id(speed_id)
    if normalize_burn_codec(codec) == "hevc":
        return f"burn:{sid}+hevc"
    return f"burn:{sid}"


def resolve_burn_speed(speed, codec="h264") -> dict:
    """Full speed config dict; unknown/empty -> exact-match mode.

    encoder: "cpu" (x264) | "cpu_hevc" (x265) | "nvenc" (H.264) |
    "nvenc_hevc". The codec switches both CPU (x264/x265) and NVENC
    (H.264/HEVC) outputs."""
    try:
        key = str(speed or "match").strip().lower()
    except Exception:
        key = "match"
    cfg = BURN_SPEEDS.get(key)
    if not isinstance(cfg, dict):
        cfg = BURN_SPEEDS["match"]
        key = "match"
    out = dict(cfg)
    if normalize_burn_codec(codec) == "hevc":
        if out.get("encoder") == "nvenc":
            out["encoder"] = "nvenc_hevc"
        elif out.get("encoder") == "cpu":
            out["encoder"] = "cpu_hevc"
    return out


def _burn_speed_id(speed) -> str:
    """Validated burn speed id (same fallback as resolve_burn_speed).
    Used for the per-speed ETA key - one speed's samples must never leak
    into another's (ultrafast vs 2-pass differ ~10x)."""
    try:
        key = _eta_safe_key(speed, "match")
    except Exception:
        key = "match"
    return key if key in BURN_SPEEDS else "match"


def probe_media(path, ffmpeg: str) -> dict:
    """ffprobe-less probe via `ffmpeg -i` stderr (imageio-ffmpeg ships no
    ffprobe binary). Raises RuntimeError when duration/size unreadable."""
    proc = subprocess.run(
        [ffmpeg, "-hide_banner", "-i", str(path)],
        stdout=subprocess.DEVNULL, stderr=subprocess.PIPE,
        text=True, errors="replace")
    err = proc.stderr or ""
    info = {"duration": 0.0, "fps": 0.0, "width": 0, "height": 0,
            "vcodec": "", "acodec": "", "audio_bps": 0,
            "has_audio": False, "overall_bps": 0, "size": 0}
    m = re.search(r"Duration:\s*(\d+):(\d+):([\d.]+)", err)
    if m:
        try:
            info["duration"] = (int(m.group(1)) * 3600 + int(m.group(2))
                                + float(m.group(3)))
        except Exception:
            pass
    m = re.search(r"bitrate:\s*(\d+)\s*kb/s", err)
    if m:
        try:
            info["overall_bps"] = int(m.group(1)) * 1000
        except Exception:
            pass
    for line in err.splitlines():
        s = line.strip()
        if ": Video:" in s and not info["vcodec"]:
            mv = re.search(r"Video:\s*([a-z0-9_]+)", s)
            if mv:
                info["vcodec"] = mv.group(1).lower()
            mr = re.search(r"(\d{3,5})x(\d{3,5})", s)
            if mr:
                try:
                    info["width"], info["height"] = int(mr.group(1)), int(mr.group(2))
                except Exception:
                    pass
            mf = re.search(r"(\d+(?:\.\d+)?)\s*fps", s)
            if not mf:
                mf = re.search(r"(\d+(?:\.\d+)?)\s*tbr", s)
            if mf:
                try:
                    info["fps"] = float(mf.group(1))
                except Exception:
                    pass
        if ": Audio:" in s and not info["has_audio"]:
            info["has_audio"] = True
            ma = re.search(r"Audio:\s*([a-z0-9_]+)", s)
            if ma:
                info["acodec"] = ma.group(1).lower()
            mb = re.search(r"(\d+)\s*kb/s", s)
            if mb:
                try:
                    info["audio_bps"] = int(mb.group(1)) * 1000
                except Exception:
                    pass
    try:
        info["size"] = Path(path).stat().st_size
    except Exception:
        info["size"] = 0
    if info["duration"] <= 0 or info["size"] <= 0:
        raise RuntimeError(f"could not probe media: {path}")
    _fixup_duration(info)
    if info["duration"] <= 0:
        raise RuntimeError(f"could not probe media: {path}")
    return info


def _fixup_duration(info: dict) -> dict:
    """Correct a lying container Duration (observed: 67s reported for a
    914s VFR/edit-list MP4 - 27,412 frames at "408fps", while 27,412/30fps,
    the SRT ETA, and the output size all agreed on ~15min).

    File size never lies: size*8/bitrate must roughly equal duration. When
    all three readings exist but size*8/bitrate disagrees with the header
    duration by 3x or more, trust size/bitrate and flag it. Otherwise (or
    with any reading missing) the probe is left untouched. Never raises."""
    try:
        info["duration_source"] = "header"
        d = float(info.get("duration") or 0)
        s = int(info.get("size") or 0)
        b = int(info.get("overall_bps") or 0)
    except Exception:
        return info
    if d > 0 and s > 0 and b > 0:
        try:
            implied = s * 8.0 / b
        except Exception:
            return info
        if implied > d * 3.0 or implied < d / 3.0:
            info["duration"] = implied
            info["duration_source"] = "size/bitrate"
    return info


def plan_burn_bitrates(info: dict, target_bytes: Optional[int] = None):
    """Split a total-size budget into (video_bps, audio_bps, audio_copy).

    Default target = original file size, so a 1GB video burns to ~1GB:
    audio is copied when MP4-compatible (exact bytes) else re-encoded near
    its probed rate, and x264 two-pass spends whatever is left on video.
    """
    try:
        total = int(target_bytes) if target_bytes else int(info["size"])
    except Exception:
        total = int(info.get("size", 0))
    dur = max(0.1, float(info.get("duration", 0) or 0))
    if info.get("has_audio"):
        acodec = (info.get("acodec") or "").lower()
        if acodec in MP4_OK_AUDIO:
            audio_copy = True
            audio_bps = int(info.get("audio_bps") or 128000)
        else:
            audio_copy = False
            try:
                audio_bps = min(int(info.get("audio_bps") or 128000), 192000)
            except Exception:
                audio_bps = 128000
    else:
        audio_copy, audio_bps = False, 0
    usable_bps = total * 8 * (1.0 - BURN_OVERHEAD) / dur
    video_bps = int(max(100000, usable_bps - audio_bps))
    return video_bps, audio_bps, audio_copy


def _has_cjk(text: str) -> bool:
    for c in text or "":
        o = ord(c)
        if 0x3040 <= o <= 0x30FF or 0x4E00 <= o <= 0x9FFF or 0xAC00 <= o <= 0xD7AF:
            return True
    return False


def _esc_filter_path(p: str) -> str:
    return (str(p).replace("\\", "/").replace(":", "\\:").replace("'", "\\'"))


def stage_subtitles_filter(srt_path, font_size: int, dest_dir) -> str:
    """Copy the SRT under a plain-ASCII name into dest_dir and build the
    libass filter string. Returns the -vf value (caller keeps dest_dir
    alive for the encode).

    Font policy (all empirically verified against this exact libass
    build - see commit history, it is picky):
    - Latin-only SRTs -> Arial + full outline style (pixel-proven).
    - Any CJK       -> MS Gothic, size only. Full outline styles and every
      other CJK family render NOTHING here; MS Gothic minimal renders.
    """
    try:
        size = max(10, min(40, int(font_size)))
    except Exception:
        size = 18
    safe_srt = Path(dest_dir) / "subs.srt"
    shutil.copy(str(srt_path), str(safe_srt))
    try:
        cjk = _has_cjk(Path(srt_path).read_text(encoding="utf-8",
                                                errors="ignore"))
    except Exception:
        cjk = False
    if cjk:
        style = "FontName=MS Gothic,FontSize=%d" % size
    else:
        style = ("FontName=Arial,FontSize=%d,PrimaryColour=&H00FFFFFF,"
                 "OutlineColour=&H80000000,BorderStyle=1,Outline=1,"
                 "Shadow=0,MarginV=28" % size)
    return ("subtitles='%s':fontsdir='%s':force_style='%s'" % (
        _esc_filter_path(str(safe_srt)),
        _esc_filter_path("C:/Windows/Fonts"), style))


def first_cue_at(srt_path, fallback_ratio: float = 0.25) -> float:
    """Start time (seconds) of the first SRT cue +0.4s, for frame previews.
    Falls back to duration*fraction when unparseable (needs probe: caller
    may pass duration via fallback... kept simple - middle fallback)."""
    try:
        data = Path(srt_path).read_text(encoding="utf-8", errors="ignore")
        m = re.search(r"(\d+):(\d+):([\d.,]+)\s*-->", data)
        if m:
            sec = (int(m.group(1)) * 3600 + int(m.group(2)) * 60
                   + float(m.group(3).replace(",", ".")))
            return max(0.0, sec + 0.4)
    except Exception:
        pass
    return -1.0  # caller decides fallback (e.g. probe duration * ratio)


def parse_time_to_seconds(text, duration: float = 0.0) -> float:
    """Parse 'ss', 'mm:ss' or 'hh:mm:ss' (fractions allowed) into seconds,
    clamped to [0, duration-1] when duration is known. Garbage/empty ->
    10% into the file (or 30s default), so previews land on content."""
    try:
        t = str(text or "").strip().replace(",", ".")
    except Exception:
        t = ""
    val = None
    if t:
        try:
            parts = [float(x) for x in t.split(":")]
            if len(parts) == 1:
                val = parts[0]
            elif len(parts) == 2:
                val = parts[0] * 60 + parts[1]
            elif len(parts) == 3:
                val = parts[0] * 3600 + parts[1] * 60 + parts[2]
        except Exception:
            val = None
    if val is None or val < 0:
        try:
            dur = float(duration or 0)
        except Exception:
            dur = 0.0
        val = dur * 0.10 if dur > 0 else 30.0
    try:
        dur = float(duration or 0)
        if dur > 2:
            val = max(0.0, min(val, dur - 1.0))
        else:
            val = max(0.0, val)
    except Exception:
        pass
    return float(val)


def extract_clip(src_path, dst_wav, ffmpeg: str, start_s: float,
                 dur_s: float, cancel_event=None):
    """Cut [start, start+dur) to 16kHz mono wav. Fast-seeks to 5s before
    the mark, then accurately seeks the rest (instant even in long files,
    frame-accurate). Raises on failure."""
    try:
        start_s = max(0.0, float(start_s))
        dur_s = max(1.0, float(dur_s))
    except Exception:
        start_s, dur_s = 0.0, 15.0
    pre = max(0.0, start_s - 5.0)
    cmd = [ffmpeg, "-hide_banner", "-y", "-v", "error",
           "-ss", f"{pre:.3f}", "-i", str(src_path),
           "-ss", f"{start_s - pre:.3f}", "-t", f"{dur_s:.3f}",
           "-ac", "1", "-ar", "16000", "-vn", str(dst_wav)]
    try:
        proc = subprocess.Popen(cmd, stdout=subprocess.DEVNULL,
                                stderr=subprocess.PIPE)
    except FileNotFoundError:
        raise RuntimeError(f"ffmpeg not executable: {ffmpeg}")
    try:
        while proc.poll() is None:
            if cancel_event is not None and cancel_event.is_set():
                try:
                    proc.terminate()
                except Exception:
                    pass
                try:
                    proc.wait(timeout=5)
                except Exception:
                    try:
                        proc.kill()
                    except Exception:
                        pass
                raise InterruptedError("cancelled")
            try:
                proc.wait(timeout=0.2)
            except subprocess.TimeoutExpired:
                pass
        _, err = proc.communicate(timeout=30)
    except InterruptedError:
        raise
    except Exception as e:
        try:
            proc.kill()
        except Exception:
            pass
        raise RuntimeError(f"ffmpeg clip extract failed: {e}")
    if proc.returncode != 0 or not Path(dst_wav).exists():
        msg = ""
        try:
            msg = (err.decode("utf-8", "ignore")
                   if isinstance(err, bytes) else (err or ""))[:200]
        except Exception:
            pass
        raise RuntimeError(f"ffmpeg clip extract failed: {msg}")


_CUE_TS = re.compile(
    r"(\d+):(\d+):([\d.,]+)\s*-->\s*(\d+):(\d+):([\d.,]+)")


def _cue_secs(h1, m1, s1) -> float:
    return int(h1) * 3600 + int(m1) * 60 + float(str(s1).replace(",", "."))


def offset_srt_file(src_srt, offset_s: float, dst_srt):
    """Rewrite an SRT with every cue shifted by offset seconds (used to map
    a sample-clip SRT back onto the original video timeline)."""
    try:
        off = float(offset_s)
    except Exception:
        off = 0.0

    def _fmt(t: float) -> str:
        # format_ts clamps negatives and carries minutes/hours correctly.
        return format_ts(t)

    def _rep(m):
        a = _cue_secs(m.group(1), m.group(2), m.group(3)) + off
        b = _cue_secs(m.group(4), m.group(5), m.group(6)) + off
        return f"{_fmt(a)} --> {_fmt(b)}"

    data = Path(src_srt).read_text(encoding="utf-8", errors="ignore")
    Path(dst_srt).write_text(_CUE_TS.sub(_rep, data), encoding="utf-8")
    return str(dst_srt)


def _burn_popen_wait(proc, total_frames: int, base: float, span: float,
                     progress_cb=None, cancel_event=None, label: str = "",
                     stall_s: float = 600.0, stats: Optional[dict] = None):
    """Drain stderr on a thread (ffmpeg status lines use \\r, not \\n),
    poll for cancel, report frame-based progress. Raises on failure.

    Honesty rules: the bar NEVER claims 100% mid-encode. total_frames is
    only a duration×fps estimate (VFR / mis-probed fps break it), so the
    fraction is capped at 99% of span and only the caller's explicit
    completion update may show 100%. When frames overshoot the estimate a
    diagnostic flag lands in `stats` (caller logs it once).
    Two hang guards: the shared buffer is capped at 16KB (an hours-long
    encode would otherwise grow it unboundedly and re-scanning it each
    poll turns quadratic), and a stall detector kills ffmpeg when no new
    frame arrives for `stall_s` (corrupt input hanging x264 forever).
    On success, stats (if given) gets frames/elapsed/avg_fps/exceeded.
    """
    buf = bytearray()
    lock = threading.Lock()

    def _drain():
        try:
            while True:
                data = proc.stderr.read(65536)
                if not data:
                    break
                with lock:
                    buf.extend(data)
                    if len(buf) > 16384:
                        del buf[:-16384]
        except Exception:
            pass

    t = threading.Thread(target=_drain, daemon=True)
    t.start()
    last = 0
    last_change = time.monotonic()
    t_start = time.monotonic()
    try:
        while proc.poll() is None:
            if cancel_event is not None and cancel_event.is_set():
                try:
                    proc.terminate()
                except Exception:
                    pass
                try:
                    proc.wait(timeout=5)
                except Exception:
                    try:
                        proc.kill()
                    except Exception:
                        pass
                raise InterruptedError("cancelled")
            try:
                with lock:
                    snap = bytes(buf)
                if snap:
                    found = re.findall(rb"frame=\s*(\d+)", snap)
                    if found:
                        try:
                            cur = int(found[-1])
                        except Exception:
                            cur = last
                        if cur > last:
                            last, last_change = cur, time.monotonic()
            except Exception:
                pass
            if last > 0 and (time.monotonic() - last_change) > stall_s:
                try:
                    proc.terminate()
                except Exception:
                    pass
                try:
                    proc.wait(timeout=5)
                except Exception:
                    try:
                        proc.kill()
                    except Exception:
                        pass
                raise RuntimeError(
                    f"ffmpeg burn {label}stalled (no progress for "
                    f"{int(stall_s)}s) - input may be corrupt")
            if progress_cb and total_frames > 0:
                try:
                    denom = max(int(total_frames), last, 1)
                    frac = base + span * min(0.99, last / denom)
                    progress_cb(max(0.0, min(1.0, frac)),
                                f"{label}frame {last}/{total_frames}")
                except Exception:
                    pass
            try:
                proc.wait(timeout=0.5)
            except subprocess.TimeoutExpired:
                pass
    finally:
        try:
            t.join(timeout=10)
        except Exception:
            pass
    tail = ""
    try:
        with lock:
            tail = bytes(buf)[-600:].decode("utf-8", "ignore")
    except Exception:
        pass
    if proc.returncode != 0:
        raise RuntimeError(f"ffmpeg burn {label}failed: {tail.strip()[-300:]}")
    if last == 0:
        # Fast encode finished before the poll loop ever parsed a status
        # line (sub-second clip on many cores): recover the final count
        # from the leftover buffer so stats/fps stay truthful.
        try:
            with lock:
                _tailfound = re.findall(rb"frame=\s*(\d+)", bytes(buf))
            if _tailfound:
                last = int(_tailfound[-1])
        except Exception:
            pass
    if stats is not None:
        try:
            el = max(0.1, time.monotonic() - t_start)
            stats["frames"] = last
            stats["elapsed"] = el
            stats["avg_fps"] = (last / el) if last > 0 else 0.0
            stats["exceeded"] = bool(last > total_frames)
        except Exception:
            pass
    return tail


def burn_subtitles(src_path: str, srt_path: str, out_path: str, ffmpeg: str,
                   video_kbps: int, audio_kbps: int = 128,
                   audio_copy: bool = True, threads: int = 0,
                   font_size: int = 18, speed: str = "match",
                   progress_cb: Optional[Callable[[float, str], None]] = None,
                   log_cb: Optional[Callable[[str], None]] = None,
                   cancel_event: Optional[threading.Event] = None,
                   codec: str = "h264"):
    """Burn an SRT into a new MP4 at a fixed video bitrate (size targeting
    lives in plan_burn_bitrates). Same resolution/fps as the source.
    speed: "match" (x264 medium 2-pass, size within ~1-3%), "fast"
    (veryfast 1-pass, ~half the time, size within ~±10%), "fastest"
    (ultrafast 1-pass, several times faster, visibly softer). Unknown
    speeds fall back to "match". codec "hevc" switches the NVENC speeds
    to hevc_nvenc (~30% smaller, same GPU speed); CPU speeds stay x264.
    Returns (out_path, in_bytes, out_bytes)."""
    import tempfile as _tf
    src, srtp, out = Path(src_path), Path(srt_path), Path(out_path)
    if not srtp.exists():
        raise FileNotFoundError(f"SRT not found: {srtp} (Generate SRT first)")
    info = probe_media(src, ffmpeg)
    if not info.get("vcodec"):
        raise RuntimeError(f"no video stream to burn into: {src.name}")
    if info.get("duration_source") == "size/bitrate" and log_cb:
        # The header Duration disagreed grossly with size/bitrate (lying
        # VFR/edit-list container) - say which number won, or the Auto
        # budget and progress math silently target the wrong length.
        try:
            log_cb(f"probe: header duration disagreed with size/bitrate - "
                   f"using {float(info.get('duration') or 0):.0f}s")
        except Exception:
            pass
    _spd = resolve_burn_speed(speed, codec)
    preset, passes = _spd["preset"], _spd["passes"]
    _tune = _spd.get("tune") or "hq"
    _spd_id = _burn_speed_id(speed)
    _codec = normalize_burn_codec(codec)
    _enc = _spd.get("encoder") or "cpu"
    use_nvenc = _enc.startswith("nvenc")
    _nvenc_name = "hevc_nvenc" if _enc == "nvenc_hevc" else "h264_nvenc"
    if use_nvenc:
        # Defensive at the library layer too (the app pre-checks and the
        # GUI reverts, but direct callers deserve the clear error).
        try:
            import gpu as _gpumod
            _nv_ok = bool(_gpumod.nvenc_available(ffmpeg, _nvenc_name))
        except Exception:
            _nv_ok = False
        if not _nv_ok:
            raise RuntimeError(
                f"NVENC burn needs an NVIDIA GPU + {_nvenc_name} encoder - "
                "none detected. Pick a CPU burn speed instead.")
    beta = EtaTracker(_burn_eta_key(speed, _codec))
    beta.set_duration(info["duration"])

    def _bprog(f, m):
        if progress_cb:
            try:
                progress_cb(max(0.0, min(1.0, f)), str(m) + beta.suffix(f))
            except Exception:
                pass

    total_frames = max(1, int(info["duration"] * (info["fps"] or 30.0)))
    in_bytes = info["size"]
    try:
        _burn_wall0 = time.time()
    except Exception:
        _burn_wall0 = 0.0
    # NOTE on fonts: see stage_subtitles_filter - Arial+full-style for
    # Latin, MS Gothic minimal for CJK. Anything else renders blank here.
    tmpd = Path(_tf.mkdtemp(prefix="burn_"))
    try:
        # Plain-ASCII temp SRT copy: sidesteps filter-escaping pitfalls
        # (quotes/brackets/unicode in real filenames) entirely.
        vf = stage_subtitles_filter(srtp, font_size, tmpd)
        vbps = max(100, int(video_kbps))
        # Closed-loop overshoot compensation: some encoders (notably NVENC
        # 1-pass) systematically overshoot the requested rate, so without
        # this a speed that runs x1.4 hot stays x1.4 hot forever no matter
        # what kbps is set. With 2+ past burns at this speed, request
        # proportionally less so the file lands on target instead.
        # Undershoot is never compensated (smaller-than-target harms
        # nothing); x264 2-pass measures ~1.00 and is unaffected.
        try:
            _fudge, _fudge_n = burn_size_fudge(_spd_id, _codec)
            if _fudge_n >= 2 and _fudge is not None and _fudge > 1.02:
                _vbps_req = vbps
                vbps = max(100, int(round(vbps / _fudge)))
                if log_cb:
                    log_cb(f"overshoot compensation: target {_vbps_req}k, "
                           f"encoding {vbps}k (learned x{_fudge:.2f} from "
                           f"{_fudge_n} burns)")
        except Exception:
            pass
        base = [ffmpeg, "-hide_banner", "-y", "-v", "info", "-i", str(src),
                "-map", "0:v:0", "-map", "0:a:0?"]
        if use_nvenc:
            # NVENC VBR with headroom caps (multipass flag added at run time
            # below for the 2-pass mode). -threads is x264-only; NVENC
            # scales internally. -rc vbr is stated explicitly: without it
            # some builds fall back to a quality mode that silently ignores
            # -b:v/-maxrate (observed: ~3x oversize, immune to kbps cuts).
            # Lookahead + AQ cost a little speed but spend the SAME bits
            # visibly better (scene-cut awareness, detail-weighted
            # quantization) - free quality at identical file sizes.
            base += ["-c:v", _nvenc_name, "-preset", preset, "-tune", _tune,
                     "-rc", "vbr",
                     "-rc-lookahead", "32",
                     "-spatial-aq", "1", "-aq-strength", "8",
                     "-temporal-aq", "1",
                     "-b:v", f"{vbps}k",
                     "-maxrate", f"{int(vbps * 1.5)}k",
                     "-bufsize", f"{int(vbps * 2)}k",
                     "-pix_fmt", "yuv420p"]
            if _enc == "nvenc_hevc":
                # hvc1 (not hev1) brand: required by Apple players, harmless
                # everywhere else.
                base += ["-tag:v", "hvc1"]
        elif _enc == "cpu_hevc":
            # x265 mirrors the x264 flags 1:1 (same presets, same 2-pass
            # stats discipline, same -threads). ≈30-40% smaller, much slower.
            base += ["-c:v", "libx265", "-b:v", f"{vbps}k",
                     "-preset", preset, "-pix_fmt", "yuv420p",
                     "-tag:v", "hvc1"]
            if threads and int(threads) > 0:
                base += ["-threads", str(int(threads))]
        else:
            base += ["-c:v", "libx264", "-b:v", f"{vbps}k",
                     "-preset", preset, "-pix_fmt", "yuv420p"]
            if threads and int(threads) > 0:
                base += ["-threads", str(int(threads))]
        base += ["-vf", vf]
        if info["has_audio"]:
            if audio_copy:
                base += ["-c:a", "copy"]
            else:
                base += ["-c:a", "aac", "-b:a", f"{max(32, int(audio_kbps))}k",
                         "-ac", "2", "-ar", "48000"]
        else:
            base += ["-an"]
        passlog = str(tmpd / "x264pass")
        try:
            if out.exists():
                out.unlink()
        except Exception:
            pass
        def _speed_line(st: dict, what: str) -> None:
            # "pass 1/2: avg 412 fps" - a NUMBER beats fan-noise guessing.
            # Flags VFR-style estimate overshoot once (bar held at 99%).
            if log_cb is None:
                return
            try:
                fps = float((st or {}).get("avg_fps") or 0)
                if fps > 0:
                    log_cb(f"  {what}: avg {fps:.0f} fps")
            except Exception:
                pass
            try:
                if (st or {}).get("exceeded"):
                    log_cb(f"  note: {st.get('frames', '?')} frames vs "
                            f"{total_frames} estimated (VFR/high-fps?) - "
                            f"bar held at 99% instead of a false 100%")
            except Exception:
                pass

        if use_nvenc:
            # NVENC multipass=2-pass VBR happens INSIDE one ffmpeg run (not
            # two runs like x264) - size lands approximately (~±5-10%).
            if passes == 2:
                multi, mlabel = ["-multipass", "fullres"], "multipass 2-pass VBR"
            else:
                multi, mlabel = [], "1-pass VBR"
            if log_cb:
                try:
                    log_cb(f"burn ({_nvenc_name} {preset}, {mlabel}, {vbps} kbps video, size approx)...")
                except Exception:
                    pass
            p = subprocess.Popen(base + multi + [str(out)],
                                 stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
            _st: dict = {}
            _burn_popen_wait(p, total_frames, 0.0, 1.0, _bprog,
                             cancel_event, label="", stats=_st)
            _speed_line(_st, "encode")
        elif passes == 2:
            # NOTE: pass 1 and pass 2 MUST use the same x264 preset. A
            # faster pass-1 preset was tried (veryfast stats + medium
            # final) and x264 hard-fails pass 2 with "different weightp
            # setting than first pass" - the stats are NOT preset-agnostic.
            # (x264 already runs its own internal fast-firstpass; the two
            # speeds stay identical here by requirement, not by caution.)
            if log_cb:
                try:
                    log_cb(f"burn pass 1/2 (analysis, {vbps} kbps video)...")
                except Exception:
                    pass
            p1 = subprocess.Popen(base + ["-pass", "1", "-passlogfile", passlog,
                                          "-f", "mp4", os.devnull],
                                  stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
            _st1: dict = {}
            _burn_popen_wait(p1, total_frames, 0.0, 0.45, _bprog,
                             cancel_event, label="pass 1/2 ", stats=_st1)
            _speed_line(_st1, "pass 1/2")
            if log_cb:
                try:
                    log_cb("burn pass 2/2 (final encode)...")
                except Exception:
                    pass
            p2 = subprocess.Popen(base + ["-pass", "2", "-passlogfile", passlog,
                                          str(out)],
                                  stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
            _st2: dict = {}
            _burn_popen_wait(p2, total_frames, 0.45, 0.55, _bprog,
                             cancel_event, label="pass 2/2 ", stats=_st2)
            _speed_line(_st2, "pass 2/2")
        else:
            # Single ABR pass: ~half the time, size lands within ~±10%
            # instead of ~1-3% (logged honestly, totals still reported).
            if log_cb:
                try:
                    log_cb(f"burn single pass ({preset}, {vbps} kbps video, size approx)...")
                except Exception:
                    pass
            p = subprocess.Popen(base + [str(out)],
                                 stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
            _st0: dict = {}
            _burn_popen_wait(p, total_frames, 0.0, 1.0, _bprog,
                             cancel_event, label="", stats=_st0)
            _speed_line(_st0, "encode")
        try:
            out_bytes = out.stat().st_size if out.exists() else 0
        except Exception:
            out_bytes = 0
        if out_bytes <= 0:
            raise RuntimeError("burn produced no output file")
        # Honest 100%: the file exists and verified non-empty above.
        try:
            if progress_cb:
                progress_cb(1.0, f"Done: {out.name}")
        except Exception:
            pass
        if log_cb:
            try:
                _bel = beta.elapsed()
                _bdur = float(info.get("duration") or 0)
                log_cb(f"Burn total: {format_eta(_bel)} for {_bdur:.1f}s video "
                       f"({_bdur / max(0.1, _bel):.1f}x realtime)")
            except Exception:
                pass
        try:
            record_eta_sample(beta.key, info["duration"], beta.elapsed())
        except Exception:
            pass
        try:
            # Feed the size learner: probed audio rate when copied, the AAC
            # rate otherwise. Record the COMPENSATED kbps actually requested
            # (not the pre-compensation target) so the learned ratio keeps
            # measuring the encoder - not our own correction - and converges
            # instead of oscillating.
            _aeff = int(info.get("audio_bps") or 0) if audio_copy else \
                max(32, int(audio_kbps)) * 1000
            record_burn_sample(_spd_id, vbps, _aeff,
                               float(info.get("duration") or 0), out_bytes,
                               _codec)
        except Exception:
            pass
        return str(out), in_bytes, out_bytes
    except BaseException:
        # Never leave a half-encoded file masquerading as finished
        # (notably Cancel during pass 2) - but only remove files THIS call
        # wrote (mtime check against call start), never a previous output.
        try:
            if out.exists() and out.stat().st_mtime >= _burn_wall0 - 5:
                out.unlink()
        except Exception:
            pass
        raise
    finally:
        try:
            shutil.rmtree(str(tmpd), ignore_errors=True)
        except Exception:
            pass
