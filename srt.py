# srt.py
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
ETA_PATH = Path(__file__).parent / "srt_eta.json"
ETA_DEFAULTS = {"whisper": 1.0, "canary": 3.0, "moonshine": 0.3, "burn": 1.5,
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
        if float(node.get("audio") or 0) > 1.0:
            return node, True
    except Exception:
        pass
    return None, True
def _eta_factor(key: str, duration=None) -> float:
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
    try:
        s = str(part or fallback or "").strip().lower()
    except Exception:
        s = str(fallback or "")
    s = re.sub(r"[^a-z0-9_+-]", "", s) or str(fallback or "")
    return s[:40] or str(fallback or "")
def clear_eta_history(which: str = "all") -> int:
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
BURN_SIZE_PATH = Path(__file__).parent / "burn_size.json"
_BURN_LOCK = threading.Lock()
def record_burn_sample(speed_id, video_kbps, audio_bps, duration_s,
                       actual_bytes, codec="h264"):
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
        return
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
    if not (vkbps > 0) or vkbps != vkbps:
        return None
    try:
        return max(1, int(round(vkbps)))
    except Exception:
        return None
def record_eta_sample(key: str, audio_s: float, proc_s: float):
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
    w = 0
    for c in s:
        w += 2 if unicodedata.east_asian_width(c) in ("W", "F") else 1
    return w
def _width_cut(s: str, limit: int) -> Tuple[str, str]:
    w = 0
    for i, c in enumerate(s):
        w += 2 if unicodedata.east_asian_width(c) in ("W", "F") else 1
        if w > limit:
            return s[:i], s[i:]
    return s, ""
PORTABLE_ROOT = Path(__file__).parent
MAX_LINE_CHARS = 42
MAX_LINES_PER_CUE = 2
MAX_CUE_SECONDS = 7.0
MIN_CUE_SECONDS = 0.6
MIN_GAP_SECONDS = 0.08
SUPPORTED_EXTS = (".mp4", ".mkv", ".avi", ".mov", ".wmv", ".flv", ".webm",
                  ".mp3", ".wav", ".m4a", ".aac", ".ogg", ".flac", ".wma")
AUDIO_ONLY_EXTS = frozenset({".mp3", ".wav", ".m4a", ".aac",
                             ".flac", ".wma", ".opus"})
def is_audio_only_path(path) -> bool:
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
    return max(1, int(round(cpu_count() * 0.8)))
def configure_cpu(n: int):
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
    for cand in (PORTABLE_ROOT / "ffmpeg" / "bin" / "ffmpeg.exe",
                 PORTABLE_ROOT / "ffmpeg" / "ffmpeg.exe"):
        try:
            if cand.exists() and cand.stat().st_size > 1_000_000:
                return str(cand)
        except Exception:
            pass
    try:
        import imageio_ffmpeg
        exe = imageio_ffmpeg.get_ffmpeg_exe()
        if exe and Path(exe).exists():
            return exe
    except Exception:
        pass
    found = shutil.which("ffmpeg")
    if found:
        return found
    return None
def format_ts(seconds: float) -> str:
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
        ratio = 16000 / float(sr)
        out_len = int(len(audio) * ratio)
        if out_len <= 0:
            audio = np.zeros(0, dtype=np.float32)
        else:
            pos = np.arange(out_len, dtype=np.float64) / ratio
            i0 = np.floor(pos).astype(np.int64)
            i1 = np.minimum(i0 + 1, len(audio) - 1)
            i0 = np.clip(i0, 0, len(audio) - 1)
            frac = (pos - i0).astype(np.float32)
            audio = (audio[i0] * (1.0 - frac) + audio[i1] * frac).astype(np.float32)
        sr = 16000
    return audio, sr
def vad_segments(audio: np.ndarray, sr: int = 16000,
                 frame_ms: int = 30, min_silence_s: float = 0.5,
                 max_chunk_s: float = 20.0,
                 target_chunk_s: float = 12.0) -> List[Tuple[float, float]]:
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
    merged: List[Tuple[float, float]] = []
    cur_s, cur_e = None, None
    max_frames = int(max_chunk_s * 1000 / frame_ms)
    target_frames = int(target_chunk_s * 1000 / frame_ms)
    for s, e in spans:
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
HALLUCINATION_MARKERS = frozenset({
    "[music]", "*music*", "(music)", "music",
    "[musique]", "[musica]", "[musik]",
    "[applause]", "*applause*", "(applause)",
    "[laughter]", "*laughter*",
    "♪", "♫", "♪♪", "[♪]", "(♪)",
})
REFINE_SEARCH_S = 1.0
REFINE_PRE_PAD_S = 0.15
REFINE_POST_PAD_S = 0.30
REFINE_MIN_DUR_S = 0.4
def refine_segments_with_vad(segments: List[Tuple[float, float, str]],
                             activity, frame_ms: int = 30, words=None):
    stats = {"snapped": 0, "dropped": 0, "markers": 0, "anchored": 0}
    if activity is None or len(activity) == 0:
        return list(segments), stats
    try:
        if not bool(activity.any()):
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
        f0 = -1
        if i_hi > i_s:
            idx = np.flatnonzero(activity[i_s:i_hi])
            if idx.size:
                f0 = i_s + int(idx[0])
        if f0 < 0 and i_s > i_lo:
            idx = np.flatnonzero(activity[i_lo:i_s])
            if idx.size:
                f0 = i_lo + int(idx[-1])
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
    text = re.sub(r"\s+", " ", (text or "")).strip()
    if not text:
        return []
    parts = []
    for m in _SENT_END.finditer(text):
        s = m.group(1).strip()
        if s:
            parts.append(s)
    out = []
    for p in parts:
        if len(p) > MAX_LINE_CHARS * MAX_LINES_PER_CUE * 2:
            chunks = re.split(r"(?<=[。！？])", p)
            out.extend(c.strip() for c in chunks if c.strip())
        else:
            out.append(p)
    return out or ([text] if text else [])
def _wrap_lines(text: str) -> List[str]:
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
            while _dlen(cur) > MAX_LINE_CHARS:
                fit, cur = _width_cut(cur, MAX_LINE_CHARS)
                if not fit:
                    fit, cur = cur[:1], cur[1:]
                lines.append(fit)
    if cur:
        lines.append(cur)
    return lines[:MAX_LINES_PER_CUE] if len(lines) > MAX_LINES_PER_CUE else lines
def select_seg_words(s: float, e: float, words) -> list:
    if not words:
        return []
    lo, hi = s - 0.5, e + 0.5
    sel = [(ws, we, wt) for (ws, we, wt) in (words or [])
           if ws < hi and we > lo]
    sel.sort(key=lambda x: (x[0], x[1]))
    return sel
def _ns_spans(strs) -> List[Tuple[int, int]]:
    spans = []
    pos = 0
    for x in strs:
        n = sum(1 for c in x if not c.isspace())
        spans.append((pos, pos + n))
        pos += n
    return spans
def map_sentences_to_words(sents: List[str], words) -> list:
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
            if len(lines) >= MAX_LINES_PER_CUE and _dlen(joined) > MAX_LINE_CHARS * MAX_LINES_PER_CUE:
                chunk_size = MAX_LINE_CHARS * MAX_LINES_PER_CUE
                if have_words:
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
    return _reserve_names(pairs, default_out_path)
def reserve_burn_names(pairs) -> dict:
    return _reserve_names(pairs, default_burn_path)
def transcribe_chunk_moonshine(transcriber, audio_f32: np.ndarray, sr: int,
                               chunk_start: float) -> List[Tuple[float, float, str]]:
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
                ld = max(0.8, len(txt) * 0.06)
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
    srt_input_lang = (srt_input_lang or "auto").strip().lower()
    srt_output_lang = (srt_output_lang or "en").strip().lower()
    log(f"Language: input={srt_input_lang}, output={srt_output_lang}")
    def _wait_for_model(eng, name: str, loading_msg: str, load_pct: float):
        import time as _t
        if eng is None:
            raise RuntimeError(f"{name} engine unavailable")
        if eng.is_ready:
            return
        prog(load_pct, loading_msg)
        log(f"Loading {name} model...")
        eng.load()
        waited = 0
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
        segments: List[Tuple[float, float, str]] = []
        _words_ready: list = []
        first_engine_error: Optional[str] = None
        if engine_kind == "Canary-1B":
            eng = get_canary_engine()
            eff_src = srt_input_lang if srt_input_lang != "auto" else (canary_src or "ja")
            eff_tgt = "en" if canary_task == "translate" else (srt_output_lang or "en")
            _wait_for_model(eng, "Canary-1B",
                            "Loading Canary-1B (3.8GB, slow first time)...", 0.16)
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
            eff_wsrc = srt_input_lang or "auto"
            eff_wtgt = "en" if whisper_task == "translate" else (srt_output_lang or "en")
            _wait_for_model(eng, f"Whisper {getattr(eng, 'model_id', 'Large v3')}",
                            f"Loading Whisper {getattr(eng, 'model_id', 'Large v3')} "
                            f"(slow first time - downloads when missing)...", 0.16)
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
            _pump = ProgressPump(prog, 0.20, 0.70,
                                 max(10.0, duration * max(0.2, eta.factor)),
                                 "Whisper transcribing").start()
            try:
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
        audio = None
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
MP4_OK_AUDIO = {"aac", "ac3", "mp3", "opus"}
BURN_OVERHEAD = 0.015
BURN_SUFFIX = ".burned.mp4"
def default_burn_path(src: Path, out_dir: Optional[str]) -> Path:
    if out_dir and str(out_dir).strip():
        d = Path(str(out_dir).strip())
        d.mkdir(parents=True, exist_ok=True)
        return d / (src.stem + BURN_SUFFIX)
    return src.parent / (src.stem + BURN_SUFFIX)
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
BURN_CODECS = {"h264": "H.264", "hevc": "HEVC (H.265)"}
def normalize_burn_codec(codec) -> str:
    try:
        return "hevc" if str(codec or "").strip().lower() == "hevc" else "h264"
    except Exception:
        return "h264"
def _burn_history_key(speed_id, codec="h264") -> str:
    sid = _burn_speed_id(speed_id)
    if normalize_burn_codec(codec) == "hevc":
        return f"{sid}+hevc"
    return sid
def _burn_eta_key(speed_id, codec="h264") -> str:
    sid = _burn_speed_id(speed_id)
    if normalize_burn_codec(codec) == "hevc":
        return f"burn:{sid}+hevc"
    return f"burn:{sid}"
def resolve_burn_speed(speed, codec="h264") -> dict:
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
    try:
        key = _eta_safe_key(speed, "match")
    except Exception:
        key = "match"
    return key if key in BURN_SPEEDS else "match"
def probe_media(path, ffmpeg: str) -> dict:
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
        style = ("FontName=MS Gothic,FontSize=%d,PrimaryColour=&H00FFFFFF,"
                 "OutlineColour=&H80000000,BorderStyle=1,Outline=1,"
                 "Shadow=0,MarginV=28" % size)
    else:
        style = ("FontName=Arial,FontSize=%d,PrimaryColour=&H00FFFFFF,"
                 "OutlineColour=&H80000000,BorderStyle=1,Outline=1,"
                 "Shadow=0,MarginV=28" % size)
    return ("subtitles='%s':fontsdir='%s':force_style='%s'" % (
        _esc_filter_path(str(safe_srt)),
        _esc_filter_path("C:/Windows/Fonts"), style))
def first_cue_at(srt_path, fallback_ratio: float = 0.25) -> float:
    try:
        data = Path(srt_path).read_text(encoding="utf-8", errors="ignore")
        m = re.search(r"(\d+):(\d+):([\d.,]+)\s*-->", data)
        if m:
            sec = (int(m.group(1)) * 3600 + int(m.group(2)) * 60
                   + float(m.group(3).replace(",", ".")))
            return max(0.0, sec + 0.4)
    except Exception:
        pass
    return -1.0
def parse_time_to_seconds(text, duration: float = 0.0) -> float:
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
    try:
        off = float(offset_s)
    except Exception:
        off = 0.0
    def _fmt(t: float) -> str:
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
    import tempfile as _tf
    src, srtp, out = Path(src_path), Path(srt_path), Path(out_path)
    if not srtp.exists():
        raise FileNotFoundError(f"SRT not found: {srtp} (Generate SRT first)")
    info = probe_media(src, ffmpeg)
    if not info.get("vcodec"):
        raise RuntimeError(f"no video stream to burn into: {src.name}")
    if info.get("duration_source") == "size/bitrate" and log_cb:
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
    tmpd = Path(_tf.mkdtemp(prefix="burn_"))
    try:
        vf = stage_subtitles_filter(srtp, font_size, tmpd)
        vbps = max(100, int(video_kbps))
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
                base += ["-tag:v", "hvc1"]
        elif _enc == "cpu_hevc":
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
            _aeff = int(info.get("audio_bps") or 0) if audio_copy else \
                max(32, int(audio_kbps)) * 1000
            record_burn_sample(_spd_id, vbps, _aeff,
                               float(info.get("duration") or 0), out_bytes,
                               _codec)
        except Exception:
            pass
        return str(out), in_bytes, out_bytes
    except BaseException:
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