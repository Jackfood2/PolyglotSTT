import bisect
import json
import logging
import math
import os
import pathlib
import re
import shutil
import subprocess
import tempfile
import threading
import time
import unicodedata
import uuid
from fractions import Fraction
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np


_LOG = logging.getLogger(__name__)

PORTABLE_ROOT = Path(__file__).resolve().parent
ETA_PATH = PORTABLE_ROOT / "srt_eta.json"
BURN_SIZE_PATH = PORTABLE_ROOT / "burn_size.json"

ETA_DEFAULTS = {
    "whisper": 1.0,
    "canary": 3.0,
    "moonshine": 0.3,
    "burn": 1.5,
    "whisper:tiny": 0.08,
    "whisper:base": 0.12,
    "whisper:small": 0.2,
    "whisper:medium": 0.4,
    "whisper:large": 1.0,
    "whisper:large-v1": 1.0,
    "whisper:large-v2": 1.0,
    "whisper:large-v3": 1.0,
    "burn:match": 1.5,
    "burn:fast": 0.7,
    "burn:fastest": 0.35,
    "burn:nvenc_draft": 0.25,
    "burn:nvenc_turbo": 0.2,
    "burn:nvenc_balanced": 0.4,
}

ETA_BUCKETS = (
    ("s", 60.0),
    ("m", 600.0),
    ("l", float("inf")),
)

_ETA_LOCK = threading.RLock()
_BURN_LOCK = threading.RLock()

MAX_LINE_CHARS = 42
MAX_LINES_PER_CUE = 2
MAX_CUE_SECONDS = 7.0
MIN_CUE_SECONDS = 0.6
MIN_GAP_SECONDS = 0.0

CUE_START_LEAD_S = 0.060
CUE_END_EARLY_S = 0.030
WORD_PAUSE_BREAK_S = 0.65

# Strict-timing tolerance: fail the job only when MORE than this fraction
# of segments lack matching word timestamps (systemic alignment failure -
# wrong engine/language, broken words). A minority of unmatched segments
# is normal - especially in Whisper translate mode, where the generated
# English tokens don't align 1:1 to the source audio - so those keep exact
# word timing for the matched majority and get estimated timing instead
# of failing the whole batch.
STRICT_TIMING_MAX_MISSING_RATIO = 0.5

SUPPORTED_EXTS = (
    ".mp4", ".mkv", ".avi", ".mov", ".wmv", ".flv", ".webm",
    ".m4v", ".mpeg", ".mpg", ".ts", ".mts", ".m2ts",
    ".mp3", ".wav", ".m4a", ".aac", ".ogg", ".flac",
    ".wma", ".opus", ".aiff", ".aif",
)

AUDIO_ONLY_EXTS = frozenset({
    ".mp3", ".wav", ".m4a", ".aac", ".flac", ".wma",
    ".opus", ".aiff", ".aif",
})

HALLUCINATION_MARKERS = frozenset({
    "[music]", "*music*", "(music)", "music",
    "[musique]", "[musica]", "[musik]",
    "[applause]", "*applause*", "(applause)",
    "[laughter]", "*laughter*",
    "♪", "♫", "♪♪", "[♪]", "(♪)",
})

REFINE_SEARCH_S = 0.25
REFINE_PRE_PAD_S = 0.0
REFINE_POST_PAD_S = 0.0
REFINE_MIN_DUR_S = 0.001

_ERROR_PREFIXES = (
    "[error:",
    "[whisper error:",
    "[canary error:",
    "[moonshine error:",
)

_ABBREVIATIONS = frozenset({
    "mr.", "mrs.", "ms.", "dr.", "prof.", "sr.", "jr.",
    "st.", "vs.", "etc.", "e.g.", "i.e.", "fig.", "no.",
    "approx.",
})

_SENT_END = re.compile(
    r"[。！？…]+[\"'”’」』）)\]]*"
    r"|[.!?;；]+[\"'”’」』）)\]]*(?=\s|$)"
)

_CUE_TS = re.compile(
    r"(\d+):([0-5]\d):([0-5]\d[.,]\d{1,3})"
    r"\s*-->\s*"
    r"(\d+):([0-5]\d):([0-5]\d[.,]\d{1,3})"
)


class SubtitleTimingError(ValueError):
    pass


def _number(value, default=None):
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError):
        return default
    return result if math.isfinite(result) else default


def _positive(value, name):
    result = _number(value)
    if result is None or result <= 0:
        raise ValueError(f"{name} must be a positive finite number")
    return result


def _safe_call(callback, *args):
    if callback is None:
        return
    try:
        callback(*args)
    except Exception:
        _LOG.exception("Application callback failed")


def _cancelled(event):
    return event is not None and event.is_set()


def _check_cancel(event):
    if _cancelled(event):
        raise InterruptedError("cancelled")


def _read_json(path):
    try:
        with Path(path).open("r", encoding="utf-8") as handle:
            result = json.load(handle)
        return result if isinstance(result, dict) else {}
    except (OSError, ValueError, TypeError):
        return {}


def _atomic_text(path, text):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=str(path.parent),
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        try:
            Path(temporary).unlink(missing_ok=True)
        except OSError:
            _LOG.debug("Temporary file cleanup failed", exc_info=True)


def _write_json(path, data):
    _atomic_text(
        path,
        json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
    )


def _eta_bucket(audio_s) -> str:
    value = _number(audio_s)
    if value is None or value <= 0:
        return "m"
    for name, limit in ETA_BUCKETS:
        if value < limit:
            return name
    return "l"


def _load_eta_stats() -> Dict:
    with _ETA_LOCK:
        return _read_json(ETA_PATH)


def _valid_eta_entry(entry):
    return (
        isinstance(entry, dict)
        and (_number(entry.get("audio"), 0.0) >= 1.0)
        and (_number(entry.get("proc"), 0.0) > 0.0)
    )


def _bucket_entry(stats: Dict, key: str, bucket: Optional[str]):
    node = stats.get(key)
    if not isinstance(node, dict):
        return None, True
    if bucket and _valid_eta_entry(node.get(bucket)):
        return node[bucket], False
    if _valid_eta_entry(node.get("all")):
        return node["all"], True
    if _valid_eta_entry(node):
        return node, True
    return None, True


def _eta_factor(key: str, duration=None) -> float:
    key = str(key or "")
    bucket = _eta_bucket(duration) if duration is not None else None
    stats = _load_eta_stats()

    for candidate in dict.fromkeys((key, key.split(":")[0])):
        entry, _ = _bucket_entry(stats, candidate, bucket)
        if entry is not None:
            return min(
                20.0,
                max(0.01, float(entry["proc"]) / float(entry["audio"])),
            )

    return float(
        ETA_DEFAULTS.get(key, ETA_DEFAULTS.get(key.split(":")[0], 1.0))
    )


def _eta_safe_key(part: str, fallback: str = "") -> str:
    value = str(part or fallback or "").strip().lower()
    value = re.sub(r"\s+", "-", value)
    value = re.sub(r"[^a-z0-9_+-]", "", value)
    return value[:80] or str(fallback or "")


def clear_eta_history(which: str = "all") -> int:
    which = str(which or "all").strip().lower()
    with _ETA_LOCK:
        stats = _read_json(ETA_PATH)
        if which == "all":
            selected = list(stats)
        elif which == "burn":
            selected = [
                key for key in stats
                if key == "burn" or key.startswith("burn:")
            ]
        elif which in {"srt", "transcribe", "transcription"}:
            selected = [
                key for key in stats
                if key != "burn" and not key.startswith("burn:")
            ]
        else:
            selected = [
                key for key in stats
                if key == which or key.startswith(which + ":")
            ]

        if not selected:
            return 0

        for key in selected:
            del stats[key]

        try:
            _write_json(ETA_PATH, stats)
        except OSError:
            _LOG.exception("Could not clear ETA history")
            return 0
        return len(selected)


def record_eta_sample(key: str, audio_s: float, proc_s: float):
    audio = _number(audio_s)
    processing = _number(proc_s)
    if not key or audio is None or processing is None:
        return
    if audio < 1.0 or processing <= 0:
        return

    with _ETA_LOCK:
        stats = _read_json(ETA_PATH)
        node = stats.get(str(key))
        if not isinstance(node, dict):
            node = {}

        if _valid_eta_entry(node) and "all" not in node:
            node = {"all": dict(node)}

        for slot in (_eta_bucket(audio), "all"):
            entry = node.get(slot)
            if not isinstance(entry, dict):
                entry = {}
            count = max(0, int(_number(entry.get("n"), 0)))
            old_audio = max(0.0, _number(entry.get("audio"), 0.0))
            old_proc = max(0.0, _number(entry.get("proc"), 0.0))
            node[slot] = {
                "n": count + 1,
                "audio": old_audio + audio,
                "proc": old_proc + processing,
            }

        stats[str(key)] = node
        try:
            _write_json(ETA_PATH, stats)
        except (OSError, ValueError):
            _LOG.exception("Could not save ETA sample")


def format_eta(seconds) -> str:
    value = _number(seconds)
    if value is None:
        return ""
    seconds = max(0, round(value))
    if seconds < 60:
        return f"{seconds}s"
    if seconds < 3600:
        return f"{seconds // 60}:{seconds % 60:02d}"
    return f"{seconds // 3600}h{seconds % 3600 // 60:02d}m"


class ProgressPump:
    def __init__(
        self,
        prog,
        base: float,
        span: float,
        total_s: float,
        label: str = "",
    ):
        self._prog = prog
        self._base = min(1.0, max(0.0, _number(base, 0.0)))
        self._span = min(
            1.0 - self._base,
            max(0.0, _number(span, 0.0)),
        )
        self._total = max(1.0, _number(total_s, 60.0))
        self._label = str(label or "")
        self._stop = threading.Event()
        self._lock = threading.Lock()
        self._thread = None
        self._t0 = None

    def start(self):
        with self._lock:
            if self._thread is not None:
                return self
            self._stop.clear()
            self._t0 = time.monotonic()
            self._thread = threading.Thread(
                target=self._tick,
                name="SubtitleProgress",
                daemon=True,
            )
            self._thread.start()
        return self

    def _tick(self):
        while not self._stop.wait(0.5):
            elapsed = time.monotonic() - self._t0
            fraction = 0.95 * (1.0 - math.exp(-elapsed / self._total))
            _safe_call(
                self._prog,
                self._base + self._span * fraction,
                f"{self._label} ({int(elapsed)}s)".strip(),
            )

    def stop(self):
        with self._lock:
            self._stop.set()
            worker = self._thread
        if worker is not None and worker is not threading.current_thread():
            worker.join()

    def __enter__(self):
        return self.start()

    def __exit__(self, *exc):
        self.stop()
        return False


def _eta_key_for(engine_kind, arch=0) -> str:
    kind = str(engine_kind or "").strip().lower()
    if "whisper" in kind:
        return "whisper"
    if "canary" in kind:
        return "canary"
    if "moonshine" in kind:
        return f"moonshine:{int(_number(arch, 0))}"
    return _eta_safe_key(kind, "job")


class EtaTracker:
    def __init__(self, key: str):
        self.key = key or "job"
        self.t0 = time.monotonic()
        self.factor = _eta_factor(self.key)
        self.dur = None
        self.total = None

    def set_duration(self, seconds):
        duration = _number(seconds)
        self.dur = duration if duration is not None and duration > 0 else None
        if self.dur is None:
            self.total = None
            return
        self.factor = _eta_factor(self.key, self.dur)
        self.total = max(1.0, self.dur * self.factor)

    def elapsed(self) -> float:
        return max(0.0, time.monotonic() - self.t0)

    def remaining(self, progress=None):
        elapsed = self.elapsed()
        progress = _number(progress)
        if self.total is not None and elapsed < self.total:
            return self.total - elapsed
        if progress is not None and 0.05 < progress < 1.0:
            return max(0.0, elapsed * (1.0 - progress) / progress)
        return None

    def suffix(self, progress) -> str:
        progress = _number(progress, 0.0)
        if progress >= 1.0:
            return ""
        remaining = self.remaining(progress)
        return "" if remaining is None else f" · ETA {format_eta(remaining)}"


def _display_units(text):
    units = []
    for char in unicodedata.normalize("NFC", str(text)):
        if units and (
            unicodedata.combining(char)
            or unicodedata.category(char) in {"Mn", "Me"}
            or char in {"\u200d", "\ufe0e", "\ufe0f"}
            or units[-1].endswith("\u200d")
        ):
            units[-1] += char
        else:
            units.append(char)
    return units


def _unit_width(unit):
    widths = [
        2 if unicodedata.east_asian_width(char) in {"W", "F"} else 1
        for char in unit
        if not unicodedata.combining(char)
        and unicodedata.category(char) not in {"Mn", "Me", "Cf"}
    ]
    if "\u200d" in unit:
        return max(widths, default=0)
    return sum(widths)


def _dlen(s: str) -> int:
    return sum(_unit_width(unit) for unit in _display_units(s))


def _width_cut(s: str, limit: int) -> Tuple[str, str]:
    limit = int(limit)
    if limit <= 0:
        raise ValueError("Width limit must be positive")
    units = _display_units(s)
    width = 0
    for index, unit in enumerate(units):
        candidate = width + _unit_width(unit)
        if candidate > limit:
            return "".join(units[:index]), "".join(units[index:])
        width = candidate
    return "".join(units), ""


def is_audio_only_path(path) -> bool:
    try:
        return Path(path).suffix.lower() in AUDIO_ONLY_EXTS
    except (TypeError, ValueError):
        return False


def cpu_count() -> int:
    return max(1, os.cpu_count() or 1)


def default_workers() -> int:
    return max(1, round(cpu_count() * 0.8))


def configure_cpu(n: int):
    n = min(cpu_count(), max(1, int(_number(n, default_workers()))))
    for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ[key] = str(n)
    try:
        import torch

        torch.set_num_threads(n)
        try:
            torch.set_num_interop_threads(min(n, 4))
        except RuntimeError:
            pass
    except ImportError:
        pass
    return n


def _is_executable(path):
    path = Path(path)
    return path.is_file() and (
        os.name == "nt" or os.access(path, os.X_OK)
    )


def get_ffmpeg_exe() -> Optional[str]:
    name = "ffmpeg.exe" if os.name == "nt" else "ffmpeg"
    candidates = (
        PORTABLE_ROOT / "ffmpeg" / "bin" / name,
        PORTABLE_ROOT / "ffmpeg" / name,
        PORTABLE_ROOT / name,
    )
    for candidate in candidates:
        if _is_executable(candidate):
            return str(candidate)

    found = shutil.which("ffmpeg")
    if found:
        return found

    try:
        import imageio_ffmpeg

        candidate = imageio_ffmpeg.get_ffmpeg_exe()
        if candidate and _is_executable(candidate):
            return str(candidate)
    except (ImportError, OSError, RuntimeError):
        pass
    return None


def _get_ffprobe_exe(ffmpeg):
    name = "ffprobe.exe" if os.name == "nt" else "ffprobe"
    candidate = Path(ffmpeg).resolve().parent / name
    if _is_executable(candidate):
        return str(candidate)
    return shutil.which("ffprobe")


def format_ts(seconds: float) -> str:
    value = _number(seconds)
    if value is None:
        raise ValueError("Subtitle timestamp must be finite")
    milliseconds = max(0, int(math.floor(value * 1000.0 + 0.5)))
    hours, remainder = divmod(milliseconds, 3600000)
    minutes, remainder = divmod(remainder, 60000)
    seconds, milliseconds = divmod(remainder, 1000)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d},{milliseconds:03d}"


def _run_process(command, cancel_event=None, timeout=None):
    _check_cancel(cancel_event)
    start = time.monotonic()
    with tempfile.TemporaryFile(mode="w+b") as errors:
        try:
            process = subprocess.Popen(
                [str(part) for part in command],
                stdin=subprocess.DEVNULL,
                stdout=subprocess.DEVNULL,
                stderr=errors,
            )
        except FileNotFoundError as exc:
            raise RuntimeError(
                f"Executable not found: {command[0]}"
            ) from exc

        try:
            while True:
                _check_cancel(cancel_event)
                if timeout is not None and time.monotonic() - start > timeout:
                    raise TimeoutError("External process timed out")
                try:
                    returncode = process.wait(timeout=0.1)
                    break
                except subprocess.TimeoutExpired:
                    continue
            _check_cancel(cancel_event)
        finally:
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=3)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()

        errors.seek(0, os.SEEK_END)
        errors.seek(max(0, errors.tell() - 8000))
        detail = errors.read().decode("utf-8", errors="replace").strip()

        if returncode:
            raise RuntimeError(
                f"External process failed ({returncode}): {detail}"
            )
        return detail


def _extract_wav(
    src,
    dst_wav,
    ffmpeg,
    cancel_event=None,
    normalize_audio=False,
    start_s=None,
    duration_s=None,
):
    import soundfile as sf

    src = Path(src)
    dst_wav = Path(dst_wav)

    if not src.is_file():
        raise FileNotFoundError(f"Input not found: {src}")
    if src.resolve() == dst_wav.resolve():
        raise ValueError("Audio output must differ from the input")

    dst_wav.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(
        prefix=".subtitle_audio_",
        suffix=".wav",
        dir=str(dst_wav.parent),
    )
    os.close(descriptor)
    temporary = Path(name)

    command = [
        ffmpeg, "-hide_banner", "-nostdin", "-y", "-v", "error",
    ]

    if start_s is None:
        command += ["-copyts", "-start_at_zero"]

    command += ["-i", str(src)]

    if start_s is not None:
        command += ["-ss", f"{start_s:.6f}"]
    if duration_s is not None:
        command += ["-t", f"{duration_s:.6f}"]

    filters = []
    if normalize_audio:
        filters.append("loudnorm=I=-16:TP=-1.5:LRA=11")
    if start_s is None:
        filters.append("aresample=16000:async=1:first_pts=0")
    else:
        filters.append("aresample=16000")

    command += [
        "-map", "0:a:0",
        "-vn", "-sn", "-dn",
        "-af", ",".join(filters),
        "-ac", "1",
        "-ar", "16000",
        "-c:a", "pcm_s16le",
        "-f", "wav",
        str(temporary),
    ]

    try:
        _run_process(command, cancel_event)
        info = sf.info(str(temporary))
        if info.frames <= 0 or info.samplerate != 16000 or info.channels != 1:
            raise RuntimeError("Audio extraction produced invalid or empty audio")
        _check_cancel(cancel_event)
        os.replace(temporary, dst_wav)
    finally:
        temporary.unlink(missing_ok=True)


def extract_audio(
    src: Path,
    dst_wav: Path,
    ffmpeg: str,
    cancel_event=None,
    normalize_audio: bool = False,
):
    _extract_wav(
        src,
        dst_wav,
        ffmpeg,
        cancel_event,
        normalize_audio,
    )


def load_wav_16k(wav_path: Path) -> Tuple[np.ndarray, int]:
    import soundfile as sf

    audio, sample_rate = sf.read(
        str(wav_path),
        dtype="float32",
        always_2d=True,
    )
    if sample_rate <= 0:
        raise ValueError("Invalid audio sample rate")

    audio = audio.mean(axis=1, dtype=np.float32)
    if not np.isfinite(audio).all():
        raise ValueError("Audio contains non-finite samples")

    if sample_rate != 16000 and len(audio):
        try:
            from scipy.signal import resample_poly
        except ImportError as exc:
            raise RuntimeError(
                "Install scipy to resample audio, or extract it as 16 kHz first"
            ) from exc

        divisor = math.gcd(int(sample_rate), 16000)
        audio = resample_poly(
            audio,
            16000 // divisor,
            int(sample_rate) // divisor,
        ).astype(np.float32)

    return np.ascontiguousarray(audio, dtype=np.float32), 16000


def _mono_audio(audio):
    audio = np.asarray(audio, dtype=np.float32)
    if audio.ndim == 2:
        audio = audio.mean(axis=1, dtype=np.float32)
    if audio.ndim != 1:
        raise ValueError("Audio must have one sample axis and optional channels")
    if not np.isfinite(audio).all():
        raise ValueError("Audio contains non-finite samples")
    return audio


def _rms_frames(audio, frame_len):
    if not len(audio):
        return np.zeros(0, dtype=np.float64)
    count = math.ceil(len(audio) / frame_len)
    padded = np.pad(audio, (0, count * frame_len - len(audio)))
    frames = padded.reshape(count, frame_len).astype(np.float64)
    energy = np.sum(frames * frames, axis=1)
    lengths = np.full(count, frame_len, dtype=np.float64)
    lengths[-1] = len(audio) - (count - 1) * frame_len
    return np.sqrt(energy / lengths)


def _activity_from_rms(rms):
    rms = np.asarray(rms, dtype=np.float64)
    if not len(rms):
        return np.zeros(0, dtype=bool)
    noise = float(np.percentile(rms, 10))
    high = float(np.percentile(rms, 90))
    if high < 0.0001:
        return np.zeros(len(rms), dtype=bool)
    threshold = max(0.0001, min(noise * 2.5, high * 0.35))
    return rms > threshold


def vad_segments(
    audio: np.ndarray,
    sr: int = 16000,
    frame_ms: int = 30,
    min_silence_s: float = 0.5,
    max_chunk_s: float = 20.0,
    target_chunk_s: float = 12.0,
) -> List[Tuple[float, float]]:
    audio = _mono_audio(audio)
    sr = int(_positive(sr, "sample rate"))
    frame_ms = _positive(frame_ms, "frame_ms")
    maximum = _positive(max_chunk_s, "max_chunk_s")
    target = min(maximum, _positive(target_chunk_s, "target_chunk_s"))
    silence = _positive(min_silence_s, "min_silence_s")

    if not len(audio):
        return []

    frame_len = max(1, round(sr * frame_ms / 1000.0))
    step = frame_len / sr
    active = _activity_from_rms(_rms_frames(audio, frame_len))
    indices = np.flatnonzero(active)

    if not len(indices):
        return []

    silence_frames = max(1, math.ceil(silence / step))
    regions = []
    begin = previous = int(indices[0])

    for raw_index in indices[1:]:
        index = int(raw_index)
        if index - previous - 1 >= silence_frames:
            regions.append((begin * step, (previous + 1) * step))
            begin = index
        previous = index

    regions.append((begin * step, (previous + 1) * step))
    duration = len(audio) / sr

    padded = []
    for start, end in regions:
        start = max(0.0, start - 0.15)
        end = min(duration, end + 0.15)
        if padded and start <= padded[-1][1]:
            padded[-1] = (padded[-1][0], max(end, padded[-1][1]))
        else:
            padded.append((start, end))

    result = []
    pending = None

    for start, end in padded:
        if pending is not None:
            if end - pending[0] <= target:
                pending = (pending[0], end)
                continue
            result.append(pending)
            pending = None

        while end - start > maximum:
            result.append((start, start + maximum))
            start += maximum

        if end > start:
            pending = (start, end)

    if pending is not None:
        result.append(pending)

    return result


def speech_activity(wav_path, frame_ms: int = 30):
    import soundfile as sf

    frame_ms = _positive(frame_ms, "frame_ms")
    energies = []

    with sf.SoundFile(str(wav_path)) as handle:
        sample_rate = int(handle.samplerate)
        frame_len = max(1, round(sample_rate * frame_ms / 1000.0))
        remainder = np.zeros(0, dtype=np.float32)

        while True:
            block = handle.read(
                sample_rate * 30,
                dtype="float32",
                always_2d=True,
            )
            if not len(block):
                break

            mono = block.mean(axis=1, dtype=np.float32)
            if not np.isfinite(mono).all():
                raise ValueError("Audio contains non-finite samples")

            buffered = np.concatenate((remainder, mono))
            complete = len(buffered) // frame_len
            if complete:
                count = complete * frame_len
                energies.extend(_rms_frames(buffered[:count], frame_len))
                remainder = buffered[count:]
            else:
                remainder = buffered

        if len(remainder):
            energies.extend(_rms_frames(remainder, frame_len))

    return _activity_from_rms(energies), sample_rate


def _clean_segments(segments, duration=None):
    cleaned = []
    limit = _number(duration)

    for item in segments or []:
        try:
            if isinstance(item, dict):
                start, end, text = item["start"], item["end"], item["text"]
            elif hasattr(item, "start") and hasattr(item, "end"):
                start, end, text = item.start, item.end, item.text
            else:
                start, end, text = item

            start = _number(start)
            end = _number(end)
            text = re.sub(r"\s+", " ", str(text or "")).strip()

            if not text:
                continue
            if text.lower().startswith(_ERROR_PREFIXES):
                raise RuntimeError(text)
            if start is None or end is None or end <= start:
                raise SubtitleTimingError(f"Invalid segment timestamps: {item}")

            start = max(0.0, start)
            if limit is not None:
                end = min(limit, end)
            if end <= start:
                raise SubtitleTimingError("Segment lies outside the audio")

            cleaned.append((start, end, text))
        except (KeyError, TypeError) as exc:
            raise SubtitleTimingError("Invalid segment record") from exc

    return sorted(cleaned, key=lambda item: (item[0], item[1]))


def _clean_words(words):
    cleaned = []
    for item in words or []:
        try:
            if isinstance(item, dict):
                start = item["start"]
                end = item["end"]
                text = item.get("word", item.get("text", ""))
            elif hasattr(item, "start") and hasattr(item, "end"):
                start = item.start
                end = item.end
                text = getattr(item, "word", getattr(item, "text", ""))
            else:
                start, end, text = item

            start = _number(start)
            end = _number(end)
            text = str(text or "")

            if not text.strip():
                continue
            if start is None or end is None or start < 0 or end <= start:
                raise SubtitleTimingError(f"Invalid word timestamps: {item}")

            cleaned.append((start, end, text))
        except (KeyError, TypeError) as exc:
            raise SubtitleTimingError("Invalid word timestamp record") from exc

    previous_start = -1.0
    previous_end = -1.0
    for start, end, _ in cleaned:
        if start < previous_start or end < previous_end:
            raise SubtitleTimingError("Word timestamps are not in spoken order")
        previous_start, previous_end = start, end

    return cleaned


def _canonical(text):
    normalized = unicodedata.normalize("NFKC", str(text)).casefold()
    return "".join(
        char for char in normalized
        if not char.isspace()
        and unicodedata.category(char)[0] not in {"P", "Z", "C"}
    )


def _canonical_offsets(text):
    normalized = unicodedata.normalize("NFC", text)
    chars = []
    offsets = []
    for index, char in enumerate(normalized):
        for value in _canonical(char):
            chars.append(value)
            offsets.append(index)
    return normalized, "".join(chars), offsets


def select_seg_words(s: float, e: float, words) -> list:
    start = _number(s)
    end = _number(e)
    if start is None or end is None or end <= start:
        return []
    return [
        word for word in _clean_words(words)
        if start <= (word[0] + word[1]) / 2.0 < end
    ]


def _assign_words_to_segments(segments, words):
    words = _clean_words(words)
    if not words:
        return [[] for _ in segments]

    keys = [_canonical(text) for _, _, text in words]
    joined = "".join(keys)
    starts = []
    ends = []
    offset = 0

    for key in keys:
        starts.append(offset)
        offset += len(key)
        ends.append(offset)

    cursor = 0
    assigned = []

    for start, end, text in segments:
        target = _canonical(text)
        if not target:
            assigned.append([])
            continue

        position = joined.find(target, cursor)
        selected = None

        while position >= 0:
            finish = position + len(target)
            left = bisect.bisect_right(ends, position)
            right = bisect.bisect_left(starts, finish)
            candidates = words[left:right]

            if candidates:
                exact_edges = starts[left] == position and ends[right - 1] == finish
                near = (
                    candidates[0][0] < end + 0.5
                    and candidates[-1][1] > start - 0.5
                )
                if exact_edges and near:
                    selected = candidates
                    cursor = finish
                    break

            position = joined.find(target, position + 1)

        assigned.append(selected or [])

    return assigned


def refine_segments_with_vad(
    segments: List[Tuple[float, float, str]],
    activity,
    frame_ms: int = 30,
    words=None,
):
    segments = _clean_segments(segments)
    stats = {"snapped": 0, "dropped": 0, "markers": 0, "anchored": 0}
    frame_seconds = _positive(frame_ms, "frame_ms") / 1000.0
    activity = (
        np.asarray(activity, dtype=bool).reshape(-1)
        if activity is not None
        else np.zeros(0, dtype=bool)
    )
    assigned = _assign_words_to_segments(segments, words) if words else None
    result = []

    for index, (start, end, text) in enumerate(segments):
        selected = assigned[index] if assigned else []
        if selected:
            new_start, new_end = selected[0][0], selected[-1][1]
            stats["anchored"] += 1
        elif activity.any():
            lo = max(0, int(math.floor(start / frame_seconds)))
            hi = min(len(activity), int(math.ceil(end / frame_seconds)))
            active = np.flatnonzero(activity[lo:hi])
            if len(active):
                first = (lo + int(active[0])) * frame_seconds
                last = (lo + int(active[-1]) + 1) * frame_seconds
                new_start = first if abs(first - start) <= REFINE_SEARCH_S else start
                new_end = last if abs(last - end) <= REFINE_SEARCH_S else end
            else:
                new_start, new_end = start, end
        else:
            new_start, new_end = start, end

        if new_end <= new_start:
            new_start, new_end = start, end

        if abs(new_start - start) > 0.001 or abs(new_end - end) > 0.001:
            stats["snapped"] += 1

        result.append((new_start, new_end, text))

    return result, stats


def _abbreviation_at(text):
    tail = text.rstrip("\"'”’」』）)] ").split()
    if not tail:
        return False
    token = tail[-1]
    return (
        token.lower() in _ABBREVIATIONS
        or bool(re.fullmatch(r"[A-Z]\.", token))
        or bool(re.fullmatch(r"(?:[A-Za-z]\.){2,}", token))
    )


def split_sentences(text: str) -> List[str]:
    text = re.sub(r"\s+", " ", str(text or "")).strip()
    if not text:
        return []

    result = []
    start = 0

    for match in _SENT_END.finditer(text):
        candidate = text[start:match.end()].strip()
        if _abbreviation_at(candidate):
            continue
        if candidate:
            result.append(candidate)
        start = match.end()

    tail = text[start:].strip()
    if tail:
        result.append(tail)
    return result or [text]


def _wrap_lines(text: str) -> List[str]:
    text = re.sub(r"\s+", " ", str(text or "")).strip()
    if not text:
        return []

    lines = []
    remaining = text

    while remaining:
        if _dlen(remaining) <= MAX_LINE_CHARS:
            lines.append(remaining)
            break

        fit, rest = _width_cut(remaining, MAX_LINE_CHARS)
        if not fit:
            raise ValueError("A display unit exceeds the configured line width")

        split_at = fit.rfind(" ")
        if split_at > 0:
            rest = fit[split_at + 1:] + rest
            fit = fit[:split_at]

        lines.append(fit.rstrip())
        remaining = rest.lstrip()

    return lines


def _ns_spans(strs) -> List[Tuple[int, int]]:
    result = []
    position = 0
    for text in strs:
        length = len(_canonical(text))
        result.append((position, position + length))
        position += length
    return result


def map_sentences_to_words(sents: List[str], words) -> list:
    words = _clean_words(words)
    if not sents or not words:
        return [None] * len(sents)

    if "".join(_canonical(text) for text in sents) != "".join(
        _canonical(text) for _, _, text in words
    ):
        return [None] * len(sents)

    sentence_spans = _ns_spans(sents)
    word_spans = _ns_spans([text for _, _, text in words])
    result = []

    for start, end in sentence_spans:
        selected = [
            word
            for word, (word_start, word_end) in zip(words, word_spans)
            if word_start < end and word_end > start
        ]
        result.append(
            (selected[0][0], selected[-1][1]) if selected else None
        )

    return result


def _timed_text_units(text, words):
    words = _clean_words(words)
    text, target, offsets = _canonical_offsets(text)
    lexical = []
    pending_start = None

    for start, end, word in words:
        key = _canonical(word)
        if not key:
            if lexical:
                lexical[-1][1] = max(lexical[-1][1], end)
            else:
                pending_start = start if pending_start is None else pending_start
            continue
        lexical.append([
            min(start, pending_start) if pending_start is not None else start,
            end,
            key,
        ])
        pending_start = None

    if not target or "".join(item[2] for item in lexical) != target:
        raise SubtitleTimingError(
            "Word timestamp text does not match the segment text"
        )

    units = []
    consumed = 0
    text_position = 0

    for index, (start, end, key) in enumerate(lexical):
        consumed += len(key)
        boundary = len(text) if index == len(lexical) - 1 else offsets[consumed]
        piece = text[text_position:boundary]
        units.append((start, end, piece))
        text_position = boundary

    return units


def _approximate_units(text, start, end):
    text = re.sub(r"\s+", " ", str(text or "")).strip()
    if not text:
        return []

    start = _number(start)
    end = _number(end)
    if start is None or end is None or end <= start:
        raise SubtitleTimingError("Cannot estimate timing for an invalid segment")

    pieces = []
    for token in re.findall(r"\S+\s*", text):
        if _has_cjk(token):
            units = _display_units(token)
            for unit in units:
                if unit.isspace() and pieces:
                    pieces[-1] += unit
                else:
                    pieces.append(unit)
        else:
            pieces.append(token)

    weights = [max(1, _dlen(piece.strip())) for piece in pieces]
    total_weight = sum(weights)
    duration = end - start
    consumed = 0
    result = []

    for index, (piece, weight) in enumerate(zip(pieces, weights)):
        unit_start = start + duration * consumed / total_weight
        consumed += weight
        unit_end = (
            end
            if index == len(pieces) - 1
            else start + duration * consumed / total_weight
        )
        result.append((unit_start, unit_end, piece))

    return result


def _sentence_boundary(text):
    text = text.rstrip()
    if not text or _abbreviation_at(text):
        return False
    return bool(
        re.search(r"[.!?。！？…]+[\"'”’」』）)\]]*$", text)
    )


def _phrase_boundary(text):
    return bool(
        re.search(r"[,;:，、；：]+[\"'”’」』）)\]]*$", text.rstrip())
    )


def _group_timed_units(units, strict_timing):
    groups = []
    pending = []

    def fits(candidate):
        text = "".join(item[2] for item in candidate).strip()
        lines = _wrap_lines(text)
        duration = candidate[-1][1] - candidate[0][0]
        return (
            len(lines) <= MAX_LINES_PER_CUE
            and duration <= MAX_CUE_SECONDS
        )

    for unit in units:
        start, end, text = unit
        if not text:
            continue

        if pending and start - pending[-1][1] >= WORD_PAUSE_BREAK_S:
            groups.append(pending)
            pending = []

        if pending and not fits(pending + [unit]):
            split_at = None

            for position in range(len(pending) - 1, 0, -1):
                left = pending[:position]
                right = pending[position:] + [unit]

                if (
                    _phrase_boundary(left[-1][2])
                    and left[-1][1] - left[0][0] >= MIN_CUE_SECONDS
                    and fits(right)
                ):
                    split_at = position
                    break

            if split_at is None:
                groups.append(pending)
                pending = []
            else:
                groups.append(pending[:split_at])
                pending = pending[split_at:]

        if not fits([unit]):
            if strict_timing:
                raise SubtitleTimingError(
                    "A single timed token exceeds the subtitle width or "
                    "duration limit. More detailed alignment is required."
                )

            if pending:
                groups.append(pending)
                pending = []

            groups.append([unit])
            continue

        pending.append(unit)

        duration = pending[-1][1] - pending[0][0]
        if _sentence_boundary(text) and duration >= MIN_CUE_SECONDS:
            groups.append(pending)
            pending = []

    if pending:
        groups.append(pending)

    return groups


def _cue_milliseconds(value):
    return int(math.floor(value * 1000.0 + 0.5))


def _finalize_cues(cues, strict_timing=True):
    if not cues:
        return []

    ordered = sorted(cues, key=lambda item: (item[0], item[1]))
    result = []

    for start, end, text in ordered:
        text = "\n".join(
            line.strip()
            for line in str(text).replace("\r\n", "\n").replace("\r", "\n").split("\n")
            if line.strip()
        )
        if not text:
            continue

        start_ms = max(0, _cue_milliseconds(start))
        end_ms = _cue_milliseconds(end)

        if end_ms <= start_ms:
            raise SubtitleTimingError(
                "A subtitle becomes empty after millisecond timestamp rounding"
            )

        if result:
            previous_start, previous_end, previous_text = result[-1]
            gap_ms = max(0, _cue_milliseconds(MIN_GAP_SECONDS))

            if previous_end + gap_ms > start_ms:
                adjusted_end = start_ms - gap_ms

                if adjusted_end <= previous_start:
                    raise SubtitleTimingError(
                        "Subtitle timestamps cannot be made non-overlapping "
                        "without losing a cue. Check word alignment or "
                        "simultaneous speakers."
                    )

                result[-1] = (
                    previous_start,
                    adjusted_end,
                    previous_text,
                )

        if strict_timing:
            lines = text.splitlines()
            if len(lines) > MAX_LINES_PER_CUE:
                raise SubtitleTimingError("Subtitle exceeds the line limit")
            if any(_dlen(line) > MAX_LINE_CHARS for line in lines):
                raise SubtitleTimingError("Subtitle exceeds the line width")

        result.append((start_ms, end_ms, text))

    return [
        (start / 1000.0, end / 1000.0, text)
        for start, end, text in result
    ]


def build_cues(
    segments: List[Tuple[float, float, str]],
    seg_words=None,
    *,
    strict_timing: bool = True,
    start_lead_s: float = CUE_START_LEAD_S,
    end_early_s: float = CUE_END_EARLY_S,
) -> List[Tuple[float, float, str]]:
    lead = _number(start_lead_s)
    early = _number(end_early_s)

    if lead is None or early is None or lead < 0 or early < 0:
        raise ValueError("Subtitle timing offsets must be finite and nonnegative")

    source = list(segments or [])
    supplied_words = list(seg_words) if seg_words is not None else None

    if supplied_words is not None and len(supplied_words) != len(source):
        raise SubtitleTimingError(
            "Segment word lists must have the same length as segments"
        )

    records = []
    for index, segment in enumerate(source):
        cleaned = _clean_segments([segment])
        if not cleaned:
            continue
        words = supplied_words[index] if supplied_words is not None else []
        records.append((cleaned[0], words))

    records.sort(key=lambda item: (item[0][0], item[0][1]))
    cues = []

    for segment_index, ((start, end, text), words) in enumerate(records):
        units = None

        if words:
            try:
                units = _timed_text_units(text, words)
            except SubtitleTimingError:
                if strict_timing:
                    raise

        if not units:
            if strict_timing:
                raise SubtitleTimingError(
                    f"Segment {segment_index + 1} has no matching word "
                    "timestamps. Supply a word aligner or explicitly use "
                    "strict_timing=False for approximate timing."
                )
            units = _approximate_units(text, start, end)

        groups = _group_timed_units(units, strict_timing)

        for group in groups:
            speech_start = group[0][0]
            speech_end = group[-1][1]
            cue_start = max(0.0, speech_start - lead)

            last_word_start = group[-1][0]
            last_word_duration = speech_end - last_word_start
            safe_early = min(early, last_word_duration * 0.25)
            cue_end = speech_end - safe_early

            if cue_end <= cue_start:
                raise SubtitleTimingError("Invalid aligned subtitle duration")

            text = "".join(unit[2] for unit in group).strip()
            lines = _wrap_lines(text)

            if len(lines) > MAX_LINES_PER_CUE:
                raise SubtitleTimingError(
                    "A token cannot fit in one cue. Supply finer token timing "
                    "or increase the configured subtitle width."
                )

            cues.append((cue_start, cue_end, "\n".join(lines)))

    return _finalize_cues(cues, strict_timing)


def write_srt(cues: List[Tuple[float, float, str]], path: Path):
    blocks = []
    previous_end_ms = -1

    for index, cue in enumerate(cues, 1):
        if len(cue) != 3:
            raise ValueError("Each subtitle cue must contain start, end, and text")

        start, end, text = cue
        start = _number(start)
        end = _number(end)

        if start is None or end is None or start < 0:
            raise SubtitleTimingError(f"Invalid timestamps in cue {index}")

        start_ms = _cue_milliseconds(start)
        end_ms = _cue_milliseconds(end)

        if end_ms <= start_ms:
            raise SubtitleTimingError(f"Cue {index} has no positive duration")
        if start_ms < previous_end_ms:
            raise SubtitleTimingError(f"Cue {index} overlaps the preceding cue")

        text = str(text).replace("\r\n", "\n").replace("\r", "\n")
        text = "\n".join(
            line.strip() for line in text.split("\n") if line.strip()
        )

        if not text or "\x00" in text:
            raise ValueError(f"Invalid subtitle text in cue {index}")

        blocks.append(
            f"{index}\n"
            f"{format_ts(start_ms / 1000.0)} --> "
            f"{format_ts(end_ms / 1000.0)}\n"
            f"{text}\n\n"
        )
        previous_end_ms = end_ms

    _atomic_text(Path(path), "".join(blocks))


def default_out_path(src: Path, out_dir: Optional[str]) -> Path:
    src = Path(src)
    if out_dir and str(out_dir).strip():
        directory = Path(str(out_dir).strip()).expanduser()
        directory.mkdir(parents=True, exist_ok=True)
        return directory / f"{src.stem}.srt"
    return src.with_suffix(".srt")


def _reservation_key(path):
    return str(Path(path).resolve()).casefold()


def _reserve_names(pairs, base_fn) -> dict:
    # Dedupe within the batch only; existing files on disk are overwritten
    # in place (run_srt_job overwrite=True). Checking exists() here would
    # silently rename re-runs to " (2).srt" instead of overwriting.
    seen = set()
    result = {}

    for index, (source, out_dir) in enumerate(pairs or []):
        base = Path(base_fn(Path(source), out_dir))
        candidate = base
        number = 2

        while _reservation_key(candidate) in seen:
            candidate = base.with_name(
                f"{base.stem} ({number}){base.suffix}"
            )
            number += 1

        seen.add(_reservation_key(candidate))
        result[index] = candidate

    return result


def reserve_batch_names(pairs) -> dict:
    return _reserve_names(pairs, default_out_path)


def reserve_burn_names(pairs) -> dict:
    return _reserve_names(pairs, default_burn_path)


def transcribe_chunk_moonshine(
    transcriber,
    audio_f32: np.ndarray,
    sr: int,
    chunk_start: float,
) -> List[Tuple[float, float, str]]:
    audio = _mono_audio(audio_f32)
    sr = int(_positive(sr, "sample rate"))
    offset = _number(chunk_start)

    if offset is None or offset < 0:
        raise ValueError("Chunk start must be finite and nonnegative")
    if not len(audio):
        return []

    transcript = transcriber.transcribe_without_streaming(
        np.clip(audio, -1.0, 1.0).tolist(),
        sample_rate=sr,
    )

    lines = list(getattr(transcript, "lines", None) or [])
    duration = len(audio) / sr
    result = []

    for index, line in enumerate(lines):
        text = str(getattr(line, "text", "") or "").strip()
        if not text:
            continue

        start = _number(getattr(line, "start_time", None))
        length = _number(getattr(line, "duration", None))

        if start is None or start < 0:
            if len(lines) == 1:
                start = 0.0
            else:
                raise SubtitleTimingError(
                    "Moonshine returned a line without a valid start time"
                )

        if length is not None and length > 0:
            end = start + length
        elif index + 1 < len(lines):
            end = _number(getattr(lines[index + 1], "start_time", None))
            if end is None or end <= start:
                raise SubtitleTimingError(
                    "Moonshine returned invalid line boundaries"
                )
        else:
            end = duration

        start = min(duration, start)
        end = min(duration, end)

        if end <= start:
            raise SubtitleTimingError(
                "Moonshine returned a line outside the chunk duration"
            )

        result.append((offset + start, offset + end, text))

    return result


def _engine_text(value):
    if value is None:
        return ""
    if not isinstance(value, str):
        raise TypeError("Transcription engine must return text")
    text = value.strip()
    if text.lower().startswith(_ERROR_PREFIXES):
        raise RuntimeError(text)
    return text


def _engine_ready(engine):
    ready = getattr(engine, "is_ready", False)
    return bool(ready() if callable(ready) else ready)


def _validate_file_words(words, duration):
    words = _clean_words(words)
    maximum = duration + 0.05

    for start, end, text in words:
        if start >= duration or end > maximum:
            raise SubtitleTimingError(
                f"Word timestamp lies outside the audio: {text!r}"
            )

    return [
        (start, min(end, duration), text)
        for start, end, text in words
    ]


def run_srt_job(
    src_path: str,
    out_dir: str,
    engine_kind: str,
    moonshine_arch: int,
    canary_task: str,
    canary_src: str,
    cpu_workers: int,
    get_moonshine_transcriber: Callable,
    get_canary_engine: Callable,
    progress_cb: Optional[Callable[[float, str], None]] = None,
    log_cb: Optional[Callable[[str], None]] = None,
    cancel_event: Optional[threading.Event] = None,
    whisper_task: str = "translate",
    whisper_src: str = "ja",
    get_whisper_engine: Optional[Callable] = None,
    srt_input_lang: str = "auto",
    srt_output_lang: str = "en",
    out_path: Optional[str] = None,
    normalize_audio: bool = False,
    *,
    strict_timing: bool = True,
    word_aligner: Optional[Callable] = None,
    overwrite: bool = True,
    close_moonshine: bool = True,
) -> str:
    import soundfile as sf

    source = Path(src_path).expanduser()
    if not source.is_file():
        raise FileNotFoundError(f"Input not found: {source}")
    if source.suffix.lower() not in SUPPORTED_EXTS:
        raise ValueError(f"Unsupported input extension: {source.suffix}")

    destination = (
        Path(out_path).expanduser()
        if out_path
        else default_out_path(source, out_dir)
    )

    if destination.resolve() == source.resolve():
        raise ValueError("Subtitle output must not overwrite the source media")
    if destination.suffix.lower() != ".srt":
        raise ValueError("Subtitle output filename must end in .srt")
    if destination.exists() and not overwrite:
        raise FileExistsError(f"Subtitle output already exists: {destination}")

    destination.parent.mkdir(parents=True, exist_ok=True)

    kind = str(engine_kind or "").strip().lower()
    if "whisper" in kind:
        family = "whisper"
    elif "canary" in kind:
        family = "canary"
    elif "moonshine" in kind:
        family = "moonshine"
    else:
        raise ValueError(f"Unknown transcription engine: {engine_kind}")

    input_language = str(srt_input_lang or "auto").strip().lower()
    output_language = str(srt_output_lang or "en").strip().lower()

    if output_language == "auto":
        output_language = input_language

    eta = EtaTracker(_eta_key_for(engine_kind, moonshine_arch))
    progress_lock = threading.RLock()
    last_progress = 0.0

    def prog(fraction, message):
        nonlocal last_progress
        fraction = min(1.0, max(0.0, _number(fraction, 0.0)))
        with progress_lock:
            last_progress = max(last_progress, fraction)
            _safe_call(
                progress_cb,
                last_progress,
                str(message) + eta.suffix(last_progress),
            )

    def log(message):
        _safe_call(log_cb, str(message))

    def wait_for_model(engine, name):
        if engine is None:
            raise RuntimeError(f"{name} engine unavailable")
        if _engine_ready(engine):
            return

        _check_cancel(cancel_event)
        prog(0.12, f"Loading {name}...")
        log(f"Loading {name}...")
        load = getattr(engine, "load", None)
        if not callable(load):
            raise RuntimeError(f"{name} does not expose a model loader")

        started = time.monotonic()
        next_report = 30.0

        with ProgressPump(prog, 0.12, 0.06, 300.0, f"Loading {name}"):
            load()

            while not _engine_ready(engine):
                _check_cancel(cancel_event)
                elapsed = time.monotonic() - started

                if elapsed >= 900.0:
                    raise TimeoutError(f"{name} model loading timed out")

                loading = getattr(engine, "_loading", None)
                if loading is not None and not bool(loading):
                    error = getattr(engine, "_last_error", None)
                    raise RuntimeError(
                        f"{name} failed to load: {error or 'unknown error'}"
                    )

                if elapsed >= next_report:
                    log(f"Still loading {name}: {int(elapsed)}s")
                    next_report += 30.0

                if cancel_event is not None:
                    cancel_event.wait(0.2)
                else:
                    time.sleep(0.2)

        _check_cancel(cancel_event)

    workers = configure_cpu(cpu_workers)
    log(f"Audio preparation CPU threads: {workers}")
    log(f"Language: input={input_language}, output={output_language}")
    log(f"Output: {destination}")

    ffmpeg = get_ffmpeg_exe()
    if not ffmpeg:
        raise RuntimeError(
            "ffmpeg was not found. Install ffmpeg or imageio-ffmpeg."
        )

    _check_cancel(cancel_event)
    segments = []
    all_words = []

    with tempfile.TemporaryDirectory(prefix="srt_job_") as directory:
        work_dir = Path(directory)
        wav_path = work_dir / "audio_16k.wav"

        prog(0.02, "Extracting 16 kHz mono audio...")
        extract_audio(
            source,
            wav_path,
            ffmpeg,
            cancel_event,
            normalize_audio,
        )

        info = sf.info(str(wav_path))
        duration = info.frames / info.samplerate
        if duration <= 0:
            raise RuntimeError("The extracted audio is empty")

        eta.set_duration(duration)
        log(f"Audio duration: {duration:.3f}s")
        prog(0.10, f"Audio ready ({duration:.1f}s)")

        audio = None
        sample_rate = 16000
        spans = None

        def ensure_chunks():
            nonlocal audio, sample_rate, spans
            if spans is not None:
                return

            _check_cancel(cancel_event)
            audio, sample_rate = load_wav_16k(wav_path)
            spans = vad_segments(audio, sample_rate)

            if not spans:
                raise RuntimeError("No audible activity detected")

            log(
                f"Audio chunks: {len(spans)}. "
                "Energy detection is used for chunking, not word alignment."
            )

        if family == "whisper":
            if not callable(get_whisper_engine):
                raise RuntimeError("Whisper engine is not configured")

            task = str(whisper_task or "transcribe").strip().lower()
            if task not in {"transcribe", "translate"}:
                raise ValueError(f"Unsupported Whisper task: {task}")

            if task == "translate":
                if output_language not in {"en", "english"}:
                    raise ValueError(
                        "Whisper translation produces English. "
                        "Another output language requires a translation stage."
                    )
            elif (
                input_language != "auto"
                and output_language not in {input_language, "same", "source"}
            ):
                raise ValueError(
                    "Whisper transcription preserves the source language. "
                    "Use translation or select the source output language."
                )

            engine = get_whisper_engine()
            wait_for_model(engine, "Whisper")

            eta.key = "whisper:" + _eta_safe_key(
                getattr(engine, "model_id", ""),
                "large-v3",
            )
            eta.set_duration(duration)

            device = getattr(engine, "device_info", "unknown")
            model = getattr(engine, "model_id", "unknown")
            log(f"Whisper model={model}, device={device}, task={task}")

            word_method = getattr(engine, "transcribe_file_words", None)
            segment_method = getattr(engine, "transcribe_file_segments", None)
            text_method = getattr(engine, "transcribe_file", None)
            native_error = None

            with ProgressPump(
                prog,
                0.20,
                0.64,
                max(5.0, duration * eta.factor),
                "Whisper transcribing",
            ):
                try:
                    _check_cancel(cancel_event)
                    if callable(word_method):
                        native, words = word_method(
                            str(wav_path),
                            task=task,
                            source_lang=input_language,
                        )
                        segments = _clean_segments(native, duration)
                        try:
                            all_words = _validate_file_words(words, duration)
                        except SubtitleTimingError as exc:
                            log(f"Native word timestamps rejected: {exc}")
                            all_words = []
                    elif callable(segment_method):
                        native = segment_method(
                            str(wav_path),
                            task=task,
                            source_lang=input_language,
                        )
                        segments = _clean_segments(native, duration)
                    elif not callable(text_method):
                        raise RuntimeError(
                            "Whisper engine exposes no supported transcription API"
                        )
                except InterruptedError:
                    raise
                except Exception as exc:
                    native_error = exc
                    segments = []
                    all_words = []
                    log(f"Full-file transcription failed: {exc}")

            _check_cancel(cancel_event)

            if not segments:
                ensure_chunks()
                if not any(
                    callable(method)
                    for method in (word_method, segment_method, text_method)
                ):
                    raise RuntimeError(
                        "Whisper has no usable transcription method"
                    ) from native_error

                log("Using chunk transcription")

                for index, (start, end) in enumerate(spans):
                    _check_cancel(cancel_event)
                    chunk_path = work_dir / f"whisper_{index:06d}.wav"
                    first = round(start * sample_rate)
                    last = min(len(audio), round(end * sample_rate))
                    actual_start = first / sample_rate
                    chunk_duration = (last - first) / sample_rate

                    sf.write(
                        str(chunk_path),
                        audio[first:last],
                        sample_rate,
                        subtype="PCM_16",
                    )

                    local_segments = []
                    local_words = []

                    try:
                        if callable(word_method):
                            native, words = word_method(
                                str(chunk_path),
                                task=task,
                                source_lang=input_language,
                            )
                            local_segments = _clean_segments(
                                native, chunk_duration
                            )
                            try:
                                local_words = _validate_file_words(
                                    words, chunk_duration
                                )
                            except SubtitleTimingError as exc:
                                log(
                                    f"Chunk {index + 1} word timing rejected: {exc}"
                                )
                        elif callable(segment_method):
                            native = segment_method(
                                str(chunk_path),
                                task=task,
                                source_lang=input_language,
                            )
                            local_segments = _clean_segments(
                                native, chunk_duration
                            )
                        else:
                            text = _engine_text(
                                text_method(
                                    str(chunk_path),
                                    task=task,
                                    source_lang=input_language,
                                )
                            )
                            if text:
                                local_segments = [
                                    (0.0, chunk_duration, text)
                                ]

                        segments.extend(
                            (
                                actual_start + local_start,
                                actual_start + local_end,
                                text,
                            )
                            for local_start, local_end, text in local_segments
                        )
                        all_words.extend(
                            (
                                actual_start + local_start,
                                actual_start + local_end,
                                text,
                            )
                            for local_start, local_end, text in local_words
                        )
                    finally:
                        chunk_path.unlink(missing_ok=True)

                    prog(
                        0.20 + 0.64 * (index + 1) / len(spans),
                        f"Whisper chunk {index + 1}/{len(spans)} complete",
                    )

        elif family == "canary":
            if not callable(get_canary_engine):
                raise RuntimeError("Canary engine is not configured")

            task = str(canary_task or "transcribe").strip().lower()
            if task not in {"transcribe", "translate"}:
                raise ValueError(f"Unsupported Canary task: {task}")

            source_language = (
                input_language
                if input_language != "auto"
                else str(canary_src or "auto").strip().lower()
            )
            target_language = (
                source_language
                if task == "transcribe"
                else output_language
            )

            if (
                task == "transcribe"
                and source_language != "auto"
                and output_language not in {
                    source_language, "same", "source"
                }
            ):
                raise ValueError(
                    "Canary transcription preserves the source language. "
                    "Choose translate to change languages."
                )

            if target_language in {"auto", "same", "source"} and task == "translate":
                raise ValueError("Canary translation requires a target language")

            engine = get_canary_engine()
            wait_for_model(engine, "Canary")

            checker = getattr(engine, "_check_lang_support", None)
            supported = getattr(engine, "supported_source_langs", None)

            if callable(checker) and supported:
                error = checker(
                    supported,
                    task,
                    source_language,
                    target_language,
                )
                if error:
                    raise RuntimeError(str(error))

            ensure_chunks()
            log(
                f"Canary task={task}, "
                f"language={source_language}->{target_language}"
            )

            for index, (start, end) in enumerate(spans):
                _check_cancel(cancel_event)
                first = round(start * sample_rate)
                last = min(len(audio), round(end * sample_rate))
                chunk_path = work_dir / f"canary_{index:06d}.wav"

                sf.write(
                    str(chunk_path),
                    audio[first:last],
                    sample_rate,
                    subtype="PCM_16",
                )

                base = 0.20 + 0.64 * index / len(spans)
                span = 0.64 / len(spans)

                try:
                    with ProgressPump(
                        prog,
                        base,
                        span,
                        max(2.0, (end - start) * eta.factor),
                        f"Canary chunk {index + 1}/{len(spans)}",
                    ):
                        text = _engine_text(
                            engine.transcribe_file(
                                str(chunk_path),
                                task=task,
                                source_lang=source_language,
                                target_lang=target_language,
                            )
                        )

                    if text:
                        segments.append((
                            first / sample_rate,
                            last / sample_rate,
                            text,
                        ))
                finally:
                    chunk_path.unlink(missing_ok=True)

                prog(
                    base + span,
                    f"Canary chunk {index + 1}/{len(spans)} complete",
                )

        else:
            if not callable(get_moonshine_transcriber):
                raise RuntimeError("Moonshine transcriber is not configured")

            if (
                input_language != "auto"
                and output_language not in {input_language, "same", "source"}
            ):
                raise ValueError(
                    "This Moonshine interface performs transcription, "
                    "not translation."
                )

            ensure_chunks()
            prog(0.16, "Loading Moonshine...")
            transcriber, architecture_name = get_moonshine_transcriber()
            log(f"Moonshine architecture: {architecture_name}")

            try:
                for index, (start, end) in enumerate(spans):
                    _check_cancel(cancel_event)
                    first = round(start * sample_rate)
                    last = min(len(audio), round(end * sample_rate))

                    segments.extend(
                        transcribe_chunk_moonshine(
                            transcriber,
                            audio[first:last],
                            sample_rate,
                            first / sample_rate,
                        )
                    )

                    prog(
                        0.20 + 0.64 * (index + 1) / len(spans),
                        f"Moonshine chunk {index + 1}/{len(spans)} complete",
                    )
            finally:
                if close_moonshine:
                    close = getattr(transcriber, "close", None)
                    if callable(close):
                        close()

        _check_cancel(cancel_event)
        segments = _clean_segments(segments, duration)

        if not segments:
            raise RuntimeError("No speech was transcribed")

        all_words = _validate_file_words(all_words, duration)
        assigned = _assign_words_to_segments(segments, all_words)
        missing = [
            index
            for index, selected in enumerate(assigned)
            if not selected
        ]

        if missing and word_aligner is not None:
            if not callable(word_aligner):
                raise TypeError("word_aligner must be callable")

            prog(0.86, "Aligning transcription to speech...")
            log(f"Aligning {len(segments)} segments to the audio")

            if family == "whisper" and whisper_task == "translate":
                alignment_language = "en"
            elif family == "canary" and canary_task == "translate":
                alignment_language = output_language
            else:
                alignment_language = input_language

            aligned = word_aligner(
                str(wav_path),
                list(segments),
                alignment_language,
                cancel_event,
            )

            _check_cancel(cancel_event)
            all_words = _validate_file_words(aligned, duration)
            assigned = _assign_words_to_segments(segments, all_words)
            missing = [
                index
                for index, selected in enumerate(assigned)
                if not selected
            ]

        if missing and strict_timing and all_words:
            missing_ratio = len(missing) / max(1, len(segments))
            if missing_ratio > STRICT_TIMING_MAX_MISSING_RATIO:
                raise SubtitleTimingError(
                    f"{len(missing)} of {len(segments)} segments lack "
                    "matching word timestamps. Accurate sentence splitting "
                    "requires word timestamps or an appropriate alignment "
                    "engine. No approximate SRT was written."
                )

        if missing:
            log(
                f"Approximate timing enabled for {len(missing)} of "
                f"{len(segments)} segments. "
                "Their word boundaries are estimated, not measured "
                "(common in translate mode, where generated tokens "
                "don't align 1:1 to the source audio)."
            )
        else:
            log(
                f"Using matched word timestamps for all {len(segments)} segments"
            )

        audio = None
        prog(0.94, "Building timed subtitle cues...")

        # Engines without any word timestamps (Canary/Moonshine, Whisper
        # chunk fallback) can only use approximate timing. Forcing strict
        # here would fail every such job, so relax automatically.
        effective_strict = bool(strict_timing and not missing)

        cues = build_cues(
            segments,
            assigned,
            strict_timing=effective_strict,
        )

        if not cues:
            raise RuntimeError("No subtitle cues were generated")

        for start, end, _ in cues:
            if start < 0 or end > duration + 0.001:
                raise SubtitleTimingError(
                    "Generated subtitle timestamp lies outside the media"
                )

        _check_cancel(cancel_event)

        if overwrite:
            write_srt(cues, destination)
        else:
            descriptor, staging_name = tempfile.mkstemp(
                prefix=f".{destination.stem}_",
                suffix=".srt",
                dir=str(destination.parent),
            )
            os.close(descriptor)
            staging_path = Path(staging_name)

            try:
                write_srt(cues, staging_path)
                _check_cancel(cancel_event)
                try:
                    os.link(staging_path, destination)
                except FileExistsError as exc:
                    raise FileExistsError(
                        f"Subtitle output already exists: {destination}"
                    ) from exc
                except OSError as exc:
                    raise RuntimeError(
                        "The filesystem cannot publish this SRT without "
                        "overwriting. Choose another destination or enable "
                        "overwrite explicitly."
                    ) from exc
            finally:
                staging_path.unlink(missing_ok=True)

        elapsed = eta.elapsed()
        record_eta_sample(eta.key, duration, elapsed)

        log(f"Wrote {len(cues)} subtitle cues to {destination}")
        log(
            f"Elapsed: {format_eta(elapsed)}; "
            f"processing speed: {duration / max(elapsed, 0.001):.2f}x realtime"
        )
        prog(1.0, f"Done: {destination.name} ({len(cues)} cues)")

    return str(destination)


def _batch_short(path: str, limit: int = 40) -> str:
    limit = max(2, int(limit))
    name = Path(path).name or str(path)
    return name if len(name) <= limit else name[:limit - 1] + "…"


def run_srt_batch(
    src_paths,
    run_one: Callable,
    progress_cb: Optional[Callable[[float, str], None]] = None,
    log_cb: Optional[Callable[[str], None]] = None,
    cancel_event: Optional[threading.Event] = None,
    file_cb: Optional[Callable] = None,
):
    paths = [
        str(path)
        for path in (src_paths or [])
        if path is not None and str(path).strip()
    ]
    total = len(paths)
    results = []

    if not total:
        _safe_call(progress_cb, 1.0, "No files to process")
        return results, False

    def emit(kind, path, info):
        _safe_call(file_cb, kind, path, info)

    def skip_remaining(start):
        for index in range(start, total):
            path = paths[index]
            results.append((path, False, "Skipped (cancelled)"))
            emit(
                "skip",
                path,
                {
                    "index": index,
                    "total": total,
                    "reason": "cancelled",
                },
            )

    for index, path in enumerate(paths):
        tag = f"[{index + 1}/{total} {_batch_short(path)}]"

        if _cancelled(cancel_event):
            skip_remaining(index)
            _safe_call(log_cb, f"{tag} batch cancelled")
            return results, True

        emit("start", path, {"index": index, "total": total})
        _safe_call(progress_cb, index / total, f"{tag} starting...")

        state = {"progress": 0.0}
        callback_lock = threading.Lock()

        def progress(fraction, message, file_index=index, label=tag):
            fraction = min(1.0, max(0.0, _number(fraction, 0.0)))
            with callback_lock:
                state["progress"] = max(state["progress"], fraction)
                overall = (file_index + state["progress"]) / total
            _safe_call(progress_cb, overall, f"{label} {message}")

        def log(message, label=tag):
            _safe_call(log_cb, f"{label} {message}")

        try:
            output = run_one(path, progress, log)
            results.append((path, True, output))
            emit(
                "done",
                path,
                {
                    "index": index,
                    "total": total,
                    "ok": True,
                    "out": output,
                },
            )
        except InterruptedError:
            results.append((path, False, "Cancelled by user"))
            emit(
                "done",
                path,
                {
                    "index": index,
                    "total": total,
                    "ok": False,
                    "error": "Cancelled by user",
                },
            )
            skip_remaining(index + 1)
            return results, True
        except Exception as exc:
            error = str(exc) or type(exc).__name__
            results.append((path, False, error))
            emit(
                "done",
                path,
                {
                    "index": index,
                    "total": total,
                    "ok": False,
                    "error": error,
                },
            )
            log(f"FAILED: {error}")
        finally:
            if not _cancelled(cancel_event):
                _safe_call(
                    progress_cb,
                    (index + 1) / total,
                    f"{tag} finished",
                )

    return results, _cancelled(cancel_event)


MP4_OK_AUDIO = {"aac", "ac3", "mp3", "opus"}
BURN_OVERHEAD = 0.015
BURN_SUFFIX = ".burned.mp4"

BURN_SPEEDS = {
    "match": {
        "encoder": "cpu",
        "preset": "medium",
        "passes": 2,
        "label": "Match size (2-pass x264)",
    },
    "fast": {
        "encoder": "cpu",
        "preset": "veryfast",
        "passes": 1,
        "label": "Fast (1-pass x264)",
    },
    "fastest": {
        "encoder": "cpu",
        "preset": "ultrafast",
        "passes": 1,
        "label": "Fastest (ultrafast 1-pass)",
    },
    "nvenc_draft": {
        "encoder": "nvenc",
        "preset": "p1",
        "passes": 1,
        "tune": "hq",
        "label": "Draft (NVENC fast 1-pass)",
    },
    "nvenc_turbo": {
        "encoder": "nvenc",
        "preset": "p1",
        "passes": 1,
        "tune": "ull",
        "label": "Turbo (NVENC ultra-fast 1-pass)",
    },
    "nvenc_balanced": {
        "encoder": "nvenc",
        "preset": "p4",
        "passes": 2,
        "tune": "hq",
        "label": "Balanced (NVENC 2-pass)",
    },
}

BURN_SPEED_LABELS = {
    key: value["label"] for key, value in BURN_SPEEDS.items()
}
BURN_SPEED_IDS = {
    value: key for key, value in BURN_SPEED_LABELS.items()
}
BURN_CODECS = {"h264": "H.264", "hevc": "HEVC (H.265)"}


def default_burn_path(src: Path, out_dir: Optional[str]) -> Path:
    src = Path(src)
    if out_dir and str(out_dir).strip():
        directory = Path(str(out_dir).strip()).expanduser()
        directory.mkdir(parents=True, exist_ok=True)
        return directory / f"{src.stem}{BURN_SUFFIX}"
    return src.parent / f"{src.stem}{BURN_SUFFIX}"


def normalize_burn_codec(codec) -> str:
    value = str(codec or "h264").strip().lower()
    if value in {"hevc", "h265", "h.265", "hevc (h.265)", "x265"}:
        return "hevc"
    if value in {"h264", "h.264", "avc", "x264"}:
        return "h264"
    raise ValueError(f"Unsupported video codec: {codec}")


def _burn_speed_id(speed) -> str:
    value = str(speed or "match").strip()
    if value in BURN_SPEED_IDS:
        return BURN_SPEED_IDS[value]

    normalized = value.lower()
    if normalized in BURN_SPEEDS:
        return normalized

    for label, key in BURN_SPEED_IDS.items():
        if label.lower() == normalized:
            return key

    raise ValueError(f"Unsupported burn speed: {speed}")


def _burn_history_key(speed_id, codec="h264") -> str:
    speed = _burn_speed_id(speed_id)
    return speed + ("+hevc" if normalize_burn_codec(codec) == "hevc" else "")


def _burn_eta_key(speed_id, codec="h264") -> str:
    return "burn:" + _burn_history_key(speed_id, codec)


def resolve_burn_speed(speed, codec="h264") -> dict:
    key = _burn_speed_id(speed)
    resolved = dict(BURN_SPEEDS[key])

    if normalize_burn_codec(codec) == "hevc":
        if resolved["encoder"] == "nvenc":
            resolved["encoder"] = "nvenc_hevc"
        else:
            resolved["encoder"] = "cpu_hevc"
            if resolved["preset"] == "ultrafast":
                resolved["preset"] = "ultrafast"
            resolved["label"] = resolved["label"].replace("x264", "x265")

    return resolved


def record_burn_sample(
    speed_id,
    video_kbps,
    audio_bps,
    duration_s,
    actual_bytes,
    codec="h264",
):
    video = _number(video_kbps)
    audio = _number(audio_bps, 0.0)
    duration = _number(duration_s)
    actual = _number(actual_bytes)

    if (
        video is None
        or duration is None
        or actual is None
        or video <= 0
        or audio < 0
        or duration <= 1.0
        or actual <= 0
    ):
        return

    expected = (video * 1000.0 + audio) * duration / 8.0
    if not math.isfinite(expected) or expected <= 0:
        return

    ratio = actual / expected
    if not 0.2 < ratio < 5.0:
        return

    key = _burn_history_key(speed_id, codec)

    with _BURN_LOCK:
        stats = _read_json(BURN_SIZE_PATH)
        node = stats.get(key)
        if not isinstance(node, dict):
            node = {}

        count = max(0, int(_number(node.get("n"), 0)))
        total = max(0.0, _number(node.get("sum"), 0.0))

        stats[key] = {"n": count + 1, "sum": total + ratio}

        try:
            _write_json(BURN_SIZE_PATH, stats)
        except (OSError, ValueError):
            _LOG.exception("Could not save burn-size sample")


def burn_size_fudge(speed_id, codec="h264"):
    key = _burn_history_key(speed_id, codec)

    with _BURN_LOCK:
        node = _read_json(BURN_SIZE_PATH).get(key)

    if not isinstance(node, dict):
        return None, 0

    count = int(_number(node.get("n"), 0))
    total = _number(node.get("sum"), 0.0)

    if count <= 0 or total <= 0:
        return None, 0

    ratio = total / count
    if not 0.2 < ratio < 5.0:
        return None, 0

    return ratio, count


def clear_burn_size_history() -> int:
    with _BURN_LOCK:
        stats = _read_json(BURN_SIZE_PATH)
        try:
            _write_json(BURN_SIZE_PATH, {})
        except OSError:
            _LOG.exception("Could not clear burn-size history")
            return 0
        return len(stats)


def _burn_entries(entries):
    result = []

    for entry in entries or []:
        if not isinstance(entry, dict):
            continue
        duration = _number(entry.get("duration"))
        if duration is None or duration <= 0:
            continue

        audio = _number(entry.get("audio_bps"), 0.0)
        source_bytes = _number(entry.get("src_bytes"), 0.0)

        result.append({
            "duration": duration,
            "audio_bps": max(0.0, audio),
            "src_bytes": max(0, int(source_bytes)),
        })

    return result


def estimate_burn_batch(
    entries,
    speed_id,
    vbr_auto,
    vbr_kbps,
    codec="h264",
):
    files = _burn_entries(entries)
    empty = {
        "mode": "none",
        "bytes": None,
        "basis": 0,
        "files": len(files),
        "calibrated": False,
    }

    if not files:
        return empty

    if vbr_auto:
        total = sum(entry["src_bytes"] for entry in files)
        return {
            "mode": "auto",
            "bytes": total or None,
            "basis": 0,
            "files": len(files),
            "calibrated": True,
        }

    try:
        video_kbps = max(100, int(vbr_kbps or 0))
    except (TypeError, ValueError):
        return empty

    raw = 0.0
    for entry in files:
        raw += (
            (video_kbps * 1000.0 + entry["audio_bps"])
            / 8.0
            * entry["duration"]
        )

    fudge, basis = burn_size_fudge(speed_id, codec)
    calibrated = fudge is not None and basis >= 1 and fudge > 0
    if not calibrated:
        fudge = 1.0

    return {
        "mode": "manual",
        "bytes": int(raw * fudge),
        "basis": basis,
        "files": len(files),
        "calibrated": calibrated,
    }


def solve_burn_kbps(entries, speed_id, target_mb, codec="h264"):
    target = _number(target_mb)
    if target is None or target <= 0:
        return None

    files = _burn_entries(entries)
    if not files:
        return None

    fudge, _ = burn_size_fudge(speed_id, codec)
    if fudge is None or not fudge > 0:
        fudge = 1.0

    total_duration = sum(entry["duration"] for entry in files)
    audio_bytes = sum(
        entry["audio_bps"] * entry["duration"] for entry in files
    ) / 8.0

    if total_duration <= 0:
        return None

    video_kbps = (
        (target * 1e6 / fudge - audio_bytes) * 8.0 / (1000.0 * total_duration)
    )
    if not math.isfinite(video_kbps) or video_kbps <= 0:
        return None

    return max(1, int(round(video_kbps)))


def probe_media(path, ffmpeg: str) -> dict:
    try:
        proc = subprocess.run(
            [ffmpeg, "-hide_banner", "-i", str(path)],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            text=True,
            errors="replace",
            timeout=60,
        )
    except subprocess.TimeoutExpired as exc:
        raise TimeoutError(f"media probe timed out: {path}") from exc
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
    source = Path(srt_path)
    if not source.is_file():
        raise FileNotFoundError(f"SRT not found: {source}")
    destination_dir = Path(dest_dir)
    destination_dir.mkdir(parents=True, exist_ok=True)
    safe_srt = destination_dir / f"subs_{os.getpid()}_{threading.get_ident()}.srt"
    shutil.copy2(str(source), str(safe_srt))
    if not safe_srt.is_file() or safe_srt.stat().st_size == 0:
        raise IOError(f"failed to stage SRT: {source}")
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
# _CUE_TS is defined once at the top of this module (strict version).
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
