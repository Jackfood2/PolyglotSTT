"""SRT file generation backend - portable, CPU-only, all engines.

Pipeline: video/audio -> ffmpeg (16kHz mono wav) ->
  Moonshine/Canary: energy VAD segments -> per-chunk transcription ->
  Whisper Large v3: single native pass with timestamps (no VAD needed) ->
  Whisper timings snapped to measured speech energy (native boundaries are
  approximate - often early / spanning silence) ->
sentence-aware cue packing -> .srt next to the source file by default.
"""

import os
import re
import shutil
import subprocess
import threading
import unicodedata
from pathlib import Path
from typing import Callable, List, Optional, Tuple

import numpy as np


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
    seconds = max(0.0, float(seconds))
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    ms = int(round((seconds - int(seconds)) * 1000))
    if ms >= 1000:
        s += 1
        ms -= 1000
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def extract_audio(src: Path, dst_wav: Path, ffmpeg: str, cancel_event=None):
    """Extract 16kHz mono wav. Raises on failure. Polls cancel_event (Popen,
    not run(): a 2h video extract would otherwise ignore Cancel for minutes)."""
    cmd = [ffmpeg, "-y", "-v", "error", "-i", str(src),
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
                srt_output_lang: str = "en") -> str:
    """Blocking SRT job. progress_cb(fraction 0..1, message). Returns output .srt path."""
    def prog(f, m):
        if progress_cb:
            try:
                progress_cb(max(0.0, min(1.0, f)), m)
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
    log(f"CPU: {cpu_count()} cores detected, using {workers} threads (80% default, CPU-only)")
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
            if waited % 10 == 0:
                prog(load_pct, f"{loading_msg} ({waited}s)")
        if not eng.is_ready:
            raise RuntimeError(f"{name} timed out loading.")

    ffmpeg = get_ffmpeg_exe()
    if not ffmpeg:
        raise RuntimeError("ffmpeg not found. It ships in wheels\\ (imageio-ffmpeg) - run setup.bat once.")
    log(f"ffmpeg: {ffmpeg}")

    out_path = default_out_path(src, out_dir)
    log(f"Output: {out_path} (default: same folder as source)")
    tmp_wav = src.parent / (src.stem + ".srt_tmp16k.wav")
    if out_path.resolve() == tmp_wav.resolve():
        tmp_wav = tmp_wav.with_name(src.stem + "_audio16k.wav")

    try:
        # Stage 1: extract (0-10%)
        if cancelled():
            raise InterruptedError("cancelled")
        prog(0.02, "Extracting audio (ffmpeg 16kHz mono)...")
        log("Extracting audio...")
        extract_audio(src, tmp_wav, ffmpeg, cancel_event)
        if cancelled():
            raise InterruptedError("cancelled")
        import soundfile as sf
        info = sf.info(str(tmp_wav))
        duration = float(info.frames) / float(info.samplerate or 16000)
        log(f"Audio: {duration:.1f}s, {info.samplerate}Hz")
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
            log(f"Transcribing {len(spans)} chunks with Canary-1B ({canary_task} {eff_src}->{eff_tgt})...")
            import soundfile as _sf
            import tempfile as _tf
            for i, (s, e) in enumerate(spans):
                if cancelled():
                    raise InterruptedError("cancelled")
                # Announce BEFORE the blocking call: a single Canary chunk can
                # take 30s+ on CPU, and without this the bar/log look frozen.
                prog(0.15 + 0.75 * i / len(spans),
                     f"Canary chunk {i + 1}/{len(spans)} ({s:.0f}s, {e - s:.1f}s audio)...")
                log(f"  chunk {i + 1}/{len(spans)}: transcribing {e - s:.1f}s audio...")
                s_i, e_i = int(s * sr), int(e * sr)
                chunk = audio[s_i:e_i]
                with _tf.NamedTemporaryFile(suffix=".wav", delete=False) as tf:
                    tmp_c = tf.name
                got_text = False
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
            _wait_for_model(eng, "Whisper Large v3",
                            "Loading Whisper Large v3 (~3GB, slow first time)...", 0.16)
            log(f"Transcribing full audio with Whisper Large v3 ({whisper_task} {eff_wsrc}->{eff_wtgt}, native timestamps)...")
            prog(0.20, "Whisper transcribing (single pass)...")
            if cancelled():
                raise InterruptedError("cancelled")
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
        return str(out_path)
    finally:
        try:
            if tmp_wav.exists():
                tmp_wav.unlink()
        except Exception:
            pass
