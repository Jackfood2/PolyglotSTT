"""Faster-Whisper large-v3 portable wrapper - offline, same interface as CanaryEngine.

Model: large-v3, official Systran conversion (~3GB, JA->EN capable).
Cache: models_cache/whisper-models/ (portable, travels in zip).
Backend: faster-whisper (ctranlate2, CPU int8) - no torch CUDA needed.

NOTE (2026-09-03): large-v3-turbo was replaced, and NO turbo conversion can
fix it: OpenAI fine-tuned turbo on transcription data only, explicitly
EXCLUDING translation data (openai/whisper#2363), so the <|translate|> token
is untrained - turbo transcribes instead of translating, silently. Verified
empirically here with both mobiuslabsgmbh and deepdml conversions (fluent
Japanese output despite a correct translate prompt; tiny + large-v3
translate the same clip fine). JA->EN needs the full large-v3.
"""

import threading
import numpy as np
import tempfile
import os
from pathlib import Path
from typing import Optional, Callable, List, Tuple

PORTABLE_ROOT = Path(__file__).parent
WHISPER_MODELS_ROOT = PORTABLE_ROOT / "models_cache" / "whisper-models"
HF_CACHE = PORTABLE_ROOT / "models_cache" / "huggingface"

os.environ.setdefault("HF_HOME", str(HF_CACHE))
os.environ.setdefault("HUGGINGFACE_HUB_CACHE", str(HF_CACHE))
os.environ.setdefault("HF_HUB_OFFLINE", "0")

WHISPER_MODEL_ID = "large-v3"

# Cast-off broken conversion (see module docstring): removed on load so the
# portable folder does not carry 1.6GB of dead weights after the update.
STALE_MODEL_DIRS = (
    "models--mobiuslabsgmbh--faster-whisper-large-v3-turbo",
)

# Tasks: transcribe = keep source language, translate = always to English (Whisper limitation)
WHISPER_TASKS = ["transcribe", "translate"]
# Subset of Whisper's ~100 langs - covers translation needs, keeps GUI usable.
# Full whisper list is much longer; add more codes here if needed (they just pass through).
WHISPER_SOURCE_LANGS = [
    "auto", "en", "ja", "zh", "ko",
    "de", "fr", "es", "it", "pt", "nl", "ru",
    "ar", "hi", "tr", "id", "uk", "vi", "th",
]
# Downloadable sizes (faster-whisper Systran aliases; auto-fetched on first
# use when online, then cached offline). NOTE: no turbo here - OpenAI
# trained turbo on transcription data only, so it cannot translate.
WHISPER_MODEL_CHOICES = {
    "Tiny (75MB, fastest)": "tiny",
    "Base (145MB)": "base",
    "Small (500MB)": "small",
    "Medium (1.5GB)": "medium",
    "Large v3 (3GB, best)": "large-v3",
}
WHISPER_MODEL_CHOICES_REV = {v: k for k, v in WHISPER_MODEL_CHOICES.items()}


class WhisperEngine:
    def __init__(self, task: str = "translate", source_lang: str = "ja",
                 target_lang: str = "en", model_id: str = WHISPER_MODEL_ID,
                 on_ready: Optional[Callable] = None):
        self.task = task if task in WHISPER_TASKS else "transcribe"
        self.source_lang = source_lang if source_lang else "auto"
        self.target_lang = "en" if self.task == "translate" else (target_lang or "en")
        self.model_id = model_id or WHISPER_MODEL_ID
        self._model = None
        self._ready = False
        self._on_ready = on_ready
        self._lock = threading.Lock()
        self._infer_lock = threading.Lock()
        self._loading = False
        self._last_error: Optional[str] = None
        self._switch_cb: Optional[Callable] = None  # one-shot switch_model callback

    @property
    def is_ready(self) -> bool:
        return self._ready

    @property
    def current_arch_name(self) -> str:
        return f"whisper-{self.model_id} ({self.task} {self.source_lang}->{self.target_lang})"

    def _ensure_dirs(self):
        for p in [WHISPER_MODELS_ROOT, HF_CACHE]:
            try:
                p.mkdir(parents=True, exist_ok=True)
            except Exception:
                pass
        # One-time cleanup of the broken turbo snapshot (exact dir name only,
        # never anything else).
        try:
            if "mobiuslabsgmbh" not in (self.model_id or ""):
                import shutil
                for stale in STALE_MODEL_DIRS:
                    d = WHISPER_MODELS_ROOT / stale
                    if d.exists():
                        print(f"[Whisper] removing stale broken model cache: {d}")
                        shutil.rmtree(d, ignore_errors=True)
        except Exception:
            pass

    def _cpu_threads(self) -> int:
        for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS"):
            try:
                v = int(os.environ.get(k, "0") or 0)
                if v > 0:
                    return v
            except Exception:
                pass
        try:
            import os as _os
            return max(1, _os.cpu_count() or 4)
        except Exception:
            return 4

    def load(self, on_progress: Optional[Callable] = None):
        with self._lock:
            if self._loading:
                return
            self._loading = True
            self._ready = False

        def _load():
            try:
                self._ensure_dirs()
                from faster_whisper import WhisperModel
                cpu_threads = self._cpu_threads()
                # Offline-first: try cached only, fallback to download when online.
                # (local_files_only=False would hit network for a version check
                # and stall/fail on the offline PC.)
                last_err = None
                for local_only in (True, False):
                    try:
                        model = WhisperModel(
                            self.model_id,
                            device="cpu",
                            compute_type="int8",
                            cpu_threads=cpu_threads,
                            download_root=str(WHISPER_MODELS_ROOT),
                            local_files_only=local_only,
                        )
                        break
                    except Exception as e:
                        last_err = e
                        continue
                else:
                    raise RuntimeError(f"Whisper load failed: {last_err}")
                with self._lock:
                    self._model = model
                    self._ready = True
                    self._last_error = None
                    self._loading = False
                    switch_cb, self._switch_cb = self._switch_cb, None
                if self._on_ready:
                    self._on_ready(True, None)
                if switch_cb:
                    try:
                        switch_cb(True, None)
                    except Exception:
                        pass
            except Exception as e:
                import traceback
                traceback.print_exc()
                with self._lock:
                    self._ready = False
                    self._last_error = str(e)
                    self._loading = False
                    switch_cb, self._switch_cb = self._switch_cb, None
                if self._on_ready:
                    self._on_ready(False, str(e))
                if switch_cb:
                    try:
                        switch_cb(False, str(e))
                    except Exception:
                        pass

        threading.Thread(target=_load, daemon=True).start()

    def switch_options(self, task: Optional[str] = None,
                       source_lang: Optional[str] = None,
                       target_lang: Optional[str] = None,
                       on_ready: Optional[Callable] = None):
        """Update task/lang without reloading weights (same model handles all).

        Thread-safe: live transcribe() snapshots options under the same
        lock, so a GUI change can never tear a read mid-inference.
        """
        with self._lock:
            if task in WHISPER_TASKS:
                self.task = task
                # translate always -> en
                if task == "translate":
                    self.target_lang = "en"
                elif target_lang:
                    self.target_lang = target_lang
            elif target_lang and self.task == "transcribe":
                self.target_lang = target_lang
            if source_lang:
                self.source_lang = source_lang
        if on_ready:
            on_ready(True, None)

    def switch_model(self, model_id: str,
                     on_ready: Optional[Callable] = None):
        """Switch downloadable size (tiny/base/small/medium/large-v3).

        Drops the loaded weights and reloads (downloading first time when
        online). In-flight inference sees not-ready and skips cleanly.
        """
        if not model_id:
            if on_ready:
                on_ready(False, "empty model id")
            return
        with self._lock:
            if model_id == self.model_id and self._ready and self._model is not None:
                same_cb = on_ready
                already = True
            else:
                same_cb, already = None, False
                self.model_id = model_id
                self._model = None
                self._ready = False
                self._last_error = None
                self._loading = False
                self._switch_cb = on_ready
        if already:
            try:
                same_cb(True, None)
            except Exception:
                pass
            return
        self.load()

    def _snapshot_opts(self) -> tuple:
        """Atomic (task, source_lang, target_lang) snapshot for one inference call."""
        with self._lock:
            return self.task, self.source_lang, self.target_lang

    def _transcribe_opts(self, task: Optional[str] = None,
                         source_lang: Optional[str] = None) -> dict:
        # Per-call overrides win; otherwise use the atomic snapshot.
        # This lets SRT jobs pass their own langs without mutating the
        # shared live engine (no live/SRT race).
        snap_task, snap_src, _ = self._snapshot_opts()
        eff_task = task if task in WHISPER_TASKS else snap_task
        eff_src = source_lang if source_lang else snap_src
        opts = {
            "beam_size": 5,
            "condition_on_previous_text": False,
            "word_timestamps": False,
        }
        # auto -> None (whisper auto-detects)
        lang = None if (eff_src or "auto") == "auto" else eff_src
        if lang:
            opts["language"] = lang
        opts["task"] = "translate" if eff_task == "translate" else "transcribe"
        return opts

    def _run(self, wav_path: str, with_timestamps: bool = False,
             task: Optional[str] = None,
             source_lang: Optional[str] = None,
             word_timestamps: bool = False):
        with self._lock:
            model = self._model if (self._ready and self._model is not None) else None
        if model is None:
            return [], None
        opts = self._transcribe_opts(task=task, source_lang=source_lang)
        # Word-level alignment costs extra inference but is the only way to
        # place short/one-word cues exactly (segment stamps are coarse).
        opts["word_timestamps"] = bool(word_timestamps)
        with self._infer_lock:
            segments_gen, info = model.transcribe(wav_path, **opts)
            segs = list(segments_gen)
        return segs, info

    @staticmethod
    def _norm_audio(audio_data: np.ndarray) -> np.ndarray:
        arr = np.asarray(audio_data).flatten()
        if arr.dtype == np.int16:
            out = arr.astype(np.float32) / 32768.0
        elif arr.dtype == np.int32:
            out = arr.astype(np.float32) / 2147483648.0
        else:
            out = arr.astype(np.float32)
            try:
                if np.abs(out).max() > 1.5:
                    out = out / 32768.0
            except Exception:
                pass
            out = np.clip(out, -1.0, 1.0)
        return out

    def transcribe(self, audio_data: np.ndarray, sample_rate: int = 16000) -> str:
        with self._lock:
            if not self._ready or self._model is None:
                return ""
        try:
            import soundfile as sf
            audio_float = self._norm_audio(audio_data)
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tf:
                tmp_path = tf.name
            try:
                sf.write(tmp_path, audio_float, samplerate=sample_rate)
                segs, _ = self._run(tmp_path, with_timestamps=False)
                texts = [(s.text or "").strip() for s in segs]
                return " ".join(t for t in texts if t).strip()
            finally:
                try:
                    os.unlink(tmp_path)
                except Exception:
                    pass
        except Exception as e:
            import traceback
            traceback.print_exc()
            return f"[Whisper Error: {e}]"

    def transcribe_file(self, wav_path: str, task: Optional[str] = None,
                        source_lang: Optional[str] = None) -> str:
        """Single-chunk text (used by VAD-chunk SRT path).

        Optional per-call task/source_lang overrides let SRT jobs use
        their own languages without mutating the shared live engine.
        """
        with self._lock:
            if not self._ready or self._model is None:
                return ""
        try:
            segs, _ = self._run(wav_path, with_timestamps=False,
                                task=task, source_lang=source_lang)
            texts = [(s.text or "").strip() for s in segs]
            return " ".join(t for t in texts if t).strip()
        except Exception as e:
            import traceback
            traceback.print_exc()
            return f"[Whisper Error: {e}]"

    def transcribe_file_segments(self, wav_path: str,
                                 task: Optional[str] = None,
                                 source_lang: Optional[str] = None) -> List[Tuple[float, float, str]]:
        """Full-file native segments with timestamps (preferred SRT path - one pass)."""
        with self._lock:
            if not self._ready or self._model is None:
                return []
        segs, _ = self._run(wav_path, with_timestamps=True,
                            task=task, source_lang=source_lang)
        out: List[Tuple[float, float, str]] = []
        for s in segs:
            try:
                txt = (s.text or "").strip()
                if not txt:
                    continue
                out.append((float(s.start), float(s.end), txt))
            except Exception:
                continue
        return out

    def transcribe_file_words(self, wav_path: str,
                              task: Optional[str] = None,
                              source_lang: Optional[str] = None):
        """Native segments PLUS flat word stamps [(w_start, w_end, word)].

        Same single pass (word_timestamps=True adds alignment cost). Words
        may be [] on models/versions without alignment - callers must fall
        back to segment/VAD timing then.
        """
        with self._lock:
            if not self._ready or self._model is None:
                return [], []
        segs, _ = self._run(wav_path, with_timestamps=True,
                            task=task, source_lang=source_lang,
                            word_timestamps=True)
        out_segs: List[Tuple[float, float, str]] = []
        out_words: List[Tuple[float, float, str]] = []
        for s in segs:
            try:
                txt = (s.text or "").strip()
                if txt:
                    out_segs.append((float(s.start), float(s.end), txt))
                for w in (getattr(s, "words", None) or []):
                    try:
                        wt = (getattr(w, "word", "") or "").strip()
                        ws = float(getattr(w, "start", -1.0))
                        we = float(getattr(w, "end", -1.0))
                        if wt and we > ws >= 0.0:
                            out_words.append((ws, we, wt))
                    except Exception:
                        continue
            except Exception:
                continue
        return out_segs, out_words
