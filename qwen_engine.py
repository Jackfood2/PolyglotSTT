"""Qwen3-ASR engine (GGUF via transcribe.cpp).

Qwen3-ASR is a 30-language offline ASR model (Apache-2.0). It auto-detects the
spoken language and emits the transcript in that language. It is
transcription-only: there is no translation task and no timestamp output, so
SRT generation runs it over VAD chunks (same as Canary) and derives cue timing
from the chunk boundaries.

The heavy lifting is done by the `transcribe-cpp` Python package, which wraps
the transcribe.cpp/ggml runtime (CPU + Vulkan backends). Models are single
`.gguf` files kept in models_cache/qwen3-asr/ so the project stays portable and
fully offline once the file is present.
"""

import gc
import os
import threading
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np

PORTABLE_ROOT = Path(__file__).resolve().parent
QWEN_CACHE = PORTABLE_ROOT / "models_cache" / "qwen3-asr"
HF_CACHE = PORTABLE_ROOT / "models_cache" / "huggingface"

os.environ.setdefault("HF_HOME", str(HF_CACHE))
os.environ.setdefault("HUGGINGFACE_HUB_CACHE", str(HF_CACHE))

QWEN_REPO = "handy-computer/Qwen3-ASR-1.7B-gguf"

# Label -> short id used in config files.
QWEN_MODEL_CHOICES = {
    "Qwen3-ASR 1.7B Q8_0 (2.2GB, best)": "qwen3-asr-1.7b-q8_0",
    "Qwen3-ASR 1.7B Q6_K (1.7GB)": "qwen3-asr-1.7b-q6_k",
    "Qwen3-ASR 1.7B Q5_K_M (1.5GB)": "qwen3-asr-1.7b-q5_k_m",
    "Qwen3-ASR 1.7B Q4_K_M (1.3GB, smallest)": "qwen3-asr-1.7b-q4_k_m",
}

QWEN_MODEL_CHOICES_REV = {v: k for k, v in QWEN_MODEL_CHOICES.items()}

QWEN_MODEL_FILES = {
    "qwen3-asr-1.7b-q8_0": "Qwen3-ASR-1.7B-Q8_0.gguf",
    "qwen3-asr-1.7b-q6_k": "Qwen3-ASR-1.7B-Q6_K.gguf",
    "qwen3-asr-1.7b-q5_k_m": "Qwen3-ASR-1.7B-Q5_K_M.gguf",
    "qwen3-asr-1.7b-q4_k_m": "Qwen3-ASR-1.7B-Q4_K_M.gguf",
}

QWEN_DEFAULT_MODEL = "qwen3-asr-1.7b-q8_0"

# Resident RAM estimates (weights + encoder/LM scratch), in MB.
QWEN_RESIDENT_MB = {
    "qwen3-asr-1.7b-q8_0": 2800,
    "qwen3-asr-1.7b-q6_k": 2300,
    "qwen3-asr-1.7b-q5_k_m": 2150,
    "qwen3-asr-1.7b-q4_k_m": 1950,
}

# Qwen3-ASR speaks 30 languages; the model reports the exact list at load time.
QWEN_LANGUAGES = [
    "auto", "en", "zh", "yue", "ja", "ko", "de", "fr", "es", "pt",
    "it", "ru", "ar", "hi", "th", "vi", "tr", "id", "ms", "nl",
    "sv", "da", "fi", "pl", "cs", "fil", "fa", "el", "ro", "hu", "mk",
]

QWEN_TASKS = ["transcribe"]

_LOG_PREFIX = "[qwen]"


def _log(message: str) -> None:
    try:
        print(f"{_LOG_PREFIX} {message}")
    except Exception:
        pass


def qwen_model_id(model_id: Optional[str]) -> str:
    mid = str(model_id or "").strip()
    return mid if mid in QWEN_MODEL_FILES else QWEN_DEFAULT_MODEL


def qwen_model_filename(model_id: Optional[str]) -> str:
    return QWEN_MODEL_FILES[qwen_model_id(model_id)]


def qwen_model_path(model_id: Optional[str] = None) -> Path:
    return QWEN_CACHE / qwen_model_filename(model_id)


def qwen_downloaded_map() -> Dict[str, bool]:
    out = {}
    for mid, filename in QWEN_MODEL_FILES.items():
        try:
            path = QWEN_CACHE / filename
            out[mid] = path.is_file() and path.stat().st_size > 100_000_000
        except Exception:
            out[mid] = False
    return out


def qwen_cache_info() -> Dict[str, int]:
    out = {}
    for mid, filename in QWEN_MODEL_FILES.items():
        try:
            path = QWEN_CACHE / filename
            out[mid] = path.stat().st_size if path.is_file() else 0
        except Exception:
            out[mid] = 0
    return out


def delete_qwen_model(model_id: Optional[str] = None) -> int:
    path = qwen_model_path(model_id)
    if not path.is_file():
        raise FileNotFoundError(f"not downloaded: {path.name}")
    size = path.stat().st_size
    path.unlink()
    return size


def _resident_model_mb(model_id: Optional[str]) -> int:
    return int(QWEN_RESIDENT_MB.get(qwen_model_id(model_id), 2800))


def _to_16k_mono(audio_data: np.ndarray, sample_rate: int) -> np.ndarray:
    """Float32 mono PCM at 16 kHz (the only rate transcribe.cpp accepts)."""
    raw = np.asarray(audio_data)
    if raw.size == 0:
        return np.empty(0, dtype=np.float32)

    if raw.ndim == 2:
        arr = raw.mean(axis=1)
    else:
        arr = raw.flatten()

    if raw.dtype == np.int16:
        audio = arr.astype(np.float32) / 32768.0
    elif raw.dtype == np.int32:
        audio = arr.astype(np.float32) / 2147483648.0
    else:
        audio = arr.astype(np.float32)

    if audio.size and np.abs(audio).max() > 1.5:
        audio = audio / 32768.0

    np.nan_to_num(audio, copy=False, nan=0.0, posinf=1.0, neginf=-1.0)
    np.clip(audio, -1.0, 1.0, out=audio)

    try:
        rate = int(sample_rate)
    except Exception:
        rate = 16000
    if rate <= 0:
        rate = 16000

    if rate != 16000 and audio.size > 1:
        duration = audio.size / float(rate)
        target = max(1, int(round(duration * 16000)))
        source_index = np.arange(audio.size, dtype=np.float64)
        target_index = np.linspace(0.0, audio.size - 1.0, target)
        audio = np.interp(target_index, source_index, audio).astype(np.float32)

    return np.ascontiguousarray(audio, dtype=np.float32)


class QwenEngine:
    """Drop-in sibling of CanaryEngine/WhisperEngine for Qwen3-ASR.

    Public surface used by the app: load(), is_ready, _loading, _last_error,
    unload(), switch_options(), switch_model(), transcribe(),
    transcribe_file(), device, device_info, current_arch_name,
    supported_source_langs.
    """

    _load_gate = threading.Lock()

    def __init__(
        self,
        task: str = "transcribe",
        source_lang: str = "auto",
        target_lang: str = "en",
        model_id: str = QWEN_DEFAULT_MODEL,
        device: str = "auto",
        on_ready: Optional[Callable] = None,
    ):
        # Qwen3-ASR cannot translate; task is kept for API symmetry only.
        self.task = "transcribe"
        self.source_lang = (
            source_lang if source_lang in QWEN_LANGUAGES else "auto"
        )
        self.target_lang = "en"
        self.model_id = qwen_model_id(model_id)
        self.device = (device or "auto").strip().lower() or "auto"
        if self.device == "gpu":
            self.device = "cuda"
        if self.device not in ("auto", "cpu", "cuda", "vulkan"):
            self.device = "auto"

        self._model = None
        self._session = None
        self._ready = False
        self._loading = False
        self._last_error: Optional[str] = None
        self._load_generation = 0
        self._device_used = "cpu"
        self._backend_used = ""
        self._on_ready = on_ready
        self._lock = threading.Lock()
        self._infer_lock = threading.Lock()
        self._load_lock = threading.Lock()
        self.supported_source_langs = set(QWEN_LANGUAGES) - {"auto"}
        self.supports_translate = False
        self.supports_timestamps = False
        # Optional user dictionary (vocabulary + context prompt).
        self._dict_terms: List[str] = []
        self._dict_context: str = ""
        self._dict_enabled = False

    # ------------------------------------------------------------------ state
    @property
    def is_ready(self) -> bool:
        with self._lock:
            return self._ready and self._model is not None

    @property
    def device_info(self) -> str:
        with self._lock:
            return self._device_used or "cpu"

    @property
    def current_arch_name(self) -> str:
        with self._lock:
            return f"qwen3-asr {self.model_id}"

    # -------------------------------------------------------------- model file
    def _ensure_dirs(self) -> None:
        for path in (QWEN_CACHE, HF_CACHE):
            try:
                path.mkdir(parents=True, exist_ok=True)
            except Exception:
                pass

    def _resolve_model_path(
        self,
        model_id: str,
        generation: int,
        on_progress: Optional[Callable],
    ) -> str:
        self._ensure_dirs()
        filename = qwen_model_filename(model_id)
        local = QWEN_CACHE / filename

        if local.is_file() and local.stat().st_size > 100_000_000:
            return str(local)

        if not self._is_current(generation):
            raise RuntimeError("superseded")

        offline = os.environ.get("HF_HUB_OFFLINE", "").strip().upper() in {
            "1", "TRUE", "YES", "ON",
        }

        if offline:
            raise FileNotFoundError(
                f"{filename} is not in models_cache/qwen3-asr and "
                f"HF_HUB_OFFLINE is set"
            )

        try:
            from huggingface_hub import hf_hub_download
        except Exception as error:
            raise RuntimeError(
                f"huggingface_hub is required to download {filename}: {error}"
            )

        self._report(on_progress, f"Downloading {filename}")
        _log(f"downloading {QWEN_REPO}/{filename}")

        path = hf_hub_download(
            repo_id=QWEN_REPO,
            filename=filename,
            local_dir=str(QWEN_CACHE),
        )

        if not self._is_current(generation):
            raise RuntimeError("superseded")
        return str(path)

    # ------------------------------------------------------------------ load
    def _is_current(self, generation: int) -> bool:
        with self._lock:
            return generation == self._load_generation

    def _report(self, callback: Optional[Callable], message: str) -> None:
        if callback is None:
            return
        try:
            callback(message)
        except Exception:
            pass

    def _backend_choice(self) -> str:
        device = (self.device or "auto").strip().lower()
        if device == "cpu":
            return "cpu"
        if device in ("cuda", "vulkan"):
            return "auto"
        return "auto"

    def load(self, on_progress: Optional[Callable] = None) -> None:
        with self._lock:
            self._load_generation += 1
            generation = self._load_generation
            self._loading = True
            self._ready = False
            self._last_error = None
            model_id = self.model_id

        threading.Thread(
            target=self._load_worker,
            args=(generation, model_id, on_progress),
            name=f"QwenLoader-{generation}",
            daemon=True,
        ).start()

    def _load_worker(
        self,
        generation: int,
        model_id: str,
        on_progress: Optional[Callable],
    ) -> None:
        model = None
        session = None
        installed = False

        try:
            with QwenEngine._load_gate:
                if not self._is_current(generation):
                    return

                with self._infer_lock:
                    with self._lock:
                        if generation != self._load_generation:
                            return
                        old_model = self._model
                        old_session = self._session
                        self._model = None
                        self._session = None
                        self._ready = False
                    self._close_session(old_session)
                    self._close_model(old_model)

                path = self._resolve_model_path(
                    model_id, generation, on_progress
                )

                if not self._is_current(generation):
                    return

                self._report(on_progress, "Loading Qwen3-ASR model")

                import transcribe_cpp

                backend = self._backend_choice()
                model = transcribe_cpp.Model(path, backend=backend)

                if not self._is_current(generation):
                    self._close_model(model)
                    return

                session = model.session()

                backend_name = str(getattr(model, "backend", "") or "cpu")
                caps = getattr(model, "capabilities", None)
                languages = None
                try:
                    languages = set(getattr(caps, "languages", ()) or ())
                except Exception:
                    languages = None

                with self._lock:
                    if generation != self._load_generation:
                        self._close_session(session)
                        self._close_model(model)
                        return
                    self._model = model
                    self._session = session
                    self._ready = True
                    self._loading = False
                    self._last_error = None
                    self._device_used = backend_name
                    self._backend_used = backend_name
                    if languages:
                        self.supported_source_langs = languages | {"auto"}
                    model = None
                    session = None
                    installed = True

                _log(
                    f"loaded {qwen_model_filename(model_id)} on {backend_name}"
                )
                self._report(on_progress, "Model ready")

        except Exception as error:
            with self._lock:
                if generation != self._load_generation:
                    return
                self._ready = False
                self._loading = False
                self._last_error = str(error)
            _log(f"load failed: {error}")
            import traceback

            traceback.print_exc()
        finally:
            if not installed:
                self._close_session(session)
                self._close_model(model)

        ready = self.is_ready
        callback = self._on_ready
        if callback is not None:
            try:
                if ready:
                    callback(True, None)
                else:
                    callback(False, self._last_error)
            except Exception:
                pass

    @staticmethod
    def _close_session(session) -> None:
        if session is None:
            return
        try:
            close = getattr(session, "close", None)
            if callable(close):
                close()
        except Exception:
            pass

    @staticmethod
    def _close_model(model) -> None:
        if model is None:
            return
        try:
            close = getattr(model, "close", None)
            if callable(close):
                close()
        except Exception:
            pass

    def unload(self) -> bool:
        with self._infer_lock:
            with self._lock:
                had = self._model is not None or self._loading
                self._load_generation += 1
                model = self._model
                session = self._session
                self._model = None
                self._session = None
                self._ready = False
                self._loading = False
                self._last_error = None
            self._close_session(session)
            self._close_model(model)

        try:
            gc.collect()
        except Exception:
            pass
        return bool(had)

    def switch_model(
        self,
        model_id: str,
        on_ready: Optional[Callable] = None,
    ) -> None:
        mid = qwen_model_id(model_id)
        with self._lock:
            self.model_id = mid
        if on_ready is not None:
            self._on_ready = on_ready
        self.load()

    def switch_options(
        self,
        task: Optional[str] = None,
        source_lang: Optional[str] = None,
        target_lang: Optional[str] = None,
        on_ready: Optional[Callable] = None,
    ) -> None:
        with self._lock:
            if source_lang:
                self.source_lang = (
                    source_lang if source_lang in QWEN_LANGUAGES else "auto"
                )
            self.task = "transcribe"
            self.target_lang = "en"
        if on_ready is not None:
            try:
                on_ready(True, None)
            except Exception:
                pass

    def _snapshot_opts(self) -> Tuple[str, str, str]:
        with self._lock:
            return self.task, self.source_lang, self.target_lang

    # ------------------------------------------------------------- inference
    def _check_lang_support(
        self,
        source_lang: Optional[str],
    ) -> Optional[str]:
        supported = self.supported_source_langs or set()
        src = (source_lang or "auto").strip().lower()
        if src == "auto" or not supported:
            return None
        if src in supported:
            return None
        return (
            "Qwen3-ASR does not list language '%s'. It supports: %s. "
            "Use auto-detect or pick a listed language." %
            (src, ", ".join(sorted(supported)))
        )

    def _run_pcm(self, pcm: np.ndarray, language: Optional[str]) -> str:
        with self._infer_lock:
            with self._lock:
                session = self._session
                if not self._ready or session is None:
                    return ""
            kwargs = {}
            if language and language != "auto":
                kwargs["language"] = language
            # User dictionary: priority-ordered vocabulary + free-text prompt.
            # Both are advertised as supported by Qwen3-ASR (probed via
            # model.supports()), but degrade gracefully on other providers.
            vocab, prompt = self.dictionary_payload()
            if vocab:
                kwargs["vocabulary"] = vocab
            if prompt:
                kwargs["prompt"] = prompt
            try:
                result = session.run(pcm, **kwargs)
            except Exception as error:
                msg = str(error).lower()
                optional = ("vocabulary", "prompt")
                rejected = isinstance(error, TypeError) or any(
                    key in msg for key in optional) or (
                    "invalid argument" in msg and len(kwargs) > 1)
                if not rejected:
                    raise
                # Drop the dictionary first, keep the language hint.
                fallback = {k: v for k, v in kwargs.items()
                            if k not in optional}
                try:
                    result = session.run(pcm, **fallback)
                except TypeError:
                    result = session.run(pcm)
                except Exception as retry_error:
                    if ("language" in str(retry_error).lower()
                            and "language" in fallback):
                        result = session.run(pcm)
                    else:
                        raise
            text = getattr(result, "text", None)
            if text is None:
                return ""
            return str(text).strip()

    # -------------------------------------------------------------- dictionary
    def set_dictionary(
        self,
        terms: Optional[List[str]] = None,
        context: Optional[str] = None,
        enabled: bool = True,
    ) -> None:
        """Attach (or clear) the user dictionary for the next decodes."""
        try:
            import dictionary as _dict
        except Exception:
            _dict = None
        try:
            clean = (_dict.norm_terms(terms) if _dict is not None
                     else [str(t).strip() for t in (terms or []) if str(t).strip()])
            ctx = (_dict.norm_context(context) if _dict is not None
                   else str(context or "").strip())
        except Exception:
            clean, ctx = [], ""
        with self._lock:
            self._dict_terms = list(clean)
            self._dict_context = ctx
            self._dict_enabled = bool(enabled)

    def dictionary_payload(self) -> Tuple[Optional[List[str]], Optional[str]]:
        """(vocabulary, prompt) for the current decode, or (None, None)."""
        with self._lock:
            if not self._dict_enabled:
                return None, None
            terms = list(self._dict_terms)
            context = self._dict_context
        if not terms and not context:
            return None, None
        try:
            import dictionary as _dict

            info = _dict.build("Qwen3-ASR", terms, context)
            return info.get("vocabulary"), info.get("prompt")
        except Exception:
            return (list(terms) or None), (context or None)

    def transcribe(
        self,
        audio_data: np.ndarray,
        sample_rate: int = 16000,
        task: Optional[str] = None,
        source_lang: Optional[str] = None,
    ) -> str:
        if not self.is_ready:
            return ""
        try:
            pcm = _to_16k_mono(audio_data, sample_rate)
            if pcm.size < 1600:  # under 0.1 s: nothing to decode
                return ""
            _, snap_src, _ = self._snapshot_opts()
            lang = source_lang if source_lang is not None else snap_src
            error = self._check_lang_support(lang)
            if error:
                return f"[Qwen Error: {error}]"
            return self._run_pcm(pcm, lang)
        except Exception as error:
            import traceback

            traceback.print_exc()
            return f"[Qwen Error: {error}]"

    def transcribe_file(
        self,
        wav_path: str,
        task: Optional[str] = None,
        source_lang: Optional[str] = None,
        target_lang: Optional[str] = None,
    ) -> str:
        if not self.is_ready:
            return ""
        try:
            import soundfile as sf

            audio, rate = sf.read(str(wav_path), dtype="float32", always_2d=False)
            _, snap_src, _ = self._snapshot_opts()
            lang = source_lang if source_lang is not None else snap_src
            error = self._check_lang_support(lang)
            if error:
                return f"[Qwen Error: {error}]"
            pcm = _to_16k_mono(audio, rate)
            if pcm.size < 1600:
                return ""
            return self._run_pcm(pcm, lang)
        except Exception as error:
            import traceback

            traceback.print_exc()
            return f"[Qwen Error: {error}]"

    def transcribe_file_segments(self, wav_path: str) -> List[Tuple[float, float, str]]:
        """Single-pass transcription with a whole-file cue.

        Qwen3-ASR reports no timestamps, so callers that need cues should
        chunk with VAD (SRT does) rather than rely on this."""
        text = self.transcribe_file(wav_path)
        if not text or text.startswith("[Qwen Error"):
            return []
        try:
            import soundfile as sf

            info = sf.info(str(wav_path))
            duration = float(info.frames) / float(info.samplerate or 16000)
        except Exception:
            duration = 0.0
        return [(0.0, max(duration, 0.1), text)]
