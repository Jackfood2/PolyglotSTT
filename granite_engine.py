"""Granite Speech 5.0 470M TurboCTC engine (GGUF via transcribe.cpp).

IBM's Granite Speech 5.0 TurboCTC is a 470M-parameter encoder-only CTC model
(Apache-2.0) for English. It is dramatically faster than the LLM-backed engines
(measured 20-25x realtime on this project's Vulkan path, vs ~2.7x for Qwen3-ASR)
which makes it the engine of choice for Live dictation.

Trade-offs, all verified against the transcribe.cpp 0.4.1 build:

  * English only — no language selection, no language detect.
  * Transcription only — no translation task.
  * No timestamps of any kind (`max_timestamp_kind=none`), so SRT generation
    runs it over VAD chunks and derives cue timing from chunk boundaries,
    exactly like Canary/Qwen3-ASR.
  * No punctuation and no capitalisation. CTC output is raw lowercase text.
    Qwen3-ASR gets PnC for free from its language model; this model cannot, and
    the runtime reports `supports pnc/itn: False`. The app's ITN pass and the
    sentence formatter still apply, but expect flatter text than Qwen.
  * No vocabulary/context-prompt support, so the user dictionary does not
    apply to this engine (dictionary.build() reports it as ineligible).
  * Hallucinates on non-speech. Silence can decode to a stray word ("the",
    "thank you"), so a low-level gate below suppresses near-silent input. The
    recorder's Silero VAD does the real work; this is defence in depth.

The heavy lifting is done by the `transcribe-cpp` Python package (CPU + Vulkan
backends). Models are single `.gguf` files kept in models_cache/granite/ so the
project stays portable and fully offline once the file is present.
"""

import gc
import os
import threading
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np

PORTABLE_ROOT = Path(__file__).resolve().parent
GRANITE_CACHE = PORTABLE_ROOT / "models_cache" / "granite"
HF_CACHE = PORTABLE_ROOT / "models_cache" / "huggingface"

os.environ.setdefault("HF_HOME", str(HF_CACHE))
os.environ.setdefault("HUGGINGFACE_HUB_CACHE", str(HF_CACHE))

GRANITE_REPO = "handy-computer/granite-speech-5.0-470m-turboctc-gguf"

# Label -> short id used in config files.
GRANITE_MODEL_CHOICES = {
    "Granite 5.0 470M Q8_0 (506MB, best)": "granite-5.0-470m-q8_0",
    "Granite 5.0 470M Q6_K (392MB)": "granite-5.0-470m-q6_k",
    "Granite 5.0 470M Q5_K_M (336MB)": "granite-5.0-470m-q5_k_m",
    "Granite 5.0 470M Q4_K_M (279MB, smallest)": "granite-5.0-470m-q4_k_m",
}

GRANITE_MODEL_CHOICES_REV = {v: k for k, v in GRANITE_MODEL_CHOICES.items()}

GRANITE_MODEL_FILES = {
    "granite-5.0-470m-q8_0": "granite-speech-5.0-470m-turboctc-Q8_0.gguf",
    "granite-5.0-470m-q6_k": "granite-speech-5.0-470m-turboctc-Q6_K.gguf",
    "granite-5.0-470m-q5_k_m": "granite-speech-5.0-470m-turboctc-Q5_K_M.gguf",
    "granite-5.0-470m-q4_k_m": "granite-speech-5.0-470m-turboctc-Q4_K_M.gguf",
}

GRANITE_DEFAULT_MODEL = "granite-5.0-470m-q8_0"

# Resident RAM estimates (weights + encoder scratch), in MB. Measured weights
# are ~482 MiB at Q8_0; the rest is the ggml sched buffers.
GRANITE_RESIDENT_MB = {
    "granite-5.0-470m-q8_0": 700,
    "granite-5.0-470m-q6_k": 560,
    "granite-5.0-470m-q5_k_m": 480,
    "granite-5.0-470m-q4_k_m": 420,
}

# English only. "auto" is accepted so the shared UI language plumbing works.
GRANITE_LANGUAGES = ["auto", "en"]

GRANITE_TASKS = ["transcribe"]

# Minimum GGUF size that counts as a real download (bytes).
_MIN_MODEL_BYTES = 100_000_000

# RMS below this is treated as silence and never decoded. -78 dBFS: well under
# any real speech, high enough to catch digital silence and room noise that
# would otherwise make the CTC head emit a stray word.
_SILENCE_RMS = 2.0e-4

_LOG_PREFIX = "[granite]"


def _log(message: str) -> None:
    try:
        print(f"{_LOG_PREFIX} {message}")
    except Exception:
        pass


def granite_model_id(model_id: Optional[str]) -> str:
    mid = str(model_id or "").strip()
    return mid if mid in GRANITE_MODEL_FILES else GRANITE_DEFAULT_MODEL


def granite_model_filename(model_id: Optional[str]) -> str:
    return GRANITE_MODEL_FILES[granite_model_id(model_id)]


def granite_model_path(model_id: Optional[str] = None) -> Path:
    return GRANITE_CACHE / granite_model_filename(model_id)


def granite_downloaded_map() -> Dict[str, bool]:
    out = {}
    for mid, filename in GRANITE_MODEL_FILES.items():
        try:
            path = GRANITE_CACHE / filename
            out[mid] = path.is_file() and path.stat().st_size > _MIN_MODEL_BYTES
        except Exception:
            out[mid] = False
    return out


def granite_cache_info() -> Dict[str, int]:
    out = {}
    for mid, filename in GRANITE_MODEL_FILES.items():
        try:
            path = GRANITE_CACHE / filename
            out[mid] = path.stat().st_size if path.is_file() else 0
        except Exception:
            out[mid] = 0
    return out


def delete_granite_model(model_id: Optional[str] = None) -> int:
    path = granite_model_path(model_id)
    if not path.is_file():
        raise FileNotFoundError(f"not downloaded: {path.name}")
    size = path.stat().st_size
    path.unlink()
    return size


def _resident_model_mb(model_id: Optional[str]) -> int:
    return int(GRANITE_RESIDENT_MB.get(granite_model_id(model_id), 700))


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

    rate = int(sample_rate or 16000)
    if rate != 16000 and audio.size:
        n_out = int(round(audio.size * 16000.0 / float(rate)))
        if n_out > 0 and n_out != audio.size:
            x_old = np.linspace(0.0, 1.0, audio.size, endpoint=False)
            x_new = np.linspace(0.0, 1.0, n_out, endpoint=False)
            audio = np.interp(x_new, x_old, audio).astype(np.float32)

    return audio


def _rms(pcm: np.ndarray) -> float:
    try:
        if pcm.size == 0:
            return 0.0
        return float(np.sqrt(np.mean(np.square(pcm, dtype=np.float64))))
    except Exception:
        return 0.0


class GraniteEngine:
    """Drop-in sibling of QwenEngine/WhisperEngine for Granite Speech TurboCTC.

    Public surface used by the app: load(), is_ready, _loading, _last_error,
    unload(), switch_options(), switch_model(), transcribe(),
    transcribe_file(), transcribe_file_segments(), device, device_info,
    current_arch_name, supported_source_langs, supports_translate,
    supports_timestamps.

    It mirrors QwenEngine deliberately: the app treats every heavy engine the
    same way, and a different shape here would need more branches there.
    """

    _load_gate = threading.Lock()

    def __init__(
        self,
        task: str = "transcribe",
        source_lang: str = "auto",
        target_lang: str = "en",
        model_id: str = GRANITE_DEFAULT_MODEL,
        device: str = "auto",
        on_ready: Optional[Callable] = None,
    ):
        # Granite TurboCTC cannot translate or pick a language.
        self.task = "transcribe"
        self.source_lang = (
            source_lang if source_lang in GRANITE_LANGUAGES else "auto"
        )
        self.target_lang = "en"
        self.model_id = granite_model_id(model_id)
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
        self.supported_source_langs = {"en"}
        self.supports_translate = False
        self.supports_timestamps = False
        # Accepted for API symmetry; the model has no vocabulary or prompt.
        self._dict_terms: List[str] = []
        self._dict_context: str = ""
        self._dict_enabled = False
        self._silence_skips = 0

    # ------------------------------------------------------------------ state
    @property
    def is_ready(self) -> bool:
        with self._lock:
            return self._ready and self._model is not None

    @property
    def is_loading(self) -> bool:
        with self._lock:
            return self._loading

    @property
    def last_error(self) -> Optional[str]:
        with self._lock:
            return self._last_error

    @property
    def device_info(self) -> str:
        with self._lock:
            return self._device_used or "cpu"

    @property
    def current_arch_name(self) -> str:
        with self._lock:
            return f"granite-turboctc {self.model_id}"

    # -------------------------------------------------------------- model file
    def _ensure_dirs(self) -> None:
        for path in (GRANITE_CACHE, HF_CACHE):
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
        filename = granite_model_filename(model_id)
        local = GRANITE_CACHE / filename

        if local.is_file() and local.stat().st_size > _MIN_MODEL_BYTES:
            return str(local)

        if not self._is_current(generation):
            raise RuntimeError("superseded")

        offline = os.environ.get("HF_HUB_OFFLINE", "").strip().upper() in {
            "1", "TRUE", "YES", "ON",
        }

        if offline:
            raise FileNotFoundError(
                f"{filename} is not in models_cache/granite and "
                f"HF_HUB_OFFLINE is set"
            )

        try:
            from huggingface_hub import hf_hub_download
        except Exception as error:
            raise RuntimeError(
                f"huggingface_hub is required to download {filename}: {error}"
            )

        self._report(on_progress, f"Downloading {filename}")
        _log(f"downloading {GRANITE_REPO}/{filename}")

        path = hf_hub_download(
            repo_id=GRANITE_REPO,
            filename=filename,
            local_dir=str(GRANITE_CACHE),
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
            name=f"GraniteLoader-{generation}",
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
            with GraniteEngine._load_gate:
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

                self._report(on_progress, "Loading Granite Speech model")

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
                        self.supported_source_langs = languages
                    model = None
                    session = None
                    installed = True

                _log(
                    "ready: %s on %s (%.2f MB)"
                    % (self.model_id, backend_name,
                       _resident_model_mb(self.model_id))
                )

                callback = self._on_ready
                if callback is not None:
                    try:
                        callback(True, None)
                    except Exception:
                        pass

        except Exception as error:
            message = str(error)
            _log(f"load failed: {message}")
            self._close_session(session)
            self._close_model(model)
            with self._lock:
                if generation == self._load_generation:
                    self._ready = False
                    self._loading = False
                    self._model = None
                    self._session = None
                    self._last_error = message
            callback = self._on_ready
            if callback is not None:
                try:
                    callback(False, message)
                except Exception:
                    pass
        finally:
            if not installed:
                with self._lock:
                    if generation == self._load_generation:
                        self._loading = False

    # --------------------------------------------------------------- teardown
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
        mid = granite_model_id(model_id)
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
                    source_lang if source_lang in GRANITE_LANGUAGES else "auto"
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

    def _check_lang_support(
        self,
        source_lang: Optional[str],
    ) -> Optional[str]:
        supported = self.supported_source_langs or {"en"}
        src = (source_lang or "auto").strip().lower()
        if src == "auto" or src in supported:
            return None
        return (
            "Granite TurboCTC is English only; it cannot transcribe '%s'. "
            "Pick English or use another engine for that language." % src
        )

    # ------------------------------------------------------------- inference
    def _run_pcm(self, pcm: np.ndarray, language: Optional[str]) -> str:
        with self._infer_lock:
            with self._lock:
                session = self._session
                if not self._ready or session is None:
                    return ""

            if _rms(pcm) < _SILENCE_RMS:
                # The CTC head invents short words on non-speech, so never
                # decode near-silence. Real speech sits far above this floor.
                with self._lock:
                    self._silence_skips += 1
                return ""

            kwargs = {}
            if language and language not in ("auto", ""):
                kwargs["language"] = language

            try:
                result = session.run(pcm, **kwargs)
            except Exception as error:
                msg = str(error).lower()
                if kwargs and (isinstance(error, TypeError)
                               or "language" in msg):
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
        """Accepted for API symmetry; TurboCTC has no vocabulary or prompt."""
        self._dict_terms = []
        self._dict_context = ""
        self._dict_enabled = False

    def dictionary_payload(self) -> Tuple[Optional[List[str]], Optional[str]]:
        return None, None

    # ------------------------------------------------------------- transcribe
    def transcribe(
        self,
        audio_data: np.ndarray,
        sample_rate: int = 16000,
        task: Optional[str] = None,
        source_lang: Optional[str] = None,
        extra_prompt: Optional[str] = None,
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
                return f"[Granite Error: {error}]"
            return self._run_pcm(pcm, lang)
        except Exception as error:
            import traceback

            traceback.print_exc()
            return f"[Granite Error: {error}]"

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

            audio, rate = sf.read(str(wav_path), dtype="float32",
                                  always_2d=False)
            _, snap_src, _ = self._snapshot_opts()
            lang = source_lang if source_lang is not None else snap_src
            error = self._check_lang_support(lang)
            if error:
                return f"[Granite Error: {error}]"
            pcm = _to_16k_mono(audio, rate)
            if pcm.size < 1600:
                return ""
            return self._run_pcm(pcm, lang)
        except Exception as error:
            import traceback

            traceback.print_exc()
            return f"[Granite Error: {error}]"

    def transcribe_file_segments(
        self, wav_path: str
    ) -> List[Tuple[float, float, str]]:
        """Single-pass transcription with a whole-file cue.

        Granite TurboCTC reports no timestamps, so callers that need cues
        should chunk with VAD (SRT does) rather than rely on this."""
        text = self.transcribe_file(wav_path)
        if not text or text.startswith("[Granite Error"):
            return []
        try:
            import soundfile as sf

            info = sf.info(str(wav_path))
            duration = float(info.frames) / float(info.samplerate or 16000)
        except Exception:
            duration = 0.0
        return [(0.0, max(duration, 0.1), text)]

