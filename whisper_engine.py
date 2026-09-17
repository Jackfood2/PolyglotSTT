import io
import logging
import os
import shutil
import threading
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple, Union

import numpy as np


PORTABLE_ROOT = Path(__file__).resolve().parent
WHISPER_MODELS_ROOT = PORTABLE_ROOT / "models_cache" / "whisper-models"
HF_CACHE = PORTABLE_ROOT / "models_cache" / "huggingface"

os.environ.setdefault("HF_HOME", str(HF_CACHE))
os.environ.setdefault("HF_HUB_CACHE", str(HF_CACHE / "hub"))
os.environ.setdefault(
    "HUGGINGFACE_HUB_CACHE",
    os.environ["HF_HUB_CACHE"],
)

WHISPER_MODEL_ID = "large-v3"

STALE_MODEL_DIRS = (
    "models--mobiuslabsgmbh--faster-whisper-large-v3-turbo",
)

WHISPER_TASKS = ["transcribe", "translate"]

WHISPER_SOURCE_LANGS = [
    "auto", "en", "ja", "zh", "ko",
    "de", "fr", "es", "it", "pt", "nl", "ru",
    "ar", "hi", "tr", "id", "uk", "vi", "th",
]

WHISPER_MODEL_CHOICES = {
    "Tiny (75MB, fastest)": "tiny",
    "Base (145MB)": "base",
    "Small (500MB)": "small",
    "Medium (1.5GB)": "medium",
    "Large v1 (3GB, oldest)": "large-v1",
    "Large v2 (3GB)": "large-v2",
    "Large v3 (3GB, best)": "large-v3",
}

WHISPER_MODEL_CHOICES_REV = {
    model_id: label
    for label, model_id in WHISPER_MODEL_CHOICES.items()
}

_MODEL_REPOS = {
    "tiny": "Systran/faster-whisper-tiny",
    "tiny.en": "Systran/faster-whisper-tiny.en",
    "base": "Systran/faster-whisper-base",
    "base.en": "Systran/faster-whisper-base.en",
    "small": "Systran/faster-whisper-small",
    "small.en": "Systran/faster-whisper-small.en",
    "medium": "Systran/faster-whisper-medium",
    "medium.en": "Systran/faster-whisper-medium.en",
    "large": "Systran/faster-whisper-large-v3",
    "large-v1": "Systran/faster-whisper-large-v1",
    "large-v2": "Systran/faster-whisper-large-v2",
    "large-v3": "Systran/faster-whisper-large-v3",
}

_LOG = logging.getLogger(__name__)

ReadyCallback = Callable[[bool, Optional[str]], None]
ProgressCallback = Callable[[str], None]
TimestampedText = Tuple[float, float, str]
AudioInput = Union[str, np.ndarray, io.BytesIO]


def _invoke_callback(callback: Optional[Callable], *args) -> None:
    if callback is None:
        return

    try:
        callback(*args)
    except Exception:
        _LOG.exception("Whisper callback failed")


def _whisper_dir_size(path) -> int:
    total = 0
    seen = set()

    for root, _, files in os.walk(path, followlinks=False):
        for name in files:
            file_path = Path(root) / name

            try:
                stat = file_path.stat()
            except OSError:
                continue

            identity = (
                (stat.st_dev, stat.st_ino)
                if stat.st_ino
                else str(file_path.resolve())
            )

            if identity in seen:
                continue

            seen.add(identity)
            total += stat.st_size

    return total


def whisper_repo_for(model_id: str) -> str:
    model_id = (model_id or "").strip()

    if not model_id:
        raise ValueError("Empty Whisper model ID")

    try:
        from faster_whisper.utils import _MODELS
    except ImportError:
        _MODELS = {}

    repo = _MODELS.get(model_id) or _MODEL_REPOS.get(model_id)

    if repo:
        return repo

    parts = model_id.split("/")

    if (
        len(parts) == 2
        and all(part and part not in {".", ".."} for part in parts)
        and "\\" not in model_id
    ):
        from huggingface_hub.utils import validate_repo_id

        validate_repo_id(model_id)
        return model_id

    raise ValueError(f"Unknown Whisper model: {model_id}")


def _whisper_cache_dir(models_root, repo: str) -> Path:
    root = (
        Path(models_root).expanduser()
        if models_root is not None
        else WHISPER_MODELS_ROOT
    )
    return root / ("models--" + repo.replace("/", "--"))


def _model_files_complete(path: Path) -> bool:
    required_files = (
        "model.bin",
        "config.json",
        "tokenizer.json",
    )
    vocabulary_files = (
        "vocabulary.json",
        "vocabulary.txt",
    )

    try:
        if not all(
            (path / name).is_file()
            and (path / name).stat().st_size > 0
            for name in required_files
        ):
            return False
        return any(
            (path / name).is_file()
            and (path / name).stat().st_size > 0
            for name in vocabulary_files
        )
    except OSError:
        return False


def _whisper_repo_complete(repo_dir: Path) -> bool:
    repo_dir = Path(repo_dir)

    if _model_files_complete(repo_dir):
        return True

    snapshots = repo_dir / "snapshots"
    main_ref = repo_dir / "refs" / "main"

    try:
        if main_ref.is_file():
            revision = main_ref.read_text(encoding="utf-8").strip()

            if (
                not revision
                or revision in {".", ".."}
                or "/" in revision
                or "\\" in revision
            ):
                return False

            return _model_files_complete(snapshots / revision)

        if snapshots.is_dir():
            return any(
                path.is_dir() and _model_files_complete(path)
                for path in snapshots.iterdir()
            )
    except (OSError, UnicodeError):
        return False

    return False


def whisper_downloaded_map(models_root=None) -> Dict[str, bool]:
    result = {}

    for model_id in WHISPER_MODEL_CHOICES.values():
        try:
            repo = whisper_repo_for(model_id)
            cache_dir = _whisper_cache_dir(models_root, repo)

            if _whisper_repo_complete(cache_dir):
                result[model_id] = True
        except Exception:
            _LOG.debug(
                "Could not inspect cache for %s",
                model_id,
                exc_info=True,
            )

    return result


def whisper_cache_info(models_root=None) -> Dict[str, int]:
    result = {}

    for model_id in WHISPER_MODEL_CHOICES.values():
        try:
            repo = whisper_repo_for(model_id)
            result[model_id] = _whisper_dir_size(
                _whisper_cache_dir(models_root, repo)
            )
        except Exception:
            result[model_id] = 0
            _LOG.debug(
                "Could not measure cache for %s",
                model_id,
                exc_info=True,
            )

    return result


def delete_whisper_model(
    models_root=None,
    model_id: str = WHISPER_MODEL_ID,
) -> Tuple[int, List[str]]:
    repo = whisper_repo_for(model_id)
    cache_dir = _whisper_cache_dir(models_root, repo)

    if cache_dir.is_symlink():
        raise ValueError(f"Refusing to delete a linked cache: {cache_dir}")

    if not cache_dir.is_dir():
        raise FileNotFoundError(f"Not downloaded: {model_id}")

    size = _whisper_dir_size(cache_dir)
    affected = [
        candidate
        for candidate in WHISPER_MODEL_CHOICES.values()
        if whisper_repo_for(candidate) == repo
    ]

    shutil.rmtree(cache_dir)

    return size, affected


class WhisperEngine:
    def __init__(
        self,
        task: str = "translate",
        source_lang: str = "ja",
        target_lang: str = "en",
        model_id: str = WHISPER_MODEL_ID,
        device: str = "auto",
        on_ready: Optional[ReadyCallback] = None,
    ):
        self.task = task if task in WHISPER_TASKS else "transcribe"
        self.source_lang = self._normalize_language(source_lang)
        self.target_lang = (
            "en"
            if self.task == "translate"
            else (target_lang or "en").strip().lower()
        )
        self.model_id = (model_id or WHISPER_MODEL_ID).strip()
        self.device = (device or "auto").strip().lower() or "auto"

        self._device_used = "cpu"
        self._compute_used = "int8"
        self._device_reason = ""
        self._model = None
        self._ready = False
        self._loading = False
        self._last_error: Optional[str] = None
        self._load_generation = 0
        # Requested model/device for the current load and installed model.
        self._load_request = None
        self._loaded_request = None
        self._on_ready = on_ready

        self._lock = threading.Lock()
        self._infer_lock = threading.Lock()
        self._load_lock = threading.Lock()

    @staticmethod
    def _normalize_language(language: Optional[str]) -> str:
        return (language or "auto").strip().lower() or "auto"

    @staticmethod
    def _request_key(model_id: str, device: str) -> Tuple[str, str]:
        device = (device or "auto").strip().lower()
        if device == "gpu":
            device = "cuda"
        return model_id, device

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
    def current_arch_name(self) -> str:
        with self._lock:
            target = (
                "en"
                if self.task == "translate"
                else self.source_lang
            )
            return (
                f"whisper-{self.model_id} "
                f"({self.task} {self.source_lang}->{target})"
            )

    @property
    def device_info(self) -> str:
        with self._lock:
            return f"{self._device_used} {self._compute_used}"

    @property
    def device_reason(self) -> str:
        with self._lock:
            return self._device_reason

    def _resolve_device(
        self,
        model_id: Optional[str] = None,
        requested_device: Optional[str] = None,
    ) -> Tuple[str, str, str]:
        if model_id is None or requested_device is None:
            with self._lock:
                model_id = model_id or self.model_id
                requested_device = requested_device or self.device

        device = requested_device.strip().lower()

        if device == "gpu":
            device = "cuda"

        if device not in {"auto", "cpu", "cuda"}:
            raise ValueError(f"Unsupported Whisper device: {device}")

        if device == "cpu":
            return "cpu", "int8", "CPU requested"

        try:
            import gpu
        except ImportError:
            gpu = None

        if gpu is not None:
            try:
                recommended, compute, reason = gpu.recommend_whisper(
                    model_id
                )

                if recommended in {"cpu", "cuda"} and compute:
                    return recommended, compute, str(reason)
            except Exception:
                _LOG.debug("GPU recommendation failed", exc_info=True)

        try:
            import ctranslate2

            if ctranslate2.get_cuda_device_count() > 0:
                supported = ctranslate2.get_supported_compute_types(
                    "cuda"
                )

                for compute in (
                    "float16",
                    "int8_float16",
                    "float32",
                    "int8",
                ):
                    if compute in supported:
                        return "cuda", compute, "CUDA available"
        except Exception:
            _LOG.debug("CUDA detection failed", exc_info=True)

        return "cpu", "int8", "CUDA unavailable"

    def _ensure_dirs(self) -> None:
        WHISPER_MODELS_ROOT.mkdir(parents=True, exist_ok=True)
        HF_CACHE.mkdir(parents=True, exist_ok=True)

    def _cpu_threads(self) -> int:
        for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS"):
            try:
                count = int(os.environ.get(name, "0"))
            except (TypeError, ValueError):
                continue

            if count > 0:
                return count

        try:
            return max(1, len(os.sched_getaffinity(0)))
        except (AttributeError, OSError):
            return max(1, os.cpu_count() or 4)

    @staticmethod
    def _dispose_model(model) -> None:
        if model is None:
            return

        try:
            close = getattr(model, "close", None)

            if callable(close):
                close()
                return

            backend = getattr(model, "model", None)
            unload = getattr(backend, "unload_model", None)

            if callable(unload):
                unload()
        except Exception:
            _LOG.warning("Could not release Whisper model", exc_info=True)

    def _is_current(self, generation: int) -> bool:
        with self._lock:
            return generation == self._load_generation

    def _report_progress(
        self,
        generation: int,
        callback: Optional[ProgressCallback],
        message: str,
    ) -> None:
        if self._is_current(generation):
            _invoke_callback(callback, message)

    def _model_path(
        self,
        model_id: str,
        generation: int,
        on_progress: Optional[ProgressCallback],
    ) -> str:
        local_path = Path(model_id).expanduser()

        if local_path.is_dir():
            return str(local_path.resolve())

        from faster_whisper.utils import download_model
        from huggingface_hub import constants

        self._report_progress(
            generation,
            on_progress,
            "Checking the local model cache",
        )

        try:
            return download_model(
                model_id,
                cache_dir=str(WHISPER_MODELS_ROOT),
                local_files_only=True,
            )
        except Exception:
            offline = (
                constants.HF_HUB_OFFLINE
                or os.environ.get("HF_HUB_OFFLINE", "").strip().upper()
                in {"1", "TRUE", "YES", "ON"}
            )

            if offline or not self._is_current(generation):
                raise

        self._report_progress(
            generation,
            on_progress,
            "Downloading model files",
        )

        return download_model(
            model_id,
            cache_dir=str(WHISPER_MODELS_ROOT),
            local_files_only=False,
        )

    def _load_worker(
        self,
        generation: int,
        model_id: str,
        requested_device: str,
        on_progress: Optional[ProgressCallback],
        switch_callback: Optional[ReadyCallback],
    ) -> None:
        model = None

        try:
            with self._load_lock:
                if not self._is_current(generation):
                    return

                with self._infer_lock:
                    with self._lock:
                        if generation != self._load_generation:
                            return

                        old_model = self._model
                        self._model = None
                        self._loaded_request = None

                    self._dispose_model(old_model)
                    del old_model

                self._ensure_dirs()

                from faster_whisper import WhisperModel

                path = self._model_path(
                    model_id,
                    generation,
                    on_progress,
                )

                if not self._is_current(generation):
                    return

                device, compute, reason = self._resolve_device(
                    model_id,
                    requested_device,
                )

                self._report_progress(
                    generation,
                    on_progress,
                    f"Loading model on {device} with {compute}",
                )

                kwargs = {
                    "cpu_threads": self._cpu_threads(),
                    "local_files_only": True,
                }

                try:
                    model = WhisperModel(
                        path,
                        device=device,
                        compute_type=compute,
                        **kwargs,
                    )
                except Exception as cuda_error:
                    if device != "cuda":
                        raise

                    if not self._is_current(generation):
                        return

                    _LOG.warning(
                        "CUDA model loading failed; trying CPU: %s",
                        cuda_error,
                    )

                    self._report_progress(
                        generation,
                        on_progress,
                        "CUDA loading failed; retrying on CPU",
                    )

                    device = "cpu"
                    compute = "int8"
                    reason = f"CUDA loading failed: {cuda_error}"

                    model = WhisperModel(
                        path,
                        device=device,
                        compute_type=compute,
                        **kwargs,
                    )

                with self._infer_lock:
                    with self._lock:
                        if generation != self._load_generation:
                            return

                        self._model = model
                        model = None
                        self._ready = True
                        self._loading = False
                        self._last_error = None
                        self._device_used = device
                        self._compute_used = compute
                        self._device_reason = reason
                        self._loaded_request = self._request_key(model_id, requested_device)

        except Exception as error:
            with self._lock:
                if generation != self._load_generation:
                    return

                self._ready = False
                self._loading = False
                self._last_error = str(error)

            _LOG.exception("Whisper model loading failed")

            if self._is_current(generation):
                _invoke_callback(self._on_ready, False, str(error))

            if self._is_current(generation):
                _invoke_callback(switch_callback, False, str(error))

            return

        finally:
            self._dispose_model(model)

        self._report_progress(generation, on_progress, "Model ready")

        if self._is_current(generation):
            _invoke_callback(self._on_ready, True, None)

        if self._is_current(generation):
            _invoke_callback(switch_callback, True, None)

    def _start_load(
        self,
        model_id: Optional[str] = None,
        on_progress: Optional[ProgressCallback] = None,
        switch_callback: Optional[ReadyCallback] = None,
        force: bool = False,
    ) -> None:
        with self._lock:
            selected_model = (
                self.model_id if model_id is None else model_id
            )
            selected_device = self.device
            request = self._request_key(selected_model, selected_device)

            already_ready = (
                not force
                and self._ready
                and self._model is not None
                and not self._loading
                and self._loaded_request == request
            )

            already_loading = (
                not force
                and self._loading
                and self._load_request == request
            )

            # Ordinary repeated load requests do not supersede an
            # identical load. A switch request with a completion callback
            # retains the existing superseding-request semantics.
            if already_loading and switch_callback is None:
                return

            if not already_ready:
                self.model_id = selected_model
                self._load_generation += 1
                generation = self._load_generation
                self._load_request = request
                self._ready = False
                self._loading = True
                self._last_error = None

        if already_ready:
            _invoke_callback(on_progress, "Model already ready")
            if switch_callback is not None:
                _invoke_callback(switch_callback, True, None)
            else:
                _invoke_callback(self._on_ready, True, None)
            return

        try:
            worker = threading.Thread(
                target=self._load_worker,
                args=(
                    generation,
                    selected_model,
                    selected_device,
                    on_progress,
                    switch_callback,
                ),
                name=f"WhisperLoader-{generation}",
                daemon=True,
            )
            worker.start()

        except Exception as error:
            with self._lock:
                if generation != self._load_generation:
                    return

                self._ready = False
                self._loading = False
                self._last_error = str(error)

            _LOG.exception("Could not start Whisper loader")

            if self._is_current(generation):
                _invoke_callback(self._on_ready, False, str(error))

            if self._is_current(generation):
                _invoke_callback(switch_callback, False, str(error))

    def load(
        self,
        on_progress: Optional[ProgressCallback] = None,
        *,
        force: bool = False,
    ) -> None:
        """Start loading asynchronously.

        Repeated ordinary calls preserve a matching ready/loading model.
        Use force=True to retry device resolution or rebuild the model.
        """
        self._start_load(
            on_progress=on_progress,
            force=force,
        )

    def switch_options(
        self,
        task: Optional[str] = None,
        source_lang: Optional[str] = None,
        target_lang: Optional[str] = None,
        on_ready: Optional[ReadyCallback] = None,
    ) -> None:
        with self._lock:
            if task in WHISPER_TASKS:
                self.task = task

            if source_lang is not None:
                self.source_lang = self._normalize_language(source_lang)

            if target_lang:
                self.target_lang = target_lang.strip().lower() or "en"

            if self.task == "translate":
                self.target_lang = "en"

        _invoke_callback(on_ready, True, None)

    def switch_model(
        self,
        model_id: str,
        on_ready: Optional[ReadyCallback] = None,
    ) -> None:
        model_id = (model_id or "").strip()

        if not model_id:
            _invoke_callback(on_ready, False, "Empty model ID")
            return

        self._start_load(
            model_id=model_id,
            switch_callback=on_ready,
        )

    def unload(self) -> bool:
        with self._infer_lock:
            with self._lock:
                had_work = self._model is not None or self._loading
                self._load_generation += 1

                model = self._model
                self._model = None
                self._ready = False
                self._loading = False
                self._last_error = None
                self._load_request = None
                self._loaded_request = None
                self._device_reason = "Unloaded"

            self._dispose_model(model)

        return had_work

    def _snapshot_opts(self) -> Tuple[str, str, str]:
        with self._lock:
            return self.task, self.source_lang, self.target_lang

    def _transcribe_opts(
        self,
        task: Optional[str] = None,
        source_lang: Optional[str] = None,
    ) -> dict:
        current_task, current_source, _ = self._snapshot_opts()

        effective_task = (
            task if task in WHISPER_TASKS else current_task
        )
        effective_source = self._normalize_language(
            source_lang if source_lang is not None else current_source
        )

        return {
            "task": effective_task,
            "language": (
                None if effective_source == "auto" else effective_source
            ),
            "beam_size": 5,
            "condition_on_previous_text": False,
            "word_timestamps": False,
        }

    def _run(
        self,
        wav_path: AudioInput,
        with_timestamps: bool = False,
        task: Optional[str] = None,
        source_lang: Optional[str] = None,
        word_timestamps: bool = False,
    ):
        with self._infer_lock:
            with self._lock:
                if not self._ready or self._model is None:
                    return [], None

                model = self._model

            options = self._transcribe_opts(
                task=task,
                source_lang=source_lang,
            )
            options["word_timestamps"] = bool(word_timestamps)

            segments, info = model.transcribe(wav_path, **options)
            return list(segments), info

    @staticmethod
    def _norm_audio(audio_data: np.ndarray) -> np.ndarray:
        array = np.asarray(audio_data)

        if array.ndim not in {1, 2}:
            raise ValueError(
                "Audio must have shape (samples,) or (samples, channels)"
            )

        if array.dtype.kind not in {"i", "u", "f"}:
            raise TypeError("Audio must contain real numeric samples")

        if array.ndim == 2 and array.shape[1] == 0:
            raise ValueError("Audio must contain at least one channel")

        if array.size == 0:
            return np.empty(0, dtype=np.float32)

        with np.errstate(over="ignore", invalid="ignore"):
            if array.dtype.kind == "i":
                limits = np.iinfo(array.dtype)
                scale = np.float32(
                    max(abs(int(limits.min)), int(limits.max))
                )
                output = array.astype(np.float32, copy=True) / scale
            elif array.dtype.kind == "u":
                limits = np.iinfo(array.dtype)
                midpoint = np.float32((int(limits.max) + 1) / 2.0)
                output = (
                    array.astype(np.float32, copy=True) - midpoint
                ) / midpoint
            else:
                output = array.astype(np.float32, copy=True)

        np.nan_to_num(
            output,
            copy=False,
            nan=0.0,
            posinf=1.0,
            neginf=-1.0,
        )
        np.clip(output, -1.0, 1.0, out=output)

        if output.ndim == 2:
            output = output.mean(axis=1, dtype=np.float32)

        return np.ascontiguousarray(output, dtype=np.float32)

    @staticmethod
    def _join_text(segments) -> str:
        return " ".join(
            text
            for segment in segments
            if (text := (segment.text or "").strip())
        )

    @staticmethod
    def _timestamped_text(
        start,
        end,
        text,
    ) -> Optional[TimestampedText]:
        try:
            start = float(start)
            end = float(end)
            text = (text or "").strip()
        except (TypeError, ValueError, AttributeError, OverflowError):
            return None

        if (
            not text
            or not np.isfinite(start)
            or not np.isfinite(end)
            or start < 0
            or end <= start
        ):
            return None

        return start, end, text

    def transcribe(
        self,
        audio_data: np.ndarray,
        sample_rate: int = 16000,
    ) -> str:
        if not self.is_ready:
            return ""

        try:
            if (
                isinstance(sample_rate, (bool, np.bool_))
                or not isinstance(sample_rate, (int, np.integer))
                or sample_rate <= 0
            ):
                raise ValueError("sample_rate must be a positive integer")

            audio = self._norm_audio(audio_data)

            if audio.size == 0:
                return ""

            if sample_rate == 16000:
                segments, _ = self._run(audio)
            else:
                import soundfile as sf

                with io.BytesIO() as buffer:
                    sf.write(
                        buffer,
                        audio,
                        samplerate=int(sample_rate),
                        format="WAV",
                        subtype="FLOAT",
                    )
                    buffer.seek(0)
                    segments, _ = self._run(buffer)

            return self._join_text(segments)

        except Exception as error:
            _LOG.exception("Whisper audio transcription failed")
            return f"[Whisper Error: {error}]"

    def transcribe_file(
        self,
        wav_path: str,
        task: Optional[str] = None,
        source_lang: Optional[str] = None,
    ) -> str:
        if not self.is_ready:
            return ""

        try:
            segments, _ = self._run(
                os.fspath(wav_path),
                task=task,
                source_lang=source_lang,
            )
            return self._join_text(segments)

        except Exception as error:
            _LOG.exception("Whisper file transcription failed")
            return f"[Whisper Error: {error}]"

    def transcribe_file_segments(
        self,
        wav_path: str,
        task: Optional[str] = None,
        source_lang: Optional[str] = None,
    ) -> List[TimestampedText]:
        if not self.is_ready:
            return []

        segments, _ = self._run(
            os.fspath(wav_path),
            with_timestamps=True,
            task=task,
            source_lang=source_lang,
        )

        output = []

        for segment in segments:
            item = self._timestamped_text(
                segment.start,
                segment.end,
                segment.text,
            )

            if item is not None:
                output.append(item)

        return output

    def transcribe_file_words(
        self,
        wav_path: str,
        task: Optional[str] = None,
        source_lang: Optional[str] = None,
    ) -> Tuple[List[TimestampedText], List[TimestampedText]]:
        if not self.is_ready:
            return [], []

        segments, _ = self._run(
            os.fspath(wav_path),
            with_timestamps=True,
            task=task,
            source_lang=source_lang,
            word_timestamps=True,
        )

        output_segments = []
        output_words = []

        for segment in segments:
            item = self._timestamped_text(
                segment.start,
                segment.end,
                segment.text,
            )

            if item is not None:
                output_segments.append(item)

            for word in getattr(segment, "words", None) or []:
                item = self._timestamped_text(
                    getattr(word, "start", None),
                    getattr(word, "end", None),
                    getattr(word, "word", ""),
                )

                if item is not None:
                    output_words.append(item)

        return output_segments, output_words