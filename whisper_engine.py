import pathlib
# whisper_engine.py
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
    "Large (3GB)": "large",
    "Large v1 (3GB)": "large-v1",
    "Large v2 (3GB)": "large-v2",
    "Large v3 (3GB, best)": "large-v3",
}
WHISPER_MODEL_CHOICES_REV = {v: k for k, v in WHISPER_MODEL_CHOICES.items()}
def _whisper_dir_size(path) -> int:
    import os as _os
    try:
        total = 0
        for root, _dirs, files in _os.walk(str(path)):
            for fn in files:
                try:
                    total += _os.path.getsize(_os.path.join(root, fn))
                except Exception:
                    pass
        return total
    except Exception:
        return 0
def whisper_repo_for(model_id: str) -> str:
    try:
        from faster_whisper.utils import _MODELS
        repo = (_MODELS or {}).get(model_id)
        if repo:
            return repo
    except Exception:
        pass
    fallback = {
        "tiny": "Systran/faster-whisper-tiny",
        "base": "Systran/faster-whisper-base",
        "small": "Systran/faster-whisper-small",
        "medium": "Systran/faster-whisper-medium",
        "large": "Systran/faster-whisper-large-v3",
        "large-v1": "Systran/faster-whisper-large-v1",
        "large-v2": "Systran/faster-whisper-large-v2",
        "large-v3": "Systran/faster-whisper-large-v3",
    }
    if model_id in fallback:
        return fallback[model_id]
    raise ValueError(f"unknown whisper model: {model_id}")
def _whisper_cache_dir(models_root, repo: str) -> Path:
    root = Path(models_root) if models_root else WHISPER_MODELS_ROOT
    return root / ("models--" + repo.replace("/", "--"))
def _whisper_repo_complete(repo_dir: Path) -> bool:
    try:
        snap = repo_dir / "snapshots"
        if not snap.exists():
            return _whisper_dir_size(repo_dir) > 50_000_000
        for sub in snap.iterdir():
            try:
                b = sub / "model.bin"
                if b.exists() and b.stat().st_size > 50_000_000:
                    return True
            except Exception:
                continue
    except Exception:
        pass
    return False
def whisper_downloaded_map(models_root=None) -> dict:
    out = {}
    for mid in WHISPER_MODEL_CHOICES.values():
        try:
            if _whisper_repo_complete(
                    _whisper_cache_dir(models_root, whisper_repo_for(mid))):
                out[mid] = True
        except Exception:
            pass
    return out
def whisper_cache_info(models_root=None) -> dict:
    out = {}
    for mid in WHISPER_MODEL_CHOICES.values():
        try:
            d = _whisper_cache_dir(models_root, whisper_repo_for(mid))
            out[mid] = _whisper_dir_size(d) if d.exists() else 0
        except Exception:
            out[mid] = 0
    return out
def delete_whisper_model(models_root=None, model_id: str = "large-v3"):
    import shutil
    repo = whisper_repo_for(model_id)
    d = _whisper_cache_dir(models_root, repo)
    if not d.exists():
        raise FileNotFoundError(f"not downloaded: {model_id}")
    freed = _whisper_dir_size(d)
    shutil.rmtree(str(d), ignore_errors=False)
    affected = [mid for mid in WHISPER_MODEL_CHOICES.values()
                if whisper_repo_for(mid) == repo]
    return freed, affected
class WhisperEngine:
    def __init__(self, task: str = "translate", source_lang: str = "ja",
                 target_lang: str = "en", model_id: str = WHISPER_MODEL_ID,
                 device: str = "auto",
                 on_ready: Optional[Callable] = None):
        self.task = task if task in WHISPER_TASKS else "transcribe"
        self.source_lang = source_lang if source_lang else "auto"
        self.target_lang = "en" if self.task == "translate" else (target_lang or "en")
        self.model_id = model_id or WHISPER_MODEL_ID
        self.device = (device or "auto").strip().lower() or "auto"
        self._device_used = "cpu"
        self._compute_used = "int8"
        self._device_reason = ""
        self._model = None
        self._ready = False
        self._on_ready = on_ready
        self._lock = threading.Lock()
        self._infer_lock = threading.Lock()
        self._loading = False
        self._last_error: Optional[str] = None
        self._load_generation = 0
        self._switch_cb: Optional[Callable] = None
    @property
    def is_ready(self) -> bool:
        return self._ready
    @property
    def current_arch_name(self) -> str:
        return f"whisper-{self.model_id} ({self.task} {self.source_lang}->{self.target_lang})"
    @property
    def device_info(self) -> str:
        try:
            return f"{self._device_used} {self._compute_used}".strip()
        except Exception:
            return "cpu"
    def _resolve_device(self):
        try:
            import gpu as _gpumod
        except Exception:
            _gpumod = None
        try:
            d = (self.device or "auto").strip().lower() or "auto"
        except Exception:
            d = "auto"
        if d == "gpu":
            d = "cuda"
        if d == "cpu" or _gpumod is None:
            return "cpu", "int8", ("forced" if d == "cpu"
                                   else "no gpu probe")
        if d == "cuda":
            try:
                dev, comp, reason = _gpumod.recommend_whisper(self.model_id)
            except Exception:
                return "cpu", "int8", "probe failed"
            if dev == "cuda":
                return dev, comp, "forced (" + reason + ")"
            print(f"[Whisper] forced cuda unusable ({reason}) - using CPU")
            return "cpu", "int8", reason
        try:
            return _gpumod.recommend_whisper(self.model_id)
        except Exception:
            return "cpu", "int8", "probe failed"
    def _ensure_dirs(self):
        for p in [WHISPER_MODELS_ROOT, HF_CACHE]:
            try:
                p.mkdir(parents=True, exist_ok=True)
            except Exception:
                pass
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
            self._load_generation += 1
            generation = self._load_generation
            self._loading = True
            self._ready = False
        def _load():
            try:
                self._ensure_dirs()
                from faster_whisper import WhisperModel
                cpu_threads = self._cpu_threads()
                device, compute, reason = self._resolve_device()
                print(f"[Whisper] device={device} compute={compute} ({reason})")
                last_err = None
                for local_only in (True, False):
                    try:
                        model = WhisperModel(
                            self.model_id,
                            device=device,
                            compute_type=compute,
                            cpu_threads=cpu_threads,
                            download_root=str(WHISPER_MODELS_ROOT),
                            local_files_only=local_only,
                        )
                        break
                    except Exception as e:
                        if device == "cuda":
                            print(f"[Whisper] cuda load failed ({e}) - retrying on CPU")
                            try:
                                model = WhisperModel(
                                    self.model_id,
                                    device="cpu",
                                    compute_type="int8",
                                    cpu_threads=cpu_threads,
                                    download_root=str(WHISPER_MODELS_ROOT),
                                    local_files_only=local_only,
                                )
                                device, compute = "cpu", "int8"
                                reason = f"cuda failed, CPU fallback ({reason})"
                                break
                            except Exception as e2:
                                last_err = e2
                                continue
                        last_err = e
                        continue
                else:
                    raise RuntimeError(f"Whisper load failed: {last_err}")
                with self._lock:
                    self._model = model
                    self._ready = True
                    self._last_error = None
                    self._loading = False
                    self._device_used = device
                    self._compute_used = compute
                    self._device_reason = reason
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
        with self._lock:
            if task in WHISPER_TASKS:
                self.task = task
                if task == "translate":
                    self.target_lang = "en"
            if target_lang:
                self.target_lang = target_lang
            if source_lang:
                self.source_lang = source_lang
            if on_ready:
                on_ready(True, None)
    def switch_model(self, model_id: str,
                     on_ready: Optional[Callable] = None):
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
            self._load_generation += 1
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
    def unload(self) -> bool:
        with self._infer_lock:
            with self._lock:
                if self._model is None:
                    return False
                self._load_generation += 1
                model = self._model
                self._model = None
                self._ready = False
                self._last_error = None
                self._loading = False
        try:
            close = getattr(model, "close", None)
            if callable(close):
                close()
        except Exception:
            pass
        try:
            import gc
            gc.collect()
        except Exception:
            pass
        return True
    def _snapshot_opts(self) -> tuple:
        with self._lock:
            return self.task, self.source_lang, self.target_lang
    def _transcribe_opts(self, task: Optional[str] = None,
                         source_lang: Optional[str] = None) -> dict:
        snap_task, snap_src, _ = self._snapshot_opts()
        eff_task = task if task in WHISPER_TASKS else snap_task
        eff_src = source_lang if source_lang else snap_src
        opts = {
            "beam_size": 5,
            "condition_on_previous_text": False,
            "word_timestamps": False,
        }
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