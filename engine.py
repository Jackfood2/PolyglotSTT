# engine.py
import os
import threading
import numpy as np
from pathlib import Path
from typing import Optional, Callable
PORTABLE_CACHE_ROOT = Path(__file__).parent / "models_cache"
def _migrate_cache_to_portable(src_path: str, portable_root: Path):
    if not src_path or portable_root is None:
        return
    try:
        import shutil
        from pathlib import Path as _P
        src = _P(src_path)
        parts = src.parts
        try:
            idx = parts.index("download.moonshine.ai")
        except ValueError:
            return
        rel = _P(*parts[idx:])
        dst = portable_root / rel
        if dst.exists():
            return
        dst.parent.mkdir(parents=True, exist_ok=True)
        if src.is_dir():
            shutil.copytree(src, dst, dirs_exist_ok=True)
        elif src.is_file():
            shutil.copy2(src, dst)
    except Exception:
        pass
MODEL_ARCH_NAMES = {
    0: "tiny",
    1: "base",
    2: "tiny-streaming",
    3: "base-streaming",
    4: "small-streaming",
    5: "medium-streaming",
}
def _dir_size_bytes(path) -> int:
    try:
        total = 0
        for root, _dirs, files in os.walk(str(path)):
            for fn in files:
                try:
                    total += os.path.getsize(os.path.join(root, fn))
                except Exception:
                    pass
        return total
    except Exception:
        return 0
def moonshine_model_dir(cache_root=None, arch: int = 5):
    from pathlib import Path as _P
    root = _P(cache_root) if cache_root else PORTABLE_CACHE_ROOT
    name = MODEL_ARCH_NAMES.get(int(arch), str(arch))
    return root / "download.moonshine.ai" / "model" / f"{name}-en"
def moonshine_downloaded_map(cache_root=None) -> dict:
    out = {}
    for arch in MODEL_ARCH_NAMES:
        try:
            d = moonshine_model_dir(cache_root, arch)
            if d.exists() and _dir_size_bytes(d) > 1_000_000:
                out[int(arch)] = True
        except Exception:
            pass
    return out
def moonshine_cache_info(cache_root=None) -> dict:
    out = {}
    for arch in MODEL_ARCH_NAMES:
        try:
            d = moonshine_model_dir(cache_root, arch)
            out[int(arch)] = _dir_size_bytes(d) if d.exists() else 0
        except Exception:
            out[int(arch)] = 0
    return out
def delete_moonshine_model(cache_root=None, arch: int = 5) -> int:
    import shutil
    d = moonshine_model_dir(cache_root, arch)
    if not d.exists():
        raise FileNotFoundError(f"not downloaded: {d.name}")
    freed = _dir_size_bytes(d)
    shutil.rmtree(str(d), ignore_errors=False)
    return freed
class TranscriptionEngine:
    def __init__(self, language: str = "en", model_arch: Optional[int] = None, on_ready: Optional[Callable] = None):
        self.language = language
        self._wanted_arch = model_arch
        self._transcriber = None
        self._model_path = None
        self._model_arch = None
        self._ready = False
        self._base_ready = on_ready
        self._switch_cb: Optional[Callable] = None
        self._lock = threading.Lock()
        self._tx_lock = threading.Lock()
        self._loading = False
        self._last_error: Optional[str] = None
    @property
    def is_ready(self) -> bool:
        return self._ready
    @property
    def current_arch(self) -> Optional[int]:
        if self._model_arch is not None:
            try:
                return self._model_arch.value
            except Exception:
                return int(self._model_arch)
        return self._wanted_arch
    @property
    def current_arch_name(self) -> str:
        arch = self.current_arch
        return MODEL_ARCH_NAMES.get(arch, str(arch) if arch is not None else "default")
    def load(self, on_progress: Optional[Callable] = None):
        with self._lock:
            if self._loading:
                return
            self._loading = True
            self._ready = False
            wanted_arch = self._wanted_arch
        def _load():
            try:
                from moonshine_voice import get_model_for_language, Transcriber
                from moonshine_voice.moonshine_api import ModelArch
                wanted = None
                if wanted_arch is not None:
                    try:
                        wanted = ModelArch(wanted_arch)
                    except Exception:
                        wanted = None
                cache_root = PORTABLE_CACHE_ROOT
                try:
                    cache_root.mkdir(parents=True, exist_ok=True)
                except Exception:
                    cache_root = None
                try:
                    if cache_root is not None:
                        self._model_path, self._model_arch = get_model_for_language(
                            self.language, wanted, cache_root=cache_root
                        )
                    else:
                        raise RuntimeError("no portable cache")
                except Exception as e:
                    try:
                        from moonshine_voice.download import get_model_for_language as _g
                        self._model_path, self._model_arch = get_model_for_language(
                            self.language, wanted
                        )
                        try:
                            threading.Thread(target=_migrate_cache_to_portable, args=(self._model_path, cache_root), daemon=True).start()
                        except Exception:
                            pass
                    except Exception:
                        raise e
                with self._tx_lock:
                    old = self._transcriber
                    if old is not None:
                        try:
                            old.close()
                        except Exception:
                            pass
                    self._transcriber = Transcriber(
                        model_path=self._model_path,
                        model_arch=self._model_arch,
                    )
                with self._lock:
                    self._ready = True
                    self._last_error = None
                    one_shot = self._switch_cb
                    self._switch_cb = None
                    try:
                        loaded_val = self._model_arch.value
                    except Exception:
                        try:
                            loaded_val = int(self._model_arch)
                        except Exception:
                            loaded_val = None
                    wanted_now = self._wanted_arch
                    self._loading = False
                try:
                    stale = (wanted_now is not None and loaded_val is not None
                             and int(wanted_now) != int(loaded_val))
                except Exception:
                    stale = False
                if stale:
                    # Fixed: restart the load outside the lock to avoid deadlock.
                    with self._lock:
                        if self._switch_cb is None:
                            self._switch_cb = one_shot
                            one_shot = None
                    self.load()
                    return
                if self._base_ready:
                    self._base_ready(True, None)
                if one_shot:
                    one_shot(True, None)
            except Exception as e:
                import traceback
                traceback.print_exc()
                self._loading = False
                with self._lock:
                    self._ready = False
                    self._last_error = str(e)
                    one_shot = self._switch_cb
                    self._switch_cb = None
                if self._base_ready:
                    self._base_ready(False, str(e))
                if one_shot:
                    one_shot(False, str(e))
        threading.Thread(target=_load, daemon=True).start()
    def switch_model(self, new_arch: Optional[int], on_ready: Optional[Callable] = None):
        with self._lock:
            self._wanted_arch = new_arch
            self._switch_cb = on_ready
        self.load()
    def unload(self) -> bool:
        with self._lock:
            if self._loading or self._transcriber is None:
                return False
            tr = self._transcriber
            self._transcriber = None
            self._ready = False
            self._last_error = None
        with self._tx_lock:
            try:
                tr.close()
            except Exception:
                pass
        return True
    def transcribe(self, audio_data: np.ndarray, sample_rate: int = 16000) -> str:
        with self._lock:
            if not self._ready or self._transcriber is None:
                return ""
            tr = self._transcriber
        try:
            arr = np.asarray(audio_data).flatten()
            if arr.dtype == np.int16:
                audio_float = arr.astype(np.float32) / 32768.0
            elif arr.dtype == np.int32:
                audio_float = arr.astype(np.float32) / 2147483648.0
            else:
                audio_float = arr.astype(np.float32)
            if np.abs(audio_float).max() > 1.5:
                audio_float = audio_float / 32768.0
            audio_float = np.clip(audio_float, -1.0, 1.0)
            audio_list = audio_float.tolist()
            with self._tx_lock:
                transcript = tr.transcribe_without_streaming(
                    audio_list, sample_rate=sample_rate
                )
            if not transcript or not transcript.lines:
                return ""
            texts = [line.text.strip() for line in transcript.lines if line.text and line.text.strip()]
            return " ".join(texts).strip()
        except Exception as e:
            import traceback
            traceback.print_exc()
            return f"[Error: {e}]"