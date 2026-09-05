"""Moonshine v2 transcription engine wrapper - offline file API with model selection."""

import os
import threading
import numpy as np
from pathlib import Path
from typing import Optional, Callable

# Portable cache - models stored inside the app folder so zipping includes them for offline use
# Falls back to default cache if portable folder not writable
PORTABLE_CACHE_ROOT = Path(__file__).parent / "models_cache"


def _migrate_cache_to_portable(src_path: str, portable_root: Path):
    """Best-effort copy of already-downloaded default-cache model to portable folder for offline zip."""
    if not src_path or portable_root is None:
        return
    try:
        import shutil
        from pathlib import Path as _P
        src = _P(src_path)
        # src is like .../Cache/download.moonshine.ai/model/...  -> we want portable_root/download.moonshine.ai/...
        # Find the 'download.moonshine.ai' segment
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

# Friendly map - matches catalog model_arch values
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
    """On-disk dir for one arch, e.g. models_cache/download.moonshine.ai/model/medium-streaming-en."""
    from pathlib import Path as _P
    root = _P(cache_root) if cache_root else PORTABLE_CACHE_ROOT
    name = MODEL_ARCH_NAMES.get(int(arch), str(arch))
    return root / "download.moonshine.ai" / "model" / f"{name}-en"


def moonshine_downloaded_map(cache_root=None) -> dict:
    """{arch: True} for archs actually cached (ignores stray dirs)."""
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
    """{arch: bytes} for cached archs (0 when absent)."""
    out = {}
    for arch in MODEL_ARCH_NAMES:
        try:
            d = moonshine_model_dir(cache_root, arch)
            out[int(arch)] = _dir_size_bytes(d) if d.exists() else 0
        except Exception:
            out[int(arch)] = 0
    return out


def delete_moonshine_model(cache_root=None, arch: int = 5) -> int:
    """Remove one cached arch dir. Returns bytes freed; raises when absent."""
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
        self._wanted_arch = model_arch  # None = default (medium-streaming for en)
        self._transcriber = None
        self._model_path = None
        self._model_arch = None
        self._ready = False
        self._base_ready = on_ready  # stable app callback - never wrapped
        self._switch_cb: Optional[Callable] = None  # one-shot per switch_model
        self._lock = threading.Lock()
        self._tx_lock = threading.Lock()  # held during C transcribe AND close
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
        """Load the Moonshine v2 model in a background thread. Respects self._wanted_arch."""
        with self._lock:
            if self._loading:
                return
            self._loading = True
            # Mark not ready during reload
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
                # Try portable cache first (for offline zip), fallback to default if it fails
                # This makes the folder self-contained: models_cache/ is included when you zip the app
                cache_root = PORTABLE_CACHE_ROOT
                try:
                    cache_root.mkdir(parents=True, exist_ok=True)
                except Exception:
                    cache_root = None
                # Portable cache attempt
                try:
                    if cache_root is not None:
                        self._model_path, self._model_arch = get_model_for_language(
                            self.language, wanted, cache_root=cache_root
                        )
                    else:
                        raise RuntimeError("no portable cache")
                except Exception as e:
                    # Fallback: try default cache (e.g., if portable download failed or on first run before migration)
                    # Also try to migrate existing APPDATA cache to portable for future offline use
                    try:
                        from moonshine_voice.download import get_model_for_language as _g
                        # Try default cache download
                        self._model_path, self._model_arch = get_model_for_language(
                            self.language, wanted
                        )
                        # Best-effort: copy default cache to portable cache in background for next offline zip
                        try:
                            threading.Thread(target=_migrate_cache_to_portable, args=(self._model_path, cache_root), daemon=True).start()
                        except Exception:
                            pass
                    except Exception:
                        raise e
                # Close old + create new under _tx_lock: an in-flight
                # transcribe() holds the same lock, so close can never race it.
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
                # A newer switch_model() may have landed mid-load (load()
                # early-returns while _loading): chain one more load so we end
                # on the WANTED arch instead of reporting the stale one ready.
                # Terminates: each chain consumes the newest request.
                try:
                    stale = (wanted_now is not None and loaded_val is not None
                             and int(wanted_now) != int(loaded_val))
                except Exception:
                    stale = False
                if stale:
                    with self._lock:
                        if self._switch_cb is None:
                            self._switch_cb = one_shot
                            one_shot = None
                    self.load()
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
        """Switch to a different model size. new_arch None = default."""
        with self._lock:
            self._wanted_arch = new_arch
            # One-shot only: replaces any pending one, never chains.
            # (The old code wrapped _on_ready per call - unbounded growth.)
            self._switch_cb = on_ready
        self.load()

    def unload(self) -> bool:
        """Drop the ORT sessions to reclaim RAM when another engine takes
        over. Refuses while a load is in flight (the loader thread would
        resurrect the model after) and never touches the live-transcribe
        path except via _ready (in-flight transcribe() holds its own local
        ref and finishes; new calls see not-ready and skip cleanly).
        Returns True when native resources were released."""
        with self._lock:
            if self._loading or self._transcriber is None:
                return False
            tr = self._transcriber
            self._transcriber = None
            self._ready = False
            self._last_error = None
        # Same lock order as transcribe() (_lock released first, then
        # _tx_lock): close can never race an in-flight inference, and no
        # ABBA deadlock with the loader (it only clears _loading after its
        # last lock use, so _loading==False means no lock held against us).
        with self._tx_lock:
            try:
                tr.close()
            except Exception:
                pass
        return True

    def transcribe(self, audio_data: np.ndarray, sample_rate: int = 16000) -> str:
        """Transcribe audio data and return text. Thread-safe."""
        with self._lock:
            if not self._ready or self._transcriber is None:
                return ""
            tr = self._transcriber  # local ref: switch_model closes under _tx_lock

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
            # _tx_lock: switch_model() closes `tr` only while holding this.
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
