"""NVIDIA Canary-1B portable wrapper - offline, same interface as TranscriptionEngine."""

import threading
import numpy as np
import tempfile
import os
import soundfile as sf
from pathlib import Path
from typing import Optional, Callable

# Portable cache roots - all inside the app folder
PORTABLE_ROOT = Path(__file__).parent
CANARY_CACHE = PORTABLE_ROOT / "models_cache" / "canary-1b"
HF_CACHE = PORTABLE_ROOT / "models_cache" / "huggingface"
TORCH_CACHE = PORTABLE_ROOT / "models_cache" / "torch"

# Ensure env points to portable caches before importing torch/nemo
os.environ.setdefault("HF_HOME", str(HF_CACHE))
os.environ.setdefault("HUGGINGFACE_HUB_CACHE", str(HF_CACHE))
os.environ.setdefault("TORCH_HOME", str(TORCH_CACHE))
os.environ.setdefault("HF_HUB_OFFLINE", "0")  # allow download when online, offline will use cache

# Canary specific options exposed in GUI.
# Probed 2026-09-03 from the bundled canary-1b.nemo tokenizer: the ONLY
# language tokens present are <|de|> <|en|> <|es|> <|fr|>. There is NO
# <|ja|> / <|zh|> / <|ko|> - requesting them crashes NeMo with
# KeyError: '<|ja|>'. (The GUI Src menu still lists more because it is
# shared with Whisper; unsupported choices now fail with a clear error
# pointing at Whisper instead of exploding inside NeMo.)
CANARY_TASKS = ["transcribe", "translate"]  # transcribe=ASR, translate=AST
CANARY_SOURCE_LANGS = ["auto", "en", "de", "es", "fr"]
CANARY_TARGET_LANGS = {
    "transcribe": ["en", "de", "es", "fr"],
    "translate": ["en"],  # Canary-1B translates to en only
}


def _canary_dir_size(path) -> int:
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


def canary_cache_info() -> dict:
    """{"nemo": bytes} for the local canary-1b.nemo (>100MB counts as a real
    model; the shared huggingface/torch caches are deliberately untouched -
    Whisper uses them too)."""
    try:
        f = CANARY_CACHE / "canary-1b.nemo"
        if f.exists() and f.stat().st_size > 100_000_000:
            return {"nemo": f.stat().st_size}
    except Exception:
        pass
    return {"nemo": 0}


def delete_canary_model() -> int:
    """Remove local canary-1b.nemo file(s). Returns bytes freed."""
    import glob as _glob
    freed = 0
    try:
        for pat in ("canary-1b.nemo", "*.nemo"):
            for hit in _glob.glob(str(CANARY_CACHE / pat)):
                try:
                    freed += Path(hit).stat().st_size
                    os.unlink(hit)
                except Exception:
                    pass
            if freed:
                break
    except Exception:
        pass
    if not freed:
        raise FileNotFoundError("no local canary-1b.nemo to delete")
    return freed

class CanaryEngine:
    def __init__(self, task: str = "transcribe", source_lang: str = "ja", target_lang: str = "en", device: str = "auto", on_ready: Optional[Callable] = None):
        self.task = task if task in CANARY_TASKS else "transcribe"
        self.source_lang = source_lang
        self.target_lang = target_lang
        # "auto" = VRAM-gated CUDA, "cpu" = never, "cuda" = try (explicit
        # choice; OOM still fails cleanly instead of hanging).
        self.device = (device or "auto").strip().lower() or "auto"
        if self.device == "gpu":
            self.device = "cuda"  # GUI/config naming vs engine naming
        if self.device not in ("auto", "cpu", "cuda"):
            self.device = "auto"
        self._model = None
        self._ready = False
        self._on_ready = on_ready
        self._lock = threading.Lock()
        # Serializes model inference: live transcribe() and SRT
        # transcribe_file() must never run concurrently (NeMo buffers).
        self._infer_lock = threading.Lock()
        self._loading = False
        self._last_error: Optional[str] = None
        self._model_name = "nvidia/canary-1b"
        self._device_used = "cpu"
        # Populated at load() by probing model.tokenizer.special_tokens.
        # None = unknown (not loaded, or detection failed) -> skip validation.
        self.supported_source_langs = None

    @property
    def is_ready(self) -> bool:
        return self._ready

    @property
    def device_info(self) -> str:
        try:
            return str(self._device_used or "cpu")
        except Exception:
            return "cpu"

    @property
    def current_arch_name(self) -> str:
        return f"canary-1b ({self.task} {self.source_lang}->{self.target_lang})"

    def _ensure_dirs(self):
        for p in [CANARY_CACHE, HF_CACHE, TORCH_CACHE]:
            try:
                p.mkdir(parents=True, exist_ok=True)
            except Exception:
                pass

    def load(self, on_progress: Optional[Callable] = None):
        with self._lock:
            if self._loading:
                return
            self._loading = True
            self._ready = False

        def _load():
            try:
                self._ensure_dirs()
                # Lazy imports - only when needed so Moonshine still works without torch
                import torch
                # Force CPU for portable (no CUDA needed, works offline)
                # Check if nemo is available, otherwise try transformers pipeline fallback
                try:
                    import nemo.collections.asr as nemo_asr
                    # 1) Prefer manual local file: models_cache/canary-1b/canary-1b.nemo
                    #    (for slow connections - download via browser, drop here, works offline)
                    local_nemo = CANARY_CACHE / "canary-1b.nemo"
                    if local_nemo.exists() and local_nemo.stat().st_size > 100_000_000:
                        self._model = nemo_asr.models.ASRModel.restore_from(
                            restore_path=str(local_nemo), map_location="cpu"
                        )
                    else:
                        # 2) Otherwise from_pretrained will download to HF_CACHE if not present
                        # Use portable cache via HF_HOME already set
                        self._model = nemo_asr.models.ASRModel.from_pretrained(
                            model_name=self._model_name
                        )
                    # Eval first; device policy: explicit cpu never touches
                    # CUDA; explicit cuda tries it (guarded); auto uses the
                    # VRAM gate (fp32-hungry: blind .cuda() OOM-crashes
                    # small cards mid-load).
                    self._model.eval()
                    _want = (getattr(self, "device", "auto") or "auto")
                    try:
                        import gpu as _gpumod
                        if _want == "cuda":
                            _use_cuda = True
                            _reason = "forced by compute setting"
                            try:
                                _tc = _gpumod.torch_cuda()
                                if not _tc.get("ok"):
                                    _use_cuda = False
                                    _reason = f"forced cuda unusable ({_tc.get('reason', '?')})"
                            except Exception:
                                _use_cuda, _reason = False, "probe failed"
                        elif _want == "cpu":
                            _use_cuda, _reason = False, "forced by compute setting"
                        else:
                            _use_cuda, _reason = _gpumod.recommend_canary()
                    except Exception:
                        _use_cuda, _reason = False, "no gpu probe"
                    try:
                        _has_cuda = bool(torch.cuda.is_available())
                    except Exception:
                        _has_cuda = False
                    self._device_used = "cpu"
                    if _use_cuda and _has_cuda:
                        try:
                            self._model.cuda()
                            self._device_used = "cuda"
                            print(f"[Canary] using CUDA ({_reason})")
                        except Exception as e_cuda:
                            print(f"[Canary] cuda() failed ({e_cuda}) - staying on CPU")
                    else:
                        print(f"[Canary] using CPU ({_reason})")
                except Exception as e_nemo:
                    # Fallback: try transformers pipeline for canary (if nemo not installed)
                    # This keeps offline working if nemo wheels not yet installed but transformers is
                    raise RuntimeError(f"NeMo load failed: {e_nemo}. Install nemo_toolkit[asr] offline via wheels\\")
                with self._lock:
                    self._ready = True
                    self._last_error = None
                    self._loading = False
                    # Probe the real supported languages from the tokenizer so
                    # unsupported requests fail with a clear message instead of
                    # KeyError: '<|ja|>' deep inside NeMo.
                    try:
                        self.supported_source_langs = self._detect_supported_langs()
                    except Exception:
                        self.supported_source_langs = None
                if self._on_ready:
                    self._on_ready(True, None)
            except Exception as e:
                import traceback
                traceback.print_exc()
                with self._lock:
                    self._ready = False
                    self._last_error = str(e)
                    self._loading = False
                if self._on_ready:
                    self._on_ready(False, str(e))

        threading.Thread(target=_load, daemon=True).start()

    def switch_options(self, task: Optional[str] = None, source_lang: Optional[str] = None, target_lang: Optional[str] = None, on_ready: Optional[Callable] = None):
        """Update task/lang without reloading weights (Canary is multitask).

        Thread-safe: inference calls snapshot options under the same
        lock, so a GUI change can never tear a read mid-inference.
        """
        with self._lock:
            if task in CANARY_TASKS:
                self.task = task
                # translate only supports en output
                if task == "translate":
                    self.target_lang = "en"
                elif target_lang:
                    self.target_lang = target_lang
            elif target_lang and self.task == "transcribe":
                self.target_lang = target_lang
            if source_lang:
                self.source_lang = source_lang
        if on_ready:
            # Already ready, just callback
            on_ready(True, None)
        # No reload needed - same weights handle both tasks

    def _snapshot_opts(self) -> tuple:
        """Atomic (task, source_lang, target_lang) snapshot for one inference call."""
        with self._lock:
            return self.task, self.source_lang, self.target_lang

    def unload(self) -> bool:
        """Drop weights to reclaim RAM (multi-GB) when another engine takes
        over. Refuses while a load is in flight. Returns True when freed."""
        with self._lock:
            if self._loading or self._model is None:
                return False
            self._model = None
            self._ready = False
            self._last_error = None
            self.supported_source_langs = None
        try:
            import gc
            gc.collect()
        except Exception:
            pass
        try:
            import torch
            try:
                torch.cuda.empty_cache()
            except Exception:
                pass
        except Exception:
            pass
        return True

    def _detect_supported_langs(self):
        """Read <|xx|> language tokens from the loaded tokenizer.

        Returns a set like {'en','de','es','fr'}, or None if undetectable.
        Called with _lock held at load(); result is read-only afterwards.
        """
        model = self._model
        if model is None:
            return None
        tok = getattr(model, "tokenizer", None)
        if tok is None:
            return None
        st = getattr(tok, "special_tokens", None) or {}
        langs = {k[2:-2] for k in st.keys()
                 if isinstance(k, str) and len(k) == 6
                 and k.startswith("<|") and k.endswith("|>")
                 and k[2:-2].isalpha()}
        return langs or None

    @staticmethod
    def _check_lang_support(supported, task, source_lang, target_lang):
        """Return an actionable error string for unsupported langs, else None.

        Static + pure so it is unit-testable without a model. `supported`
        None means unknown -> allow (NeMo decides).
        """
        if not supported:
            return None
        eff_task = task if task in CANARY_TASKS else "transcribe"
        bad = []
        if (source_lang or "auto") != "auto" and source_lang not in supported:
            bad.append("source language '%s'" % source_lang)
        eff_tgt = "en" if eff_task == "translate" else (target_lang or "en")
        if eff_tgt != "auto" and eff_tgt not in supported:
            bad.append("target language '%s'" % eff_tgt)
        if not bad:
            return None
        return ("Canary-1B supports only [%s]; %s not supported. "
                "For Japanese/Chinese/Korean use Whisper Large v3."
                % (", ".join(sorted(supported)), " and ".join(bad)))

    def transcribe(self, audio_data: np.ndarray, sample_rate: int = 16000) -> str:
        with self._lock:
            if not self._ready or self._model is None:
                return ""
            supported = self.supported_source_langs

        # Fail before touching NeMo: unsupported langs crash it with
        # KeyError: '<|xx|>' plus a Windows temp-file cleanup cascade.
        # Single snapshot used for BOTH validation and the call below, so a
        # GUI switch mid-call cannot slip an unvalidated combo through.
        eff_task, eff_src, eff_tgt = self._snapshot_opts()
        err = self._check_lang_support(supported, eff_task, eff_src, eff_tgt)
        if err:
            return "[Canary Error: %s]" % err

        try:
            # Normalize to float32 [-1,1] like Moonshine
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

            # Canary expects 16kHz mono, write temp wav for nemo transcribe
            # Use soundfile to write
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tf:
                tmp_path = tf.name
            try:
                sf.write(tmp_path, audio_float, samplerate=sample_rate)
                # Call nemo transcribe
                # Canary API: model.transcribe([path], source_lang, target_lang, task, batch_size)
                # For translate, task='ast' or 'translate' depending on version.
                # Uses the validated snapshot above (not a fresh read).
                task_arg = "ast" if eff_task == "translate" else "asr"
                # Older nemo uses pnc for source_lang auto
                src = eff_src if eff_src != "auto" else None
                tgt = eff_tgt
                with self._lock:
                    model = self._model
                if model is None:
                    # Unloaded (engine switch) between check and call.
                    return ""
                # Try new API first (serialized: see _infer_lock)
                try:
                    # NeMo Canary 1.1B API (verbose=False keeps the per-chunk
                    # tqdm bar off stdout; the GUI gets progress via callbacks)
                    with self._infer_lock:
                        results = model.transcribe(
                            [tmp_path],
                            batch_size=1,
                            source_lang=src or "en",
                            target_lang=tgt,
                            task=task_arg,
                            pnc="yes",
                            verbose=False,
                        )
                except TypeError:
                    # Fallback older API
                    with self._infer_lock:
                        results = model.transcribe([tmp_path])
                # Results is list of strings or list of Transcription
                if results and len(results) > 0:
                    r = results[0]
                    if isinstance(r, str):
                        return r.strip()
                    # Sometimes returns object with .text
                    if hasattr(r, "text"):
                        return r.text.strip()
                    return str(r).strip()
                return ""
            finally:
                try:
                    os.unlink(tmp_path)
                except Exception:
                    pass
        except Exception as e:
            if isinstance(e, KeyError) and str(e).startswith("'<|"):
                return ("[Canary Error: language %s not supported by this "
                        "model. For Japanese/Chinese/Korean use Whisper Large v3.]"
                        % str(e).strip("'"))
            import traceback
            traceback.print_exc()
            return f"[Canary Error: {e}]"

    def transcribe_file(self, wav_path: str, task: Optional[str] = None,
                        source_lang: Optional[str] = None,
                        target_lang: Optional[str] = None) -> str:
        """Transcribe an existing wav file path (used by SRT jobs - no temp copy).

        Optional per-call overrides let SRT jobs use their own languages
        without mutating the shared live engine (no live/SRT race).
        """
        with self._lock:
            if not self._ready or self._model is None:
                return ""
            model = self._model
            snap_task, snap_src, snap_tgt = self.task, self.source_lang, self.target_lang
            supported = self.supported_source_langs
        try:
            eff_task = task if task in CANARY_TASKS else snap_task
            eff_src = source_lang if source_lang else snap_src
            eff_tgt = target_lang if target_lang else snap_tgt
            if eff_task == "translate":
                eff_tgt = "en"  # Canary translate only supports en output
            # Fail before touching NeMo (see transcribe()).
            err = self._check_lang_support(supported, eff_task, eff_src, eff_tgt)
            if err:
                return "[Canary Error: %s]" % err
            task_arg = "ast" if eff_task == "translate" else "asr"
            src = eff_src if eff_src != "auto" else None
            tgt = eff_tgt
            with self._lock:
                # Re-check liveness: unload() may have run since entry.
                if not self._ready or self._model is None:
                    return ""
                model = self._model
            try:
                with self._infer_lock:
                    results = model.transcribe(
                        [wav_path],
                        batch_size=1,
                        source_lang=src or "en",
                        target_lang=tgt,
                        task=task_arg,
                        pnc="yes",
                        verbose=False,
                    )
            except TypeError:
                with self._infer_lock:
                    results = model.transcribe([wav_path])
            if results and len(results) > 0:
                r = results[0]
                if isinstance(r, str):
                    return r.strip()
                if hasattr(r, "text"):
                    return r.text.strip()
                return str(r).strip()
            return ""
        except Exception as e:
            # Known NeMo failure (unsupported <|xx|> prompt token): report
            # cleanly without the multi-page traceback + Windows temp-file
            # PermissionError cascade that follows it.
            if isinstance(e, KeyError) and str(e).startswith("'<|"):
                return ("[Canary Error: language %s not supported by this "
                        "model. For Japanese/Chinese/Korean use Whisper Large v3.]"
                        % str(e).strip("'"))
            import traceback
            traceback.print_exc()
            return f"[Canary Error: {e}]"
