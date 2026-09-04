"""MoonshineSTT - Hold F2 to record, release to transcribe and paste (focus-aware)."""

import os
import sys
import time
import threading
import queue
import json

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from pynput import keyboard
from engine import TranscriptionEngine
from recorder import AudioRecorder
from input_sim import (
    insert_text,
    set_clipboard_text,
    get_foreground_window,
    get_focused_child,
    get_window_title,
    user32,
    GA_ROOT,
)

try:
    from gui import MoonshineGUI, SUCCESS, DANGER, WARNING, FG_SECONDARY
except Exception as e:
    print(f"GUI not available: {e}")
    MoonshineGUI = None
    SUCCESS = "#00B894"
    DANGER = "#E17055"
    WARNING = "#FDCB6E"
    FG_SECONDARY = "#B2BEC3"

RECORD_KEY = keyboard.Key.f2
SAMPLE_RATE = 16000
CONFIG_PATH = os.path.join(os.path.dirname(__file__), "moonshine_config.json")
_CONFIG_LOCK = threading.RLock()  # reentrant: callers may hold it while save_local_config() re-acquires

DEFAULT_CONFIG = {
    "typing_method": "clipboard",   # clipboard | unicode
    "suffix": "none",               # none | space | newline | period_space
    "typing_delay_ms": 0,
    "model_arch": 5,                # 0=tiny,1=base,2=tiny-streaming,4=small-streaming,5=medium-streaming (default/best)
    "engine": "Moonshine v2",       # Moonshine v2 | Canary-1B | Whisper Large v3
    "canary_task": "transcribe",    # transcribe | translate (to EN)
    "canary_src_lang": "auto",      # canary-1b tokenizer only has de/en/es/fr (+auto->en)
    "whisper_task": "translate",    # transcribe | translate (->en, Whisper only outputs EN on translate)
    "whisper_src_lang": "ja",       # auto + whisper langs (ja/zh/ko/en/...)
    "whisper_model": "large-v3",    # tiny | base | small | medium | large-v3 (downloaded on demand)
    "srt_cpu": 0,                   # 0 = auto (80% of cores), else thread count
    "srt_out_dir": "",              # "" = same folder as source file
    "srt_input_lang": "ja",         # SRT input language code (auto/en/ja/zh/ko/...)
    "srt_output_lang": "en",        # SRT output language code (en/ja/zh/ko/...)
    "burn_font_size": 18,           # burned subtitle size (ASS units, 10-40)
}

def load_local_config():
    try:
        if os.path.exists(CONFIG_PATH):
            with open(CONFIG_PATH, "r", encoding="utf-8") as f:
                cfg = json.load(f)
                out = dict(DEFAULT_CONFIG)
                out.update(cfg)
                # Auto-migrate file if missing keys (ensures all settings persist after updates)
                if any(k not in cfg for k in DEFAULT_CONFIG):
                    try:
                        with open(CONFIG_PATH, "w", encoding="utf-8") as wf:
                            json.dump(out, wf, indent=2)
                    except Exception:
                        pass
                return out
    except Exception:
        pass
    return dict(DEFAULT_CONFIG)

def save_local_config(cfg):
    # Serialized + atomic (tmp write + os.replace) so a concurrent save
    # or a crash mid-write can never leave a half-written JSON file.
    try:
        with _CONFIG_LOCK:
            tmp = CONFIG_PATH + ".tmp"
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump(cfg, f, indent=2)
            os.replace(tmp, CONFIG_PATH)
    except Exception:
        pass

def apply_suffix(text: str, suffix: str) -> str:
    if suffix == "space":
        return text + " "
    if suffix == "newline":
        return text + "\n"
    if suffix == "period_space":
        return text + ". "
    return text

class MoonshineSTTApp:
    def __init__(self):
        self.config = load_local_config()
        # ensure all default keys exist and clamp - ensures persistence across updates
        needs_save = False
        for k, v in DEFAULT_CONFIG.items():
            if k not in self.config:
                self.config[k] = v
                needs_save = True
        try:
            arch = int(self.config.get("model_arch", 5))
            if arch not in (0,1,2,3,4,5):
                arch = 5
                self.config["model_arch"] = 5
                needs_save = True
        except Exception:
            arch = 5
            self.config["model_arch"] = 5
            needs_save = True
        # Clamp engine choices (+ migrate the old "Whisper Turbo v3" name,
        # whose model weights turned out not to translate, to Large v3)
        if self.config.get("engine") == "Whisper Turbo v3":
            self.config["engine"] = "Whisper Large v3"
            needs_save = True
        if self.config.get("engine") not in ("Moonshine v2", "Canary-1B", "Whisper Large v3"):
            self.config["engine"] = "Moonshine v2"
            needs_save = True
        if self.config.get("canary_task") not in ("transcribe", "translate"):
            self.config["canary_task"] = "transcribe"
            needs_save = True
        # Clamp to the model's REAL capability (probed: de/en/es/fr only).
        # Old configs may hold "ja"/"zh" (never worked - crashed NeMo), so
        # migrate those to "auto" rather than preserving a broken value.
        if self.config.get("canary_src_lang") not in ("auto", "en", "de", "es", "fr"):
            self.config["canary_src_lang"] = "auto"
            needs_save = True
        if self.config.get("whisper_task") not in ("transcribe", "translate"):
            self.config["whisper_task"] = "translate"
            needs_save = True
        try:
            from whisper_engine import WHISPER_SOURCE_LANGS, WHISPER_MODEL_CHOICES
            _wlangs = tuple(WHISPER_SOURCE_LANGS)
            _wmodels = tuple(WHISPER_MODEL_CHOICES.values())
        except Exception:
            _wlangs = ("auto", "en", "ja", "zh", "ko", "de", "fr", "es")
            _wmodels = ("tiny", "base", "small", "medium", "large-v3")
        if self.config.get("whisper_src_lang") not in _wlangs:
            self.config["whisper_src_lang"] = "ja"
            needs_save = True
        if self.config.get("whisper_model") not in _wmodels:
            self.config["whisper_model"] = "large-v3"
            needs_save = True
        # Validate SRT language codes
        try:
            from gui import SRT_LANGS
            _srt_langs = tuple(SRT_LANGS)
        except Exception:
            _srt_langs = ("auto", "en", "ja", "zh", "ko", "de", "fr", "es")
        if self.config.get("srt_input_lang") not in _srt_langs:
            self.config["srt_input_lang"] = "ja"
            needs_save = True
        if self.config.get("srt_output_lang") not in _srt_langs:
            self.config["srt_output_lang"] = "en"
            needs_save = True
        try:
            _bf = int(self.config.get("burn_font_size", 18))
        except Exception:
            _bf = 18
        if _bf < 10 or _bf > 40:
            _bf = 18
            needs_save = True
        if self.config.get("burn_font_size") != _bf:
            self.config["burn_font_size"] = _bf
            needs_save = True
        if needs_save:
            save_local_config(self.config)

        # Three engines - Moonshine lightweight, Canary + Whisper heavy (lazy)
        # Locks first: _get_*_engine() (called just below) needs _eng_lock.
        self._eng_lock = threading.Lock()  # guards lazy heavy-engine creation
        self.moonshine_engine = TranscriptionEngine(language="en", model_arch=arch, on_ready=self._model_ready)
        self.canary_engine = None  # created on demand
        self.whisper_engine = None  # created on demand (large-v3)
        # Active engine pointer
        if self.config.get("engine") == "Canary-1B":
            self.engine = self._get_canary_engine()
        elif self.config.get("engine") == "Whisper Large v3":
            self.engine = self._get_whisper_engine()
        else:
            self.engine = self.moonshine_engine

        self.recorder = AudioRecorder(sample_rate=SAMPLE_RATE)
        self._recording = False
        # True while a model is being swapped (record/F2 refused: inference
        # would return empty on a half-loaded engine).
        self._model_switching = False
        # Serializes the recorder start/stop sequence: a fast F2 hammer
        # could otherwise start() while the old stream is still closing
        # ("Device unavailable") or double-start two streams.
        self._rec_lock = threading.Lock()
        self._key_pressed = False
        self._gui_queue: queue.Queue = queue.Queue()
        self.gui = None
        self._our_root_hwnd = 0
        self._target_top_hwnd = None
        self._target_child_hwnd = None

        # Queue system - ported from test.py audio_queue / processing_thread
        self.audio_queue: queue.Queue = queue.Queue()
        self.currently_processing = False
        self._processing_thread = threading.Thread(target=self._processing_worker, daemon=True)
        self._processing_thread.start()

        # SRT file jobs (separate serial worker so live dictation keeps working)
        self._srt_busy = False
        self._srt_lock = threading.Lock()  # guards _srt_busy check-and-set across threads
        self._srt_thread = None  # tracked so a dead job can never wedge "busy" forever
        self._srt_cancel = threading.Event()
        # (self._eng_lock is created earlier, before engine init)

        if MoonshineGUI is not None:
            try:
                self.gui = MoonshineGUI()
                self.gui.set_record_callback(self._gui_record_start, self._gui_record_stop)
                self.gui.set_settings(
                    self.config.get("typing_method", "clipboard"),
                    self.config.get("suffix", "none"),
                    self._on_method_changed,
                    self._on_suffix_changed,
                )
                try:
                    self._refresh_model_row()
                except Exception:
                    pass
                # Engine selector + Canary/Whisper options (shared Task/Src widgets)
                try:
                    _eng = self.config.get("engine", "Moonshine v2")
                    if _eng == "Whisper Large v3":
                        _t, _s = (self.config.get("whisper_task", "translate"),
                                  self.config.get("whisper_src_lang", "ja"))
                    else:
                        _t, _s = (self.config.get("canary_task", "transcribe"),
                                  self.config.get("canary_src_lang", "auto"))
                    self.gui.set_engine(
                        _eng, _t, _s,
                        self._on_engine_changed,
                        self._on_canary_task_changed,
                        self._on_canary_lang_changed,
                    )
                except Exception:
                    pass
                try:
                    self.gui.set_srt_callbacks(self._srt_start, self._srt_cancel,
                                               self._burn_start)
                    self.gui.set_srt_preview_callback(self._burn_preview)
                    try:
                        self.gui.set_model_manage_callback(self._open_model_manager)
                    except Exception:
                        pass
                    try:
                        _bfs = int(self.config.get("burn_font_size", 18))
                        self.gui.burn_font_slider.set(_bfs)
                        self.gui._on_burn_fontsize_changed(_bfs)
                    except Exception:
                        pass
                    # SRT language dropdowns (only meaningful for Canary/Whisper)
                    try:
                        self.gui.set_srt_languages(
                            self.config.get("srt_input_lang", "ja"),
                            self.config.get("srt_output_lang", "en"),
                            self._on_srt_input_lang_changed,
                            self._on_srt_output_lang_changed,
                        )
                        self.gui.set_srt_lang_state(self.config.get("engine", "Moonshine v2"))
                        # set_engine() above ran before these widgets were
                        # restored, so coerce once more for the saved engine
                        # (e.g. Canary + saved Japanese -> Auto-detect).
                        try:
                            self.gui.refresh_lang_options(
                                self.config.get("engine", "Moonshine v2"))
                        except Exception:
                            pass
                    except Exception:
                        pass
                    # Restore persisted SRT prefs
                    try:
                        if self.config.get("srt_out_dir"):
                            self.gui.srt_out_entry.delete(0, "end")
                            self.gui.srt_out_entry.insert(0, self.config.get("srt_out_dir"))
                    except Exception:
                        pass
                    try:
                        import srt as _srtmod
                        cpu = int(self.config.get("srt_cpu", 0) or 0)
                        if cpu > 0:
                            cpu = max(1, min(cpu, _srtmod.cpu_count()))
                            self.gui.srt_cpu_slider.set(cpu)
                            self.gui._on_srt_cpu_changed(cpu)
                    except Exception:
                        pass
                except Exception:
                    pass
                self.gui.protocol("WM_DELETE_WINDOW", self._on_close)
                self.gui.update_idletasks()
                try:
                    our_hwnd = int(self.gui.winfo_id())
                    self._our_root_hwnd = user32.GetAncestor(our_hwnd, GA_ROOT)
                except Exception:
                    self._our_root_hwnd = 0
            except Exception as e:
                print(f"GUI init failed: {e}")
                import traceback; traceback.print_exc()
                self.gui = None

        self.recorder.set_level_callback(self._level_callback)

    def _get_canary_engine(self, apply_live_options: bool = True):
        # Double-checked creation: GUI thread and SRT worker may race here.
        # SRT jobs pass apply_live_options=False so they never mutate the
        # live engine's settings (they use per-call overrides instead).
        if self.canary_engine is None:
            with self._eng_lock:
                if self.canary_engine is None:
                    try:
                        from canary_engine import CanaryEngine
                        self.canary_engine = CanaryEngine(
                            task=self.config.get("canary_task", "transcribe"),
                            source_lang=self.config.get("canary_src_lang", "ja"),
                            target_lang="en",
                            on_ready=self._model_ready,
                        )
                    except Exception as e:
                        print(f"Canary engine failed to create: {e}")
                        import traceback; traceback.print_exc()
                        # Fallback to moonshine
                        return self.moonshine_engine
        if apply_live_options:
            # Update options if changed
            try:
                self.canary_engine.switch_options(
                    task=self.config.get("canary_task"),
                    source_lang=self.config.get("canary_src_lang"),
                )
            except Exception:
                pass
        return self.canary_engine

    def _get_whisper_engine(self, apply_live_options: bool = True):
        if self.whisper_engine is None:
            with self._eng_lock:
                if self.whisper_engine is None:
                    try:
                        from whisper_engine import WhisperEngine
                        self.whisper_engine = WhisperEngine(
                            task=self.config.get("whisper_task", "translate"),
                            source_lang=self.config.get("whisper_src_lang", "ja"),
                            target_lang="en",
                            model_id=self.config.get("whisper_model", "large-v3"),
                            on_ready=self._model_ready,
                        )
                    except Exception as e:
                        print(f"Whisper engine failed to create: {e}")
                        import traceback; traceback.print_exc()
                        return self.moonshine_engine
        if apply_live_options:
            try:
                self.whisper_engine.switch_options(
                    task=self.config.get("whisper_task"),
                    source_lang=self.config.get("whisper_src_lang"),
                )
            except Exception:
                pass
        return self.whisper_engine

    def _on_method_changed(self, value):
        self.config["typing_method"] = value
        save_local_config(self.config)
        self._log(f"Typing method -> {value}")

    def _on_suffix_changed(self, value):
        self.config["suffix"] = value
        save_local_config(self.config)
        self._log(f"Suffix -> {value}")

    def _model_status_fn(self):
        """Badge provider for the Model menu: '✓' downloaded, '↓' missing.
        Computed once per refresh (dir walk, no model loads)."""
        try:
            eng = self.config.get("engine", "Moonshine v2")
            if eng == "Whisper Large v3":
                from whisper_engine import (WHISPER_MODEL_CHOICES,
                                            whisper_downloaded_map)
                try:
                    dl = whisper_downloaded_map()
                except Exception:
                    dl = {}
                rev = {v: k for k, v in WHISPER_MODEL_CHOICES.items()}

                def _fn(base_label, _rev=rev, _dl=dl):
                    try:
                        mid = _rev.get(base_label)
                        if not mid:
                            return ""
                        return "✓" if _dl.get(mid) else "↓"
                    except Exception:
                        return ""
                return _fn
            if eng == "Canary-1B":
                return None
            from gui import MODEL_CHOICES
            from engine import moonshine_downloaded_map
            try:
                dl = moonshine_downloaded_map()
            except Exception:
                dl = {}
            rev = {v: k for k, v in MODEL_CHOICES.items()}

            def _fn2(base_label, _rev=rev, _dl=dl):
                try:
                    arch = _rev.get(base_label)
                    if arch is None:
                        return ""
                    return "✓" if _dl.get(int(arch)) else "↓"
                except Exception:
                    return ""
            return _fn2
        except Exception:
            return None

    def _refresh_model_row(self):
        """Show the active engine's choices in the shared Model row."""
        if not self.gui:
            return
        try:
            eng = self.config.get("engine", "Moonshine v2")
            try:
                self.gui.set_model_title(eng)
            except Exception:
                pass
            fn = self._model_status_fn()
            if eng == "Whisper Large v3":
                from gui import WHISPER_MODEL_CHOICES, WHISPER_MODEL_CHOICES_REV
                cur = WHISPER_MODEL_CHOICES_REV.get(
                    self.config.get("whisper_model", "large-v3"),
                    "Large v3 (3GB, best)")
                self.gui.set_model_options(list(WHISPER_MODEL_CHOICES.keys()),
                                           cur, self._on_model_changed, fn)
            elif eng == "Canary-1B":
                from gui import CANARY_MODEL_LABEL
                self.gui.set_model_options([CANARY_MODEL_LABEL], CANARY_MODEL_LABEL,
                                           self._on_model_changed, None)
            else:
                from gui import MODEL_CHOICES, MODEL_CHOICES_REV
                cur_arch = int(self.config.get("model_arch", 5))
                cur_lbl = MODEL_CHOICES_REV.get(
                    cur_arch, "Medium Streaming (110MB, best)")
                self.gui.set_model_options(list(MODEL_CHOICES.keys()),
                                           cur_lbl, self._on_model_changed, fn)
        except Exception:
            pass

    def _open_model_manager(self):
        if not self.gui:
            return
        try:
            self.gui.show_model_manager(
                self._model_manager_data,
                self._model_manager_delete,
                self._model_manager_delete_all)
        except Exception as e:
            print(f"model manager failed: {e}")

    def _model_manager_data(self):
        """Snapshot for the storage dialog (fast dir walks, no loads)."""
        from gui import MODEL_CHOICES, WHISPER_MODEL_CHOICES
        items = []
        total = 0
        try:
            active = self.config.get("engine", "Moonshine v2")
        except Exception:
            active = "Moonshine v2"
        # Moonshine archs
        try:
            from engine import moonshine_cache_info
            minfo = moonshine_cache_info()
        except Exception:
            minfo = {}
        try:
            cur_arch = int(self.config.get("model_arch", 5))
        except Exception:
            cur_arch = 5
        for label, arch in MODEL_CHOICES.items():
            try:
                size = int(minfo.get(int(arch), 0))
            except Exception:
                size = 0
            dl = size > 1_000_000
            if dl:
                total += size
            items.append({"engine": "Moonshine v2", "kind": "moonshine",
                          "id": int(arch), "label": label,
                          "size": size if dl else None,
                          "downloaded": dl,
                          "in_use": active == "Moonshine v2" and cur_arch == int(arch)})
        # Whisper sizes (large + large-v3 share one repo dir)
        try:
            from whisper_engine import whisper_cache_info
            winfo = whisper_cache_info()
        except Exception:
            winfo = {}
        wmid = self.config.get("whisper_model", "large-v3")
        for label, mid in WHISPER_MODEL_CHOICES.items():
            try:
                size = int(winfo.get(mid, 0))
            except Exception:
                size = 0
            dl = size > 20_000_000
            if dl:
                total += size
            items.append({"engine": "Whisper Large v3", "kind": "whisper",
                          "id": mid, "label": label,
                          "size": size if dl else None,
                          "downloaded": dl,
                          "in_use": active == "Whisper Large v3" and wmid == mid})
        # Canary (single .nemo; shared HF/torch caches untouched)
        try:
            from canary_engine import canary_cache_info
            csize = int((canary_cache_info() or {}).get("nemo", 0))
        except Exception:
            csize = 0
        cdl = csize > 100_000_000
        if cdl:
            total += csize
        items.append({"engine": "Canary-1B", "kind": "canary",
                      "id": "canary-1b", "label": "Canary-1B (3.9GB)",
                      "size": csize if cdl else None, "downloaded": cdl,
                      "in_use": active == "Canary-1B"})
        return {"items": items, "total": total}

    def _model_manager_delete(self, engine, kind, ident, refresh=True):
        """Delete one cached model. Returns (ok, msg)."""
        if self._srt_busy:
            return False, "Stop the running SRT/burn job first."
        try:
            if kind == "moonshine":
                arch = int(ident)
                if (self.config.get("engine") == "Moonshine v2"
                        and int(self.config.get("model_arch", 5)) == arch):
                    return False, "Switch to another model first."
                from engine import delete_moonshine_model
                freed = delete_moonshine_model(None, arch)
            elif kind == "whisper":
                if (self.config.get("engine") == "Whisper Large v3"
                        and self.config.get("whisper_model") == ident):
                    return False, "Switch to another model first."
                from whisper_engine import delete_whisper_model
                freed, affected = delete_whisper_model(None, ident)
                extra = [a for a in (affected or []) if a != ident]
                note = (f" (shared cache also removed: {', '.join(extra)})"
                        if extra else "")
                self._refresh_model_row()
                return True, f"Deleted {ident} ({self._fmt_gb(freed)} freed).{note}"
            elif kind == "canary":
                if self.config.get("engine") == "Canary-1B":
                    return False, "Switch to another engine first."
                from canary_engine import delete_canary_model
                freed = delete_canary_model()
            else:
                return False, "Unknown model kind."
            if refresh:
                try:
                    self._refresh_model_row()
                except Exception:
                    pass
            return True, f"Deleted ({self._fmt_gb(freed)} freed)."
        except FileNotFoundError as e:
            return False, str(e)
        except Exception as e:
            import traceback
            traceback.print_exc()
            return False, f"Delete failed: {e}"

    def _model_manager_delete_all(self):
        """Delete every downloaded, non-active model. Returns (ok, msg)."""
        if self._srt_busy:
            return False, "Stop the running SRT/burn job first."
        try:
            data = self._model_manager_data() or {}
        except Exception as e:
            return False, str(e)
        freed_total, removed, skipped = 0, [], []
        for it in (data.get("items") or []):
            if not it.get("downloaded") or it.get("in_use"):
                if it.get("downloaded"):
                    skipped.append(str(it.get("label", it.get("id"))))
                continue
            ok, _msg = self._model_manager_delete(
                it.get("engine"), it.get("kind"), it.get("id"), refresh=False)
            if ok:
                removed.append(str(it.get("label", it.get("id"))))
                try:
                    freed_total += int(it.get("size") or 0)
                except Exception:
                    pass
            else:
                skipped.append(str(it.get("label", it.get("id"))))
        try:
            self._refresh_model_row()
        except Exception:
            pass
        if not removed:
            return False, "Nothing deletable right now."
        msg = (f"Deleted {len(removed)} model(s), "
               f"{self._fmt_gb(freed_total)} freed.")
        if skipped:
            msg += f" Skipped (in use): {', '.join(skipped[:3])}."
        return True, msg

    def _on_model_changed(self, display_label: str):
        if self.config.get("engine") == "Whisper Large v3":
            self._on_whisper_model_changed(display_label)
            return
        if self.config.get("engine") not in ("Moonshine v2",):
            return  # fixed-model engines have nothing to switch
        try:
            from gui import MODEL_CHOICES
            new_arch = MODEL_CHOICES.get(display_label, 5)
        except Exception:
            new_arch = 5
        old_arch = int(self.config.get("model_arch", 5))
        if new_arch == old_arch:
            return
        self.config["model_arch"] = new_arch
        save_local_config(self.config)
        self._log(f"Model -> {display_label} (arch {new_arch}), reloading...")
        if self.gui:
            self.gui.set_model_status(f"Switching to {display_label}...", WARNING)
            # Disable record button during switch
            try:
                self.gui.record_btn.configure(state="disabled")
            except Exception:
                pass

        def _model_switched(success, err):
            def _ui():
                try:
                    self.gui.record_btn.configure(state="normal")
                except Exception:
                    pass
                self._model_switching = False
                if success:
                    self.gui.set_status(f"Ready \u2022 {display_label}", SUCCESS)
                else:
                    self.gui.set_status(f"Model error: {err}", DANGER)
            if self.gui:
                try:
                    self.gui.after(0, _ui)
                except Exception:
                    pass

        # ensure active engine is moonshine for model switch
        if self.config.get("engine") == "Canary-1B":
            # If user switched model while Canary active, switch back to Moonshine
            self.config["engine"] = "Moonshine v2"
            save_local_config(self.config)
            self.engine = self.moonshine_engine
        self._model_switching = True
        self.engine.switch_model(new_arch, on_ready=_model_switched)

    def _on_whisper_model_changed(self, display_label: str):
        try:
            from gui import WHISPER_MODEL_CHOICES, WHISPER_MODEL_CHOICES_REV
        except Exception:
            return
        new_id = WHISPER_MODEL_CHOICES.get(display_label)
        if not new_id:
            return
        old_id = self.config.get("whisper_model", "large-v3")
        if new_id == old_id:
            return
        if self._srt_busy:
            # The SRT/burn job shares this engine - switching mid-job would
            # blank its chunks. Revert the menu and say so.
            self._log("Stop the SRT/burn job before switching Whisper model")
            try:
                self._refresh_model_row()
            except Exception:
                pass
            return
        self.config["whisper_model"] = new_id
        save_local_config(self.config)
        self._log(f"Whisper model -> {display_label} ({new_id}), reloading...")
        if self.gui:
            self.gui.set_model_status(f"Switching to {display_label}...", WARNING)
            try:
                self.gui.record_btn.configure(state="disabled")
            except Exception:
                pass

        def _whisper_switched(success, err):
            def _ui():
                try:
                    self.gui.record_btn.configure(state="normal")
                except Exception:
                    pass
                self._model_switching = False
                if success:
                    self.gui.set_status(f"Ready \u2022 Whisper {new_id}", SUCCESS)
                else:
                    self.gui.set_status(f"Model error: {err}", DANGER)
            if self.gui:
                try:
                    self.gui.after(0, _ui)
                except Exception:
                    pass

        eng = self._get_whisper_engine()
        try:
            self._model_switching = True
            eng.switch_model(new_id, on_ready=_whisper_switched)
        except AttributeError:
            # Fallback target (moonshine) has no whisper switch - shouldn't
            # happen, but never wedge the UI (or the switching flag).
            self._model_switching = False
            _whisper_switched(False, "engine unavailable")

    def _on_engine_changed(self, display_label: str):
        old = self.config.get("engine", "Moonshine v2")
        if display_label == old:
            return
        self.config["engine"] = display_label
        save_local_config(self.config)
        self._log(f"Engine -> {display_label}")
        # Sync shared Task/Src widgets to the newly active engine's saved values
        try:
            if self.gui:
                if display_label == "Whisper Large v3":
                    self.gui.canary_task_var.set(self.config.get("whisper_task", "translate"))
                    self.gui.canary_lang_var.set(self.config.get("whisper_src_lang", "ja"))
                elif display_label == "Canary-1B":
                    self.gui.canary_task_var.set(self.config.get("canary_task", "transcribe"))
                    self.gui.canary_lang_var.set(self.config.get("canary_src_lang", "auto"))
                # Restore this engine's SRT languages into the widgets; the
                # GUI then coerces anything the engine cannot do (e.g. SRT
                # input Japanese under Canary -> Auto-detect). Config keeps
                # each engine's last values so switching back restores them.
                try:
                    self.gui.set_srt_languages(
                        self.config.get("srt_input_lang", "auto"),
                        self.config.get("srt_output_lang", "en"),
                        self._on_srt_input_lang_changed,
                        self._on_srt_output_lang_changed,
                    )
                except Exception:
                    pass
                try:
                    self.gui._refresh_srt_engine_label()
                except Exception:
                    pass
                # Enable/disable SRT language dropdowns based on engine
                try:
                    self.gui.set_srt_lang_state(display_label)
                except Exception:
                    pass
                # Show the new engine's choices in the shared Model row
                # (Moonshine sizes / Whisper downloadable sizes / Canary fixed)
                try:
                    self._refresh_model_row()
                except Exception:
                    pass
        except Exception:
            pass
        if display_label == "Canary-1B":
            new_engine = self._get_canary_engine()
            self.engine = new_engine
            if self.gui:
                self.gui.set_status("Loading Canary-1B...", WARNING)
                try:
                    self.gui.record_btn.configure(state="disabled")
                except Exception:
                    pass
            if not new_engine.is_ready:
                # Trigger load, _model_ready will re-enable
                new_engine.load()
            else:
                if self.gui:
                    try:
                        self.gui.after(0, lambda e=new_engine: self.gui.set_status(f"Ready \u2022 Canary-1B ({e.task} {e.source_lang}->en)", SUCCESS))
                        self.gui.after(0, lambda: self.gui.record_btn.configure(state="normal"))
                    except Exception:
                        pass
        elif display_label == "Whisper Large v3":
            new_engine = self._get_whisper_engine()
            self.engine = new_engine
            if self.gui:
                self.gui.set_status("Loading Whisper Large v3...", WARNING)
                try:
                    self.gui.record_btn.configure(state="disabled")
                except Exception:
                    pass
            if not new_engine.is_ready:
                new_engine.load()
            else:
                if self.gui:
                    try:
                        self.gui.after(0, lambda e=new_engine: self.gui.set_status(f"Ready \u2022 Whisper Large v3 ({e.task} {e.source_lang}->en)", SUCCESS))
                        self.gui.after(0, lambda: self.gui.record_btn.configure(state="normal"))
                    except Exception:
                        pass
        else:
            self.engine = self.moonshine_engine
            if self.gui:
                if not self.moonshine_engine.is_ready:
                    self.gui.set_status("Loading Moonshine...", WARNING)
                    self.moonshine_engine.load()
                else:
                    self.gui.set_status(f"Ready \u2022 Moonshine {self.moonshine_engine.current_arch_name}", SUCCESS)
        # Free the other heavy engine's weights (each is 3-4GB; keeping both
        # resident alongside torch + a burn encode pages 16GB machines into a
        # severe hang). Skipped while a job runs - it may still need them.
        self._unload_idle_engines()

    def _unload_idle_engines(self):
        """Release whichever heavy engine is NOT active (multi-GB RAM back).
        Never touches the active engine, a loading engine, or anything while
        an SRT/burn job is running (the job may hold the old engine)."""
        try:
            if self._srt_busy:
                return
            active = self.config.get("engine", "Moonshine v2")
            for name in ("canary_engine", "whisper_engine"):
                try:
                    eng = getattr(self, name, None)
                    if eng is None or eng is self.engine:
                        continue
                    if active == "Canary-1B" and name == "canary_engine":
                        continue
                    if active == "Whisper Large v3" and name == "whisper_engine":
                        continue
                    try:
                        if eng.unload():
                            self._log(f"Unloaded idle {name} (RAM reclaimed)")
                    except Exception:
                        pass
                except Exception:
                    pass
        except Exception:
            pass

    def _on_canary_task_changed(self, value):
        # Shared Task widget - routes to the active heavy engine
        if self.config.get("engine") == "Whisper Large v3":
            self.config["whisper_task"] = value
            save_local_config(self.config)
            if self.whisper_engine:
                try:
                    self.whisper_engine.switch_options(task=value)
                except Exception:
                    pass
            self._log(f"Whisper task -> {value} (translate always outputs EN)")
            return
        self.config["canary_task"] = value
        save_local_config(self.config)
        if self.canary_engine:
            try:
                self.canary_engine.switch_options(task=value)
            except Exception:
                pass
        self._log(f"Canary task -> {value}")

    def _on_canary_lang_changed(self, value):
        # Shared Src widget - routes to the active heavy engine
        if self.config.get("engine") == "Whisper Large v3":
            self.config["whisper_src_lang"] = value
            save_local_config(self.config)
            if self.whisper_engine:
                try:
                    self.whisper_engine.switch_options(source_lang=value)
                except Exception:
                    pass
            self._log(f"Whisper src -> {value}")
            return
        self.config["canary_src_lang"] = value
        save_local_config(self.config)
        if self.canary_engine:
            try:
                self.canary_engine.switch_options(source_lang=value)
            except Exception:
                pass
        self._log(f"Canary src -> {value}")

    def _on_srt_input_lang_changed(self, value):
        from gui import SRT_LANG_CODE_FROM_DISPLAY, SRT_LANGS
        code = SRT_LANG_CODE_FROM_DISPLAY.get(value, "auto")
        if code not in SRT_LANGS:
            code = "auto"
        with _CONFIG_LOCK:
            self.config["srt_input_lang"] = code
            save_local_config(self.config)
        self._log(f"SRT input lang -> {code}")

    def _on_srt_output_lang_changed(self, value):
        from gui import SRT_LANG_CODE_FROM_DISPLAY, SRT_LANGS
        code = SRT_LANG_CODE_FROM_DISPLAY.get(value, "en")
        if code not in SRT_LANGS:
            code = "en"
        with _CONFIG_LOCK:
            self.config["srt_output_lang"] = code
            save_local_config(self.config)
        self._log(f"SRT output lang -> {code}")

    def _log(self, msg, color=None):
        # Mirror to GUI log if available, else print
        print(f"[MoonshineSTT] {msg}")
        if self.gui and color:
            pass  # GUI status handles this via queue

    def _capture_target_window(self):
        """Ported from test.py _capture_target_window - save window under cursor at key-down."""
        try:
            hwnd = get_foreground_window()
            target_root = user32.GetAncestor(hwnd, GA_ROOT) if hwnd else 0
            if hwnd and target_root != self._our_root_hwnd:
                self._target_top_hwnd = hwnd
                child = get_focused_child(hwnd)
                # Drop stale/invalid child hwnds here (Chrome omnibox etc.):
                # an invalid child makes verify_focus_ready loop and flash focus.
                try:
                    if child and not user32.IsWindow(child):
                        child = None
                except Exception:
                    child = None
                self._target_child_hwnd = child
                title = get_window_title(hwnd)
                short = title[:40] + "..." if len(title) > 40 else title
                self._log(f"Target captured: {short} (hwnd={hwnd}, child={self._target_child_hwnd})")
                if self.gui:
                    # Show target in status briefly
                    self._gui_queue.put(("status", (f"Target: {short}", FG_SECONDARY)))
            else:
                # No valid target (we are focused on self) - clear so insert will fallback to fg at paste time
                self._target_top_hwnd = None
                self._target_child_hwnd = None
        except Exception as e:
            print(f"capture target failed: {e}")
            self._target_top_hwnd = None
            self._target_child_hwnd = None

    def _model_ready(self, success: bool, error: str = None):
        if self.gui:
            try:
                # Guard: gui may not have mainloop yet (headless tests)
                if self.gui.winfo_exists():
                    if success:
                        self.gui.after(0, lambda: self.gui.set_status("Ready \u2022 Hold F2 to speak", SUCCESS))
                    else:
                        self.gui.after(0, lambda: self.gui.set_status(f"Model error: {error}", DANGER))
                    return
            except Exception:
                pass
        print("Model ready!" if success else f"Model error: {error}")

    def _level_callback(self, level: float):
        # PortAudio fires this ~64x/sec - throttle to ~15Hz so the Tk event
        # queue is not flooded with redraws (was a GUI-freeze risk).
        now = time.monotonic()
        if now - getattr(self, "_last_level_t", 0.0) < 0.065:
            return
        self._last_level_t = now
        if self.gui:
            try:
                self.gui.after(0, lambda l=level: self.gui.update_level(l))
            except Exception:
                pass

    def start(self):
        if self.gui:
            self.gui.set_status("Loading model...", WARNING)
        else:
            print("Loading Moonshine v2 model...")
        self.engine.load()

        t = threading.Thread(target=self._keyboard_listener, daemon=True)
        t.start()

        if self.gui:
            self._poll_queue()
            self.gui.mainloop()
        else:
            print("Hold F2 to record, release to transcribe. Ctrl+C to quit.")
            try:
                while True:
                    time.sleep(0.2)
            except KeyboardInterrupt:
                print("\nExiting...")

    def _poll_queue(self):
        if not self.gui:
            return
        try:
            while True:
                msg_type, payload = self._gui_queue.get_nowait()
                if msg_type == "transcription":
                    self.gui.set_transcription(payload)
                elif msg_type == "status":
                    self.gui.set_status(payload[0], payload[1])
                elif msg_type == "srt_progress":
                    self.gui.set_srt_progress(payload[0], payload[1])
                elif msg_type == "srt_log":
                    self.gui.srt_log(payload)
                elif msg_type == "srt_done":
                    self.gui.srt_done(payload[0], payload[1])
                elif msg_type == "srt_file_status":
                    try:
                        self.gui.set_srt_file_status(payload[0], payload[1])
                    except Exception:
                        pass
        except queue.Empty:
            pass
        self.gui.after(100, self._poll_queue)

    def _keyboard_listener(self):
        # NOTE: any exception escaping on_press/on_release kills this
        # listener thread and F2 silently stops working - hence the guards.
        def on_press(key):
            try:
                if key == RECORD_KEY and not self._key_pressed and not self._recording:
                    self._key_pressed = True
                    self._capture_target_window()
                    self._start_recording()
            except Exception as e:
                print(f"hotkey press error: {e}")

        def on_release(key):
            try:
                if key == RECORD_KEY and self._key_pressed:
                    self._key_pressed = False
                    self._stop_recording()
            except Exception as e:
                print(f"hotkey release error: {e}")

        with keyboard.Listener(on_press=on_press, on_release=on_release) as listener:
            listener.join()

    def _start_recording(self):
        with self._rec_lock:
            if self._recording:
                return
            if getattr(self, "_model_switching", False):
                self._gui_queue.put(("status", ("Switching model - wait a moment", WARNING)))
                return
            try:
                self.recorder.start()
            except Exception as e:
                msg = f"Mic error: {e}"
                print(msg)
                if self.gui:
                    self._gui_queue.put(("status", (msg, DANGER)))
                    self.gui.after(0, lambda: self.gui.set_recording_state(False))
                self._recording = False
                return

            self._recording = True
        if self.gui:
            self.gui.after(0, lambda: self.gui.set_recording_state(True))
            self.gui.after(0, lambda: self.gui.set_status("Recording... F2 held", DANGER))

    def _stop_recording(self):
        with self._rec_lock:
            if not self._recording:
                return
            self._recording = False
            try:
                audio = self.recorder.stop()
            except Exception as e:
                print(f"recorder stop error: {e}")
                audio = None
        if self.gui:
            self.gui.after(0, lambda: self.gui.set_recording_state(False))
            self.gui.after(0, lambda: self._update_indicator())
        # Snapshot all insertion settings + engine at stop time
        saved_top = self._target_top_hwnd
        saved_child = self._target_child_hwnd
        settings_snapshot = {
            "target_top": saved_top,
            "target_child": saved_child,
            "suffix": self.config.get("suffix", "none"),
            "typing_method": self.config.get("typing_method", "clipboard"),
            "typing_delay_ms": int(self.config.get("typing_delay_ms", 0)),
            "our_root_hwnd": self._our_root_hwnd,
            "engine": self.engine,
            "engine_name": self.config.get("engine", "Moonshine v2"),
        }

        if audio is None or len(audio) == 0:
            self._gui_queue.put(("status", ("No audio", WARNING)))
            if self.gui:
                self.gui.after(0, lambda: self.gui.set_transcription("(no audio captured)"))
            self._update_indicator()
            return
        if len(audio) < int(SAMPLE_RATE * 0.3):
            self._gui_queue.put(("status", ("Too short", WARNING)))
            if self.gui:
                self.gui.after(0, lambda: self.gui.set_transcription("(too short)"))
            self._update_indicator()
            return

        # Enqueue for serial processing - allows recording next chunk while this one transcribes
        try:
            if self.audio_queue.qsize() >= 25:
                # Pathological hammering guard: ~25 queued clips is minutes
                # of audio and hundreds of MB - tell the user to wait.
                self._gui_queue.put(("status", ("Queue full - wait for processing", WARNING)))
                self._update_indicator()
                return
        except Exception:
            pass
        self.audio_queue.put((audio, settings_snapshot))
        # Update indicator to show queue depth
        self._update_indicator()

    def _processing_worker(self):
        """Serial queue worker - ported from test.py _processing_worker."""
        while True:
            task = self.audio_queue.get()
            if task is None:
                break
            audio, settings = task
            self.currently_processing = True
            try:
                self._update_indicator()
                self._process_queued_audio(audio, settings)
            except Exception as e:
                print(f"Queue worker error: {e}")
                import traceback; traceback.print_exc()
            finally:
                self.currently_processing = False
                self.audio_queue.task_done()
                self._update_indicator()

    def _process_queued_audio(self, audio, settings):
        # Show transcribing status with queue depth
        q_size = self.audio_queue.qsize()
        if self.gui:
            if q_size > 0:
                self._gui_queue.put(("status", (f"Transcribing... ({q_size} queued)", WARNING)))
            else:
                self._gui_queue.put(("status", ("Transcribing...", WARNING)))
        engine = settings.get("engine", self.engine)
        text = engine.transcribe(audio, SAMPLE_RATE)
        clean = text.strip() if text else ""
        if clean and not clean.startswith("["):
            final_text = apply_suffix(clean, settings.get("suffix", "none"))
            self._gui_queue.put(("transcription", clean))
            ok = insert_text(
                final_text,
                settings.get("target_top"),
                settings.get("target_child"),
                "F2",
                our_root_hwnd=settings.get("our_root_hwnd", 0),
                method=settings.get("typing_method", "clipboard"),
                allow_self=False,
                wake=False,
                delay_ms=settings.get("typing_delay_ms", 0),
                log_func=lambda m: print(f"[insert] {m}"),
            )
            if ok:
                self._gui_queue.put(("status", ("Pasted \u2713  (clipboard also)", SUCCESS)))
            else:
                self._gui_queue.put(("status", ("Copied to clipboard \u2014 press Ctrl+V", WARNING)))
        else:
            display = clean if clean else "(no speech detected)"
            self._gui_queue.put(("transcription", display))
            self._gui_queue.put(("status", ("Ready", FG_SECONDARY)))

    def _update_indicator(self):
        """Update GUI indicator based on recording/processing/queue state (ported from test.py update_indicator)."""
        if not self.gui:
            return
        try:
            q_size = self.audio_queue.qsize()
            is_proc = self.currently_processing
            is_rec = self._recording
            if is_rec:
                # Recording - status already set by _start_recording, but ensure queue info
                if is_proc or q_size > 0:
                    self._gui_queue.put(("status", (f"Recording... ({q_size} queued)", DANGER)))
                return
            if is_proc or q_size > 0:
                if q_size > 0:
                    self._gui_queue.put(("status", (f"Processing ({q_size} queued)...", WARNING)))
                else:
                    self._gui_queue.put(("status", ("Processing...", WARNING)))
            else:
                # Idle - let _model_ready or last status handle, but ensureReady shown if no other
                pass
        except Exception:
            pass

    # ---------------- SRT file jobs ----------------
    def _get_srt_moonshine_transcriber(self):
        """Dedicated per-job Moonshine Transcriber (live engine untouched)."""
        from moonshine_voice import get_model_for_language, Transcriber
        from moonshine_voice.moonshine_api import ModelArch
        from engine import PORTABLE_CACHE_ROOT, MODEL_ARCH_NAMES
        arch = int(self.config.get("model_arch", 5))
        try:
            wanted = ModelArch(arch)
        except Exception:
            wanted = None
        try:
            PORTABLE_CACHE_ROOT.mkdir(parents=True, exist_ok=True)
            model_path, model_arch = get_model_for_language("en", wanted, cache_root=PORTABLE_CACHE_ROOT)
        except Exception:
            model_path, model_arch = get_model_for_language("en", wanted)
        tr = Transcriber(model_path=model_path, model_arch=model_arch)
        try:
            arch_name = MODEL_ARCH_NAMES.get(model_arch.value, str(model_arch))
        except Exception:
            arch_name = str(arch)
        return tr, arch_name

    def _srt_start(self, input_paths, out_dir: str, cpu_workers: int,
                   srt_in: str = "", srt_out: str = "", srt_task: str = ""):
        # Atomic check-and-set: two rapid Starts must not launch two jobs.
        # Self-heal: if a previous job thread died without clearing the flag
        # (must never happen, but a stuck "already running" bricks the tab),
        # detect the dead thread and allow a fresh start.
        with self._srt_lock:
            if self._srt_busy:
                _t = self._srt_thread
                if _t is not None and _t.is_alive():
                    self._gui_queue.put(("srt_log", "A batch is already running. Cancel it first."))
                    return
                self._gui_queue.put(("srt_log", "Clearing stale job flag from a previous run..."))
                self._srt_busy = False
                self._srt_thread = None
            self._srt_busy = True
        # Everything between busy-set and thread-spawn must be failure-proof:
        # any exception here would otherwise wedge busy=True with no job.
        try:
            # Snapshot everything the job needs NOW (worker thread must never
            # touch Tk widgets, and config may change mid-job from the GUI).
            # First arg is the batch queue (a single stray string is tolerated).
            if isinstance(input_paths, str):
                paths = [input_paths]
            else:
                try:
                    paths = list(input_paths or [])
                except Exception:
                    paths = []
            paths = [str(p).strip().strip('"') for p in paths if str(p or "").strip()]
            if not paths:
                raise ValueError("queue is empty - add video/audio files first")
            from gui import SRT_LANGS as _SRT_LANGS
            srt_in = (srt_in or "").strip().lower() or None
            srt_out = (srt_out or "").strip().lower() or None
            srt_task = (srt_task or "").strip().lower()
            if srt_task not in ("transcribe", "translate"):
                srt_task = ""
            with _CONFIG_LOCK:
                if srt_in in _SRT_LANGS:
                    self.config["srt_input_lang"] = srt_in
                if srt_out in _SRT_LANGS:
                    self.config["srt_output_lang"] = srt_out
                # The widget the SRT label is rendered from is the source of
                # truth for the task: persist it to the ACTIVE engine's key so
                # label, job, and config can never disagree again.
                _eng = self.config.get("engine", "Moonshine v2")
                if srt_task and _eng == "Whisper Large v3":
                    self.config["whisper_task"] = srt_task
                elif srt_task and _eng == "Canary-1B":
                    self.config["canary_task"] = srt_task
                self.config["srt_out_dir"] = out_dir or ""
                try:
                    self.config["srt_cpu"] = int(cpu_workers)
                except Exception:
                    self.config["srt_cpu"] = 0
                save_local_config(self.config)
                job = {
                    "engine_kind": self.config.get("engine", "Moonshine v2"),
                    "moonshine_arch": int(self.config.get("model_arch", 5)),
                    "canary_task": self.config.get("canary_task", "transcribe"),
                    "canary_src": self.config.get("canary_src_lang", "ja"),
                    "whisper_task": self.config.get("whisper_task", "translate"),
                    "whisper_src": self.config.get("whisper_src_lang", "ja"),
                    "srt_input_lang": self.config.get("srt_input_lang", "auto"),
                    "srt_output_lang": self.config.get("srt_output_lang", "en"),
                    "cpu_workers": int(cpu_workers),
                    "src_paths": paths,
                    "out_dir": out_dir,
                }
            if job["engine_kind"] not in ("Moonshine v2", "Canary-1B", "Whisper Large v3"):
                job["engine_kind"] = "Moonshine v2"
            self._srt_cancel.clear()
            if self.gui:
                self.gui.after(0, lambda: self.gui.set_srt_running(True))
                _n = len(job["src_paths"])
                self.gui.after(0, lambda _n=_n: self.gui.set_srt_progress(
                    0, f"Starting batch ({_n} file(s))..."))
        except Exception as e:
            with self._srt_lock:
                self._srt_busy = False
            self._gui_queue.put(("srt_done", (False, f"Could not start job: {e}")))
            return

        def _job():
            try:
                import os as _os
                import srt as srtmod
                # Frozen snapshot - immune to GUI changes mid-job.
                def _run_one(path, progress_cb, log_cb):
                    return srtmod.run_srt_job(
                        src_path=path,
                        out_dir=job["out_dir"],
                        engine_kind=job["engine_kind"],
                        moonshine_arch=job["moonshine_arch"],
                        canary_task=job["canary_task"],
                        canary_src=job["canary_src"],
                        cpu_workers=job["cpu_workers"],
                        get_moonshine_transcriber=self._get_srt_moonshine_transcriber,
                        # apply_live_options=False: SRT uses per-call overrides
                        # and must never mutate the live engine from this worker.
                        get_canary_engine=lambda: self._get_canary_engine(False),
                        progress_cb=progress_cb,
                        log_cb=log_cb,
                        cancel_event=self._srt_cancel,
                        whisper_task=job["whisper_task"],
                        whisper_src=job["whisper_src"],
                        get_whisper_engine=lambda: self._get_whisper_engine(False),
                        srt_input_lang=job["srt_input_lang"],
                        srt_output_lang=job["srt_output_lang"],
                    )

                def _file_cb(kind, path, info):
                    try:
                        idx = int((info or {}).get("index", -1))
                    except Exception:
                        idx = -1
                    total = len(job["src_paths"])
                    if kind == "start":
                        self._gui_queue.put(
                            ("srt_file_status", (idx, "working…")))
                    elif kind == "done":
                        if (info or {}).get("ok"):
                            try:
                                name = _os.path.basename((info or {}).get("out") or path)
                            except Exception:
                                name = "saved"
                            self._gui_queue.put(
                                ("srt_file_status", (idx, f"✓ {name}")))
                        else:
                            err = str((info or {}).get("error") or "error")[:60]
                            self._gui_queue.put(
                                ("srt_file_status", (idx, f"✗ {err}")))
                    elif kind == "skip":
                        self._gui_queue.put(
                            ("srt_file_status", (idx, "– skipped")))

                results, cancelled = srtmod.run_srt_batch(
                    job["src_paths"], _run_one,
                    progress_cb=lambda f, m: self._gui_queue.put(("srt_progress", (f, m))),
                    log_cb=lambda m: self._gui_queue.put(("srt_log", m)),
                    cancel_event=self._srt_cancel,
                    file_cb=_file_cb,
                )
                ok_paths = [p for (p, ok, _m) in results if ok]
                bad = [(p, m) for (p, ok, m) in results if not ok]
                total = len(results)
                if cancelled:
                    self._gui_queue.put(
                        ("srt_done", (False, f"Cancelled after {len(ok_paths)}/{total} files")))
                elif not bad:
                    self._gui_queue.put(
                        ("srt_done", (True, f"Batch done: all {total} saved")))
                elif ok_paths:
                    try:
                        first_bad = _os.path.basename(bad[0][0])
                    except Exception:
                        first_bad = "a file"
                    self._gui_queue.put(
                        ("srt_done", (True, f"Batch done: {len(ok_paths)}/{total} saved "
                                            f"({len(bad)} failed, e.g. {first_bad})")))
                else:
                    self._gui_queue.put(
                        ("srt_done", (False, f"Batch failed: 0/{total} saved")))
            except InterruptedError:
                self._gui_queue.put(("srt_done", (False, "Cancelled by user")))
            except Exception as e:
                import traceback
                traceback.print_exc()
                self._gui_queue.put(("srt_done", (False, f"Error: {e}")))
            finally:
                with self._srt_lock:
                    self._srt_busy = False
                    if threading.current_thread() is self._srt_thread:
                        self._srt_thread = None

        _t = threading.Thread(target=_job, daemon=True)
        with self._srt_lock:
            self._srt_thread = _t
        _t.start()

    def _srt_cancel(self):
        self._srt_cancel.set()
        self._gui_queue.put(("srt_log", "Cancelling..."))

    @staticmethod
    def _fmt_gb(nbytes: int) -> str:
        try:
            v = float(nbytes)
        except Exception:
            return "?GB"
        if v >= 1e9:
            return f"{v / 1e9:.2f}GB"
        if v >= 1e6:
            return f"{v / 1e6:.0f}MB"
        return f"{int(v)}B"

    def _burn_start(self, input_paths, out_dir: str, cpu_workers: int,
                    font_size: int = 0):
        # Same single-flight machinery as SRT: one heavy job at a time, same
        # Cancel button/event, same running UI state.
        with self._srt_lock:
            if self._srt_busy:
                _t = self._srt_thread
                if _t is not None and _t.is_alive():
                    self._gui_queue.put(("srt_log", "A job is already running. Cancel it first."))
                    return
                self._gui_queue.put(("srt_log", "Clearing stale job flag from a previous run..."))
                self._srt_busy = False
                self._srt_thread = None
            self._srt_busy = True
        try:
            if isinstance(input_paths, str):
                paths = [input_paths]
            else:
                try:
                    paths = list(input_paths or [])
                except Exception:
                    paths = []
            paths = [str(p).strip().strip('"') for p in paths if str(p or "").strip()]
            if not paths:
                raise ValueError("queue is empty - add video files first")
            with _CONFIG_LOCK:
                self.config["srt_out_dir"] = out_dir or ""
                try:
                    self.config["srt_cpu"] = int(cpu_workers)
                except Exception:
                    self.config["srt_cpu"] = 0
                try:
                    _fs = int(font_size) if font_size else int(
                        self.config.get("burn_font_size", 18))
                except Exception:
                    _fs = 18
                _fs = max(10, min(40, _fs))
                self.config["burn_font_size"] = _fs
                save_local_config(self.config)
                job = {"paths": paths, "out_dir": out_dir,
                       "cpu_workers": int(cpu_workers), "font_size": _fs}
            self._srt_cancel.clear()
            if self.gui:
                self.gui.after(0, lambda: self.gui.set_srt_running(True))
                _n = len(job["paths"])
                _cpu = job["cpu_workers"]
                self.gui.after(0, lambda _n=_n: self.gui.set_srt_progress(
                    0, f"Starting burn ({_n} file(s))..."))
                self._gui_queue.put(
                    ("srt_log", f"Burn: {_n} file(s), {_cpu} threads (x264 multi-core), "
                                f"size-match target"))
        except Exception as e:
            with self._srt_lock:
                self._srt_busy = False
            self._gui_queue.put(("srt_done", (False, f"Could not start burn: {e}")))
            return

        def _job():
            try:
                import os as _os
                import srt as srtmod
                ffmpeg = srtmod.get_ffmpeg_exe()
                if not ffmpeg:
                    raise RuntimeError("ffmpeg not found - run setup.bat once.")

                def _run_one(path, progress_cb, log_cb):
                    from pathlib import Path as _P
                    src = _P(path)
                    srt_path = srtmod.default_out_path(src, job["out_dir"])
                    if not srt_path.exists():
                        raise FileNotFoundError(
                            f"no SRT for {src.name} - Generate SRT first")
                    info = srtmod.probe_media(src, ffmpeg)
                    vbps, abps, acopy = srtmod.plan_burn_bitrates(info)
                    try:
                        in_gb = self._fmt_gb(info["size"])
                    except Exception:
                        in_gb = "?"
                    log_cb(f"budget: video {vbps // 1000} kbps, "
                           f"audio {'copy' if acopy else f'AAC {abps // 1000} kbps'} "
                           f"(in {in_gb}, target match)")
                    out_path = srtmod.default_burn_path(src, job["out_dir"])
                    return srtmod.burn_subtitles(
                        str(src), str(srt_path), str(out_path), ffmpeg,
                        vbps // 1000, (abps // 1000) if not acopy else 128,
                        acopy, job["cpu_workers"], job["font_size"],
                        progress_cb=progress_cb, log_cb=log_cb,
                        cancel_event=self._srt_cancel)

                def _file_cb(kind, path, info):
                    try:
                        idx = int((info or {}).get("index", -1))
                    except Exception:
                        idx = -1
                    if kind == "start":
                        self._gui_queue.put(("srt_file_status", (idx, "burning…")))
                    elif kind == "done":
                        if (info or {}).get("ok"):
                            try:
                                _o, _ib, _ob = (info or {}).get("out"), 0, 0
                                if isinstance(_o, (list, tuple)) and len(_o) >= 3:
                                    _ib, _ob = _o[1], _o[2]
                                tag = (f"\u2713 {self._fmt_gb(_ob)}"
                                       if _ob else "\u2713 burned")
                            except Exception:
                                tag = "\u2713 burned"
                            self._gui_queue.put(("srt_file_status", (idx, tag)))
                        else:
                            err = str((info or {}).get("error") or "error")[:60]
                            self._gui_queue.put(("srt_file_status", (idx, f"\u2717 {err}")))
                    elif kind == "skip":
                        self._gui_queue.put(("srt_file_status", (idx, "– skipped")))

                results, cancelled = srtmod.run_srt_batch(
                    job["paths"], _run_one,
                    progress_cb=lambda f, m: self._gui_queue.put(("srt_progress", (f, m))),
                    log_cb=lambda m: self._gui_queue.put(("srt_log", m)),
                    cancel_event=self._srt_cancel,
                    file_cb=_file_cb,
                )
                ok = [(p, m) for (p, ok_, m) in results if ok_]
                bad = [(p, m) for (p, ok_, m) in results if not ok_]
                total = len(results)
                in_sum = out_sum = 0
                for (_p, m) in ok:
                    try:
                        if isinstance(m, (list, tuple)) and len(m) >= 3:
                            in_sum += int(m[1] or 0)
                            out_sum += int(m[2] or 0)
                    except Exception:
                        pass
                sizes = (f" ({self._fmt_gb(in_sum)}\u2192{self._fmt_gb(out_sum)})"
                         if in_sum and out_sum else "")
                if cancelled:
                    self._gui_queue.put(
                        ("srt_done", (False, f"Burn cancelled after {len(ok)}/{total} files")))
                elif not bad:
                    self._gui_queue.put(
                        ("srt_done", (True, f"Burned all {total}{sizes}")))
                elif ok:
                    try:
                        first_bad = _os.path.basename(bad[0][0])
                    except Exception:
                        first_bad = "a file"
                    self._gui_queue.put(
                        ("srt_done", (True, f"Burned {len(ok)}/{total}{sizes} "
                                            f"({len(bad)} failed, e.g. {first_bad})")))
                else:
                    self._gui_queue.put(
                        ("srt_done", (False, f"Burn failed: 0/{total}")))
            except InterruptedError:
                self._gui_queue.put(("srt_done", (False, "Cancelled by user")))
            except Exception as e:
                import traceback
                traceback.print_exc()
                self._gui_queue.put(("srt_done", (False, f"Error: {e}")))
            finally:
                with self._srt_lock:
                    self._srt_busy = False
                    if threading.current_thread() is self._srt_thread:
                        self._srt_thread = None

        _t = threading.Thread(target=_job, daemon=True)
        with self._srt_lock:
            self._srt_thread = _t
        _t.start()

    def _burn_preview(self, input_paths, out_dir: str, font_size: int):
        """Render ONE frame with the burn filter (current size) and open it.

        Fast by design (single frame, own thread, no busy flag): the user
        drags the size slider and re-previews until the subtitles look
        right, then runs the FULL encode. First queued file that already
        has an SRT wins; otherwise "Generate SRT first".
        """
        def _done():
            if self.gui:
                try:
                    self.gui.after(0, lambda: self.gui.set_srt_preview_done())
                except Exception:
                    pass

        def _work():
            try:
                import subprocess as _sp
                import tempfile as _tf
                from pathlib import Path as _P
                import srt as srtmod
                if isinstance(input_paths, str):
                    paths = [input_paths]
                else:
                    try:
                        paths = list(input_paths or [])
                    except Exception:
                        paths = []
                paths = [str(p).strip().strip('"') for p in paths
                         if str(p or "").strip()]
                try:
                    size = max(10, min(40, int(font_size or 0)))
                except Exception:
                    size = 18
                if not size:
                    size = 18
                with _CONFIG_LOCK:
                    self.config["burn_font_size"] = size
                    save_local_config(self.config)
                ffmpeg = srtmod.get_ffmpeg_exe()
                if not ffmpeg:
                    raise RuntimeError("ffmpeg not found - run setup.bat once.")
                target = None
                for p in paths:
                    try:
                        cand = srtmod.default_out_path(_P(p), out_dir)
                    except Exception:
                        continue
                    if cand.exists():
                        target = (_P(p), cand)
                        break
                if target is None:
                    self._gui_queue.put(("srt_log", "Preview needs an SRT - Generate SRT first"))
                    self._gui_queue.put(("srt_progress", (0.0, "Generate SRT first")))
                    return
                src, srt_path = target
                try:
                    ts = srtmod.first_cue_at(srt_path)
                except Exception:
                    ts = -1.0
                if ts is None or ts < 0:
                    try:
                        info = srtmod.probe_media(src, ffmpeg)
                        ts = max(0.0, float(info.get("duration") or 0) * 0.25)
                    except Exception:
                        ts = 5.0
                out_png = ((_P(out_dir) if out_dir and str(out_dir).strip() else src.parent)
                           / (src.stem + ".burn_preview.png"))
                try:
                    out_png.parent.mkdir(parents=True, exist_ok=True)
                except Exception:
                    pass
                tmpd = _P(_tf.mkdtemp(prefix="burnprev_"))
                try:
                    vf = srtmod.stage_subtitles_filter(srt_path, size, tmpd)
                    # Fast-seek near the cue, then accurate-seek the last
                    # seconds: instant even in a 2h movie, frame-accurate.
                    pre = max(0.0, ts - 10.0)
                    cmd = [ffmpeg, "-hide_banner", "-y", "-v", "error",
                           "-ss", f"{pre:.3f}", "-i", str(src),
                           "-ss", f"{ts - pre:.3f}",
                           "-frames:v", "1", "-vf", vf, "-an", str(out_png)]
                    proc = _sp.run(cmd, stdout=_sp.DEVNULL,
                                   stderr=_sp.PIPE, timeout=180,
                                   text=True, errors="replace")
                    if proc.returncode != 0 or not out_png.exists():
                        raise RuntimeError(
                            f"preview encode failed: {(proc.stderr or '')[-200:]}")
                    self._gui_queue.put(("srt_log", f"Preview (size {size}): {out_png}"))
                    self._gui_queue.put(("srt_progress", (0.0, f"Preview saved: {out_png.name}")))
                    try:
                        import os as _os
                        _os.startfile(str(out_png))
                    except Exception:
                        pass
                finally:
                    try:
                        import shutil as _sh
                        _sh.rmtree(str(tmpd), ignore_errors=True)
                    except Exception:
                        pass
            except Exception as e:
                import traceback
                traceback.print_exc()
                self._gui_queue.put(("srt_log", f"Preview failed: {e}"))
                self._gui_queue.put(("srt_progress", (0.0, f"Preview failed: {e}")))
            finally:
                _done()

        threading.Thread(target=_work, daemon=True).start()

    def _gui_record_start(self):
        if not self._recording:
            self._capture_target_window()
            self._start_recording()
        else:
            self._gui_queue.put(("status", ("Already recording", WARNING)))

    def _gui_record_stop(self):
        if self._recording:
            self._stop_recording()

    def _on_close(self):
        if self._recording:
            try:
                self.recorder.stop()
            except Exception:
                pass
        if self.gui:
            self.gui.destroy()
        os._exit(0)


def main():
    app = MoonshineSTTApp()
    app.start()


if __name__ == "__main__":
    main()
