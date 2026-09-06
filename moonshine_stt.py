import pathlib
# moonshine_stt.py
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
APP_VERSION = "1.2.18"
CONFIG_PATH = os.path.join(os.path.dirname(__file__), "moonshine_config.json")
_CONFIG_LOCK = threading.RLock()
DEFAULT_CONFIG = {
    "typing_method": "clipboard",
    "suffix": "none",
    "typing_delay_ms": 0,
    "model_arch": 5,
    "engine": "Moonshine v2",
    "canary_task": "transcribe",
    "canary_src_lang": "auto",
    "whisper_task": "translate",
    "whisper_src_lang": "auto",
    "whisper_model": "large-v3",
    "whisper_device": "auto",
    "compute": "auto",
    "srt_cpu": 0,
    "srt_out_dir": "",
    "srt_input_lang": "auto",
    "srt_output_lang": "en",
    "burn_font_size": 18,
    "burn_sample_start": "0:30",
    "burn_sample_len": 15,
    "burn_vbr_auto": True,
    "burn_vbr_kbps": 2000,
    "burn_speed": "match",
    "burn_codec": "h264",
    "srt_tab": "Live",
    "srt_norm": False,
    "burn_after": False,
    "auto_shutdown": False,
    "completion_alert": True,
    "theme": "dark",
}
def load_local_config():
    try:
        if os.path.exists(CONFIG_PATH):
            with open(CONFIG_PATH, "r", encoding="utf-8") as f:
                cfg = json.load(f)
            out = dict(DEFAULT_CONFIG)
            out.update(cfg)
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
    try:
        with _CONFIG_LOCK:
            tmp = CONFIG_PATH + ".tmp"
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump(cfg, f, indent=2)
            os.replace(tmp, CONFIG_PATH)
    except Exception:
        pass
def update_config(config, **changes):
    """Atomically update and persist a configuration snapshot."""
    with _CONFIG_LOCK:
        snapshot = dict(config or {})
        snapshot.update(changes)
        config.clear()
        config.update(snapshot)
        save_local_config(dict(snapshot))
        return dict(snapshot)


def apply_suffix(text: str, suffix: str) -> str:
    if suffix == "space":
        return text + " "
    if suffix == "newline":
        return text + "\n"
    if suffix == "period_space":
        return text + ". "
    return text
def _speed_label(speed_id: str) -> str:
    try:
        from srt import BURN_SPEED_LABELS
        return BURN_SPEED_LABELS.get(speed_id, "Match size (2-pass x264)")
    except Exception:
        return "Match size (2-pass x264)"
class MoonshineSTTApp:
    def __init__(self):
        self.config = load_local_config()
        needs_save = False
        for k, v in DEFAULT_CONFIG.items():
            if k not in self.config:
                self.config[k] = v
                needs_save = True
        try:
            arch = int(self.config.get("model_arch", 5))
            if arch not in (0, 1, 2, 3, 4, 5):
                arch = 5
                self.config["model_arch"] = 5
                needs_save = True
        except Exception:
            arch = 5
            self.config["model_arch"] = 5
            needs_save = True
        if self.config.get("engine") in ("Whisper Turbo v3", "Whisper Large v3"):
            self.config["engine"] = "Whisper"
            needs_save = True
        if self.config.get("engine") not in ("Moonshine v2", "Canary-1B", "Whisper"):
            self.config["engine"] = "Moonshine v2"
            needs_save = True
        if self.config.get("canary_task") not in ("transcribe", "translate"):
            self.config["canary_task"] = "transcribe"
            needs_save = True
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
        if self.config.get("whisper_device") not in ("auto", "cpu", "cuda"):
            self.config["whisper_device"] = "auto"
            needs_save = True
        if self.config.get("compute") not in ("auto", "cpu", "gpu"):
            self.config["compute"] = "auto"
            needs_save = True
        try:
            _th = str(self.config.get("theme", "dark")).lower()
        except Exception:
            _th = "dark"
        if _th not in ("dark", "light"):
            _th = "dark"
        if self.config.get("theme") != _th:
            self.config["theme"] = _th
            needs_save = True
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
        try:
            _bl = int(self.config.get("burn_sample_len", 15))
        except Exception:
            _bl = 15
        if _bl not in (10, 15, 30, 60):
            _bl = 15
            needs_save = True
        if self.config.get("burn_sample_len") != _bl:
            self.config["burn_sample_len"] = _bl
            needs_save = True
        if not isinstance(self.config.get("burn_sample_start"), str):
            self.config["burn_sample_start"] = "0:30"
            needs_save = True
        if not isinstance(self.config.get("burn_vbr_auto"), bool):
            try:
                self.config["burn_vbr_auto"] = bool(int(self.config.get("burn_vbr_auto", 1)))
            except Exception:
                self.config["burn_vbr_auto"] = True
            needs_save = True
        for _bk, _bdef in (("srt_norm", False), ("burn_after", False),
                           ("auto_shutdown", False),
                           ("completion_alert", True)):
            _bv = self.config.get(_bk, _bdef)
            if isinstance(_bv, bool):
                _nb = _bv
            elif isinstance(_bv, (int, float)):
                _nb = bool(_bv)
            elif isinstance(_bv, str):
                _s = _bv.strip().lower()
                if _s in ("1", "true", "yes", "on"):
                    _nb = True
                elif _s in ("0", "false", "no", "off", ""):
                    _nb = False
                else:
                    _nb = _bdef
            else:
                _nb = _bdef
            if self.config.get(_bk) is not _nb:
                self.config[_bk] = _nb
                needs_save = True
        try:
            _vk = int(self.config.get("burn_vbr_kbps", 2000))
        except Exception:
            _vk = 2000
        if _vk < 300 or _vk > 10000:
            _vk = 2000
            needs_save = True
        if self.config.get("burn_vbr_kbps") != _vk:
            self.config["burn_vbr_kbps"] = _vk
            needs_save = True
        if self.config.get("burn_speed") not in ("match", "fast", "fastest",
                                                 "nvenc_draft", "nvenc_turbo",
                                                 "nvenc_balanced"):
            self.config["burn_speed"] = "match"
            needs_save = True
        try:
            _bc = str(self.config.get("burn_codec", "h264")).strip().lower()
        except Exception:
            _bc = "h264"
        if _bc not in ("h264", "hevc"):
            _bc = "h264"
        if self.config.get("burn_codec") != _bc:
            self.config["burn_codec"] = _bc
            needs_save = True
        if self.config.get("srt_tab") not in ("Live", "SRT File"):
            self.config["srt_tab"] = "Live"
            needs_save = True
        if needs_save:
            save_local_config(self.config)
        self._eng_lock = threading.Lock()
        self.moonshine_engine = TranscriptionEngine(language="en", model_arch=arch, on_ready=self._model_ready)
        self.canary_engine = None
        self.whisper_engine = None
        self._tab_sel = {}
        self._shared_sel = {"kind": "Moonshine v2", "arch": 5, "wmodel": "large-v3"}
        self._tab_cache = {
            "srt": {"canary": None, "whisper": None, "wmodel": None},
            "note": {"canary": None, "whisper": None, "wmodel": None},
        }
        self._note_engine_obj = None
        try:
            self._init_tab_engines()
        except Exception:
            pass
        if self.config.get("engine") == "Canary-1B":
            self.engine = self._get_canary_engine()
        elif self.config.get("engine") == "Whisper":
            self.engine = self._get_whisper_engine()
        else:
            self.engine = self.moonshine_engine
        self.recorder = AudioRecorder(sample_rate=SAMPLE_RATE)
        self._recording = False
        self._model_switching = False
        self._rec_lock = threading.Lock()
        self._key_pressed = False
        self._gui_queue: queue.Queue = queue.Queue()
        self.gui = None
        self._our_root_hwnd = 0
        self._target_top_hwnd = None
        self._target_child_hwnd = None
        self.audio_queue: queue.Queue = queue.Queue(maxsize=25)
        self.currently_processing = False
        self._processing_thread = threading.Thread(target=self._processing_worker, daemon=True)
        self._processing_thread.start()
        self._srt_busy = False
        self._srt_lock = threading.Lock()
        self._srt_thread = None
        self._srt_cancel = threading.Event()
        if MoonshineGUI is not None:
            try:
                self.gui = MoonshineGUI()
                self.gui.set_record_callback(self._gui_record_start, self._gui_record_stop)
                self.gui.set_tab_callback(self._on_tab_changed)
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
                try:
                    _eng = self.config.get("engine", "Moonshine v2")
                    if _eng == "Whisper":
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
                    self.gui.set_srt_callbacks(self._srt_start, self._cancel_srt_job,
                                               self._burn_start, self.srt_start_request)
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
                    try:
                        _auto = bool(self.config.get("burn_vbr_auto", True))
                        _kbps = int(self.config.get("burn_vbr_kbps", 2000))
                        _kbps = max(300, min(10000, _kbps))
                        self.gui.burn_vbr_auto_var.set(_auto)
                        self.gui.burn_vbr_slider.set(_kbps)
                        self.gui._on_burn_vbr_toggled()
                    except Exception:
                        pass
                    try:
                        self.gui.sample_start_var.set(
                            str(self.config.get("burn_sample_start", "0:30")))
                        _bsl = f"{int(self.config.get('burn_sample_len', 15))}s"
                        if _bsl in ("10s", "15s", "30s", "60s"):
                            self.gui.sample_len_var.set(_bsl)
                    except Exception:
                        pass
                    try:
                        from srt import BURN_SPEED_LABELS as _BSL
                        _bl = _BSL.get(self.config.get("burn_speed", "match"),
                                       _BSL.get("match"))
                        if _bl:
                            self.gui.burn_speed_var.set(_bl)
                        try:
                            self.gui._refresh_burn_speed_desc()
                        except Exception:
                            pass
                    except Exception:
                        pass
                    try:
                        self.gui.set_srt_languages(
                            self.config.get("srt_input_lang", "ja"),
                            self.config.get("srt_output_lang", "en"),
                            self._on_srt_input_lang_changed,
                            self._on_srt_output_lang_changed,
                        )
                        self.gui.set_srt_lang_state(self.config.get("engine", "Moonshine v2"))
                        try:
                            self.gui.refresh_lang_options(
                                self.config.get("engine", "Moonshine v2"))
                        except Exception:
                            pass
                    except Exception:
                        pass
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
                    try:
                        self.gui.set_compute(self.config.get("compute", "auto"),
                                             self._on_compute_changed)
                    except Exception:
                        pass
                    try:
                        self.gui.set_compute(self.config.get("compute", "auto"),
                                             self._on_compute_changed)
                    except Exception:
                        pass
                except Exception:
                    pass
                try:
                    self.gui.set_srt_options(
                        {k: self.config.get(k)
                         for k in ("srt_norm", "burn_after",
                                   "auto_shutdown", "completion_alert")},
                        self._save_srt_opts)
                except Exception:
                    pass
                try:
                    self.gui.set_theme(self.config.get("theme", "dark"))
                    self.gui.set_theme_callback(self._on_theme_changed)
                except Exception:
                    pass
                try:
                    self.gui.set_burn_codec(self.config.get("burn_codec", "h264"))
                except Exception:
                    pass
                try:
                    self.gui.set_footer_version(APP_VERSION)
                    self.gui.set_title_version(APP_VERSION)
                except Exception:
                    pass
                try:
                    self.gui.set_note_transcribe_fn(self._note_transcribe)
                except Exception:
                    pass
                try:
                    self.gui.set_note_record_callback(self.note_record_request,
                                                      self.note_record_confirm,
                                                      self.note_engine_ready)
                except Exception:
                    pass
                try:
                    self.gui.set_tab_engine_callbacks(
                        self.plan_tab_change, self.apply_tab_change,
                        lambda t: self.tab_selection(t))
                    for _t, _setter in (("srt", self.gui.set_srt_engine_state),
                                        ("note", self.gui.set_note_engine_state)):
                        try:
                            _s = self.tab_selection(_t)
                            _setter(_s["kind"], _s["arch"], _s["wmodel"])
                        except Exception:
                            pass
                except Exception:
                    pass

                self.gui.protocol("WM_DELETE_WINDOW", self._on_close)
                self.gui.update_idletasks()
                try:
                    self.gui.set_active_tab(self.config.get("srt_tab", "Live"))
                except Exception:
                    pass
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
        try:
            # Orphan session WAVs from crashed/hung runs (close-cleanup never
            # ran). Cheap directory scan; previous runs' MP3s are untouched.
            from pathlib import Path as _P
            from note_engine import cleanup_stale_session_wavs as _sweep
            _n = _sweep(_P(__file__).parent / "notes_audio")
            if _n:
                self._log(f"Cleared {_n} orphan note audio file(s) from a previous run")
        except Exception:
            pass
        try:
            self._log(f"PolyglotSTT v{APP_VERSION} (engine={self.config.get('engine', '?')}, "
                      f"compute={self.config.get('compute', '?')}, "
                      f"theme={self.config.get('theme', '?')})")
        except Exception:
            pass
    def _tab_canary(self, tab):
        """Dedicated tab instance for jobs (never the live singleton, so
        live switches can't rebind a running batch). None on failure -
        the job then fails loudly via _wait_for_model. Never raises."""
        try:
            if tab == "live":
                return self._get_canary_engine(False)
            sel = self.tab_selection(tab)
            if sel["kind"] != "Canary-1B":
                return None
            return self._tab_heavy(tab, "Canary-1B")
        except Exception:
            return None

    def _tab_whisper(self, tab):
        """Dedicated tab instance for jobs (see _tab_canary)."""
        try:
            if tab == "live":
                return self._get_whisper_engine(False)
            sel = self.tab_selection(tab)
            if sel["kind"] != "Whisper":
                return None
            return self._tab_heavy(tab, "Whisper", sel["wmodel"])
        except Exception:
            return None

    def _save_srt_opts(self, opts):
        try:
            with _CONFIG_LOCK:
                for k in ("srt_norm", "burn_after",
                          "auto_shutdown", "completion_alert"):
                    try:
                        if opts is not None and k in opts:
                            self.config[k] = bool(opts[k])
                    except Exception:
                        pass
                save_local_config(self.config)
        except Exception:
            pass
    def _on_theme_changed(self, mode):
        try:
            with _CONFIG_LOCK:
                self.config["theme"] = "light" if str(mode).lower() == "light" else "dark"
                save_local_config(self.config)
        except Exception:
            pass
    def _get_canary_engine(self, apply_live_options: bool = True):
        if self.canary_engine is None:
            with self._eng_lock:
                if self.canary_engine is None:
                    try:
                        from canary_engine import CanaryEngine
                        self.canary_engine = CanaryEngine(
                            task=self.config.get("canary_task", "transcribe"),
                            source_lang=self.config.get("canary_src_lang", "auto"),
                            target_lang="en",
                            device=self.config.get("compute", "auto"),
                            on_ready=self._model_ready,
                        )
                    except Exception as e:
                        print(f"Canary engine failed to create: {e}")
                        import traceback; traceback.print_exc()
                        return self.moonshine_engine
        if apply_live_options:
            try:
                self.canary_engine.switch_options(
                    task=self.config.get("canary_task"),
                    source_lang=self.config.get("canary_src_lang"),
                )
            except Exception:
                pass
            try:
                _dev = self.config.get("compute", "auto")
                if (_dev in ("auto", "cpu", "cuda", "gpu") and self.canary_engine is not None
                        and getattr(self.canary_engine, "device", "auto") != _dev):
                    self.canary_engine.device = _dev
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
                            device=self.config.get("compute", "auto"),
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
            try:
                _dev = self.config.get("compute", "auto")
                if (_dev in ("auto", "cpu", "cuda", "gpu") and self.whisper_engine is not None
                        and getattr(self.whisper_engine, "device", "auto") != _dev):
                    self.whisper_engine.device = _dev
            except Exception:
                pass
        return self.whisper_engine
    # ---------------- Per-tab engines (Live / SRT / Note) ----------------
    # Each tab picks its own engine+model. Live keeps the pre-existing
    # singleton behavior untouched; SRT/Note get dedicated heavy instances
    # (Moonshine is always the shared live object - tiny, serialized).
    # Rule: changing a tab while another session is ACTIVE asks first
    # (dual engines = dual RAM); with others idle, their cached instances
    # are simply unloaded. Selections mirror as shared defaults across
    # idle, non-diverged tabs. plan_* is pure logic (unit-testable, no Tk,
    # no loads); apply_* performs side effects (GUI thread only).
    TAB_IDS = ("live", "srt", "note")
    ENGINE_KINDS = ("Moonshine v2", "Canary-1B", "Whisper")
    _WHISPER_IDS = ("tiny", "base", "small", "medium", "large",
                    "large-v1", "large-v2", "large-v3")
    _ARCHES = (0, 1, 2, 3, 4, 5)
    # Rough extra-RAM for the dual-engine confirm dialog (MB).
    _ENGINE_RAM_MB = {"Moonshine v2": 300, "Canary-1B": 5500}
    _WHISPER_RAM_MB = {"tiny": 200, "base": 300, "small": 800,
                       "medium": 1800, "large": 3200, "large-v1": 3200,
                       "large-v2": 3200, "large-v3": 3200}

    @staticmethod
    def _norm_tab_sel(kind, arch, wmodel):
        try:
            k = str(kind if kind is not None else "Moonshine v2")
        except Exception:
            k = "Moonshine v2"
        if k not in MoonshineSTTApp.ENGINE_KINDS:
            k = "Moonshine v2"
        try:
            a = int(arch)
        except Exception:
            a = 5
        if a not in MoonshineSTTApp._ARCHES:
            a = 5
        try:
            w = str(wmodel if wmodel is not None else "large-v3")
        except Exception:
            w = "large-v3"
        if w not in MoonshineSTTApp._WHISPER_IDS:
            w = "large-v3"
        return {"kind": k, "arch": a, "wmodel": w}

    @staticmethod
    def _engine_ram_mb(kind, arch=None, wmodel=None):
        try:
            if kind == "Whisper":
                return int(MoonshineSTTApp._WHISPER_RAM_MB.get(
                    wmodel or "large-v3", 3200))
            return int(MoonshineSTTApp._ENGINE_RAM_MB.get(kind, 300))
        except Exception:
            return 3000

    def _init_tab_engines(self):
        """Seed per-tab selections (migrate legacy live keys). Never raises."""
        try:
            legacy = self._norm_tab_sel(self.config.get("engine", "Moonshine v2"),
                                        self.config.get("model_arch", 5),
                                        self.config.get("whisper_model", "large-v3"))
        except Exception:
            legacy = {"kind": "Moonshine v2", "arch": 5, "wmodel": "large-v3"}
        needs_save = False
        try:
            raw = self.config.get("tab_engines")
            shared_raw = self.config.get("shared_engine")
        except Exception:
            raw, shared_raw = None, None
        tabs = {}
        if isinstance(raw, dict):
            for t in self.TAB_IDS:
                try:
                    node = raw.get(t) or {}
                    tabs[t] = {
                        "kind": node.get("kind", legacy["kind"]),
                        "arch": node.get("arch", legacy["arch"]),
                        "wmodel": node.get("wmodel", legacy["wmodel"]),
                        "explicit": bool(node.get("explicit", False)),
                    }
                except Exception:
                    tabs[t] = dict(legacy, explicit=False)
        else:
            for t in self.TAB_IDS:
                tabs[t] = dict(legacy, explicit=False)
            needs_save = True
        for t in self.TAB_IDS:
            try:
                n = self._norm_tab_sel(tabs[t].get("kind"), tabs[t].get("arch"),
                                       tabs[t].get("wmodel"))
                tabs[t].update(n)
                tabs[t]["explicit"] = bool(tabs[t].get("explicit", False))
            except Exception:
                tabs[t] = dict(legacy, explicit=False)
        try:
            if isinstance(shared_raw, dict):
                shared = self._norm_tab_sel(shared_raw.get("kind"),
                                            shared_raw.get("arch"),
                                            shared_raw.get("wmodel"))
            else:
                shared = dict(legacy)
                needs_save = True
        except Exception:
            shared = dict(legacy)
        self._tab_sel = tabs
        self._shared_sel = {"kind": shared["kind"], "arch": shared["arch"],
                            "wmodel": shared["wmodel"]}
        self._tab_cache = {
            "srt": {"canary": None, "whisper": None, "wmodel": None},
            "note": {"canary": None, "whisper": None, "wmodel": None},
        }
        self._note_engine_obj = None
        try:
            with _CONFIG_LOCK:
                self.config["tab_engines"] = {t: dict(v) for t, v in tabs.items()}
                self.config["shared_engine"] = dict(self._shared_sel)
                if needs_save:
                    save_local_config(self.config)
        except Exception:
            pass

    def _save_tab_sel(self):
        try:
            with _CONFIG_LOCK:
                self.config["tab_engines"] = {t: dict(v)
                                              for t, v in self._tab_sel.items()}
                self.config["shared_engine"] = dict(self._shared_sel)
                save_local_config(self.config)
        except Exception:
            pass

    def tab_selection(self, tab):
        """Validated (kind, arch, wmodel) selection for a tab. Live reads
        the long-standing config keys (unchanged behavior); srt/note read
        the registry. Never raises."""
        try:
            if tab == "live":
                return self._norm_tab_sel(self.config.get("engine", "Moonshine v2"),
                                          self.config.get("model_arch", 5),
                                          self.config.get("whisper_model", "large-v3"))
            node = (self._tab_sel or {}).get(tab) or {}
            return self._norm_tab_sel(node.get("kind"), node.get("arch"),
                                      node.get("wmodel"))
        except Exception:
            return {"kind": "Moonshine v2", "arch": 5, "wmodel": "large-v3"}

    def _describe_tab_engine(self, tab):
        """Human label of what a tab would run, e.g. 'Whisper medium'."""
        try:
            sel = self.tab_selection(tab)
            k = sel["kind"]
            if k == "Moonshine v2":
                try:
                    from engine import MODEL_ARCH_NAMES
                    name = MODEL_ARCH_NAMES.get(int(sel["arch"]), str(sel["arch"]))
                except Exception:
                    name = str(sel["arch"])
                return f"Moonshine {name}"
            if k == "Whisper":
                return f"Whisper {sel['wmodel']}"
            return "Canary-1B"
        except Exception:
            return "?"

    def _live_session_active(self):
        try:
            if bool(getattr(self, "_recording", False)):
                return True
            try:
                if self.audio_queue.qsize() > 0:
                    return True
            except Exception:
                pass
            if bool(getattr(self, "currently_processing", False)):
                return True
            if bool(getattr(self, "_model_switching", False)):
                return True
            for e in (getattr(self, "moonshine_engine", None),
                      getattr(self, "canary_engine", None),
                      getattr(self, "whisper_engine", None)):
                try:
                    if e is not None and getattr(e, "is_ready", True) is False \
                            and getattr(e, "_loading", False):
                        return True
                except Exception:
                    pass
        except Exception:
            pass
        return False

    def _note_session_active(self):
        try:
            g = self.gui
            if g is None:
                return False
            try:
                if bool(g.note_session_active()):
                    return True
            except Exception:
                pass
            if bool(getattr(g, "_note_recording", False)):
                return True
            try:
                sub = int(getattr(g, "_note_submitted", 0) or 0)
                done = int(getattr(g, "_note_done", 0) or 0)
                if sub > done:
                    return True
            except Exception:
                pass
        except Exception:
            pass
        return False

    def tab_active(self, tab):
        """Is this tab's session live right now (recording / processing /
        loading)? Never raises; missing GUI counts as inactive."""
        try:
            if tab == "live":
                return self._live_session_active()
            if tab == "srt":
                return bool(getattr(self, "_srt_busy", False))
            if tab == "note":
                return self._note_session_active()
        except Exception:
            pass
        return False

    def _tab_active_desc(self, tab):
        try:
            if tab == "live":
                if bool(getattr(self, "_recording", False)):
                    return "Live is recording"
                return "Live is processing"
            if tab == "srt":
                return "SRT batch is running"
            if tab == "note":
                try:
                    if self.gui is not None and bool(getattr(self.gui, "_note_recording", False)):
                        return "Note is recording"
                except Exception:
                    pass
                return "Note is transcribing"
        except Exception:
            pass
        return "session active"

    def plan_tab_change(self, tab, kind=None, arch=None, wmid=None):
        """Pure verdict for a tab engine/model pick (no Tk, no loads).
        kind/arch/wmid None = keep current. Returns {"action":...}:
        ok | confirm{message, ram_mb, others} | revert{reason}."""
        try:
            if tab not in self.TAB_IDS:
                return {"action": "revert", "reason": "unknown tab"}
            cur = self.tab_selection(tab)
            want_kind = str(kind) if kind is not None else cur["kind"]
            if want_kind not in self.ENGINE_KINDS:
                return {"action": "revert", "reason": "unknown engine"}
            want_arch, want_wmid = cur["arch"], cur["wmodel"]
            if arch is not None:
                try:
                    want_arch = int(arch)
                except Exception:
                    return {"action": "revert", "reason": "bad model"}
                if want_arch not in self._ARCHES:
                    return {"action": "revert", "reason": "bad model"}
            if wmid is not None:
                want_wmid = str(wmid)
                if want_wmid not in self._WHISPER_IDS:
                    return {"action": "revert", "reason": "bad model"}
            if want_kind == cur["kind"] and want_arch == cur["arch"] \
                    and want_wmid == cur["wmodel"]:
                return {"action": "ok", "kind": want_kind, "arch": want_arch,
                        "wmodel": want_wmid, "noop": True, "mirrored": []}
            others = []
            for t in self.TAB_IDS:
                if t == tab:
                    continue
                try:
                    if self.tab_active(t):
                        others.append((t, self._tab_active_desc(t),
                                       self.tab_selection(t)))
                except Exception:
                    pass
            if others:
                ram = self._engine_ram_mb(want_kind, want_arch, want_wmid)
                who = "; ".join(d for _, d, _s in others)
                if want_kind == "Moonshine v2":
                    what = f"Moonshine (shared live engine, ~{ram} MB)"
                elif want_kind == "Canary-1B":
                    what = f"Canary-1B (~{ram / 1000:.1f} GB extra RAM)"
                else:
                    what = f"Whisper {want_wmid} (~{ram / 1000:.1f} GB extra RAM)"
                return {"action": "confirm", "kind": want_kind,
                        "arch": want_arch, "wmodel": want_wmid,
                        "ram_mb": ram, "others": [t for t, _d, _s in others],
                        "message": f"{who} still active.\nLoad {what} "
                                   f"alongside it too?\nYes = run both engines "
                                   f"(heavy RAM). No = stay as you are."}
            return {"action": "ok", "kind": want_kind, "arch": want_arch,
                    "wmodel": want_wmid, "noop": False, "mirrored": []}
        except Exception as e:
            return {"action": "revert", "reason": str(e)}

    def _drop_tab_cache(self, tab):
        """Unload + forget a tab's dedicated heavy instances. Returns how
        many live objects were dropped. Never raises."""
        dropped = 0
        try:
            cache = (self._tab_cache or {}).get(tab)
            if not isinstance(cache, dict):
                return 0
            for slot in ("canary", "whisper"):
                try:
                    eng = cache.get(slot)
                    if eng is not None:
                        dropped += 1
                        try:
                            eng.unload()
                        except Exception:
                            pass
                except Exception:
                    pass
                try:
                    cache[slot] = None
                except Exception:
                    pass
            try:
                cache["wmodel"] = None
            except Exception:
                pass
        except Exception:
            pass
        return dropped

    def _make_engine(self, kind, arch=None, wmid=None):
        """Build an (unloaded) dedicated engine for srt/note tabs.
        Monkeypatchable in tests - production constructs the real ones."""
        try:
            if kind == "Canary-1B":
                from canary_engine import CanaryEngine
                return CanaryEngine(
                    task=self.config.get("canary_task", "transcribe"),
                    source_lang=self.config.get("canary_src_lang", "auto"),
                    target_lang="en",
                    device=self.config.get("compute", "auto"),
                    on_ready=None,
                )
            if kind == "Whisper":
                from whisper_engine import WhisperEngine
                return WhisperEngine(
                    task=self.config.get("whisper_task", "translate"),
                    source_lang=self.config.get("whisper_src_lang", "ja"),
                    target_lang="en",
                    model_id=wmid or self.config.get("whisper_model", "large-v3"),
                    device=self.config.get("compute", "auto"),
                    on_ready=None,
                )
        except Exception as e:
            print(f"tab engine build failed: {e}")
        return None

    @staticmethod
    def _whisper_cache_ok(eng, want):
        """Cached dedicated Whisper instance matches the wanted model id.
        Missing model_id attr + ready counts as a match (never rebuild-loop
        on exotic engine objects)."""
        try:
            if eng is None:
                return False
            try:
                got = str(getattr(eng, "model_id", "") or "")
            except Exception:
                got = ""
            if not got:
                return bool(getattr(eng, "is_ready", False))
            return bool(getattr(eng, "is_ready", False)) and got == str(want)
        except Exception:
            return False

    def _tab_heavy(self, tab, kind, wmid=None):
        """Get-or-create this tab's dedicated Canary/Whisper instance (no
        load). Moonshine returns None (shared live object). Never raises."""
        try:
            if kind == "Canary-1B":
                slot = "canary"
            elif kind == "Whisper":
                slot = "whisper"
            else:
                return None
            cache = (self._tab_cache or {}).get(tab)
            if not isinstance(cache, dict):
                return None
            eng = cache.get(slot)
            if slot == "whisper":
                try:
                    want = str(wmid or self.tab_selection(tab)["wmodel"])
                except Exception:
                    want = "large-v3"
                if eng is not None and not self._whisper_cache_ok(eng, want):
                    try:
                        eng.unload()
                    except Exception:
                        pass
                    eng = None
                if eng is None:
                    eng = self._make_engine(kind, None, want)
                    try:
                        cache[slot] = eng
                        cache["wmodel"] = want
                    except Exception:
                        pass
            else:
                if eng is None:
                    eng = self._make_engine(kind)
                    try:
                        cache[slot] = eng
                    except Exception:
                        pass
            return eng
        except Exception:
            return None

    def apply_tab_change(self, tab, kind, arch, wmid, dual_ok=False):
        """Apply a planned change (GUI thread only - syncs follower menus).
        Unloads other tabs' idle caches; preloads live/note selections.
        Returns {"action": "ok", "mirrored": [(tab, kind, arch, wmid), ...]}
        or {"action": "revert", "reason"}. Never raises."""
        try:
            if tab not in self.TAB_IDS:
                return {"action": "revert", "reason": "unknown tab"}
            # None = keep this tab's current value (callers pass verdict
            # fields through; GUI verdicts are always fully resolved).
            try:
                _cur0 = self.tab_selection(tab)
            except Exception:
                _cur0 = {"kind": "Moonshine v2", "arch": 5, "wmodel": "large-v3"}
            if kind is None:
                kind = _cur0["kind"]
            if arch is None:
                arch = _cur0["arch"]
            if wmid is None:
                wmid = _cur0["wmodel"]
            norm = self._norm_tab_sel(kind, arch, wmid)
            kind, arch, wmid = norm["kind"], norm["arch"], norm["wmodel"]
            if not dual_ok:
                for t in self.TAB_IDS:
                    if t != tab:
                        try:
                            if self.tab_active(t):
                                return {"action": "revert",
                                        "reason": "other session active"}
                        except Exception:
                            pass
            # Commit selection + shared default.
            try:
                if tab == "live":
                    with _CONFIG_LOCK:
                        self.config["engine"] = kind
                        if kind == "Moonshine v2":
                            self.config["model_arch"] = int(arch)
                        elif kind == "Whisper":
                            self.config["whisper_model"] = str(wmid)
                        save_local_config(self.config)
                node = (self._tab_sel or {}).get(tab)
                if isinstance(node, dict):
                    node.update({"kind": kind, "arch": int(arch),
                                 "wmodel": str(wmid), "explicit": True})
                self._shared_sel = {"kind": kind, "arch": int(arch),
                                    "wmodel": str(wmid)}
                self._save_tab_sel()
            except Exception as e:
                return {"action": "revert", "reason": str(e)}
            # Mirror to idle, non-diverged followers (srt/note only - live
            # is the anchor: its menus always show the running engine).
            # Callers sync follower menus from the returned list.
            mirrored = []
            try:
                for t in ("srt", "note"):
                    if t == tab:
                        continue
                    try:
                        if self.tab_active(t):
                            continue
                        node = (self._tab_sel or {}).get(t)
                        if not isinstance(node, dict):
                            continue
                        if bool(node.get("explicit", False)):
                            continue
                        node.update({"kind": kind, "arch": int(arch),
                                     "wmodel": str(wmid)})
                        mirrored.append((t, kind, int(arch), str(wmid)))
                    except Exception:
                        pass
                self._save_tab_sel()
            except Exception:
                pass
            # Lifecycle: drop this tab's stale heavies, then every OTHER
            # tab's cache that is idle right now. Active sessions always
            # keep theirs (their turn comes when they go idle).
            try:
                self._drop_tab_cache(tab)
            except Exception:
                pass
            try:
                freed_tabs = []
                for t in self.TAB_IDS:
                    if t == tab:
                        continue
                    try:
                        if self.tab_active(t):
                            continue
                    except Exception:
                        pass
                    try:
                        if self._drop_tab_cache(t) > 0:
                            freed_tabs.append(t)
                    except Exception:
                        pass
                if freed_tabs:
                    try:
                        self._log("Unloaded idle engine(s) for %s (RAM reclaimed)"
                                  % ", ".join(freed_tabs))
                    except Exception:
                        pass
            except Exception:
                pass
            # Preload for interactive tabs (SRT loads inside its job w/ progress).
            try:
                if tab in ("live", "note") and kind in ("Canary-1B", "Whisper"):
                    if tab == "live":
                        pass  # existing live flow below loads it
                    else:
                        eng = self._tab_heavy(tab, kind, wmid)
                        try:
                            if eng is not None and not eng.is_ready:
                                eng.load()
                        except Exception:
                            pass
            except Exception:
                pass
            # Push follower menu states (best effort, GUI thread).
            # Live menus are owned by the existing engine/model flows, which
            # repaint from config as they continue - mirrored srt/note menus
            # are synced by the caller from the returned list.
            return {"action": "ok", "mirrored": mirrored}
        except Exception as e:
            return {"action": "revert", "reason": str(e)}

    def engine_start_cost(self, tab):
        """(need_new_heavy_load, ram_mb, describe) for starting work on tab
        now. Moonshine never needs a dialog (shared object, tiny load)."""
        try:
            sel = self.tab_selection(tab)
            kind = sel["kind"]
            if kind == "Moonshine v2":
                return False, 0, "Moonshine (shared)"
            if kind == "Canary-1B":
                try:
                    eng = ((self._tab_cache or {}).get(tab) or {}).get("canary")
                    ready = bool(eng is not None and eng.is_ready)
                except Exception:
                    ready = False
                if ready:
                    return False, 0, "Canary-1B (ready)"
                return True, self._engine_ram_mb(kind), "Canary-1B (not loaded)"
            # Whisper
            try:
                want = str(sel["wmodel"])
                eng = ((self._tab_cache or {}).get(tab) or {}).get("whisper")
                ready = self._whisper_cache_ok(eng, want)
            except Exception:
                ready = False
            if ready:
                return False, 0, f"Whisper {sel['wmodel']} (ready)"
            return True, self._engine_ram_mb(kind, None, sel["wmodel"]), \
                f"Whisper {sel['wmodel']} (not loaded)"
        except Exception:
            return False, 0, "?"

    def srt_start_request(self):
        """GUI-thread pre-flight for the SRT start button. Returns
        {"go": True} | {"confirm": message, "ram_mb": n} (user picks in
        dialog; Yes -> spawn job, No -> abort start)."""
        try:
            need, ram, _desc = self.engine_start_cost("srt")
            if not need:
                return {"go": True}
            others = []
            for t in self.TAB_IDS:
                if t == "srt":
                    continue
                try:
                    if self.tab_active(t):
                        others.append(self._tab_active_desc(t))
                except Exception:
                    pass
            if not others:
                return {"go": True}
            sel = self.tab_selection("srt")
            what = "Canary-1B" if sel["kind"] == "Canary-1B" \
                else f"Whisper {sel['wmodel']}"
            return {"confirm": f"{'; '.join(others)} still active.\n"
                               f"Start the SRT job with {what} too?\n"
                               f"Yes = run both engines (~{ram / 1000:.1f} GB "
                               f"extra RAM). No = stay as you are.",
                    "ram_mb": ram}
        except Exception:
            return {"go": True}

    def note_record_request(self):
        """GUI-thread pre-flight for Note RECORD. Returns {"go": True}
        (engine snapshotted) | {"confirm": ...} | {"wait": msg} | {"abort"}.
        Moonshine uses the shared live object (snapshot it); heavies must
        be cached+ready (preloaded at selection) or they load now."""
        try:
            sel = self.tab_selection("note")
            kind = sel["kind"]
            if kind == "Moonshine v2":
                try:
                    eng = self.moonshine_engine
                    if eng is not None and eng.is_ready:
                        self._note_engine_obj = eng
                        return {"go": True}
                except Exception:
                    pass
                return {"wait": "Note engine (Moonshine) still loading — "
                                "recording starts automatically."}
            try:
                cache = (self._tab_cache or {}).get("note") or {}
            except Exception:
                cache = {}
            slot = "canary" if kind == "Canary-1B" else "whisper"
            eng = cache.get(slot)
            ok = False
            try:
                if eng is not None and eng.is_ready:
                    if slot == "whisper":
                        ok = self._whisper_cache_ok(eng, sel["wmodel"])
                    else:
                        ok = True
            except Exception:
                ok = False
            if ok:
                self._note_engine_obj = eng
                return {"go": True}
            # Need a load: dual cost only matters against active others.
            others = []
            for t in self.TAB_IDS:
                if t == "note":
                    continue
                try:
                    if self.tab_active(t):
                        others.append(self._tab_active_desc(t))
                except Exception:
                    pass
            if others:
                ram = self._engine_ram_mb(kind, sel["arch"], sel["wmodel"])
                what = "Canary-1B" if kind == "Canary-1B" \
                    else f"Whisper {sel['wmodel']}"
                return {"confirm": f"{'; '.join(others)} still active.\n"
                                   f"Load {what} for Note too?\n"
                                   f"Yes = run both engines (~{ram / 1000:.1f} GB "
                                   f"extra RAM), then press record again. "
                                   f"No = stay as you are.",
                        "ram_mb": ram}
            try:
                fresh = self._tab_heavy("note", kind, sel["wmodel"])
                if fresh is not None and not fresh.is_ready:
                    fresh.load()
            except Exception:
                pass
            return {"wait": "Loading note engine — recording starts automatically."}
        except Exception:
            return {"abort": True}

    def note_engine_ready(self):
        """(ready_bool, problem_str) for the Note tab's current selection.
        Kicks a missing load (never blocks). GUI thread safe."""
        try:
            sel = self.tab_selection("note")
            kind = sel["kind"]
            if kind == "Moonshine v2":
                try:
                    eng = self.moonshine_engine
                    if eng is not None and eng.is_ready:
                        return True, ""
                    if eng is not None and not getattr(eng, "_loading", False):
                        try:
                            eng.load()
                        except Exception:
                            pass
                except Exception:
                    pass
                return False, ""
            try:
                cache = (self._tab_cache or {}).get("note") or {}
            except Exception:
                cache = {}
            slot = "canary" if kind == "Canary-1B" else "whisper"
            eng = cache.get(slot)
            if eng is None:
                try:
                    eng = self._tab_heavy("note", kind, sel["wmodel"])
                    if eng is not None and not eng.is_ready:
                        eng.load()
                except Exception:
                    pass
                return False, ""
            try:
                if eng.is_ready:
                    if slot == "whisper" and not self._whisper_cache_ok(
                            eng, sel["wmodel"]):
                        return False, ""
                    return True, ""
            except Exception:
                pass
            try:
                if not getattr(eng, "_loading", True):
                    err = getattr(eng, "_last_error", None) or "unknown error"
                    return False, f"Note engine failed: {err}"
            except Exception:
                pass
            return False, ""
        except Exception:
            return False, ""

    def note_record_confirm(self, dual_ok):
        """Follow-up after the record pre-flight asked to confirm."""
        try:
            if not dual_ok:
                return {"abort": True}
            sel = self.tab_selection("note")
            try:
                fresh = self._tab_heavy("note", sel["kind"], sel["wmodel"])
                if fresh is not None and not fresh.is_ready:
                    fresh.load()
            except Exception:
                pass
            return {"wait": "Loading note engine — recording starts automatically."}
        except Exception:
            return {"abort": True}

    def tab_shown(self, tabid):
        """Tab-switch adopt: an srt/note tab that was never explicitly
        picked follows the shared default (menus only - engines load at
        use). Live is the anchor and never auto-adopts. Never raises."""
        try:
            mapping = {"SRT File": "srt", "Note": "note"}
            tab = mapping.get(tabid)
            if tab is None:
                return
            node = (self._tab_sel or {}).get(tab)
            if not isinstance(node, dict) or bool(node.get("explicit", False)):
                return
            try:
                sh = self._shared_sel
            except Exception:
                return
            node.update({"kind": sh["kind"], "arch": int(sh["arch"]),
                         "wmodel": str(sh["wmodel"])})
            self._save_tab_sel()
            try:
                g = self.gui
                if g is not None:
                    if tab == "srt":
                        g.set_srt_engine_state(sh["kind"], int(sh["arch"]),
                                               str(sh["wmodel"]))
                    elif tab == "note":
                        g.set_note_engine_state(sh["kind"], int(sh["arch"]),
                                                str(sh["wmodel"]))
            except Exception:
                pass
        except Exception:
            pass

    def _sync_mirrored_menus(self, mirrored):
        """Paint follower (srt/note) menu states after an apply. The live
        tab owns its menus via the engine/model flows. Never raises."""
        try:
            g = self.gui
            if g is None:
                return
            for (t, k, a, w) in (mirrored or []):
                try:
                    if t == "srt":
                        g.set_srt_engine_state(k, a, w)
                    elif t == "note":
                        g.set_note_engine_state(k, a, w)
                except Exception:
                    pass
        except Exception:
            pass

    def _sync_live_menus(self):
        """Repaint Live engine/model rows from config (mirror revert path).
        Never raises; programmatic sets don't fire menu commands."""
        try:
            g = self.gui
            if g is None:
                return
            _eng = self.config.get("engine", "Moonshine v2")
            try:
                if _eng == "Whisper":
                    _t = self.config.get("whisper_task", "translate")
                    _s = self.config.get("whisper_src_lang", "ja")
                else:
                    _t = self.config.get("canary_task", "transcribe")
                    _s = self.config.get("canary_src_lang", "auto")
                g.set_engine(_eng, _t, _s, self._on_engine_changed,
                             self._on_canary_task_changed,
                             self._on_canary_lang_changed)
            except Exception:
                pass
            try:
                self._refresh_model_row()
            except Exception:
                pass
        except Exception:
            pass

    def _unload_idle_tab_engines(self):
        """Unload stale tab caches (selection drifted) and never touch an
        active tab's instances. Called with the regular idle sweep."""
        try:
            for tab in ("srt", "note"):
                try:
                    if self.tab_active(tab):
                        continue
                    sel = self.tab_selection(tab)
                    cache = (self._tab_cache or {}).get(tab)
                    if not isinstance(cache, dict):
                        continue
                    # Whisper slot stale?
                    try:
                        eng = cache.get("whisper")
                        if eng is not None:
                            try:
                                same = (sel["kind"] == "Whisper"
                                        and str(getattr(eng, "model_id", "") or "")
                                        == str(sel["wmodel"]))
                            except Exception:
                                same = False
                            if not same:
                                try:
                                    eng.unload()
                                except Exception:
                                    pass
                                try:
                                    cache["whisper"] = None
                                    cache["wmodel"] = None
                                except Exception:
                                    pass
                    except Exception:
                        pass
                    # Canary slot stale?
                    try:
                        eng = cache.get("canary")
                        if eng is not None and sel["kind"] != "Canary-1B":
                            try:
                                eng.unload()
                            except Exception:
                                pass
                            try:
                                cache["canary"] = None
                            except Exception:
                                pass
                    except Exception:
                        pass
                except Exception:
                    pass
        except Exception:
            pass

    def _on_tab_changed(self, name: str):
        if name not in ("Live", "SRT File", "Note"):
            return
        if name in ("Live", "SRT File"):
            with _CONFIG_LOCK:
                self.config["srt_tab"] = name
                save_local_config(self.config)
        # Adopt-shared-default for never-explicitly-picked tabs.
        try:
            mapping = {"Live": "live", "SRT File": "srt", "Note": "note"}
            if name in mapping:
                self.tab_shown(mapping[name])
        except Exception:
            pass
    def _on_compute_changed(self, display: str):
        code = {"Auto": "auto", "CPU": "cpu", "GPU": "gpu"}.get(
            (display or "").strip(), "auto")
        if code == "gpu":
            try:
                import gpu as _gpumod
                if _gpumod.best_gpu() is None:
                    raise RuntimeError("no dGPU")
            except Exception:
                self._log("No NVIDIA GPU found - staying on CPU")
                if self.gui:
                    try:
                        self.gui.set_compute(self.config.get("compute", "auto"),
                                             self._on_compute_changed)
                    except Exception:
                        pass
                return
        old = self.config.get("compute", "auto")
        if code == old:
            return
        with _CONFIG_LOCK:
            self.config["compute"] = code
            save_local_config(self.config)
        try:
            import gpu as _gpumod2
            _gpumod2.refresh_gpu_cache()
        except Exception:
            pass
        self._log(f"Compute -> {display} (applies on engine load)")
        active = self.config.get("engine", "Moonshine v2")
        target = None
        if active == "Whisper" and self.whisper_engine is not None:
            try:
                self.whisper_engine.device = code
            except Exception:
                pass
            target = self.whisper_engine
        elif active == "Canary-1B" and self.canary_engine is not None:
            try:
                self.canary_engine.device = code
            except Exception:
                pass
            target = self.canary_engine
        else:
            if self.gui:
                self.gui.set_status(f"Compute: {display} (Moonshine is CPU-only)", SUCCESS)
            return
        try:
            if code in ("cpu", "cuda", "gpu"):
                want = "cpu" if code == "cpu" else "cuda"
                if (getattr(target, "is_ready", False)
                        and getattr(target, "_device_used", "") == want):
                    self._log(f"Compute: already on {display}, no reload")
                    if self.gui:
                        self.gui.set_status(f"Ready \u2022 compute {display}", SUCCESS)
                    return
        except Exception:
            pass
        if self.gui:
            self.gui.set_status(f"Switching compute to {display}...", WARNING)
            try:
                self.gui.record_btn.configure(state="disabled")
            except Exception:
                pass
        self._model_switching = True
        def _compute_switched(success, err):
            def _ui():
                self._model_switching = False
                try:
                    self.gui.record_btn.configure(state="normal")
                except Exception:
                    pass
                if success:
                    self.gui.set_status(f"Ready \u2022 compute {display}", SUCCESS)
                else:
                    self.gui.set_status(f"Compute error: {err}", DANGER)
            if self.gui:
                try:
                    self.gui.after(0, _ui)
                except Exception:
                    self._model_switching = False
            else:
                self._model_switching = False
        try:
            target.load()
            threading.Thread(target=self._watch_compute_load,
                             args=(target, _compute_switched), daemon=True).start()
        except Exception as e:
            self._model_switching = False
            _compute_switched(False, str(e))
    def _watch_compute_load(self, eng, done_cb):
        import time as _t
        for _ in range(900):
            try:
                if getattr(eng, "is_ready", False):
                    done_cb(True, None)
                    return
                if not getattr(eng, "_loading", True):
                    done_cb(False, getattr(eng, "_last_error", None) or "load failed")
                    return
            except Exception:
                pass
            _t.sleep(1)
        try:
            done_cb(False, "timed out")
        except Exception:
            pass
    def _on_method_changed(self, value):
        self.config["typing_method"] = value
        save_local_config(self.config)
        self._log(f"Typing method -> {value}")
    def _on_suffix_changed(self, value):
        self.config["suffix"] = value
        save_local_config(self.config)
        self._log(f"Suffix -> {value}")
    def _model_status_fn(self):
        try:
            eng = self.config.get("engine", "Moonshine v2")
            if eng == "Whisper":
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
        if not self.gui:
            return
        try:
            eng = self.config.get("engine", "Moonshine v2")
            try:
                self.gui.set_model_title(eng)
            except Exception:
                pass
            fn = self._model_status_fn()
            if eng == "Whisper":
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
        from gui import MODEL_CHOICES, WHISPER_MODEL_CHOICES
        items = []
        total = 0
        try:
            active = self.config.get("engine", "Moonshine v2")
        except Exception:
            active = "Moonshine v2"
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
            items.append({"engine": "Whisper", "kind": "whisper",
                          "id": mid, "label": label,
                          "size": size if dl else None,
                          "downloaded": dl,
                          "in_use": active == "Whisper" and wmid == mid})
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
                if (self.config.get("engine") == "Whisper"
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
        if self.config.get("engine") == "Whisper":
            self._on_whisper_model_changed(display_label)
            return
        if self.config.get("engine") not in ("Moonshine v2",):
            return
        try:
            from gui import MODEL_CHOICES
            new_arch = MODEL_CHOICES.get(display_label, 5)
        except Exception:
            new_arch = 5
        old_arch = int(self.config.get("model_arch", 5))
        if new_arch == old_arch:
            return
        try:
            if not self._live_engine_pick("Moonshine v2", new_arch, None):
                return
        except Exception:
            pass
        self.config["model_arch"] = new_arch
        save_local_config(self.config)
        self._log(f"Model -> {display_label} (arch {new_arch}), reloading...")
        if self.gui:
            self.gui.set_model_status(f"Switching to {display_label}...", WARNING)
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
            self._log("Stop the SRT/burn job before switching Whisper model")
            try:
                self._refresh_model_row()
            except Exception:
                pass
            return
        try:
            if not self._live_engine_pick("Whisper", None, new_id):
                return
        except Exception:
            pass
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
            self._model_switching = False
            _whisper_switched(False, "engine unavailable")
    def _ask_dual(self, message):
        """GUI-thread Yes/No dialog for dual-engine loads. False on any
        failure or headless run (safe direction: don't load)."""
        try:
            from tkinter import messagebox as _mb
            try:
                parent = self.gui if self.gui is not None else None
            except Exception:
                parent = None
            return bool(_mb.askyesno("Load second engine?",
                                     str(message or "Another session is active."),
                                     parent=parent))
        except Exception:
            return False

    def _live_engine_pick(self, kind, arch=None, wmid=None):
        """Shared dual-guard for the three live menu flows (engine, model,
        whisper size). Runs plan -> optional dialog -> apply -> follower
        menu sync. Returns True when the caller should proceed with its
        existing load flow; False means reverted (menus restored).
        GUI thread. Never raises (fail-open: existing flow validates)."""
        try:
            _pv = self.plan_tab_change("live", kind=kind, arch=arch,
                                       wmid=wmid)
        except Exception:
            return True
        try:
            act = (_pv or {}).get("action")
            if act == "confirm":
                _yes = self._ask_dual((_pv or {}).get("message"))
                _ap = self.apply_tab_change(
                    "live", (_pv or {}).get("kind"), (_pv or {}).get("arch"),
                    (_pv or {}).get("wmodel"), dual_ok=bool(_yes))
                if not _yes or (_ap or {}).get("action") != "ok":
                    try:
                        self._sync_live_menus()
                    except Exception:
                        pass
                    return False
            elif act == "revert":
                try:
                    self._sync_live_menus()
                except Exception:
                    pass
                return False
            else:
                _ap = self.apply_tab_change(
                    "live", (_pv or {}).get("kind", kind),
                    (_pv or {}).get("arch", arch),
                    (_pv or {}).get("wmodel", wmid), dual_ok=True)
                if (_ap or {}).get("action") != "ok":
                    try:
                        self._sync_live_menus()
                    except Exception:
                        pass
                    return False
            try:
                self._sync_mirrored_menus((_ap or {}).get("mirrored"))
            except Exception:
                pass
            return True
        except Exception:
            return True

    def _on_engine_changed(self, display_label: str):
        old = self.config.get("engine", "Moonshine v2")
        if display_label == old:
            return
        # Per-tab rule: another live session blocks silent switches.
        try:
            if not self._live_engine_pick(display_label):
                return
        except Exception:
            pass
        self.config["engine"] = display_label
        save_local_config(self.config)
        self._log(f"Engine -> {display_label}")
        try:
            if self.gui:
                if display_label == "Whisper":
                    self.gui.canary_task_var.set(self.config.get("whisper_task", "translate"))
                    self.gui.canary_lang_var.set(self.config.get("whisper_src_lang", "ja"))
                elif display_label == "Canary-1B":
                    self.gui.canary_task_var.set(self.config.get("canary_task", "transcribe"))
                    self.gui.canary_lang_var.set(self.config.get("canary_src_lang", "auto"))
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
                try:
                    self.gui.set_srt_lang_state(display_label)
                except Exception:
                    pass
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
                new_engine.load()
            else:
                if self.gui:
                    try:
                        self.gui.after(0, lambda e=new_engine: self.gui.set_status(f"Ready \u2022 Canary-1B ({e.task} {e.source_lang}->en)", SUCCESS))
                        self.gui.after(0, lambda: self.gui.record_btn.configure(state="normal"))
                    except Exception:
                        pass
        elif display_label == "Whisper":
            new_engine = self._get_whisper_engine()
            self.engine = new_engine
            if self.gui:
                self.gui.set_status("Loading Whisper...", WARNING)
                try:
                    self.gui.record_btn.configure(state="disabled")
                except Exception:
                    pass
            if not new_engine.is_ready:
                new_engine.load()
            else:
                if self.gui:
                    try:
                        self.gui.after(0, lambda e=new_engine: self.gui.set_status(f"Ready \u2022 Whisper ({e.task} {e.source_lang}->en)", SUCCESS))
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
        self._unload_idle_engines()
    def _unload_idle_engines(self):
        """Release whichever heavy engine is NOT active (multi-GB RAM back)."""
        try:
            self._unload_idle_tab_engines()
        except Exception:
            pass
        try:
            if self._srt_busy:
                return
            try:
                if self.audio_queue.qsize() > 0 or self.currently_processing:
                    return
            except Exception:
                pass
            try:
                if self.gui is not None and getattr(self.gui, "_note_recording", False):
                    return
            except Exception:
                pass
            active = self.config.get("engine", "Moonshine v2")
            for name in ("canary_engine", "whisper_engine", "moonshine_engine"):
                try:
                    eng = getattr(self, name, None)
                    if eng is None or eng is self.engine:
                        continue
                    if active == "Canary-1B" and name == "canary_engine":
                        continue
                    if active == "Whisper" and name == "whisper_engine":
                        continue
                    if active == "Moonshine v2" and name == "moonshine_engine":
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
        if self.config.get("engine") == "Whisper":
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
        if self.config.get("engine") == "Whisper":
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
        print(f"[MoonshineSTT] {msg}")
        if self.gui and color:
            pass
    def _capture_target_window(self):
        try:
            hwnd = get_foreground_window()
            target_root = user32.GetAncestor(hwnd, GA_ROOT) if hwnd else 0
            if hwnd and target_root != self._our_root_hwnd:
                self._target_top_hwnd = hwnd
                child = get_focused_child(hwnd)
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
                    self._gui_queue.put(("status", (f"Target: {short}", FG_SECONDARY)))
            else:
                self._target_top_hwnd = None
                self._target_child_hwnd = None
        except Exception as e:
            print(f"capture target failed: {e}")
            self._target_top_hwnd = None
            self._target_child_hwnd = None
    def _model_ready(self, success: bool, error: str = None):
        if self.gui:
            try:
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
        now = time.monotonic()
        if now - getattr(self, "_last_level_t", 0.0) < 0.065:
            return
        self._last_level_t = now
        try:
            if self._recording and now - getattr(self, "_rec_start_t", now) > 300:
                self._gui_queue.put(("status", ("Auto-stopped (5 min max)", WARNING)))
                threading.Thread(target=self._stop_recording, daemon=True).start()
                return
        except Exception:
            pass
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
        while True:
            try:
                msg_type, payload = self._gui_queue.get_nowait()
            except queue.Empty:
                break
            except Exception:
                continue
            try:
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
                    try:
                        self._on_job_finished(payload[0], payload[1])
                    except Exception:
                        pass
                elif msg_type == "srt_file_status":
                    try:
                        self.gui.set_srt_file_status(payload[0], payload[1])
                    except Exception:
                        pass
            except Exception as e:
                try:
                    print(f"[MoonshineSTT] gui queue item failed: {e}")
                except Exception:
                    pass
                continue
        try:
            self.gui.after(100, self._poll_queue)
        except Exception:
            pass
    def _keyboard_listener(self):
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
                if not getattr(self.engine, "is_ready", False):
                    self._gui_queue.put(("status", ("Model not ready yet", WARNING)))
                    return
            except Exception:
                pass
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
            try:
                self._rec_start_t = time.monotonic()
            except Exception:
                pass
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
        try:
            self.audio_queue.put_nowait((audio, settings_snapshot))
        except queue.Full:
            self._gui_queue.put(("status", (
                "Queue full - wait for processing", WARNING)))
            self._update_indicator()
            return
        self._update_indicator()
    def _processing_worker(self):
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
        q_size = self.audio_queue.qsize()
        if self.gui:
            if q_size > 0:
                self._gui_queue.put(("status", (f"Transcribing... ({q_size} queued)", WARNING)))
            else:
                self._gui_queue.put(("status", ("Transcribing...", WARNING)))
        engine = settings.get("engine", self.engine)
        if not getattr(engine, "is_ready", False):
            self._gui_queue.put(("status", ("Engine not ready - skipped clip", WARNING)))
            return
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
                self._gui_queue.put(("status", ("Pasted ✓  (clipboard also)", SUCCESS)))
            else:
                self._gui_queue.put(("status", ("Copied to clipboard — press Ctrl+V", WARNING)))
        else:
            display = clean if clean else "(no speech detected)"
            self._gui_queue.put(("transcription", display))
            self._gui_queue.put(("status", ("Ready", FG_SECONDARY)))
    def _update_indicator(self):
        if not self.gui:
            return
        try:
            q_size = self.audio_queue.qsize()
            is_proc = self.currently_processing
            is_rec = self._recording
            if is_rec:
                if is_proc or q_size > 0:
                    self._gui_queue.put(("status", (f"Recording... ({q_size} queued)", DANGER)))
                return
            if is_proc or q_size > 0:
                if q_size > 0:
                    self._gui_queue.put(("status", (f"Processing ({q_size} queued)...", WARNING)))
                else:
                    self._gui_queue.put(("status", ("Processing...", WARNING)))
            else:
                pass
        except Exception:
            pass
    def _get_srt_moonshine_transcriber(self):
        from moonshine_voice import get_model_for_language, Transcriber
        from moonshine_voice.moonshine_api import ModelArch
        from engine import PORTABLE_CACHE_ROOT, MODEL_ARCH_NAMES
        # SRT tab's own arch (not Live's): per-tab selections.
        try:
            arch = int(self.tab_selection("srt")["arch"])
        except Exception:
            try:
                arch = int(self.config.get("model_arch", 5))
            except Exception:
                arch = 5
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
    def _srt_start(self, input_paths, order=None, out_dir: str = "",
                   cpu_workers: int = 1,
                   srt_in: str = "", srt_out: str = "", srt_task: str = "",
                   normalize_audio: bool = False, burn_after: bool = False,
                   burn_font_size: int = 0, burn_speed: str = "match",
                   vbr_auto: bool = True, vbr_kbps: int = 2000,
                   burn_codec: str = "h264"):
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
                raise ValueError("queue is empty - add video/audio files first")
            try:
                order = [int(i) for i in (order or [])]
            except Exception:
                order = []
            if len(order) != len(paths):
                order = list(range(len(paths)))
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
                _eng = self.config.get("engine", "Moonshine v2")
                if srt_task and _eng == "Whisper":
                    self.config["whisper_task"] = srt_task
                elif srt_task and _eng == "Canary-1B":
                    self.config["canary_task"] = srt_task
                self.config["srt_out_dir"] = out_dir or ""
                try:
                    self.config["srt_cpu"] = int(cpu_workers)
                except Exception:
                    self.config["srt_cpu"] = 0
                self.config["srt_norm"] = bool(normalize_audio)
                self.config["burn_after"] = bool(burn_after)
                try:
                    _bfs = max(10, min(40, int(burn_font_size or 18)))
                except Exception:
                    _bfs = 18
                self.config["burn_font_size"] = _bfs
                _bspd = str(burn_speed or "match").strip().lower()
                if _bspd not in ("match", "fast", "fastest",
                                 "nvenc_draft", "nvenc_turbo",
                                 "nvenc_balanced"):
                    _bspd = "match"
                self.config["burn_speed"] = _bspd
                try:
                    _bcode = str(burn_codec or "h264").strip().lower()
                except Exception:
                    _bcode = "h264"
                if _bcode not in ("h264", "hevc"):
                    _bcode = "h264"
                self.config["burn_codec"] = _bcode
                try:
                    _bauto = bool(vbr_auto) if vbr_auto is not None else True
                    _bkbps = max(300, min(10000, int(vbr_kbps or 2000)))
                except Exception:
                    _bauto, _bkbps = True, 2000
                self.config["burn_vbr_auto"] = _bauto
                self.config["burn_vbr_kbps"] = _bkbps
                save_local_config(self.config)
            try:
                _st = self.tab_selection("srt")
            except Exception:
                _st = {"kind": "Moonshine v2", "arch": 5, "wmodel": "large-v3"}
            job = {
                "engine_kind": _st["kind"],
                "moonshine_arch": int(_st["arch"]),
                "canary_task": self.config.get("canary_task", "transcribe"),
                "canary_src": self.config.get("canary_src_lang", "auto"),
                "whisper_task": self.config.get("whisper_task", "translate"),
                "whisper_src": self.config.get("whisper_src_lang", "ja"),
                "srt_input_lang": self.config.get("srt_input_lang", "auto"),
                "srt_output_lang": self.config.get("srt_output_lang", "en"),
                "cpu_workers": int(cpu_workers),
                "src_paths": paths,
                "order": order,
                "out_dir": out_dir,
                "normalize_audio": bool(normalize_audio),
                "burn_after": bool(burn_after),
                "burn_font_size": _bfs,
                "burn_speed": _bspd,
                "burn_vbr_auto": _bauto,
                "burn_vbr_kbps": _bkbps,
                "burn_codec": _bcode,
            }
            if job["engine_kind"] not in ("Moonshine v2", "Canary-1B", "Whisper"):
                job["engine_kind"] = "Moonshine v2"
            self._srt_cancel.clear()
            if self._abort_shutdown():
                self._gui_queue.put(
                    ("srt_log", "Aborted a pending auto-shutdown (new job started)"))
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
                from collections import deque as _deque
                try:
                    _reserved = srtmod.reserve_batch_names(
                        [(p, job["out_dir"]) for p in job["src_paths"]])
                    _by_path = {}
                    for _idx in sorted(_reserved):
                        _pp = job["src_paths"][_idx]
                        _by_path.setdefault(_pp, _deque()).append(_reserved[_idx])
                except Exception:
                    _by_path = {}
                def _run_one(path, progress_cb, log_cb):
                    _q = _by_path.get(path)
                    _out = _q.popleft() if _q else None
                    return srtmod.run_srt_job(
                        src_path=path,
                        out_dir=job["out_dir"],
                        engine_kind=job["engine_kind"],
                        moonshine_arch=job["moonshine_arch"],
                        canary_task=job["canary_task"],
                        canary_src=job["canary_src"],
                        cpu_workers=job["cpu_workers"],
                        get_moonshine_transcriber=self._get_srt_moonshine_transcriber,
                        get_canary_engine=lambda: self._tab_canary("srt"),
                        progress_cb=progress_cb,
                        log_cb=log_cb,
                        cancel_event=self._srt_cancel,
                        whisper_task=job["whisper_task"],
                        whisper_src=job["whisper_src"],
                        get_whisper_engine=lambda: self._tab_whisper("srt"),
                        srt_input_lang=job["srt_input_lang"],
                        srt_output_lang=job["srt_output_lang"],
                        out_path=str(_out) if _out is not None else None,
                        normalize_audio=job.get("normalize_audio", False),
                    )
                def _file_cb(kind, path, info):
                    try:
                        idx = int((info or {}).get("index", -1))
                        _order = job.get("order") or []
                        if 0 <= idx < len(_order):
                            idx = int(_order[idx])
                    except Exception:
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
                burn_msg = ""
                burn_cancelled = False
                if (not cancelled) and job.get("burn_after"):
                    _bmsg, burn_cancelled = self._auto_burn_after_srt(job)
                    if _bmsg:
                        burn_msg = " | " + str(_bmsg)
                if cancelled or burn_cancelled:
                    self._gui_queue.put(
                        ("srt_done", (False, f"Cancelled after {len(ok_paths)}/{total} files{burn_msg}")))
                elif not bad:
                    self._gui_queue.put(
                        ("srt_done", (True, f"Batch done: all {total} saved{burn_msg}")))
                elif ok_paths:
                    try:
                        first_bad = _os.path.basename(bad[0][0])
                    except Exception:
                        first_bad = "a file"
                    self._gui_queue.put(
                        ("srt_done", (True, f"Batch done: {len(ok_paths)}/{total} saved "
                                      f"({len(bad)} failed, e.g. {first_bad}){burn_msg}")))
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
    @staticmethod
    def _shutdown_pc():
        try:
            import subprocess as _sp
            _sp.Popen(["shutdown", "/s", "/f", "/t", "60"],
                      stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        except Exception:
            pass
    @staticmethod
    def _abort_shutdown() -> bool:
        try:
            import subprocess as _sp
            _r = _sp.run(["shutdown", "/a"],
                         stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                         timeout=10)
            return bool(getattr(_r, "returncode", 1) == 0)
        except Exception:
            return False
    def _on_job_finished(self, ok, msg):
        try:
            text = str(msg or "")
        except Exception:
            text = ""
        if not ok:
            return
        full = "all " in text
        try:
            want_sd = bool(self.config.get("auto_shutdown", False))
        except Exception:
            want_sd = False
        if full and want_sd:
            self._gui_queue.put(
                ("srt_log", "Auto-shutdown ON: PC powers off in 60s "
                 "(run `shutdown /a` in cmd to abort)"))
            self._shutdown_pc()
            return
        try:
            want_alert = bool(self.config.get("completion_alert", True))
        except Exception:
            want_alert = True
        if want_alert and self.gui:
            try:
                self.gui.notify_completion("Processing complete", text)
            except Exception:
                pass
    def _cancel_srt_job(self):
        try:
            self._srt_cancel.set()
        except Exception:
            pass
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
    def _execute_burn(self, bjob):
        import os as _os
        import srt as srtmod
        from collections import deque as _deque2
        ffmpeg = srtmod.get_ffmpeg_exe()
        if not ffmpeg:
            raise RuntimeError("ffmpeg not found - run setup.bat once.")
        if bjob.get("speed", "match") in ("nvenc_draft", "nvenc_turbo",
                                          "nvenc_balanced"):
            try:
                import gpu as _gpumod
                _bcode = str(bjob.get("burn_codec", "h264")).strip().lower()
                _enc = "hevc_nvenc" if _bcode == "hevc" else "h264_nvenc"
                _nv_ok = bool(_gpumod.nvenc_available(ffmpeg, _enc))
            except Exception:
                _nv_ok = False
                _enc = "h264_nvenc"
            if not _nv_ok:
                raise RuntimeError(
                    f"NVENC burn needs an NVIDIA GPU + {_enc} encoder - "
                    "none detected. Pick a CPU burn speed instead.")
        try:
            _bres = srtmod.reserve_burn_names(
                [(p, bjob["out_dir"]) for p in bjob["paths"]])
            _bby_path = {}
            for _idx in sorted(_bres):
                _pp = bjob["paths"][_idx]
                _bby_path.setdefault(_pp, _deque2()).append(_bres[_idx])
        except Exception:
            _bby_path = {}
        def _run_one(path, progress_cb, log_cb):
            from pathlib import Path as _P
            src = _P(path)
            srt_path = srtmod.default_out_path(src, bjob["out_dir"])
            if not srt_path.exists():
                raise FileNotFoundError(
                    f"no SRT for {src.name} - Generate SRT first")
            info = srtmod.probe_media(src, ffmpeg)
            _vbps_auto, abps, acopy = srtmod.plan_burn_bitrates(info)
            try:
                in_gb = self._fmt_gb(info["size"])
            except Exception:
                in_gb = "?"
            if bjob.get("vbr_auto", True):
                vbps = _vbps_auto
                budget_note = f"(in {in_gb}, target match)"
            else:
                vbps = max(100000, int(bjob.get("vbr_kbps", 2000)) * 1000)
                try:
                    _est = (vbps + (abps if info.get("has_audio") else 0)) \
                        * max(1.0, float(info.get("duration") or 0)) / 8
                    est_note = f"est. ~{self._fmt_gb(_est)}, size-match OFF"
                except Exception:
                    est_note = "size-match OFF"
                budget_note = f"(in {in_gb}, manual {vbps // 1000} kbps, {est_note})"
            log_cb(f"budget: video {vbps // 1000} kbps, "
                   f"audio {'copy' if acopy else f'AAC {abps // 1000} kbps'} "
                   f"{budget_note}")
            _bq = _bby_path.get(path)
            if _bq:
                out_path = _bq.popleft()
            else:
                out_path = srtmod.default_burn_path(src, bjob["out_dir"])
            return srtmod.burn_subtitles(
                str(src), str(srt_path), str(out_path), ffmpeg,
                vbps // 1000, (abps // 1000) if not acopy else 128,
                acopy, bjob["cpu_workers"], bjob["font_size"],
                bjob["speed"],
                progress_cb=progress_cb, log_cb=log_cb,
                cancel_event=self._srt_cancel,
                codec=bjob.get("burn_codec", "h264"))
        def _file_cb(kind, path, info):
            try:
                idx = int((info or {}).get("index", -1))
                _order = bjob.get("order") or []
                if 0 <= idx < len(_order):
                    idx = int(_order[idx])
            except Exception:
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
            bjob["paths"], _run_one,
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
            return results, True, False, f"Burn cancelled after {len(ok)}/{total} files"
        elif not bad:
            return results, False, True, f"Burned all {total}{sizes}"
        elif ok:
            try:
                first_bad = _os.path.basename(bad[0][0])
            except Exception:
                first_bad = "a file"
            return (results, False, True,
                    f"Burned {len(ok)}/{total}{sizes} "
                    f"({len(bad)} failed, e.g. {first_bad})")
        else:
            return results, False, False, "Burn failed: 0/{total}".format(total=total)
    def _auto_burn_after_srt(self, job):
        try:
            import srt as srtmod
            from pathlib import Path as _P
            order = list(job.get("order") or [])
            srcs = list(job.get("src_paths") or [])
            out_dir = job.get("out_dir") or ""
            keep_pos = []
            kept_existing = 0
            for pos, p in enumerate(srcs):
                try:
                    row = order[pos] if pos < len(order) else pos
                    if srtmod.is_audio_only_path(p):
                        self._gui_queue.put(
                            ("srt_file_status", (row, "– audio-only")))
                        continue
                    if not srtmod.default_out_path(_P(p), out_dir).exists():
                        self._gui_queue.put(
                            ("srt_file_status", (row, "– no SRT")))
                        continue
                    if srtmod.default_burn_path(_P(p), out_dir).exists():
                        self._gui_queue.put(
                            ("srt_file_status", (row, "– kept existing")))
                        kept_existing += 1
                        continue
                except Exception:
                    continue
                keep_pos.append(pos)
            if kept_existing:
                self._gui_queue.put(
                    ("srt_log",
                     f"Auto-burn: {kept_existing} existing MP4(s) kept "
                     f"(Convert to MP4 to overwrite)"))
            if not keep_pos:
                return "", False
            self._gui_queue.put(
                ("srt_log",
                 f"Auto-burn: {len(keep_pos)} file(s) with SRT -> MP4..."))
            self._gui_queue.put(
                ("srt_progress", (0.0, "Starting auto-burn...")))
            bjob = {"paths": [srcs[i] for i in keep_pos],
                    "order": [order[i] if i < len(order) else i
                              for i in keep_pos],
                    "out_dir": out_dir,
                    "cpu_workers": int(job.get("cpu_workers") or 0),
                    "font_size": int(job.get("burn_font_size") or 18),
                    "speed": job.get("burn_speed") or "match",
                    "vbr_auto": job.get("burn_vbr_auto", True),
                    "vbr_kbps": job.get("burn_vbr_kbps", 2000),
                    "burn_codec": job.get("burn_codec", "h264")}
            _results, _cancelled, _ok, _msg = self._execute_burn(bjob)
            return _msg, bool(_cancelled)
        except Exception as e:
            import traceback
            traceback.print_exc()
            return f"auto-burn skipped: {e}", False
    def _burn_start(self, input_paths, order=None, out_dir: str = "",
                    cpu_workers: int = 1, font_size: int = 0,
                    speed: str = "match", vbr_auto: bool = True,
                    vbr_kbps: int = 2000, burn_codec: str = "h264"):
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
            try:
                order = [int(i) for i in (order or [])]
            except Exception:
                order = []
            if len(order) != len(paths):
                order = list(range(len(paths)))
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
                _spd = str(speed or "").strip().lower()
                if _spd not in ("match", "fast", "fastest",
                                "nvenc_draft", "nvenc_turbo",
                                "nvenc_balanced"):
                    _spd = "match"
                self.config["burn_speed"] = _spd
                try:
                    _bcode = str(burn_codec or "h264").strip().lower()
                except Exception:
                    _bcode = "h264"
                if _bcode not in ("h264", "hevc"):
                    _bcode = "h264"
                self.config["burn_codec"] = _bcode
                try:
                    _vauto = bool(vbr_auto) if vbr_auto is not None else True
                    _vkbps = max(300, min(10000, int(vbr_kbps or 2000)))
                except Exception:
                    _vauto, _vkbps = True, 2000
                self.config["burn_vbr_auto"] = _vauto
                self.config["burn_vbr_kbps"] = _vkbps
                save_local_config(self.config)
            job = {"paths": paths, "out_dir": out_dir,
                   "cpu_workers": int(cpu_workers), "font_size": _fs,
                   "speed": _spd, "order": order,
                   "vbr_auto": _vauto, "vbr_kbps": _vkbps,
                   "burn_codec": _bcode}
            self._srt_cancel.clear()
            if self._abort_shutdown():
                self._gui_queue.put(
                    ("srt_log", "Aborted a pending auto-shutdown (new job started)"))
            if self.gui:
                self.gui.after(0, lambda: self.gui.set_srt_running(True))
                _n = len(job["paths"])
                _cpu = job["cpu_workers"]
                self.gui.after(0, lambda _n=_n: self.gui.set_srt_progress(
                    0, f"Starting burn ({_n} file(s))..."))
                self._gui_queue.put(
                    ("srt_log", f"Burn: {_n} file(s), {_cpu} threads (x264 multi-core), "
                     f"{_speed_label(job.get('speed', 'match'))}"))
            try:
                _spd = job.get("speed", "match")
                _cm = self.config.get("compute", "auto")
                _cpu_burn = not str(_spd).startswith("nvenc_")
                _gpu_wanted = (_cm == "gpu")
                if not _gpu_wanted and _cm == "auto":
                    try:
                        import gpu as _gpumod2
                        _gpu_wanted = _gpumod2.best_gpu() is not None
                    except Exception:
                        _gpu_wanted = False
                if _cpu_burn and _gpu_wanted:
                    self._gui_queue.put(
                        ("srt_log", "note: Compute=GPU covers inference; this burn "
                         "encodes on CPU (x264) - pick an NVENC burn "
                         "speed for GPU encoding"))
            except Exception:
                pass
        except Exception as e:
            with self._srt_lock:
                self._srt_busy = False
            self._gui_queue.put(("srt_done", (False, f"Could not start burn: {e}")))
            return
        def _job():
            try:
                _results, _cancelled, _ok, _msg = self._execute_burn(job)
                self._gui_queue.put(("srt_done", (_ok, _msg)))
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
    def _burn_preview(self, input_paths, out_dir: str, font_size: int,
                      sample_start: str = "", sample_len: int = 15):
        def _done():
            if self.gui:
                try:
                    self.gui.after(0, lambda: self.gui.set_srt_preview_done())
                except Exception:
                    pass
        def _work():
            import subprocess as _sp
            import tempfile as _tf
            from pathlib import Path as _P
            import srt as srtmod
            tmpd = _P(_tf.mkdtemp(prefix="burnprev_"))
            try:
                if isinstance(input_paths, str):
                    paths = [input_paths]
                else:
                    try:
                        paths = list(input_paths or [])
                    except Exception:
                        paths = []
                paths = [str(p).strip().strip('"') for p in paths
                         if str(p or "").strip()]
                if not paths:
                    self._gui_queue.put(("srt_progress", (0.0, "Add video files first")))
                    return
                try:
                    size = max(10, min(40, int(font_size or 0)))
                except Exception:
                    size = 18
                if not size:
                    size = 18
                try:
                    s_len = int(sample_len or 15)
                except Exception:
                    s_len = 15
                s_len = max(5, min(120, s_len))
                with _CONFIG_LOCK:
                    self.config["burn_font_size"] = size
                    self.config["burn_sample_start"] = str(sample_start or "")
                    self.config["burn_sample_len"] = s_len
                    save_local_config(self.config)
                snap = {
                    "engine_kind": self.config.get("engine", "Moonshine v2"),
                    "moonshine_arch": int(self.config.get("model_arch", 5)),
                    "canary_task": self.config.get("canary_task", "transcribe"),
                    "canary_src": self.config.get("canary_src_lang", "auto"),
                    "whisper_task": self.config.get("whisper_task", "translate"),
                    "whisper_src": self.config.get("whisper_src_lang", "ja"),
                    "srt_input_lang": self.config.get("srt_input_lang", "auto"),
                    "srt_output_lang": self.config.get("srt_output_lang", "en"),
                    "cpu_workers": int(self.config.get("srt_cpu", 0) or 0),
                    "normalize_audio": bool(self.config.get("srt_norm", False)),
                }
                if snap["engine_kind"] not in ("Moonshine v2", "Canary-1B", "Whisper"):
                    snap["engine_kind"] = "Moonshine v2"
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
                        target = (_P(p), cand, False)
                        break
                sample_origin = 0.0
                if target is None:
                    src0 = _P(paths[0])
                    try:
                        info0 = srtmod.probe_media(src0, ffmpeg)
                        dur0 = max(1.0, float(info0.get("duration") or 0))
                    except Exception:
                        dur0 = 0.0
                    start = srtmod.parse_time_to_seconds(sample_start, dur0)
                    if dur0 > 0:
                        s_len = max(5, min(s_len, int(max(5.0, dur0 - start))))
                    sample_origin = start
                    self._gui_queue.put(("srt_log",
                                         f"Preview: no SRT yet - transcribing {s_len}s sample "
                                         f"@{start:.0f}s with {snap['engine_kind']}..."))
                    self._gui_queue.put(("srt_progress",
                                         (0.0, f"Sample @{start:.0f}s ({s_len}s) transcribing...")))
                    clip_wav = tmpd / "sample16k.wav"
                    srtmod.extract_clip(str(src0), str(clip_wav), ffmpeg,
                                        start, s_len)
                    sample_srt = srtmod.run_srt_job(
                        src_path=str(clip_wav), out_dir=str(tmpd),
                        engine_kind=snap["engine_kind"],
                        moonshine_arch=snap["moonshine_arch"],
                        canary_task=snap["canary_task"],
                        canary_src=snap["canary_src"],
                        cpu_workers=snap["cpu_workers"] or 4,
                        get_moonshine_transcriber=self._get_srt_moonshine_transcriber,
                        get_canary_engine=lambda: self._get_canary_engine(False),
                        progress_cb=lambda f, m: self._gui_queue.put(
                            ("srt_progress", (f, f"[sample] {m}"))),
                        log_cb=lambda m: self._gui_queue.put(("srt_log", f"[sample] {m}")),
                        cancel_event=None,
                        whisper_task=snap["whisper_task"],
                        whisper_src=snap["whisper_src"],
                        get_whisper_engine=lambda: self._get_whisper_engine(False),
                        srt_input_lang=snap["srt_input_lang"],
                        srt_output_lang=snap["srt_output_lang"],
                        normalize_audio=snap["normalize_audio"],
                    )
                    shifted = tmpd / "sample_shifted.srt"
                    srtmod.offset_srt_file(sample_srt, sample_origin, shifted)
                    target = (src0, shifted, True)
                src, srt_path, _is_sample = target
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
                vf = srtmod.stage_subtitles_filter(srt_path, size, tmpd)
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
            except Exception as e:
                import traceback
                traceback.print_exc()
                self._gui_queue.put(("srt_log", f"Preview failed: {e}"))
                self._gui_queue.put(("srt_progress", (0.0, f"Preview failed: {e}")))
            finally:
                try:
                    import shutil as _sh
                    _sh.rmtree(str(tmpd), ignore_errors=True)
                except Exception:
                    pass
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
    def _note_transcribe(self, audio, sample_rate=16000):
        """Transcribe audio for Note mode using the NOTE tab's snapshotted
        engine (stable for the session even if Live switches mid-note).
        Self-heals: a missing snapshot re-resolves (loads are already
        kicked by the pre-flight, so this is just a lookup)."""
        try:
            engine = self._note_engine_obj
            if engine is None:
                try:
                    v = self.note_record_request() or {}
                    if isinstance(v, dict) and v.get("go"):
                        engine = self._note_engine_obj
                except Exception:
                    pass
            if engine is None:
                return "[Error: note engine not ready]"
            return engine.transcribe(audio, sample_rate)
        except Exception as e:
            return f"[Error: {e}]"

    def _on_close(self):
        try:
            if self.gui and not self.gui.confirm_note_close():
                return
        except Exception:
            pass
        try:
            # Drop this run's unsaved Note session audio (kept iff the user
            # explicitly saved it via Save Audio).
            if self.gui:
                self.gui.cleanup_note_audio()
        except Exception:
            pass
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