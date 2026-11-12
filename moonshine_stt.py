import os
import sys
import time
import threading
import queue
import json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
if sys.stdout is None or sys.stderr is None:


    try:
        _logdir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "logs")
        os.makedirs(_logdir, exist_ok=True)
        _winlog = open(os.path.join(_logdir, "app.log"), "a",
                       encoding="utf-8", errors="replace", buffering=1)
        if sys.stdout is None:
            sys.stdout = _winlog
        if sys.stderr is None:
            sys.stderr = _winlog
    except Exception:
        pass
from pynput import keyboard
from engine import TranscriptionEngine
from recorder import AudioRecorder
from input_sim import (
    insert_text,
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

_SINGLE_INSTANCE_MUTEX = None


try:
    from core.config import (
        APP_VERSION,
        CONFIG_PATH,
        NOTES_AUDIO_DIR,
        _CONFIG_LOCK,
        DEFAULT_CONFIG,
        load_local_config,
        save_local_config,
        apply_suffix,
    )
except Exception:  # pragma: no cover - fallback if core/ is missing
    import json as _json
    APP_VERSION = "1.4.18"
    CONFIG_PATH = os.path.join(os.path.dirname(__file__), "moonshine_config.json")
    NOTES_AUDIO_DIR = os.path.join(os.path.dirname(__file__), "notes_audio")
    _CONFIG_LOCK = threading.RLock()
    DEFAULT_CONFIG = {
        "typing_method": "clipboard",
        "suffix": "none",
        "typing_delay_ms": 0,
        "model_arch": 5,
        "engine": "Moonshine v2",
        "canary_task": "transcribe",
        "canary_src_lang": "auto",
        "whisper_task": "transcribe",
        "whisper_src_lang": "auto",
        "qwen_model": "qwen3-asr-1.7b-q8_0",
        "qwen_src_lang": "auto",
        "dict_terms": [],
        "dict_context": "",
        "dict_engines": {"Whisper": False, "Qwen3-ASR": False},
        "note_whisper_lang": "auto",
        "note_mic_id": "",
        "note_desktop_audio": False,
        "whisper_model": "large-v3-turbo",
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
        "windowless": False,
        "input_device": -1,
        "input_device_name": "",
        "auto_input": True,
    }

    def load_local_config():
        try:
            if os.path.exists(CONFIG_PATH):
                with open(CONFIG_PATH, "r", encoding="utf-8") as f:
                    cfg = _json.load(f)
                out = dict(DEFAULT_CONFIG)
                out.update(cfg)
                if any(k not in cfg for k in DEFAULT_CONFIG):
                    try:
                        with open(CONFIG_PATH, "w", encoding="utf-8") as wf:
                            _json.dump(out, wf, indent=2)
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
                    _json.dump(cfg, f, indent=2)
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
        if self.config.get("engine") not in ("Moonshine v2", "Canary-1B", "Whisper", "Qwen3-ASR"):
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
            _wmodels = ("tiny", "base", "small", "medium", "large-v3",
                        "large-v3-turbo")
        if self.config.get("whisper_src_lang") not in _wlangs:
            self.config["whisper_src_lang"] = "auto"
            needs_save = True
        if self.config.get("note_whisper_lang") not in _wlangs:
            self.config["note_whisper_lang"] = "auto"
            needs_save = True
        if self.config.get("whisper_model") not in _wmodels:
            self.config["whisper_model"] = "large-v3"
            needs_save = True




        if (self.config.get("whisper_model") == "large-v3-turbo"
                and self.config.get("whisper_task") == "translate"):
            self.config["whisper_model"] = "large-v3"
            needs_save = True
        if self.config.get("whisper_device") not in ("auto", "cpu", "cuda"):
            self.config["whisper_device"] = "auto"
            needs_save = True
        try:
            from qwen_engine import QWEN_LANGUAGES, QWEN_MODEL_FILES
            _qlangs = tuple(QWEN_LANGUAGES)
            _qmodels = tuple(QWEN_MODEL_FILES)
        except Exception:
            _qlangs = ("auto", "en", "zh", "ja", "ko", "de", "fr", "es")
            _qmodels = ("qwen3-asr-1.7b-q8_0",)
        if self.config.get("qwen_src_lang") not in _qlangs:
            self.config["qwen_src_lang"] = "auto"
            needs_save = True
        if self.config.get("qwen_model") not in _qmodels:
            self.config["qwen_model"] = "qwen3-asr-1.7b-q8_0"
            needs_save = True
        try:
            import dictionary as _dictmod
            _clean_terms = _dictmod.norm_terms(self.config.get("dict_terms"))
            _clean_ctx = _dictmod.norm_context(self.config.get("dict_context"))
            _clean_eng = _dictmod.enabled_map(self.config.get("dict_engines"))
        except Exception:
            _clean_terms = list(self.config.get("dict_terms") or [])
            _clean_ctx = str(self.config.get("dict_context") or "")
            _clean_eng = {"Whisper": False, "Qwen3-ASR": False}
        if self.config.get("dict_terms") != _clean_terms:
            self.config["dict_terms"] = _clean_terms
            needs_save = True
        if self.config.get("dict_context") != _clean_ctx:
            self.config["dict_context"] = _clean_ctx
            needs_save = True
        if self.config.get("dict_engines") != _clean_eng:
            self.config["dict_engines"] = _clean_eng
            needs_save = True
        if self.config.get("compute") not in ("auto", "cpu", "gpu"):
            self.config["compute"] = "auto"
            needs_save = True
        if not isinstance(self.config.get("windowless"), bool):
            self.config["windowless"] = False
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
                           ("completion_alert", True),
                           ("itn_live", True), ("itn_hint", True)):
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
        if self.config.get("srt_tab") not in ("Live", "SRT File", "Note", "Import"):
            self.config["srt_tab"] = "Live"
            needs_save = True
        if needs_save:
            save_local_config(self.config)
        self._eng_lock = threading.Lock()
        self.moonshine_engine = TranscriptionEngine(language="en", model_arch=arch, on_ready=self._model_ready)
        self.canary_engine = None
        self.whisper_engine = None
        self.qwen_engine = None
        self._tab_sel = {}
        self._shared_sel = {"kind": "Moonshine v2", "arch": 5, "wmodel": "large-v3"}
        self._tab_cache = {
            "srt": {"canary": None, "whisper": None, "qwen": None, "wmodel": None},
            "note": {"canary": None, "whisper": None, "qwen": None, "wmodel": None},
            "import": {"canary": None, "whisper": None, "qwen": None, "wmodel": None},
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
        elif self.config.get("engine") == "Qwen3-ASR":
            self.engine = self._get_qwen_engine()
        else:
            self.engine = self.moonshine_engine
        self.recorder = AudioRecorder(sample_rate=SAMPLE_RATE)
        try:
            self.recorder.set_device(self.config.get("input_device", -1),
                                     self.config.get("input_device_name", ""))
        except Exception:
            pass
        self._mic_watch_stop = threading.Event()
        self._recording = False
        self._rec_session = 0
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
        self._note_file_busy = False
        self._note_file_lock = threading.Lock()
        self._note_file_thread = None
        self._note_cancel_event = threading.Event()

        self._import_file_busy = False
        self._import_file_lock = threading.Lock()
        self._import_file_thread = None
        self._import_cancel_event = threading.Event()
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
                    self.config.get("itn_live", True),
                    self.config.get("itn_hint", True),
                    self._on_itn_live_changed,
                    self._on_itn_hint_changed,
                )
                try:
                    self._refresh_model_row()
                except Exception:
                    pass
                try:
                    _eng = self.config.get("engine", "Moonshine v2")
                    if _eng == "Whisper":
                        _t, _s = (self.config.get("whisper_task", "translate"),
                                  self.config.get("whisper_src_lang", "auto"))
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
                        self.gui.set_dictionary_callback(self._open_dictionary)
                        import dictionary as _dictmod
                        self.gui.set_dictionary_status(_dictmod.status_text(
                            self.config.get("dict_terms") or [],
                            self.config.get("dict_context") or "",
                            _dictmod.enabled_map(
                                self.config.get("dict_engines"))))
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
                    self.gui.set_windowless(self.config.get("windowless", False),
                                            self._on_windowless_changed)
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
                    self.gui.set_note_unload_callback(
                        lambda: self.unload_other_tab_engines("note"))
                except Exception:
                    pass
                try:
                    self.gui.set_note_self_unload_callback(
                        lambda: self.unload_tab_engines("note"))
                except Exception:
                    pass
                try:
                    self.gui.set_note_lang_callback(
                        self._on_note_lang_changed)
                    self.gui.set_note_language(
                        self.config.get("note_whisper_lang", "auto"))
                except Exception:
                    pass
                try:
                    self.gui.set_note_source_callback(
                        self._on_note_source_changed)
                    self.gui.set_note_source(
                        self.config.get("note_mic_id", ""),
                        bool(self.config.get("note_desktop_audio", False)))
                except Exception:
                    pass
                try:
                    self.gui.set_unload_callback(
                        lambda: self.unload_tab_engines("live"))
                except Exception:
                    pass
                try:
                    self.gui.set_input_device_callback(
                        self._on_input_device_changed)
                    self._refresh_input_devices(refresh=True)
                except Exception:
                    pass
                try:
                    threading.Thread(target=self._watch_input_devices,
                                     daemon=True).start()
                except Exception:
                    pass
                try:
                    self.gui.set_note_file_callbacks(self._note_file_start,
                                                     self._note_file_cancel,
                                                     self.note_file_request)
                except Exception:
                    pass
                try:
                    self.gui.set_import_file_callbacks(self._import_file_start,
                                                       self._import_file_cancel,
                                                       self.import_file_request)
                except Exception:
                    pass
                try:
                    self.gui.set_tab_engine_callbacks(
                        self.plan_tab_change, self.apply_tab_change,
                        lambda t: self.tab_selection(t))
                    for _t, _setter in (("srt", self.gui.set_srt_engine_state),
                                        ("note", self.gui.set_note_engine_state),
                                        ("import", self.gui.set_import_engine_state)):
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



            from note_engine import recover_stale_session_wavs as _recover
            _n = _recover(NOTES_AUDIO_DIR)
            if _n:
                self._log(f"Recovered {_n} unfinished note recording(s) "
                          f"to notes_audio/recovered")
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

    def _tab_qwen(self, tab):
        """Dedicated tab instance for jobs (see _tab_canary)."""
        try:
            if tab == "live":
                return self._get_qwen_engine(False)
            sel = self.tab_selection(tab)
            if sel["kind"] != "Qwen3-ASR":
                return None
            return self._tab_heavy(tab, "Qwen3-ASR", sel["wmodel"])
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


                        try:
                            self._gui_queue.put(
                                ("status", ("Canary unavailable - using Moonshine",
                                            WARNING)))
                        except Exception:
                            pass
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
                            source_lang=self.config.get("whisper_src_lang", "auto"),
                            target_lang="en",
                            model_id=self.config.get("whisper_model", "large-v3"),
                            device=self.config.get("compute", "auto"),
                            on_ready=self._model_ready,
                        )
                    except Exception as e:
                        print(f"Whisper engine failed to create: {e}")
                        import traceback; traceback.print_exc()

                        try:
                            self._gui_queue.put(
                                ("status", ("Whisper unavailable - using Moonshine",
                                            WARNING)))
                        except Exception:
                            pass
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
        self._apply_dictionary(self.whisper_engine)
        return self.whisper_engine
    def _get_qwen_engine(self, apply_live_options: bool = True):
        if self.qwen_engine is None:
            with self._eng_lock:
                if self.qwen_engine is None:
                    try:
                        from qwen_engine import QwenEngine
                        self.qwen_engine = QwenEngine(
                            task="transcribe",
                            source_lang=self.config.get("qwen_src_lang", "auto"),
                            target_lang="en",
                            model_id=self.config.get("qwen_model",
                                                     "qwen3-asr-1.7b-q8_0"),
                            device=self.config.get("compute", "auto"),
                            on_ready=self._model_ready,
                        )
                    except Exception as e:
                        print(f"Qwen3-ASR engine failed to create: {e}")
                        import traceback; traceback.print_exc()

                        try:
                            self._gui_queue.put(
                                ("status", ("Qwen3-ASR unavailable - using Moonshine",
                                            WARNING)))
                        except Exception:
                            pass
                        return self.moonshine_engine
        if apply_live_options:
            try:
                self.qwen_engine.switch_options(
                    source_lang=self.config.get("qwen_src_lang"),
                )
            except Exception:
                pass
            try:
                _dev = self.config.get("compute", "auto")
                if (_dev in ("auto", "cpu", "cuda", "gpu") and self.qwen_engine is not None
                        and getattr(self.qwen_engine, "device", "auto") != _dev):
                    self.qwen_engine.device = _dev
            except Exception:
                pass
        self._apply_dictionary(self.qwen_engine)
        return self.qwen_engine










    TAB_IDS = ("live", "srt", "note", "import")
    ENGINE_KINDS = ("Moonshine v2", "Canary-1B", "Whisper", "Qwen3-ASR")
    _WHISPER_IDS = ("tiny", "base", "small", "medium", "large",
                    "large-v1", "large-v2", "large-v3", "large-v3-turbo")
    _QWEN_IDS = ("qwen3-asr-1.7b-q8_0", "qwen3-asr-1.7b-q6_k",
                 "qwen3-asr-1.7b-q5_k_m", "qwen3-asr-1.7b-q4_k_m")
    _ARCHES = (0, 1, 2, 3, 4, 5)



    _ENGINE_RAM_MB = {"Moonshine v2": 300, "Canary-1B": 5500,
                      "Qwen3-ASR": 2800}
    _WHISPER_RAM_MB = {"tiny": 900, "base": 950, "small": 1250,
                       "medium": 1900, "large": 3400, "large-v1": 3400,
                       "large-v2": 3400, "large-v3": 3400,
                       "large-v3-turbo": 2000}

    @staticmethod
    def _engine_slot(kind):
        """Cache slot name for an engine kind (None for shared Moonshine)."""
        return {
            "Canary-1B": "canary",
            "Whisper": "whisper",
            "Qwen3-ASR": "qwen",
        }.get(str(kind or ""))

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
        if k == "Qwen3-ASR":
            if w not in MoonshineSTTApp._QWEN_IDS:
                w = "qwen3-asr-1.7b-q8_0"
        elif w not in MoonshineSTTApp._WHISPER_IDS:
            w = "large-v3"
        return {"kind": k, "arch": a, "wmodel": w}

    @staticmethod
    def _engine_ram_mb(kind, arch=None, wmodel=None):
        try:
            if kind == "Whisper":


                try:
                    import whisper_engine

                    return int(whisper_engine._resident_model_mb(
                        wmodel or "large-v3"))
                except Exception:
                    return int(MoonshineSTTApp._WHISPER_RAM_MB.get(
                        wmodel or "large-v3", 3400))
            if kind == "Qwen3-ASR":
                try:
                    import qwen_engine

                    return int(qwen_engine._resident_model_mb(
                        wmodel or "qwen3-asr-1.7b-q8_0"))
                except Exception:
                    return 2800
            return int(MoonshineSTTApp._ENGINE_RAM_MB.get(kind, 300))
        except Exception:
            return 3400

    def _ram_warning(self, kind, arch=None, wmodel=None):
        """Reason string when this engine clearly will not fit in free RAM, else
        None. Catches the "second engine at 4:35" case before it is clicked."""
        try:
            import whisper_engine

            free = whisper_engine._available_ram_mb()

            if free is None:
                return None

            need = self._engine_ram_mb(kind, arch, wmodel)
            headroom = whisper_engine._RESIDENT_HEADROOM_MB

            if free - need < headroom:
                return (f"Only {free} MB RAM free; {kind} needs about "
                        f"{need} MB. Free memory or pick a smaller model, "
                        f"otherwise transcription will fail mid-session.")
        except Exception:
            pass

        return None

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
            "import": {"canary": None, "whisper": None, "wmodel": None},
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

    def _set_universal_whisper_model(self, mid, source=""):
        """One model picker for all tabs: config + every tab node + every
        menu follow the same Whisper model, so only one model is ever
        wanted (and only one instance ever loaded). Returns the id."""
        try:
            valid = set(getattr(MoonshineSTTApp, "_WHISPER_IDS", ())
                        or ("tiny", "base", "small", "medium",
                            "large-v1", "large-v2", "large-v3",
                            "large-v3-turbo"))
        except Exception:
            valid = ("tiny", "base", "small", "medium",
                     "large-v1", "large-v2", "large-v3",
                     "large-v3-turbo")
        mid = str(mid or "large-v3")
        if mid not in valid:
            mid = "large-v3"
        if (mid == "large-v3-turbo"
                and self.config.get("whisper_task") == "translate"):





            self._log(
                "Whisper turbo is transcription-only; "
                "switching task to transcribe"
            )
            try:
                with _CONFIG_LOCK:
                    self.config["whisper_task"] = "transcribe"
                    save_local_config(self.config)
            except Exception:
                pass
            try:
                if getattr(self, "gui", None) is not None:
                    self.gui.canary_task_var.set("transcribe")
            except Exception:
                pass
            try:
                _we = getattr(self, "whisper_engine", None)
                if _we is not None:
                    _we.switch_options(task="transcribe")
            except Exception:
                pass
            try:
                if getattr(self, "gui", None) is not None:
                    self.gui.set_status(
                        "Whisper Turbo selected - Task switched to Transcribe.",
                        WARNING,
                    )
            except Exception:
                pass
        try:
            with _CONFIG_LOCK:
                self.config["whisper_model"] = mid
                save_local_config(self.config)
        except Exception:
            pass
        try:
            for _node in ((self._tab_sel or {}).values()):
                try:
                    if isinstance(_node, dict):
                        _node["wmodel"] = mid
                except Exception:
                    continue
            self._save_tab_sel()
        except Exception:
            pass
        try:
            self._repaint_whisper_model_menus(mid)
        except Exception:
            pass
        return mid

    def _repaint_whisper_model_menus(self, mid):
        """Repaint every model menu to the universal Whisper id (menus only,
        no callbacks fired). Never raises."""
        try:
            g = self.gui
            if g is None:
                return
            try:
                self._refresh_model_row()
            except Exception:
                pass
            for _tab, _setter in (("srt", "set_srt_engine_state"),
                                  ("note", "set_note_engine_state"),
                                  ("import", "set_import_engine_state")):
                try:
                    sel = self.tab_selection(_tab)
                    if sel.get("kind") != "Whisper":
                        continue
                    fn = getattr(g, _setter, None)
                    if callable(fn):
                        fn(sel.get("kind"), sel.get("arch"), mid)
                except Exception:
                    continue
        except Exception:
            pass

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
            if k == "Qwen3-ASR":
                return f"Qwen3-ASR {sel['wmodel']}"
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

    def _live_hard_active(self) -> bool:
        """True only for recording/queued/processing/switching - NOT mere
        model loading. A pure load may be abandoned (its generation is
        bumped and the new pick supersedes it). Never raises."""
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
        except Exception:
            pass
        return False

    def _tab_hard_active(self, tab) -> bool:
        """Blocking activity for engine retargeting. Live excludes pure
        loading (abandonable); other tabs' tab_active() already means a
        real session. Never raises."""
        try:
            if tab == "live":
                return self._live_hard_active()
            return bool(self.tab_active(tab))
        except Exception:
            return True

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
            if tab == "import":
                return bool(getattr(self, "_import_file_busy", False)
                            or getattr(self, "_note_file_busy", False))
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
            if tab == "import":
                return "Import file is transcribing"
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
                if want_wmid not in self._WHISPER_IDS \
                        and want_wmid not in self._QWEN_IDS:
                    return {"action": "revert", "reason": "bad model"}
            if want_kind == cur["kind"] and want_arch == cur["arch"] \
                    and want_wmid == cur["wmodel"]:
                return {"action": "ok", "kind": want_kind, "arch": want_arch,
                        "wmodel": want_wmid, "noop": True, "mirrored": []}







            if self.tab_active(tab):
                try:
                    hard = self._tab_hard_active(tab)
                except Exception:
                    hard = True
                if hard:
                    return {
                        "action": "revert",
                        "reason": ("Stop this tab's recording or processing "
                                   "before changing its engine."),
                    }

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
                elif want_kind == "Qwen3-ASR":
                    what = f"Qwen3-ASR {want_wmid} (~{ram / 1000:.1f} GB extra RAM)"
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

    def _tab_cache_loading(self, tab) -> bool:
        """True when one of the tab's cached heavy engines is mid-load.
        Callers must not unload/replace then: the load owns the instance
        until it settles (engines discard stale generations themselves)."""
        try:
            cache = (self._tab_cache or {}).get(tab) or {}
            return any(bool(getattr(cache.get(s), "_loading", False))
                       for s in ("canary", "whisper", "qwen"))
        except Exception:
            return False

    def _drop_tab_cache(self, tab, except_engines=()):
        """Unload + forget a tab's dedicated heavy instances. Returns how
        many live objects were dropped. Engines in except_engines are left
        untouched (e.g. Note's shared Whisper mid-load). Never raises."""
        dropped = 0
        skip_ids = {id(e) for e in (except_engines or ()) if e is not None}
        try:
            cache = (self._tab_cache or {}).get(tab)
            if not isinstance(cache, dict):
                return 0
            for slot in ("canary", "whisper", "qwen"):
                try:
                    eng = cache.get(slot)
                    if eng is not None and id(eng) in skip_ids:
                        continue
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
                if not any(cache.get(s) is not None
                           for s in ("canary", "whisper", "qwen")):
                    cache["wmodel"] = None
            except Exception:
                pass
        except Exception:
            pass
        return dropped

    @staticmethod
    def _heavy_occupied(eng) -> bool:
        """True when an engine holds (or is about to hold) gigabytes: ready,
        or mid-load. A failed/never-loaded instance holds nothing."""
        try:
            if eng is None:
                return False
            if bool(getattr(eng, "is_ready", False)):
                return True
            return bool(getattr(eng, "_loading", False))
        except Exception:
            return False

    def _resident_heavy_holders(self, except_tab=None, except_engines=()):
        """Tabs holding (or loading) a heavy (Whisper/Canary) engine:
        [(tab, slot, desc, ram_mb)]. A mid-load engine counts: it will own
        the RAM within seconds, which is exactly the race that OOMs the
        second loader. Moonshine is light and shared, so it never counts.
        Engines in except_engines (by identity) are not reported, and are
        also skipped when their holder is later swapped out. Never raises."""
        holders = []
        skip_ids = {id(e) for e in (except_engines or ()) if e is not None}
        try:
            for tab in self.TAB_IDS:
                if tab == except_tab:
                    continue
                try:
                    if tab == "live":
                        candidates = (
                            ("whisper", getattr(self, "whisper_engine", None)),
                            ("canary", getattr(self, "canary_engine", None)),
                            ("qwen", getattr(self, "qwen_engine", None)),
                        )
                    else:
                        cache = (self._tab_cache or {}).get(tab) or {}
                        candidates = (
                            ("whisper", cache.get("whisper")),
                            ("canary", cache.get("canary")),
                            ("qwen", cache.get("qwen")),
                        )
                    for slot, eng in candidates:
                        try:
                            if id(eng) in skip_ids:
                                continue
                            if not self._heavy_occupied(eng):
                                continue
                            if slot == "whisper":
                                mid = str(getattr(eng, "model_id", "")
                                          or "large-v3")
                                ram = self._engine_ram_mb("Whisper", None, mid)
                                desc = (f"Live Whisper {mid}" if tab == "live"
                                        else f"{tab} Whisper {mid}")
                            elif slot == "qwen":
                                mid = str(getattr(eng, "model_id", "")
                                          or "qwen3-asr-1.7b-q8_0")
                                ram = self._engine_ram_mb("Qwen3-ASR", None, mid)
                                desc = (f"Live Qwen3-ASR {mid}" if tab == "live"
                                        else f"{tab} Qwen3-ASR {mid}")
                            else:
                                ram = self._engine_ram_mb("Canary-1B")
                                desc = ("Live Canary-1B" if tab == "live"
                                        else f"{tab} Canary-1B")
                            try:
                                if not bool(getattr(eng, "is_ready", False)):
                                    desc += " (loading)"
                            except Exception:
                                pass
                            holders.append((tab, slot, desc, ram))
                        except Exception:
                            continue
                except Exception:
                    continue
        except Exception:
            pass
        return holders

    def _swap_out_tab_heavies(self, tab, except_engines=()):
        """Unload a tab's heavy engines but keep the objects, so the tab can
        reload them on next use. Covers ready engines and mid-load ones
        (unload bumps the generation, so the in-flight load discards itself
        instead of claiming RAM). Engines in except_engines are preserved.
        Returns ~MB freed. Moonshine (light, shared) is never touched.
        Never raises."""
        freed = 0
        skip_ids = {id(e) for e in (except_engines or ()) if e is not None}
        try:
            if tab == "live":
                targets = (
                    ("whisper_engine", "Whisper"),
                    ("canary_engine", "Canary-1B"),
                    ("qwen_engine", "Qwen3-ASR"),
                )
                for attr, kind in targets:
                    try:
                        eng = getattr(self, attr, None)
                        if id(eng) in skip_ids:
                            continue
                        if not self._heavy_occupied(eng):
                            continue
                        if kind == "Whisper":
                            mid = str(getattr(eng, "model_id", "")
                                      or "large-v3")
                            freed += self._engine_ram_mb(kind, None, mid)
                        else:
                            freed += self._engine_ram_mb(kind)
                        try:
                            eng.unload()
                        except Exception:
                            pass
                    except Exception:
                        continue
            else:
                try:
                    cache = (self._tab_cache or {}).get(tab) or {}
                    for slot in ("whisper", "canary", "qwen"):
                        try:
                            eng = cache.get(slot)
                            if id(eng) in skip_ids:
                                continue
                            if not self._heavy_occupied(eng):
                                continue
                            if slot == "whisper":
                                mid = str(getattr(eng, "model_id", "")
                                          or cache.get("wmodel")
                                          or "large-v3")
                                freed += self._engine_ram_mb("Whisper", None,
                                                             mid)
                            elif slot == "qwen":
                                mid = str(getattr(eng, "model_id", "")
                                          or cache.get("wmodel")
                                          or "qwen3-asr-1.7b-q8_0")
                                freed += self._engine_ram_mb("Qwen3-ASR", None,
                                                             mid)
                            else:
                                freed += self._engine_ram_mb("Canary-1B")
                        except Exception:
                            continue
                except Exception:
                    pass
                self._drop_tab_cache(tab, except_engines=except_engines)
        except Exception:
            pass
        return freed

    def _prepare_heavy_load(self, except_tabs=(), except_engines=()):
        """Unload every idle heavy holder (ready or mid-load) except the
        listed tabs/engines, so a model switch/load starts from free RAM.
        Tabs with a running session keep theirs. A foreground switch/record
        intent wins over a background load: the orphaned load discards itself
        by generation. Returns {"freed": mb, "swapped": [(tab, desc)],
        "kept": [(tab, desc)]}. GUI-thread only. Never raises."""
        result = {"freed": 0, "swapped": [], "kept": []}
        try:
            skip = set(except_tabs or ())
            try:
                holders = self._resident_heavy_holders(
                    except_engines=except_engines)
            except Exception:
                holders = []
            for (holder_tab, _slot, desc, _mb) in holders:
                if holder_tab in skip:
                    continue
                try:
                    if self._tab_hard_active(holder_tab):
                        result["kept"].append((holder_tab, desc))
                        continue
                except Exception:
                    result["kept"].append((holder_tab, desc))
                    continue
                try:
                    result["freed"] += (
                        self._swap_out_tab_heavies(
                            holder_tab, except_engines=except_engines) or 0)
                    result["swapped"].append((holder_tab, desc))
                except Exception:
                    pass
        except Exception:
            pass
        return result

    def unload_tab_engines(self, tab):
        """Manual 'Unload' action for a tab's own engines. Refuses while the
        tab runs a session (never yanks a live recording); otherwise unloads
        and reports what was freed. Returns a one-line status message."""
        try:
            try:
                if self._tab_hard_active(tab):
                    return (f"{self._tab_active_desc(tab)} — stop it first, "
                            f"then unload.")
            except Exception:
                pass
            holders = [h for h in self._resident_heavy_holders()
                       if h[0] == tab]
            if not holders:
                return "No loaded engine on this tab — nothing to free."
            freed = self._swap_out_tab_heavies(tab)
            names = ", ".join(sorted({d for _t, _s, d, _m in holders}))
            return (f"Unloaded {names} "
                    f"(~{freed / 1000:.1f} GB freed).")
        except Exception as e:
            return f"Unload failed: {e}"

    def unload_other_tab_engines(self, tab):
        """Manual 'Free memory' action: unload every OTHER tab's idle heavy
        engines. Tabs with a running session keep theirs (named in the
        message). Returns a one-line status message."""
        try:
            holders = self._resident_heavy_holders(except_tab=tab)
            if not holders:
                return "No other tab holds an engine — nothing to free."
            freed, kept = 0, []
            for (holder, _slot, desc, _mb) in holders:
                try:
                    if self._tab_hard_active(holder):
                        kept.append(desc)
                        continue
                except Exception:
                    kept.append(desc)
                    continue
                freed += self._swap_out_tab_heavies(holder) or 0
            msg = f"Freed ~{freed / 1000:.1f} GB from idle engines."
            if kept:
                msg += f" Kept (in use): {'; '.join(kept)}."
            return msg
        except Exception as e:
            return f"Free-memory failed: {e}"

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
                    source_lang=self.config.get("whisper_src_lang", "auto"),
                    target_lang="en",
                    model_id=wmid or self.config.get("whisper_model", "large-v3"),
                    device=self.config.get("compute", "auto"),
                    on_ready=None,
                )
            if kind == "Qwen3-ASR":
                from qwen_engine import QwenEngine
                return QwenEngine(
                    task="transcribe",
                    source_lang=self.config.get("qwen_src_lang", "auto"),
                    target_lang="en",
                    model_id=wmid or self.config.get("qwen_model",
                                                     "qwen3-asr-1.7b-q8_0"),
                    device=self.config.get("compute", "auto"),
                    on_ready=None,
                )
        except Exception as e:
            print(f"tab engine build failed: {e}")
        return None

    @staticmethod
    def _whisper_cache_ok(eng, want):
        """Ready-for-use check: instance matches the wanted model id AND is
        ready. Missing model_id attr + ready counts as a match (never
        rebuild-loop on exotic engine objects)."""
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

    @staticmethod
    def _whisper_instance_matches(eng, want, cached_model=None):
        """Identity check independent of loading/readiness state. A matching
        engine that is still loading must be kept, never unloaded and
        recreated (that churn is pure waste)."""
        if eng is None:
            return False
        try:
            model_id = getattr(eng, "model_id", None)
            if not model_id:
                model_id = cached_model
            return str(model_id or "") == str(want)
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
            elif kind == "Qwen3-ASR":
                slot = "qwen"
            else:
                return None
            cache = (self._tab_cache or {}).get(tab)
            if not isinstance(cache, dict):
                return None
            eng = cache.get(slot)
            if slot == "whisper":



                try:
                    return self._get_whisper_engine(False)
                except Exception:
                    return None
            else:
                if eng is None:
                    eng = self._make_engine(kind, wmid=wmid)
                    try:
                        cache[slot] = eng
                        if slot == "qwen":
                            cache["wmodel"] = wmid
                    except Exception:
                        pass
                elif slot == "qwen":

                    try:
                        want = str(wmid or self.config.get(
                            "qwen_model", "qwen3-asr-1.7b-q8_0"))
                        if str(getattr(eng, "model_id", "") or "") != want:
                            eng.switch_model(want)
                            cache["wmodel"] = want
                    except Exception:
                        pass
            self._apply_dictionary(eng)
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



            plan = self.plan_tab_change(tab, kind=kind, arch=arch,
                                        wmid=wmid)
            action = (plan or {}).get("action")
            if action == "revert":
                return plan
            if action not in ("ok", "confirm"):
                return {"action": "revert",
                        "reason": "invalid engine-change verdict"}
            if plan.get("noop"):
                return {"action": "ok", "mirrored": []}
            if action == "confirm" and not dual_ok:
                return {"action": "revert",
                        "reason": "other session active"}
            kind = plan["kind"]
            arch = plan["arch"]
            wmid = plan["wmodel"]


            try:
                if kind == "Whisper":
                    wmid = self._set_universal_whisper_model(
                        wmid, source=tab)
            except Exception:
                pass

            try:
                if tab == "live":
                    with _CONFIG_LOCK:
                        self.config["engine"] = kind
                        if kind == "Moonshine v2":
                            self.config["model_arch"] = int(arch)
                        elif kind == "Whisper":
                            self.config["whisper_model"] = str(wmid)
                        elif kind == "Qwen3-ASR":
                            self.config["qwen_model"] = str(wmid)
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



            mirrored = []
            try:
                for t in ("srt", "note", "import"):
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





            try:
                if not self._tab_cache_loading(tab):
                    self._drop_tab_cache(tab)
            except Exception:
                pass
            try:
                prep = self._prepare_heavy_load(except_tabs=(tab,))
                freed_tabs = sorted(
                    {t for t, _d in prep.get("swapped", [])})
                if freed_tabs:
                    try:
                        self._log("Unloaded idle engine(s) for %s (RAM reclaimed)"
                                  % ", ".join(freed_tabs))
                    except Exception:
                        pass
            except Exception:
                pass

            try:
                if tab in ("live", "note") and kind in ("Canary-1B", "Whisper",
                                                        "Qwen3-ASR"):
                    if tab == "live":
                        pass
                    else:
                        eng = self._tab_heavy(tab, kind, wmid)
                        try:
                            if eng is not None and kind in ("Whisper", "Qwen3-ASR"):
                                if not self._whisper_cache_ok(eng, wmid):
                                    eng.switch_model(wmid)
                                elif not eng.is_ready:
                                    eng.load()
                            elif eng is not None and not eng.is_ready:
                                eng.load()
                        except Exception:
                            pass
            except Exception:
                pass




            return {"action": "ok", "mirrored": mirrored}
        except Exception as e:
            return {"action": "revert", "reason": str(e)}

    def engine_start_cost(self, tab):
        """Return (needs_load, estimated_mb, description) for starting work
        on tab now. Live heavy engines live in self.canary_engine /
        self.whisper_engine (never in _tab_cache["live"]) - the old code
        always reported them "not loaded" and forced bogus dual-engine
        dialogs on Live. needs_load False for an already-loading engine
        means "no additional allocation", NOT "ready to transcribe" -
        callers must still wait for readiness."""
        try:
            if tab not in self.TAB_IDS:
                raise ValueError(f"Unknown tab: {tab}")
            sel = self.tab_selection(tab)
            kind = sel["kind"]
            if kind == "Moonshine v2":
                return False, 0, "Moonshine (shared)"
            slot = self._engine_slot(kind)
            if slot == "canary":
                label = "Canary-1B"
            elif slot == "qwen":
                label = f"Qwen3-ASR {sel['wmodel']}"
            else:
                label = f"Whisper {sel['wmodel']}"
            if slot == "whisper":


                eng = getattr(self, "whisper_engine", None)
                try:
                    cached_model = self.config.get("whisper_model",
                                                   sel["wmodel"])
                except Exception:
                    cached_model = sel["wmodel"]
            elif tab == "live":
                eng = getattr(self, f"{slot}_engine", None)
                cached_model = None
            else:
                cache = (self._tab_cache or {}).get(tab) or {}
                eng = cache.get(slot)
                cached_model = cache.get("wmodel")
            matches = eng is not None
            if matches and slot in ("whisper", "qwen"):
                model_id = getattr(eng, "model_id", None) or cached_model
                matches = str(model_id or "") == str(sel["wmodel"])
            if matches:
                try:
                    if bool(getattr(eng, "is_ready", False)):
                        return False, 0, f"{label} (ready)"
                    if bool(getattr(eng, "_loading", False)):
                        return False, 0, f"{label} (loading)"
                except Exception:
                    pass
            ram = self._engine_ram_mb(kind, sel["arch"], sel["wmodel"])
            return True, ram, f"{label} (not loaded)"
        except Exception as exc:
            try:
                self._log(f"Engine cost check failed for {tab}: {exc}")
            except Exception:
                pass

            return True, 3000, "Engine state unavailable"

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
            if sel["kind"] == "Canary-1B":
                what = "Canary-1B"
            elif sel["kind"] == "Qwen3-ASR":
                what = f"Qwen3-ASR {sel['wmodel']}"
            else:
                what = f"Whisper {sel['wmodel']}"
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
        be cached+ready (preloaded at selection) or they load now. Note wins
        over other tabs' resident models: idle holders are swapped out
        (unloaded, RAM freed) before Note loads; a tab with a running
        session is only swapped on user confirmation."""
        try:
            try:
                if bool(getattr(self, "_note_file_busy", False)):
                    return {"abort": True}
                if self.gui is not None and bool(
                        getattr(self.gui, "_note_file_running", False)):
                    return {"abort": True}
            except Exception:
                pass
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
            slot = self._engine_slot(kind)



            try:
                eng = self._tab_heavy("note", kind, sel["wmodel"])
            except Exception:
                eng = None
            ok = False
            try:
                if eng is not None and eng.is_ready:
                    if slot in ("whisper", "qwen"):
                        ok = self._whisper_cache_ok(eng, sel["wmodel"])
                    else:
                        ok = True
            except Exception:
                ok = False
            if ok:
                self._note_engine_obj = eng
                return {"go": True}







            prep = self._prepare_heavy_load(
                except_tabs=("note",),
                except_engines=(eng,) if eng is not None else ())
            hard = list(prep.get("kept") or [])
            freed = int(prep.get("freed") or 0)
            if hard:
                if kind == "Canary-1B":
                    what = "Canary-1B"
                elif kind == "Qwen3-ASR":
                    what = f"Qwen3-ASR {sel['wmodel']}"
                else:
                    what = f"Whisper {sel['wmodel']}"
                try:
                    self._note_swap_tabs = [t for t, _d in hard]
                except Exception:
                    pass
                who = "; ".join(d for _t, d in hard)
                return {"confirm": f"{who}.\n"
                                    f"Switch to Note ({what})?\n"
                                    f"Yes = unload that engine, free the RAM, "
                                    f"then load Note's. No = stay as you are."}
            try:
                self._note_swap_tabs = []
            except Exception:
                pass




            warn = self._ram_warning(kind, sel["arch"], sel["wmodel"])

            if warn:

                return {"abort": True, "reason": warn}

            try:
                self._note_kick_load(kind, sel["wmodel"])
            except Exception:
                pass
            if freed > 0:
                return {"wait": f"Unloaded idle engine(s) (~{freed / 1000:.1f} GB "
                                f"freed). Loading note engine — recording "
                                f"starts automatically."}
            return {"wait": "Loading note engine — recording starts automatically."}
        except Exception:
            return {"abort": True}

    def _note_kick_load(self, kind, wmid):
        """Start (or retarget) the load for the Note tab's selection.

        Whisper is the shared universal instance, so a model pick that does
        not match the live object switches it instead of loading the stale
        model. Never raises."""
        try:
            eng = self._tab_heavy("note", kind, wmid)
            if eng is None:
                return
            if kind in ("Whisper", "Qwen3-ASR"):
                if self._whisper_cache_ok(eng, wmid):
                    if (not bool(getattr(eng, "is_ready", False))
                            and not bool(getattr(eng, "_loading", False))):
                        eng.load()
                    return
                if bool(getattr(eng, "_loading", False)):
                    return
                eng.switch_model(wmid)
                return
            if (not bool(getattr(eng, "is_ready", False))
                    and not bool(getattr(eng, "_loading", False))):
                eng.load()
        except Exception:
            pass

    def note_engine_ready(self):
        """Return (ready, problem) for the current Note selection.

        Reports missing engines and recorded load failures instead of
        polling "still loading" forever, never restarts an engine that
        already recorded a load error on every poll (the record pre-flight
        re-kicks on the next explicit user attempt), and snapshots the
        engine when ready. Kicks a missing load, never blocks. Assumes
        eng.load() starts an asynchronous load."""
        try:
            sel = self.tab_selection("note")
            kind = sel["kind"]
            if kind == "Moonshine v2":
                eng = self.moonshine_engine
                if eng is None:
                    return False, "Note engine unavailable."
                try:
                    if bool(getattr(eng, "is_ready", False)):
                        self._note_engine_obj = eng
                        return True, ""
                    if bool(getattr(eng, "_loading", False)):
                        return False, ""
                    err = getattr(eng, "_last_error", None)
                    if err:
                        return False, f"Note engine failed: {err}"
                    eng.load()
                except Exception:
                    pass
                return False, ""
            slot = self._engine_slot(kind)




            try:
                eng = self._tab_heavy("note", kind, sel["wmodel"])
            except Exception:
                eng = None
            if eng is None:
                return False, "Note engine could not be created."
            if slot in ("whisper", "qwen"):
                try:
                    if not self._whisper_cache_ok(eng, sel["wmodel"]):
                        err = getattr(eng, "_last_error", None)
                        if err:


                            return False, f"Note engine failed: {err}"
                        if bool(getattr(eng, "_loading", False)):
                            return False, ""
                        mid = str(getattr(eng, "model_id", "") or "")
                        if mid and mid != str(sel["wmodel"]):


                            self._note_kick_load(kind, sel["wmodel"])
                            return False, ""


                except Exception:
                    pass
            try:
                if bool(getattr(eng, "_loading", False)):
                    return False, ""
                if bool(getattr(eng, "is_ready", False)):
                    self._note_engine_obj = eng
                    return True, ""
                err = getattr(eng, "_last_error", None)
                if err:
                    return False, f"Note engine failed: {err}"
                eng.load()
                if bool(getattr(eng, "is_ready", False)):

                    self._note_engine_obj = eng
                    return True, ""
                err = getattr(eng, "_last_error", None)
                if err and not bool(getattr(eng, "_loading", False)):
                    return False, f"Note engine failed: {err}"
            except Exception:
                pass
            return False, ""
        except Exception as exc:
            import traceback
            traceback.print_exc()
            return False, f"Note engine failed: {exc}"

    def note_record_confirm(self, dual_ok):
        """Follow-up after the record pre-flight asked to confirm a swap:
        Yes unloads the holding tab's engine (freeing its RAM), then Note
        loads; No stays as-is."""
        try:
            if not dual_ok:
                return {"abort": True}
            try:
                for tab in list(getattr(self, "_note_swap_tabs", None) or []):
                    self._swap_out_tab_heavies(tab)
            except Exception:
                pass
            try:
                self._note_swap_tabs = []
            except Exception:
                pass
            sel = self.tab_selection("note")
            try:
                self._note_kick_load(sel["kind"], sel["wmodel"])
            except Exception:
                pass
            return {"wait": "Loading note engine — recording starts automatically."}
        except Exception:
            return {"abort": True}

    def tab_shown(self, tabid):
        """Tab-switch adopt: an idle srt/note/import tab that was never
        explicitly picked follows the shared default (menus only - engines
        load at use). Live is the anchor and never auto-adopts. Never
        raises; never touches an active tab or a tab with a load in
        flight, and never rewrites config when nothing changed."""
        try:


            if tabid in ("live", "srt", "note", "import"):
                tab = tabid
            else:
                mapping = {"SRT File": "srt", "Note": "note",
                           "Import": "import", "Live": "live"}
                tab = mapping.get(tabid)
            if tab not in ("srt", "note", "import"):
                return
            if self.tab_active(tab):
                return
            node = (self._tab_sel or {}).get(tab)
            if not isinstance(node, dict) or bool(node.get("explicit", False)):
                return


            try:
                cache = (self._tab_cache or {}).get(tab) or {}
                if any(bool(getattr(cache.get(slot), "_loading", False))
                       for slot in ("canary", "whisper", "qwen")):
                    return
            except Exception:
                pass
            try:
                sh = self._shared_sel
            except Exception:
                return
            try:
                selection = self._norm_tab_sel(sh.get("kind"),
                                               sh.get("arch"),
                                               sh.get("wmodel"))
            except Exception:
                return
            if any(node.get(k) != v for k, v in selection.items()):
                try:
                    node.update(selection)
                    self._save_tab_sel()
                except Exception:
                    pass
            try:
                g = self.gui
                if g is not None:
                    if tab == "srt":
                        g.set_srt_engine_state(selection["kind"],
                                               int(selection["arch"]),
                                               str(selection["wmodel"]))
                    elif tab == "note":
                        g.set_note_engine_state(selection["kind"],
                                                int(selection["arch"]),
                                                str(selection["wmodel"]))
                    elif tab == "import":
                        g.set_import_engine_state(selection["kind"],
                                                  int(selection["arch"]),
                                                  str(selection["wmodel"]))
            except Exception:
                pass
        except Exception:
            pass

    def _sync_mirrored_menus(self, mirrored):
        """Paint follower (srt/note/import) menu states after an apply. The
        live tab owns its menus via the engine/model flows. Never raises."""
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
                    elif t == "import":
                        g.set_import_engine_state(k, a, w)
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
                    _s = self.config.get("whisper_src_lang", "auto")
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
        active tab's instances - or instances with a load in flight (the
        load owns them until it settles). Called with the regular idle
        sweep."""
        try:
            for tab in ("srt", "note", "import"):
                try:
                    if self.tab_active(tab):
                        continue
                    sel = self.tab_selection(tab)
                    cache = (self._tab_cache or {}).get(tab)
                    if not isinstance(cache, dict):
                        continue


                    try:
                        eng = cache.get("whisper")
                        if eng is not None and not bool(
                                getattr(eng, "_loading", False)):
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

                    try:
                        eng = cache.get("canary")
                        if eng is not None and sel["kind"] != "Canary-1B" \
                                and not bool(getattr(eng, "_loading", False)):
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
        if name not in ("Live", "SRT File", "Note", "Import"):
            return
        if name in ("Live", "SRT File", "Note", "Import"):
            with _CONFIG_LOCK:
                self.config["srt_tab"] = name
                save_local_config(self.config)

        if name == "Live":
            try:
                self._refresh_input_devices(refresh=True)
            except Exception:
                pass

        try:
            mapping = {"Live": "live", "SRT File": "srt", "Note": "note",
                       "Import": "import"}
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
        elif active == "Qwen3-ASR" and self.qwen_engine is not None:
            try:
                self.qwen_engine.device = code
            except Exception:
                pass
            target = self.qwen_engine
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
                    actual = getattr(target, "_device_used", "unknown")
                    compute = getattr(target, "_compute_used", "")
                    fallback = code in ("gpu", "cuda") and actual != "cuda"

                    status = f"Ready • {actual.upper()} {compute}".strip()
                    if fallback:
                        status += " (GPU request fell back to CPU)"
                        self._log(
                            "Compute fallback: "
                            + str(getattr(target, "_device_reason", ""))
                        )

                    self.gui.set_status(
                        status, WARNING if fallback else SUCCESS
                    )
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
            self._prepare_heavy_load(except_tabs=("live",))
        except Exception:
            pass
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
    def _on_itn_live_changed(self, value: bool):
        self.config["itn_live"] = bool(value)
        save_local_config(self.config)
        self._log(f"Numbers -> digits (ITN) {'on' if value else 'off'}")
    def _on_itn_hint_changed(self, value: bool):
        self.config["itn_hint"] = bool(value)
        save_local_config(self.config)
        self._log(f"Qwen digit prompt {'on' if value else 'off'}")
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
                def _fn(base_label, _table=WHISPER_MODEL_CHOICES, _dl=dl):
                    try:
                        mid = _table.get(base_label)
                        if not mid:
                            return ""
                        return "✓" if _dl.get(mid) else "↓"
                    except Exception:
                        return ""
                return _fn
            if eng == "Qwen3-ASR":
                from qwen_engine import (QWEN_MODEL_CHOICES,
                                         qwen_downloaded_map)
                try:
                    dl = qwen_downloaded_map()
                except Exception:
                    dl = {}
                def _fnq(base_label, _table=QWEN_MODEL_CHOICES, _dl=dl):
                    try:
                        mid = _table.get(base_label)
                        if not mid:
                            return ""
                        return "✓" if _dl.get(mid) else "↓"
                    except Exception:
                        return ""
                return _fnq
            if eng == "Canary-1B":
                return None
            from gui import MODEL_CHOICES
            from engine import moonshine_downloaded_map
            try:
                dl = moonshine_downloaded_map()
            except Exception:
                dl = {}
            def _fn2(base_label, _table=MODEL_CHOICES, _dl=dl):
                try:
                    arch = _table.get(base_label)
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
                model_id = str(self.config.get("whisper_model", "large-v3"))
                cur = WHISPER_MODEL_CHOICES_REV.get(
                    model_id, "Large v3 (3GB, best)")
                self.gui.set_model_options(list(WHISPER_MODEL_CHOICES.keys()),
                                           cur, self._on_model_changed, fn)


                try:
                    import capabilities as _cap
                    tasks = _cap.task_options("Whisper", model_id)
                    old_task = str(self.config.get("whisper_task",
                                                   "transcribe")
                                   or "").strip().lower()
                    task = old_task if old_task in tasks else tasks[0]
                    if task != old_task:
                        with _CONFIG_LOCK:
                            self.config["whisper_task"] = task
                        save_local_config(self.config)
                        self._log(f"Whisper {model_id} cannot {old_task}; "
                                  f"task set to {task}")
                    self.gui.set_task_options(tasks, current=task)
                except Exception:
                    pass
            elif eng == "Canary-1B":
                from gui import CANARY_MODEL_LABEL
                self.gui.set_model_options([CANARY_MODEL_LABEL], CANARY_MODEL_LABEL,
                                           self._on_model_changed, None)
            elif eng == "Qwen3-ASR":
                from qwen_engine import (QWEN_MODEL_CHOICES,
                                         QWEN_MODEL_CHOICES_REV,
                                         QWEN_DEFAULT_MODEL)
                cur = QWEN_MODEL_CHOICES_REV.get(
                    self.config.get("qwen_model", QWEN_DEFAULT_MODEL),
                    QWEN_MODEL_CHOICES_REV[QWEN_DEFAULT_MODEL])
                self.gui.set_model_options(list(QWEN_MODEL_CHOICES.keys()),
                                           cur, self._on_model_changed, fn)
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


    def _open_dictionary(self):
        """Open the dictionary editor (Live tab button)."""
        if not self.gui:
            return
        try:
            self.gui.show_dictionary_manager(self._dictionary_data,
                                             self._dictionary_save)
        except Exception as e:
            print(f"dictionary dialog failed: {e}")

    def _dictionary_data(self):
        try:
            import dictionary as _dict
        except Exception:
            return {"terms": [], "context": "", "engines": {},
                    "available": {}}
        return {
            "terms": _dict.norm_terms(self.config.get("dict_terms")),
            "context": _dict.norm_context(self.config.get("dict_context")),
            "engines": _dict.enabled_map(self.config.get("dict_engines")),
            "available": _dict.available_map(),
        }

    def _dictionary_save(self, payload):
        """Persist the dialog payload and push it to every live engine."""
        try:
            import dictionary as _dict
        except Exception as e:
            return False, f"dictionary.py missing: {e}"
        payload = payload or {}
        terms = _dict.norm_terms(payload.get("terms"))
        context = _dict.norm_context(payload.get("context"))
        engines = _dict.enabled_map(payload.get("engines"))
        self.config["dict_terms"] = terms
        self.config["dict_context"] = context
        self.config["dict_engines"] = engines
        try:
            save_local_config(self.config)
        except Exception as e:
            return False, f"Could not save config: {e}"
        self._push_dictionary()
        try:
            if self.gui:
                self.gui.set_dictionary_status(
                    _dict.status_text(terms, context, engines))
        except Exception:
            pass
        try:
            self._log(_dict.status_text(terms, context, engines))
        except Exception:
            pass
        active = [e for e, on in engines.items() if on]
        if not active:
            return True, "Dictionary saved (not enabled for any engine)."
        return True, ("Dictionary saved for " + ", ".join(active) +
                      f" — {len(terms)} term(s).")

    def _apply_dictionary(self, engine):
        """Attach the saved dictionary to one engine instance (if it has one)."""
        try:
            setter = getattr(engine, "set_dictionary", None)
            if not callable(setter):
                return
            import dictionary as _dict
            kind = None
            if engine is getattr(self, "whisper_engine", None) or \
                    type(engine).__name__ == "WhisperEngine":
                kind = "Whisper"
            elif engine is getattr(self, "qwen_engine", None) or \
                    type(engine).__name__ == "QwenEngine":
                kind = "Qwen3-ASR"
            if kind is None:
                return
            enabled = bool(_dict.enabled_map(
                self.config.get("dict_engines")).get(kind))
            setter(self.config.get("dict_terms") or [],
                   self.config.get("dict_context") or "",
                   enabled)
        except Exception:
            pass

    def _push_dictionary(self):
        """Re-apply the saved dictionary to every engine that is alive."""
        seen = set()
        for attr in ("moonshine_engine", "whisper_engine", "canary_engine",
                     "qwen_engine", "_note_engine_obj"):
            try:
                eng = getattr(self, attr, None)
            except Exception:
                continue
            if eng is None or id(eng) in seen:
                continue
            seen.add(id(eng))
            self._apply_dictionary(eng)
        for tab, cache in (self._tab_cache or {}).items():
            if not isinstance(cache, dict):
                continue
            for slot in ("whisper", "qwen"):
                try:
                    eng = cache.get(slot)
                except Exception:
                    continue
                if eng is None or id(eng) in seen:
                    continue
                seen.add(id(eng))
                self._apply_dictionary(eng)

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
        try:
            from qwen_engine import (qwen_cache_info, qwen_downloaded_map,
                                     QWEN_MODEL_CHOICES_REV)
            qinfo = qwen_cache_info()
            qdl = qwen_downloaded_map()
            qmid = self.config.get("qwen_model", "qwen3-asr-1.7b-q8_0")
            for qid, qsize in qinfo.items():
                downloaded = bool(qdl.get(qid))
                if downloaded:
                    total += int(qsize)
                items.append({
                    "engine": "Qwen3-ASR", "kind": "qwen", "id": qid,
                    "label": QWEN_MODEL_CHOICES_REV.get(qid, qid),
                    "size": int(qsize) if downloaded else None,
                    "downloaded": downloaded,
                    "in_use": active == "Qwen3-ASR" and qmid == qid,
                })
        except Exception:
            pass
        try:
            for it in items:
                if not it.get("in_use") and self._model_in_use(
                        it.get("kind"), it.get("id")):
                    it["in_use"] = True
        except Exception:
            pass
        return {"items": items, "total": total}

    def _model_in_use(self, kind, ident) -> bool:
        """True when any tab (live config or srt/note/import selection)
        references this model. Deleting a referenced model file while its
        engine may load risks corrupt loads, so the manager blocks it."""
        try:
            if kind == "moonshine":
                try:
                    want = int(ident)
                except Exception:
                    return False
                try:
                    if (self.config.get("engine") == "Moonshine v2"
                            and int(self.config.get("model_arch", 5)) == want):
                        return True
                except Exception:
                    pass
                for t in ("srt", "note", "import"):
                    try:
                        sel = self.tab_selection(t)
                        if sel.get("kind") == "Moonshine v2" \
                                and int(sel.get("arch", -1)) == want:
                            return True
                    except Exception:
                        pass
                return False
            if kind == "whisper":
                try:
                    if (self.config.get("engine") == "Whisper"
                            and str(self.config.get("whisper_model",
                                                     "large-v3")) == str(ident)):
                        return True
                except Exception:
                    pass
                for t in ("srt", "note", "import"):
                    try:
                        sel = self.tab_selection(t)
                        if sel.get("kind") == "Whisper" \
                                and str(sel.get("wmodel", "")) == str(ident):
                            return True
                    except Exception:
                        pass
                return False
            if kind == "canary":
                try:
                    if self.config.get("engine") == "Canary-1B":
                        return True
                except Exception:
                    pass
                for t in ("srt", "note", "import"):
                    try:
                        if (self.tab_selection(t).get("kind")
                                == "Canary-1B"):
                            return True
                    except Exception:
                        pass
                return False
            if kind == "qwen":
                try:
                    if (self.config.get("engine") == "Qwen3-ASR"
                            and str(self.config.get("qwen_model",
                                                    "qwen3-asr-1.7b-q8_0"))
                            == str(ident)):
                        return True
                except Exception:
                    pass
                for t in ("srt", "note", "import"):
                    try:
                        sel = self.tab_selection(t)
                        if sel.get("kind") == "Qwen3-ASR" \
                                and str(sel.get("wmodel", "")) == str(ident):
                            return True
                    except Exception:
                        pass
                return False
        except Exception:
            pass
        return False
    def _model_manager_delete(self, engine, kind, ident, refresh=True):
        if self._srt_busy:
            return False, "Stop the running SRT/burn job first."
        try:
            if self._model_in_use(kind, ident):
                return False, ("In use by Live or a tab selection - switch "
                                "every tab off it first.")
        except Exception:
            pass
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
            elif kind == "qwen":
                if (self.config.get("engine") == "Qwen3-ASR"
                        and self.config.get("qwen_model") == ident):
                    return False, "Switch to another model first."
                from qwen_engine import delete_qwen_model
                freed = delete_qwen_model(ident)
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
        if self.config.get("engine") == "Qwen3-ASR":
            self._on_qwen_model_changed(display_label)
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


            self._model_switching = False
            def _ui():
                try:
                    self.gui.record_btn.configure(state="normal")
                except Exception:
                    pass
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
        try:
            self.engine.switch_model(new_arch, on_ready=_model_switched)
        except Exception as e:
            self._model_switching = False
            try:
                if self.gui:
                    self.gui.record_btn.configure(state="normal")
                    self.gui.set_status(f"Model error: {e}", DANGER)
            except Exception:
                pass
    def _on_whisper_model_changed(self, display_label: str):
        try:
            from gui import WHISPER_MODEL_CHOICES
        except Exception:
            return
        new_id = WHISPER_MODEL_CHOICES.get(display_label)
        if not new_id:
            return
        switch_to_transcribe = False
        if (new_id == "large-v3-turbo"
                and self.config.get("whisper_task") == "translate"):
            switch_task = False
            if self.gui is not None:
                try:
                    from tkinter import messagebox
                    switch_task = bool(messagebox.askyesno(
                        "Use Whisper Turbo?",
                        "Whisper large-v3-turbo can only transcribe.\n\n"
                        "Switch Task to Transcribe and use Turbo for all tabs?\n\n"
                        "Live, SRT, Note, and Import will all switch to "
                        "transcription. To translate again, select large-v3 "
                        "and set Task to Translate.",
                        parent=self.gui,
                    ))
                except Exception:
                    switch_task = False
            if not switch_task:
                self._log(
                    "Whisper turbo is transcription-only; "
                    "switch Task to Transcribe before selecting it"
                )
                try:
                    self.gui.set_status(
                        "Turbo cannot translate. Switch Task to Transcribe first.",
                        WARNING,
                    )
                except Exception:
                    pass
                try:
                    self._refresh_model_row()
                except Exception:
                    pass
                return
            switch_to_transcribe = True
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
        if switch_to_transcribe:
            if self.gui is not None:
                try:
                    self.gui.canary_task_var.set("transcribe")
                except Exception:
                    pass
            self._on_canary_task_changed("transcribe")
            if self.gui is not None:
                try:
                    self.gui.set_task_options(["transcribe"],
                                              current="transcribe")
                except Exception:
                    pass
        self.config["whisper_model"] = new_id
        save_local_config(self.config)
        try:
            self._set_universal_whisper_model(new_id, source="live")
        except Exception:
            pass
        self._log(f"Whisper model -> {display_label} ({new_id}), reloading...")
        if self.gui:
            self.gui.set_model_status(f"Switching to {display_label}...", WARNING)
            try:
                self.gui.record_btn.configure(state="disabled")
            except Exception:
                pass
        def _whisper_switched(success, err):

            self._model_switching = False
            def _ui():
                try:
                    self.gui.record_btn.configure(state="normal")
                except Exception:
                    pass
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
            self._prepare_heavy_load(except_tabs=("live",))
        except Exception:
            pass
        try:
            self._model_switching = True
            eng.switch_model(new_id, on_ready=_whisper_switched)
        except AttributeError:
            self._model_switching = False
            _whisper_switched(False, "engine unavailable")
        except Exception as e:
            self._model_switching = False
            try:
                if self.gui:
                    self.gui.record_btn.configure(state="normal")
                    self.gui.set_status(f"Model error: {e}", DANGER)
            except Exception:
                pass
    def _on_qwen_model_changed(self, display_label: str):
        """Live tab model pick while Qwen3-ASR is the active engine."""
        try:
            from qwen_engine import QWEN_MODEL_CHOICES
        except Exception:
            return
        new_id = QWEN_MODEL_CHOICES.get(display_label)
        if not new_id:
            return
        old_id = self.config.get("qwen_model", "qwen3-asr-1.7b-q8_0")
        if new_id == old_id:
            return
        if self._srt_busy:
            self._log("Stop the SRT/burn job before switching Qwen3-ASR model")
            try:
                self._refresh_model_row()
            except Exception:
                pass
            return
        try:
            if not self._live_engine_pick("Qwen3-ASR", None, new_id):
                return
        except Exception:
            pass
        self.config["qwen_model"] = new_id
        save_local_config(self.config)
        self._log(f"Qwen3-ASR model -> {display_label} ({new_id}), reloading...")
        if self.gui:
            self.gui.set_model_status(f"Switching to {display_label}...",
                                      WARNING)
            try:
                self.gui.record_btn.configure(state="disabled")
            except Exception:
                pass
        def _qwen_switched(success, err):

            self._model_switching = False
            def _ui():
                try:
                    self.gui.record_btn.configure(state="normal")
                except Exception:
                    pass
                if success:
                    self.gui.set_status(f"Ready \u2022 Qwen3-ASR {new_id}",
                                        SUCCESS)
                else:
                    self.gui.set_status(f"Model error: {err}", DANGER)
            if self.gui:
                try:
                    self.gui.after(0, _ui)
                except Exception:
                    pass
        eng = self._get_qwen_engine()
        try:
            self._prepare_heavy_load(except_tabs=("live",))
        except Exception:
            pass
        try:
            self._model_switching = True
            eng.switch_model(new_id, on_ready=_qwen_switched)
        except AttributeError:
            self._model_switching = False
            _qwen_switched(False, "engine unavailable")
        except Exception as e:
            self._model_switching = False
            try:
                if self.gui:
                    self.gui.record_btn.configure(state="normal")
                    self.gui.set_status(f"Model error: {e}", DANGER)
            except Exception:
                pass
    def _ask_dual(self, message):
        """GUI-thread Yes/No dialog for dual-engine loads. False on any
        failure or headless run (safe direction: don't load)."""
        if self.gui is None:
            return False
        try:
            from tkinter import messagebox
            return bool(messagebox.askyesno(
                "Load second engine?",
                str(message or "Another session is active."),
                parent=self.gui,
            ))
        except Exception:
            import traceback
            traceback.print_exc()
            return False

    def _live_engine_pick(self, kind, arch=None, wmid=None):
        """Validate and confirm before changing live selections. Shared
        dual-guard for the three live menu flows (engine, model, whisper
        size). Runs plan -> optional dialog -> apply -> follower menu
        sync. Returns True when the caller should proceed with its
        existing load flow; False means reverted (menus restored).
        GUI thread only. Fail closed if validation or application fails."""
        def revert():
            try:
                self._sync_live_menus()
            except Exception:
                pass
            return False

        try:
            plan = self.plan_tab_change("live", kind=kind, arch=arch,
                                        wmid=wmid)
            action = (plan or {}).get("action")
            if action == "revert":
                return revert()
            if action not in ("ok", "confirm"):
                try:
                    self._log(f"Unexpected engine-change verdict: {plan!r}")
                except Exception:
                    pass
                return revert()
            if plan.get("noop"):
                return True
            dual_ok = False
            if action == "confirm":
                if not self._ask_dual(plan.get("message")):

                    return revert()
                dual_ok = True
            result = self.apply_tab_change("live", plan["kind"],
                                           plan["arch"], plan["wmodel"],
                                           dual_ok=dual_ok)
            if (result or {}).get("action") != "ok":
                try:
                    self._log("Engine change rejected: "
                              + str((result or {}).get("reason",
                                                       "unknown reason")))
                except Exception:
                    pass
                return revert()
            try:
                self._sync_mirrored_menus((result or {}).get("mirrored"))
            except Exception:
                pass
            return True
        except Exception:
            import traceback
            traceback.print_exc()
            return revert()

    def _on_engine_changed(self, display_label: str):
        old = self.config.get("engine", "Moonshine v2")
        if display_label == old:
            return




        try:
            _old_live_obj = self.engine
            _old_kind = str(old)
        except Exception:
            _old_live_obj, _old_kind = None, ""

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
                    from gui import SRT_LANG_NAMES as _SLN
                    self.gui.canary_task_var.set(self.config.get("whisper_task", "translate"))
                    self.gui.canary_lang_var.set(
                        _SLN.get(self.config.get("whisper_src_lang", "auto"), "Auto-detect"))
                elif display_label == "Canary-1B":
                    from gui import SRT_LANG_NAMES as _SLN2
                    self.gui.canary_task_var.set(self.config.get("canary_task", "transcribe"))
                    self.gui.canary_lang_var.set(
                        _SLN2.get(self.config.get("canary_src_lang", "auto"), "Auto-detect"))
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



                    self.gui._refresh_srt_lang_state()
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
            try:
                self._prepare_heavy_load(except_tabs=("live",))
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
            try:
                self._prepare_heavy_load(except_tabs=("live",))
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
        elif display_label == "Qwen3-ASR":
            new_engine = self._get_qwen_engine()
            self.engine = new_engine
            if self.gui:
                self.gui.set_status("Loading Qwen3-ASR...", WARNING)
                try:
                    self.gui.record_btn.configure(state="disabled")
                except Exception:
                    pass
            try:
                self._prepare_heavy_load(except_tabs=("live",))
            except Exception:
                pass
            if not new_engine.is_ready:
                new_engine.load()
            else:
                if self.gui:
                    try:
                        self.gui.after(0, lambda e=new_engine: self.gui.set_status(f"Ready \u2022 Qwen3-ASR ({e.model_id})", SUCCESS))
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
        try:



            _new_live_obj = self.engine
            if _old_live_obj is not None and _old_live_obj is not _new_live_obj \
                    and _old_kind in ("Canary-1B", "Whisper", "Qwen3-ASR"):
                try:
                    if bool(getattr(_old_live_obj, "_loading", False)):
                        try:
                            _old_live_obj.unload()
                        except Exception:
                            pass
                        try:
                            self._log(f"Cancelled { _old_kind} load (switched to {display_label})")
                        except Exception:
                            pass
                except Exception:
                    pass
        except Exception:
            pass
        self._unload_idle_engines()
    def _unload_idle_engines(self):
        """Release whichever heavy engine is NOT active (multi-GB RAM back).
        Backs off while any tab session runs - including Import/Note file
        jobs that may be mid-transcribe on the shared Moonshine object -
        and never unloads an engine with a load in flight."""
        try:
            self._unload_idle_tab_engines()
        except Exception:
            pass
        try:
            try:
                if self.tab_active("live") or self.tab_active("srt") \
                        or self.tab_active("note") or self.tab_active("import"):
                    return
            except Exception:
                pass
            try:
                if bool(getattr(self, "_model_switching", False)):
                    return
            except Exception:
                pass
            active = self.config.get("engine", "Moonshine v2")
            for name in ("canary_engine", "whisper_engine", "qwen_engine",
                         "moonshine_engine"):
                try:
                    eng = getattr(self, name, None)
                    if eng is None or eng is self.engine:
                        continue
                    if active == "Canary-1B" and name == "canary_engine":
                        continue
                    if active == "Whisper" and name == "whisper_engine":
                        continue
                    if active == "Qwen3-ASR" and name == "qwen_engine":
                        continue
                    if active == "Moonshine v2" and name == "moonshine_engine":
                        continue
                    try:
                        if bool(getattr(eng, "_loading", False)):
                            continue
                    except Exception:
                        pass
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
            if (value == "translate"
                    and self.config.get("whisper_model") == "large-v3-turbo"):
                self._log(
                    "Whisper turbo cannot translate; "
                    "switch the model to large-v3 first"
                )
                try:
                    self.gui.set_status(
                        "Turbo cannot translate. Switch model to large-v3 first.",
                        WARNING,
                    )
                    self.gui.canary_task_var.set("transcribe")
                except Exception:
                    pass
                return
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

        from gui import SRT_LANG_CODE_FROM_DISPLAY, SRT_LANGS
        code = SRT_LANG_CODE_FROM_DISPLAY.get(value, value)
        if self.config.get("engine") == "Qwen3-ASR":
            try:
                from ui.choices import QWEN_LANG_NAMES
                code = {v: k for k, v in QWEN_LANG_NAMES.items()}.get(
                    value, code)
            except Exception:
                pass
            try:
                from qwen_engine import QWEN_LANGUAGES
                _ql = tuple(QWEN_LANGUAGES)
            except Exception:
                _ql = ("auto",)
            if code not in _ql:
                code = "auto"
            self.config["qwen_src_lang"] = code
            save_local_config(self.config)
            if self.qwen_engine:
                try:
                    self.qwen_engine.switch_options(source_lang=code)
                except Exception:
                    pass
            self._log(f"Qwen3-ASR src -> {code}")
            return
        if self.config.get("engine") == "Whisper":
            if code not in SRT_LANGS:
                code = "auto"
            self.config["whisper_src_lang"] = code
            save_local_config(self.config)
            if self.whisper_engine:
                try:
                    self.whisper_engine.switch_options(source_lang=code)
                except Exception:
                    pass
            self._log(f"Whisper src -> {code}")
            return
        if code not in ("auto", "en", "de", "es", "fr"):
            code = "auto"
        self.config["canary_src_lang"] = code
        save_local_config(self.config)
        if self.canary_engine:
            try:
                self.canary_engine.switch_options(source_lang=code)
            except Exception:
                pass
        self._log(f"Canary src -> {code}")
    def _on_note_lang_changed(self, code):
        """Note tab language: spoken language for Whisper/Qwen ("auto" =
        detect). Per-call, so the shared Whisper instance keeps its Live
        options."""
        code = str(code or "auto").strip().lower()
        try:
            _kind = self.tab_selection("note").get("kind", "Moonshine v2")
        except Exception:
            _kind = "Moonshine v2"
        if _kind == "Qwen3-ASR":
            try:
                from qwen_engine import QWEN_LANGUAGES
                langs = tuple(QWEN_LANGUAGES)
            except Exception:
                langs = ("auto",)
        else:
            try:
                from whisper_engine import WHISPER_SOURCE_LANGS
                langs = tuple(WHISPER_SOURCE_LANGS)
            except Exception:
                langs = ("auto", "en", "ja", "zh", "ko", "de", "fr", "es")
        if code not in langs:
            code = "auto"
        try:
            self.config["note_whisper_lang"] = code
            save_local_config(self.config)
        except Exception:
            pass
        try:
            self._log(f"Note language -> {code}")
        except Exception:
            pass

    def _on_note_source_changed(self, mic_id, desktop):
        """Note tab audio source: a soundcard mic endpoint id ("" = system
        default, "__none__" = no mic) plus the desktop-loopback flag."""
        try:
            self.config["note_mic_id"] = str(mic_id or "")
            self.config["note_desktop_audio"] = bool(desktop)
            save_local_config(self.config)
        except Exception:
            pass
        try:
            _who = "default mic" if not mic_id else str(mic_id)
            if mic_id == "__none__":
                _who = "no mic"
            _tail = " + desktop audio" if desktop else ""
            self._log(f"Note audio source -> {_who}{_tail}")
        except Exception:
            pass

    def _on_input_device_changed(self, index):
        """Live tab mic picker: persist the choice and re-point the recorder
        (takes effect on the next hold-to-record, never mid-capture)."""
        try:
            idx = int(index)
        except Exception:
            idx = -1
        name = ""
        if idx >= 0:
            try:
                name = str(AudioRecorder.list_devices()[idx]["name"]).strip()
            except Exception:
                name = ""
        self.config["input_device"] = idx
        self.config["input_device_name"] = name
        try:
            save_local_config(self.config)
        except Exception:
            pass
        try:
            self.recorder.set_device(idx, name or None)
        except Exception:
            pass
        shown = name or ("system default" if idx < 0 else f"device {idx}")
        try:
            self._log(f"Input device -> {shown}")
        except Exception:
            pass

    def _refresh_input_devices(self, refresh: bool = False):
        """Rebuild the Live mic picker (hot-plugged mics, unplugged ones).

        The saved endpoint is tracked by *name*, because Windows hands a
        replugged USB mic a new index. While it is away the picker falls
        back to the system default so recording keeps working, and the name
        is kept so the same mic is re-selected automatically on replug.
        """
        try:
            devices = AudioRecorder.list_input_devices(refresh=refresh)
        except Exception as _e:
            devices = []
        by_index = {}
        for d in devices:
            try:
                by_index[int(d.get("index", -1))] = str(
                    d.get("name") or "").strip()
            except Exception:
                continue
        try:
            old = int(self.config.get("input_device", -1))
        except Exception:
            old = -1
        want_name = str(self.config.get("input_device_name", "") or "").strip()
        # One function decides, so the rule can be tested without a GUI (see
        # recorder.choose_input): saved mic by name, then a plugged-in USB
        # mic, then the system default (the laptop's own microphone).
        try:
            from recorder import choose_input
            want, chosen_name = choose_input(
                devices, old, want_name,
                auto=bool(self.config.get("auto_input", True)))
        except Exception:
            want, chosen_name = old, want_name
        if chosen_name:
            want_name = str(chosen_name).strip()
        if want != old:
            self.config["input_device"] = want
            if want_name:
                self.config["input_device_name"] = want_name
            try:
                save_local_config(self.config)
            except Exception:
                pass
            try:
                if want >= 0:
                    self._log(f"Mic -> {by_index.get(want, want_name)}")
                else:
                    self._log("Mic -> system default")
            except Exception:
                pass
        try:
            self.recorder.set_device(want, want_name or None)
        except Exception:
            pass
        self._apply_input_devices(devices, want)
        return devices

    def _apply_input_devices(self, devices, want):
        """Push the device list into the pickers.

        Tk widgets may only be touched from the thread that created them,
        and this runs from the hotkey thread too (after a mic error), so an
        off-thread call hands the work to the GUI queue instead.
        """
        if getattr(self, "gui", None) is None:
            return
        if threading.current_thread() is not threading.main_thread():
            try:
                self._gui_queue.put(("mic_devices_ui", (devices, want)))
            except Exception:
                pass
            return
        try:
            self.gui.set_input_devices(devices, want)
        except Exception:
            pass
        try:
            note_refresh = getattr(self.gui, "_refresh_note_sources", None)
            if callable(note_refresh):
                note_refresh()
        except Exception:
            pass

    def _watch_input_devices(self):
        """Hot-plug watcher. PortAudio caches the endpoint list at init, so
        an unplugged or replugged mic is invisible until it is re-scanned.
        Poll every few seconds and rebuild the picker when the list moves,
        so a replug works without restarting the app."""
        last = None
        while not self._mic_watch_stop.wait(5.0):
            try:
                if self.recorder.is_recording or self._recording:
                    continue
                note = getattr(self.gui, "_note_recorder", None)
                if note is not None and getattr(note, "is_recording", False):
                    continue
            except Exception:
                continue
            try:
                devices = AudioRecorder.list_input_devices(refresh=True)
            except Exception:
                continue
            try:
                sig = tuple(sorted(
                    (int(d.get("index", -1)), str(d.get("name") or ""))
                    for d in devices))
            except Exception:
                continue
            if sig == last:
                continue
            last = sig
            try:
                self._gui_queue.put(("mic_devices", devices))
            except Exception:
                pass

    def _srt_lang_codes_for_tab(self):
        """Language codes the SRT tab's own engine accepts (input side)."""
        try:
            kind = self.tab_selection("srt").get("kind", "Moonshine v2")
        except Exception:
            kind = "Moonshine v2"
        if kind == "Canary-1B":
            return ("auto", "en", "de", "es", "fr")
        if kind == "Qwen3-ASR":
            try:
                from qwen_engine import QWEN_LANGUAGES
                return tuple(QWEN_LANGUAGES)
            except Exception:
                return ("auto",)
        try:
            from ui.choices import WHISPER_SRT_IN_CODES
            return tuple(WHISPER_SRT_IN_CODES)
        except Exception:
            return ("auto",)

    def _on_srt_input_lang_changed(self, value):
        from ui.choices import lang_code_from_display
        code = lang_code_from_display(value, "auto")
        if code not in self._srt_lang_codes_for_tab():
            code = "auto"
        with _CONFIG_LOCK:
            self.config["srt_input_lang"] = code
            save_local_config(self.config)
        self._log(f"SRT input lang -> {code}")
    def _on_srt_output_lang_changed(self, value):
        from ui.choices import lang_code_from_display
        code = lang_code_from_display(value, "en")
        try:
            if self.tab_selection("srt").get("kind") == "Qwen3-ASR":

                code = str(self.config.get("srt_input_lang", "auto")
                           or "auto")
        except Exception:
            pass
        if code not in self._srt_lang_codes_for_tab():
            code = "en"
        with _CONFIG_LOCK:
            self.config["srt_output_lang"] = code
            save_local_config(self.config)
        self._log(f"SRT output lang -> {code}")
    def _on_windowless_changed(self, value: bool):
        with _CONFIG_LOCK:
            self.config["windowless"] = bool(value)
            save_local_config(self.config)
        mode = "windowless (no console)" if value else "console window"
        self._log(f"Next launch via run.bat uses {mode} (takes effect on restart)")

    def _log(self, msg, color=None):
        print(f"[MoonshineSTT] {msg}")
        try:
            from core.applog import log as _filelog
            _filelog(msg)
        except Exception:
            pass
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



        for _ in range(200):
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
                elif msg_type == "srt_live":
                    try:
                        self.gui.set_srt_heartbeat(payload)
                    except Exception:
                        pass
                elif msg_type == "srt_done":
                    self.gui.srt_done(payload[0], payload[1])
                    try:
                        try:
                            full = bool(payload[2]) if len(payload) > 2 else False
                        except Exception:
                            full = False
                        self._on_job_finished(payload[0], payload[1], full)
                    except Exception:
                        pass
                elif msg_type == "srt_file_status":
                    try:
                        self.gui.set_srt_file_status(payload[0], payload[1])
                    except Exception:
                        pass
                elif msg_type == "mic_devices":
                    try:
                        self._refresh_input_devices()
                    except Exception:
                        pass
                elif msg_type == "mic_devices_ui":
                    try:
                        self._apply_input_devices(payload[0], payload[1])
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
                eng = self.engine
                if not getattr(eng, "is_ready", False):




                    try:
                        loading = bool(getattr(eng, "_loading", False))
                    except Exception:
                        loading = False
                    if not loading:
                        try:
                            eng.load()
                        except Exception:
                            pass
                        self._gui_queue.put(("status", ("Loading model — hold F2 again when ready", WARNING)))
                    else:
                        self._gui_queue.put(("status", ("Model not ready yet", WARNING)))
                    return
            except Exception:
                pass
            try:
                self.recorder.start()
            except Exception as e:
                raw = str(e)
                print(f"Mic error: {raw}")
                low = raw.lower()
                if ("9999" in raw or "unanticipated" in low
                        or "2005401480" in raw or "invalidated" in low):
                    msg = ("Mic error: device was reset - replug it or "
                           "pick another mic")
                else:
                    msg = f"Mic error: {raw}"
                if self.gui:
                    self._gui_queue.put(("status", (msg, DANGER)))
                    self.gui.after(0, lambda: self.gui.set_recording_state(False))
                self._recording = False
                try:
                    self._refresh_input_devices(refresh=True)
                except Exception:
                    pass
                return
            self._recording = True
            try:
                self._rec_start_t = time.monotonic()
            except Exception:
                pass
            try:
                self._rec_session = int(getattr(self, "_rec_session", 0)) + 1
                session = self._rec_session
            except Exception:
                session = 0
            try:
                threading.Thread(target=self._watch_silent_stream,
                                 args=(session,), daemon=True).start()
            except Exception:
                pass
            if self.gui:
                self.gui.after(0, lambda: self.gui.set_recording_state(True))
                self.gui.after(0, lambda: self.gui.set_status("Recording... F2 held", DANGER))
    def _reopen_mic(self, why: str) -> bool:
        """Drop and re-open the capture stream, reporting the outcome."""
        try:
            ok = bool(self.recorder.reopen())
        except Exception:
            ok = False
        try:
            self._log(f"Mic {why} - reopen {'ok' if ok else 'failed'}")
        except Exception:
            pass
        return ok

    def _watch_silent_stream(self, session: int = 0):
        """Watch the live capture while it runs.

        Three failure modes: an endpoint that opens but never delivers a
        block (dead device, or another app holding it exclusively), an
        endpoint that is unplugged mid-session, and a stream that sleeps
        after minutes of silence (USB selective suspend). None of them may
        end the capture: the audio already collected still gets
        transcribed, and once the mic is back -- replugged, or re-enumerated
        after Windows settles -- capture resumes on its own instead of
        leaving the app "recording" nothing until a restart.
        """
        started = time.monotonic()
        last_try = 0.0
        backoff = 1.0
        warned = False

        def alive() -> bool:
            try:
                return (bool(self._recording)
                        and int(getattr(self, "_rec_session", 0)) == session)
            except Exception:
                return False

        def status(msg: str, colour=DANGER):
            try:
                self._gui_queue.put(("status", (msg, colour)))
            except Exception:
                pass

        while alive():
            try:
                time.sleep(1.0)
            except Exception:
                return
            if not alive():
                return
            try:
                rec = self.recorder
                frames = len(getattr(rec, "_frames", None) or [])
                stalled = float(rec.stalled_for())
                silent = float(rec.silent_for())
                has_stream = bool(getattr(rec, "has_stream", True))
            except Exception:
                return
            elapsed = time.monotonic() - started
            if has_stream and frames > 0 and stalled < 6.0 and silent < 30.0:
                warned = False
                continue
            if has_stream and frames == 0 and elapsed < 2.5:
                continue
            now = time.monotonic()
            if now - last_try < backoff:
                continue
            last_try = now
            if frames == 0:
                why = "delivers no audio"
                if not warned:
                    warned = True
                    status("Mic open but silent - pick another mic or replug it")
            elif stalled >= 6.0:
                why = "stopped delivering audio"
            else:
                why = "went silent"
            if self._reopen_mic(why):
                warned = False
                if frames > 0:
                    started = time.monotonic()
                    backoff = 1.0
                    status("Mic re-opened", WARNING)
                else:
                    # The endpoint opens but still delivers nothing (muted,
                    # held exclusively, or a dead USB endpoint). Keep trying,
                    # but stop hammering it every second.
                    backoff = min(5.0, backoff + 1.0)
                    status("Mic open but silent - replug it or pick another mic")
            else:
                backoff = min(5.0, backoff + 1.0)
                status("Microphone disconnected - reconnecting when it returns")

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



            try:
                saved_top = self._target_top_hwnd
                saved_child = self._target_child_hwnd
                snap_engine = self.engine
                snap_engine_name = self.config.get("engine", "Moonshine v2")
                try:
                    if snap_engine_name == "Whisper":
                        snap_task = self.config.get("whisper_task",
                                                    "translate")
                        snap_lang = self.config.get("whisper_src_lang",
                                                    "auto")
                    else:
                        snap_task, snap_lang = None, None
                except Exception:
                    snap_task, snap_lang = None, None
                snap_suffix = self.config.get("suffix", "none")
                snap_itn = bool(self.config.get("itn_live", True))
                snap_itn_hint = bool(self.config.get("itn_hint", True))
                snap_method = self.config.get("typing_method", "clipboard")
                snap_delay = int(self.config.get("typing_delay_ms", 0))
                snap_root = self._our_root_hwnd
            except Exception:
                saved_top = saved_child = None
                snap_engine, snap_engine_name = None, "Moonshine v2"
                snap_task, snap_lang = None, None
                snap_suffix, snap_method, snap_delay = "none", "clipboard", 0
                snap_itn, snap_itn_hint = True, True
                snap_root = 0
        if self.gui:
            self.gui.after(0, lambda: self.gui.set_recording_state(False))
            self.gui.after(0, lambda: self._update_indicator())
        settings_snapshot = {
            "target_top": saved_top,
            "target_child": saved_child,
            "suffix": snap_suffix,
            "typing_method": snap_method,
            "typing_delay_ms": snap_delay,
            "itn": snap_itn,
            "itn_hint": snap_itn_hint,
            "our_root_hwnd": snap_root,
            "engine": snap_engine,
            "engine_name": snap_engine_name,
            "task": snap_task,
            "source_lang": snap_lang,
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
        try:
            if settings.get("engine_name") == "Whisper":
                text = engine.transcribe(
                    audio, SAMPLE_RATE,
                    task=settings.get("task"),
                    source_lang=settings.get("source_lang"))
            elif settings.get("engine_name") == "Qwen3-ASR":
                hint = None
                if settings.get("itn_hint", True):
                    try:
                        import itn as _itn
                        hint = _itn.QWEN_HINT
                    except Exception:
                        hint = None
                text = engine.transcribe(audio, SAMPLE_RATE,
                                         extra_prompt=hint)
            else:
                text = engine.transcribe(audio, SAMPLE_RATE)
        except TypeError:
            text = engine.transcribe(audio, SAMPLE_RATE)
        clean = text.strip() if text else ""
        if clean and not clean.startswith("["):
            if settings.get("itn", True):
                try:
                    import itn as _itn
                    clean = _itn.normalize(clean)
                except Exception as _itn_err:
                    print(f"ITN failed: {_itn_err}")

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
                self._gui_queue.put(("status", ("Pasted ✓", SUCCESS)))
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
            try:
                _st = self.tab_selection("srt")
            except Exception:
                _st = {"kind": "Moonshine v2", "arch": 5, "wmodel": "large-v3"}
            _srt_kind = _st.get("kind", "Moonshine v2")
            with _CONFIG_LOCK:
                if srt_in in _SRT_LANGS:
                    self.config["srt_input_lang"] = srt_in
                if srt_out in _SRT_LANGS:
                    self.config["srt_output_lang"] = srt_out
                if srt_task and _srt_kind == "Whisper":
                    self.config["whisper_task"] = srt_task
                elif srt_task and _srt_kind == "Canary-1B":
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
            job = {
                "engine_kind": _st["kind"],
                "moonshine_arch": int(_st["arch"]),
                "canary_task": (
                    srt_task if srt_task and _srt_kind == "Canary-1B"
                    else self.config.get("canary_task", "transcribe")),
                "canary_src": self.config.get("canary_src_lang", "auto"),
                "whisper_task": (
                    srt_task if srt_task and _srt_kind == "Whisper"
                    else self.config.get("whisper_task", "translate")),
                "whisper_src": self.config.get("whisper_src_lang", "auto"),
                "qwen_src": self.config.get("qwen_src_lang", "auto"),
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
            if job["engine_kind"] not in ("Moonshine v2", "Canary-1B", "Whisper", "Qwen3-ASR"):
                job["engine_kind"] = "Moonshine v2"
            try:
                if _st["kind"] == "Whisper":
                    _sh = self._get_whisper_engine(False)
                    _want = str(self.config.get("whisper_model", "large-v3"))
                    if _sh is not None and not self._whisper_cache_ok(
                            _sh, _want):
                        self._prepare_heavy_load(except_tabs=("srt",))
                        _sh.switch_model(_want)
            except Exception:
                pass
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
                        live_cb=lambda m: self._gui_queue.put(("srt_live", m)),
                        get_qwen_engine=lambda: self._tab_qwen("srt"),
                        qwen_src=job.get("qwen_src", "auto"),
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
                burn_ok = True
                if (not cancelled) and job.get("burn_after"):
                    _bmsg, burn_cancelled, burn_ok = \
                        self._auto_burn_after_srt(job)
                    if _bmsg:
                        burn_msg = " | " + str(_bmsg)




                full_ok = (not cancelled and not burn_cancelled
                           and not bad and total > 0 and bool(burn_ok))
                if cancelled or burn_cancelled:
                    self._gui_queue.put(
                        ("srt_done", (False, f"Cancelled after {len(ok_paths)}/{total} files{burn_msg}", False)))
                elif not bad:
                    self._gui_queue.put(
                        ("srt_done", (True, f"Batch done: all {total} saved{burn_msg}", full_ok)))
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
    def _shutdown_pc(self) -> bool:
        try:
            import subprocess as _sp
            _r = _sp.run(["shutdown", "/s", "/f", "/t", "60"],
                         stdout=_sp.DEVNULL, stderr=_sp.DEVNULL,
                         timeout=10)
            return bool(getattr(_r, "returncode", 1) == 0)
        except Exception:
            return False
    @staticmethod
    def _abort_shutdown() -> bool:
        try:
            import subprocess as _sp
            _r = _sp.run(["shutdown", "/a"],
                         stdout=_sp.DEVNULL, stderr=_sp.DEVNULL,
                         timeout=10)
            return bool(getattr(_r, "returncode", 1) == 0)
        except Exception:
            return False
    def _on_job_finished(self, ok, msg, full_ok=False):
        """Completion gate for auto-shutdown + pop-up. full_ok must be
        passed explicitly by the producer (True only when every requested
        stage - transcription AND any requested burn - fully succeeded).
        Never infer success from message text: e.g. "Batch done: all 3
        saved | Burn failed: ..." contains "all " but must NOT shut down."""
        try:
            text = str(msg or "")
        except Exception:
            text = ""
        if not ok:
            return
        try:
            want_sd = bool(self.config.get("auto_shutdown", False))
        except Exception:
            want_sd = False
        if bool(full_ok) and want_sd:
            try:
                _sd_ok = self._shutdown_pc()
            except Exception:
                _sd_ok = False
            if _sd_ok:
                self._gui_queue.put(
                    ("srt_log", "Auto-shutdown ON: PC powers off in 60s "
                     "(run `shutdown /a` in cmd to abort)"))
            else:
                self._gui_queue.put(
                    ("srt_log", "Auto-shutdown FAILED: could not schedule "
                     "shutdown (run `shutdown /s /f /t 60` in cmd; "
                     "abort with `shutdown /a`)"))
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
                        _o, _ob = (info or {}).get("out"), 0
                        if isinstance(_o, (list, tuple)) and len(_o) >= 3:
                            _ob = _o[2]
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


                return "", False, True
            self._gui_queue.put(
                ("srt_log",
                 f"Auto-burn: {len(keep_pos)} file(s) with SRT -> MP4..."))
            self._gui_queue.put(
                ("srt_live", "Burn stage (ffmpeg) - see burn progress below"))
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
            try:
                _burn_ok = (not bool(_cancelled) and bool(_results)
                            and all(bool(ok_) for (_, ok_, _)
                                    in (_results or [])))
            except Exception:
                _burn_ok = False
            return _msg, bool(_cancelled), bool(_burn_ok)
        except Exception as e:
            import traceback
            traceback.print_exc()
            return f"auto-burn skipped: {e}", False, False
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
                try:
                    _full = (not bool(_cancelled) and bool(_results)
                             and all(bool(ok_) for (_, ok_, _)
                                     in (_results or [])))
                except Exception:
                    _full = False
                self._gui_queue.put(("srt_done", (_ok, _msg, bool(_full))))
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
                    "whisper_src": self.config.get("whisper_src_lang", "auto"),
                    "qwen_src": self.config.get("qwen_src_lang", "auto"),
                    "srt_input_lang": self.config.get("srt_input_lang", "auto"),
                    "srt_output_lang": self.config.get("srt_output_lang", "en"),
                    "cpu_workers": int(self.config.get("srt_cpu", 0) or 0),
                    "normalize_audio": bool(self.config.get("srt_norm", False)),
                }
                if snap["engine_kind"] not in ("Moonshine v2", "Canary-1B", "Whisper", "Qwen3-ASR"):
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
                        cpu_workers=snap["cpu_workers"] or srtmod.default_workers(),
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
                        get_qwen_engine=lambda: self._tab_qwen("srt"),
                        qwen_src=snap.get("qwen_src", "auto"),
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


            try:
                _is_whisper = type(engine).__name__ == "WhisperEngine"
            except Exception:
                _is_whisper = False
            if not _is_whisper:
                try:
                    _is_whisper = \
                        self.tab_selection("note").get("kind") == "Whisper"
                except Exception:
                    _is_whisper = False
            if _is_whisper:
                _nt = "transcribe"
                try:
                    _ns = str(self.config.get("note_whisper_lang", "auto")
                              or "auto")
                except Exception:
                    _ns = "auto"
            else:
                try:
                    _nt = str(self.config.get("whisper_task", "translate")
                              or "translate")
                except Exception:
                    _nt = "translate"
                try:
                    _ns = str(self.config.get("whisper_src_lang", "auto")
                              or "auto")
                except Exception:
                    _ns = "auto"


            try:
                if (type(engine).__name__ == "QwenEngine"
                        or self.tab_selection("note").get("kind") == "Qwen3-ASR"):
                    _nt = "transcribe"
                    _ns = str(self.config.get("note_whisper_lang", "auto")
                              or "auto")
            except Exception:
                pass
            try:
                return engine.transcribe(audio, sample_rate,
                                         task=_nt, source_lang=_ns)
            except TypeError:
                return engine.transcribe(audio, sample_rate)
        except Exception as e:
            return f"[Error: {e}]"




    def import_file_request(self):
        """GUI-thread pre-flight for Import Transcribe. Returns
        {"go": True} | {"confirm": msg} | {"wait": msg} | {"abort": True}.
        Refuses while Note mic records or another file job runs."""
        try:
            try:
                if self.gui is not None and bool(
                        getattr(self.gui, "_note_recording", False)):
                    return {"abort": True}
            except Exception:
                pass
            if bool(getattr(self, "_import_file_busy", False)
                    or getattr(self, "_note_file_busy", False)):
                return {"abort": True}
            try:
                if self.gui is not None and bool(
                        getattr(self.gui, "_import_file_running", False)
                        or getattr(self.gui, "_note_file_running", False)) \
                        and not bool(getattr(self, "_import_file_busy", False)
                                     or getattr(self, "_note_file_busy", False)):

                    pass
            except Exception:
                pass
            sel = self.tab_selection("import")
            kind = sel["kind"]

            try:
                if kind == "Moonshine v2":
                    eng = self.moonshine_engine
                    if eng is not None and eng.is_ready:
                        return {"go": True}
                else:
                    cache = (self._tab_cache or {}).get("import") or {}
                    slot = self._engine_slot(kind)



                    if slot in ("whisper", "qwen"):
                        eng = (self._get_whisper_engine(False)
                               if slot == "whisper" else cache.get("qwen"))
                    else:
                        eng = cache.get(slot)
                    ok = False
                    try:
                        if eng is not None and eng.is_ready:
                            if slot in ("whisper", "qwen"):
                                ok = self._whisper_cache_ok(eng, sel["wmodel"])
                            else:
                                ok = True
                    except Exception:
                        ok = False
                    if ok:
                        return {"go": True}
            except Exception:
                pass

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
                try:
                    ram = self._engine_ram_mb(kind, sel["arch"], sel["wmodel"])
                except Exception:
                    ram = 3000
                what = ("Canary-1B" if kind == "Canary-1B"
                        else (f"Qwen3-ASR {sel['wmodel']}"
                              if kind == "Qwen3-ASR"
                              else (f"Whisper {sel['wmodel']}" if kind == "Whisper"
                                    else "Moonshine (shared)")))
                return {"confirm": f"{'; '.join(others)} still active.\n"
                                    f"Transcribe this file with {what} too?\n"
                                    f"Yes = run both engines (~{ram / 1000:.1f} GB "
                                    f"extra RAM). No = stay as you are.",
                        "ram_mb": ram}


            try:
                if kind == "Moonshine v2":
                    try:
                        if (self.moonshine_engine is not None
                                and not self.moonshine_engine.is_ready
                                and not getattr(self.moonshine_engine,
                                                "_loading", False)):
                            self.moonshine_engine.load()
                    except Exception:
                        pass
                else:
                    if kind == "Whisper":
                        try:
                            _sh = self._get_whisper_engine(False)
                            _want = str(self.config.get(
                                "whisper_model",
                                sel.get("wmodel", "large-v3")))
                            if _sh is not None and not self._whisper_cache_ok(
                                    _sh, _want):
                                self._prepare_heavy_load(
                                    except_tabs=("import",))
                                _sh.switch_model(_want)
                            elif _sh is not None and not _sh.is_ready:
                                _sh.load()
                        except Exception:
                            pass
                    else:
                        fresh = self._tab_heavy("import", kind, sel["wmodel"])
                        if fresh is not None and not fresh.is_ready:
                            fresh.load()
            except Exception:
                pass
            return {"wait": "Loading import engine - press Transcribe File again when ready."}
        except Exception:
            return {"abort": True}

    def note_file_request(self):
        """Legacy alias (Note file UI moved to the Import tab)."""
        try:
            return self.import_file_request()
        except Exception:
            return {"abort": True}

    def _import_file_cancel(self):
        """Cancel the running Import file job (thread-safe, never raises)."""
        try:
            self._import_cancel_event.set()
        except Exception:
            pass
        try:
            self._note_cancel_event.set()
        except Exception:
            pass
        try:
            if self.gui is not None:
                try:
                    self.gui.after(
                        0, lambda: self.gui.set_import_file_progress(
                            0, "Cancelling..."))
                except Exception:
                    pass
        except Exception:
            pass

    def _note_file_cancel(self):
        """Legacy alias."""
        try:
            return self._import_file_cancel()
        except Exception:
            pass

    def _import_file_start(self, path: str):
        """Background entry for Import file transcription (GUI spawns a thread
        around this; this method blocks until done). Never raises."""
        acquired = False
        try:
            with self._import_file_lock:
                if self._import_file_busy or self._note_file_busy:
                    return
                self._import_file_busy = True
                self._note_file_busy = True
                acquired = True
                self._import_cancel_event.clear()
                self._note_cancel_event.clear()
            self._import_file_work(str(path or ""))
        except Exception as e:
            try:
                if self.gui is not None:
                    self.gui.after(
                        0, lambda m=str(e): self.gui.import_file_done(
                            False, f"File error: {m}"))
            except Exception:
                pass
        finally:
            if acquired:
                with self._import_file_lock:
                    self._import_file_busy = False
                    self._note_file_busy = False

    def _note_file_start(self, path: str):
        """Legacy alias."""
        try:
            return self._import_file_start(path)
        except Exception:
            pass

    def _import_file_prog(self, frac: float, msg: str = ""):
        try:
            g = self.gui
            if g is None:
                return
            try:
                g.after(0, lambda f=frac, m=msg: g.set_import_file_progress(f, m))
            except Exception:
                pass
        except Exception:
            pass

    def _note_file_prog(self, frac: float, msg: str = ""):
        """Legacy alias."""
        try:
            return self._import_file_prog(frac, msg)
        except Exception:
            pass

    def _import_file_resolve_engine(self, sel, cancel_event, prog):
        """Return (engine, kind) for the Import tab selection, waiting for the
        load (with progress + cancel). Raises RuntimeError on failure."""
        import time as _t
        kind = sel["kind"]
        if kind == "Moonshine v2":
            eng = self.moonshine_engine
            if eng is None:
                raise RuntimeError("Moonshine engine unavailable")
            if not eng.is_ready:
                try:
                    if not getattr(eng, "_loading", False):
                        eng.load()
                except Exception:
                    pass
                waited = 0
                while not eng.is_ready:
                    if cancel_event.is_set():
                        raise InterruptedError("cancelled")
                    if not getattr(eng, "_loading", True):
                        err = getattr(eng, "_last_error", None) or "load failed"
                        raise RuntimeError(f"Moonshine failed to load: {err}")
                    if waited % 5 == 0:
                        prog(0.05, f"Loading Moonshine ({waited}s)...")
                    _t.sleep(0.5)
                    waited += 1
                    if waited > 240:
                        raise RuntimeError("Moonshine load timed out")
            return eng, kind

        eng = self._tab_heavy("import", kind, sel.get("wmodel"))
        if eng is None:
            raise RuntimeError(f"{kind} engine unavailable")
        if kind in ("Whisper", "Qwen3-ASR"):
            want = sel.get("wmodel")
            try:
                got = str(getattr(eng, "model_id", "") or "")
                if want and got != str(want):
                    eng.switch_model(want)
            except Exception:
                pass
        if not eng.is_ready:
            try:
                if not getattr(eng, "_loading", False):
                    eng.load()
            except Exception:
                pass
        waited = 0
        if kind == "Canary-1B":
            label = "Canary-1B"
        elif kind == "Qwen3-ASR":
            label = f"Qwen3-ASR {sel.get('wmodel', '')}"
        else:
            label = f"Whisper {sel.get('wmodel', '')}"
        while not eng.is_ready:
            if cancel_event.is_set():
                raise InterruptedError("cancelled")
            if not getattr(eng, "_loading", True):
                err = getattr(eng, "_last_error", None) or "load failed"
                raise RuntimeError(f"{label} failed to load: {err}")
            if waited % 5 == 0:
                prog(0.05, f"Loading {label} ({waited}s)...")
            _t.sleep(0.5)
            waited += 1
            if waited > 1200:
                raise RuntimeError(f"{label} load timed out")
        if kind in ("Whisper", "Qwen3-ASR"):
            try:
                if not self._whisper_cache_ok(eng, sel.get("wmodel")):
                    raise RuntimeError(f"{label} model mismatch - retry")
            except RuntimeError:
                raise
            except Exception:
                pass
        return eng, kind

    def _import_file_work(self, path: str):
        """Extract -> transcribe (chunked) -> sentences -> append. Runs off
        the GUI thread; all Tk touches go via after()."""
        import os as _os
        from pathlib import Path as _P
        cancel_event = self._import_cancel_event
        prog = self._import_file_prog
        gui = self.gui

        def _done(ok, msg):
            try:
                if gui is not None:
                    gui.after(0, lambda: gui.import_file_done(ok, msg))
            except Exception:
                pass

        try:
            if not path or not _os.path.exists(path):
                _done(False, "File not found - pick it again")
                return
            try:
                from srt import SUPPORTED_EXTS
                if _P(path).suffix.lower() not in SUPPORTED_EXTS:
                    _done(False,
                          f"Unsupported type {_P(path).suffix} "
                          f"({', '.join(SUPPORTED_EXTS)})")
                    return
            except Exception:
                pass
            if cancel_event.is_set():
                _done(False, "Cancelled")
                return
            sel = self.tab_selection("import")
            prog(0.03, "Resolving import engine...")
            try:
                eng, kind = self._import_file_resolve_engine(sel, cancel_event,
                                                             prog)
            except InterruptedError:
                _done(False, "Cancelled")
                return
            except Exception as e:
                _done(False, f"Engine error: {e}")
                return


            try:
                import srt as _srtmod
                ffmpeg = _srtmod.get_ffmpeg_exe()
            except Exception:
                ffmpeg = None
            if not ffmpeg:
                _done(False, "ffmpeg not found - run setup.bat once")
                return
            src = _P(path)


            try:
                import tempfile as _tf2
                _tmpd = _P(_tf2.gettempdir())
                _tmpd.mkdir(parents=True, exist_ok=True)
                _fd, _tmp = _tf2.mkstemp(prefix=src.stem + ".import_",
                                         suffix="_16k.wav",
                                         dir=str(_tmpd))
                try:
                    _os.close(_fd)
                except Exception:
                    pass
                tmp_wav = _P(_tmp)
            except Exception:
                try:
                    _tmpd = _P(_os.environ.get("TEMP") or _os.getcwd())
                except Exception:
                    _tmpd = src.parent
                tmp_wav = _tmpd / (src.stem + ".import_tmp16k.wav")
            try:
                if tmp_wav.resolve() == src.resolve():
                    tmp_wav = tmp_wav.with_name(src.stem + "_import16k.wav")
            except Exception:
                pass
            try:
                prog(0.08, "Extracting audio (ffmpeg 16kHz mono)...")
                _srtmod.extract_audio(src, tmp_wav, ffmpeg, cancel_event,
                                      normalize_audio=False)
            except InterruptedError:
                try:
                    tmp_wav.unlink(missing_ok=True)
                except Exception:
                    pass
                _done(False, "Cancelled")
                return
            except Exception as e:
                try:
                    tmp_wav.unlink(missing_ok=True)
                except Exception:
                    pass
                _done(False, f"Audio extract failed: {e}")
                return
            try:
                if cancel_event.is_set():
                    _done(False, "Cancelled")
                    return
                texts = self._import_file_transcribe(
                    eng, kind, sel, tmp_wav, cancel_event, prog)
            finally:
                try:
                    tmp_wav.unlink(missing_ok=True)
                except Exception:
                    pass
            if cancel_event.is_set():
                _done(False, "Cancelled")
                return
            raw = " ".join(t for t in (texts or []) if t and t.strip()).strip()
            if not raw:
                _done(False, "No speech detected in file")
                return
            try:
                from services.note_format import (format_note_sentences as _fmt,
                                                 format_note_file_text as _head)
            except Exception:
                try:
                    from note_engine import (format_note_sentences as _fmt,
                                             format_note_file_text as _head)
                except Exception:
                    _fmt = lambda t: (t or "").strip()  # noqa: E731
                    _head = lambda n, e, b: (b or "").strip()  # noqa: E731
            try:
                engine_label = self._describe_tab_engine("import")
            except Exception:
                engine_label = str(kind)
            sentences = _fmt(raw)
            if not sentences:
                sentences = raw.strip()
            try:
                formatted = _head(src.name, engine_label, sentences)
            except Exception:
                formatted = sentences
            try:
                if gui is not None:
                    gui.after(
                        0, lambda f=formatted: gui.append_import_file_text(f))
            except Exception:
                pass

            try:
                n_sent = len([l for l in sentences.split("\n") if l.strip()])
            except Exception:
                n_sent = 0
            _done(True,
                  f"File done: {src.name} ({n_sent} sentence"
                  f"{'' if n_sent == 1 else 's'}) - review, Save TXT or Copy")
        except InterruptedError:
            _done(False, "Cancelled")
        except Exception as e:
            import traceback
            traceback.print_exc()
            _done(False, f"File error: {e}")

    def _note_file_resolve_engine(self, sel, cancel_event, prog):
        """Legacy alias."""
        try:
            return self._import_file_resolve_engine(sel, cancel_event, prog)
        except Exception as e:
            raise e

    def _note_file_work(self, path: str):
        """Legacy alias."""
        try:
            return self._import_file_work(path)
        except Exception:
            pass

    def _import_file_transcribe(self, eng, kind, sel, tmp_wav, cancel_event,
                                prog):
        """Chunked file transcription -> list of plain texts. Raises on
        cancel; engine failure strings become RuntimeError (never sentences)."""
        import os as _os
        import tempfile as _tf
        texts = []
        first_error = None

        def _is_err(t):
            try:
                s = str(t or "").strip()
                return (not s) or s.startswith(
                    ("[Error:", "[Whisper Error:", "[Canary Error:",
                     "[Qwen Error:"))
            except Exception:
                return True


        if kind == "Whisper":
            try:
                eff_src = "auto"
                try:


                    eff_src = str(self.config.get("whisper_src_lang",
                                                  "auto") or "auto")
                except Exception:
                    eff_src = "auto"
                try:
                    eff_task = str(self.config.get("whisper_task",
                                                   "translate") or "translate")
                except Exception:
                    eff_task = "translate"
                prog(0.15, "Whisper transcribing (single pass)...")
                fn = getattr(eng, "transcribe_file_segments", None)
                if cancel_event.is_set():
                    raise InterruptedError("cancelled")
                if callable(fn):
                    native = fn(str(tmp_wav), task=eff_task,
                                source_lang=eff_src)
                else:
                    native = []
                joined = " ".join(
                    str(t or "").strip() for _, _, t in (native or [])
                    if str(t or "").strip())
                if joined and not _is_err(joined):
                    return [joined]
                if _is_err(joined) and joined:
                    first_error = joined.strip()

            except InterruptedError:
                raise
            except Exception as e:
                import traceback
                traceback.print_exc()
                first_error = first_error or f"[Whisper Error: {e}]"

        try:
            import srt as _srtmod
            import soundfile as _sf
            audio, sr = _srtmod.load_wav_16k(tmp_wav)
            spans = _srtmod.vad_segments(audio, sr)
            if not spans:
                if first_error:
                    raise RuntimeError(first_error)
                return []
            n = max(1, len(spans))
            for i, (s, e) in enumerate(spans):
                if cancel_event.is_set():
                    raise InterruptedError("cancelled")
                base = 0.15 + 0.75 * i / n
                prog(base, f"Transcribing {i + 1}/{n} ({s:.0f}s)...")
                s_i, e_i = int(s * sr), int(e * sr)
                chunk = audio[s_i:e_i]
                text = ""
                try:
                    if kind == "Canary-1B":

                        with _tf.NamedTemporaryFile(suffix=".wav",
                                                     delete=False) as tf:
                            tmp_c = tf.name
                        try:
                            _sf.write(tmp_c, chunk, samplerate=sr)
                            text = eng.transcribe_file(tmp_c)
                        finally:
                            try:
                                _os.unlink(tmp_c)
                            except Exception:
                                pass
                    else:



                        if kind == "Qwen3-ASR":
                            _ct = "transcribe"
                            try:
                                _cs = str(self.config.get("qwen_src_lang",
                                                          "auto") or "auto")
                            except Exception:
                                _cs = "auto"
                        else:
                            try:
                                _ct = str(self.config.get("whisper_task",
                                                          "translate")
                                          or "translate")
                            except Exception:
                                _ct = "translate"
                            try:
                                _cs = str(self.config.get("whisper_src_lang",
                                                          "auto") or "auto")
                            except Exception:
                                _cs = "auto"



                            if _ct == "translate":
                                try:
                                    import capabilities as _cap
                                    _mid = str(self.config.get(
                                        "whisper_model", "large-v3"))
                                    if not _cap.model_supports(
                                            "Whisper", _mid, _cap.TRANSLATE):
                                        _ct = "transcribe"
                                        self._log(
                                            f"Whisper {_mid} cannot translate; "
                                            "import uses transcribe")
                                except Exception:
                                    pass
                        try:
                            text = eng.transcribe(chunk, sr, task=_ct,
                                                  source_lang=_cs)
                        except TypeError:
                            text = eng.transcribe(chunk, sr)
                except Exception as ex:
                    text = f"[Error: {ex}]"
                t = str(text or "").strip()
                if t and not _is_err(t):
                    texts.append(t)
                elif t and first_error is None:
                    first_error = t
                prog(0.15 + 0.75 * (i + 1) / n,
                     f"Chunk {i + 1}/{n} done")
            if texts:
                return texts
            if first_error:
                raise RuntimeError(first_error)
            return []
        except InterruptedError:
            raise
        except RuntimeError:
            raise
        except Exception as e:
            import traceback
            traceback.print_exc()
            raise RuntimeError(f"transcription failed: {e}")

    def _note_file_transcribe(self, eng, kind, sel, tmp_wav, cancel_event,
                              prog):
        """Legacy alias."""
        try:
            return self._import_file_transcribe(eng, kind, sel, tmp_wav,
                                               cancel_event, prog)
        except Exception as e:
            raise e

    def _on_close(self):
        try:
            if self.gui and not self.gui.confirm_note_processing():
                return
        except Exception:
            pass
        try:
            if self.gui and not self.gui.confirm_note_close():
                return
        except Exception:
            pass
        try:
            if self.gui and not self.gui.confirm_import_close():
                return
        except Exception:
            pass
        try:
            if bool(getattr(self, "_srt_busy", False)):
                from tkinter import messagebox as _mb
                if not _mb.askyesno(
                        "SRT job running",
                        "An SRT job is still running.\n\n"
                        "Quit and cancel it?", parent=self.gui):
                    return
        except Exception:
            pass
        try:
            self._import_cancel_event.set()
        except Exception:
            pass
        try:
            self._note_cancel_event.set()
        except Exception:
            pass
        try:
            self._cancel_srt_job()
        except Exception:
            pass
        try:


            _deadline = time.monotonic() + 3.0
            while (time.monotonic() < _deadline
                   and bool(getattr(self, "_srt_busy", False))):
                time.sleep(0.05)
        except Exception:
            pass
        try:


            if self.gui:
                self.gui.cleanup_note_audio()
        except Exception:
            pass
        if self._recording:
            try:
                self.recorder.stop()
            except Exception:
                pass
        try:
            # Stop the hot-plug watcher before the window goes away, so it
            # cannot touch a destroyed Tk widget or restart PortAudio while
            # the process is shutting down.
            self._mic_watch_stop.set()
        except Exception:
            pass
        if self.gui:
            self.gui.destroy()
        try:
            from core.applog import flush as _logflush
            _logflush()
        except Exception:
            pass
        os._exit(0)
def _single_instance_guard():
    """One app per user session. Two copies would both install the F2 hook
    and fight over the microphone. Best-effort: never blocks startup."""
    global _SINGLE_INSTANCE_MUTEX
    try:
        if os.name != "nt":
            return
        import ctypes
        from ctypes import wintypes
        k32 = ctypes.WinDLL("kernel32", use_last_error=True)
        k32.CreateMutexW.argtypes = (wintypes.LPVOID, wintypes.BOOL,
                                     wintypes.LPCWSTR)
        k32.CreateMutexW.restype = wintypes.HANDLE
        h = k32.CreateMutexW(None, False,
                             "Local\\PolyglotSTT_SingleInstance")
        _err = ctypes.get_last_error()
        if not h:
            return
        _SINGLE_INSTANCE_MUTEX = h
        if _err == 183:
            try:
                ctypes.windll.user32.MessageBoxW(
                    None, "PolyglotSTT is already running.",
                    "PolyglotSTT", 0x40)
            except Exception:
                pass
            os._exit(0)
    except Exception:
        pass
def _want_windowless_launch():
    """True when this launch should not keep a console window.

    Purely informational here: the app itself never hides or detaches a
    console (EDR products flag that as defense evasion). Windowless launches
    are done by running the app under pythonw.exe, which has no console at
    all - see run.bat / the desktop shortcut.
    """
    if "--windowless" in sys.argv:
        return True
    try:
        exe = os.path.basename(sys.executable or "").lower()
        if exe == "pythonw.exe":
            return True
    except Exception:
        pass
    try:
        if os.environ.get("POLYGLOTSTT_WINDOWLESS", "").strip() in ("1", "true", "yes"):
            return True
    except Exception:
        pass
    try:
        with open(CONFIG_PATH, "r", encoding="utf-8") as f:
            return bool(json.load(f).get("windowless", False))
    except Exception:
        return False


def main():
    try:
        from core.applog import setup_logging
        setup_logging()
    except Exception:
        pass
    try:
        _single_instance_guard()
    except Exception:
        pass
    app = MoonshineSTTApp()
    app.start()
if __name__ == "__main__":
    main()