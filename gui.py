import pathlib
# gui.py
import customtkinter as ctk
import threading
import time
import math
from pathlib import Path
from typing import Optional, Callable, Tuple
from ui.tabs.live import LiveTabMixin
from ui.tabs.note import NoteTabMixin
from ui.tabs.import_tab import ImportTabMixin
from ui.tabs.srt import SrtTabMixin
from ui.dialogs import DialogsMixin
from ui.widgets import LevelMeter, WaveformDisplay, StatusBadge, HistoryPanel
from ui.choices import (MODEL_CHOICES, MODEL_CHOICES_REV,
                         WHISPER_MODEL_CHOICES, WHISPER_MODEL_CHOICES_REV,
                         CANARY_MODEL_LABEL, ENGINE_CHOICES, CANARY_TASKS,
                         CANARY_LANGS, WHISPER_TASKS, WHISPER_LANGS, SRT_LANGS,
                         SRT_LANG_NAMES, SRT_LANG_DISPLAY,
                         SRT_LANG_CODE_FROM_DISPLAY, CANARY_LIVE_SRC_CODES,
                         CANARY_SRT_IN_CODES, CANARY_SRT_OUT_CODES,
                         WHISPER_SRT_IN_CODES, WHISPER_SRT_OUT_CODES,
                         BURN_SPEED_HELP, BURN_COMPARE_ROWS,
                         BURN_COMPARE_CODEC_NOTE, apply_badge_map)
ACCENT = "#6C5CE7"
ACCENT_DARK = "#5A4BD1"
ACCENT_GLOW = "#A29BFE"
SUCCESS = "#00B894"
WARNING = "#FDCB6E"
DANGER = "#E17055"
BG_DARK = "#1A1A2E"
BG_CARD = "#16213E"
BG_INPUT = "#0F3460"
FG_PRIMARY = "#FFFFFF"
FG_SECONDARY = "#B2BEC3"
FG_DIM = "#636E72"
BTN_DIM = "#2D3748"
BTN_DIM_HOVER = "#4A5568"
BTN_GO_HOVER = "#00916E"
BTN_DANGER_HOVER = "#C0392B"
BTN_BURN = "#B5651D"
BTN_BURN_HOVER = "#8E4E15"
BTN_TEXT = "#FEFEFE"
THEMES = {
    "dark": {
        "ACCENT": "#6C5CE7", "ACCENT_DARK": "#5A4BD1",
        "ACCENT_GLOW": "#A29BFE", "SUCCESS": "#00B894",
        "WARNING": "#FDCB6E", "DANGER": "#E17055",
        "BG_DARK": "#1A1A2E", "BG_CARD": "#16213E", "BG_INPUT": "#0F3460",
        "FG_PRIMARY": "#FFFFFF", "FG_SECONDARY": "#B2BEC3",
        "FG_DIM": "#636E72",
        "BTN_DIM": "#2D3748", "BTN_DIM_HOVER": "#4A5568",
        "BTN_GO_HOVER": "#00916E", "BTN_DANGER_HOVER": "#C0392B",
        "BTN_BURN": "#B5651D", "BTN_BURN_HOVER": "#8E4E15",
        "BTN_TEXT": "#FEFEFE",
        "SEG_SELECTED": "#6C5CE7", "SEG_SELECTED_HOVER": "#5A4BD1",
    },
    "light": {
        "ACCENT": "#5A48D6", "ACCENT_DARK": "#4A38B8",
        "ACCENT_GLOW": "#5E35B1", "SUCCESS": "#00755C",
        "WARNING": "#8A5A00", "DANGER": "#C0392B",
        "BG_DARK": "#E9EDF2", "BG_CARD": "#FFFFFF", "BG_INPUT": "#DCE2EA",
        "FG_PRIMARY": "#16181D", "FG_SECONDARY": "#3E4450",
        "FG_DIM": "#687182",
        "BTN_DIM": "#CBD2DC", "BTN_DIM_HOVER": "#B6BECB",
        "BTN_GO_HOVER": "#006A51", "BTN_DANGER_HOVER": "#A93226",
        "BTN_BURN": "#A05A18", "BTN_BURN_HOVER": "#7E4A12",
        "BTN_TEXT": "#FEFEFE",
        "SEG_SELECTED": "#D9D2FB", "SEG_SELECTED_HOVER": "#C4B8F5",
    },
}
SEG_SELECTED = THEMES["dark"]["SEG_SELECTED"]
SEG_SELECTED_HOVER = THEMES["dark"]["SEG_SELECTED_HOVER"]
THEME_MODE = "dark"
# SEG_* are excluded from the value maps: their dark values intentionally
# match other roles (selected strip == ACCENT), and value-based mapping
# cannot tell roles apart. The strip is configured explicitly instead.
_SKIP_REMAP = ("SEG_SELECTED", "SEG_SELECTED_HOVER")
_DARK_TO_LIGHT = {v.upper(): THEMES["light"][k]
                  for k, v in THEMES["dark"].items()
                  if k not in _SKIP_REMAP}
_LIGHT_TO_DARK = {v.upper(): THEMES["dark"][k]
                  for k, v in THEMES["light"].items()
                  if k not in _SKIP_REMAP}
def theme_color(value):
    try:
        v = str(value or "")
        if not v:
            return value
        m = _DARK_TO_LIGHT if THEME_MODE == "light" else _LIGHT_TO_DARK
        return m.get(v.upper(), value)
    except Exception:
        return value
class MoonshineGUI(LiveTabMixin, NoteTabMixin, ImportTabMixin, SrtTabMixin, DialogsMixin,
                        ctk.CTk):
    def __init__(self):
        super().__init__()
        ctk.set_appearance_mode("dark")
        ctk.set_default_color_theme("dark-blue")
        self.title("MoonshineSTT")
        self.geometry("480x850")
        self.minsize(440, 750)
        self.configure(fg_color=BG_DARK)
        self.resizable(True, True)
        self._on_record_start: Optional[Callable] = None
        self._on_record_stop: Optional[Callable] = None
        self._is_recording = False
        self._theme_callback = None
        self._theme_mode = "dark"
        self._build_ui()
    def _build_ui(self):
        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(2, weight=1)
        header = ctk.CTkFrame(self, fg_color="transparent")
        header.grid(row=0, column=0, sticky="ew", padx=20, pady=(16, 0))
        header.grid_columnconfigure(0, weight=1)
        header.grid_columnconfigure(1, weight=0)
        title_frame = ctk.CTkFrame(header, fg_color="transparent")
        title_frame.grid(row=0, column=0, sticky="w")
        ctk.CTkLabel(title_frame, text="MoonshineSTT",
                     font=("Segoe UI", 22, "bold"),
                     text_color=FG_PRIMARY).pack(side="left")
        ctk.CTkLabel(title_frame, text=" v2",
                     font=("Segoe UI", 14), text_color=ACCENT_GLOW).pack(side="left", pady=(6, 0))
        self.status_badge = StatusBadge(header)
        self.status_badge.grid(row=0, column=1, sticky="e")
        self.status_badge.set_status("Initializing", WARNING)
        header.grid_columnconfigure(2, weight=0)
        self._theme_btn = ctk.CTkButton(
            header, text="☀", width=40, height=28, font=("Segoe UI", 13),
            fg_color="transparent", hover_color=BG_CARD, text_color=FG_DIM,
            corner_radius=8, command=self._on_theme_toggled)
        self._theme_btn.grid(row=0, column=2, sticky="e", padx=(8, 0))
        subtitle = ctk.CTkLabel(self, text="Hold F2 to record \u2022 Release to transcribe",
                                font=("Segoe UI", 11), text_color=FG_DIM)
        subtitle.grid(row=1, column=0, padx=20, pady=(6, 8), sticky="w")
        self.tabs = ctk.CTkTabview(self, fg_color="transparent",
                                   segmented_button_fg_color=BG_CARD,
                                   segmented_button_selected_color=ACCENT,
                                   segmented_button_unselected_color=BG_CARD,
                                   text_color=FG_PRIMARY,
                                   command=self._on_tab_changed)
        self._tab_callback = None
        self.tabs.grid(row=2, column=0, sticky="nsew", padx=8, pady=(0, 4))
        live = self.tabs.add("Live")
        srt_tab = self.tabs.add("SRT File")
        note_tab = self.tabs.add("Note")
        import_tab = self.tabs.add("Import")
        live.grid_columnconfigure(0, weight=1)
        live.grid_rowconfigure(5, weight=1)
        srt_tab.grid_columnconfigure(0, weight=1)
        srt_tab.grid_rowconfigure(0, weight=1)
        note_tab.grid_columnconfigure(0, weight=1)
        note_tab.grid_rowconfigure(0, weight=1)
        import_tab.grid_columnconfigure(0, weight=1)
        import_tab.grid_rowconfigure(0, weight=1)
        try:
            # Palette-named colors so the theme walker remaps them.
            # fg follows the cards (no dark surround survives) and the
            # border is off entirely ("transparent" fg is rejected here,
            # BG_CARD is the opaque equivalent).
            _sb = self.tabs._segmented_button
            _sb.configure(fg_color=BG_CARD, border_width=0,
                          selected_color=SEG_SELECTED,
                          selected_hover_color=SEG_SELECTED_HOVER,
                          unselected_color=BG_CARD,
                          unselected_hover_color=BG_INPUT,
                          text_color=FG_PRIMARY)
        except Exception:
            pass
        record_card = ctk.CTkFrame(live, fg_color=BG_CARD, corner_radius=16)
        record_card.grid(row=2, column=0, sticky="ew", padx=8, pady=(0, 6))
        record_card.grid_columnconfigure(0, weight=1)
        self.meter = LevelMeter(record_card, width=420, height=10)
        self.meter.grid(row=0, column=0, padx=20, pady=(12, 4), sticky="ew")
        self.waveform = WaveformDisplay(record_card, width=420, height=70)
        self.waveform.grid(row=1, column=0, padx=16, pady=(8, 4), sticky="ew")
        btn_frame = ctk.CTkFrame(record_card, fg_color="transparent")
        btn_frame.grid(row=2, column=0, pady=(8, 12), sticky="ew")
        btn_frame.grid_columnconfigure(0, weight=1)
        btn_frame.grid_columnconfigure(1, weight=1)
        btn_frame.grid_columnconfigure(2, weight=1)
        self.record_btn = ctk.CTkButton(
            btn_frame, text="\u25CF  RECORD", font=("Segoe UI", 13, "bold"),
            fg_color=ACCENT, hover_color=ACCENT_DARK, height=40,
            text_color=BTN_TEXT,
            corner_radius=10, command=self._toggle_record)
        self.record_btn.grid(row=0, column=0, padx=(16, 4), sticky="ew")
        self.clear_btn = ctk.CTkButton(
            btn_frame, text="Clear", font=("Segoe UI", 12),
            fg_color=BTN_DIM, hover_color=BTN_DIM_HOVER, height=40,
            corner_radius=10, text_color=FG_SECONDARY,
            command=self._clear_history)
        self.clear_btn.grid(row=0, column=1, padx=4, sticky="ew")
        self.copy_last_btn = ctk.CTkButton(
            btn_frame, text="Copy", font=("Segoe UI", 12),
            fg_color=BTN_DIM, hover_color=BTN_DIM_HOVER, height=40,
            corner_radius=10, text_color=FG_SECONDARY,
            command=self._copy_last)
        self.copy_last_btn.grid(row=0, column=2, padx=(4, 16), sticky="ew")
        output_card = ctk.CTkFrame(live, fg_color=BG_CARD, corner_radius=16)
        output_card.grid(row=3, column=0, sticky="ew", padx=8, pady=(0, 6))
        output_card.grid_columnconfigure(0, weight=1)
        ctk.CTkLabel(output_card, text="Last Transcription",
                     font=("Segoe UI", 11, "bold"), text_color=FG_DIM
                     ).pack(anchor="w", padx=16, pady=(12, 2))
        self.output_text = ctk.CTkTextbox(
            output_card, font=("Consolas", 13), fg_color=BG_INPUT,
            text_color=FG_PRIMARY, corner_radius=8, height=60,
            activate_scrollbars=False, wrap="word")
        self.output_text.pack(fill="x", padx=12, pady=(0, 12))
        self.output_text.insert("1.0", "Waiting for voice input...")
        self.output_text.configure(state="disabled")
        history_label_frame = ctk.CTkFrame(live, fg_color="transparent")
        history_label_frame.grid(row=4, column=0, sticky="ew", padx=12, pady=(8, 4))
        ctk.CTkLabel(history_label_frame, text="Transcription History",
                     font=("Segoe UI", 11, "bold"), text_color=FG_DIM
                     ).pack(side="left")
        self.history = HistoryPanel(live, height=140)
        self.history.grid(row=5, column=0, sticky="nsew", padx=8, pady=(0, 6))
        engine_frame = ctk.CTkFrame(live, fg_color=BG_CARD, corner_radius=10)
        engine_frame.grid(row=0, column=0, sticky="ew", padx=8, pady=(0, 6))
        ctk.CTkLabel(engine_frame, text="Engine:", font=("Segoe UI", 10, "bold"), text_color=FG_DIM).pack(side="left", padx=(12, 4), pady=8)
        self.engine_var = ctk.StringVar(value="Moonshine v2")
        self.engine_menu = ctk.CTkOptionMenu(engine_frame, variable=self.engine_var, values=ENGINE_CHOICES, width=150, fg_color=BG_INPUT, button_color=ACCENT, command=self._on_engine_changed)
        self.engine_menu.pack(side="left", padx=4, pady=8)
        ctk.CTkLabel(engine_frame, text="Task:", font=("Segoe UI", 10), text_color=FG_DIM).pack(side="left", padx=(12, 4), pady=8)
        self.canary_task_var = ctk.StringVar(value="transcribe")
        self.canary_task_menu = ctk.CTkOptionMenu(engine_frame, variable=self.canary_task_var, values=CANARY_TASKS, width=110, fg_color=BG_INPUT, button_color=ACCENT, command=self._on_canary_task_changed)
        self.canary_task_menu.pack(side="left", padx=4, pady=8)
        ctk.CTkLabel(engine_frame, text="Src:", font=("Segoe UI", 10), text_color=FG_DIM).pack(side="left", padx=(8, 4), pady=8)
        self.canary_lang_var = ctk.StringVar(value="Auto-detect")
        self.canary_lang_menu = ctk.CTkOptionMenu(engine_frame, variable=self.canary_lang_var, values=[SRT_LANG_NAMES[c] for c in CANARY_LANGS], width=140, fg_color=BG_INPUT, button_color=ACCENT, command=self._on_canary_lang_changed)
        self.canary_lang_menu.pack(side="left", padx=4, pady=8)
        self._engine_callback = None
        self._canary_task_callback = None
        self._canary_lang_callback = None
        settings_frame = ctk.CTkFrame(live, fg_color=BG_CARD, corner_radius=10)
        settings_frame.grid(row=6, column=0, sticky="ew", padx=8, pady=(0, 6))
        ctk.CTkLabel(settings_frame, text="Insert via:", font=("Segoe UI", 10), text_color=FG_DIM).pack(side="left", padx=(12, 4), pady=8)
        self.method_var = ctk.StringVar(value="clipboard")
        self.method_menu = ctk.CTkOptionMenu(settings_frame, variable=self.method_var, values=["clipboard", "unicode"], width=110, fg_color=BG_INPUT, button_color=ACCENT, command=self._on_method_changed)
        self.method_menu.pack(side="left", padx=4, pady=8)
        ctk.CTkLabel(settings_frame, text="Suffix:", font=("Segoe UI", 10), text_color=FG_DIM).pack(side="left", padx=(12, 4), pady=8)
        self.suffix_var = ctk.StringVar(value="none")
        self.suffix_menu = ctk.CTkOptionMenu(settings_frame, variable=self.suffix_var, values=["none", "space", "newline", "period_space"], width=110, fg_color=BG_INPUT, button_color=ACCENT, command=self._on_suffix_changed)
        self.suffix_menu.pack(side="left", padx=4, pady=8)
        self._method_callback = None
        self._suffix_callback = None
        model_frame = ctk.CTkFrame(live, fg_color=BG_CARD, corner_radius=10)
        model_frame.grid(row=1, column=0, sticky="ew", padx=8, pady=(0, 6))
        self.model_title_label = ctk.CTkLabel(model_frame, text="Model:",
                                              font=("Segoe UI", 10, "bold"), text_color=FG_DIM)
        self.model_title_label.pack(side="left", padx=(12, 4), pady=8)
        self.model_var = ctk.StringVar(value="Medium Streaming (110MB, best)")
        self.model_menu = ctk.CTkOptionMenu(model_frame, variable=self.model_var, values=list(MODEL_CHOICES.keys()), width=260, fg_color=BG_INPUT, button_color=ACCENT, command=self._on_model_changed)
        self.model_menu.pack(side="left", padx=4, pady=8, fill="x", expand=True)
        self.model_manage_btn = ctk.CTkButton(
            model_frame, text="Manage…", width=80, font=("Segoe UI", 12),
            fg_color=BTN_DIM, hover_color=BTN_DIM_HOVER, height=32,
            corner_radius=8, text_color=FG_SECONDARY,
            command=self._on_model_manage)
        self.model_manage_btn.pack(side="left", padx=4, pady=8)
        ctk.CTkLabel(model_frame, text="Restart not needed", font=("Segoe UI", 9), text_color=FG_DIM).pack(side="left", padx=(8, 12), pady=8)
        self._model_callback = None
        self._model_manage_cb = None
        self._model_value_map = {}
        self.model_frame = model_frame
        self.engine_frame = engine_frame
        footer = ctk.CTkFrame(self, fg_color="transparent")
        footer.grid(row=3, column=0, sticky="ew", padx=20, pady=(0, 12))
        self._footer_label = ctk.CTkLabel(
            footer, text="Moonshine v2 + Canary-1B + Whisper \u2022 On-device \u2022 No API keys",
            font=("Segoe UI", 10), text_color=FG_DIM)
        self._footer_label.pack(side="left")
        self.windowless_var = ctk.BooleanVar(value=False)
        self.windowless_check = ctk.CTkCheckBox(
            footer, text="Windowless next launch",
            variable=self.windowless_var, font=("Segoe UI", 10),
            text_color=FG_DIM, fg_color=ACCENT,
            command=self._on_windowless_toggled)
        self.windowless_check.pack(side="right")
        self._windowless_callback = None
        self._build_srt_tab(srt_tab)
        self._build_note_tab(note_tab)
        self._build_import_tab(import_tab)
        try:
            self._fix_menu_text()
        except Exception:
            pass
        try:
            self._tune_scrollers()
        except Exception:
            pass
    def set_footer_version(self, version: str):
        try:
            base = "Moonshine v2 + Canary-1B + Whisper \u2022 On-device \u2022 No API keys"
            v = str(version or "").strip()
            self._footer_label.configure(
                text=f"{base} \u2022 v{v}" if v else base)
        except Exception:
            pass
    def set_title_version(self, version: str):
        try:
            v = str(version or "").strip()
            self.title(f"MoonshineSTT v{v}" if v else "MoonshineSTT")
        except Exception:
            pass
    def _on_windowless_toggled(self):
        try:
            value = bool(self.windowless_var.get())
        except Exception:
            value = False
        if self._windowless_callback:
            try:
                self._windowless_callback(value)
            except Exception:
                pass
    def set_windowless(self, value: bool, callback):
        self._windowless_callback = callback if callable(callback) else None
        try:
            self.windowless_var.set(bool(value))
        except Exception:
            pass


    def _on_tab_changed(self, value=None):
        try:
            name = self.tabs.get()
        except Exception:
            name = value
        if self._tab_callback and name in ("Live", "SRT File", "Note", "Import"):
            try:
                self._tab_callback(name)
            except Exception:
                pass
    def set_tab_callback(self, cb: Callable):
        self._tab_callback = cb if callable(cb) else None
    def set_active_tab(self, name: str):
        try:
            if name in ("Live", "SRT File", "Note", "Import"):
                self.tabs.set(name)
        except Exception:
            pass
    def _fix_menu_text(self):
        for _mn in ("engine_menu", "canary_task_menu", "canary_lang_menu",
                    "method_menu", "suffix_menu", "model_menu",
                    "compute_menu", "srt_input_lang_menu",
                    "srt_output_lang_menu", "sample_len_menu",
                    "burn_speed_menu", "burn_codec_menu",
                    "srt_engine_menu", "srt_model_menu",
                    "note_engine_menu", "note_model_menu",
                    "import_engine_menu", "import_model_menu"):
            try:
                _w = getattr(self, _mn, None)
                if _w is not None:
                    _w.configure(text_color=FG_PRIMARY,
                                 dropdown_text_color=FG_PRIMARY)
            except Exception:
                pass
    def set_theme_callback(self, cb: Optional[Callable]):
        self._theme_callback = cb if callable(cb) else None
    def _on_theme_toggled(self):
        try:
            cur = getattr(self, "_theme_mode", None) or THEME_MODE
            nxt = "light" if cur != "light" else "dark"
        except Exception:
            nxt = "dark"
        try:
            self.set_theme(nxt)
        except Exception:
            pass
        cb = getattr(self, "_theme_callback", None)
        if cb:
            try:
                cb(nxt)
            except Exception:
                pass
    def _tune_scrollers(self, root=None):
        """Wheel step-up for our scroll areas. CTk scrolls 20 units of 1px
        per notch on Windows (~20px - endless on a heavy tab); 12px units
        travel the same tab in ~8 notches. Re-asserted on theme swap.
        Inner text boxes/listboxes keep their own scrolling."""
        try:
            import customtkinter as _ctk
        except Exception:
            return
        try:
            base = root if root is not None else self
            stack = [base]
            while stack:
                try:
                    w = stack.pop()
                except Exception:
                    break
                try:
                    if isinstance(w, _ctk.CTkScrollableFrame):
                        try:
                            w._parent_canvas.configure(yscrollincrement=12)
                        except Exception:
                            pass
                except Exception:
                    pass
                try:
                    stack.extend(list(w.winfo_children()))
                except Exception:
                    pass
        except Exception:
            pass

    def set_theme(self, mode: str = "dark"):
        global THEME_MODE
        mode = "light" if str(mode or "").lower() == "light" else "dark"
        pal = THEMES[mode]
        try:
            old_vals = {k: globals().get(k) for k in pal}
        except Exception:
            old_vals = {}
        for k, v in pal.items():
            try:
                globals()[k] = v
            except Exception:
                pass
        try:
            # SEG_* excluded like the module maps: their dark values match
            # other roles and value-mapping cannot tell roles apart (this
            # exact omission once repainted every accent button lavender).
            forward = {str(old_vals.get(k, v)).upper(): v
                       for k, v in pal.items() if k not in _SKIP_REMAP}
        except Exception:
            forward = {}
        try:
            ctk.set_appearance_mode("Light" if mode == "light" else "Dark")
        except Exception:
            pass
        try:
            self.configure(fg_color=pal["BG_DARK"])
        except Exception:
            pass
        try:
            self._remap_theme(self, forward)
        except Exception:
            pass
        try:
            for _w in self._walk_widgets(self):
                try:
                    import customtkinter as _ctk
                    _is_scroll = isinstance(_w, _ctk.CTkScrollableFrame)
                except Exception:
                    _is_scroll = False
                if not _is_scroll:
                    continue
                try:
                    import tkinter as _tk
                    _tk.Frame.configure(_w, bg=BG_DARK)
                except Exception:
                    pass
                try:
                    _w._parent_canvas.configure(bg=BG_DARK)
                except Exception:
                    pass
        except Exception:
            pass
        try:
            _sb2 = getattr(getattr(self, "tabs", None),
                           "_segmented_button", None)
            if _sb2 is not None:
                _sb2.configure(fg_color=BG_CARD, border_width=0,
                               selected_color=SEG_SELECTED,
                               selected_hover_color=SEG_SELECTED_HOVER,
                               unselected_color=BG_CARD,
                               unselected_hover_color=BG_INPUT,
                               text_color=FG_PRIMARY)
                # Inner segment buttons own CTkCanvases invisible to the
                # walker - paint them the strip color directly.
                try:
                    import tkinter as _tk4
                    for _ch in list(_sb2.winfo_children()):
                        try:
                            _cv = getattr(_ch, "_canvas", None)
                            if _cv is not None:
                                _tk4.Canvas.configure(_cv, bg=BG_CARD)
                        except Exception:
                            pass
                except Exception:
                    pass
        except Exception:
            pass
        THEME_MODE = mode
        try:
            # v1.4.0: ui/theme.py is the shared palette home (ui/widgets.py
            # reads it dynamically). Mirror the new palette + mode there so
            # packaged widgets repaint with the same toggle. The tab mixins
            # and dialogs hold import-time palette snapshots, so push the
            # same values into those modules as well.
            import ui.theme as _pkg_theme
            try:
                for _k, _v in pal.items():
                    setattr(_pkg_theme, _k, _v)
                _pkg_theme.THEME_MODE = mode
            except Exception:
                pass
            try:
                import ui.tabs.live as _m_live
                import ui.tabs.note as _m_note
                import ui.tabs.import_tab as _m_imp
                import ui.tabs.srt as _m_srt
                import ui.dialogs as _m_dlg
                for _mod in (_m_live, _m_note, _m_imp, _m_srt, _m_dlg):
                    try:
                        for _k, _v in pal.items():
                            setattr(_mod, _k, _v)
                    except Exception:
                        pass
            except Exception:
                pass
        except Exception:
            pass
        try:
            self._theme_mode = mode
            self._theme_btn.configure(text="☾" if mode == "light" else "☀")
        except Exception:
            pass
        try:
            if getattr(self, "meter", None) is not None:
                try:
                    import tkinter as _tk2
                    _tk2.Canvas.configure(self.meter, bg=BG_CARD)
                except Exception:
                    try:
                        self.meter.configure(bg=BG_CARD)
                    except Exception:
                        pass
                self.meter.set_level(float(getattr(self.meter, "level", 0.0)))
        except Exception:
            pass
        try:
            _nm = getattr(self, "note_meter", None)
            if _nm is not None:
                try:
                    _nm.configure(bg=BG_CARD)
                except Exception:
                    pass
                _nm.set_level(float(getattr(_nm, "level", 0.0)))
        except Exception:
            pass
        try:
            _wf = getattr(self, "waveform", None)
            if _wf is not None and getattr(_wf, "canvas", None) is not None:
                try:
                    import tkinter as _tk3
                    _tk3.Canvas.configure(_wf.canvas, bg=BG_INPUT)
                except Exception:
                    try:
                        _wf.canvas.configure(bg=BG_INPUT)
                    except Exception:
                        pass
        except Exception:
            pass
        try:
            self._tune_scrollers()
        except Exception:
            pass
        return mode
    def _walk_widgets(self, widget):
        try:
            yield widget
        except Exception:
            return
        try:
            kids = list(widget.winfo_children())
        except Exception:
            return
        for ch in kids:
            try:
                for sub in self._walk_widgets(ch):
                    yield sub
            except Exception:
                continue
    def _remap_theme(self, widget, mapping):
        try:
            kids = list(widget.winfo_children())
        except Exception:
            kids = []
        for ch in kids:
            try:
                self._remap_theme(ch, mapping)
            except Exception:
                pass
        if not mapping:
            return
        for opt in ("fg_color", "bg_color", "bg", "fg", "text_color",
                    "button_color", "button_hover_color", "hover_color",
                    "border_color", "progress_color",
                    "scrollbar_button_color", "scrollbar_button_hover_color",
                    "selectbackground", "selectforeground",
                    "dropdown_fg_color", "dropdown_hover_color",
                    "dropdown_text_color", "segmented_button_fg_color",
                    "segmented_button_selected_color",
                    "segmented_button_unselected_color",
                    "checkmark_color"):
            try:
                cur = widget.cget(opt)
            except Exception:
                continue
            try:
                key = str(cur).upper()
            except Exception:
                continue
            if key in mapping:
                # CTkCanvas (button art, meters, waveform) goes straight to
                # the tkinter level: its CTk configure() accepts bg without
                # applying it, so the normal path would silently no-op.
                try:
                    import tkinter as _tk0
                    if type(widget).__name__ == "CTkCanvas" and opt in (
                            "bg", "background"):
                        _tk0.Canvas.configure(widget, bg=mapping[key])
                        continue
                except Exception:
                    pass
                try:
                    widget.configure(**{opt: mapping[key]})
                except Exception:
                    try:
                        import tkinter as _tk
                        if type(widget).__name__ == "CTkCanvas" and opt in (
                                "bg", "background"):
                            _tk.Canvas.configure(widget, bg=mapping[key])
                    except Exception:
                        pass


    def _srt_ids_from_menus(self):
        """(kind, arch|None, wmid|None) from current menu values."""
        kind = self.get_srt_engine_kind()
        arch, wmid = None, None
        try:
            label = self.srt_model_var.get()
        except Exception:
            label = ""
        try:
            if kind == "Whisper":
                wmid = WHISPER_MODEL_CHOICES.get(label)
            elif kind == "Moonshine v2":
                arch = MODEL_CHOICES.get(label)
        except Exception:
            pass
        return kind, arch, wmid

    def _revert_srt_menus(self):
        """Restore menus from the app registry (after a declined dialog)."""
        try:
            get_cb = getattr(self, "_tab_get_cb", None)
            if get_cb is None:
                return
            sel = get_cb("srt") or {}
            self.set_srt_engine_state(sel.get("kind"), sel.get("arch"),
                                      sel.get("wmodel"))
        except Exception:
            pass

    def set_tab_engine_callbacks(self, plan_cb, apply_cb, get_cb):
        """App plan/apply/getter for per-tab engine picks (SRT + Note +
        Import menus share this one flow). All None-safe."""
        try:
            self._tab_plan_cb = plan_cb if callable(plan_cb) else None
            self._tab_apply_cb = apply_cb if callable(apply_cb) else None
            self._tab_get_cb = get_cb if callable(get_cb) else None
        except Exception:
            pass

    def _translate_engine_pick(self, tab, kind_display=None,
                               model_display=None):
        """Menu display values -> (kind, arch|None, wmid|None) ids."""
        try:
            if tab == "srt":
                kvar, mvar = self.srt_engine_var, self.srt_model_var
            elif tab == "note":
                kvar, mvar = self.note_engine_var, self.note_model_var
            elif tab == "import":
                kvar, mvar = self.import_engine_var, self.import_model_var
            else:
                return "Moonshine v2", None, None
            try:
                kind = (kind_display if kind_display is not None
                        else kvar.get() or "Moonshine v2").strip()
            except Exception:
                kind = "Moonshine v2"
            if kind not in ENGINE_CHOICES:
                kind = "Moonshine v2"
            try:
                label = (model_display if model_display is not None
                         else mvar.get() or "")
            except Exception:
                label = ""
            arch, wmid = None, None
            try:
                if kind == "Whisper":
                    wmid = WHISPER_MODEL_CHOICES.get(label)
                elif kind == "Moonshine v2":
                    arch = MODEL_CHOICES.get(label)
            except Exception:
                pass
            return kind, arch, wmid
        except Exception:
            return "Moonshine v2", None, None

    def _sync_tab_menus(self, tab, kind, arch=None, wmid=None):
        """Paint one tab's menus from ids (mirror/restore path)."""
        try:
            if tab == "srt":
                self.set_srt_engine_state(kind, arch, wmid)
            elif tab == "note":
                self.set_note_engine_state(kind, arch, wmid)
            elif tab == "import":
                self.set_import_engine_state(kind, arch, wmid)
        except Exception:
            pass

    def _revert_tab_menus(self, tab):
        try:
            get_cb = getattr(self, "_tab_get_cb", None)
            if get_cb is None:
                return
            sel = get_cb(tab) or {}
            self._sync_tab_menus(tab, sel.get("kind"), sel.get("arch"),
                                 sel.get("wmodel"))
        except Exception:
            pass

    def _request_tab_engine(self, tab, kind_display=None,
                            model_display=None):
        """Apply menu changes only after validation and any confirmation."""
        try:
            plan_cb = getattr(self, "_tab_plan_cb", None)
            apply_cb = getattr(self, "_tab_apply_cb", None)
            if not callable(plan_cb) or not callable(apply_cb):
                self._revert_tab_menus(tab)
                return

            kind, arch, wmid = self._translate_engine_pick(
                tab, kind_display, model_display)
            verdict = plan_cb(tab, kind, arch, wmid) or {}
            action = verdict.get("action")
            if action not in ("ok", "confirm"):
                self._revert_tab_menus(tab)
                return

            dual_ok = False
            if action == "confirm":
                from tkinter import messagebox
                dual_ok = bool(messagebox.askyesno(
                    "Load second engine?",
                    str(verdict.get("message") or
                        "Another session is active."),
                    parent=self))
                if not dual_ok:
                    self._revert_tab_menus(tab)
                    return

            result = apply_cb(
                tab, verdict["kind"], verdict["arch"],
                verdict["wmodel"], dual_ok) or {}
            if result.get("action") != "ok":
                self._revert_tab_menus(tab)
                return

            self._revert_tab_menus(tab)  # Read back committed selection.
            for other, kind, arch, wmid in result.get("mirrored") or []:
                if other != tab:
                    self._sync_tab_menus(other, kind, arch, wmid)
        except Exception:
            import logging
            logging.getLogger(__name__).exception("Engine selection failed")
            self._revert_tab_menus(tab)


    def import_session_active(self):
        """True while an Import file job runs. Never raises."""
        try:
            return bool(getattr(self, "_import_file_running", False))
        except Exception:
            return False

    def note_session_active(self):
        """True while Note records or transcribes mic chunks."""
        try:
            if bool(getattr(self, "_note_recording", False)):
                return True
            if bool(getattr(self, "_import_file_running", False)):
                return True
            if bool(getattr(self, "_note_file_running", False)):
                return True
            sub = int(getattr(self, "_note_submitted", 0) or 0)
            done = int(getattr(self, "_note_done", 0) or 0)
            return sub > done
        except Exception:
            return False

    def _enable_drop_to(self, widget, handler) -> bool:
        """Register a drop target with an explicit handler (Note vs SRT).
        Never raises; returns True when tkinterdnd2 accepted the widget."""
        try:
            from tkinterdnd2 import DND_FILES
            import tkinterdnd2.TkinterDnD as _dndmod
            _dndmod._require(widget)
            _dndmod.DnDWrapper.drop_target_register(widget, DND_FILES)
            _dndmod.DnDWrapper.dnd_bind(widget, "<<Drop>>", handler)
            return True
        except Exception:
            return False

    def _enable_drop(self, widget) -> bool:
        try:
            return self._enable_drop_to(widget, self._on_drop_files)
        except Exception:
            return False
    def _parse_drop(self, data: str):
        import tkinter
        try:
            return list(self.tk.splitlist(data)) if data else []
        except tkinter.TclError:
            return []


