import pathlib
# gui.py
import customtkinter as ctk
import threading
import time
import math
from pathlib import Path
from typing import Optional, Callable, Tuple
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
MODEL_CHOICES = {
    "Tiny (26MB, fastest)": 0,
    "Tiny Streaming (45MB, light)": 2,
    "Base (60MB, balanced)": 1,
    "Base Streaming (60MB, light)": 3,
    "Small Streaming (80MB, good)": 4,
    "Medium Streaming (110MB, best)": 5,
}
MODEL_CHOICES_REV = {v: k for k, v in MODEL_CHOICES.items()}
try:
    from whisper_engine import (WHISPER_MODEL_CHOICES as _WMC,
                                WHISPER_MODEL_CHOICES_REV as _WMCR)
    WHISPER_MODEL_CHOICES = dict(_WMC)
    WHISPER_MODEL_CHOICES_REV = dict(_WMCR)
except Exception:
    WHISPER_MODEL_CHOICES = {
        "Tiny (75MB, fastest)": "tiny",
        "Base (145MB)": "base",
        "Small (500MB)": "small",
        "Medium (1.5GB)": "medium",
        "Large v1 (3GB, oldest)": "large-v1",
        "Large v2 (3GB)": "large-v2",
        "Large v3 (3GB, best)": "large-v3",
    }
    WHISPER_MODEL_CHOICES_REV = {v: k for k, v in WHISPER_MODEL_CHOICES.items()}
CANARY_MODEL_LABEL = "Canary-1B (3.9GB, fixed)"
BURN_SPEED_HELP = {
    "Match size (2-pass x264)": "Exact size (±1–3%). Slowest — encodes the video twice.",
    "Fast (1-pass x264)": "About half the time, size within ~±10%.",
    "Fastest (ultrafast 1-pass)": "Several times faster, visibly softer. For quick checks, not keeps.",
    "Draft (NVENC fast 1-pass)": "Needs NVIDIA GPU. Fastest encode, size approximate (~±10%).",
    "Turbo (NVENC ultra-fast 1-pass)": "Needs NVIDIA GPU. Lowest-latency tune, a touch softer than Draft.",
    "Balanced (NVENC 2-pass)": "Needs NVIDIA GPU. GPU two-pass, closer size (~±5–10%).",
}
BURN_COMPARE_ROWS = (
    ("Match size (2-pass)", "x264 CPU", "±1–3% (exact)",
     "1× baseline (slow)", "everything incl. old devices",
     "archival keeps, exact-size needs"),
    ("Fast (1-pass)", "x264 CPU", "~±10%",
     "~2×", "everything incl. old devices",
     "quick CPU burns"),
    ("Fastest (ultrafast 1-pass)", "x264 CPU", "~±10%, visibly softer",
     "~4–6×", "everything incl. old devices",
     "drafts and checks"),
    ("Draft (NVENC 1-pass)", "NVENC GPU", "~±10% (VBR)",
     "very fast (100s of fps)", "modern devices (~2016+)",
     "fast GPU burns"),
    ("Turbo (NVENC ultra-fast)", "NVENC GPU", "~±10%, a touch softer",
     "fastest", "modern devices (~2016+)",
     "speed runs, previews"),
    ("Balanced (NVENC 2-pass)", "NVENC GPU", "~±5–10%",
     "fast", "modern devices (~2016+)",
     "best GPU quality per minute"),
)
BURN_COMPARE_CODEC_NOTE = (
    "Codec switch: HEVC (H.265) ≈30% smaller than H.264 at the same visual "
    "quality - via NVENC at the same GPU speed, or via x265 on CPU "
    "(≈30–40% smaller but much slower). Needs ~2016+ playback hardware; "
    "H.264 plays on everything. Learned estimates, the MB box, and "
    "overshoot compensation track each speed+codec separately."
)
def apply_badge_map(values, status_fn=None):
    mapping = {}
    displays = []
    for v in (values or []):
        mark = ""
        if status_fn is not None:
            try:
                mark = (status_fn(v) or "").strip()
            except Exception:
                mark = ""
        d = f"{v} {mark}" if mark else str(v)
        k, n = d, 2
        while k in mapping:
            k = f"{d} ({n})"
            n += 1
        mapping[k] = v
        displays.append(k)
    return displays, mapping
ENGINE_CHOICES = ["Moonshine v2", "Canary-1B", "Whisper"]
CANARY_TASKS = ["transcribe", "translate"]
CANARY_LANGS = ["auto", "en", "de", "es", "fr"]
WHISPER_TASKS = ["transcribe", "translate"]
WHISPER_LANGS = ["auto", "en", "ja", "zh", "ko", "de", "es", "fr",
                 "it", "pt", "nl", "ru", "ar", "hi", "tr", "id", "uk", "vi", "th"]
SRT_LANGS = ["auto", "en", "ja", "zh", "ko", "de", "es", "fr",
             "it", "pt", "nl", "ru", "ar", "hi", "tr", "id", "uk", "vi", "th"]
SRT_LANG_NAMES = {
    "auto": "Auto-detect", "en": "English", "ja": "Japanese", "zh": "Chinese",
    "ko": "Korean", "de": "German", "es": "Spanish", "fr": "French",
    "it": "Italian", "pt": "Portuguese", "nl": "Dutch", "ru": "Russian",
    "ar": "Arabic", "hi": "Hindi", "tr": "Turkish", "id": "Indonesian",
    "uk": "Ukrainian", "vi": "Vietnamese", "th": "Thai",
}
SRT_LANG_DISPLAY = [SRT_LANG_NAMES[c] for c in SRT_LANGS]
SRT_LANG_CODE_FROM_DISPLAY = {v: k for k, v in SRT_LANG_NAMES.items()}
CANARY_LIVE_SRC_CODES = ["auto", "en", "de", "es", "fr"]
CANARY_SRT_IN_CODES = ["auto", "en", "de", "es", "fr"]
CANARY_SRT_OUT_CODES = ["en", "de", "es", "fr"]
WHISPER_SRT_IN_CODES = list(SRT_LANGS)
WHISPER_SRT_OUT_CODES = [c for c in SRT_LANGS if c != "auto"]
class LevelMeter(ctk.CTkCanvas):
    def __init__(self, master, width=300, height=8, **kwargs):
        super().__init__(master, width=width, height=height,
                         highlightthickness=0, bg=BG_CARD, **kwargs)
        self.meter_width = width
        self.meter_height = height
        self.level = 0.0
        self._draw()
    def set_level(self, level: float):
        self.level = max(0.0, min(1.0, level))
        self._draw()
    def _draw(self):
        self.delete("all")
        self.create_rectangle(0, 0, self.meter_width, self.meter_height,
                              fill=BTN_DIM, outline="")
        if self.level > 0.01:
            w = self.meter_width * self.level
            if self.level < 0.6:
                color = SUCCESS
            elif self.level < 0.85:
                color = WARNING
            else:
                color = DANGER
            self.create_rectangle(0, 0, w, self.meter_height,
                                  fill=color, outline="", width=0)
class WaveformDisplay(ctk.CTkFrame):
    def __init__(self, master, width=400, height=80, **kwargs):
        super().__init__(master, width=width, height=height,
                         fg_color=BG_INPUT, corner_radius=12, **kwargs)
        self.canvas = ctk.CTkCanvas(self, width=width - 4, height=height - 4,
                                    highlightthickness=0, bg=BG_INPUT)
        self.canvas.pack(padx=2, pady=2, fill="both", expand=True)
        self.amplitude = 0.0
        self._animating = False
    def set_amplitude(self, amp: float):
        self.amplitude = max(0.0, min(1.0, amp))
    def start_animation(self):
        if self._animating:
            return
        self._animating = True
        self._animate()
    def stop_animation(self):
        self._animating = False
        self.canvas.delete("all")
        w = self.canvas.winfo_width() or 400
        h = self.canvas.winfo_height() or 80
        self.canvas.create_line(0, h // 2, w, h // 2,
                                fill=FG_DIM, width=2, dash=(4, 4))
    def _animate(self):
        if not self._animating:
            return
        self.canvas.delete("all")
        w = self.canvas.winfo_width() or 400
        h = self.canvas.winfo_height() or 80
        mid = h // 2
        t = time.time()
        points = []
        for x in range(0, w, 2):
            freq1 = 3.0
            freq2 = 7.0
            amp = self.amplitude * (mid - 4)
            y = mid + amp * math.sin(freq1 * (x / w) * 2 * math.pi + t * 4)
            y += amp * 0.5 * math.sin(freq2 * (x / w) * 2 * math.pi + t * 6)
            points.append((x, y))
        if len(points) > 1:
            flat = []
            for p in points:
                flat.extend(p)
            self.canvas.create_line(*flat, fill=ACCENT_GLOW, width=2, smooth=True)
            glow_points = []
            for p in points:
                glow_points.extend([p[0], p[1] + 2])
            self.canvas.create_line(*glow_points, fill=ACCENT, width=1, smooth=True,
                                    dash=(2, 4))
        self.after(33, self._animate)
class StatusBadge(ctk.CTkFrame):
    def __init__(self, master, **kwargs):
        super().__init__(master, fg_color="transparent", **kwargs)
        self.indicator = ctk.CTkLabel(self, text="\u25CF", font=("Segoe UI", 12),
                                      text_color=FG_DIM)
        self.indicator.pack(side="left", padx=(0, 6))
        self.label = ctk.CTkLabel(self, text="Ready", font=("Segoe UI", 13, "bold"),
                                  text_color=FG_SECONDARY)
        self.label.pack(side="left")
    def set_status(self, text: str, color: str):
        color = theme_color(color)
        self.label.configure(text=text, text_color=color)
        self.indicator.configure(text_color=color)
class HistoryPanel(ctk.CTkScrollableFrame):
    MAX_ENTRIES = 100
    def __init__(self, master, **kwargs):
        super().__init__(master, fg_color=BG_INPUT, corner_radius=12,
                         scrollbar_button_color=ACCENT, **kwargs)
        self._entries = []
    def add_entry(self, text: str, timestamp: str):
        entry_frame = ctk.CTkFrame(self, fg_color=BG_CARD, corner_radius=8)
        entry_frame.pack(fill="x", padx=4, pady=3)
        ts_label = ctk.CTkLabel(entry_frame, text=timestamp,
                                font=("Segoe UI", 10), text_color=FG_DIM)
        ts_label.pack(anchor="w", padx=8, pady=(6, 0))
        text_label = ctk.CTkLabel(entry_frame, text=text,
                                  font=("Consolas", 12), text_color=FG_PRIMARY,
                                  wraplength=360, justify="left")
        text_label.pack(anchor="w", padx=8, pady=(2, 6))
        self._entries.append(entry_frame)
        while len(self._entries) > self.MAX_ENTRIES:
            try:
                self._entries.pop(0).destroy()
            except Exception:
                break
        try:
            self._parent_canvas.yview_moveto(1.0)
        except Exception:
            try:
                self._canvas.yview_moveto(1.0)
            except Exception:
                pass
    def clear(self):
        for entry in self._entries:
            entry.destroy()
        self._entries.clear()
class MoonshineGUI(ctk.CTk):
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
        live.grid_columnconfigure(0, weight=1)
        live.grid_rowconfigure(5, weight=1)
        srt_tab.grid_columnconfigure(0, weight=1)
        srt_tab.grid_rowconfigure(0, weight=1)
        note_tab.grid_columnconfigure(0, weight=1)
        note_tab.grid_rowconfigure(0, weight=1)
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
        self.canary_lang_var = ctk.StringVar(value="auto")
        self.canary_lang_menu = ctk.CTkOptionMenu(engine_frame, variable=self.canary_lang_var, values=CANARY_LANGS, width=80, fg_color=BG_INPUT, button_color=ACCENT, command=self._on_canary_lang_changed)
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
        self._footer_label.pack()
        self._build_srt_tab(srt_tab)
        self._build_note_tab(note_tab)
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
    def _build_srt_tab(self, tab):
        import os as _os
        try:
            from srt import cpu_count, default_workers
            max_cpu = cpu_count()
            def_cpu = default_workers()
        except Exception:
            max_cpu = _os.cpu_count() or 4
            def_cpu = max(1, int(round(max_cpu * 0.8)))
        self._srt_max_cpu = max(1, int(max_cpu))
        self._srt_start_cb = None
        self._srt_cancel_cb = None
        self._srt_input_paths: list = []
        self._srt_file_status: dict = {}
        self._srt_running = False
        scroll = ctk.CTkScrollableFrame(tab, fg_color="transparent")
        scroll.pack(fill="both", expand=True, padx=4, pady=4)
        scroll.grid_columnconfigure(0, weight=1)
        eng_card = ctk.CTkFrame(scroll, fg_color=BG_CARD, corner_radius=12)
        eng_card.grid(row=0, column=0, sticky="ew", padx=4, pady=(0, 6))
        eng_card.grid_columnconfigure(1, weight=1)
        file_card = ctk.CTkFrame(scroll, fg_color=BG_CARD, corner_radius=12)
        file_card.grid(row=1, column=0, sticky="ew", padx=4, pady=(0, 6))
        file_card.grid_columnconfigure(0, weight=1)
        ctk.CTkLabel(file_card, text="Video / Audio Files  (queue - runs one by one)",
                     font=("Segoe UI", 11, "bold"), text_color=FG_DIM
                     ).pack(anchor="w", padx=12, pady=(8, 2))
        self.srt_drop = ctk.CTkTextbox(file_card, font=("Segoe UI", 11),
                                       fg_color=BG_INPUT, text_color=FG_SECONDARY,
                                       corner_radius=8, height=58,
                                       activate_scrollbars=False, wrap="word")
        self.srt_drop.pack(fill="x", padx=10, pady=(0, 6))
        self.srt_drop.insert("1.0", "Drag & drop video/audio files here,\nor click Browse Files...")
        self.srt_drop.configure(state="disabled")
        self._srt_dnd_ok = self._enable_drop(self.srt_drop)
        self._enable_drop(file_card)
        import tkinter as _tk
        _listrow = ctk.CTkFrame(file_card, fg_color="transparent")
        _listrow.pack(fill="x", padx=10, pady=(0, 6))
        _listrow.grid_columnconfigure(0, weight=1)
        self.srt_file_list = _tk.Listbox(
            _listrow, height=5, font=("Segoe UI", 10),
            bg=BG_INPUT, fg=FG_PRIMARY, selectbackground=ACCENT,
            selectforeground=BTN_TEXT, highlightthickness=0,
            relief="flat", activestyle="none")
        self.srt_file_list.grid(row=0, column=0, sticky="ew")
        try:
            _sb = _tk.Scrollbar(_listrow, orient="vertical",
                                command=self.srt_file_list.yview)
            _sb.grid(row=0, column=1, sticky="ns")
            self.srt_file_list.configure(yscrollcommand=_sb.set)
        except Exception:
            pass
        try:
            self._enable_drop(self.srt_file_list)
        except Exception:
            pass
        fbtn = ctk.CTkFrame(file_card, fg_color="transparent")
        fbtn.pack(fill="x", padx=10, pady=(0, 10))
        fbtn.grid_columnconfigure(0, weight=1)
        fbtn.grid_columnconfigure(1, weight=1)
        self.srt_browse_btn = ctk.CTkButton(fbtn, text="Browse Files...", font=("Segoe UI", 12),
                                            fg_color=ACCENT, hover_color=ACCENT_DARK, height=36,
                                            corner_radius=8, text_color=BTN_TEXT, command=self._srt_browse_file)
        self.srt_browse_btn.grid(row=0, column=0, padx=(0, 4), sticky="ew")
        self.srt_clear_btn = ctk.CTkButton(fbtn, text="Clear", font=("Segoe UI", 12),
                                           fg_color=BTN_DIM, hover_color=BTN_DIM_HOVER, height=36,
                                           corner_radius=8, text_color=FG_SECONDARY,
                                           command=self._srt_clear_file)
        self.srt_clear_btn.grid(row=0, column=1, padx=(4, 0), sticky="ew")
        out_card = ctk.CTkFrame(scroll, fg_color=BG_CARD, corner_radius=12)
        out_card.grid(row=6, column=0, sticky="ew", padx=4, pady=(0, 6))
        out_card.grid_columnconfigure(0, weight=1)
        ctk.CTkLabel(out_card, text="Output Folder  (empty = same folder as video)",
                     font=("Segoe UI", 11, "bold"), text_color=FG_DIM
                     ).pack(anchor="w", padx=12, pady=(8, 2))
        orow = ctk.CTkFrame(out_card, fg_color="transparent")
        orow.pack(fill="x", padx=10, pady=(0, 10))
        orow.grid_columnconfigure(0, weight=1)
        self.srt_out_entry = ctk.CTkEntry(orow, font=("Segoe UI", 11),
                                          fg_color=BG_INPUT, text_color=FG_PRIMARY,
                                          corner_radius=8, height=34,
                                          placeholder_text="Same folder as video (default)")
        self.srt_out_entry.grid(row=0, column=0, sticky="ew", padx=(0, 6))
        ctk.CTkButton(orow, text="Browse...", width=90, font=("Segoe UI", 12),
                      fg_color=BTN_DIM, hover_color=BTN_DIM_HOVER, height=34,
                      corner_radius=8, text_color=FG_SECONDARY,
                      command=self._srt_browse_outdir
                      ).grid(row=0, column=1, sticky="e")
        perf_card = ctk.CTkFrame(scroll, fg_color=BG_CARD, corner_radius=12)
        perf_card.grid(row=7, column=0, sticky="ew", padx=4, pady=(0, 6))
        perf_card.grid_columnconfigure(1, weight=1)
        ctk.CTkLabel(eng_card, text="Engine:", font=("Segoe UI", 10, "bold"),
                     text_color=FG_DIM).grid(row=0, column=0, sticky="w", padx=(12, 4), pady=(8, 2))
        self.srt_engine_var = ctk.StringVar(value="Moonshine v2")
        self.srt_engine_menu = ctk.CTkOptionMenu(
            eng_card, variable=self.srt_engine_var,
            values=list(ENGINE_CHOICES), width=150,
            fg_color=BG_INPUT, button_color=ACCENT,
            command=self._on_srt_engine_changed)
        self.srt_engine_menu.grid(row=0, column=1, sticky="ew", padx=4, pady=(8, 2))
        self.srt_model_var = ctk.StringVar(value="")
        self.srt_model_menu = ctk.CTkOptionMenu(
            eng_card, variable=self.srt_model_var,
            values=[], width=190,
            fg_color=BG_INPUT, button_color=ACCENT,
            command=self._on_srt_model_changed)
        self.srt_model_menu.grid(row=0, column=2, sticky="ew", padx=(4, 12), pady=(8, 2))
        self._srt_engine_cb = None
        self._srt_model_cb = None
        try:
            self._refresh_srt_model_menu()
        except Exception:
            pass
        self.srt_cpu_info = ctk.CTkLabel(
            perf_card, text=f"CPU: {self._srt_max_cpu} cores detected (CPU-only)",
            font=("Segoe UI", 10), text_color=FG_DIM)
        self.srt_cpu_info.grid(row=1, column=0, columnspan=3, sticky="w", padx=12, pady=(2, 0))
        self.srt_cpu_var = ctk.IntVar(value=def_cpu)
        self.srt_cpu_slider = ctk.CTkSlider(
            perf_card, from_=1, to=max(2, self._srt_max_cpu),
            number_of_steps=max(1, self._srt_max_cpu - 1),
            button_color=ACCENT, progress_color=ACCENT,
            command=self._on_srt_cpu_changed)
        self.srt_cpu_slider.grid(row=2, column=0, columnspan=2, sticky="ew", padx=12, pady=(2, 8))
        self.srt_cpu_slider.set(def_cpu)
        if self._srt_max_cpu <= 1:
            try:
                self.srt_cpu_slider.configure(state="disabled")
            except Exception:
                pass
        self.srt_cpu_value = ctk.CTkLabel(perf_card, text=f"{def_cpu} threads",
                                          font=("Segoe UI", 11, "bold"), text_color=FG_PRIMARY)
        self.srt_cpu_value.grid(row=2, column=2, sticky="e", padx=(4, 12), pady=(2, 8))
        ctk.CTkLabel(perf_card, text="Compute:", font=("Segoe UI", 10, "bold"),
                     text_color=FG_DIM).grid(row=3, column=0, sticky="w", padx=(12, 4), pady=(2, 8))
        try:
            import gpu as _gpumod2
            _has_dgpu = _gpumod2.best_gpu() is not None
        except Exception:
            _has_dgpu = False
        _compute_vals = ["Auto", "CPU", "GPU"] if _has_dgpu else ["CPU only"]
        self.compute_var = ctk.StringVar(value=_compute_vals[0])
        self.compute_menu = ctk.CTkOptionMenu(
            perf_card, variable=self.compute_var,
            values=_compute_vals, width=110,
            fg_color=BG_INPUT, button_color=ACCENT,
            command=self._on_compute_changed)
        self.compute_menu.grid(row=3, column=1, sticky="w", padx=4, pady=(2, 8))
        if not _has_dgpu:
            try:
                self.compute_menu.configure(state="disabled")
            except Exception:
                pass
        self.srt_norm_var = ctk.BooleanVar(value=False)
        self.srt_norm_check = ctk.CTkCheckBox(
            perf_card, text="Boost quiet audio (normalize before transcription)",
            variable=self.srt_norm_var, font=("Segoe UI", 10),
            text_color=FG_DIM, fg_color=ACCENT,
            command=self._on_srt_opt_toggled)
        self.srt_norm_check.grid(row=4, column=0, columnspan=3, sticky="w",
                                 padx=12, pady=(2, 8))
        ctk.CTkLabel(perf_card, text="(Whisper/Canary)",
                     font=("Segoe UI", 9), text_color=FG_DIM
                     ).grid(row=3, column=2, sticky="e", padx=(4, 12), pady=(2, 8))
        self._compute_callback = None
        lang_card = ctk.CTkFrame(scroll, fg_color=BG_CARD, corner_radius=12)
        lang_card.grid(row=8, column=0, sticky="ew", padx=4, pady=(0, 6))
        lang_card.grid_columnconfigure(1, weight=1)
        lang_card.grid_columnconfigure(3, weight=1)
        ctk.CTkLabel(lang_card, text="Input Language:",
                     font=("Segoe UI", 10, "bold"), text_color=FG_DIM
                     ).grid(row=0, column=0, sticky="w", padx=(12, 4), pady=(8, 2))
        self.srt_input_lang_var = ctk.StringVar(value="Japanese")
        self.srt_input_lang_menu = ctk.CTkOptionMenu(
            lang_card, variable=self.srt_input_lang_var, values=SRT_LANG_DISPLAY,
            width=170, fg_color=BG_INPUT, button_color=ACCENT,
            command=self._on_srt_input_lang_changed)
        self.srt_input_lang_menu.grid(row=0, column=1, sticky="ew", padx=4, pady=(8, 2))
        ctk.CTkLabel(lang_card, text="Output Language:",
                     font=("Segoe UI", 10, "bold"), text_color=FG_DIM
                     ).grid(row=0, column=2, sticky="w", padx=(12, 4), pady=(8, 2))
        self.srt_output_lang_var = ctk.StringVar(value="English")
        self.srt_output_lang_menu = ctk.CTkOptionMenu(
            lang_card, variable=self.srt_output_lang_var, values=SRT_LANG_DISPLAY,
            width=170, fg_color=BG_INPUT, button_color=ACCENT,
            command=self._on_srt_output_lang_changed)
        self.srt_output_lang_menu.grid(row=0, column=3, sticky="ew", padx=4, pady=(8, 2))
        self.srt_lang_hint = ctk.CTkLabel(
            lang_card, text="Only Whisper and Canary-1B support language selection",
            font=("Segoe UI", 9), text_color=FG_DIM)
        self.srt_lang_hint.grid(row=1, column=0, columnspan=4, sticky="w", padx=12, pady=(2, 8))
        self._srt_input_lang_cb = None
        self._srt_output_lang_cb = None
        self.lang_card = lang_card
        style_card = ctk.CTkFrame(scroll, fg_color=BG_CARD, corner_radius=12)
        style_card.grid(row=9, column=0, sticky="ew", padx=4, pady=(0, 6))
        style_card.grid_columnconfigure(1, weight=1)
        ctk.CTkLabel(style_card, text="Subtitle size:",
                     font=("Segoe UI", 10, "bold"), text_color=FG_DIM
                     ).grid(row=0, column=0, sticky="w", padx=(12, 4), pady=(8, 2))
        self.burn_font_size_var = ctk.IntVar(value=18)
        self.burn_font_slider = ctk.CTkSlider(
            style_card, from_=12, to=32, number_of_steps=20,
            button_color=ACCENT, progress_color=ACCENT,
            command=self._on_burn_fontsize_changed)
        self.burn_font_slider.grid(row=0, column=1, sticky="ew", padx=4, pady=(8, 2))
        self.burn_font_slider.set(18)
        self.burn_font_value = ctk.CTkLabel(style_card, text="18",
                                            font=("Segoe UI", 11, "bold"),
                                            text_color=FG_PRIMARY)
        self.burn_font_value.grid(row=0, column=2, sticky="e", padx=(4, 6), pady=(8, 2))
        self.srt_preview_btn = ctk.CTkButton(
            style_card, text="Preview Frame", font=("Segoe UI", 12),
            fg_color=BTN_DIM, hover_color=BTN_DIM_HOVER, height=34,
            corner_radius=8, text_color=FG_SECONDARY,
            command=self._on_srt_preview)
        self.srt_preview_btn.grid(row=0, column=3, sticky="e", padx=(6, 12), pady=(8, 2))
        try:
            import gpu as _gpumod
            _ggpu = _gpumod.best_gpu()
            if _ggpu:
                _ghint = (f" GPU: {_ggpu.get('name', 'NVIDIA')} "
                          f"({_ggpu.get('free_mb', '?')}MB free) - NVENC modes enabled.")
            else:
                _ghint = " GPU: none detected - CPU modes only."
        except Exception:
            _ghint = ""
        ctk.CTkLabel(style_card,
                     text="Preview Frame tests THESE settings on one still image. "
                     "Sample from/len only pick the test spot (and the short clip "
                     "transcribed when no SRT exists yet) - Generate SRT/MP4 always "
                     "processes the FULL file." + _ghint,
                     font=("Segoe UI", 9), text_color=FG_DIM, wraplength=420,
                     justify="left").grid(row=5, column=0, columnspan=4,
                                          sticky="w", padx=12, pady=(2, 8))
        estrow = ctk.CTkFrame(style_card, fg_color="transparent")
        estrow.grid(row=6, column=0, columnspan=4, sticky="ew",
                    padx=12, pady=(2, 10))
        ctk.CTkLabel(estrow, text="Est. size:",
                     font=("Segoe UI", 10, "bold"),
                     text_color=FG_DIM).pack(side="left")
        self.burn_est_entry = ctk.CTkEntry(
            estrow, width=90, height=28, font=("Segoe UI", 11, "bold"),
            fg_color=BG_INPUT, text_color=FG_PRIMARY, corner_radius=8,
            placeholder_text="MB")
        self.burn_est_entry.pack(side="left", padx=(6, 4))
        self.burn_est_entry.bind("<Return>", self._on_burn_mb_commit)
        self.burn_est_entry.bind("<FocusOut>", self._on_burn_mb_commit)
        ctk.CTkLabel(estrow, text="MB", font=("Segoe UI", 10),
                     text_color=FG_DIM).pack(side="left", padx=(0, 8))
        self.burn_est_basis = ctk.CTkLabel(
            estrow, text="", font=("Segoe UI", 10), text_color=FG_DIM,
            wraplength=300, justify="left")
        self.burn_est_basis.pack(side="left", fill="x", expand=True)
        self._burn_probe_cache = {}
        self._burn_est_token = 0
        self._burn_est_sig = None
        self._burn_est_ctx = None
        self._burn_est_shown = ""
        self._burn_vbr_exact = None
        self._srt_preview_cb = None
        self._preview_running = False
        ctk.CTkLabel(style_card, text="Sample from:",
                     font=("Segoe UI", 10), text_color=FG_DIM
                     ).grid(row=1, column=0, sticky="w", padx=(12, 4), pady=(2, 2))
        self.sample_start_var = ctk.StringVar(value="0:30")
        self.sample_start_entry = ctk.CTkEntry(
            style_card, textvariable=self.sample_start_var, width=70,
            font=("Segoe UI", 11), fg_color=BG_INPUT, text_color=FG_PRIMARY,
            corner_radius=8, height=30)
        self.sample_start_entry.grid(row=1, column=1, sticky="w", padx=4, pady=(2, 2))
        ctk.CTkLabel(style_card, text="Sample len:",
                     font=("Segoe UI", 10), text_color=FG_DIM
                     ).grid(row=1, column=2, sticky="e", padx=(4, 4), pady=(2, 2))
        self.sample_len_var = ctk.StringVar(value="15s")
        self.sample_len_menu = ctk.CTkOptionMenu(
            style_card, variable=self.sample_len_var,
            values=["10s", "15s", "30s", "60s"], width=80,
            fg_color=BG_INPUT, button_color=ACCENT)
        self.sample_len_menu.grid(row=1, column=3, sticky="e", padx=(4, 12), pady=(2, 2))
        ctk.CTkLabel(style_card, text="Burn speed:",
                     font=("Segoe UI", 10), text_color=FG_DIM
                     ).grid(row=2, column=0, sticky="w", padx=(12, 4), pady=(2, 2))
        try:
            from srt import BURN_SPEED_LABELS as _BSL
            _speed_vals = [_BSL[k] for k in
                           ("match", "fast", "fastest",
                            "nvenc_draft", "nvenc_turbo", "nvenc_balanced")
                           if k in _BSL]
        except Exception:
            _speed_vals = ["Match size (2-pass x264)", "Fast (1-pass x264)",
                           "Fastest (ultrafast 1-pass)"]
        if not _speed_vals:
            _speed_vals = ["Match size (2-pass)"]
        try:
            import gpu as _gpumod0
            from srt import get_ffmpeg_exe as _ff0
            try:
                _exe0 = _ff0()
            except Exception:
                _exe0 = None
            _nv_any = bool(_gpumod0.nvenc_available(_exe0, "h264_nvenc"))
            _nv_hevc = bool(_gpumod0.nvenc_available(_exe0, "hevc_nvenc"))
        except Exception:
            _nv_any, _nv_hevc = False, False
        self._has_nvenc = bool(_nv_any or _nv_hevc)
        self._has_nvenc_hevc = bool(_nv_hevc)
        if not self._has_nvenc:
            _cpu_only = [v for v in _speed_vals if "NVENC" not in v]
            if _cpu_only:
                _speed_vals = _cpu_only
        self.burn_speed_var = ctk.StringVar(value=_speed_vals[0])
        self.burn_speed_menu = ctk.CTkOptionMenu(
            style_card, variable=self.burn_speed_var,
            values=_speed_vals, width=200,
            fg_color=BG_INPUT, button_color=ACCENT,
            command=self._on_burn_speed_changed)
        self.burn_speed_menu.grid(row=2, column=1, sticky="ew",
                                  padx=4, pady=(2, 2))
        self.burn_codec_var = ctk.StringVar(value="H.264")
        self.burn_codec_menu = ctk.CTkOptionMenu(
            style_card, variable=self.burn_codec_var,
            values=["H.264", "HEVC (H.265)"], width=130,
            fg_color=BG_INPUT, button_color=ACCENT,
            command=self._on_burn_codec_changed)
        self.burn_codec_menu.grid(row=2, column=2, sticky="e",
                                  padx=4, pady=(2, 2))
        self.burn_compare_btn = ctk.CTkButton(
            style_card, text="ⓘ", width=32, height=28, font=("Segoe UI", 13),
            fg_color=BTN_DIM, hover_color=BTN_DIM_HOVER, text_color=FG_SECONDARY,
            corner_radius=8, command=self.show_burn_compare)
        self.burn_compare_btn.grid(row=2, column=3, sticky="e",
                                   padx=(4, 12), pady=(2, 2))
        self.burn_speed_desc = ctk.CTkLabel(
            style_card, text="", font=("Segoe UI", 10),
            text_color=ACCENT_GLOW, wraplength=420, justify="left")
        self.burn_speed_desc.grid(row=4, column=0, columnspan=4,
                                  sticky="w", padx=12, pady=(0, 2))
        self._refresh_burn_speed_desc()
        ctk.CTkLabel(style_card, text="Video bitrate:",
                     font=("Segoe UI", 10), text_color=FG_DIM
                     ).grid(row=3, column=0, sticky="w", padx=(12, 4), pady=(2, 2))
        self.burn_vbr_var = ctk.IntVar(value=2000)
        self.burn_vbr_slider = ctk.CTkSlider(
            style_card, from_=300, to=10000, number_of_steps=97,
            button_color=ACCENT, progress_color=ACCENT,
            command=self._on_burn_vbr_changed)
        self.burn_vbr_slider.grid(row=3, column=1, sticky="ew", padx=4, pady=(2, 2))
        self.burn_vbr_slider.set(2000)
        self.burn_vbr_value = ctk.CTkLabel(style_card, text="auto",
                                           font=("Segoe UI", 11, "bold"),
                                           text_color=FG_PRIMARY)
        self.burn_vbr_value.grid(row=3, column=2, sticky="e", padx=(4, 6), pady=(2, 2))
        self.burn_vbr_auto_var = ctk.BooleanVar(value=True)
        self.burn_vbr_check = ctk.CTkCheckBox(
            style_card, text="Auto", variable=self.burn_vbr_auto_var,
            font=("Segoe UI", 10), text_color=FG_DIM, fg_color=ACCENT,
            command=self._on_burn_vbr_toggled)
        self.burn_vbr_check.grid(row=3, column=3, sticky="e", padx=(6, 12), pady=(2, 2))
        try:
            self.burn_vbr_slider.configure(state="disabled")
        except Exception:
            pass
        prog_card = ctk.CTkFrame(scroll, fg_color=BG_CARD, corner_radius=12)
        prog_card.grid(row=5, column=0, sticky="ew", padx=4, pady=(0, 6))
        prog_card.grid_columnconfigure(0, weight=1)
        toprow = ctk.CTkFrame(prog_card, fg_color="transparent")
        toprow.pack(fill="x", padx=12, pady=(8, 2))
        ctk.CTkLabel(toprow, text="Progress",
                     font=("Segoe UI", 11, "bold"), text_color=FG_DIM).pack(side="left")
        self.srt_pct = ctk.CTkLabel(toprow, text="0%",
                                    font=("Segoe UI", 11, "bold"), text_color=FG_SECONDARY)
        self.srt_pct.pack(side="right")
        self.burn_hist_btn = ctk.CTkButton(
            toprow, text="↺ Burn", width=66, height=24, font=("Segoe UI", 10),
            fg_color=BTN_DIM, hover_color=BTN_DIM_HOVER, text_color=FG_SECONDARY,
            corner_radius=6, command=self._on_clear_burn_hist)
        self.burn_hist_btn.pack(side="right", padx=(0, 6))
        self.srt_hist_btn = ctk.CTkButton(
            toprow, text="↺ SRT", width=60, height=24, font=("Segoe UI", 10),
            fg_color=BTN_DIM, hover_color=BTN_DIM_HOVER, text_color=FG_SECONDARY,
            corner_radius=6, command=self._on_clear_srt_hist)
        self.srt_hist_btn.pack(side="right", padx=(0, 6))
        self.srt_bar = ctk.CTkProgressBar(prog_card, fg_color=BG_INPUT,
                                          progress_color=ACCENT, height=12,
                                          corner_radius=7)
        self.srt_bar.pack(fill="x", padx=12, pady=(0, 4))
        self.srt_bar.set(0)
        self.srt_status = ctk.CTkLabel(prog_card, text="Idle - pick a file to begin",
                                       font=("Segoe UI", 10), text_color=FG_DIM,
                                       wraplength=400, justify="left")
        self.srt_status.pack(anchor="w", padx=12, pady=(0, 4))
        self.srt_log_box = ctk.CTkTextbox(prog_card, font=("Consolas", 10),
                                          fg_color=BG_INPUT, text_color=FG_SECONDARY,
                                          corner_radius=8, height=88, wrap="word")
        self.srt_log_box.pack(fill="x", padx=10, pady=(0, 6))
        self.srt_log_box.insert("1.0", "SRT log ready.\n")
        self.srt_log_box.configure(state="disabled")
        abtn = ctk.CTkFrame(scroll, fg_color="transparent")
        abtn.grid(row=2, column=0, sticky="ew", padx=4, pady=(0, 6))
        abtn.grid_columnconfigure(0, weight=3)
        abtn.grid_columnconfigure(1, weight=1)
        abtn.grid_columnconfigure(2, weight=1)
        self.srt_start_btn = ctk.CTkButton(
            abtn, text="\u25B6  Generate SRT", font=("Segoe UI", 13, "bold"),
            fg_color=SUCCESS, hover_color=BTN_GO_HOVER, height=36,
            text_color=BTN_TEXT,
            corner_radius=10, command=self._on_srt_start)
        self.srt_start_btn.grid(row=0, column=0, padx=(0, 4), sticky="ew")
        self.srt_cancel_btn = ctk.CTkButton(
            abtn, text="Cancel", font=("Segoe UI", 12),
            fg_color=DANGER, hover_color=BTN_DANGER_HOVER, height=36,
            text_color=BTN_TEXT,
            corner_radius=10, state="disabled", command=self._on_srt_cancel)
        self.srt_cancel_btn.grid(row=0, column=1, padx=(4, 0), sticky="ew")
        self.srt_burn_btn = ctk.CTkButton(
            abtn, text="Burn SRT into MP4", font=("Segoe UI", 12, "bold"),
            fg_color=BTN_BURN, hover_color=BTN_BURN_HOVER, height=36,
            text_color=BTN_TEXT,
            corner_radius=10, command=self._on_srt_burn)
        self.srt_burn_btn.grid(row=1, column=0, padx=(0, 4), pady=(8, 0), sticky="ew")
        ctk.CTkButton(abtn, text="Open Folder", font=("Segoe UI", 12),
                      fg_color=BTN_DIM, hover_color=BTN_DIM_HOVER, height=36,
                      corner_radius=10, text_color=FG_SECONDARY,
                      command=self._on_srt_open_folder
                      ).grid(row=0, column=2, padx=(4, 0), sticky="ew")
        self._srt_burn_cb = None
        self._srt_mode = "generate"
        self._srt_running_mode = "generate"
        self.burn_after_var = ctk.BooleanVar(value=False)
        self.burn_after_check = ctk.CTkCheckBox(
            scroll, text="Burn MP4 automatically after SRT (one-click hardcode)",
            variable=self.burn_after_var, font=("Segoe UI", 11),
            text_color=FG_PRIMARY, fg_color=ACCENT,
            command=self._on_srt_opt_toggled)
        self.burn_after_check.grid(row=3, column=0, sticky="w", padx=16, pady=(0, 2))
        finrow = ctk.CTkFrame(scroll, fg_color="transparent")
        finrow.grid(row=4, column=0, sticky="ew", padx=16, pady=(0, 6))
        self.shutdown_var = ctk.BooleanVar(value=False)
        self.shutdown_check = ctk.CTkCheckBox(
            finrow, text="Shut down PC when done",
            variable=self.shutdown_var, font=("Segoe UI", 11),
            text_color=FG_PRIMARY, fg_color=DANGER,
            command=self._on_srt_opt_toggled)
        self.shutdown_check.pack(side="left", padx=(0, 18))
        self.alert_var = ctk.BooleanVar(value=True)
        self.alert_check = ctk.CTkCheckBox(
            finrow, text="Pop-up alert when done",
            variable=self.alert_var, font=("Segoe UI", 11),
            text_color=FG_PRIMARY, fg_color=ACCENT,
            command=self._on_srt_opt_toggled)
        self.alert_check.pack(side="left")
        self._srt_opt_callback = None
        self._show_srt_action("generate")

    # ═══════════════════════════════════════════════════════════
    # NOTE TAB
    # ═══════════════════════════════════════════════════════════
    def _build_note_tab(self, tab):
        """Build the Note tab UI - professional dictation layout."""
        import customtkinter as ctk

        self._note_recording = False
        self._note_start_time = 0
        self._note_timer_id = None
        self._note_submitted = 0  # chunks cut by the recorder
        self._note_done = 0  # chunks fully transcribed (ok or not)
        self._note_dirty = False  # unsaved note content present
        self._note_mic_warned = False  # mic-dead popup latch (per episode)
        self._note_last_level_t = 0.0  # level-post throttle stamp
        self._note_pending = False  # auto-start armed while engine loads
        self._note_auto_mp3s = set()  # session MP3s auto-saved this run
        self._note_saved_mp3s = set()  # ...of which the user kept via Save Audio

        scroll = ctk.CTkScrollableFrame(tab, fg_color="transparent")
        scroll.pack(fill="both", expand=True, padx=4, pady=4)
        scroll.grid_columnconfigure(0, weight=1)

        # ── Header card ──
        header_card = ctk.CTkFrame(scroll, fg_color=BG_CARD, corner_radius=16)
        header_card.grid(row=1, column=0, sticky="ew", padx=4, pady=(0, 8))
        header_card.grid_columnconfigure(0, weight=1)

        ctk.CTkLabel(header_card, text="Note Mode",
                     font=("Segoe UI", 18, "bold"),
                     text_color=FG_PRIMARY).pack(anchor="w", padx=20, pady=(16, 2))
        ctk.CTkLabel(header_card,
                     text="Record and transcribe in real-time. Audio is processed in "
                          "40-80s chunks at natural pauses.",
                     font=("Segoe UI", 11), text_color=FG_DIM,
                     wraplength=400, justify="left"
                     ).pack(anchor="w", padx=20, pady=(0, 12))

        # ── Record control card ──
        ctrl_card = ctk.CTkFrame(scroll, fg_color=BG_CARD, corner_radius=16)
        ctrl_card.grid(row=2, column=0, sticky="ew", padx=4, pady=(0, 8))
        ctrl_card.grid_columnconfigure(0, weight=1)

        # Timer display
        self.note_timer_label = ctk.CTkLabel(
            ctrl_card, text="00:00:00",
            font=("Consolas", 32, "bold"), text_color=FG_PRIMARY)
        self.note_timer_label.pack(pady=(20, 4))

        # Status line
        self.note_status_label = ctk.CTkLabel(
            ctrl_card, text="Ready to record",
            font=("Segoe UI", 11), text_color=FG_DIM)
        self.note_status_label.pack(pady=(0, 8))

        # Live input meter (proves the mic is alive long before chunk 1
        # lands 40-80s in; doubles as the mic-death visual).
        try:
            self.note_meter = LevelMeter(ctrl_card, width=360, height=8)
            self.note_meter.pack(pady=(0, 8))
        except Exception:
            self.note_meter = None

        # Record / Stop button
        btn_row = ctk.CTkFrame(ctrl_card, fg_color="transparent")
        btn_row.pack(pady=(0, 20))

        self.note_record_btn = ctk.CTkButton(
            btn_row, text="●  RECORD", width=180, height=48,
            font=("Segoe UI", 15, "bold"),
            fg_color=ACCENT, hover_color=ACCENT_DARK,
            text_color=BTN_TEXT, corner_radius=24,
            command=self._on_note_toggle)
        self.note_record_btn.pack(side="left", padx=6)

        self.note_clear_btn = ctk.CTkButton(
            btn_row, text="Clear", width=90, height=48,
            font=("Segoe UI", 12),
            fg_color=BTN_DIM, hover_color=BTN_DIM_HOVER,
            text_color=FG_SECONDARY, corner_radius=24,
            command=self._on_note_clear)
        self.note_clear_btn.pack(side="left", padx=6)

        # Chunk info
        self.note_chunk_label = ctk.CTkLabel(
            ctrl_card, text="", font=("Segoe UI", 10), text_color=FG_DIM)
        self.note_chunk_label.pack(pady=(0, 12))

        # ── Transcription output card ──
        out_card = ctk.CTkFrame(scroll, fg_color=BG_CARD, corner_radius=16)
        out_card.grid(row=3, column=0, sticky="nsew", padx=4, pady=(0, 8))
        out_card.grid_columnconfigure(0, weight=1)
        out_card.grid_rowconfigure(1, weight=1)

        out_header = ctk.CTkFrame(out_card, fg_color="transparent")
        out_header.grid(row=0, column=0, sticky="ew", padx=16, pady=(12, 4))

        ctk.CTkLabel(out_header, text="Transcription",
                     font=("Segoe UI", 12, "bold"),
                     text_color=FG_DIM).pack(side="left")

        self.note_save_btn = ctk.CTkButton(
            out_header, text="Save as TXT", width=110, height=30,
            font=("Segoe UI", 11),
            fg_color=SUCCESS, hover_color=BTN_GO_HOVER,
            text_color=BTN_TEXT, corner_radius=8,
            command=self._on_note_save)
        self.note_save_btn.pack(side="right")

        self.note_save_audio_btn = ctk.CTkButton(
            out_header, text="Save Audio", width=100, height=30,
            font=("Segoe UI", 11),
            fg_color=BTN_DIM, hover_color=BTN_DIM_HOVER,
            text_color=FG_SECONDARY, corner_radius=8,
            command=self._on_note_save_audio)
        self.note_save_audio_btn.pack(side="right", padx=(0, 6))

        self.note_copy_btn = ctk.CTkButton(
            out_header, text="Copy All", width=90, height=30,
            font=("Segoe UI", 11),
            fg_color=BTN_DIM, hover_color=BTN_DIM_HOVER,
            text_color=FG_SECONDARY, corner_radius=8,
            command=self._on_note_copy)
        self.note_copy_btn.pack(side="right", padx=(0, 6))

        self.note_text = ctk.CTkTextbox(
            out_card, font=("Segoe UI", 13),
            fg_color=BG_INPUT, text_color=FG_PRIMARY,
            corner_radius=10, wrap="word", height=280)
        self.note_text.grid(row=1, column=0, sticky="nsew", padx=12, pady=(4, 12))
        self.note_text.insert("1.0", "Transcription will appear here as you speak...")
        try:
            self.note_text.bind("<Key>", self._note_mark_dirty)
        except Exception:
            pass

        # ── Engine pick (per-tab: independent from Live/SRT) ──
        info_card = ctk.CTkFrame(scroll, fg_color=BG_CARD, corner_radius=12)
        info_card.grid(row=0, column=0, sticky="ew", padx=4, pady=(0, 6))
        info_card.grid_columnconfigure(1, weight=1)

        self.note_engine_label = ctk.CTkLabel(
            info_card, text="Engine: following shared default",
            font=("Segoe UI", 10), text_color=FG_DIM)
        self.note_engine_label.grid(row=0, column=0, columnspan=3,
                                     sticky="w", padx=16, pady=(10, 2))
        ctk.CTkLabel(info_card, text="Engine:",
                     font=("Segoe UI", 10, "bold"),
                     text_color=FG_DIM).grid(row=1, column=0, sticky="w",
                                             padx=(16, 4), pady=(2, 10))
        self.note_engine_var = ctk.StringVar(value="Moonshine v2")
        self.note_engine_menu = ctk.CTkOptionMenu(
            info_card, variable=self.note_engine_var,
            values=list(ENGINE_CHOICES), width=150,
            fg_color=BG_INPUT, button_color=ACCENT,
            command=self._on_note_engine_changed)
        self.note_engine_menu.grid(row=1, column=1, sticky="ew",
                                    padx=4, pady=(2, 10))
        self.note_model_var = ctk.StringVar(value="")
        self.note_model_menu = ctk.CTkOptionMenu(
            info_card, variable=self.note_model_var,
            values=[], width=190,
            fg_color=BG_INPUT, button_color=ACCENT,
            command=self._on_note_model_changed)
        self.note_model_menu.grid(row=1, column=2, sticky="ew",
                                   padx=(4, 16), pady=(2, 10))
        self._note_engine_cb = None
        self._note_model_cb = None
        try:
            self._refresh_note_model_menu()
        except Exception:
            pass

        # Init note recorder
        try:
            from note_engine import NoteRecorder, NoteTranscriber
            self._note_recorder = NoteRecorder(sample_rate=16000)
            self._note_transcriber = NoteTranscriber()
            self._note_recorder.set_callbacks(
                on_chunk_ready=self._note_on_chunk,
                on_level=self._note_on_level,
                on_status=self._note_on_rec_status)
            self._note_transcriber.set_callbacks(
                on_text=self._note_on_text,
                on_status=self._note_on_tx_status,
                on_done=self._note_on_done)
        except Exception as e:
            print(f"[Note] init failed: {e}")
            self._note_recorder = None
            self._note_transcriber = None

    def _on_note_toggle(self):
        if self._note_recording:
            self._note_stop()
        elif getattr(self, "_note_pending", False):
            # Second press while waiting for the engine: cancel the wait.
            try:
                self._note_pending = False
            except Exception:
                pass
            try:
                self.note_status_label.configure(text="Ready to record",
                                                 text_color=FG_DIM)
            except Exception:
                pass
        else:
            self._note_start()

    def _note_mark_dirty(self, event=None):
        try:
            self._note_dirty = True
        except Exception:
            pass

    def _note_prepare_box(self):
        """Start-of-session textbox prep. A fresh/placeholder box is cleared;
        a continued note keeps its text and gets a timestamp separator, so
        RECORD-after-STOP appends instead of wiping. Sets the dirty flag."""
        try:
            current = self.note_text.get("1.0", "end").strip()
        except Exception:
            current = ""
        try:
            if not current or current == "Transcription will appear here as you speak...":
                self.note_text.delete("1.0", "end")
                self._note_dirty = False
            else:
                self.note_text.insert("end", f"\n─── {time.strftime('%H:%M')} ───\n\n")
                try:
                    self.note_text.see("end")
                except Exception:
                    pass
                self._note_dirty = True
        except Exception:
            pass

    def set_note_record_callback(self, cb, confirm_cb=None, ready_cb=None):
        """App pre-flight for RECORD: returns go/confirm/wait/abort verdict.
        confirm_cb(dual_ok) follows a confirm verdict; ready_cb() reports
        (ready_bool, problem_str) for the pending auto-start poll. None-safe."""
        try:
            self._note_record_cb = cb if callable(cb) else None
            self._note_record_confirm = confirm_cb if callable(confirm_cb) else None
            self._note_ready_cb = ready_cb if callable(ready_cb) else None
        except Exception:
            self._note_record_cb = None
            self._note_record_confirm = None
            self._note_ready_cb = None

    def _note_start(self):
        if self._note_recorder is None:
            return
        try:
            # App pre-flight: snapshots the Note tab engine, confirms dual
            # loads, or refuses while it loads. Runs on the GUI thread.
            if getattr(self, "_note_record_cb", None) is not None:
                try:
                    verdict = self._note_record_cb() or {"go": True}
                except Exception:
                    verdict = {"go": True}
                if isinstance(verdict, dict) and "confirm" in verdict:
                    try:
                        from tkinter import messagebox as _mb
                        _yes = bool(_mb.askyesno(
                            "Load second engine?",
                            str(verdict.get("confirm") or
                                "Another session is active."),
                            parent=self))
                    except Exception:
                        _yes = False
                    try:
                        follow = self._note_record_confirm(bool(_yes)) \
                            if getattr(self, "_note_record_confirm", None) \
                            is not None else {"abort": True}
                    except Exception:
                        follow = {"abort": True}
                    if not (isinstance(follow, dict) and follow.get("go")):
                        if isinstance(follow, dict) and "wait" in follow:
                            # Dual load accepted: same pending auto-start.
                            try:
                                self.note_status_label.configure(
                                    text="Loading note engine — recording starts automatically…",
                                    text_color=WARNING)
                            except Exception:
                                pass
                            try:
                                self._note_pending = True
                                self._note_wait_engine()
                            except Exception:
                                pass
                            return
                        try:
                            self.note_status_label.configure(
                                text=str((follow or {}).get("wait") or
                                         "Note record cancelled"),
                                text_color=WARNING)
                        except Exception:
                            pass
                        return
                elif isinstance(verdict, dict) and "wait" in verdict:
                    # Engine still loading: arm a pending auto-start instead
                    # of dead-ending. The 1/sec poll below begins capture
                    # the moment it is ready; second RECORD press cancels.
                    try:
                        self.note_status_label.configure(
                            text="Loading note engine — recording starts automatically…",
                            text_color=WARNING)
                    except Exception:
                        pass
                    try:
                        self._note_pending = True
                        self._note_wait_engine()
                    except Exception:
                        pass
                    return
                elif isinstance(verdict, dict) and "abort" in verdict:
                    return
            self._note_begin_capture()
        except Exception as e:
            self.note_status_label.configure(text=f"Error: {e}", text_color=DANGER)

    def _note_wait_engine(self):
        """Pending auto-start poll (GUI-thread after() chain, no threads).
        Ends by beginning capture, surfacing a load failure, or on cancel."""
        try:
            if not getattr(self, "_note_pending", False):
                return
            if getattr(self, "_note_recording", False):
                try:
                    self._note_pending = False
                except Exception:
                    pass
                return
            ready, problem = False, ""
            try:
                chk = getattr(self, "_note_ready_cb", None)
                if chk is not None:
                    ready, problem = chk()
                else:
                    ready, problem = True, ""
            except Exception:
                ready, problem = False, ""
            if ready:
                try:
                    self._note_pending = False
                except Exception:
                    pass
                try:
                    self._note_begin_capture()
                except Exception as e:
                    try:
                        self.note_status_label.configure(
                            text=f"Error: {e}", text_color=DANGER)
                    except Exception:
                        pass
                return
            if problem:
                try:
                    self._note_pending = False
                except Exception:
                    pass
                try:
                    self.note_status_label.configure(text=str(problem),
                                                     text_color=DANGER)
                except Exception:
                    pass
                return
            try:
                self.after(1000, self._note_wait_engine)
            except Exception:
                pass
        except Exception:
            pass

    def _note_begin_capture(self):
        """Start recorder + transcriber + UI state. Re-runs the pre-flight
        first so the engine snapshot is always fresh - the wait/confirm
        paths load engines but never snapshot them."""
        try:
            if getattr(self, "_note_record_cb", None) is not None:
                try:
                    _v = self._note_record_cb() or {"go": True}
                except Exception:
                    _v = {"go": True}
                if isinstance(_v, dict) and not _v.get("go"):
                    try:
                        self.note_status_label.configure(
                            text="Note engine not ready — press RECORD to retry",
                            text_color=WARNING)
                    except Exception:
                        pass
                    return
            # Set transcription function based on current engine
            self._note_setup_transcribe_fn()
            try:
                from pathlib import Path as _P
                _notes_dir = _P(__file__).parent / "notes_audio"
            except Exception:
                _notes_dir = None
            self._note_recorder.start(session_dir=_notes_dir)
            self._note_transcriber.start()
            self._note_recording = True
            self._note_start_time = time.time()
            self._note_submitted = 0
            self._note_done = 0
            self._note_update_counter()

            self.note_record_btn.configure(
                text="■  STOP", fg_color=DANGER, hover_color=BTN_DANGER_HOVER)
            self.note_status_label.configure(text="Recording...", text_color=DANGER)
            self._note_prepare_box()
            self._note_mic_warned = False

            self._note_update_timer()
        except Exception as e:
            self.note_status_label.configure(text=f"Error: {e}", text_color=DANGER)

    def _note_stop(self):
        if not self._note_recording:
            return
        self._note_recording = False

        # Stop timer
        if self._note_timer_id:
            try:
                self.after_cancel(self._note_timer_id)
            except Exception:
                pass
            self._note_timer_id = None

        # Get remaining audio
        remaining = self._note_recorder.stop()
        if remaining is not None and len(remaining) > 16000:
            self._note_transcriber.submit_chunk(remaining, self._note_recorder.chunk_count + 1)

        self._note_transcriber.stop()

        # Full-session MP3 archive in the background: transcription and UI
        # never wait for it; any failure only logs (WAV is kept then).
        try:
            _wav = getattr(self._note_recorder, "session_wav", None)
        except Exception:
            _wav = None
        if _wav:
            try:
                import threading as _th
                _th.Thread(target=self._note_export_mp3, args=(str(_wav),),
                           daemon=True).start()
            except Exception:
                pass

        self.note_record_btn.configure(
            text="●  RECORD", fg_color=ACCENT, hover_color=ACCENT_DARK)
        self.note_status_label.configure(text="Stopped - review and save", text_color=SUCCESS)
        self.note_chunk_label.configure(text="")

    def _note_export_mp3(self, wav_path):
        """Background worker: session WAV -> notes_audio/note_*.mp3, then a
        one-line status. Never touches Tk except via after()."""
        def _say(text, color):
            try:
                self.after(0, lambda: self.note_status_label.configure(
                    text=text, text_color=color))
            except Exception:
                pass

        try:
            from pathlib import Path as _P
            from note_engine import wav_to_mp3
            try:
                import imageio_ffmpeg as _iff
                _ff = _iff.get_ffmpeg_exe()
            except Exception:
                _ff = None
            if not _ff:
                _say("Audio archive skipped (ffmpeg missing)", WARNING)
                return
            wav = _P(str(wav_path))
            mp3 = wav.with_suffix(".mp3")
            ok, msg = wav_to_mp3(_ff, wav, mp3, 128)
            if ok:
                try:
                    self._note_auto_mp3s.add(str(mp3))
                except Exception:
                    pass
                _say(f"Audio saved: {msg}", SUCCESS)
            else:
                _say(f"Audio archive kept as WAV ({msg}): {wav.name}", WARNING)
        except Exception as e:
            _say(f"Audio archive failed: {e}", WARNING)

    def _note_setup_transcribe_fn(self):
        """Wire up the transcription function from the app's current engine."""
        try:
            # This will be set by the main app via set_note_transcribe_fn
            if hasattr(self, '_note_transcribe_fn') and self._note_transcribe_fn:
                self._note_transcriber.set_transcribe_fn(self._note_transcribe_fn)
        except Exception:
            pass

    def set_note_transcribe_fn(self, fn):
        """Called by main app to set the transcription function."""
        self._note_transcribe_fn = fn
        if self._note_transcriber:
            self._note_transcriber.set_transcribe_fn(fn)

    def _note_update_counter(self):
        """Honest progress: transcribed/completed out of cut chunks."""
        try:
            done = int(getattr(self, "_note_done", 0) or 0)
            sub = int(getattr(self, "_note_submitted", 0) or 0)
            self.note_chunk_label.configure(
                text=f"Transcribed {done}/{sub} chunk{'s' if sub != 1 else ''}")
        except Exception:
            pass

    def _note_on_done(self, index, ok):
        """Worker-thread completion signal: marshal count to GUI thread."""
        try:
            self.after(0, self._note_count_done)
        except Exception:
            pass

    def _note_count_done(self):
        try:
            self._note_done = int(getattr(self, "_note_done", 0) or 0) + 1
        except Exception:
            self._note_done = 1
        self._note_update_counter()

    def _note_on_chunk(self, audio, index):
        """Called from recorder thread when a chunk is ready."""
        if self._note_transcriber:
            self._note_transcriber.submit_chunk(audio, index)
        try:
            if int(index) > int(getattr(self, "_note_submitted", 0) or 0):
                self._note_submitted = int(index)
        except Exception:
            pass
        try:
            self.after(0, self._note_update_counter)
        except Exception:
            pass

    def _note_on_level(self, level):
        # Audio-thread callback (~64/s): throttle, hop to the GUI thread,
        # and never touch Tk off-thread (a canvas call from here would
        # eventually crash the interpreter).
        try:
            now = time.monotonic()
            if now - float(getattr(self, "_note_last_level_t", 0.0)) < 0.066:
                return
            self._note_last_level_t = now
            meter = getattr(self, "note_meter", None)
            try:
                value = max(0.0, min(1.0, float(level or 0.0)))
            except Exception:
                return
            self.after(0, lambda: self._note_safe_level(meter, value))
        except Exception:
            pass

    @staticmethod
    def _note_safe_level(meter, value):
        try:
            if meter is not None and meter.winfo_exists():
                meter.set_level(value)
        except Exception:
            pass

    def _note_on_rec_status(self, msg):
        try:
            self.after(0, lambda m=msg: self.note_status_label.configure(
                text=m, text_color=FG_SECONDARY))
        except Exception:
            pass

    def _note_on_text(self, text, index):
        """Called when a chunk is transcribed. The box stays editable, so
        appends go to the end without disturbing text being edited."""
        def _append():
            try:
                current = self.note_text.get("1.0", "end").strip()
                if current == "Transcription will appear here as you speak...":
                    self.note_text.delete("1.0", "end")
                self.note_text.insert("end", text + "\n\n")
                self.note_text.see("end")
                self._note_dirty = True
            except Exception:
                pass
        try:
            self.after(0, _append)
        except Exception:
            pass

    def _note_on_tx_status(self, msg):
        try:
            self.after(0, lambda m=msg: self.note_status_label.configure(
                text=m, text_color=FG_SECONDARY))
        except Exception:
            pass

    def _note_update_timer(self):
        if not self._note_recording:
            return
        elapsed = time.time() - self._note_start_time
        h = int(elapsed // 3600)
        m = int((elapsed % 3600) // 60)
        s = int(elapsed % 60)
        self.note_timer_label.configure(text=f"{h:02d}:{m:02d}:{s:02d}")
        # Mic watchdog (this tick is the only per-second work added: one
        # timestamp read, no threads, no audio work - nil perf impact).
        try:
            rec = getattr(self, "_note_recorder", None)
            if rec is not None:
                state, _detail = rec.mic_health()
                if state == "dead":
                    self.note_status_label.configure(
                        text="No microphone input — check mic", text_color=DANGER)
                    # Popup-focus once per dead episode, recording only.
                    # Latch clears when input resumes; no modal (a modal
                    # would block the STOP button mid-record).
                    if not getattr(self, "_note_mic_warned", False):
                        self._note_mic_warned = True
                        try:
                            self.lift()
                        except Exception:
                            pass
                        try:
                            self.attributes("-topmost", True)
                            self.after(1500, self._note_drop_topmost)
                        except Exception:
                            pass
                elif state == "silent":
                    self.note_status_label.configure(
                        text="Microphone silent — check input level",
                        text_color=WARNING)
                elif getattr(self, "_note_mic_warned", False):
                    self._note_mic_warned = False
                    self.note_status_label.configure(text="Recording...",
                                                     text_color=DANGER)
        except Exception:
            pass
        self._note_timer_id = self.after(1000, self._note_update_timer)

    def _note_drop_topmost(self):
        try:
            self.attributes("-topmost", False)
        except Exception:
            pass

    def _on_note_clear(self):
        if self._note_recording:
            return
        self.note_text.delete("1.0", "end")
        self.note_text.insert("1.0", "Transcription will appear here as you speak...")
        self.note_chunk_label.configure(text="")
        self._note_dirty = False
        self.note_status_label.configure(text="Ready to record", text_color=FG_DIM)

    def _on_note_copy(self):
        try:
            text = self.note_text.get("1.0", "end").strip()
            if text and text != "Transcription will appear here as you speak...":
                from input_sim import copy_to_clipboard
                copy_to_clipboard(text)
                self.note_status_label.configure(text="Copied to clipboard", text_color=SUCCESS)
        except Exception:
            pass

    def _on_note_save(self):
        try:
            text = self.note_text.get("1.0", "end").strip()

            if not text or text == "Transcription will appear here as you speak...":
                self.note_status_label.configure(text="Nothing to save", text_color=WARNING)
                return

            from tkinter import filedialog
            import datetime
            default_name = f"note_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
            path = filedialog.asksaveasfilename(
                title="Save Note",
                defaultextension=".txt",
                initialfile=default_name,
                filetypes=[("Text files", "*.txt"), ("All files", "*.*")])
            if path:
                with open(path, "w", encoding="utf-8") as f:
                    f.write(text)
                self._note_dirty = False
                self.note_status_label.configure(
                    text=f"Saved: {Path(path).name}", text_color=SUCCESS)
        except Exception as e:
            self.note_status_label.configure(text=f"Save error: {e}", text_color=DANGER)

    def note_has_unsaved(self) -> bool:
        """True when the box holds real, unsaved note content."""
        try:
            if not bool(getattr(self, "_note_dirty", False)):
                return False
            text = self.note_text.get("1.0", "end").strip()
            return bool(text) and text != "Transcription will appear here as you speak..."
        except Exception:
            return False

    def confirm_note_close(self) -> bool:
        """True = proceed with close. Asks once (Save / discard / cancel)
        only when unsaved note content exists; otherwise returns True
        untouched. Never raises (close must never brick)."""
        try:
            if not self.note_has_unsaved():
                return True
            from tkinter import messagebox as _mb
            r = _mb.askyesnocancel(
                "Save note?",
                "You have an unsaved note.\nSave it before closing?",
                parent=self)
        except Exception:
            return True
        if r is None:
            return False
        if r is True:
            try:
                self._on_note_save()
            except Exception:
                return False
            # Save clears dirty only on success; a cancelled file dialog
            # keeps the text, so closing aborts (safe direction).
            try:
                return not self.note_has_unsaved()
            except Exception:
                return False
        return True

    def _on_note_save_audio(self):
        """Save Audio button: copy the last auto-saved session MP3 to a
        user-chosen location (marks it kept for close-cleanup)."""
        try:
            from pathlib import Path as _P
            from tkinter import filedialog as _fd
            from tkinter import messagebox as _mb
            try:
                cands = sorted(
                    (str(p) for p in (getattr(self, "_note_auto_mp3s", set())
                                      or set())),
                    reverse=True)
            except Exception:
                cands = []
            src = None
            for c in cands:
                try:
                    if _P(c).exists() and _P(c).stat().st_size > 0:
                        src = _P(c)
                        break
                except Exception:
                    continue
            if src is None:
                try:
                    self.note_status_label.configure(
                        text="No audio saved yet — record a note first",
                        text_color=WARNING)
                except Exception:
                    pass
                return
            try:
                dest = _fd.asksaveasfilename(
                    title="Save note audio",
                    defaultextension=".mp3",
                    initialfile=src.name,
                    filetypes=[("MP3 audio", "*.mp3"),
                               ("All files", "*.*")],
                    parent=self)
            except Exception:
                return
            if not dest:
                return
            try:
                import shutil as _sh
                _sh.copy2(str(src), str(dest))
            except Exception as e:
                try:
                    _mb.showwarning("Save Audio", f"Could not save: {e}",
                                    parent=self)
                except Exception:
                    pass
                return
            try:
                self._note_saved_mp3s.add(str(src))
            except Exception:
                pass
            try:
                self.note_status_label.configure(
                    text=f"Audio saved to: {_P(dest).name}", text_color=SUCCESS)
            except Exception:
                pass
        except Exception:
            pass

    def cleanup_note_audio(self):
        """Close-time sweep: stop any live Note capture, then delete this
        run's auto session files (WAV leftovers + MP3s) UNLESS explicitly
        kept via Save Audio. Previous runs' files are never touched.
        Never raises (close must never brick)."""
        try:
            try:
                if bool(getattr(self, "_note_recording", False)):
                    try:
                        self._note_recording = False
                    except Exception:
                        pass
                    try:
                        rec = getattr(self, "_note_recorder", None)
                        if rec is not None:
                            rec.stop()
                    except Exception:
                        pass
            except Exception:
                pass
            try:
                auto = set(getattr(self, "_note_auto_mp3s", set()) or set())
                saved = set(getattr(self, "_note_saved_mp3s", set()) or set())
            except Exception:
                auto, saved = set(), set()
            try:
                rec = getattr(self, "_note_recorder", None)
                sw = getattr(rec, "session_wav", None) if rec else None
                if sw:
                    auto.add(str(sw))
            except Exception:
                pass
            from pathlib import Path as _P
            for p in auto:
                try:
                    if p in saved:
                        continue
                    _f = _P(str(p))
                    if _f.exists():
                        _f.unlink()
                except Exception:
                    continue
        except Exception:
            pass


    def _on_tab_changed(self, value=None):
        try:
            name = self.tabs.get()
        except Exception:
            name = value
        if self._tab_callback and name in ("Live", "SRT File", "Note"):
            try:
                self._tab_callback(name)
            except Exception:
                pass
    def set_tab_callback(self, cb: Callable):
        self._tab_callback = cb if callable(cb) else None
    def set_active_tab(self, name: str):
        try:
            if name in ("Live", "SRT File"):
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
                    "note_engine_menu", "note_model_menu"):
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
    def _toggle_record(self):
        if self._is_recording:
            self._is_recording = False
            if self._on_record_stop:
                threading.Thread(target=self._on_record_stop, daemon=True).start()
        else:
            self._is_recording = True
            if self._on_record_start:
                threading.Thread(target=self._on_record_start, daemon=True).start()
    def _clear_history(self):
        self.history.clear()
        self.output_text.configure(state="normal")
        self.output_text.delete("1.0", "end")
        self.output_text.insert("1.0", "Cleared.")
        self.output_text.configure(state="disabled")
    def _copy_last(self):
        from input_sim import copy_to_clipboard
        self.output_text.configure(state="normal")
        text = self.output_text.get("1.0", "end").strip()
        self.output_text.configure(state="disabled")
        if text and text not in ("Waiting for voice input...", "Cleared."):
            copy_to_clipboard(text)
    def set_recording_state(self, recording: bool):
        self._is_recording = recording
        if recording:
            self.record_btn.configure(text="\u25A0  STOP", fg_color=DANGER,
                                      hover_color=BTN_DANGER_HOVER)
            self.status_badge.set_status("Recording", DANGER)
            self.waveform.start_animation()
        else:
            self.record_btn.configure(text="\u25CF  RECORD", fg_color=ACCENT,
                                      hover_color=ACCENT_DARK)
            self.waveform.stop_animation()
            self.meter.set_level(0)
    def set_status(self, text: str, color: str = FG_SECONDARY):
        self.status_badge.set_status(text, color)
    def set_transcription(self, text: str):
        self.output_text.configure(state="normal")
        self.output_text.delete("1.0", "end")
        self.output_text.insert("1.0", text if text else "(no speech detected)")
        self.output_text.configure(state="disabled")
        if text and text.strip():
            import datetime
            ts = datetime.datetime.now().strftime("%H:%M:%S")
            self.history.add_entry(text, ts)
    def update_level(self, level: float):
        self.meter.set_level(level)
        self.waveform.set_amplitude(level)
    def _on_method_changed(self, value):
        if self._method_callback:
            self._method_callback(value)
    def _on_suffix_changed(self, value):
        if self._suffix_callback:
            self._suffix_callback(value)
    def _on_model_changed(self, value):
        try:
            self._refresh_srt_engine_label()
        except Exception:
            pass
        if self._model_callback:
            try:
                base = (self._model_value_map or {}).get(value, value)
            except Exception:
                base = value
            self._model_callback(base)
    def _refresh_srt_engine_label(self):
        """Legacy entry point (task/lang changes used to rewrite a summary
        label). The label is now real Engine/Model menus, so this just keeps
        the model menu's options in step with the engine menu."""
        try:
            self._refresh_srt_model_menu()
        except Exception:
            pass

    def _srt_model_options(self, kind):
        """Display labels valid for an engine kind."""
        try:
            if kind == "Whisper":
                return list(WHISPER_MODEL_CHOICES.keys())
            if kind == "Canary-1B":
                return [CANARY_MODEL_LABEL]
            return list(MODEL_CHOICES.keys())
        except Exception:
            return []

    def _refresh_srt_model_menu(self):
        try:
            kind = (self.srt_engine_var.get() or "Moonshine v2").strip()
        except Exception:
            kind = "Moonshine v2"
        try:
            vals = self._srt_model_options(kind)
            if vals:
                cur = None
                try:
                    cur = self.srt_model_var.get()
                except Exception:
                    cur = None
                self.srt_model_menu.configure(values=vals)
                if cur in vals:
                    self.srt_model_var.set(cur)
                else:
                    self.srt_model_var.set(vals[0])
        except Exception:
            pass

    def _srt_menu_label(self, kind, arch=None, wmid=None):
        """Display model label for ids (restore/mirror path)."""
        try:
            if kind == "Whisper":
                return WHISPER_MODEL_CHOICES_REV.get(str(wmid or "large-v3"))
            if kind == "Canary-1B":
                return CANARY_MODEL_LABEL
            return MODEL_CHOICES_REV.get(int(arch if arch is not None else 5))
        except Exception:
            return None

    def set_srt_engine_state(self, kind, arch=None, wmid=None):
        """Programmatic set (mirror/adopt/restore). No callbacks fired."""
        try:
            if kind in ENGINE_CHOICES:
                self.srt_engine_var.set(kind)
        except Exception:
            pass
        try:
            self._refresh_srt_model_menu()
        except Exception:
            pass
        try:
            label = self._srt_menu_label(kind, arch, wmid)
            if label:
                try:
                    vals = list(self.srt_model_menu.cget("values"))
                except Exception:
                    vals = []
                if label in vals:
                    self.srt_model_var.set(label)
        except Exception:
            pass

    def get_srt_engine_kind(self):
        try:
            k = (self.srt_engine_var.get() or "Moonshine v2").strip()
        except Exception:
            k = "Moonshine v2"
        return k if k in ENGINE_CHOICES else "Moonshine v2"

    def set_srt_engine_callbacks(self, eng_cb, model_cb):
        self._srt_engine_cb = eng_cb if callable(eng_cb) else None
        self._srt_model_cb = model_cb if callable(model_cb) else None

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
        """App plan/apply/getter for per-tab engine picks (SRT + Note menus
        share this one flow). All None-safe."""
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
        """Shared SRT/Note menu-pick flow: plan -> optional dual dialog ->
        apply -> sync self + mirrored tabs. GUI thread only."""
        try:
            plan_cb = getattr(self, "_tab_plan_cb", None)
            apply_cb = getattr(self, "_tab_apply_cb", None)
            if plan_cb is None or apply_cb is None:
                return
            kind, arch, wmid = self._translate_engine_pick(
                tab, kind_display, model_display)
            try:
                v = plan_cb(tab, kind, arch, wmid) or {}
            except Exception:
                return
            act = (v or {}).get("action")
            if act == "confirm":
                try:
                    from tkinter import messagebox as _mb
                    yes = bool(_mb.askyesno(
                        "Load second engine?",
                        str(v.get("message") or "Another session is active."),
                        parent=self))
                except Exception:
                    yes = False
                try:
                    r = apply_cb(tab, v.get("kind"), v.get("arch"),
                                 v.get("wmodel"), bool(yes))
                except Exception:
                    r = {"action": "revert"}
                if not yes or (r or {}).get("action") != "ok":
                    self._revert_tab_menus(tab)
                    return
            elif act == "ok":
                try:
                    r = apply_cb(tab, v.get("kind"), v.get("arch"),
                                 v.get("wmodel"), True)
                except Exception:
                    r = {"action": "revert"}
                if (r or {}).get("action") != "ok":
                    self._revert_tab_menus(tab)
                    return
            else:
                self._revert_tab_menus(tab)
                return
            try:
                self._sync_tab_menus(tab, v.get("kind"), v.get("arch"),
                                     v.get("wmodel"))
                for (t, k, a, w) in ((r or {}).get("mirrored") or []):
                    if t != tab:
                        self._sync_tab_menus(t, k, a, w)
            except Exception:
                pass
        except Exception:
            pass

    def _on_srt_engine_changed(self, value):
        try:
            self._refresh_srt_model_menu()
        except Exception:
            pass
        try:
            self._request_tab_engine("srt", kind_display=value)
        except Exception:
            pass

    def _on_srt_model_changed(self, value):
        try:
            self._request_tab_engine("srt", model_display=value)
        except Exception:
            pass

    def _on_note_engine_changed(self, value):
        try:
            self._refresh_note_model_menu()
        except Exception:
            pass
        try:
            self._request_tab_engine("note", kind_display=value)
        except Exception:
            pass

    def _on_note_model_changed(self, value):
        try:
            self._request_tab_engine("note", model_display=value)
        except Exception:
            pass

    def _note_model_options(self, kind):
        try:
            if kind == "Whisper":
                return list(WHISPER_MODEL_CHOICES.keys())
            if kind == "Canary-1B":
                return [CANARY_MODEL_LABEL]
            return list(MODEL_CHOICES.keys())
        except Exception:
            return []

    def _refresh_note_model_menu(self):
        try:
            kind = (self.note_engine_var.get() or "Moonshine v2").strip()
        except Exception:
            kind = "Moonshine v2"
        try:
            vals = self._note_model_options(kind)
            if vals:
                cur = None
                try:
                    cur = self.note_model_var.get()
                except Exception:
                    cur = None
                self.note_model_menu.configure(values=vals)
                if cur in vals:
                    self.note_model_var.set(cur)
                else:
                    self.note_model_var.set(vals[0])
        except Exception:
            pass

    def _note_menu_label(self, kind, arch=None, wmid=None):
        try:
            if kind == "Whisper":
                return WHISPER_MODEL_CHOICES_REV.get(str(wmid or "large-v3"))
            if kind == "Canary-1B":
                return CANARY_MODEL_LABEL
            return MODEL_CHOICES_REV.get(int(arch if arch is not None else 5))
        except Exception:
            return None

    def set_note_engine_state(self, kind, arch=None, wmid=None):
        """Programmatic set (mirror/adopt/restore). No callbacks fired."""
        try:
            if kind in ENGINE_CHOICES:
                self.note_engine_var.set(kind)
        except Exception:
            pass
        try:
            self._refresh_note_model_menu()
        except Exception:
            pass
        try:
            label = self._note_menu_label(kind, arch, wmid)
            if label:
                try:
                    vals = list(self.note_model_menu.cget("values"))
                except Exception:
                    vals = []
                if label in vals:
                    self.note_model_var.set(label)
        except Exception:
            pass
        try:
            self.note_engine_label.configure(
                text=f"Engine: {kind}" + (f" ({label})" if label else ""))
        except Exception:
            pass

    def get_note_engine_kind(self):
        try:
            k = (self.note_engine_var.get() or "Moonshine v2").strip()
        except Exception:
            k = "Moonshine v2"
        return k if k in ENGINE_CHOICES else "Moonshine v2"

    def note_session_active(self):
        """True while Note records or transcriptions are outstanding."""
        try:
            if bool(getattr(self, "_note_recording", False)):
                return True
            sub = int(getattr(self, "_note_submitted", 0) or 0)
            done = int(getattr(self, "_note_done", 0) or 0)
            return sub > done
        except Exception:
            return False

    def _on_engine_changed(self, value):
        self._refresh_srt_engine_label()
        is_canary = value == "Canary-1B"
        is_whisper = value == "Whisper"
        try:
            self.model_menu.configure(state=self._model_menu_state(value))
            self.canary_task_menu.configure(
                state="normal" if is_canary or is_whisper else "disabled")
            self.canary_lang_menu.configure(
                state="normal" if is_canary or is_whisper else "disabled")
        except Exception:
            pass
        if self._engine_callback:
            self._engine_callback(value)
        try:
            self.refresh_lang_options(value)
        except Exception:
            pass
    def _on_canary_task_changed(self, value):
        self._refresh_srt_engine_label()
        if self._canary_task_callback:
            self._canary_task_callback(value)
    def _on_canary_lang_changed(self, value):
        self._refresh_srt_engine_label()
        if self._canary_lang_callback:
            self._canary_lang_callback(value)
    def set_settings(self, method: str, suffix: str, method_cb: Callable, suffix_cb: Callable):
        self._method_callback = method_cb if callable(method_cb) else None
        self._suffix_callback = suffix_cb if callable(suffix_cb) else None
        self.method_var.set(method if method in ["clipboard", "unicode"] else "clipboard")
        self.suffix_var.set(suffix if suffix in ["none", "space", "newline", "period_space"] else "none")
    def set_model(self, arch: int, callback: Callable):
        self.set_model_options(
            list(MODEL_CHOICES.keys()),
            MODEL_CHOICES_REV.get(arch, "Medium Streaming (110MB, best)"),
            callback)
    @staticmethod
    def _model_menu_state(engine_kind: str) -> str:
        return ("normal" if engine_kind in ("Moonshine v2", "Whisper")
                else "disabled")
    def set_model_options(self, values, current: str, callback: Callable,
                          status_fn: Optional[Callable] = None):
        self._model_callback = callback if callable(callback) else None
        displays, mapping = apply_badge_map(list(values or []), status_fn)
        self._model_value_map = mapping
        try:
            self.model_menu.configure(values=displays)
        except Exception:
            pass
        try:
            want = None
            for d, b in mapping.items():
                if b == current:
                    want = d
                    break
            self.model_var.set(want if want is not None else (displays[0] if displays else ""))
        except Exception:
            pass
    def set_model_title(self, engine_name: str):
        try:
            self.model_title_label.configure(text=f"Model ({engine_name}):")
        except Exception:
            pass
    def set_model_manage_callback(self, cb: Callable):
        self._model_manage_cb = cb if callable(cb) else None
    def _on_model_manage(self):
        if self._model_manage_cb:
            try:
                self._model_manage_cb()
            except Exception:
                pass
    def show_model_manager(self, get_data, on_delete, on_delete_all):
        try:
            from tkinter import messagebox as _mb
        except Exception:
            _mb = None
        try:
            _old = getattr(self, "_model_manager_win", None)
            if _old is not None:
                try:
                    if _old.winfo_exists():
                        _old.lift()
                        try:
                            _old.focus_force()
                        except Exception:
                            pass
                        return
                except Exception:
                    pass
        except Exception:
            pass
        win = ctk.CTkToplevel(self)
        self._model_manager_win = win
        try:
            win.title("Model storage")
            win.geometry("540x480")
            win.minsize(480, 380)
            win.configure(fg_color=BG_DARK)
            try:
                win.transient(self)
            except Exception:
                pass
        except Exception:
            pass
        header = ctk.CTkFrame(win, fg_color="transparent")
        header.pack(fill="x", padx=16, pady=(14, 4))
        ctk.CTkLabel(header, text="Downloaded models",
                     font=("Segoe UI", 15, "bold"), text_color=FG_PRIMARY
                     ).pack(side="left")
        total_lbl = ctk.CTkLabel(header, text="", font=("Segoe UI", 11),
                                 text_color=FG_DIM)
        total_lbl.pack(side="right")
        ctk.CTkLabel(win, text="✓ downloaded   ↓ downloads on first pick",
                     font=("Segoe UI", 10), text_color=FG_DIM
                     ).pack(anchor="w", padx=16)
        body = ctk.CTkScrollableFrame(win, fg_color="transparent")
        body.pack(fill="both", expand=True, padx=12, pady=4)
        body.grid_columnconfigure(0, weight=1)
        try:
            self._tune_scrollers(win)
        except Exception:
            pass
        foot = ctk.CTkFrame(win, fg_color="transparent")
        foot.pack(fill="x", padx=16, pady=(4, 14))
        foot.grid_columnconfigure(0, weight=1)
        foot.grid_columnconfigure(1, weight=1)
        def _gb(n):
            try:
                v = float(n or 0)
            except Exception:
                return "—"
            if v >= 1e9:
                return f"{v / 1e9:.2f} GB"
            if v >= 1e6:
                return f"{v / 1e6:.0f} MB"
            if v > 0:
                return f"{int(v)} B"
            return "—"
        def _rebuild():
            try:
                for child in body.winfo_children():
                    child.destroy()
            except Exception:
                pass
            try:
                data = get_data() or {}
            except Exception:
                data = {}
            items = (data or {}).get("items", []) or []
            try:
                total_lbl.configure(
                    text=f"on disk: {_gb((data or {}).get('total', 0))}")
            except Exception:
                pass
            if not items:
                ctk.CTkLabel(body, text="No downloadable models found.",
                             font=("Segoe UI", 11), text_color=FG_DIM
                             ).grid(row=0, column=0, sticky="w", padx=8, pady=8)
                return
            last_eng = None
            row = 0
            for it in items:
                try:
                    eng = it.get("engine", "")
                    if eng != last_eng:
                        last_eng = eng
                        ctk.CTkLabel(body, text=eng,
                                     font=("Segoe UI", 11, "bold"),
                                     text_color=ACCENT_GLOW
                                     ).grid(row=row, column=0, columnspan=3,
                                            sticky="w", padx=8, pady=(8, 2))
                        row += 1
                    name = str(it.get("label", it.get("id", "?")))
                    if it.get("downloaded"):
                        sub = f"{_gb(it.get('size'))} downloaded"
                    else:
                        sub = "not downloaded"
                    if it.get("in_use"):
                        sub += " • in use"
                    card = ctk.CTkFrame(body, fg_color=BG_CARD, corner_radius=8)
                    card.grid(row=row, column=0, columnspan=3, sticky="ew",
                              padx=4, pady=3)
                    card.grid_columnconfigure(0, weight=1)
                    ctk.CTkLabel(card, text=name, font=("Segoe UI", 11),
                                 text_color=FG_PRIMARY
                                 ).grid(row=0, column=0, sticky="w", padx=10, pady=(6, 0))
                    ctk.CTkLabel(card, text=sub, font=("Segoe UI", 10),
                                 text_color=FG_DIM
                                 ).grid(row=1, column=0, sticky="w", padx=10, pady=(0, 6))
                    can_del = bool(it.get("downloaded")) and not bool(it.get("in_use"))
                    btn = ctk.CTkButton(
                        card, text="Delete", width=80, height=30,
                        font=("Segoe UI", 11),
                        fg_color=DANGER if can_del else BTN_DIM,
                        hover_color=BTN_DANGER_HOVER if can_del else BTN_DIM,
                        text_color=FG_PRIMARY if can_del else FG_DIM,
                        state="normal" if can_del else "disabled",
                        command=lambda e=eng, k=it.get("kind"),
                        i=it.get("id"), n=name,
                        s=it.get("size"): _ask_delete(e, k, i, n, s))
                    btn.grid(row=0, column=1, rowspan=2, padx=10, pady=6, sticky="e")
                    row += 1
                except Exception:
                    continue
        def _ask_delete(eng, kind, ident, name, size):
            if _mb is None:
                return
            try:
                if not _mb.askyesno(
                        "Delete model",
                        f"Delete {name} ({_gb(size)}) from disk?\n"
                        f"It re-downloads automatically next time you pick it."):
                    return
            except Exception:
                return
            try:
                ok, msg = on_delete(eng, kind, ident)
            except Exception as ex:
                ok, msg = False, str(ex)
            try:
                if _mb is not None:
                    if ok:
                        _mb.showinfo("Model storage", msg or "Deleted.")
                    else:
                        _mb.showwarning("Model storage", msg or "Could not delete.")
            except Exception:
                pass
            _rebuild()
        def _ask_delete_all():
            if _mb is None:
                return
            try:
                data = get_data() or {}
                n = sum(1 for it in (data.get("items") or [])
                        if it.get("downloaded") and not it.get("in_use"))
                if n <= 0:
                    _mb.showinfo("Model storage", "Nothing deletable right now.")
                    return
                if not _mb.askyesno(
                        "Delete all",
                        f"Delete all {n} downloaded model(s) not in use?\n"
                        f"They re-download automatically when picked again."):
                    return
            except Exception:
                return
            try:
                ok, msg = on_delete_all()
            except Exception as ex:
                ok, msg = False, str(ex)
            try:
                if _mb is not None:
                    if ok:
                        _mb.showinfo("Model storage", msg or "Deleted.")
                    else:
                        _mb.showwarning("Model storage", msg or "Could not delete.")
            except Exception:
                pass
            _rebuild()
        ctk.CTkButton(foot, text="Delete all downloaded", font=("Segoe UI", 12),
                      fg_color=DANGER, hover_color=BTN_DANGER_HOVER, height=38,
                      text_color=BTN_TEXT,
                      corner_radius=8, command=_ask_delete_all
                      ).grid(row=0, column=0, padx=(0, 4), sticky="ew")
        ctk.CTkButton(foot, text="Close", font=("Segoe UI", 12),
                      fg_color=BTN_DIM, hover_color=BTN_DIM_HOVER, height=38,
                      corner_radius=8, text_color=FG_SECONDARY,
                      command=win.destroy
                      ).grid(row=0, column=1, padx=(4, 0), sticky="ew")
        _rebuild()
        try:
            win.lift()
            win.focus_force()
        except Exception:
            pass
    def set_engine(self, engine: str, task: str, src_lang: str, engine_cb: Callable, task_cb: Callable, lang_cb: Callable):
        self._engine_callback = engine_cb if callable(engine_cb) else None
        self._canary_task_callback = task_cb if callable(task_cb) else None
        self._canary_lang_callback = lang_cb if callable(lang_cb) else None
        self.engine_var.set(engine if engine in ENGINE_CHOICES else "Moonshine v2")
        self.canary_task_var.set(task if task in CANARY_TASKS else "transcribe")
        self.canary_lang_var.set(src_lang if src_lang in CANARY_LANGS else "auto")
        is_heavy = (engine in ("Canary-1B", "Whisper"))
        try:
            self.model_menu.configure(state=self._model_menu_state(engine))
            self.canary_task_menu.configure(state="normal" if is_heavy else "disabled")
            self.canary_lang_menu.configure(state="normal" if is_heavy else "disabled")
        except Exception:
            pass
        self._refresh_srt_engine_label()
        try:
            self.refresh_lang_options(engine)
        except Exception:
            pass
    def set_model_status(self, text: str, color: str = WARNING):
        self.set_status(text, color)
    def set_record_callback(self, on_start: Callable, on_stop: Callable):
        self._on_record_start = on_start if callable(on_start) else None
        self._on_record_stop = on_stop if callable(on_stop) else None
    def _enable_drop(self, widget) -> bool:
        try:
            from tkinterdnd2 import DND_FILES
            import tkinterdnd2.TkinterDnD as _dndmod
            _dndmod._require(widget)
            _dndmod.DnDWrapper.drop_target_register(widget, DND_FILES)
            _dndmod.DnDWrapper.dnd_bind(widget, "<<Drop>>", self._on_drop_files)
            return True
        except Exception:
            return False
    def _parse_drop(self, data: str):
        import re as _re
        data = (data or "").strip()
        if not data:
            return []
        parts = _re.findall(r"\{([^}]+)\}|(\S+)", data)
        out = []
        for a, b in parts:
            p = (a or b).strip()
            if p:
                out.append(p)
        return out
    def _on_drop_files(self, event):
        try:
            files = self._parse_drop(event.data)
            if files:
                self.add_srt_files(files)
        except Exception:
            pass
    def _srt_browse_file(self):
        try:
            from tkinter import filedialog
            from srt import SUPPORTED_EXTS
            exts = " ".join(f"*{e}" for e in SUPPORTED_EXTS)
            paths = filedialog.askopenfilenames(
                title="Pick video or audio files (multi-select for batch)",
                filetypes=[("Media", exts), ("All files", "*.*")])
            if paths:
                self.add_srt_files(list(paths))
        except Exception:
            pass
    def add_srt_files(self, paths) -> int:
        try:
            from srt import SUPPORTED_EXTS as _exts
        except Exception:
            _exts = ()
        added = 0
        try:
            import os as _os
            if self._srt_input_paths and not getattr(self, "_srt_running", False):
                all_done = all(
                    str(self._srt_file_status.get(i, "")).startswith(
                        ("\u2713", "\u2717", "\u2013"))
                    for i in range(len(self._srt_input_paths))
                )
                if all_done:
                    self._srt_input_paths = []
                    self._srt_file_status = {}
            try:
                seen = {_os.path.normcase(p) for p in self._srt_input_paths}
            except Exception:
                seen = set()
            for raw in (paths or []):
                p = str(raw or "").strip().strip('"')
                if not p:
                    continue
                if _exts and _os.path.splitext(p)[1].lower() not in _exts:
                    try:
                        self.srt_log(f"skip (unsupported type): {_os.path.basename(p)}")
                    except Exception:
                        pass
                    continue
                if _os.path.normcase(p) in seen:
                    try:
                        self.srt_log(f"skip (already queued): {_os.path.basename(p)}")
                    except Exception:
                        pass
                    continue
                seen.add(_os.path.normcase(p))
                self._srt_input_paths.append(p)
                self._srt_file_status[len(self._srt_input_paths) - 1] = "queued"
                added += 1
            self._refresh_srt_list()
            if added and not getattr(self, "_srt_running", False):
                try:
                    last = self._srt_input_paths[-1]
                    self.set_srt_progress(
                        0, f"Ready: {added} file(s) - latest: {_os.path.basename(last)}")
                except Exception:
                    pass
        except Exception:
            pass
        try:
            self.update_action_states()
        except Exception:
            pass
        return added

    def _refresh_srt_list(self):
        try:
            self.srt_file_list.delete(0, "end")
        except Exception:
            return
        try:
            import os as _os
            for i, p in enumerate(self._srt_input_paths):
                st = self._srt_file_status.get(i, "queued")
                self.srt_file_list.insert("end", f"{i + 1}. {_os.path.basename(p)}  [{st}]")
            n = len(self._srt_input_paths)
            self.srt_drop.configure(state="normal")
            self.srt_drop.delete("1.0", "end")
            if n:
                self.srt_drop.insert(
                    "1.0", f"{n} file(s) queued - runs one by one.\n"
                    "Drop more files or Browse Files... to add.")
            else:
                self.srt_drop.insert(
                    "1.0", "Drag & drop video/audio files here,\nor click Browse Files...")
            self.srt_drop.configure(state="disabled")
        except Exception:
            pass
    def get_srt_input_paths(self) -> list:
        return [p for p in list(self._srt_input_paths or []) if p]
    def _srt_out_dir(self) -> str:
        try:
            return self.srt_out_entry.get().strip()
        except Exception:
            return ""
    @staticmethod
    def _safe_exists(fn, path_obj, out_dir) -> bool:
        try:
            return bool(fn(path_obj, out_dir).exists())
        except Exception:
            return False
    def _srt_existing_outputs(self, kind: str):
        found = {}
        try:
            from pathlib import Path as _P
            if kind == "burn":
                from srt import default_burn_path as _fn
            else:
                from srt import default_out_path as _fn
            out_dir = self._srt_out_dir()
            for i, p in enumerate(list(self._srt_input_paths or [])):
                try:
                    if _fn(_P(p), out_dir).exists():
                        found[i] = str(_fn(_P(p), out_dir))
                except Exception:
                    continue
        except Exception:
            pass
        return found
    def _show_srt_action(self, mode: str):
        try:
            gen, brn = self.srt_start_btn, self.srt_burn_btn
        except Exception:
            return
        try:
            if mode == "burn":
                try:
                    gen.grid_remove()
                except Exception:
                    pass
                try:
                    brn.configure(text="\u25B6 Convert to MP4")
                except Exception:
                    pass
                brn.grid(row=0, column=0, padx=(0, 4), sticky="ew")
            else:
                try:
                    brn.grid_remove()
                except Exception:
                    pass
                try:
                    gen.configure(text="\u25B6  Generate SRT")
                except Exception:
                    pass
                gen.grid(row=0, column=0, padx=(0, 4), sticky="ew")
            self._srt_mode = mode
        except Exception:
            pass
    def _set_burn_after_enabled(self, enabled: bool):
        try:
            self.burn_after_check.configure(state="normal" if enabled else "disabled")
        except Exception:
            pass
    def get_srt_options(self) -> dict:
        try:
            norm = bool(self.srt_norm_var.get())
        except Exception:
            norm = False
        try:
            after = bool(self.burn_after_var.get())
        except Exception:
            after = False
        try:
            sd = bool(self.shutdown_var.get())
        except Exception:
            sd = False
        try:
            al = bool(self.alert_var.get())
        except Exception:
            al = True
        return {"srt_norm": norm, "burn_after": after,
                "auto_shutdown": sd, "completion_alert": al}
    def set_srt_options(self, opts=None, callback: Optional[Callable] = None):
        o = opts or {}
        for var_name, key, dflt in (("srt_norm_var", "srt_norm", False),
                                    ("burn_after_var", "burn_after", False),
                                    ("shutdown_var", "auto_shutdown", False),
                                    ("alert_var", "completion_alert", True)):
            try:
                var = getattr(self, var_name, None)
                if var is not None:
                    var.set(bool(o.get(key, dflt)))
            except Exception:
                pass
        self._srt_opt_callback = callback if callable(callback) else None
    def _on_srt_opt_toggled(self):
        cb = getattr(self, "_srt_opt_callback", None)
        if cb:
            try:
                cb(self.get_srt_options())
            except Exception:
                pass
    def notify_completion(self, title: str, msg: str):
        try:
            self.lift()
        except Exception:
            pass
        try:
            self.attributes("-topmost", True)
        except Exception:
            pass
        try:
            from tkinter import messagebox as _mb
            _mb.showinfo(title or "Done", msg or "Finished.", parent=self)
        except Exception:
            pass
        try:
            self.attributes("-topmost", False)
        except Exception:
            pass
        try:
            self.focus_force()
        except Exception:
            pass
    def update_action_states(self, touch_progress: bool = True):
        if getattr(self, "_srt_running", False):
            return
        try:
            paths = self.get_srt_input_paths()
            try:
                _sig = (tuple(paths), self._srt_out_dir())
            except Exception:
                _sig = None
            if _sig != getattr(self, "_burn_est_sig", None):
                self._burn_est_sig = _sig
                try:
                    self._refresh_burn_est()
                except Exception:
                    pass
            if not paths:
                self._show_srt_action("generate")
                try:
                    self.srt_start_btn.configure(state="normal")
                except Exception:
                    pass
                self._set_burn_after_enabled(True)
                return
            existing = self._srt_existing_outputs("srt")
            all_done = len(existing) >= len(paths)
            any_done = len(existing) > 0
            if all_done:
                self._show_srt_action("burn")
                try:
                    self.srt_burn_btn.configure(state="normal")
                except Exception:
                    pass
                self._set_burn_after_enabled(False)
            else:
                self._show_srt_action("generate")
                try:
                    self.srt_start_btn.configure(state="normal")
                except Exception:
                    pass
                self._set_burn_after_enabled(True)
            if not touch_progress:
                return
            if all_done:
                self.set_srt_progress(
                    0, "All queued files already have SRTs - Convert to MP4")
            elif not any_done:
                try:
                    cur = self.srt_status.cget("text")
                except Exception:
                    cur = ""
                if "already have SRTs" in (cur or ""):
                    self.set_srt_progress(0, "Ready - Generate SRT first")
        except Exception:
            pass
    def set_srt_file_status(self, index: int, status: str):
        try:
            self._srt_file_status[int(index)] = str(status or "")
            self._refresh_srt_list()
            try:
                self.srt_file_list.see(int(index))
            except Exception:
                pass
        except Exception:
            pass
    def _srt_clear_file(self):
        if getattr(self, "_srt_running", False):
            return
        self._srt_input_paths = []
        self._srt_file_status = {}
        self._refresh_srt_list()
        try:
            self.update_action_states()
        except Exception:
            pass
        try:
            self.srt_bar.set(0)
            self.srt_pct.configure(text="0%")
            self.srt_status.configure(text="Idle - pick a file to begin")
        except Exception:
            pass

    def _srt_browse_outdir(self):
        try:
            from tkinter import filedialog
            d = filedialog.askdirectory(title="SRT output folder (Cancel = same as video)")
            if d:
                self.srt_out_entry.delete(0, "end")
                self.srt_out_entry.insert(0, d)
                try:
                    self.update_action_states()
                except Exception:
                    pass
        except Exception:
            pass
    def _on_compute_changed(self, value):
        if self._compute_callback:
            try:
                self._compute_callback(value)
            except Exception:
                pass
    def set_compute(self, code: str, callback: Callable):
        self._compute_callback = callback if callable(callback) else None
        try:
            vals = list(self.compute_menu.cget("values"))
        except Exception:
            vals = ["Auto", "CPU", "GPU"]
        inv = {"auto": "Auto", "cpu": "CPU", "gpu": "GPU"}
        want = inv.get((code or "auto").strip().lower(), "Auto")
        if want not in vals:
            want = vals[0] if vals else "Auto"
        try:
            self.compute_var.set(want)
        except Exception:
            pass
    def _on_srt_cpu_changed(self, value):
        try:
            n = max(1, int(round(float(value))))
        except Exception:
            n = 1
        pct = int(round(n / max(1, self._srt_max_cpu) * 100))
        self.srt_cpu_value.configure(text=f"{n} threads ({pct}%)")
    def _on_burn_fontsize_changed(self, value):
        try:
            n = max(12, min(32, int(round(float(value)))))
        except Exception:
            n = 18
        try:
            self.burn_font_value.configure(text=f"{n}")
        except Exception:
            pass
    def _slider_kbps(self) -> int:
        try:
            return max(300, min(10000, int(round(float(self.burn_vbr_slider.get()) / 100.0)) * 100))
        except Exception:
            return 2000
    def _update_burn_vbr_display(self):
        try:
            auto = bool(self.burn_vbr_auto_var.get())
        except Exception:
            auto = True
        try:
            if auto:
                text = "auto"
            else:
                try:
                    exact = int(getattr(self, "_burn_vbr_exact", 0) or 0)
                except Exception:
                    exact = 0
                text = f"{exact}k" if exact >= 300 else f"{self._slider_kbps()}k"
            self.burn_vbr_value.configure(text=text)
        except Exception:
            pass
    def _on_burn_vbr_changed(self, value):
        try:
            self._burn_vbr_exact = None
        except Exception:
            pass
        try:
            self._update_burn_vbr_display()
        except Exception:
            pass
        try:
            self._refresh_burn_est()
        except Exception:
            pass
    def _on_burn_vbr_toggled(self):
        try:
            auto = bool(self.burn_vbr_auto_var.get())
        except Exception:
            auto = True
        try:
            self.burn_vbr_slider.configure(state="disabled" if auto else "normal")
            if auto:
                self.burn_vbr_value.configure(text="auto")
            else:
                self._on_burn_vbr_changed(self.burn_vbr_slider.get())
        except Exception:
            pass
        try:
            self._refresh_burn_est()
        except Exception:
            pass
    def get_burn_vbr(self):
        try:
            auto = bool(self.burn_vbr_auto_var.get())
        except Exception:
            auto = True
        if auto:
            return True, 0
        try:
            exact = int(getattr(self, "_burn_vbr_exact", 0) or 0)
        except Exception:
            exact = 0
        if exact >= 300:
            return False, min(10000, exact)
        return False, self._slider_kbps()
    def get_burn_codec(self) -> str:
        try:
            v = (self.burn_codec_var.get() or "").strip().lower()
        except Exception:
            return "h264"
        return "hevc" if ("hevc" in v or v == "h265") else "h264"
    def set_burn_codec(self, code: str):
        try:
            want = "HEVC (H.265)" if str(code or "").strip().lower() == "hevc" else "H.264"
            vals = list(self.burn_codec_menu.cget("values"))
            if want in vals:
                self.burn_codec_var.set(want)
        except Exception:
            pass
    def _on_burn_codec_changed(self, value=None):
        try:
            self._refresh_burn_speed_desc()
        except Exception:
            pass
        try:
            self._refresh_burn_est()
        except Exception:
            pass
    def _refresh_burn_codec_state(self):
        try:
            self.burn_codec_menu.configure(state="normal")
        except Exception:
            pass
    def show_burn_compare(self):
        try:
            _old = getattr(self, "_burn_compare_win", None)
            if _old is not None:
                try:
                    if _old.winfo_exists():
                        _old.lift()
                        try:
                            _old.focus_force()
                        except Exception:
                            pass
                        return
                except Exception:
                    pass
        except Exception:
            pass
        try:
            win = ctk.CTkToplevel(self)
        except Exception:
            return
        self._burn_compare_win = win
        try:
            win.title("Burn speeds compared")
            win.geometry("680x430")
            win.minsize(560, 360)
            win.configure(fg_color=BG_DARK)
            try:
                win.transient(self)
            except Exception:
                pass
        except Exception:
            pass
        try:
            ctk.CTkLabel(win, text="Which burn to pick",
                         font=("Segoe UI", 15, "bold"),
                         text_color=FG_PRIMARY).pack(anchor="w", padx=16,
                                                     pady=(14, 2))
            ctk.CTkLabel(win, text="Typical figures — your content moves them; "
                         "the learner calibrates to your machine.",
                         font=("Segoe UI", 10), text_color=FG_DIM,
                         wraplength=640, justify="left").pack(anchor="w",
                                                              padx=16, pady=(0, 6))
        except Exception:
            pass
        try:
            grid = ctk.CTkScrollableFrame(win, fg_color="transparent")
            grid.pack(fill="both", expand=True, padx=12, pady=4)
            try:
                self._tune_scrollers(win)
            except Exception:
                pass
            headers = ("Mode", "Encoder", "Size accuracy", "Speed",
                       "Plays on", "Best for")
            for c, h in enumerate(headers):
                try:
                    ctk.CTkLabel(grid, text=h, font=("Segoe UI", 10, "bold"),
                                 text_color=ACCENT_GLOW).grid(
                        row=0, column=c, sticky="w",
                        padx=6, pady=(4, 2))
                except Exception:
                    pass
            for r, row in enumerate(BURN_COMPARE_ROWS, start=1):
                for c, val in enumerate(row):
                    try:
                        ctk.CTkLabel(grid, text=str(val),
                                     font=("Segoe UI", 10),
                                     text_color=FG_PRIMARY if c == 0 else FG_SECONDARY,
                                     wraplength=150, justify="left").grid(
                            row=r, column=c, sticky="nw",
                            padx=6, pady=2)
                    except Exception:
                        pass
            try:
                ctk.CTkLabel(grid, text=BURN_COMPARE_CODEC_NOTE,
                             font=("Segoe UI", 10), text_color=ACCENT_GLOW,
                             wraplength=640, justify="left").grid(
                    row=len(BURN_COMPARE_ROWS) + 1, column=0,
                    columnspan=6, sticky="w", padx=6, pady=(8, 2))
            except Exception:
                pass
        except Exception:
            pass
        try:
            ctk.CTkButton(win, text="Close", font=("Segoe UI", 12),
                          fg_color=BTN_DIM, hover_color=BTN_DIM_HOVER,
                          height=36, corner_radius=8, text_color=FG_SECONDARY,
                          command=win.destroy).pack(fill="x", padx=16,
                                                    pady=(4, 14))
        except Exception:
            pass
        try:
            win.lift()
            win.focus_force()
        except Exception:
            pass
    def _refresh_burn_est(self):
        try:
            token = int(getattr(self, "_burn_est_token", 0) or 0) + 1
            self._burn_est_token = token
        except Exception:
            token = 0
        try:
            from srt import BURN_SPEED_IDS as _BSI
            speed = _BSI.get((self.burn_speed_var.get() or "").strip(), "match")
        except Exception:
            speed = "match"
        try:
            vauto, vkbps = self.get_burn_vbr()
        except Exception:
            vauto, vkbps = True, 0
        try:
            vcodec = self.get_burn_codec()
        except Exception:
            vcodec = "h264"
        try:
            paths = list(self.get_srt_input_paths() or [])
            out_dir = self._srt_out_dir()
        except Exception:
            paths, out_dir = [], ""
        try:
            import threading as _th
            _th.Thread(target=self._burn_est_work,
                       args=(token, paths, out_dir, speed, vauto, vkbps,
                             vcodec),
                       daemon=True).start()
        except Exception:
            pass
    def _burn_est_work(self, token, paths, out_dir, speed, vauto, vkbps,
                       vcodec="h264"):
        try:
            import os as _os
            from pathlib import Path as _P
            from srt import (probe_media as _probe, get_ffmpeg_exe as _ff,
                             is_audio_only_path as _audio_only,
                             estimate_burn_batch as _est)
            try:
                ff = _ff()
            except Exception:
                ff = None
            if not isinstance(self._burn_probe_cache, dict):
                self._burn_probe_cache = {}
            if len(self._burn_probe_cache) > 300:
                self._burn_probe_cache = {}
            entries = []
            for p in paths or []:
                try:
                    if _audio_only(p):
                        continue
                    st = _os.stat(p)
                    key = (str(p), st.st_mtime_ns, st.st_size)
                    hit = self._burn_probe_cache.get(key)
                    if hit is None and ff:
                        try:
                            info = _probe(_P(p), ff)
                            hit = {"duration": float(info.get("duration") or 0),
                                   "audio_bps": int(info.get("audio_bps") or 0),
                                   "src_bytes": int(info.get("size") or st.st_size)}
                        except Exception:
                            hit = {}
                        try:
                            self._burn_probe_cache[key] = hit
                        except Exception:
                            pass
                    hit = hit or {}
                    if float(hit.get("duration") or 0) > 1.0:
                        entries.append(hit)
                except Exception:
                    continue
            res = _est(entries, speed, vauto, vkbps, vcodec)
            try:
                from srt import burn_size_fudge as _fg
                _fn = _fg(speed, vcodec)[1]
            except Exception:
                _fn = 0
            try:
                ctx = {"entries": entries, "speed": speed,
                       "codec": vcodec, "fudge_n": _fn}
            except Exception:
                ctx = {"entries": [], "speed": speed, "codec": vcodec,
                       "fudge_n": 0}
        except Exception:
            res = {"mode": "none"}
            ctx = {"entries": [], "speed": "match"}
        try:
            self.after(0, lambda: self._apply_burn_est(token, res, ctx))
        except Exception:
            pass
    @staticmethod
    def _fmt_est_mb(nbytes) -> str:
        try:
            mb = float(nbytes) / 1e6
        except Exception:
            return ""
        return f"{mb:.0f}" if mb >= 100 else f"{mb:.1f}"
    def _set_est_entry(self, text: str, enabled: bool, placeholder: str = ""):
        try:
            self.burn_est_entry.configure(state="normal" if enabled else "disabled")
        except Exception:
            pass
        try:
            if placeholder != getattr(self, "_est_placeholder", None):
                try:
                    self.burn_est_entry.configure(placeholder_text=placeholder)
                except Exception:
                    pass
                self._est_placeholder = placeholder
        except Exception:
            pass
        try:
            self.burn_est_entry.delete(0, "end")
            if text:
                self.burn_est_entry.insert(0, text)
            self._burn_est_shown = text or ""
        except Exception:
            pass
    def _apply_burn_est(self, token, res, ctx=None):
        try:
            if int(token) != int(getattr(self, "_burn_est_token", 0) or 0):
                return
        except Exception:
            pass
        try:
            self._burn_est_ctx = dict(ctx or {})
        except Exception:
            self._burn_est_ctx = {}
        try:
            typing = False
            try:
                typing = (self.focus_get() == self.burn_est_entry)
            except Exception:
                typing = False
            try:
                running = bool(getattr(self, "_srt_running", False))
            except Exception:
                running = False
            res = res or {}
            mode = res.get("mode")
            n = int(res.get("basis") or 0)
            if mode == "manual" and res.get("bytes"):
                if n >= 2:
                    basis = f"(learned, {n} burns)"
                elif n == 1:
                    basis = "(1 burn — rough, one more sharpens)"
                else:
                    basis = "(uncalibrated — burn to sharpen)"
                try:
                    self.burn_est_basis.configure(text=basis)
                except Exception:
                    pass
                if not typing:
                    self._set_est_entry(self._fmt_est_mb(res["bytes"]),
                                        not running, "")
            elif mode == "manual":
                try:
                    self.burn_est_basis.configure(text="(could not measure queue)")
                except Exception:
                    pass
                if not typing:
                    self._set_est_entry("", False, "MB")
            elif mode == "auto" and res.get("bytes"):
                try:
                    self.burn_est_basis.configure(text="(size-match ≈ source)")
                except Exception:
                    pass
                if not typing:
                    self._set_est_entry(self._fmt_est_mb(res["bytes"]),
                                        False, "")
            else:
                try:
                    self.burn_est_basis.configure(text="")
                except Exception:
                    pass
                if not typing:
                    self._set_est_entry("", False, "MB")
        except Exception:
            pass
    def _on_burn_mb_commit(self, event=None):
        if getattr(self, "_srt_running", False):
            return
        try:
            from srt import BURN_SPEED_IDS as _BSI, solve_burn_kbps as _solve
            speed = _BSI.get((self.burn_speed_var.get() or "").strip(), "match")
            codec = self.get_burn_codec()
        except Exception:
            return
        try:
            ctx = getattr(self, "_burn_est_ctx", None) or {}
            if ctx.get("speed") != speed or ctx.get("codec", "h264") != codec:
                try:
                    self._refresh_burn_est()
                except Exception:
                    pass
                return
            raw = str(self.burn_est_entry.get() or "").strip().lower()
            if raw.endswith("mb"):
                raw = raw[:-2].strip()
            if not raw:
                try:
                    self._refresh_burn_est()
                except Exception:
                    pass
                return
            if "," in raw and "." not in raw:
                raw = raw.replace(",", ".")
            else:
                raw = raw.replace(",", "")
            mb = float(raw)
            kbps = _solve(ctx.get("entries") or [], speed, mb, codec)
            if kbps is None:
                self.srt_log("Size target needs a target above the audio floor "
                             "(type MB, e.g. 850)")
                try:
                    self._set_est_entry(getattr(self, "_burn_est_shown", ""),
                                        len(self._burn_est_shown or "") > 0,
                                        "" if getattr(self, "_burn_est_shown", "") else "MB")
                except Exception:
                    pass
                return
            try:
                _hn = int(ctx.get("fudge_n") or 0)
            except Exception:
                _hn = 0
            lo, hi = 300, 10000
            clamped = kbps < lo or kbps > hi
            kbps = max(lo, min(hi, kbps))
            try:
                self.burn_vbr_auto_var.set(False)
            except Exception:
                pass
            self._burn_vbr_exact = kbps
            try:
                self.burn_vbr_slider.configure(state="normal")
            except Exception:
                pass
            try:
                self.burn_vbr_slider.set(max(300, min(10000, int(round(kbps / 100.0)) * 100)))
            except Exception:
                pass
            self._update_burn_vbr_display()
            try:
                _cal = "" if _hn >= 2 else " (uncalibrated - analytic only)"
            except Exception:
                _cal = ""
            self.srt_log(f"Target ≈{mb:g} MB -> {kbps} kbps video"
                         f"{' (clamped to slider range)' if clamped else ''}{_cal} [{speed}]")
            try:
                self._refresh_burn_est()
            except Exception:
                pass
        except Exception as e:
            try:
                self.srt_log(f"Size target not understood (type MB, e.g. 850): {e}")
            except Exception:
                pass
            try:
                self._refresh_burn_est()
            except Exception:
                pass
    def _refresh_burn_speed_desc(self):
        try:
            label = (self.burn_speed_var.get() or "").strip()
        except Exception:
            label = ""
        try:
            text = BURN_SPEED_HELP.get(label, "")
        except Exception:
            text = ""
        if "NVENC" in label:
            try:
                import gpu as _gpumod
                from srt import get_ffmpeg_exe as _get_ff
                try:
                    exe = _get_ff()
                except Exception:
                    exe = None
                if _gpumod.nvenc_available(exe):
                    try:
                        import gpu as _g2
                        _best = _g2.best_gpu() or {}
                        text += f" Ready: {_best.get('name', 'NVIDIA GPU')}."
                    except Exception:
                        text += " Ready on this machine."
                else:
                    text += " Not available on this machine."
            except Exception:
                pass
        try:
            from srt import BURN_SPEED_IDS as _BSI2
            _sid2 = _BSI2.get(label, "match")
            _is_nvenc = _sid2.startswith("nvenc_")
            if self.get_burn_codec() == "hevc" and _is_nvenc:
                text += (" HEVC ≈30% smaller than H.264 at the same "
                         "visual quality, same GPU speed (needs ~2016+ "
                         "playback).")
            elif self.get_burn_codec() == "hevc":
                text += (" HEVC via x265: ≈30–40% smaller, much slower "
                         "(needs ~2016+ playback).")
            elif _is_nvenc:
                text += " Switch codec to HEVC for ≈30% smaller files."
            else:
                text += " Switch codec to HEVC (x265) for ≈30–40% smaller files."
        except Exception:
            pass
        try:
            self.burn_speed_desc.configure(text=text)
        except Exception:
            pass
        try:
            self._refresh_burn_codec_state()
        except Exception:
            pass
    def _on_burn_speed_changed(self, value):
        try:
            self._refresh_burn_speed_desc()
        except Exception:
            pass
        try:
            self._refresh_burn_est()
        except Exception:
            pass
        try:
            from srt import BURN_SPEED_IDS as _BSI
            sid = _BSI.get((value or "").strip(), "match")
        except Exception:
            return
        if sid not in ("nvenc_draft", "nvenc_turbo", "nvenc_balanced"):
            return
        ok = False
        try:
            import gpu as _gpumod
            from srt import get_ffmpeg_exe as _get_ff
            try:
                exe = _get_ff()
            except Exception:
                exe = None
            ok = bool(_gpumod.nvenc_available(exe))
        except Exception:
            ok = False
        if not ok:
            try:
                from srt import BURN_SPEED_LABELS as _BSL
                self.burn_speed_var.set(_BSL.get("match", value))
            except Exception:
                pass
            try:
                self._refresh_burn_speed_desc()
            except Exception:
                pass
            self.set_srt_progress(0, "NVENC needs an NVIDIA GPU - reverted to Match size")
            try:
                self.srt_log("NVENC unavailable (no NVIDIA GPU/encoder) - pick a CPU mode.")
            except Exception:
                pass
    def get_burn_font_size(self) -> int:
        try:
            return max(10, min(40, int(round(float(self.burn_font_slider.get())))))
        except Exception:
            return 18
    def get_preview_sample(self):
        try:
            start_text = (self.sample_start_var.get() or "").strip()
        except Exception:
            start_text = ""
        try:
            length = int(str(self.sample_len_var.get() or "15s").strip()
                         .lower().rstrip("s"))
        except Exception:
            length = 15
        if length not in (10, 15, 30, 60):
            length = 15
        return start_text, length
    def set_srt_preview_callback(self, cb: Callable):
        self._srt_preview_cb = cb if callable(cb) else None
    def _on_srt_preview(self):
        if getattr(self, "_preview_running", False):
            return
        if not self.get_srt_input_paths():
            self.set_srt_progress(0, "Add video files first (preview needs the queue)")
            return
        if self._srt_preview_cb:
            try:
                out_dir = self.srt_out_entry.get().strip()
            except Exception:
                out_dir = ""
            self._preview_running = True
            try:
                self.srt_preview_btn.configure(state="disabled")
            except Exception:
                pass
            threading.Thread(target=self._srt_preview_cb,
                             args=(self.get_srt_input_paths(), out_dir,
                                   self.get_burn_font_size(),
                                   *self.get_preview_sample()),
                             daemon=True).start()
    def set_srt_preview_done(self):
        self._preview_running = False
        try:
            self.srt_preview_btn.configure(state="normal")
        except Exception:
            pass
    def _on_srt_start(self):
        if self._srt_running:
            return
        paths = self.get_srt_input_paths()
        if not paths:
            self.set_srt_progress(0, "Add video/audio files first")
            return
        order = list(range(len(paths)))
        try:
            from tkinter import messagebox as _mb
            from pathlib import Path as _P
            from srt import default_out_path as _fn
            out_dir = self._srt_out_dir()
            existing = [i for i, p in enumerate(paths)
                        if self._safe_exists(_fn, _P(p), out_dir)]
            if existing:
                if len(existing) >= len(paths):
                    self.set_srt_progress(
                        0, "All queued files already have SRTs - Clear or add more")
                    return
                r = _mb.askyesnocancel(
                    "SRTs already exist",
                    f"{len(existing)} of {len(paths)} file(s) already have SRTs.\n"
                    f"Yes = overwrite them all\nNo = choose per file\nCancel = abort")
                if r is None:
                    return
                if r is False:
                    keep = []
                    for i, p in enumerate(paths):
                        if i not in existing:
                            keep.append(i)
                            continue
                        try:
                            name = _P(p).name
                        except Exception:
                            name = p
                        if _mb.askyesno("Overwrite SRT?",
                                        f"Overwrite the SRT for:\n{name}"):
                            keep.append(i)
                        else:
                            self.set_srt_file_status(i, "– skipped")
                    if not keep:
                        self.set_srt_progress(0, "All skipped - nothing to do")
                        return
                    order = keep
        except Exception:
            pass
        for i in order:
            self.set_srt_file_status(i, "queued")
        # Dual-engine pre-flight (another session may be running): the app
        # verdict arrives synchronously on this GUI thread.
        if getattr(self, "_srt_prequest_cb", None) is not None:
            try:
                _pv = self._srt_prequest_cb() or {"go": True}
            except Exception:
                _pv = {"go": True}
            if isinstance(_pv, dict) and "confirm" in _pv:
                try:
                    from tkinter import messagebox as _mb2
                    _yes = bool(_mb2.askyesno(
                        "Load second engine?",
                        str(_pv.get("confirm") or "Another session is active."),
                        parent=self))
                except Exception:
                    _yes = False
                if not _yes:
                    try:
                        self.set_srt_progress(0, "SRT start cancelled")
                    except Exception:
                        pass
                    return
        if self._srt_start_cb:
            try:
                out_dir = self.srt_out_entry.get().strip()
            except Exception:
                out_dir = ""
            try:
                cpu = max(1, int(round(float(self.srt_cpu_slider.get()))))
            except Exception:
                cpu = 1
            try:
                srt_in, srt_out = self.get_srt_lang_codes()
            except Exception:
                srt_in, srt_out = "auto", "en"
            try:
                srt_task = (self.canary_task_var.get() or "").strip().lower()
            except Exception:
                srt_task = ""
            if srt_task not in ("transcribe", "translate"):
                srt_task = ""
            try:
                opts = self.get_srt_options()
            except Exception:
                opts = {}
            try:
                _bfont = self.get_burn_font_size()
            except Exception:
                _bfont = 18
            try:
                from srt import BURN_SPEED_IDS as _BSI
                _bspd = _BSI.get((self.burn_speed_var.get() or "").strip(), "match")
            except Exception:
                _bspd = "match"
            try:
                _bvauto, _bvk = self.get_burn_vbr()
            except Exception:
                _bvauto, _bvk = True, 2000
            try:
                _bcodec = self.get_burn_codec()
            except Exception:
                _bcodec = "h264"
            self._srt_running_mode = "generate"
            threading.Thread(target=self._srt_start_cb,
                             args=([paths[i] for i in order], order,
                                   out_dir, cpu,
                                   srt_in, srt_out, srt_task,
                                   bool(opts.get("srt_norm", False)),
                                   bool(opts.get("burn_after", False)),
                                   _bfont, _bspd, _bvauto, _bvk, _bcodec),
                             daemon=True).start()
    def _on_srt_cancel(self):
        if self._srt_cancel_cb:
            try:
                threading.Thread(target=self._srt_cancel_cb, daemon=True).start()
            except Exception:
                pass
    def _on_srt_burn(self):
        if self._srt_running:
            return
        paths = self.get_srt_input_paths()
        if not paths:
            self.set_srt_progress(0, "Add video files first (burn needs the queue)")
            return
        order = list(range(len(paths)))
        try:
            from tkinter import messagebox as _mb
            from pathlib import Path as _P
            from srt import default_out_path as _srtfn
            from srt import default_burn_path as _bfn
            out_dir = self._srt_out_dir()
            runnable = []
            for i, p in enumerate(paths):
                if not self._safe_exists(_srtfn, _P(p), out_dir):
                    self.set_srt_file_status(i, "– no SRT")
                else:
                    runnable.append(i)
            if not runnable:
                self.set_srt_progress(0, "Generate SRT first - nothing burnable queued")
                return
            existing = [i for i in runnable
                        if self._safe_exists(_bfn, _P(paths[i]), out_dir)]
            if existing:
                r = _mb.askyesnocancel(
                    "Burned videos already exist",
                    f"{len(existing)} of {len(runnable)} file(s) already have burned MP4s.\n"
                    f"Yes = overwrite them all\nNo = choose per file\nCancel = abort")
                if r is None:
                    return
                if r is True:
                    order = runnable
                else:
                    keep = []
                    for i in runnable:
                        if i not in existing:
                            keep.append(i)
                            continue
                        try:
                            name = _P(paths[i]).name
                        except Exception:
                            name = paths[i]
                        if _mb.askyesno("Overwrite burned MP4?",
                                        f"Overwrite the burned video for:\n{name}"):
                            keep.append(i)
                        else:
                            self.set_srt_file_status(i, "– skipped")
                    if not keep:
                        self.set_srt_progress(0, "All skipped - nothing to do")
                        return
                    order = keep
            else:
                order = runnable
        except Exception:
            pass
        for i in order:
            self.set_srt_file_status(i, "queued")
        if self._srt_burn_cb:
            try:
                out_dir = self.srt_out_entry.get().strip()
            except Exception:
                out_dir = ""
            try:
                cpu = max(1, int(round(float(self.srt_cpu_slider.get()))))
            except Exception:
                cpu = 1
            try:
                from srt import BURN_SPEED_IDS as _BSI
                speed = _BSI.get((self.burn_speed_var.get() or "").strip(), "match")
            except Exception:
                speed = "match"
            try:
                vbr_auto, vbr_kbps = self.get_burn_vbr()
            except Exception:
                vbr_auto, vbr_kbps = True, 2000
            try:
                _bcodec = self.get_burn_codec()
            except Exception:
                _bcodec = "h264"
            self._srt_running_mode = "burn"
            threading.Thread(target=self._srt_burn_cb,
                             args=([paths[i] for i in order], order, out_dir, cpu,
                                   self.get_burn_font_size(), speed,
                                   vbr_auto, vbr_kbps, _bcodec),
                             daemon=True).start()
    def _on_srt_open_folder(self):
        import os as _os
        import subprocess as _sp
        from pathlib import Path as _P
        target = self.srt_out_entry.get().strip() if hasattr(self, "srt_out_entry") else ""
        if not target:
            try:
                first = (self.get_srt_input_paths() or [""])[0]
                if first:
                    target = str(_P(first).parent)
            except Exception:
                target = ""
        if not target:
            return
        try:
            _os.startfile(target)
        except Exception:
            try:
                _sp.Popen(["explorer", target])
            except Exception:
                pass
    def set_srt_input(self, path: str):
        try:
            if getattr(self, "_srt_running", False):
                return
            self._srt_input_paths = []
            self._srt_file_status = {}
        except Exception:
            pass
        self.add_srt_files([path])
    def set_srt_callbacks(self, on_start: Callable, on_cancel: Callable,
                          on_burn: Optional[Callable] = None,
                          on_prequest: Optional[Callable] = None):
        self._srt_start_cb = on_start if callable(on_start) else None
        self._srt_cancel_cb = on_cancel if callable(on_cancel) else None
        self._srt_burn_cb = on_burn if callable(on_burn) else None
        self._srt_prequest_cb = on_prequest if callable(on_prequest) else None
    def set_srt_engine_label(self, text: str):
        """Legacy no-op (the summary label is real Engine/Model menus now).
        Kept so old callers never break."""
        try:
            w = getattr(self, "srt_engine_label", None)
            if w is not None:
                w.configure(text=text)
        except Exception:
            pass
    def set_srt_running(self, running: bool):
        self._srt_running = running
        try:
            if running:
                try:
                    self._show_srt_action(
                        getattr(self, "_srt_running_mode", None) or "generate")
                except Exception:
                    pass
                try:
                    self.burn_est_entry.configure(state="disabled")
                except Exception:
                    pass
            try:
                self.srt_start_btn.configure(
                    state="disabled" if running else "normal")
            except Exception:
                pass
            try:
                self.srt_cancel_btn.configure(
                    state="normal" if running else "disabled")
            except Exception:
                pass
            try:
                self.srt_burn_btn.configure(
                    state="disabled" if running else "normal")
            except Exception:
                pass
            for w in (getattr(self, "srt_browse_btn", None),
                      getattr(self, "srt_clear_btn", None)):
                try:
                    if w is not None:
                        w.configure(state="disabled" if running else "normal")
                except Exception:
                    pass
            try:
                self.srt_file_list.configure(
                    state="disabled" if running else "normal")
            except Exception:
                pass
            if running:
                try:
                    self.srt_bar.set(0)
                    self.srt_pct.configure(text="0%")
                except Exception:
                    pass
            else:
                try:
                    self.update_action_states(touch_progress=False)
                except Exception:
                    pass
        except Exception:
            pass

    def set_srt_progress(self, frac: float, msg: str = ""):
        try:
            frac = max(0.0, min(1.0, float(frac)))
        except Exception:
            frac = 0.0
        try:
            self.srt_bar.set(frac)
            self.srt_pct.configure(text=f"{int(round(frac * 100))}%")
            if msg:
                self.srt_status.configure(text=msg)
        except Exception:
            pass
    def srt_log(self, msg: str):
        try:
            self.srt_log_box.configure(state="normal")
            self.srt_log_box.insert("end", str(msg).rstrip() + "\n")
            self._srt_log_lines = getattr(self, "_srt_log_lines", 0) + 1
            if self._srt_log_lines >= 20:
                self._srt_log_lines = 0
                try:
                    total = int(float(self.srt_log_box.index("end-1c").split(".")[0]))
                    if total > 800:
                        self.srt_log_box.delete("1.0", "300.0")
                except Exception:
                    pass
            self.srt_log_box.see("end")
            self.srt_log_box.configure(state="disabled")
        except Exception:
            pass
    def _on_clear_srt_hist(self):
        try:
            from srt import clear_eta_history as _clr
            n = int(_clr("srt") or 0)
            word = "entry" if n == 1 else "entries"
            self.srt_log(f"SRT timing history cleared ({n} {word}); estimates restart from defaults")
        except Exception as e:
            try:
                self.srt_log(f"Clear SRT history failed: {e}")
            except Exception:
                pass
    def _on_clear_burn_hist(self):
        try:
            from srt import (clear_eta_history as _clr,
                             clear_burn_size_history as _clrb)
            n = int(_clr("burn") or 0)
            m = int(_clrb() or 0)
            self.srt_log(f"Burn history cleared ({n} timing + {m} size entries); recalibrates on next burn")
            try:
                self._refresh_burn_est()
            except Exception:
                pass
        except Exception as e:
            try:
                self.srt_log(f"Clear burn history failed: {e}")
            except Exception:
                pass
    def srt_done(self, ok: bool, msg: str):
        self.set_srt_running(False)
        try:
            self.srt_clear_btn.configure(state="normal")
        except Exception:
            pass
        try:
            self.srt_browse_btn.configure(state="normal")
        except Exception:
            pass
        try:
            self.srt_file_list.configure(state="normal")
        except Exception:
            pass
        try:
            self.srt_status.configure(
                text=msg, text_color=theme_color(SUCCESS if ok else DANGER))
        except Exception:
            pass
        self.srt_log(f"{'DONE' if ok else 'FAILED'}: {msg}")
        try:
            self._refresh_burn_est()
        except Exception:
            pass

    def _on_srt_input_lang_changed(self, value):
        try:
            self._refresh_srt_engine_label()
        except Exception:
            pass
        if self._srt_input_lang_cb:
            self._srt_input_lang_cb(value)
    def _on_srt_output_lang_changed(self, value):
        try:
            self._refresh_srt_engine_label()
        except Exception:
            pass
        if self._srt_output_lang_cb:
            self._srt_output_lang_cb(value)
    def set_srt_languages(self, input_lang: str, output_lang: str,
                           input_cb: Callable, output_cb: Callable):
        self._srt_input_lang_cb = input_cb
        self._srt_output_lang_cb = output_cb
        in_display = SRT_LANG_NAMES.get(input_lang, "")
        out_display = SRT_LANG_NAMES.get(output_lang, "")
        if in_display in SRT_LANG_DISPLAY:
            self.srt_input_lang_var.set(in_display)
        if out_display in SRT_LANG_DISPLAY:
            self.srt_output_lang_var.set(out_display)
    def get_srt_lang_codes(self) -> Tuple[str, str]:
        in_display = self.srt_input_lang_var.get()
        out_display = self.srt_output_lang_var.get()
        in_code = SRT_LANG_CODE_FROM_DISPLAY.get(in_display, "auto")
        out_code = SRT_LANG_CODE_FROM_DISPLAY.get(out_display, "en")
        return in_code, out_code
    def refresh_lang_options(self, engine_kind: str = ""):
        eng = engine_kind or self.engine_var.get()
        is_canary = (eng == "Canary-1B")
        live_codes = CANARY_LIVE_SRC_CODES if is_canary else WHISPER_LANGS
        if is_canary:
            srt_in_codes, srt_out_codes = CANARY_SRT_IN_CODES, CANARY_SRT_OUT_CODES
        elif eng == "Whisper":
            srt_in_codes, srt_out_codes = WHISPER_SRT_IN_CODES, WHISPER_SRT_OUT_CODES
        else:
            srt_in_codes, srt_out_codes = WHISPER_SRT_IN_CODES, WHISPER_SRT_OUT_CODES
        try:
            self.canary_lang_menu.configure(values=live_codes)
            self.srt_input_lang_menu.configure(
                values=[SRT_LANG_NAMES[c] for c in srt_in_codes])
            self.srt_output_lang_menu.configure(
                values=[SRT_LANG_NAMES[c] for c in srt_out_codes])
        except Exception:
            pass
        try:
            if self.canary_lang_var.get() not in live_codes:
                fallback = "auto" if "auto" in live_codes else live_codes[0]
                self.canary_lang_var.set(fallback)
                self._on_canary_lang_changed(fallback)
        except Exception:
            pass
        try:
            in_code, out_code = self.get_srt_lang_codes()
            if in_code not in srt_in_codes:
                self.srt_input_lang_var.set(
                    SRT_LANG_NAMES[srt_in_codes[0]])
            if out_code not in srt_out_codes:
                self.srt_output_lang_var.set(
                    SRT_LANG_NAMES[srt_out_codes[0]])
        except Exception:
            pass
        try:
            self._refresh_srt_engine_label()
        except Exception:
            pass
    def set_srt_lang_state(self, engine_kind: str):
        has_lang = engine_kind in ("Canary-1B", "Whisper")
        try:
            state = "normal" if has_lang else "disabled"
            self.srt_input_lang_menu.configure(state=state)
            self.srt_output_lang_menu.configure(state=state)
            if engine_kind == "Canary-1B":
                hint = "Canary-1B supports English, German, Spanish, French only (translate always outputs English)"
            elif engine_kind == "Whisper":
                hint = "SRT subtitles use these languages (translate task always outputs English)"
            else:
                hint = "Only Whisper and Canary-1B support language selection"
            self.srt_lang_hint.configure(text=hint)
        except Exception:
            pass
        try:
            self._refresh_srt_engine_label()
        except Exception:
            pass