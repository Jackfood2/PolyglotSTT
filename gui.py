"""Professional GUI for MoonshineSTT using customtkinter."""

import customtkinter as ctk
import threading
import time
import math
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

# Model size choices - maps display label -> arch int (from catalog 0-5)
MODEL_CHOICES = {
    "Tiny (26MB, fastest)": 0,
    "Tiny Streaming (45MB, light)": 2,
    "Base (60MB, balanced)": 1,
    "Base Streaming (60MB, light)": 3,
    "Small Streaming (80MB, good)": 4,
    "Medium Streaming (110MB, best)": 5,
}
MODEL_CHOICES_REV = {v: k for k, v in MODEL_CHOICES.items()}

# Downloadable Whisper sizes - single source of truth lives in
# whisper_engine (imported; local fallback only if that import fails).
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
        "Large (3GB)": "large",
        "Large v1 (3GB)": "large-v1",
        "Large v2 (3GB)": "large-v2",
        "Large v3 (3GB, best)": "large-v3",
    }
    WHISPER_MODEL_CHOICES_REV = {v: k for k, v in WHISPER_MODEL_CHOICES.items()}
CANARY_MODEL_LABEL = "Canary-1B (3.9GB, fixed)"


# One-line trade-off notes for the burn-speed menu, keyed by menu label.
BURN_SPEED_HELP = {
    "Match size (2-pass x264)": "Exact size (±1–3%). Slowest — encodes the video twice.",
    "Fast (1-pass x264)": "About half the time, size within ~±10%.",
    "Fastest (ultrafast 1-pass)": "Several times faster, visibly softer. For quick checks, not keeps.",
    "Draft (NVENC fast 1-pass)": "Needs NVIDIA GPU. Fastest encode, size approximate (~±10%).",
    "Turbo (NVENC ultra-fast 1-pass)": "Needs NVIDIA GPU. Lowest-latency tune, a touch softer than Draft.",
    "Balanced (NVENC 2-pass)": "Needs NVIDIA GPU. GPU two-pass, closer size (~±5–10%).",
}


def apply_badge_map(values, status_fn=None):
    """Build (display_list, {display: base}) with ✓/↓ download badges.

    Pure function (no widgets) so it is unit-testable. Display strings are
    uniquified defensively; a failing status_fn degrades to no badges.
    """
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

ENGINE_CHOICES = ["Moonshine v2", "Canary-1B", "Whisper Large v3"]
CANARY_TASKS = ["transcribe", "translate"]
# Union of Canary + Whisper src langs so one shared Src menu serves both engines.
# Canary officially uses en/de/es/fr (+ja/zh experimental); Whisper adds ko + more.
CANARY_LANGS = ["auto", "en", "ja", "zh", "ko", "de", "es", "fr",
                "it", "pt", "nl", "ru", "ar", "hi", "tr", "id", "uk", "vi", "th"]
WHISPER_TASKS = ["transcribe", "translate"]
WHISPER_LANGS = CANARY_LANGS
# Full language list for SRT input/output dropdowns (superset of all engines).
# Displayed by FULL NAME only - no "ja - Japanese" code shortcuts.
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
# Per-engine capability lists. Probed 2026-09-03: canary-1b.nemo only
# contains <|de|> <|en|> <|es|> <|fr|> - anything else crashes NeMo, so the
# menus must not offer it while Canary is active. Whisper covers the full
# list. Output menus exclude "auto" (auto-detect as an OUTPUT is meaningless;
# Canary would crash on target_lang="auto").
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
                              fill="#2D3748", outline="")
        if self.level > 0.01:
            w = self.meter_width * self.level
            if self.level < 0.6:
                color = SUCCESS
            elif self.level < 0.85:
                color = WARNING
            else:
                color = DANGER
            # Plain fill only: the old per-pixel sine loop created up to
            # ~420 canvas objects per redraw and froze the UI at 15Hz.
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
            return  # already looping - a second loop would double-draw forever
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
        self.label.configure(text=text, text_color=color)
        self.indicator.configure(text_color=color)


class HistoryPanel(ctk.CTkScrollableFrame):
    MAX_ENTRIES = 100  # unbounded history = unbounded widgets/RAM + scroll crawl

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
        live.grid_columnconfigure(0, weight=1)
        live.grid_rowconfigure(3, weight=1)
        srt_tab.grid_columnconfigure(0, weight=1)
        srt_tab.grid_rowconfigure(0, weight=1)

        record_card = ctk.CTkFrame(live, fg_color=BG_CARD, corner_radius=16)
        record_card.grid(row=0, column=0, sticky="ew", padx=8, pady=(0, 8))
        record_card.grid_columnconfigure(0, weight=1)

        self.meter = LevelMeter(record_card, width=420, height=10)
        self.meter.grid(row=0, column=0, padx=20, pady=(16, 4), sticky="ew")

        self.waveform = WaveformDisplay(record_card, width=420, height=70)
        self.waveform.grid(row=1, column=0, padx=16, pady=(8, 4), sticky="ew")

        btn_frame = ctk.CTkFrame(record_card, fg_color="transparent")
        btn_frame.grid(row=2, column=0, pady=(12, 16), sticky="ew")
        btn_frame.grid_columnconfigure(0, weight=1)
        btn_frame.grid_columnconfigure(1, weight=1)
        btn_frame.grid_columnconfigure(2, weight=1)

        self.record_btn = ctk.CTkButton(
            btn_frame, text="\u25CF  RECORD", font=("Segoe UI", 13, "bold"),
            fg_color=ACCENT, hover_color=ACCENT_DARK, height=40,
            corner_radius=10, command=self._toggle_record)
        self.record_btn.grid(row=0, column=0, padx=(16, 4), sticky="ew")

        self.clear_btn = ctk.CTkButton(
            btn_frame, text="Clear", font=("Segoe UI", 12),
            fg_color="#2D3748", hover_color="#4A5568", height=40,
            corner_radius=10, text_color=FG_SECONDARY,
            command=self._clear_history)
        self.clear_btn.grid(row=0, column=1, padx=4, sticky="ew")

        self.copy_last_btn = ctk.CTkButton(
            btn_frame, text="Copy", font=("Segoe UI", 12),
            fg_color="#2D3748", hover_color="#4A5568", height=40,
            corner_radius=10, text_color=FG_SECONDARY,
            command=self._copy_last)
        self.copy_last_btn.grid(row=0, column=2, padx=(4, 16), sticky="ew")

        output_card = ctk.CTkFrame(live, fg_color=BG_CARD, corner_radius=16)
        output_card.grid(row=1, column=0, sticky="ew", padx=8, pady=(0, 8))
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
        history_label_frame.grid(row=2, column=0, sticky="ew", padx=12, pady=(8, 4))
        ctk.CTkLabel(history_label_frame, text="Transcription History",
                     font=("Segoe UI", 11, "bold"), text_color=FG_DIM
                     ).pack(side="left")

        self.history = HistoryPanel(live, height=140)
        self.history.grid(row=3, column=0, sticky="nsew", padx=8, pady=(0, 8))

        # Engine selector - new for Canary-1B portable
        engine_frame = ctk.CTkFrame(live, fg_color=BG_CARD, corner_radius=10)
        engine_frame.grid(row=4, column=0, sticky="ew", padx=8, pady=(0, 8))
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

        # Settings row - ported from test.py typing method / suffix
        settings_frame = ctk.CTkFrame(live, fg_color=BG_CARD, corner_radius=10)
        settings_frame.grid(row=6, column=0, sticky="ew", padx=8, pady=(0, 8))
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

        # Model size selector - paired directly under Engine (same card flow):
        # the options + title always belong to the active engine.
        model_frame = ctk.CTkFrame(live, fg_color=BG_CARD, corner_radius=10)
        model_frame.grid(row=5, column=0, sticky="ew", padx=8, pady=(0, 8))
        self.model_title_label = ctk.CTkLabel(model_frame, text="Model:",
                                              font=("Segoe UI", 10, "bold"), text_color=FG_DIM)
        self.model_title_label.pack(side="left", padx=(12, 4), pady=8)
        self.model_var = ctk.StringVar(value="Medium Streaming (110MB, best)")
        self.model_menu = ctk.CTkOptionMenu(model_frame, variable=self.model_var, values=list(MODEL_CHOICES.keys()), width=260, fg_color=BG_INPUT, button_color=ACCENT, command=self._on_model_changed)
        self.model_menu.pack(side="left", padx=4, pady=8, fill="x", expand=True)
        self.model_manage_btn = ctk.CTkButton(
            model_frame, text="Manage…", width=80, font=("Segoe UI", 12),
            fg_color="#2D3748", hover_color="#4A5568", height=32,
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
        ctk.CTkLabel(footer, text="Moonshine v2 + Canary-1B + Whisper Large v3 \u2022 On-device \u2022 No API keys",
                     font=("Segoe UI", 10), text_color=FG_DIM).pack()

        self._build_srt_tab(srt_tab)

    # ---------------- SRT File tab ----------------
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
        self._srt_input_paths: list = []  # batch queue (ordered, may repeat)
        self._srt_file_status: dict = {}  # queue index -> status text
        self._srt_running = False

        scroll = ctk.CTkScrollableFrame(tab, fg_color="transparent")
        scroll.pack(fill="both", expand=True, padx=4, pady=4)
        scroll.grid_columnconfigure(0, weight=1)

        # File drop card (batch queue)
        file_card = ctk.CTkFrame(scroll, fg_color=BG_CARD, corner_radius=12)
        file_card.grid(row=0, column=0, sticky="ew", padx=4, pady=(0, 8))
        file_card.grid_columnconfigure(0, weight=1)
        ctk.CTkLabel(file_card, text="Video / Audio Files  (queue - runs one by one)",
                     font=("Segoe UI", 11, "bold"), text_color=FG_DIM
                     ).pack(anchor="w", padx=12, pady=(10, 2))
        self.srt_drop = ctk.CTkTextbox(file_card, font=("Segoe UI", 11),
                                       fg_color=BG_INPUT, text_color=FG_SECONDARY,
                                       corner_radius=8, height=44,
                                       activate_scrollbars=False, wrap="word")
        self.srt_drop.pack(fill="x", padx=10, pady=(0, 6))
        self.srt_drop.insert("1.0", "Drag && drop video/audio files here,\nor click Browse Files...")
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
            selectforeground=FG_PRIMARY, highlightthickness=0,
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
                      corner_radius=8, command=self._srt_browse_file)
        self.srt_browse_btn.grid(row=0, column=0, padx=(0, 4), sticky="ew")
        self.srt_clear_btn = ctk.CTkButton(fbtn, text="Clear", font=("Segoe UI", 12),
                      fg_color="#2D3748", hover_color="#4A5568", height=36,
                      corner_radius=8, text_color=FG_SECONDARY,
                      command=self._srt_clear_file)
        self.srt_clear_btn.grid(row=0, column=1, padx=(4, 0), sticky="ew")

        # Output dir card (default = source folder)
        out_card = ctk.CTkFrame(scroll, fg_color=BG_CARD, corner_radius=12)
        out_card.grid(row=1, column=0, sticky="ew", padx=4, pady=(0, 8))
        out_card.grid_columnconfigure(0, weight=1)
        ctk.CTkLabel(out_card, text="Output Folder  (empty = same folder as video)",
                     font=("Segoe UI", 11, "bold"), text_color=FG_DIM
                     ).pack(anchor="w", padx=12, pady=(10, 2))
        orow = ctk.CTkFrame(out_card, fg_color="transparent")
        orow.pack(fill="x", padx=10, pady=(0, 10))
        orow.grid_columnconfigure(0, weight=1)
        self.srt_out_entry = ctk.CTkEntry(orow, font=("Segoe UI", 11),
                                          fg_color=BG_INPUT, text_color=FG_PRIMARY,
                                          corner_radius=8, height=34,
                                          placeholder_text="Same folder as video (default)")
        self.srt_out_entry.grid(row=0, column=0, sticky="ew", padx=(0, 6))
        ctk.CTkButton(orow, text="Browse...", width=90, font=("Segoe UI", 12),
                      fg_color="#2D3748", hover_color="#4A5568", height=34,
                      corner_radius=8, text_color=FG_SECONDARY,
                      command=self._srt_browse_outdir
                      ).grid(row=0, column=1, sticky="e")

        # Engine + CPU card
        perf_card = ctk.CTkFrame(scroll, fg_color=BG_CARD, corner_radius=12)
        perf_card.grid(row=2, column=0, sticky="ew", padx=4, pady=(0, 8))
        perf_card.grid_columnconfigure(1, weight=1)
        ctk.CTkLabel(perf_card, text="Engine:", font=("Segoe UI", 10, "bold"),
                     text_color=FG_DIM).grid(row=0, column=0, sticky="w", padx=(12, 4), pady=(10, 2))
        self.srt_engine_label = ctk.CTkLabel(perf_card, text="Moonshine v2",
                                             font=("Segoe UI", 11), text_color=ACCENT_GLOW)
        self.srt_engine_label.grid(row=0, column=1, sticky="w", padx=4, pady=(10, 2))
        ctk.CTkLabel(perf_card, text="(change in Live tab)",
                     font=("Segoe UI", 9), text_color=FG_DIM
                     ).grid(row=0, column=2, sticky="e", padx=(4, 12), pady=(10, 2))
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
        self.srt_cpu_slider.grid(row=2, column=0, columnspan=2, sticky="ew", padx=12, pady=(2, 10))
        self.srt_cpu_slider.set(def_cpu)
        if self._srt_max_cpu <= 1:
            # from_==to_==1 is a degenerate slider (Tcl error risk) - lock it.
            try:
                self.srt_cpu_slider.configure(state="disabled")
            except Exception:
                pass
        self.srt_cpu_value = ctk.CTkLabel(perf_card, text=f"{def_cpu} threads",
                                          font=("Segoe UI", 11, "bold"), text_color=FG_PRIMARY)
        self.srt_cpu_value.grid(row=2, column=2, sticky="e", padx=(4, 12), pady=(2, 10))
        # Compute device for Whisper/Canary inference (Moonshine is CPU-only).
        # Greyed to "CPU only" when no NVIDIA dGPU is present.
        ctk.CTkLabel(perf_card, text="Compute:", font=("Segoe UI", 10, "bold"),
                     text_color=FG_DIM).grid(row=3, column=0, sticky="w", padx=(12, 4), pady=(2, 10))
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
        self.compute_menu.grid(row=3, column=1, sticky="w", padx=4, pady=(2, 10))
        if not _has_dgpu:
            try:
                self.compute_menu.configure(state="disabled")
            except Exception:
                pass
        # Volume boost: single-pass loudnorm on the extraction wav so quiet
        # / uneven recordings transcribe cleaner. Burn is unaffected (it
        # re-encodes from the original source, never this wav).
        self.srt_norm_var = ctk.BooleanVar(value=False)
        self.srt_norm_check = ctk.CTkCheckBox(
            perf_card, text="Boost quiet audio (normalize before transcription)",
            variable=self.srt_norm_var, font=("Segoe UI", 10),
            text_color=FG_DIM, fg_color=ACCENT,
            command=self._on_srt_opt_toggled)
        self.srt_norm_check.grid(row=4, column=0, columnspan=3, sticky="w",
                                 padx=12, pady=(2, 10))
        ctk.CTkLabel(perf_card, text="(Whisper/Canary)",
                     font=("Segoe UI", 9), text_color=FG_DIM
                     ).grid(row=3, column=2, sticky="e", padx=(4, 12), pady=(2, 10))
        self._compute_callback = None

        # Language card (SRT input/output - only for engines with language choice)
        lang_card = ctk.CTkFrame(scroll, fg_color=BG_CARD, corner_radius=12)
        lang_card.grid(row=3, column=0, sticky="ew", padx=4, pady=(0, 8))
        lang_card.grid_columnconfigure(1, weight=1)
        lang_card.grid_columnconfigure(3, weight=1)
        ctk.CTkLabel(lang_card, text="Input Language:",
                     font=("Segoe UI", 10, "bold"), text_color=FG_DIM
                     ).grid(row=0, column=0, sticky="w", padx=(12, 4), pady=(10, 2))
        self.srt_input_lang_var = ctk.StringVar(value="Japanese")
        self.srt_input_lang_menu = ctk.CTkOptionMenu(
            lang_card, variable=self.srt_input_lang_var, values=SRT_LANG_DISPLAY,
            width=170, fg_color=BG_INPUT, button_color=ACCENT,
            command=self._on_srt_input_lang_changed)
        self.srt_input_lang_menu.grid(row=0, column=1, sticky="ew", padx=4, pady=(10, 2))
        ctk.CTkLabel(lang_card, text="Output Language:",
                     font=("Segoe UI", 10, "bold"), text_color=FG_DIM
                     ).grid(row=0, column=2, sticky="w", padx=(12, 4), pady=(10, 2))
        self.srt_output_lang_var = ctk.StringVar(value="English")
        self.srt_output_lang_menu = ctk.CTkOptionMenu(
            lang_card, variable=self.srt_output_lang_var, values=SRT_LANG_DISPLAY,
            width=170, fg_color=BG_INPUT, button_color=ACCENT,
            command=self._on_srt_output_lang_changed)
        self.srt_output_lang_menu.grid(row=0, column=3, sticky="ew", padx=4, pady=(10, 2))
        self.srt_lang_hint = ctk.CTkLabel(
            lang_card, text="Only Whisper Large v3 and Canary-1B support language selection",
            font=("Segoe UI", 9), text_color=FG_DIM)
        self.srt_lang_hint.grid(row=1, column=0, columnspan=4, sticky="w", padx=12, pady=(2, 10))
        self._srt_input_lang_cb = None
        self._srt_output_lang_cb = None
        self.lang_card = lang_card

        # Burn style card (subtitle size + frame preview before full encode)
        style_card = ctk.CTkFrame(scroll, fg_color=BG_CARD, corner_radius=12)
        style_card.grid(row=4, column=0, sticky="ew", padx=4, pady=(0, 8))
        style_card.grid_columnconfigure(1, weight=1)
        ctk.CTkLabel(style_card, text="Subtitle size:",
                     font=("Segoe UI", 10, "bold"), text_color=FG_DIM
                     ).grid(row=0, column=0, sticky="w", padx=(12, 4), pady=(10, 2))
        self.burn_font_size_var = ctk.IntVar(value=18)
        self.burn_font_slider = ctk.CTkSlider(
            style_card, from_=12, to=32, number_of_steps=20,
            button_color=ACCENT, progress_color=ACCENT,
            command=self._on_burn_fontsize_changed)
        self.burn_font_slider.grid(row=0, column=1, sticky="ew", padx=4, pady=(10, 2))
        self.burn_font_slider.set(18)
        self.burn_font_value = ctk.CTkLabel(style_card, text="18",
                                            font=("Segoe UI", 11, "bold"),
                                            text_color=FG_PRIMARY)
        self.burn_font_value.grid(row=0, column=2, sticky="e", padx=(4, 6), pady=(10, 2))
        self.srt_preview_btn = ctk.CTkButton(
            style_card, text="Preview Frame", font=("Segoe UI", 12),
            fg_color="#2D3748", hover_color="#4A5568", height=34,
            corner_radius=8, text_color=FG_SECONDARY,
            command=self._on_srt_preview)
        self.srt_preview_btn.grid(row=0, column=3, sticky="e", padx=(6, 12), pady=(10, 2))
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
                     text="Preview burns one frame with the current size - instant check before the full encode." + _ghint,
                     font=("Segoe UI", 9), text_color=FG_DIM, wraplength=420,
                     justify="left").grid(row=5, column=0, columnspan=4,
                                          sticky="w", padx=12, pady=(2, 10))
        self._srt_preview_cb = None
        self._preview_running = False
        # Sample controls: when no SRT exists yet, Preview transcribes just
        # this slice (start mm:ss + length) instead of the whole video.
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
        # Burn speed: exact size (2-pass) vs fast / fastest single pass.
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
        self.burn_speed_var = ctk.StringVar(value=_speed_vals[0])
        self.burn_speed_menu = ctk.CTkOptionMenu(
            style_card, variable=self.burn_speed_var,
            values=_speed_vals, width=220,
            fg_color=BG_INPUT, button_color=ACCENT,
            command=self._on_burn_speed_changed)
        self.burn_speed_menu.grid(row=2, column=1, columnspan=3, sticky="w",
                                  padx=4, pady=(2, 2))
        self.burn_speed_desc = ctk.CTkLabel(
            style_card, text="", font=("Segoe UI", 10),
            text_color=ACCENT_GLOW, wraplength=420, justify="left")
        self.burn_speed_desc.grid(row=4, column=0, columnspan=4,
                                  sticky="w", padx=12, pady=(0, 2))
        self._refresh_burn_speed_desc()
        # Manual video bitrate: Auto (size-match budget) or a fixed kbps.
        # Manual trades the size guarantee for control - the log reports the
        # expected total so the trade is explicit before encoding.
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

        # Progress card
        prog_card = ctk.CTkFrame(scroll, fg_color=BG_CARD, corner_radius=12)
        prog_card.grid(row=5, column=0, sticky="ew", padx=4, pady=(0, 8))
        prog_card.grid_columnconfigure(0, weight=1)
        toprow = ctk.CTkFrame(prog_card, fg_color="transparent")
        toprow.pack(fill="x", padx=12, pady=(10, 2))
        ctk.CTkLabel(toprow, text="Progress",
                     font=("Segoe UI", 11, "bold"), text_color=FG_DIM).pack(side="left")
        self.srt_pct = ctk.CTkLabel(toprow, text="0%",
                                    font=("Segoe UI", 11, "bold"), text_color=FG_SECONDARY)
        self.srt_pct.pack(side="right")
        self.srt_bar = ctk.CTkProgressBar(prog_card, fg_color=BG_INPUT,
                                          progress_color=ACCENT, height=14,
                                          corner_radius=7)
        self.srt_bar.pack(fill="x", padx=12, pady=(0, 4))
        self.srt_bar.set(0)
        self.srt_status = ctk.CTkLabel(prog_card, text="Idle - pick a file to begin",
                                       font=("Segoe UI", 10), text_color=FG_DIM,
                                       wraplength=400, justify="left")
        self.srt_status.pack(anchor="w", padx=12, pady=(0, 4))
        self.srt_log_box = ctk.CTkTextbox(prog_card, font=("Consolas", 10),
                                          fg_color=BG_INPUT, text_color=FG_SECONDARY,
                                          corner_radius=8, height=110, wrap="word")
        self.srt_log_box.pack(fill="x", padx=10, pady=(0, 6))
        self.srt_log_box.insert("1.0", "SRT log ready.\n")
        self.srt_log_box.configure(state="disabled")

        # Action buttons - always visible at bottom of tab
        abtn = ctk.CTkFrame(scroll, fg_color="transparent")
        abtn.grid(row=6, column=0, sticky="ew", padx=4, pady=(0, 8))
        abtn.grid_columnconfigure(0, weight=2)
        abtn.grid_columnconfigure(1, weight=1)
        self.srt_start_btn = ctk.CTkButton(
            abtn, text="\u25B6  Generate SRT", font=("Segoe UI", 13, "bold"),
            fg_color=SUCCESS, hover_color="#00916E", height=42,
            corner_radius=10, command=self._on_srt_start)
        self.srt_start_btn.grid(row=0, column=0, padx=(0, 4), sticky="ew")
        self.srt_cancel_btn = ctk.CTkButton(
            abtn, text="Cancel", font=("Segoe UI", 12),
            fg_color=DANGER, hover_color="#C0392B", height=42,
            corner_radius=10, state="disabled", command=self._on_srt_cancel)
        self.srt_cancel_btn.grid(row=0, column=1, padx=(4, 0), sticky="ew")
        self.srt_burn_btn = ctk.CTkButton(
            abtn, text="Burn SRT into MP4", font=("Segoe UI", 12, "bold"),
            fg_color="#B5651D", hover_color="#8E4E15", height=42,
            corner_radius=10, command=self._on_srt_burn)
        self.srt_burn_btn.grid(row=1, column=0, padx=(0, 4), pady=(8, 0), sticky="ew")
        ctk.CTkButton(abtn, text="Open Folder", font=("Segoe UI", 12),
                      fg_color="#2D3748", hover_color="#4A5568", height=42,
                      corner_radius=10, text_color=FG_SECONDARY,
                      command=self._on_srt_open_folder
                      ).grid(row=1, column=1, padx=(4, 0), pady=(8, 0), sticky="ew")
        self._srt_burn_cb = None
        self._srt_mode = "generate"  # single-button rule: generate | burn
        self._srt_running_mode = "generate"
        # One-click hardcode: with SRTs missing, generate them and burn the
        # MP4s in the same run (files already carrying an SRT burn too).
        self.burn_after_var = ctk.BooleanVar(value=False)
        self.burn_after_check = ctk.CTkCheckBox(
            scroll, text="Burn MP4 automatically after SRT (one-click hardcode)",
            variable=self.burn_after_var, font=("Segoe UI", 11),
            text_color=FG_PRIMARY, fg_color=ACCENT,
            command=self._on_srt_opt_toggled)
        self.burn_after_check.grid(row=7, column=0, sticky="w", padx=16, pady=(0, 2))
        # Finish behavior: checked shutdown powers the PC off 60s after a
        # FULLY successful job (abort with `shutdown /a`); otherwise an
        # optional pop-up + window focus fires once per finished job.
        finrow = ctk.CTkFrame(scroll, fg_color="transparent")
        finrow.grid(row=8, column=0, sticky="ew", padx=16, pady=(0, 8))
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

    def _on_tab_changed(self, value=None):
        # CTkTabview calls command() without args on some versions, with the
        # tab name on others - resolve via get() either way.
        try:
            name = self.tabs.get()
        except Exception:
            name = value
        if self._tab_callback and name in ("Live", "SRT File"):
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
                                       hover_color="#C0392B")
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
            # Menu shows badged labels ("Large v3 (3GB, best) ✓"); the app
            # always gets the BASE label back.
            try:
                base = (self._model_value_map or {}).get(value, value)
            except Exception:
                base = value
            self._model_callback(base)

    def _refresh_srt_engine_label(self):
        try:
            eng = self.engine_var.get()
            if eng in ("Canary-1B", "Whisper Large v3"):
                _t = self.canary_task_var.get()
                # SRT job uses the SRT-tab languages, not the Live-tab Src.
                try:
                    _s, _o = self.get_srt_lang_codes()
                except Exception:
                    _s, _o = self.canary_lang_var.get(), "en"
                _tgt = "en" if _t == "translate" else _o
                self.set_srt_engine_label(f"{eng} ({_t} {_s}->{_tgt})")
            else:
                try:
                    _mbase = (self._model_value_map or {}).get(
                        self.model_var.get(), self.model_var.get())
                except Exception:
                    _mbase = self.model_var.get()
                self.set_srt_engine_label(f"Moonshine v2 ({_mbase})")
        except Exception:
            pass

    def _on_engine_changed(self, value):
        # Update UI state immediately
        self._refresh_srt_engine_label()
        is_heavy = (value in ("Canary-1B", "Whisper Large v3"))
        # Moonshine sizes + Whisper downloadable sizes share the Model menu;
        # Canary-1B has one fixed model. Task/Src serve Canary + Whisper.
        try:
            self.model_menu.configure(state=self._model_menu_state(value))
            self.canary_task_menu.configure(state="normal" if is_heavy else "disabled")
            self.canary_lang_menu.configure(state="normal" if is_heavy else "disabled")
        except Exception:
            pass
        if self._engine_callback:
            self._engine_callback(value)
        # AFTER the app callback: config now holds the NEW engine, so the
        # coercion callback below routes to the correct engine's keys.
        # (Doing it before would save e.g. Whisper's Src under Canary.)
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
        # Model row is usable for Moonshine sizes and Whisper downloadable
        # sizes; Canary-1B has a single fixed model.
        return ("normal" if engine_kind in ("Moonshine v2", "Whisper Large v3")
                else "disabled")

    def set_model_options(self, values, current: str, callback: Callable,
                          status_fn: Optional[Callable] = None):
        """Swap the Model row to another engine's choices (Whisper sizes /
        Canary fixed label) and select `current`. status_fn(base_label) may
        return a badge ("✓" downloaded / "↓" needs download); the app
        callback always receives the BASE label back."""
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
        """Model storage dialog.

        get_data() -> {"items": [{engine, kind, id, label, size (bytes|None),
          downloaded, in_use}], "total": bytes}. on_delete(engine, kind, id)
        -> (ok, msg); on_delete_all() -> (ok, msg). All guarded; the dialog
        rebuilds its rows after every action.
        """
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
                                            sticky="w", padx=8, pady=(10, 2))
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
                        fg_color=DANGER if can_del else "#2D3748",
                        hover_color="#C0392B" if can_del else "#2D3748",
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
                      fg_color=DANGER, hover_color="#C0392B", height=38,
                      corner_radius=8, command=_ask_delete_all
                      ).grid(row=0, column=0, padx=(0, 4), sticky="ew")
        ctk.CTkButton(foot, text="Close", font=("Segoe UI", 12),
                      fg_color="#2D3748", hover_color="#4A5568", height=38,
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
        # Shared Task/Src widgets serve Canary + Whisper (both use transcribe/translate)
        self.canary_task_var.set(task if task in CANARY_TASKS else "transcribe")
        self.canary_lang_var.set(src_lang if src_lang in CANARY_LANGS else "auto")
        # Apply initial enable/disable
        is_heavy = (engine in ("Canary-1B", "Whisper Large v3"))
        try:
            self.model_menu.configure(state=self._model_menu_state(engine))
            self.canary_task_menu.configure(state="normal" if is_heavy else "disabled")
            self.canary_lang_menu.configure(state="normal" if is_heavy else "disabled")
        except Exception:
            pass
        self._refresh_srt_engine_label()
        # Restrict menus to this engine's capabilities (coerces invalid
        # saved values, e.g. Canary Src "ja" -> "auto")
        try:
            self.refresh_lang_options(engine)
        except Exception:
            pass

    def set_model_status(self, text: str, color: str = WARNING):
        self.set_status(text, color)

    def set_record_callback(self, on_start: Callable, on_stop: Callable):
        self._on_record_start = on_start if callable(on_start) else None
        self._on_record_stop = on_stop if callable(on_stop) else None

    # ---------------- SRT tab helpers ----------------
    def _enable_drop(self, widget) -> bool:
        try:
            from tkinterdnd2 import DND_FILES
            import tkinterdnd2.TkinterDnD as _dndmod
            _dndmod._require(widget)  # load tkdnd Tcl extension into this interpreter
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
        # tkinterdnd2 wraps space-containing paths in {}
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
        """Append files to the batch queue (skips unsupported types with a
        notice). Returns the number added."""
        try:
            from srt import SUPPORTED_EXTS as _exts
        except Exception:
            _exts = ()
        added = 0
        try:
            import os as _os
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
            if added:
                try:
                    last = self._srt_input_paths[-1]
                    import os as _os
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
                    "1.0", "Drag && drop video/audio files here,\nor click Browse Files...")
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
        """{queue_index: out_path} for files whose SRT (kind='srt') or burned
        MP4 (kind='burn') already exists. Pure check, no side effects."""
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
        """Single-button rule: exactly one primary action is visible.

        "generate" -> big green Generate SRT; "burn" (every queued file
        already has an SRT) -> the same slot becomes Convert to MP4."""
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
        """The one-click-hardcode tickbox only makes sense while generating."""
        try:
            self.burn_after_check.configure(state="normal" if enabled else "disabled")
        except Exception:
            pass

    def get_srt_options(self) -> dict:
        """Snapshot of the four SRT-tab switches (GUI thread only)."""
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
        """Restore the four switches from config at startup. Never raises."""
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
        """One-shot finish alert (GUI thread only): bring the window up and
        pop a message box. Topmost is flashed, never left on."""
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
        """Single-button rule (only when idle - set_srt_running owns states
        mid-run): every queued file already has an SRT -> the primary slot
        becomes Convert to MP4; otherwise it is Generate SRT (plus the
        burn-after tickbox for one-click hardcode)."""
        if getattr(self, "_srt_running", False):
            return
        try:
            paths = self.get_srt_input_paths()
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
        """Update one queue row's status text (safe from any thread via
        the app's after() pump - but defensive try/except anyway)."""
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
            return  # never wipe the queue mid-run
        self._srt_input_paths = []
        self._srt_file_status = {}
        self._refresh_srt_list()
        try:
            self.update_action_states()
        except Exception:
            pass

    def _srt_browse_outdir(self):
        try:
            from tkinter import filedialog
            d = filedialog.askdirectory(title="SRT output folder (Cancel = same as video)")
            if d:
                self.srt_out_entry.delete(0, "end")
                self.srt_out_entry.insert(0, d)
                # Existence of SRTs depends on the folder - re-evaluate.
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
        """code in auto/cpu/gpu. On boxes without dGPU the menu is a locked
        'CPU only' row and any code maps to it."""
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

    def _on_burn_vbr_changed(self, value):
        try:
            n = max(300, min(10000, int(round(float(value) / 100.0)) * 100))
        except Exception:
            n = 2000
        try:
            self.burn_vbr_value.configure(text=f"{n}k")
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

    def get_burn_vbr(self):
        """(auto_bool, kbps) manual video-bitrate snapshot."""
        try:
            auto = bool(self.burn_vbr_auto_var.get())
        except Exception:
            auto = True
        if auto:
            return True, 0
        try:
            n = max(300, min(10000, int(round(float(self.burn_vbr_slider.get()) / 100.0)) * 100))
        except Exception:
            return True, 0
        return False, n

    def _refresh_burn_speed_desc(self):
        """One-line trade-off note for the selected burn speed. NVENC rows
        also state availability on this machine."""
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
            self.burn_speed_desc.configure(text=text)
        except Exception:
            pass

    def _on_burn_speed_changed(self, value):
        try:
            self._refresh_burn_speed_desc()
        except Exception:
            pass
        # NVENC entries stay visible (discoverability) but refuse without
        # hardware: revert to exact-match and say why (same pattern as the
        # Whisper-model busy revert).
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
        """(start_text, length_seconds) snapshot for the sample preview."""
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
        # Overwrite guard: existing SRTs ask first (all / per-file / abort).
        # Skipped rows are marked and EXCLUDED via the order map - the queue
        # itself is never mutated, so list rows stay truthful.
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
        if self._srt_start_cb:
            # Snapshot ALL Tk state here on the GUI thread. The callback
            # runs on a worker thread where any widget access risks
            # Tcl thread errors and torn reads.
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
            # Snapshot the Task widget too: the SRT label is rendered from it,
            # so the job must use THIS value, not a possibly-diverged config
            # key (a "translate" label with a "transcribe" job was observed
            # in the wild - Japanese SRT despite Output: English).
            try:
                srt_task = (self.canary_task_var.get() or "").strip().lower()
            except Exception:
                srt_task = ""
            if srt_task not in ("transcribe", "translate"):
                srt_task = ""
            # One-click switches + burn settings for a possible auto-burn
            # phase (same snapshot discipline: worker never touches widgets).
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
            self._srt_running_mode = "generate"
            threading.Thread(target=self._srt_start_cb,
                             args=([paths[i] for i in order], order,
                                   out_dir, cpu,
                                   srt_in, srt_out, srt_task,
                                   bool(opts.get("srt_norm", False)),
                                   bool(opts.get("burn_after", False)),
                                   _bfont, _bspd, _bvauto, _bvk),
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
        # Burn pre-checks on the GUI thread: files without SRT cannot burn
        # (marked, excluded); existing burned outputs ask first.
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
            # Same GUI-thread snapshot discipline as Generate: the worker
            # thread must never touch Tk widgets.
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
            self._srt_running_mode = "burn"
            threading.Thread(target=self._srt_burn_cb,
                             args=([paths[i] for i in order], order, out_dir, cpu,
                                   self.get_burn_font_size(), speed,
                                   vbr_auto, vbr_kbps),
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
        """Compat wrapper (old single-file API): replace queue with one file."""
        try:
            if getattr(self, "_srt_running", False):
                return
            self._srt_input_paths = []
            self._srt_file_status = {}
        except Exception:
            pass
        self.add_srt_files([path])

    def set_srt_callbacks(self, on_start: Callable, on_cancel: Callable,
                          on_burn: Optional[Callable] = None):
        # Coerce non-callables (e.g. a threading.Event wired by mistake -
        # that once crashed Cancel threads with "Event object is not
        # callable" while the event itself never got set).
        self._srt_start_cb = on_start if callable(on_start) else None
        self._srt_cancel_cb = on_cancel if callable(on_cancel) else None
        self._srt_burn_cb = on_burn if callable(on_burn) else None

    def set_srt_engine_label(self, text: str):
        try:
            self.srt_engine_label.configure(text=text)
        except Exception:
            pass

    def set_srt_running(self, running: bool):
        self._srt_running = running
        try:
            if running:
                # The single-button rule holds mid-run: only the starter stays.
                try:
                    self._show_srt_action(
                        getattr(self, "_srt_running_mode", None) or "generate")
                except Exception:
                    pass
            self.srt_start_btn.configure(state="disabled" if running else "normal")
            self.srt_cancel_btn.configure(state="normal" if running else "disabled")
            try:
                self.srt_burn_btn.configure(state="disabled" if running else "normal")
            except Exception:
                pass
            # Freeze the queue itself mid-run (the job owns a snapshot).
            for w in (getattr(self, "srt_browse_btn", None),
                      getattr(self, "srt_clear_btn", None)):
                try:
                    if w is not None:
                        w.configure(state="disabled" if running else "normal")
                except Exception:
                    pass
            try:
                self.srt_file_list.configure(state="disabled" if running else "normal")
            except Exception:
                pass
            if running:
                self.srt_bar.set(0)
                self.srt_pct.configure(text="0%")
            else:
                # Job ended: outputs changed on disk - re-apply grey rules
                # (buttons only; srt_done() owns the final message + bar).
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
            # Cap the log: hours-long batches would otherwise grow this
            # widget unbounded (Tk text slows dramatically past ~1k lines).
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

    def srt_done(self, ok: bool, msg: str):
        self.set_srt_running(False)
        try:
            self.srt_status.configure(text=msg, text_color=SUCCESS if ok else DANGER)
        except Exception:
            pass
        self.srt_log(f"{'DONE' if ok else 'FAILED'}: {msg}")

    # ---------------- SRT language helpers ----------------
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
        # Convert code -> full display name
        in_display = SRT_LANG_NAMES.get(input_lang, "")
        out_display = SRT_LANG_NAMES.get(output_lang, "")
        if in_display in SRT_LANG_DISPLAY:
            self.srt_input_lang_var.set(in_display)
        if out_display in SRT_LANG_DISPLAY:
            self.srt_output_lang_var.set(out_display)

    def get_srt_lang_codes(self) -> Tuple[str, str]:
        """Return (input_code, output_code) from current full-name selections."""
        in_display = self.srt_input_lang_var.get()
        out_display = self.srt_output_lang_var.get()
        in_code = SRT_LANG_CODE_FROM_DISPLAY.get(in_display, "auto")
        out_code = SRT_LANG_CODE_FROM_DISPLAY.get(out_display, "en")
        return in_code, out_code

    def refresh_lang_options(self, engine_kind: str = ""):
        """Restrict every language menu to what the active engine can do.

        Canary-1B only has <|de|en|es|fr|> tokens - offering Japanese merely
        produces an error, so it is not offered at all. Selections that
        become invalid are coerced to a safe default (live Src fires its
        callback so config + engine stay in sync; SRT widgets are silent
        because the app restores per-engine values from config on switch
        and the job snapshots the widgets at start).
        """
        eng = engine_kind or self.engine_var.get()
        is_canary = (eng == "Canary-1B")
        live_codes = CANARY_LIVE_SRC_CODES if is_canary else CANARY_LANGS
        if is_canary:
            srt_in_codes, srt_out_codes = CANARY_SRT_IN_CODES, CANARY_SRT_OUT_CODES
        elif eng == "Whisper Large v3":
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
        # Coerce live Src (fires callback -> config + engine follow)
        try:
            if self.canary_lang_var.get() not in live_codes:
                fallback = "auto" if "auto" in live_codes else live_codes[0]
                self.canary_lang_var.set(fallback)
                self._on_canary_lang_changed(fallback)
        except Exception:
            pass
        # Coerce SRT widgets silently (job snapshots widgets; app restores
        # each engine's values from config on the next switch)
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
        """Enable/disable language dropdowns based on engine. Only Canary+Whisper have real choice."""
        has_lang = engine_kind in ("Canary-1B", "Whisper Large v3")
        try:
            state = "normal" if has_lang else "disabled"
            self.srt_input_lang_menu.configure(state=state)
            self.srt_output_lang_menu.configure(state=state)
            if engine_kind == "Canary-1B":
                hint = "Canary-1B supports English, German, Spanish, French only (translate always outputs English)"
            elif engine_kind == "Whisper Large v3":
                hint = "SRT subtitles use these languages (translate task always outputs English)"
            else:
                hint = "Only Whisper Large v3 and Canary-1B support language selection"
            self.srt_lang_hint.configure(text=hint)
        except Exception:
            pass
        try:
            self._refresh_srt_engine_label()
        except Exception:
            pass
