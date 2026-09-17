# ui/tabs/srt.py
# SRT File tab: queue, actions, progress, burn controls, languages, compute, preview.
# (Split verbatim from gui.py in v1.4.0 - no behavior change.)
# Palette names below are import-time snapshots kept fresh by
# MoonshineGUI.set_theme, which pushes every toggle into this module.
import customtkinter as ctk
import threading
from pathlib import Path
from typing import Optional, Callable, Tuple
from ui.theme import ACCENT, ACCENT_DARK, ACCENT_GLOW, SUCCESS, WARNING, DANGER, BG_CARD, BG_INPUT, FG_PRIMARY, FG_SECONDARY, FG_DIM, BTN_DIM, BTN_DIM_HOVER, BTN_GO_HOVER, BTN_DANGER_HOVER, BTN_BURN, BTN_BURN_HOVER, BTN_TEXT, theme_color
from ui.choices import MODEL_CHOICES, MODEL_CHOICES_REV, WHISPER_MODEL_CHOICES, WHISPER_MODEL_CHOICES_REV, CANARY_MODEL_LABEL, ENGINE_CHOICES, WHISPER_LANGS, SRT_LANG_NAMES, SRT_LANG_DISPLAY, SRT_LANG_CODE_FROM_DISPLAY, CANARY_LIVE_SRC_CODES, CANARY_SRT_IN_CODES, CANARY_SRT_OUT_CODES, WHISPER_SRT_IN_CODES, WHISPER_SRT_OUT_CODES, BURN_SPEED_HELP


class SrtTabMixin:
    """SRT File tab: queue, actions, progress, burn controls, languages, compute, preview."""

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
        # The Input/Output language menus belong to THIS tab's engine, not
        # Live's: re-align them on every engine paint (init, mirror,
        # adopt, revert, user pick). Without this they stayed frozen on
        # whatever the Live engine was when the app started.
        try:
            self._refresh_srt_lang_state()
        except Exception:
            pass

    def _refresh_srt_lang_state(self):
        """Align SRT language menus + hint with the SRT tab's own engine.
        Never touches the Live Src menu. Never raises."""
        try:
            kind = self.get_srt_engine_kind()
        except Exception:
            kind = "Moonshine v2"
        try:
            self.set_srt_lang_state(kind)
        except Exception:
            pass
        if kind == "Canary-1B":
            srt_in_codes = list(CANARY_SRT_IN_CODES)
            srt_out_codes = list(CANARY_SRT_OUT_CODES)
        else:
            srt_in_codes = list(WHISPER_SRT_IN_CODES)
            srt_out_codes = list(WHISPER_SRT_OUT_CODES)
        try:
            self.srt_input_lang_menu.configure(
                values=[SRT_LANG_NAMES[c] for c in srt_in_codes])
            self.srt_output_lang_menu.configure(
                values=[SRT_LANG_NAMES[c] for c in srt_out_codes])
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

    def get_srt_engine_kind(self):
        try:
            k = (self.srt_engine_var.get() or "Moonshine v2").strip()
        except Exception:
            k = "Moonshine v2"
        return k if k in ENGINE_CHOICES else "Moonshine v2"

    def set_srt_engine_callbacks(self, eng_cb, model_cb):
        self._srt_engine_cb = eng_cb if callable(eng_cb) else None
        self._srt_model_cb = model_cb if callable(model_cb) else None

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
        # engine_kind drives the LIVE Src menu only. The SRT Input/Output
        # menus follow the SRT tab's own engine (see
        # _refresh_srt_lang_state) - a Live switch must never rewrite them.
        eng = engine_kind or self.engine_var.get()
        is_canary = (eng == "Canary-1B")
        live_codes = CANARY_LIVE_SRC_CODES if is_canary else WHISPER_LANGS
        live_displays = [SRT_LANG_NAMES[c] for c in live_codes]
        try:
            self.canary_lang_menu.configure(values=live_displays)
        except Exception:
            pass
        try:
            if self.canary_lang_var.get() not in live_displays:
                fallback_code = "auto" if "auto" in live_codes else live_codes[0]
                fallback = SRT_LANG_NAMES[fallback_code]
                self.canary_lang_var.set(fallback)
                self._on_canary_lang_changed(fallback)
        except Exception:
            pass
        try:
            self._refresh_srt_lang_state()
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
