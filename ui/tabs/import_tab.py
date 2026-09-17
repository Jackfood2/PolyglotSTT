# ui/tabs/import_tab.py
# Import-tab file transcription UI + legacy note_file_* aliases.
# (Split verbatim from gui.py in v1.4.0 - no behavior change.)
# Palette names below are import-time snapshots kept fresh by
# MoonshineGUI.set_theme, which pushes every toggle into this module.
from pathlib import Path
from ui.theme import ACCENT, SUCCESS, WARNING, DANGER, BG_CARD, BG_INPUT, FG_PRIMARY, FG_SECONDARY, FG_DIM, BTN_DIM, BTN_DIM_HOVER, BTN_GO_HOVER, BTN_DANGER_HOVER, BTN_TEXT
from ui.choices import MODEL_CHOICES, MODEL_CHOICES_REV, WHISPER_MODEL_CHOICES, WHISPER_MODEL_CHOICES_REV, CANARY_MODEL_LABEL, ENGINE_CHOICES


class ImportTabMixin:
    """Import-tab file transcription UI + legacy note_file_* aliases."""

    # ── Import tab (moved out of Note in v1.4.0) ──
    def _build_import_tab(self, tab):
        """Standalone file-import tab: engine row + drag&drop file card +
        sentence output box. Own engine selection, own progress, own text.
        note_file_* attribute/method names are kept as synced aliases so
        older app wiring keeps working."""
        import customtkinter as ctk

        self._import_file_path = None
        self._import_file_running = False
        self._import_file_start_cb = None
        self._import_file_cancel_cb = None
        self._import_file_prequest_cb = None
        self._import_dirty = False

        scroll = ctk.CTkScrollableFrame(tab, fg_color="transparent")
        scroll.pack(fill="both", expand=True, padx=4, pady=4)
        scroll.grid_columnconfigure(0, weight=1)

        # ── Engine pick (per-tab: independent from Live/SRT/Note) ──
        info_card = ctk.CTkFrame(scroll, fg_color=BG_CARD, corner_radius=12)
        info_card.grid(row=0, column=0, sticky="ew", padx=4, pady=(0, 6))
        info_card.grid_columnconfigure(1, weight=1)

        self.import_engine_label = ctk.CTkLabel(
            info_card, text="Engine: following shared default",
            font=("Segoe UI", 10), text_color=FG_DIM)
        self.import_engine_label.grid(row=0, column=0, columnspan=3,
                                      sticky="w", padx=16, pady=(10, 2))
        ctk.CTkLabel(info_card, text="Engine:",
                     font=("Segoe UI", 10, "bold"),
                     text_color=FG_DIM).grid(row=1, column=0, sticky="w",
                                             padx=(16, 4), pady=(2, 10))
        self.import_engine_var = ctk.StringVar(value="Moonshine v2")
        self.import_engine_menu = ctk.CTkOptionMenu(
            info_card, variable=self.import_engine_var,
            values=list(ENGINE_CHOICES), width=150,
            fg_color=BG_INPUT, button_color=ACCENT,
            command=self._on_import_engine_changed)
        self.import_engine_menu.grid(row=1, column=1, sticky="ew",
                                     padx=4, pady=(2, 10))
        self.import_model_var = ctk.StringVar(value="")
        self.import_model_menu = ctk.CTkOptionMenu(
            info_card, variable=self.import_model_var,
            values=[], width=190,
            fg_color=BG_INPUT, button_color=ACCENT,
            command=self._on_import_model_changed)
        self.import_model_menu.grid(row=1, column=2, sticky="ew",
                                    padx=(4, 16), pady=(2, 10))
        self._import_engine_cb = None
        self._import_model_cb = None
        try:
            self._refresh_import_model_menu()
        except Exception:
            pass

        # ── File card (drag & drop -> sentences) ──
        import_card = ctk.CTkFrame(scroll, fg_color=BG_CARD, corner_radius=16)
        import_card.grid(row=1, column=0, sticky="ew", padx=4, pady=(0, 8))
        import_card.grid_columnconfigure(0, weight=1)
        ctk.CTkLabel(import_card, text="Import Audio / Video File",
                     font=("Segoe UI", 12, "bold"),
                     text_color=FG_DIM).pack(anchor="w", padx=16, pady=(12, 2))
        ctk.CTkLabel(import_card,
                     text="Drop a wav, mp3, mp4, mkv… file below (or Browse). "
                          "It transcribes with this tab's engine and appends "
                          "one sentence per line. Save as TXT (UTF-8) or Copy All.",
                     font=("Segoe UI", 10), text_color=FG_DIM,
                     wraplength=400, justify="left"
                     ).pack(anchor="w", padx=16, pady=(0, 6))
        self.import_file_drop = ctk.CTkTextbox(
            import_card, font=("Segoe UI", 11),
            fg_color=BG_INPUT, text_color=FG_SECONDARY,
            corner_radius=8, height=52,
            activate_scrollbars=False, wrap="word")
        self.import_file_drop.pack(fill="x", padx=12, pady=(0, 6))
        self.import_file_drop.insert(
            "1.0", "Drag & drop audio/video here,\nor click Browse...")
        self.import_file_drop.configure(state="disabled")
        try:
            self._enable_drop_to(self.import_file_drop,
                                 self._on_import_drop_files)
        except Exception:
            try:
                self._enable_drop(self.import_file_drop)
            except Exception:
                pass
        try:
            self._enable_drop_to(import_card, self._on_import_drop_files)
        except Exception:
            pass
        self.import_file_label = ctk.CTkLabel(
            import_card, text="No file selected",
            font=("Segoe UI", 10), text_color=FG_DIM,
            wraplength=400, justify="left")
        self.import_file_label.pack(anchor="w", padx=16, pady=(0, 6))
        file_btn_row = ctk.CTkFrame(import_card, fg_color="transparent")
        file_btn_row.pack(fill="x", padx=12, pady=(0, 6))
        file_btn_row.grid_columnconfigure(0, weight=1)
        file_btn_row.grid_columnconfigure(1, weight=1)
        file_btn_row.grid_columnconfigure(2, weight=1)
        self.import_file_browse_btn = ctk.CTkButton(
            file_btn_row, text="Browse...", font=("Segoe UI", 12),
            fg_color=BTN_DIM, hover_color=BTN_DIM_HOVER, height=36,
            corner_radius=8, text_color=FG_SECONDARY,
            command=self._import_browse_file)
        self.import_file_browse_btn.grid(row=0, column=0, padx=(0, 4), sticky="ew")
        self.import_file_start_btn = ctk.CTkButton(
            file_btn_row, text="▶  Transcribe File", font=("Segoe UI", 12, "bold"),
            fg_color=SUCCESS, hover_color=BTN_GO_HOVER, height=36,
            corner_radius=8, text_color=BTN_TEXT,
            command=self._on_import_file_start)
        self.import_file_start_btn.grid(row=0, column=1, padx=(4, 4), sticky="ew")
        self.import_file_cancel_btn = ctk.CTkButton(
            file_btn_row, text="Cancel", font=("Segoe UI", 12),
            fg_color=DANGER, hover_color=BTN_DANGER_HOVER, height=36,
            corner_radius=8, text_color=BTN_TEXT, state="disabled",
            command=self._on_import_file_cancel)
        self.import_file_cancel_btn.grid(row=0, column=2, padx=(4, 0), sticky="ew")
        self.import_file_bar = ctk.CTkProgressBar(
            import_card, fg_color=BG_INPUT, progress_color=ACCENT,
            height=10, corner_radius=6)
        self.import_file_bar.pack(fill="x", padx=12, pady=(0, 4))
        try:
            self.import_file_bar.set(0)
        except Exception:
            pass
        file_prog_row = ctk.CTkFrame(import_card, fg_color="transparent")
        file_prog_row.pack(fill="x", padx=16, pady=(0, 12))
        self.import_file_pct = ctk.CTkLabel(
            file_prog_row, text="0%", font=("Segoe UI", 10, "bold"),
            text_color=FG_SECONDARY)
        self.import_file_pct.pack(side="right")
        self.import_file_status = ctk.CTkLabel(
            file_prog_row, text="Idle - pick a file to begin",
            font=("Segoe UI", 10), text_color=FG_DIM,
            wraplength=320, justify="left")
        self.import_file_status.pack(side="left")

        # ── Output card (own sentences box) ──
        out_card = ctk.CTkFrame(scroll, fg_color=BG_CARD, corner_radius=16)
        out_card.grid(row=2, column=0, sticky="nsew", padx=4, pady=(0, 8))
        out_card.grid_columnconfigure(0, weight=1)
        out_card.grid_rowconfigure(1, weight=1)
        out_header = ctk.CTkFrame(out_card, fg_color="transparent")
        out_header.grid(row=0, column=0, sticky="ew", padx=16, pady=(12, 4))
        ctk.CTkLabel(out_header, text="Imported Sentences",
                     font=("Segoe UI", 12, "bold"),
                     text_color=FG_DIM).pack(side="left")
        self.import_save_btn = ctk.CTkButton(
            out_header, text="Save as TXT", width=110, height=30,
            font=("Segoe UI", 11),
            fg_color=SUCCESS, hover_color=BTN_GO_HOVER,
            text_color=BTN_TEXT, corner_radius=8,
            command=self._on_import_save)
        self.import_save_btn.pack(side="right")
        self.import_copy_btn = ctk.CTkButton(
            out_header, text="Copy All", width=90, height=30,
            font=("Segoe UI", 11),
            fg_color=BTN_DIM, hover_color=BTN_DIM_HOVER,
            text_color=FG_SECONDARY, corner_radius=8,
            command=self._on_import_copy)
        self.import_copy_btn.pack(side="right", padx=(0, 6))
        self.import_clear_btn = ctk.CTkButton(
            out_header, text="Clear", width=70, height=30,
            font=("Segoe UI", 11),
            fg_color=BTN_DIM, hover_color=BTN_DIM_HOVER,
            text_color=FG_SECONDARY, corner_radius=8,
            command=self._on_import_clear)
        self.import_clear_btn.pack(side="right", padx=(0, 6))
        self.import_text = ctk.CTkTextbox(
            out_card, font=("Segoe UI", 13),
            fg_color=BG_INPUT, text_color=FG_PRIMARY,
            corner_radius=10, wrap="word", height=280)
        self.import_text.grid(row=1, column=0, sticky="nsew", padx=12, pady=(4, 12))
        self.import_text.insert("1.0", "Imported sentences will appear here...")
        try:
            self.import_text.bind("<Key>", self._import_mark_dirty)
        except Exception:
            pass
        self.import_status_label = ctk.CTkLabel(
            out_card, text="Ready - pick a file above",
            font=("Segoe UI", 10), text_color=FG_DIM)
        self.import_status_label.grid(row=2, column=0, sticky="w",
                                      padx=16, pady=(0, 12))

        # Backward-compat aliases: older app code drives note_file_*.
        try:
            self.note_file_drop = self.import_file_drop
            self.note_file_label = self.import_file_label
            self.note_file_browse_btn = self.import_file_browse_btn
            self.note_file_start_btn = self.import_file_start_btn
            self.note_file_cancel_btn = self.import_file_cancel_btn
            self.note_file_bar = self.import_file_bar
            self.note_file_pct = self.import_file_pct
            self.note_file_status = self.import_file_status
            self._note_file_path = None
            self._note_file_running = False
            self._note_file_start_cb = None
            self._note_file_cancel_cb = None
            self._note_file_prequest_cb = None
        except Exception:
            pass

    def _import_model_options(self, kind):
        try:
            if kind == "Whisper":
                return list(WHISPER_MODEL_CHOICES.keys())
            if kind == "Canary-1B":
                return [CANARY_MODEL_LABEL]
            return list(MODEL_CHOICES.keys())
        except Exception:
            return []

    def _import_menu_label(self, kind, arch=None, wmid=None):
        try:
            if kind == "Whisper":
                return WHISPER_MODEL_CHOICES_REV.get(str(wmid or "large-v3"))
            if kind == "Canary-1B":
                return CANARY_MODEL_LABEL
            return MODEL_CHOICES_REV.get(int(arch if arch is not None else 5))
        except Exception:
            return None

    def set_import_engine_state(self, kind, arch=None, wmid=None):
        """Programmatic set (mirror/adopt/restore). No callbacks fired."""
        try:
            if kind in ENGINE_CHOICES:
                self.import_engine_var.set(kind)
        except Exception:
            pass
        try:
            self._refresh_import_model_menu()
        except Exception:
            pass
        try:
            label = self._import_menu_label(kind, arch, wmid)
            if label:
                try:
                    vals = list(self.import_model_menu.cget("values"))
                except Exception:
                    vals = []
                if label in vals:
                    self.import_model_var.set(label)
        except Exception:
            pass
        try:
            self.import_engine_label.configure(
                text=f"Engine: {kind}" + (f" ({label})" if label else ""))
        except Exception:
            pass

    def get_import_engine_kind(self):
        try:
            k = (self.import_engine_var.get() or "Moonshine v2").strip()
        except Exception:
            k = "Moonshine v2"
        return k if k in ENGINE_CHOICES else "Moonshine v2"

    # ── Import file API (primary; note_file_* delegate below) ──
    def _import_mark_dirty(self, event=None):
        try:
            self._import_dirty = True
        except Exception:
            pass

    def _import_browse_file(self):
        if getattr(self, "_import_file_running", False):
            return
        try:
            from tkinter import filedialog
            try:
                from srt import SUPPORTED_EXTS
                exts = " ".join(f"*{e}" for e in SUPPORTED_EXTS)
            except Exception:
                exts = "*.wav *.mp3 *.mp4 *.mkv *.m4a"
            paths = filedialog.askopenfilenames(
                title="Pick an audio or video file to import",
                filetypes=[("Media", exts), ("All files", "*.*")])
            if paths:
                self.add_import_files(list(paths))
        except Exception:
            pass

    def add_import_files(self, paths) -> int:
        """Queue a single file for Import (last valid file wins).
        Returns 1 when accepted, else 0. Never raises."""
        try:
            try:
                from srt import SUPPORTED_EXTS as _exts
            except Exception:
                _exts = ()
            import os as _os
            picked = None
            for raw in (paths or []):
                p = str(raw or "").strip().strip('"')
                if not p:
                    continue
                if _exts and _os.path.splitext(p)[1].lower() not in _exts:
                    try:
                        self.set_import_file_progress(
                            0, f"skip (unsupported type): {_os.path.basename(p)}")
                    except Exception:
                        pass
                    continue
                if not _os.path.exists(p):
                    continue
                picked = p
            if picked is None:
                return 0
            if getattr(self, "_import_file_running", False):
                try:
                    self.set_import_file_progress(0, "Busy - wait for current file")
                except Exception:
                    pass
                return 0
            self._import_file_path = picked
            try:
                self._note_file_path = picked
            except Exception:
                pass
            try:
                import os as _os2
                self.import_file_label.configure(
                    text=f"Selected: {_os2.path.basename(picked)}")
            except Exception:
                pass
            try:
                self.import_file_drop.configure(state="normal")
                self.import_file_drop.delete("1.0", "end")
                import os as _os3
                self.import_file_drop.insert(
                    "1.0", f"Ready: {_os3.path.basename(picked)}\n"
                           "Press Transcribe File.")
                self.import_file_drop.configure(state="disabled")
            except Exception:
                pass
            try:
                self.set_import_file_progress(0, "Ready - press Transcribe File")
            except Exception:
                pass
            return 1
        except Exception:
            return 0

    def get_import_file_path(self):
        try:
            return str(getattr(self, "_import_file_path", None) or "")
        except Exception:
            return ""

    def set_import_file_callbacks(self, on_start=None, on_cancel=None,
                                  on_prequest=None):
        try:
            self._import_file_start_cb = on_start if callable(on_start) else None
            self._import_file_cancel_cb = on_cancel if callable(on_cancel) else None
            self._import_file_prequest_cb = (
                on_prequest if callable(on_prequest) else None)
            # Keep legacy mirrors in sync for older callers.
            self._note_file_start_cb = self._import_file_start_cb
            self._note_file_cancel_cb = self._import_file_cancel_cb
            self._note_file_prequest_cb = self._import_file_prequest_cb
        except Exception:
            pass

    def set_import_file_progress(self, frac: float, msg: str = ""):
        try:
            frac = max(0.0, min(1.0, float(frac)))
        except Exception:
            frac = 0.0
        try:
            self.import_file_bar.set(frac)
            self.import_file_pct.configure(text=f"{int(round(frac * 100))}%")
            if msg:
                self.import_file_status.configure(text=msg)
        except Exception:
            pass

    def set_import_file_running(self, running: bool):
        self._import_file_running = bool(running)
        try:
            self._note_file_running = bool(running)
        except Exception:
            pass
        try:
            self.import_file_start_btn.configure(
                state="disabled" if running else "normal")
        except Exception:
            pass
        try:
            self.import_file_cancel_btn.configure(
                state="normal" if running else "disabled")
        except Exception:
            pass
        try:
            self.import_file_browse_btn.configure(
                state="disabled" if running else "normal")
        except Exception:
            pass

    def import_file_done(self, ok: bool, msg: str):
        self.set_import_file_running(False)
        try:
            self.set_import_file_progress(1.0 if ok else 0.0, msg)
        except Exception:
            pass
        try:
            self.import_status_label.configure(
                text=msg, text_color=(SUCCESS if ok else DANGER))
        except Exception:
            pass

    def import_has_unsaved(self) -> bool:
        try:
            if not bool(getattr(self, "_import_dirty", False)):
                return False
            text = self.import_text.get("1.0", "end").strip()
            return bool(text) and text != "Imported sentences will appear here..."
        except Exception:
            return False

    def confirm_import_close(self) -> bool:
        try:
            if not self.import_has_unsaved():
                return True
            from tkinter import messagebox as _mb
            r = _mb.askyesnocancel(
                "Save import?",
                "You have unsaved imported sentences.\nSave before closing?",
                parent=self)
        except Exception:
            return True
        if r is None:
            return False
        if r is True:
            try:
                self._on_import_save()
            except Exception:
                return False
            try:
                return not self.import_has_unsaved()
            except Exception:
                return False
        return True

    # ── Legacy Note file-import names (v1.4.0: Import tab owns the UI) ──
    # Every method below delegates to the import_* implementation so older
    # app wiring (set_note_file_callbacks / note_file_done /
    # append_note_file_text / ...) keeps working unchanged.
    def _note_browse_file(self):
        try:
            return self._import_browse_file()
        except Exception:
            pass

    def add_note_files(self, paths) -> int:
        try:
            return int(self.add_import_files(paths) or 0)
        except Exception:
            return 0

    def _on_note_drop_files(self, event):
        try:
            return self._on_import_drop_files(event)
        except Exception:
            pass

    def get_note_file_path(self):
        try:
            return self.get_import_file_path()
        except Exception:
            return ""

    def set_note_file_callbacks(self, on_start=None, on_cancel=None,
                                on_prequest=None):
        try:
            return self.set_import_file_callbacks(on_start, on_cancel,
                                                 on_prequest)
        except Exception:
            pass

    def set_note_file_progress(self, frac: float, msg: str = ""):
        try:
            return self.set_import_file_progress(frac, msg)
        except Exception:
            pass

    def set_note_file_running(self, running: bool):
        try:
            return self.set_import_file_running(running)
        except Exception:
            pass

    def note_file_done(self, ok: bool, msg: str):
        try:
            return self.import_file_done(ok, msg)
        except Exception:
            pass

    def append_note_file_text(self, formatted_text: str):
        """Legacy entry: imported sentences now land in the Import tab box."""
        try:
            return self.append_import_file_text(formatted_text)
        except Exception:
            pass

    def _on_note_file_start(self):
        try:
            return self._on_import_file_start()
        except Exception:
            pass

    def _on_note_file_cancel(self):
        try:
            return self._on_import_file_cancel()
        except Exception:
            pass

    # ── Import engine row (per-tab, mirrors Note/SRT flow) ──
    def _on_import_engine_changed(self, value):
        try:
            self._refresh_import_model_menu()
        except Exception:
            pass
        try:
            self._request_tab_engine("import", kind_display=value)
        except Exception:
            pass

    def _on_import_model_changed(self, value):
        try:
            self._request_tab_engine("import", model_display=value)
        except Exception:
            pass

    def _on_import_drop_files(self, event):
        try:
            files = self._parse_drop(getattr(event, "data", ""))
            if files:
                self.add_import_files(files)
        except Exception:
            pass

    def _on_import_file_start(self):
        if getattr(self, "_import_file_running", False):
            return
        path = self.get_import_file_path()
        if not path:
            try:
                self.set_import_file_progress(0, "Pick a file first (Browse or drop)")
            except Exception:
                pass
            return
        if getattr(self, "_import_file_prequest_cb", None) is not None:
            try:
                _pv = self._import_file_prequest_cb() or {"go": True}
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
                        self.set_import_file_progress(0, "File transcription cancelled")
                    except Exception:
                        pass
                    return
            elif isinstance(_pv, dict) and "wait" in _pv:
                try:
                    self.set_import_file_progress(
                        0, str(_pv.get("wait") or "Engine loading..."))
                except Exception:
                    pass
                return
            elif isinstance(_pv, dict) and "abort" in _pv:
                try:
                    self.set_import_file_progress(0, "Engine not ready - try again")
                except Exception:
                    pass
                return
        if self._import_file_start_cb:
            try:
                import threading as _th
                self.set_import_file_running(True)
                self.set_import_file_progress(0.02, "Starting file transcription...")
                _th.Thread(target=self._import_file_start_cb,
                           args=(path,), daemon=True).start()
            except Exception:
                try:
                    self.set_import_file_running(False)
                except Exception:
                    pass

    def _on_import_file_cancel(self):
        if self._import_file_cancel_cb:
            try:
                import threading as _th
                _th.Thread(target=self._import_file_cancel_cb,
                           daemon=True).start()
            except Exception:
                pass

    def _on_import_clear(self):
        try:
            if getattr(self, "_import_file_running", False):
                return
            self.import_text.delete("1.0", "end")
            self.import_text.insert("1.0", "Imported sentences will appear here...")
            self._import_dirty = False
            try:
                self.import_status_label.configure(text="Ready - pick a file above",
                                                  text_color=FG_DIM)
            except Exception:
                pass
        except Exception:
            pass

    def _on_import_copy(self):
        try:
            text = self.import_text.get("1.0", "end").strip()
            if text and text != "Imported sentences will appear here...":
                from input_sim import copy_to_clipboard
                copy_to_clipboard(text)
                self.import_status_label.configure(text="Copied to clipboard",
                                                  text_color=SUCCESS)
        except Exception:
            pass

    def _on_import_save(self):
        try:
            text = self.import_text.get("1.0", "end").strip()
            if not text or text == "Imported sentences will appear here...":
                self.import_status_label.configure(text="Nothing to save",
                                                  text_color=WARNING)
                return
            from tkinter import filedialog
            import datetime
            default_name = f"import_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
            path = filedialog.asksaveasfilename(
                title="Save Import",
                defaultextension=".txt",
                initialfile=default_name,
                filetypes=[("Text files", "*.txt"), ("All files", "*.*")])
            if path:
                with open(path, "w", encoding="utf-8") as f:
                    f.write(text)
                self._import_dirty = False
                self.import_status_label.configure(
                    text=f"Saved: {Path(path).name}", text_color=SUCCESS)
        except Exception as e:
            try:
                self.import_status_label.configure(text=f"Save error: {e}",
                                                  text_color=DANGER)
            except Exception:
                pass


    def _refresh_import_model_menu(self):
        try:
            kind = (self.import_engine_var.get() or "Moonshine v2").strip()
        except Exception:
            kind = "Moonshine v2"
        try:
            vals = self._import_model_options(kind)
            if vals:
                cur = None
                try:
                    cur = self.import_model_var.get()
                except Exception:
                    cur = None
                self.import_model_menu.configure(values=vals)
                if cur in vals:
                    self.import_model_var.set(cur)
                else:
                    self.import_model_var.set(vals[0])
        except Exception:
            pass

    def append_import_file_text(self, formatted_text: str):
        """Append imported-file sentences to the Import box. Never raises."""
        try:
            txt = str(formatted_text or "").strip()
            if not txt:
                return
            try:
                current = self.import_text.get("1.0", "end").strip()
            except Exception:
                current = ""
            try:
                if (not current
                        or current == "Imported sentences will appear here..."):
                    self.import_text.delete("1.0", "end")
                else:
                    if not current.endswith("\n"):
                        self.import_text.insert("end", "\n")
                    self.import_text.insert("end", "\n")
                self.import_text.insert("end", txt + "\n\n")
                try:
                    self.import_text.see("end")
                except Exception:
                    pass
                self._import_dirty = True
            except Exception:
                pass
        except Exception:
            pass
