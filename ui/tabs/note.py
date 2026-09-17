# ui/tabs/note.py
# Note-tab mic dictation: recorder wiring, chunk flow, engine row, save/copy/audio export.
# (Split verbatim from gui.py in v1.4.0 - no behavior change.)
# Palette names below are import-time snapshots kept fresh by
# MoonshineGUI.set_theme, which pushes every toggle into this module.
import customtkinter as ctk
import time
from pathlib import Path
from ui.theme import ACCENT, ACCENT_DARK, SUCCESS, WARNING, DANGER, BG_CARD, BG_INPUT, FG_PRIMARY, FG_SECONDARY, FG_DIM, BTN_DIM, BTN_DIM_HOVER, BTN_GO_HOVER, BTN_DANGER_HOVER, BTN_TEXT
from ui.widgets import LevelMeter
from ui.choices import MODEL_CHOICES, MODEL_CHOICES_REV, WHISPER_MODEL_CHOICES, WHISPER_MODEL_CHOICES_REV, CANARY_MODEL_LABEL, ENGINE_CHOICES


class NoteTabMixin:
    """Note-tab mic dictation: recorder wiring, chunk flow, engine row, save/copy/audio export."""

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
        # File-import state now lives on the Import tab (import_* attrs).
        # note_file_* names below are kept as aliases (see _build_import_tab).

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

        # NOTE: file import lived here until v1.4.0 - it now has its own
        # Import tab (see _build_import_tab). Note keeps mic recording only.

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

        # Tail audio (everything up to the click) becomes one last chunk:
        # the mic stops now, but the transcriber drains it after Stop.
        try:
            tail_idx = int(self._note_recorder.chunk_count) + 1
        except Exception:
            tail_idx = int(getattr(self, "_note_submitted", 0) or 0) + 1
        remaining = self._note_recorder.stop()
        if remaining is not None and len(remaining) > 16000:
            try:
                accepted = (self._note_transcriber.submit_chunk(remaining, tail_idx)
                            if self._note_transcriber else False)
            except Exception:
                accepted = False
            if accepted is not False:
                try:
                    if tail_idx > int(getattr(self, "_note_submitted", 0) or 0):
                        self._note_submitted = tail_idx
                except Exception:
                    pass

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
        try:
            _pend = (int(getattr(self, "_note_submitted", 0) or 0)
                     - int(getattr(self, "_note_done", 0) or 0))
        except Exception:
            _pend = 0
        if _pend > 0:
            self.note_status_label.configure(
                text=f"Finishing last {_pend} chunk{'s' if _pend != 1 else ''}…",
                text_color=WARNING)
        else:
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
        try:
            # Drain finished after Stop: replace only our own interim status.
            if (not getattr(self, "_note_recording", False)
                    and int(getattr(self, "_note_done", 0) or 0)
                    >= int(getattr(self, "_note_submitted", 0) or 0) > 0
                    and "Finishing" in str(self.note_status_label.cget("text"))):
                self.note_status_label.configure(text="Stopped - review and save",
                                                 text_color=SUCCESS)
        except Exception:
            pass

    def _note_on_chunk(self, audio, index):
        """Called from recorder thread when a chunk is ready."""
        try:
            accepted = True
            if self._note_transcriber:
                accepted = self._note_transcriber.submit_chunk(audio, index)
        except Exception:
            accepted = True
        if accepted is False:
            return  # queue full: dropped, and _note_done will never come
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

    def note_processing_pending(self):
        """(recording, pending_chunks) snapshot for the close path.
        Never raises."""
        try:
            recording = bool(getattr(self, "_note_recording", False))
        except Exception:
            recording = False
        try:
            pending = (int(getattr(self, "_note_submitted", 0) or 0)
                       - int(getattr(self, "_note_done", 0) or 0))
        except Exception:
            pending = 0
        return recording, max(0, pending)

    def confirm_note_processing(self) -> bool:
        """True = proceed with close. Warns when the mic is still open,
        chunks are still transcribing, or a file import is running -
        quitting then loses audio/text in flight. Never raises."""
        try:
            recording, pending = self.note_processing_pending()
        except Exception:
            return True
        try:
            file_running = bool(getattr(self, "_import_file_running", False)
                                or getattr(self, "_note_file_running", False))
        except Exception:
            file_running = False
        if not recording and pending <= 0 and not file_running:
            return True
        try:
            from tkinter import messagebox as _mb
            if recording:
                what = "still recording"
            elif file_running:
                what = "still transcribing an imported file (Import tab)"
            else:
                what = (f"{pending} chunk{'s' if pending != 1 else ''} "
                        f"still transcribing")
            return bool(_mb.askyesno(
                "Transcription in progress",
                f"Note is {what}.\nQuit now and lose it?",
                parent=self))
        except Exception:
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
