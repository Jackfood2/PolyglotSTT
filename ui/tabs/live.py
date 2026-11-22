




import threading
from typing import Optional, Callable
from ui.theme import ACCENT, ACCENT_DARK, WARNING, DANGER, FG_SECONDARY, BTN_DANGER_HOVER
from ui.choices import MODEL_CHOICES, MODEL_CHOICES_REV, ENGINE_CHOICES, CANARY_TASKS, SRT_LANG_NAMES, apply_badge_map


class LiveTabMixin:
    """Live-tab controls: record/output/history rows, engine + model menus, typing settings."""

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
        self._refresh_history_count()

    def _copy_last(self):
        from input_sim import copy_to_clipboard
        self.output_text.configure(state="normal")
        text = self.output_text.get("1.0", "end").strip()
        self.output_text.configure(state="disabled")
        if text and text not in ("Waiting for voice input...", "Cleared."):
            copy_to_clipboard(text)

    def _copy_history_all(self):
        """Copy the whole transcription history to the clipboard."""
        try:
            n = self.history.copy_all() if hasattr(self.history, "copy_all") else 0
        except Exception:
            n = 0

        try:
            btn = getattr(self, "history_copy_all_btn", None)
            if btn is not None:
                if n > 0:
                    btn.configure(text="Copied ✓")
                else:
                    btn.configure(text="Empty")
                try:
                    self.after(1200, lambda: self._restore_copy_all_btn())
                except Exception:
                    pass
        except Exception:
            pass

    def _restore_copy_all_btn(self):
        try:
            btn = getattr(self, "history_copy_all_btn", None)
            if btn is not None:
                btn.configure(text="Copy All")
        except Exception:
            pass

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
            self._refresh_history_count()

    def update_level(self, level: float):
        self.meter.set_level(level)
        self.waveform.set_amplitude(level)

    def _on_method_changed(self, value):
        if self._method_callback:
            self._method_callback(value)

    def _on_suffix_changed(self, value):
        if self._suffix_callback:
            self._suffix_callback(value)

    def _on_itn_toggled(self):
        try:
            value = bool(self.itn_var.get())
        except Exception:
            value = True
        if self._itn_callback:
            try:
                self._itn_callback(value)
            except Exception:
                pass

    def _on_itn_hint_toggled(self):
        try:
            value = bool(self.itn_hint_var.get())
        except Exception:
            value = True
        if self._itn_hint_callback:
            try:
                self._itn_hint_callback(value)
            except Exception:
                pass

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

    def _on_engine_changed(self, value):
        self._refresh_srt_engine_label()
        is_canary = value == "Canary-1B"
        is_whisper = value == "Whisper"
        is_qwen = value == "Qwen3-ASR"
        try:
            self.model_menu.configure(state=self._model_menu_state(value))
            self._sync_task_menu(value)
            self.canary_lang_menu.configure(
                state="normal" if is_canary or is_whisper or is_qwen
                else "disabled")
        except Exception:
            pass
        if self._engine_callback:
            self._engine_callback(value)
        try:
            self.refresh_lang_options(value)
        except Exception:
            pass
        self._update_live_summary()

    def _sync_task_menu(self, engine_kind):
        """Offer only the tasks this engine can actually perform.

        The task menu doubles as the SRT/Import mode selector, so an
        impossible pick (Qwen translate, Whisper turbo translate) must never
        be offered. When the stored value is no longer valid, fall back to
        the first supported task and tell the app so the config follows.
        """
        try:
            import capabilities as _cap
            values = _cap.task_options(engine_kind)
        except Exception:
            values = ["transcribe"]
        if not values:
            values = ["transcribe"]
        try:
            self.canary_task_menu.configure(
                values=values,
                state="normal" if len(values) > 1 else "disabled")
        except Exception:
            pass
        try:
            current = str(self.canary_task_var.get() or "").strip().lower()
        except Exception:
            current = ""
        if current not in values:
            pick = values[0]
            try:
                self.canary_task_var.set(pick)
            except Exception:
                pass
            if self._canary_task_callback:
                try:
                    self._canary_task_callback(pick)
                except Exception:
                    pass
        return values

    def set_task_options(self, values, current=None):
        """Push a model-specific task list (e.g. turbo: transcribe only)."""
        vals = [str(v).strip().lower() for v in (values or [])
                if str(v).strip()]
        if not vals:
            vals = ["transcribe"]
        try:
            self.canary_task_menu.configure(
                values=vals, state="normal" if len(vals) > 1 else "disabled")
        except Exception:
            pass
        want = str(current or "").strip().lower()
        if want not in vals:
            want = vals[0]
        try:
            self.canary_task_var.set(want)
        except Exception:
            pass
        return vals

    def _on_canary_task_changed(self, value):
        self._refresh_srt_engine_label()
        if self._canary_task_callback:
            self._canary_task_callback(value)
        self._update_live_summary()

    def _on_canary_lang_changed(self, value):
        self._refresh_srt_engine_label()
        if self._canary_lang_callback:
            self._canary_lang_callback(value)
        self._update_live_summary()

    def set_settings(self, method: str, suffix: str, method_cb: Callable, suffix_cb: Callable,
                     itn: bool = True, hint: bool = True,
                     itn_cb: Optional[Callable] = None,
                     hint_cb: Optional[Callable] = None):
        self._method_callback = method_cb if callable(method_cb) else None
        self._suffix_callback = suffix_cb if callable(suffix_cb) else None
        self._itn_callback = itn_cb if callable(itn_cb) else None
        self._itn_hint_callback = hint_cb if callable(hint_cb) else None
        self.method_var.set(method if method in ["clipboard", "unicode"] else "clipboard")
        self.suffix_var.set(suffix if suffix in ["none", "space", "newline", "period_space"] else "none")
        try:
            self.itn_var.set(bool(itn))
            self.itn_hint_var.set(bool(hint))
        except Exception:
            pass

    def set_model(self, arch: int, callback: Callable):
        self.set_model_options(
            list(MODEL_CHOICES.keys()),
            MODEL_CHOICES_REV.get(arch, "Medium Streaming (110MB, best)"),
            callback)

    @staticmethod
    def _model_menu_state(engine_kind: str) -> str:
        return ("normal" if engine_kind in ("Moonshine v2", "Whisper",
                                           "Qwen3-ASR", "Granite",
                                           "Granite 2B")
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
            # The engine name is already visible in the engine menu; the
            # caption stays short so it cannot widen the config card.
            self.model_title_label.configure(text="Model")
        except Exception:
            pass
        self._update_live_summary()

    def set_model_manage_callback(self, cb: Callable):
        self._model_manage_cb = cb if callable(cb) else None

    def _on_model_manage(self):
        if self._model_manage_cb:
            try:
                self._model_manage_cb()
            except Exception:
                pass

    def set_dictionary_callback(self, cb: Callable):
        self._dictionary_cb = cb if callable(cb) else None

    def _on_dictionary(self):
        if self._dictionary_cb:
            try:
                self._dictionary_cb()
            except Exception:
                pass

    def set_dictionary_status(self, text: str):
        """Short summary on the Live card ("Dictionary off", "2 engines ...")."""
        try:
            self.dict_status_label.configure(text=str(text or "Dictionary off"))
        except Exception:
            pass
        self._update_live_summary()

    def set_unload_callback(self, cb: Callable):
        self._unload_cb = cb if callable(cb) else None

    def _on_live_unload(self):
        try:
            cb = getattr(self, "_unload_cb", None)
            msg = cb() if callable(cb) else "Unavailable"
        except Exception as e:
            msg = f"Error: {e}"
        try:
            self.set_model_status(str(msg))
        except Exception:
            pass

    def set_engine(self, engine: str, task: str, src_lang: str, engine_cb: Callable, task_cb: Callable, lang_cb: Callable):
        self._engine_callback = engine_cb if callable(engine_cb) else None
        self._canary_task_callback = task_cb if callable(task_cb) else None
        self._canary_lang_callback = lang_cb if callable(lang_cb) else None
        self.engine_var.set(engine if engine in ENGINE_CHOICES else "Moonshine v2")
        self.canary_task_var.set(task if task in CANARY_TASKS else "transcribe")
        self.canary_lang_var.set(SRT_LANG_NAMES.get(src_lang, "Auto-detect"))
        is_heavy = (engine in ("Canary-1B", "Whisper", "Qwen3-ASR"))
        try:
            self.model_menu.configure(state=self._model_menu_state(engine))



            self._sync_task_menu(engine)
            self.canary_lang_menu.configure(state="normal" if is_heavy else "disabled")
        except Exception:
            pass
        self._refresh_srt_engine_label()
        try:
            self.refresh_lang_options(engine)
        except Exception:
            pass
        self._update_live_summary()

    def set_model_status(self, text: str, color: str = WARNING):
        self.set_status(text, color)

    def set_record_callback(self, on_start: Callable, on_stop: Callable):
        self._on_record_start = on_start if callable(on_start) else None
        self._on_record_stop = on_stop if callable(on_stop) else None

    def set_input_device_callback(self, cb: Callable):
        """App callback: device index (-1 = system default) on mic change."""
        self._mic_callback = cb if callable(cb) else None

    def _on_mic_changed(self, value=None):
        try:
            idx = getattr(self, "_mic_value_map", {}).get(str(value or ""), -1)
        except Exception:
            idx = -1
        cb = getattr(self, "_mic_callback", None)
        if callable(cb):
            try:
                cb(idx)
            except Exception:
                pass

    def set_input_devices(self, devices, current=-1):
        """Fill the mic picker. The row only appears when the machine
        actually offers a choice, so single-mic setups look unchanged."""
        vals = ["System default"]
        mapping = {"System default": -1}
        seen = {"system default"}
        for d in (devices or []):
            try:
                name = str(d.get("name") or "Microphone").strip() or "Microphone"
                idx = int(d.get("index", -1))
            except Exception:
                continue
            base, n = name, 2
            while name.lower() in seen:
                name = f"{base} ({n})"
                n += 1
            seen.add(name.lower())
            vals.append(name)
            mapping[name] = idx
        self._mic_value_map = mapping
        try:
            self.mic_menu.configure(values=vals)
        except Exception:
            pass
        try:
            want = int(current)
        except Exception:
            want = -1
        label = "System default"
        if want != -1:
            for text, idx in mapping.items():
                if idx == want:
                    label = text
                    break
        try:
            self.mic_menu.set(label)
        except Exception:
            pass
        try:
            if len(vals) > 2:
                self.mic_frame.grid()
            else:
                self.mic_frame.grid_remove()
        except Exception:
            pass

    # ------------------------------------------------------------------
    # Live layout: collapsible setup + responsive card arrangement
    # ------------------------------------------------------------------
    def _toggle_setup(self):
        self._set_setup(not bool(getattr(self, "_setup_expanded", False)))

    def _set_setup(self, expanded: bool, relayout: bool = True):
        prev = bool(getattr(self, "_setup_expanded", False))
        self._setup_expanded = bool(expanded)
        frame = getattr(self, "live_setup_frame", None)
        try:
            if frame is not None:
                if self._setup_expanded:
                    frame.grid(row=5, column=0, columnspan=2, sticky="ew",
                               padx=0, pady=(0, 2))
                else:
                    frame.grid_remove()
        except Exception:
            pass
        # The waveform is decoration; hiding it while Setup is open keeps
        # the history readable on a small window.
        try:
            if self._setup_expanded:
                self.waveform.grid_remove()
            else:
                self.waveform.grid()
        except Exception:
            pass
        try:
            self.setup_btn.configure(
                text="\u2699 Setup \u25B4" if self._setup_expanded
                else "\u2699 Setup \u25BE")
        except Exception:
            pass
        if relayout and self._setup_expanded != prev:
            self._apply_live_layout(force=True)

    def _on_live_configure(self, event=None):
        live = getattr(self, "_live_frame", None)
        if live is None:
            return
        if event is not None and getattr(event, "widget", None) is not live:
            return
        try:
            if getattr(self, "_live_fit_after", None) is not None:
                self.after_cancel(self._live_fit_after)
        except Exception:
            pass
        try:
            self._live_fit_after = self.after(140, self._apply_live_layout)
        except Exception:
            self._live_fit_after = None

    def _apply_live_layout(self, mode=None, force=False):
        """Stack the cards on a narrow window, split them in two columns
        when there is room. History always keeps the slack height."""
        live = getattr(self, "_live_frame", None)
        if live is None:
            return
        self._live_fit_after = None
        if mode is None:
            try:
                width = int(live.winfo_width())
            except Exception:
                width = 0
            mode = "wide" if width >= 720 else "narrow"
        if not force and mode == getattr(self, "_live_layout_mode", None):
            return
        self._live_layout_mode = mode
        cards = []
        for name in ("live_config_card", "live_record_card",
                     "live_output_card", "live_history_card"):
            w = getattr(self, name, None)
            if w is None:
                return
            cards.append(w)
        config, record, output, history = cards
        try:
            for w in cards:
                w.grid_remove()
            for r in range(6):
                live.grid_rowconfigure(r, weight=0, minsize=0)
            for c in range(3):
                live.grid_columnconfigure(c, weight=0, minsize=0)
            # While Setup is open the last-transcription card is hidden:
            # the history is the part that must stay readable.
            setup_open = bool(getattr(self, "_setup_expanded", False))
            if mode == "wide":
                live.grid_columnconfigure(0, weight=3, minsize=380)
                live.grid_columnconfigure(1, weight=2, minsize=300)
                live.grid_rowconfigure(2, weight=1, minsize=140)
                config.grid(row=0, column=0, sticky="ew",
                            padx=(8, 4), pady=(0, 6))
                record.grid(row=1, column=0, sticky="new",
                            padx=(8, 4), pady=(0, 6))
                if not setup_open:
                    output.grid(row=2, column=0, sticky="nsew",
                                padx=(8, 4), pady=(0, 8))
                history.grid(row=0, column=1, rowspan=3, sticky="nsew",
                             padx=(4, 8), pady=(0, 8))
            else:
                live.grid_columnconfigure(0, weight=1)
                config.grid(row=0, column=0, sticky="ew",
                            padx=8, pady=(0, 6))
                record.grid(row=1, column=0, sticky="ew",
                            padx=8, pady=(0, 6))
                if setup_open:
                    live.grid_rowconfigure(2, weight=1, minsize=160)
                    history.grid(row=2, column=0, sticky="nsew",
                                 padx=8, pady=(0, 8))
                else:
                    live.grid_rowconfigure(2, weight=1, minsize=110)
                    live.grid_rowconfigure(3, weight=3, minsize=150)
                    output.grid(row=2, column=0, sticky="nsew",
                                padx=8, pady=(0, 6))
                    history.grid(row=3, column=0, sticky="nsew",
                                 padx=8, pady=(0, 8))
        except Exception:
            pass
        # grid_remove on the parent card does not un-hide the setup frame,
        # but its state must be re-applied after the card is re-gridded.
        self._set_setup(bool(getattr(self, "_setup_expanded", False)),
                        relayout=False)

    def _update_live_summary(self):
        parts = []
        try:
            parts.append(str(self.canary_task_var.get() or "transcribe"))
        except Exception:
            pass
        try:
            parts.append(str(self.canary_lang_var.get() or "Auto-detect"))
        except Exception:
            pass
        try:
            parts.append(str(self.dict_status_label.cget("text")
                             or "Dictionary off"))
        except Exception:
            pass
        try:
            self.live_summary_label.configure(
                text="  \u00b7  ".join(p for p in parts if p))
        except Exception:
            pass

    def _refresh_history_count(self):
        try:
            n = self.history.count()
        except Exception:
            n = 0
        try:
            self.history_count_label.configure(
                text=f"{n} entr{'y' if n == 1 else 'ies'}")
        except Exception:
            pass
