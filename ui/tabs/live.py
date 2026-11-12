




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

    def _on_canary_lang_changed(self, value):
        self._refresh_srt_engine_label()
        if self._canary_lang_callback:
            self._canary_lang_callback(value)

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
