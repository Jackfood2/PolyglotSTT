# ui/widgets.py
# Shared GUI widgets (moved verbatim from gui.py in v1.4.0, with one
# structural change: colors are read via the ui.theme module (dynamic
# attribute access) instead of module globals, so the dark/light toggle
# in gui.MoonshineGUI.set_theme - which syncs ui.theme - repaints these
# widgets too. Behavior is otherwise identical.
import math
import time

import customtkinter as ctk

from . import theme as _t


class LevelMeter(ctk.CTkCanvas):
    def __init__(self, master, width=300, height=8, **kwargs):
        super().__init__(master, width=width, height=height,
                         highlightthickness=0, bg=_t.BG_CARD, **kwargs)
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
                              fill=_t.BTN_DIM, outline="")
        if self.level > 0.01:
            w = self.meter_width * self.level
            if self.level < 0.6:
                color = _t.SUCCESS
            elif self.level < 0.85:
                color = _t.WARNING
            else:
                color = _t.DANGER
            self.create_rectangle(0, 0, w, self.meter_height,
                                  fill=color, outline="", width=0)


class WaveformDisplay(ctk.CTkFrame):
    def __init__(self, master, width=400, height=80, **kwargs):
        super().__init__(master, width=width, height=height,
                         fg_color=_t.BG_INPUT, corner_radius=12, **kwargs)
        self.canvas = ctk.CTkCanvas(self, width=width - 4, height=height - 4,
                                    highlightthickness=0, bg=_t.BG_INPUT)
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
                                fill=_t.FG_DIM, width=2, dash=(4, 4))

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
            self.canvas.create_line(*flat, fill=_t.ACCENT_GLOW, width=2,
                                    smooth=True)
            glow_points = []
            for p in points:
                glow_points.extend([p[0], p[1] + 2])
            self.canvas.create_line(*glow_points, fill=_t.ACCENT, width=1,
                                    smooth=True, dash=(2, 4))
        self.after(33, self._animate)


class StatusBadge(ctk.CTkFrame):
    def __init__(self, master, **kwargs):
        super().__init__(master, fg_color="transparent", **kwargs)
        self.indicator = ctk.CTkLabel(self, text="●", font=("Segoe UI", 12),
                                      text_color=_t.FG_DIM)
        self.indicator.pack(side="left", padx=(0, 6))
        self.label = ctk.CTkLabel(self, text="Ready",
                                  font=("Segoe UI", 13, "bold"),
                                  text_color=_t.FG_SECONDARY)
        self.label.pack(side="left")

    def set_status(self, text: str, color: str):
        color = _t.theme_color(color)
        self.label.configure(text=text, text_color=color)
        self.indicator.configure(text_color=color)


class HistoryPanel(ctk.CTkScrollableFrame):
    MAX_ENTRIES = 100

    def __init__(self, master, **kwargs):
        super().__init__(master, fg_color=_t.BG_INPUT, corner_radius=12,
                         scrollbar_button_color=_t.ACCENT, **kwargs)
        self._entries = []
        self._texts: list[str] = []

    def _copy_text(self, text: str) -> bool:
        """Copy to OS clipboard. Never raises."""
        try:
            if not text:
                return False
            try:
                from input_sim import copy_to_clipboard
                if copy_to_clipboard(text):
                    return True
            except Exception:
                pass
            # Fallback: tkinter clipboard.
            try:
                self.clipboard_clear()
                self.clipboard_append(text)
                return True
            except Exception:
                return False
        except Exception:
            return False

    def _copy_one(self, text: str, ts_label=None, timestamp: str = ""):
        ok = self._copy_text(text)
        # Brief "Copied" feedback on the timestamp label.
        try:
            if ts_label is not None and timestamp:
                if ok:
                    ts_label.configure(
                        text=f"{timestamp}   •   Copied ✓",
                        text_color=_t.theme_color(_t.SUCCESS))
                    try:
                        self.after(1200, lambda: self._restore_ts(
                            ts_label, timestamp))
                    except Exception:
                        pass
        except Exception:
            pass

    def _restore_ts(self, ts_label, timestamp: str):
        try:
            ts_label.configure(text=timestamp,
                               text_color=_t.theme_color(_t.FG_DIM))
        except Exception:
            pass

    def get_all_text(self, sep: str = "\n") -> str:
        """All history entries, oldest first, plain text for clipboard."""
        try:
            return sep.join(t for t in list(self._texts) if t)
        except Exception:
            return ""

    def copy_all(self) -> int:
        """Copy whole history to clipboard. Returns entries copied."""
        try:
            text = self.get_all_text()
            if not text.strip():
                return 0
            if self._copy_text(text):
                return len(self._texts)
            return 0
        except Exception:
            return 0

    def add_entry(self, text: str, timestamp: str):
        entry_frame = ctk.CTkFrame(self, fg_color=_t.BG_CARD, corner_radius=8,
                                   cursor="hand2")
        entry_frame.pack(fill="x", padx=4, pady=3)
        ts_label = ctk.CTkLabel(entry_frame, text=timestamp,
                                font=("Segoe UI", 10), text_color=_t.FG_DIM)
        ts_label.pack(anchor="w", padx=8, pady=(6, 0))
        text_label = ctk.CTkLabel(entry_frame, text=text,
                                  font=("Consolas", 12),
                                  text_color=_t.FG_PRIMARY,
                                  wraplength=360, justify="left")
        text_label.pack(anchor="w", padx=8, pady=(2, 6))
        # Click anywhere on the entry copies that entry's text.
        try:
            _cb = lambda _e=None, _t=text, _l=ts_label, _ts=timestamp: \
                self._copy_one(_t, _l, _ts)
            entry_frame.bind("<Button-1>", _cb)
            ts_label.bind("<Button-1>", _cb)
            text_label.bind("<Button-1>", _cb)
        except Exception:
            pass
        self._entries.append(entry_frame)
        self._texts.append(text)
        while len(self._entries) > self.MAX_ENTRIES:
            try:
                self._entries.pop(0).destroy()
                self._texts.pop(0)
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
        try:
            self._texts.clear()
        except Exception:
            pass
