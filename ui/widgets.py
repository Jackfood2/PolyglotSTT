# ui/widgets.py
# Shared GUI widgets (moved verbatim from gui.py in v1.4.0, with one
# structural change: colors are read via the ui.theme module (dynamic
# attribute access) instead of module globals, so the dark/light toggle
# in gui.MoonshineGUI.set_theme - which syncs ui.theme - repaints these
# widgets too. Behavior is otherwise identical.
import math
import time

import tkinter as tk

import customtkinter as ctk

from . import theme as _t


class LevelMeter(ctk.CTkCanvas):
    def __init__(self, master, width=300, height=8, **kwargs):
        super().__init__(master, width=width, height=height,
                         highlightthickness=0, bg=_t.BG_CARD, **kwargs)
        self.meter_width = width
        self.meter_height = height
        self.level = 0.0
        # The meter is packed with sticky="ew"/fill="x" in both tabs, so its
        # real width only exists after layout. Without this the bar (and its
        # fill) stops at the construction width and looks cut off.
        try:
            self.bind("<Configure>", self._on_resize)
        except Exception:
            pass
        self._draw()

    def _on_resize(self, event=None):
        try:
            w = int(getattr(event, "width", 0) or 0)
        except Exception:
            w = 0
        if w > 1 and w != self.meter_width:
            self.meter_width = w
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
        self._idle_w = max(1, width - 4)
        self._idle_h = max(1, height - 4)
        # The frame is packed with sticky="ew", so the real canvas width is
        # only known after layout - and it changes when the window resizes.
        # Redraw the idle line on every geometry change or it stays stuck at
        # the construction width and stops short of the box edge.
        try:
            self.canvas.bind("<Configure>", self._on_canvas_configure)
        except Exception:
            pass
        self._draw_idle()

    def _on_canvas_configure(self, event=None):
        if self._animating:
            return
        try:
            w = int(getattr(event, "width", 0) or 0)
            h = int(getattr(event, "height", 0) or 0)
        except Exception:
            w, h = 0, 0
        self._draw_idle(w or None, h or None)

    def _draw_idle(self, w=None, h=None):
        try:
            self.canvas.delete("all")
            try:
                w = int(w or self.canvas.winfo_width() or self._idle_w)
                h = int(h or self.canvas.winfo_height() or self._idle_h)
            except Exception:
                w, h = self._idle_w, self._idle_h
            self.canvas.create_line(0, h // 2, w, h // 2,
                                    fill=_t.FG_DIM, width=2, dash=(4, 4))
        except Exception:
            pass

    def set_amplitude(self, amp: float):
        self.amplitude = max(0.0, min(1.0, amp))

    def start_animation(self):
        if self._animating:
            return
        self._animating = True
        self._animate()

    def stop_animation(self):
        self._animating = False
        self._draw_idle()

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
    """Selectable transcription history.

    Each entry is a read-only tk.Text (selectable, Ctrl+C works even when
    disabled) instead of a Label, so long transcripts can be read and
    copied directly. Long entries are capped at MAX_LINES and can be opened
    in a resizable window from the Expand button.
    """

    MAX_ENTRIES = 100
    MAX_LINES = 6
    EMPTY_TEXT = ("No transcriptions yet.\n"
                  "Hold F2, speak, release \u2014 each transcript appears "
                  "here.\nClick the text to select it, or use Copy / Expand.")

    def __init__(self, master, **kwargs):
        super().__init__(master, fg_color=_t.BG_INPUT, corner_radius=12,
                         scrollbar_button_color=_t.ACCENT, **kwargs)
        self._entries = []
        self._texts: list[str] = []
        self._boxes = []
        self._more_btns = []
        self._refit_after = None
        self._empty = None
        self._show_empty()
        try:
            self.bind("<Configure>", self._on_resize, add="+")
        except Exception:
            pass

    # -- empty state ---------------------------------------------------
    def _show_empty(self):
        try:
            if self._empty is None:
                self._empty = ctk.CTkLabel(
                    self, text=self.EMPTY_TEXT, font=("Segoe UI", 10),
                    text_color=_t.FG_DIM, justify="left")
            self._empty.pack(padx=12, pady=18)
        except Exception:
            pass

    # -- clipboard -----------------------------------------------------
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
                        text=f"{timestamp}   \u2022   Copied \u2713",
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

    def count(self) -> int:
        try:
            return len(self._texts)
        except Exception:
            return 0

    # -- entry sizing --------------------------------------------------
    def _on_resize(self, event=None):
        if event is not None and getattr(event, "widget", None) is not self:
            return
        try:
            if self._refit_after is not None:
                self.after_cancel(self._refit_after)
        except Exception:
            pass
        try:
            self._refit_after = self.after(120, self._refit_all)
        except Exception:
            self._refit_after = None

    def _refit_all(self):
        self._refit_after = None
        for box, text, more in zip(list(self._boxes), list(self._texts),
                                   list(self._more_btns)):
            try:
                self._size_box(box, text, more)
            except Exception:
                pass

    def _size_box(self, box, text, more_btn=None):
        try:
            import tkinter.font as _font
            f = _font.Font(font=box.cget("font"))
            cw = max(6, int(f.measure("0") or 7))
        except Exception:
            cw = 7
        try:
            w = int(box.winfo_width())
            if w <= 2:
                w = max(120, int(self.winfo_width()) - 40)
        except Exception:
            w = 240
        cpl = max(12, int(w / cw))
        total = 0
        for para in (str(text).splitlines() or [""]):
            total += max(1, int(math.ceil(len(para) / float(cpl))))
        lines = max(1, min(self.MAX_LINES, total))
        try:
            box.configure(height=lines)
        except Exception:
            pass
        try:
            if more_btn is not None:
                more_btn.configure(text="Show all" if total > lines
                                   else "Expand")
        except Exception:
            pass

    # -- interaction ---------------------------------------------------
    @staticmethod
    def _select_all(event=None):
        try:
            w = event.widget
            w.tag_add("sel", "1.0", "end-1c")
            w.mark_set("insert", "1.0")
        except Exception:
            pass
        return "break"

    def _wheel_outer(self, event=None):
        """The read-only text boxes never scroll themselves; send the
        wheel to the history list instead of swallowing it."""
        try:
            delta = int(getattr(event, "delta", 0) or 0)
        except Exception:
            delta = 0
        step = -3 if delta > 0 else 3
        try:
            self._parent_canvas.yview_scroll(step, "units")
        except Exception:
            try:
                self._canvas.yview_scroll(step, "units")
            except Exception:
                pass
        return "break"

    # -- entries -------------------------------------------------------
    def add_entry(self, text: str, timestamp: str):
        try:
            if self._empty is not None:
                self._empty.pack_forget()
                self._empty.destroy()
        except Exception:
            pass
        self._empty = None
        entry_frame = ctk.CTkFrame(self, fg_color=_t.BG_CARD, corner_radius=8)
        entry_frame.pack(fill="x", padx=4, pady=3)
        head = ctk.CTkFrame(entry_frame, fg_color="transparent")
        head.pack(fill="x", padx=8, pady=(6, 0))
        ts_label = ctk.CTkLabel(head, text=timestamp,
                                font=("Segoe UI", 10), text_color=_t.FG_DIM)
        ts_label.pack(side="left")
        expand_btn = ctk.CTkButton(
            head, text="Expand", width=64, height=20, font=("Segoe UI", 9),
            fg_color="transparent", hover_color=_t.BTN_DIM, corner_radius=6,
            text_color=_t.FG_DIM,
            command=lambda _t=text, _ts=timestamp: self._show_full(_t, _ts))
        expand_btn.pack(side="right")
        copy_btn = ctk.CTkButton(
            head, text="Copy", width=46, height=20, font=("Segoe UI", 9),
            fg_color="transparent", hover_color=_t.BTN_DIM, corner_radius=6,
            text_color=_t.FG_DIM,
            command=lambda _t=text, _l=ts_label, _ts=timestamp:
                self._copy_one(_t, _l, _ts))
        copy_btn.pack(side="right", padx=(0, 2))
        box = tk.Text(entry_frame, wrap="word", height=2, width=1,
                      font=("Consolas", 11), bg=_t.BG_CARD,
                      fg=_t.FG_PRIMARY, insertbackground=_t.FG_PRIMARY,
                      selectbackground=_t.ACCENT,
                      selectforeground=_t.BTN_TEXT,
                      bd=0, highlightthickness=0, padx=2, pady=4,
                      cursor="xterm", spacing1=0, spacing3=2)
        box.insert("1.0", text)
        box.configure(state="disabled")
        box.pack(fill="x", padx=8, pady=(2, 6))
        try:
            box.bind("<Control-a>", self._select_all)
            box.bind("<Control-A>", self._select_all)
            box.bind("<MouseWheel>", self._wheel_outer)
        except Exception:
            pass
        self._entries.append(entry_frame)
        self._texts.append(text)
        self._boxes.append(box)
        self._more_btns.append(expand_btn)
        self._size_box(box, text, expand_btn)
        while len(self._entries) > self.MAX_ENTRIES:
            try:
                self._entries.pop(0).destroy()
                self._texts.pop(0)
                self._boxes.pop(0)
                self._more_btns.pop(0)
            except Exception:
                break
        try:
            self._parent_canvas.yview_moveto(1.0)
        except Exception:
            try:
                self._canvas.yview_moveto(1.0)
            except Exception:
                pass

    def _show_full(self, text: str, timestamp: str):
        """Open one entry in a resizable window with a selectable view."""
        try:
            top = ctk.CTkToplevel(self.winfo_toplevel())
            top.title(f"Transcription {timestamp}")
            top.geometry("620x420")
            top.configure(fg_color=_t.BG_DARK)
            top.transient(self.winfo_toplevel())
            ctk.CTkLabel(top,
                         text=f"{timestamp}  \u00b7  select the text "
                              f"or press Ctrl+A / Ctrl+C",
                         font=("Segoe UI", 10), text_color=_t.FG_DIM
                         ).pack(anchor="w", padx=14, pady=(12, 4))
            tb = ctk.CTkTextbox(top, fg_color=_t.BG_INPUT,
                                text_color=_t.FG_PRIMARY,
                                font=("Consolas", 12), wrap="word",
                                corner_radius=8)
            tb.pack(fill="both", expand=True, padx=12, pady=(0, 8))
            tb.insert("1.0", text)
            tb.configure(state="disabled")
            btns = ctk.CTkFrame(top, fg_color="transparent")
            btns.pack(fill="x", padx=12, pady=(0, 12))
            cbtn = ctk.CTkButton(btns, text="Copy", width=90, height=30,
                                 font=("Segoe UI", 11), fg_color=_t.ACCENT,
                                 hover_color=_t.ACCENT_DARK,
                                 text_color=_t.BTN_TEXT, corner_radius=8)
            cbtn.configure(command=lambda: self._dialog_copy(cbtn, text))
            cbtn.pack(side="right")
            ctk.CTkButton(btns, text="Close", width=90, height=30,
                          font=("Segoe UI", 11), fg_color=_t.BTN_DIM,
                          hover_color=_t.BTN_DIM_HOVER,
                          text_color=_t.FG_SECONDARY, corner_radius=8,
                          command=top.destroy).pack(side="right", padx=(0, 8))
            try:
                top.bind("<Escape>", lambda _e=None: top.destroy())
                top.after(60, top.focus_force)
            except Exception:
                pass
        except Exception:
            pass

    def _dialog_copy(self, btn, text):
        try:
            if self._copy_text(text):
                btn.configure(text="Copied \u2713")
                try:
                    btn.after(1200, lambda: btn.configure(text="Copy"))
                except Exception:
                    pass
        except Exception:
            pass

    def clear(self):
        for entry in self._entries:
            try:
                entry.destroy()
            except Exception:
                pass
        self._entries.clear()
        try:
            self._texts.clear()
            self._boxes.clear()
            self._more_btns.clear()
        except Exception:
            pass
        self._empty = None
        self._show_empty()
