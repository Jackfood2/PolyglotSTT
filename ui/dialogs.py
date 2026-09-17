# ui/dialogs.py
# Modal dialogs: model manager, burn-speed compare.
# (Split verbatim from gui.py in v1.4.0 - no behavior change.)
# Palette names below are import-time snapshots kept fresh by
# MoonshineGUI.set_theme, which pushes every toggle into this module.
import customtkinter as ctk
from ui.theme import ACCENT_GLOW, DANGER, BG_DARK, BG_CARD, FG_PRIMARY, FG_SECONDARY, FG_DIM, BTN_DIM, BTN_DIM_HOVER, BTN_DANGER_HOVER, BTN_TEXT
from ui.choices import BURN_COMPARE_ROWS, BURN_COMPARE_CODEC_NOTE


class DialogsMixin:
    """Modal dialogs: model manager, burn-speed compare."""

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
