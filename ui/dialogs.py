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

    def show_dictionary_manager(self, get_data, on_save):
        """Add / edit / delete dictionary terms and context text.

        ``get_data()`` returns {"terms", "context", "engines", "available"};
        ``on_save(payload)`` returns (ok, message). Only engines that really
        support a dictionary are offered; over-limit lists ask the user what
        to do instead of silently clipping.
        """
        try:
            from tkinter import messagebox as _mb
        except Exception:
            _mb = None
        try:
            from tkinter import simpledialog as _sd
        except Exception:
            _sd = None
        try:
            import dictionary as _dict
        except Exception:
            _dict = None
        if _dict is None:
            if _mb is not None:
                try:
                    _mb.showerror("Dictionary", "dictionary.py is missing.")
                except Exception:
                    pass
            return
        try:
            _old = getattr(self, "_dictionary_win", None)
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
            data = get_data() or {}
        except Exception:
            data = {}
        state = {
            "terms": _dict.norm_terms(data.get("terms")),
            "context": _dict.norm_context(data.get("context")),
            "engines": _dict.enabled_map(data.get("engines")),
        }
        available = dict(data.get("available") or _dict.available_map())

        win = ctk.CTkToplevel(self)
        self._dictionary_win = win
        try:
            win.title("Dictionary")
            win.geometry("620x680")
            win.minsize(540, 540)
            win.configure(fg_color=BG_DARK)
            try:
                win.transient(self)
            except Exception:
                pass
        except Exception:
            pass

        ctk.CTkLabel(win, text="Dictionary", font=("Segoe UI", 15, "bold"),
                     text_color=FG_PRIMARY).pack(anchor="w", padx=16,
                                                 pady=(14, 0))
        ctk.CTkLabel(
            win, text="Bias transcription toward names, jargon and acronyms. "
            "Moonshine v2 and Canary-1B have no equivalent setting, so only "
            "Whisper and Qwen3-ASR can use this list.",
            font=("Segoe UI", 10), text_color=FG_DIM, wraplength=580,
            justify="left").pack(anchor="w", padx=16, pady=(0, 8))

        eng_frame = ctk.CTkFrame(win, fg_color=BG_CARD, corner_radius=10)
        eng_frame.pack(fill="x", padx=14, pady=(0, 8))
        eng_frame.grid_columnconfigure(0, weight=1)
        eng_vars = {}
        eng_status = {}
        row = 0
        for engine in _dict.DICT_ENGINES:
            ok = bool(available.get(engine, True))
            var = ctk.BooleanVar(
                value=bool(state["engines"].get(engine)) and ok)
            eng_vars[engine] = var
            ctk.CTkCheckBox(
                eng_frame, text=engine, variable=var, font=("Segoe UI", 12),
                text_color=FG_PRIMARY, fg_color=ACCENT_GLOW,
                hover_color=ACCENT_GLOW, border_color=FG_DIM,
                state="normal" if ok else "disabled",
                command=lambda: _refresh_status()
            ).grid(row=row, column=0, sticky="w", padx=12, pady=(8, 0))
            note = (_dict.engine_help(engine) if ok else
                    "Not installed on this machine - skipped.")
            ctk.CTkLabel(eng_frame, text=note, font=("Segoe UI", 9),
                         text_color=FG_DIM, wraplength=460, justify="left"
                         ).grid(row=row + 1, column=0, sticky="w",
                                padx=(34, 8))
            st = ctk.CTkLabel(eng_frame, text="", font=("Segoe UI", 10),
                              text_color=FG_SECONDARY, anchor="e")
            st.grid(row=row + 1, column=1, sticky="e", padx=12)
            eng_status[engine] = st
            row += 2
        ctk.CTkLabel(eng_frame, text="", height=4).grid(row=row, column=0)

        add_frame = ctk.CTkFrame(win, fg_color="transparent")
        add_frame.pack(fill="x", padx=14, pady=(0, 6))
        add_frame.grid_columnconfigure(0, weight=1)
        entry = ctk.CTkEntry(add_frame, placeholder_text="Add a word or phrase…",
                             font=("Segoe UI", 12), fg_color=BG_CARD,
                             border_color=BTN_DIM, height=34)
        entry.grid(row=0, column=0, sticky="ew", padx=(0, 6))

        def _add():
            try:
                raw = entry.get()
            except Exception:
                raw = ""
            if not raw.strip():
                return
            parts = [p for chunk in raw.splitlines() for p in chunk.split(",")]
            state["terms"] = _dict.norm_terms(list(state["terms"]) + parts)
            try:
                entry.delete(0, "end")
            except Exception:
                pass
            _rebuild()
            _refresh_status()

        ctk.CTkButton(add_frame, text="Add", width=70, height=34,
                      font=("Segoe UI", 12), fg_color=BTN_DIM,
                      hover_color=BTN_DIM_HOVER, text_color=FG_SECONDARY,
                      corner_radius=8, command=_add
                      ).grid(row=0, column=1, sticky="e")
        entry.bind("<Return>", lambda _e: _add())

        list_wrap = ctk.CTkFrame(win, fg_color="transparent")
        list_wrap.pack(fill="both", expand=True, padx=14, pady=(0, 6))
        count_lbl = ctk.CTkLabel(list_wrap, text="", font=("Segoe UI", 10),
                                 text_color=FG_DIM, anchor="w")
        count_lbl.pack(fill="x")
        body = ctk.CTkScrollableFrame(list_wrap, fg_color="transparent")
        body.pack(fill="both", expand=True)
        body.grid_columnconfigure(0, weight=1)
        try:
            self._tune_scrollers(win)
        except Exception:
            pass

        def _remove(idx):
            try:
                del state["terms"][idx]
            except Exception:
                return
            _rebuild()
            _refresh_status()

        def _edit(idx):
            try:
                current = state["terms"][idx]
            except Exception:
                return
            new = None
            if _sd is not None:
                try:
                    new = _sd.askstring("Edit term", "Word or phrase:",
                                        initialvalue=current, parent=win)
                except Exception:
                    new = None
            if new is None:
                return
            new = str(new).strip()
            if not new:
                _remove(idx)
                return
            terms = list(state["terms"])
            terms[idx] = new[: _dict.MAX_TERM_CHARS]
            state["terms"] = _dict.norm_terms(terms)
            _rebuild()
            _refresh_status()

        def _rebuild():
            try:
                for child in body.winfo_children():
                    child.destroy()
            except Exception:
                pass
            terms = state["terms"]
            try:
                count_lbl.configure(text=f"{len(terms)} term(s)")
            except Exception:
                pass
            if not terms:
                ctk.CTkLabel(body, text="No terms yet - add names, brands or "
                             "acronyms above.", font=("Segoe UI", 11),
                             text_color=FG_DIM).grid(row=0, column=0,
                                                     sticky="w", padx=8, pady=10)
                return
            for i, term in enumerate(terms):
                card = ctk.CTkFrame(body, fg_color=BG_CARD, corner_radius=8)
                card.grid(row=i, column=0, sticky="ew", padx=2, pady=2)
                card.grid_columnconfigure(0, weight=1)
                ctk.CTkLabel(card, text=term, font=("Segoe UI", 11),
                             text_color=FG_PRIMARY, anchor="w"
                             ).grid(row=0, column=0, sticky="w", padx=10, pady=6)
                ctk.CTkButton(card, text="Edit", width=52, height=26,
                              font=("Segoe UI", 10), fg_color=BTN_DIM,
                              hover_color=BTN_DIM_HOVER, text_color=FG_SECONDARY,
                              corner_radius=6, command=lambda n=i: _edit(n)
                              ).grid(row=0, column=1, padx=(0, 4), pady=4)
                ctk.CTkButton(card, text="Remove", width=68, height=26,
                              font=("Segoe UI", 10), fg_color=DANGER,
                              hover_color=BTN_DANGER_HOVER, text_color=BTN_TEXT,
                              corner_radius=6, command=lambda n=i: _remove(n)
                              ).grid(row=0, column=2, padx=(0, 8), pady=4)

        ctk.CTkLabel(win, text="Context / prompt (optional)", font=("Segoe UI", 10),
                     text_color=FG_DIM, anchor="w"
                     ).pack(fill="x", padx=16)
        ctx_box = ctk.CTkTextbox(win, height=78, font=("Segoe UI", 12),
                                 fg_color=BG_CARD, text_color=FG_PRIMARY,
                                 wrap="word", border_width=1,
                                 border_color=BTN_DIM)
        ctx_box.pack(fill="x", padx=14, pady=(0, 8))
        if state["context"]:
            try:
                ctx_box.insert("1.0", state["context"])
            except Exception:
                pass
        ctx_box.bind("<KeyRelease>", lambda _e: _refresh_status())

        def _current_terms():
            return _dict.norm_terms(state["terms"])

        def _current_ctx():
            try:
                return _dict.norm_context(ctx_box.get("1.0", "end"))
            except Exception:
                return state["context"]

        def _enabled():
            return [e for e, v in eng_vars.items() if bool(v.get())]

        def _refresh_status():
            terms = _current_terms()
            ctx = _current_ctx()
            on = set(_enabled())
            for engine, lbl in eng_status.items():
                try:
                    if not available.get(engine, True):
                        lbl.configure(text="not installed",
                                      text_color=FG_DIM)
                        continue
                    if engine not in on:
                        lbl.configure(text="off", text_color=FG_DIM)
                        continue
                    info = _dict.build(engine, terms, ctx)
                    over = int(info.get("dropped", 0) or 0) > 0
                    lbl.configure(text=_dict.describe(engine, terms, ctx),
                                  text_color=DANGER if over else ACCENT_GLOW)
                except Exception:
                    pass

        foot = ctk.CTkFrame(win, fg_color="transparent")
        foot.pack(fill="x", padx=16, pady=(2, 14))
        foot.grid_columnconfigure(0, weight=1)
        foot.grid_columnconfigure(1, weight=1)

        def _save():
            terms = _current_terms()
            ctx = _current_ctx()
            on = _enabled()
            skipped = [e for e in on if not available.get(e, True)]
            if skipped and _mb is not None:
                try:
                    if not _mb.askyesno(
                            "Engine not available",
                            ", ".join(skipped) + " is not installed on this "
                            "machine.\n\nKeep it enabled anyway?"):
                        for e in skipped:
                            try:
                                eng_vars[e].set(False)
                            except Exception:
                                pass
                        on = _enabled()
                except Exception:
                    pass
            over = []
            for engine in on:
                info = _dict.build(engine, terms, ctx)
                if int(info.get("dropped", 0) or 0) > 0:
                    over.append((engine, info))
            if over and _mb is not None:
                engine, info = over[0]
                keep = len(info.get("kept") or [])
                unit = "tokens" if engine == "Whisper" else "terms"
                try:
                    ans = _mb.askyesnocancel(
                        "Dictionary too long",
                        f"{engine} can only take {info['limit']} {unit}, so "
                        f"{info['dropped']} of your {len(terms)} term(s) "
                        f"would be ignored.\n\n"
                        f"Yes = trim the list to {keep} term(s) now\n"
                        f"No = keep everything (extra terms are ignored)\n"
                        f"Cancel = go back and edit")
                except Exception:
                    ans = None
                if ans is None:
                    return
                if ans:
                    terms = _dict.trim_to_fit(terms, ctx, on)
                    state["terms"] = terms
                    _rebuild()
                    _refresh_status()
            warnings = []
            for engine in on:
                info = _dict.build(engine, terms, ctx)
                if info.get("warning"):
                    warnings.append(f"{engine}: {info['warning']}")
            if warnings and _mb is not None:
                try:
                    if not _mb.askyesno("Dictionary",
                                        "\n\n".join(warnings) +
                                        "\n\nSave anyway?"):
                        return
                except Exception:
                    pass
            payload = {
                "terms": terms,
                "context": ctx,
                "engines": {e: bool(eng_vars[e].get())
                            for e in _dict.DICT_ENGINES},
            }
            try:
                ok, msg = on_save(payload)
            except Exception as ex:
                ok, msg = False, str(ex)
            if not ok:
                if _mb is not None:
                    try:
                        _mb.showerror("Dictionary", msg or "Could not save.")
                    except Exception:
                        pass
                return
            try:
                win.destroy()
            except Exception:
                pass

        ctk.CTkButton(foot, text="Cancel", font=("Segoe UI", 12),
                      fg_color=BTN_DIM, hover_color=BTN_DIM_HOVER, height=38,
                      corner_radius=8, text_color=FG_SECONDARY,
                      command=win.destroy
                      ).grid(row=0, column=0, padx=(0, 4), sticky="ew")
        ctk.CTkButton(foot, text="Save", font=("Segoe UI", 12),
                      fg_color=ACCENT_GLOW, hover_color=ACCENT_GLOW, height=38,
                      corner_radius=8, text_color=BG_DARK, command=_save
                      ).grid(row=0, column=1, padx=(4, 0), sticky="ew")

        _rebuild()
        _refresh_status()
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
