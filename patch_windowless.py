"""Windowless-next-launch switch: GUI checkbox + run.bat branch + log guard.

Covers every file the feature touches. Idempotent: re-running only
reports "already applied". Unlike blind text replacement, any anchor
that does not match EXACTLY is reported as MISS and fails loudly -
no silent partial application.

Usage:  python patch_windowless.py   (stdlib only; run from repo root)
"""
from __future__ import annotations

import ast
import datetime
import pathlib
import subprocess
import sys

ROOT = pathlib.Path(__file__).resolve().parent
BACKUP_DIR = ROOT / "patch_backups" / datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

applied: list[str] = []
skipped: list[str] = []
missing: list[str] = []
_backed_up: set[str] = set()


def log(message: str) -> None:
    print(f"[patch] {message}")


def read_text(path: pathlib.Path) -> tuple[str, bool]:
    raw = path.read_bytes().decode("utf-8")
    crlf = "\r\n" in raw
    return raw.replace("\r\n", "\n"), crlf


def write_text(path: pathlib.Path, text: str, crlf: bool) -> None:
    if crlf:
        text = text.replace("\n", "\r\n")
    path.write_bytes(text.encode("utf-8"))


def backup(path: pathlib.Path) -> None:
    if path.name in _backed_up or not path.exists():
        return
    dest = BACKUP_DIR / path.name
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_bytes(path.read_bytes())
    _backed_up.add(path.name)


def patch_file(filename: str, old: str, new: str, tag: str) -> bool:
    """Replace old->new once. Skip if new already present; MISS if old absent."""
    path = ROOT / filename
    if not path.exists():
        missing.append(f"{tag} ({filename} not found)")
        return False
    text, crlf = read_text(path)
    if new in text:
        skipped.append(tag)
        log(f"already applied, skip: {tag}")
        return True
    if text.count(old) != 1:
        missing.append(f"{tag} (anchor x{text.count(old)} in {filename})")
        return False
    backup(path)
    write_text(path, text.replace(old, new, 1), crlf)
    applied.append(tag)
    log(f"patched: {tag}")
    return True


def ensure_line(filename: str, line: str, tag: str) -> bool:
    path = ROOT / filename
    if not path.exists():
        missing.append(f"{tag} ({filename} not found)")
        return False
    text, crlf = read_text(path)
    if line in text.splitlines():
        skipped.append(tag)
        log(f"already applied, skip: {tag}")
        return True
    backup(path)
    if not text.endswith("\n"):
        text += "\n"
    write_text(path, text + line + "\n", crlf)
    applied.append(tag)
    log(f"patched: {tag}")
    return True


def ensure_file(filename: str, content: str, tag: str) -> bool:
    path = ROOT / filename
    if not path.exists():
        path.write_bytes(content.replace("\n", "\r\n").encode("utf-8"))
        applied.append(tag + " (created)")
        log(f"created: {filename}")
        return True
    current = path.read_bytes().decode("utf-8").replace("\r\n", "\n")
    if current == content:
        skipped.append(tag)
        log(f"already applied, skip: {tag}")
        return True
    missing.append(f"{tag} ({filename} exists with different content - not overwritten)")
    return False


# ---------------------------------------------------------------- fragments
MS_GUARD_OLD = "sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))"
MS_GUARD_NEW = """sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
if sys.stdout is None or sys.stderr is None:
    # Windowless launch (pythonw.exe): there is no console, and any print
    # would raise. Keep a rolling log next to the script instead.
    try:
        _logdir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "logs")
        os.makedirs(_logdir, exist_ok=True)
        _winlog = open(os.path.join(_logdir, "app.log"), "a",
                       encoding="utf-8", errors="replace", buffering=1)
        if sys.stdout is None:
            sys.stdout = _winlog
        if sys.stderr is None:
            sys.stderr = _winlog
    except Exception:
        pass"""

MS_DEFAULT_OLD = '    "theme": "dark",\n}'
MS_DEFAULT_NEW = '    "theme": "dark",\n    "windowless": False,\n}'

MS_VALIDATE_OLD = """        if self.config.get("compute") not in ("auto", "cpu", "gpu"):
            self.config["compute"] = "auto"
            needs_save = True"""
MS_VALIDATE_NEW = """        if self.config.get("compute") not in ("auto", "cpu", "gpu"):
            self.config["compute"] = "auto"
            needs_save = True
        if not isinstance(self.config.get("windowless"), bool):
            self.config["windowless"] = False
            needs_save = True"""

MS_HANDLER_OLD = "    def _log(self, msg, color=None):"
MS_HANDLER_NEW = '''    def _on_windowless_changed(self, value: bool):
        with _CONFIG_LOCK:
            self.config["windowless"] = bool(value)
            save_local_config(self.config)
        mode = "windowless (no console)" if value else "console window"
        self._log(f"Next launch via run.bat uses {mode} (takes effect on restart)")

    def _log(self, msg, color=None):'''

MS_RESTORE_OLD = """                try:
                    self.gui.set_footer_version(APP_VERSION)
                    self.gui.set_title_version(APP_VERSION)
                except Exception:
                    pass"""
MS_RESTORE_NEW = """                try:
                    self.gui.set_footer_version(APP_VERSION)
                    self.gui.set_title_version(APP_VERSION)
                except Exception:
                    pass
                try:
                    self.gui.set_windowless(self.config.get("windowless", False),
                                            self._on_windowless_changed)
                except Exception:
                    pass"""

GUI_CHECK_OLD = "        self._footer_label.pack()"
GUI_CHECK_NEW = """        self._footer_label.pack(side="left")
        self.windowless_var = ctk.BooleanVar(value=False)
        self.windowless_check = ctk.CTkCheckBox(
            footer, text="Windowless next launch",
            variable=self.windowless_var, font=("Segoe UI", 10),
            text_color=FG_DIM, fg_color=ACCENT,
            command=self._on_windowless_toggled)
        self.windowless_check.pack(side="right")
        self._windowless_callback = None"""

GUI_METHODS_OLD = '''    def set_title_version(self, version: str):
        try:
            v = str(version or "").strip()
            self.title(f"MoonshineSTT v{v}" if v else "MoonshineSTT")
        except Exception:
            pass'''
GUI_METHODS_NEW = GUI_METHODS_OLD + '''
    def _on_windowless_toggled(self):
        try:
            value = bool(self.windowless_var.get())
        except Exception:
            value = False
        if self._windowless_callback:
            try:
                self._windowless_callback(value)
            except Exception:
                pass
    def set_windowless(self, value: bool, callback):
        self._windowless_callback = callback if callable(callback) else None
        try:
            self.windowless_var.set(bool(value))
        except Exception:
            pass'''

BAT_BRANCH_OLD = ''':RUN_APP
if not exist "models_cache\\download.moonshine.ai" echo Warning: models_cache not found. Will try APPDATA cache.
"%~dp0venv\\Scripts\\python.exe" moonshine_stt.py %*'''
BAT_BRANCH_NEW = ''':RUN_APP
if not exist "models_cache\\download.moonshine.ai" echo Warning: models_cache not found. Will try APPDATA cache.
rem GUI switch "Windowless next launch" (moonshine_config.json holds "windowless": true).
findstr /r /c:"windowless.: *true" "moonshine_config.json" >nul 2>&1
if not errorlevel 1 (
    start "PolyglotSTT" "%~dp0venv\\Scripts\\pythonw.exe" moonshine_stt.py %*
    goto END_APP
)
"%~dp0venv\\Scripts\\python.exe" moonshine_stt.py %*'''

BAT_END_OLD = """if errorlevel 1 (
    echo.
    echo MoonshineSTT exited with an error.
    pause
)
endlocal"""
BAT_END_NEW = """if errorlevel 1 (
    echo.
    echo MoonshineSTT exited with an error.
    pause
)
:END_APP
endlocal"""

BAT_WL = """@echo off
rem Windowless launcher: no console window stays open (a brief one may
rem flash on double-click - use run_windowless.vbs for zero flash).
cd /d "%~dp0"
if not exist "venv\\Scripts\\pythonw.exe" (
    echo No venv found - run run.bat once first (it installs everything with a window).
    pause
    exit /b 1
)
start "PolyglotSTT" "%~dp0venv\\Scripts\\pythonw.exe" moonshine_stt.py
"""

VBS_WL = """' Double-click: starts PolyglotSTT with no window at all (not even a flash).
' Needs venv already set up (run run.bat once first). Output goes to logs\\app.log.
CreateObject("Wscript.Shell").Run "venv\\Scripts\\pythonw.exe moonshine_stt.py", 0, False
"""


def main() -> int:
    log(f"workspace: {ROOT}")
    ok = True
    ok &= patch_file("moonshine_stt.py", MS_GUARD_OLD, MS_GUARD_NEW, "stdout-guard")
    ok &= patch_file("moonshine_stt.py", MS_DEFAULT_OLD, MS_DEFAULT_NEW, "config-default")
    ok &= patch_file("moonshine_stt.py", MS_VALIDATE_OLD, MS_VALIDATE_NEW, "config-validate")
    ok &= patch_file("moonshine_stt.py", MS_HANDLER_OLD, MS_HANDLER_NEW, "windowless-handler")
    ok &= patch_file("moonshine_stt.py", MS_RESTORE_OLD, MS_RESTORE_NEW, "launch-restore")
    ok &= patch_file("gui.py", GUI_CHECK_OLD, GUI_CHECK_NEW, "footer-checkbox")
    ok &= patch_file("gui.py", GUI_METHODS_OLD, GUI_METHODS_NEW, "windowless-methods")
    ok &= patch_file("run.bat", BAT_BRANCH_OLD, BAT_BRANCH_NEW, "runbat-branch")
    ok &= patch_file("run.bat", BAT_END_OLD, BAT_END_NEW, "runbat-endlabel")
    ok &= ensure_line(".gitignore", "logs/", "gitignore-logs")
    ok &= ensure_file("run_windowless.bat", BAT_WL, "launcher-bat")
    ok &= ensure_file("run_windowless.vbs", VBS_WL, "launcher-vbs")

    for fn in ("moonshine_stt.py", "gui.py"):
        try:
            ast.parse(read_text(ROOT / fn)[0], filename=fn)
        except SyntaxError as exc:
            print(f"[error] syntax error in {fn}: {exc}")
            return 1

    targets = [str(ROOT / f) for f in ("moonshine_stt.py", "gui.py")]
    proc = subprocess.run([sys.executable, "-m", "py_compile", *targets], cwd=ROOT)
    if proc.returncode != 0:
        print("[error] Python compilation failed")
        return 1
    log("Python compilation passed")

    print()
    print(f"Applied: {len(applied)}, already-applied: {len(skipped)}, MISS: {len(missing)}")
    for tag in missing:
        print(f"  MISS: {tag}")
    if missing:
        print(f"[error] restore backups from: {BACKUP_DIR}")
        return 1
    print(f"Backups: {BACKUP_DIR}")
    print()
    print("Next: launch the app, flip the footer switch, restart via run.bat,")
    print("confirm the console is gone and logs\\app.log keeps growing.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
