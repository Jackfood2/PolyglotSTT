# core/no_window.py
# Keeps PolyglotSTT's helper-processes console-free on Windows.
#
# Deliberately minimal and boring, because EDR products (CrowdStrike etc.)
# treat "hide/detach my console" tricks (FreeConsole, ShowWindow(SW_HIDE) on a
# console, CREATE_NO_WINDOW used for stealth) as defense-evasion signals. So we
# do NOT do any of that here. The app is launched with pythonw.exe, which has
# no console in the first place, and the only thing we set is
# CREATE_NO_WINDOW on our own child processes (ffmpeg / nvidia-smi), which is
# the documented, ordinary way to stop a console program from opening a window.
#
# The app's GUI and its child processes are the only targets; nothing touches
# other processes, the system, or the user's session.
import os
import subprocess

IS_WINDOWS = os.name == "nt"

# Documented Windows process-creation flag: do not allocate a console for the
# child process. Set only on children we create ourselves.
CREATE_NO_WINDOW = 0x08000000

QUIET_FLAGS = CREATE_NO_WINDOW if IS_WINDOWS else 0


def child_flags() -> int:
    """Creation flags for our own helper processes (ffmpeg, nvidia-smi)."""
    return QUIET_FLAGS


def quiet_popen_kwargs() -> dict:
    """Keyword args to merge into subprocess.Popen/run(**kwargs)."""
    if not IS_WINDOWS:
        return {}
    return {"creationflags": QUIET_FLAGS}
