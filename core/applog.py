# core/applog.py
# File logging + crash diagnostics for the packaged/windowless app.
# Everything here is best-effort: logging must never take the app down.
import logging
import os
import sys
import threading
from logging.handlers import RotatingFileHandler

LOG_DIR = os.path.join(
    os.environ.get("LOCALAPPDATA") or os.path.expanduser("~"),
    "PolyglotSTT", "logs")
LOG_PATH = os.path.join(LOG_DIR, "app.log")
CRASH_PATH = os.path.join(LOG_DIR, "crash.log")

_configured = False
_fault_file = None


def setup_logging():
    """Install file logging plus excepthook/faulthandler. Idempotent."""
    global _configured, _fault_file
    if _configured:
        return LOG_PATH
    _configured = True
    try:
        os.makedirs(LOG_DIR, exist_ok=True)
        root = logging.getLogger()
        root.setLevel(logging.INFO)
        fmt = logging.Formatter(
            "%(asctime)s %(levelname)s [%(threadName)s] %(message)s")
        fh = RotatingFileHandler(LOG_PATH, maxBytes=1_000_000,
                                 backupCount=3, encoding="utf-8",
                                 delay=True)
        fh.setFormatter(fmt)
        root.addHandler(fh)
        # Keep the console (run.bat) informative too, when there is one.
        try:
            if sys.stderr is not None:
                sh = logging.StreamHandler(sys.stderr)
                sh.setFormatter(fmt)
                root.addHandler(sh)
        except Exception:
            pass

        _orig_hook = sys.excepthook

        def _hook(exc_type, exc, tb):
            try:
                logging.getLogger("polyglot").critical(
                    "Uncaught exception", exc_info=(exc_type, exc, tb))
            except Exception:
                pass
            try:
                _orig_hook(exc_type, exc, tb)
            except Exception:
                pass

        sys.excepthook = _hook

        def _thread_hook(args):
            try:
                logging.getLogger("polyglot").critical(
                    "Unhandled thread exception in %s",
                    getattr(getattr(args, "thread", None), "name", "?"),
                    exc_info=(args.exc_type, args.exc_value,
                              args.exc_traceback))
            except Exception:
                pass

        try:
            threading.excepthook = _thread_hook
        except Exception:
            pass
        try:
            import faulthandler
            _fault_file = open(CRASH_PATH, "a", encoding="utf-8",
                               buffering=1)
            faulthandler.enable(file=_fault_file, all_threads=True)
        except Exception:
            _fault_file = None
        logging.getLogger("polyglot").info(
            "Logging initialized: %s", LOG_PATH)
    except Exception:
        pass
    return LOG_PATH


def log(msg):
    try:
        logging.getLogger("polyglot").info(str(msg))
    except Exception:
        pass


def flush():
    """Best-effort flush before os._exit()."""
    try:
        for h in logging.getLogger().handlers:
            try:
                h.flush()
            except Exception:
                pass
    except Exception:
        pass
