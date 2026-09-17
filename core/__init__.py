# core/__init__.py
# App-wide configuration + small shared helpers (split out of
# moonshine_stt.py in v1.4.0). Import from here; moonshine_stt.py
# re-exports the same names so older imports keep working.
from .config import (
    CONFIG_PATH,
    DEFAULT_CONFIG,
    APP_VERSION,
    load_local_config,
    save_local_config,
    apply_suffix,
)

__all__ = [
    "CONFIG_PATH",
    "DEFAULT_CONFIG",
    "APP_VERSION",
    "load_local_config",
    "save_local_config",
    "apply_suffix",
]
