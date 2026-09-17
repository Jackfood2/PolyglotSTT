# ui/__init__.py
# GUI package (split out of gui.py in v1.4.0). Theme constants and shared
# widgets live here; tab builders follow the same mixin pattern next.
#
# NOTE: palette values change on every dark/light toggle, so bare names
# are resolved dynamically via __getattr__ (always the live ui.theme
# value). Prefer `from ui import theme` + `theme.BG_CARD` in new code.
from . import theme as theme
from .widgets import LevelMeter, WaveformDisplay, StatusBadge, HistoryPanel

__all__ = [
    "theme",
    "ACCENT", "ACCENT_DARK", "ACCENT_GLOW", "SUCCESS", "WARNING",
    "DANGER", "BG_DARK", "BG_CARD", "BG_INPUT", "FG_PRIMARY",
    "FG_SECONDARY", "FG_DIM", "BTN_DIM", "BTN_DIM_HOVER",
    "BTN_GO_HOVER", "BTN_DANGER_HOVER", "BTN_BURN", "BTN_BURN_HOVER",
    "BTN_TEXT", "THEMES", "THEME_MODE", "theme_color",
    "LevelMeter", "WaveformDisplay", "StatusBadge", "HistoryPanel",
]


def __getattr__(name: str):
    # Proxy palette lookups to the live theme module (PEP 562).
    if name in __all__:
        try:
            return getattr(theme, name)
        except AttributeError:
            pass
    raise AttributeError(f"module 'ui' has no attribute {name!r}")

