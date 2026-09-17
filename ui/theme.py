# ui/theme.py
# Color palette + dark/light themes (moved verbatim from gui.py in v1.4.0).
ACCENT = "#6C5CE7"
ACCENT_DARK = "#5A4BD1"
ACCENT_GLOW = "#A29BFE"
SUCCESS = "#00B894"
WARNING = "#FDCB6E"
DANGER = "#E17055"
BG_DARK = "#1A1A2E"
BG_CARD = "#16213E"
BG_INPUT = "#0F3460"
FG_PRIMARY = "#FFFFFF"
FG_SECONDARY = "#B2BEC3"
FG_DIM = "#636E72"
BTN_DIM = "#2D3748"
BTN_DIM_HOVER = "#4A5568"
BTN_GO_HOVER = "#00916E"
BTN_DANGER_HOVER = "#C0392B"
BTN_BURN = "#B5651D"
BTN_BURN_HOVER = "#8E4E15"
BTN_TEXT = "#FEFEFE"
THEMES = {
    "dark": {
        "ACCENT": "#6C5CE7", "ACCENT_DARK": "#5A4BD1",
        "ACCENT_GLOW": "#A29BFE", "SUCCESS": "#00B894",
        "WARNING": "#FDCB6E", "DANGER": "#E17055",
        "BG_DARK": "#1A1A2E", "BG_CARD": "#16213E", "BG_INPUT": "#0F3460",
        "FG_PRIMARY": "#FFFFFF", "FG_SECONDARY": "#B2BEC3",
        "FG_DIM": "#636E72",
        "BTN_DIM": "#2D3748", "BTN_DIM_HOVER": "#4A5568",
        "BTN_GO_HOVER": "#00916E", "BTN_DANGER_HOVER": "#C0392B",
        "BTN_BURN": "#B5651D", "BTN_BURN_HOVER": "#8E4E15",
        "BTN_TEXT": "#FEFEFE",
        "SEG_SELECTED": "#6C5CE7", "SEG_SELECTED_HOVER": "#5A4BD1",
    },
    "light": {
        "ACCENT": "#5A48D6", "ACCENT_DARK": "#4A38B8",
        "ACCENT_GLOW": "#5E35B1", "SUCCESS": "#00755C",
        "WARNING": "#8A5A00", "DANGER": "#C0392B",
        "BG_DARK": "#E9EDF2", "BG_CARD": "#FFFFFF", "BG_INPUT": "#DCE2EA",
        "FG_PRIMARY": "#16181D", "FG_SECONDARY": "#3E4450",
        "FG_DIM": "#687182",
        "BTN_DIM": "#CBD2DC", "BTN_DIM_HOVER": "#B6BECB",
        "BTN_GO_HOVER": "#006A51", "BTN_DANGER_HOVER": "#A93226",
        "BTN_BURN": "#A05A18", "BTN_BURN_HOVER": "#7E4A12",
        "BTN_TEXT": "#FEFEFE",
        "SEG_SELECTED": "#D9D2FB", "SEG_SELECTED_HOVER": "#C4B8F5",
    },
}
THEME_MODE = "dark"
# SEG_* are excluded from the value maps: their dark values intentionally
# match other roles (selected strip == ACCENT), and value-based mapping
# cannot tell roles apart. The strip is configured explicitly instead.
_SKIP_REMAP = ("SEG_SELECTED", "SEG_SELECTED_HOVER")
_DARK_TO_LIGHT = {v.upper(): THEMES["light"][k]
                  for k, v in THEMES["dark"].items()
                  if k not in _SKIP_REMAP}
_LIGHT_TO_DARK = {v.upper(): THEMES["dark"][k]
                  for k, v in THEMES["light"].items()
                  if k not in _SKIP_REMAP}


def theme_color(value):
    try:
        v = str(value or "")
        if not v:
            return value
        m = _DARK_TO_LIGHT if THEME_MODE == "light" else _LIGHT_TO_DARK
        return m.get(v.upper(), value)
    except Exception:
        return value
