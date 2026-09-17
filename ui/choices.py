# ui/choices.py
# Static menus, labels and badge helper (split verbatim from gui.py in v1.4.0).


MODEL_CHOICES = {
    "Tiny (26MB, fastest)": 0,
    "Tiny Streaming (45MB, light)": 2,
    "Base (60MB, balanced)": 1,
    "Base Streaming (60MB, light)": 3,
    "Small Streaming (80MB, good)": 4,
    "Medium Streaming (110MB, best)": 5,
}


MODEL_CHOICES_REV = {v: k for k, v in MODEL_CHOICES.items()}


try:
    from whisper_engine import (WHISPER_MODEL_CHOICES as _WMC,
                                WHISPER_MODEL_CHOICES_REV as _WMCR)
    WHISPER_MODEL_CHOICES = dict(_WMC)
    WHISPER_MODEL_CHOICES_REV = dict(_WMCR)
except Exception:
    WHISPER_MODEL_CHOICES = {
        "Tiny (75MB, fastest)": "tiny",
        "Base (145MB)": "base",
        "Small (500MB)": "small",
        "Medium (1.5GB)": "medium",
        "Large v1 (3GB, oldest)": "large-v1",
        "Large v2 (3GB)": "large-v2",
        "Large v3 (3GB, best)": "large-v3",
    }
    WHISPER_MODEL_CHOICES_REV = {v: k for k, v in WHISPER_MODEL_CHOICES.items()}


CANARY_MODEL_LABEL = "Canary-1B (3.9GB, fixed)"


BURN_SPEED_HELP = {
    "Match size (2-pass x264)": "Exact size (±1–3%). Slowest — encodes the video twice.",
    "Fast (1-pass x264)": "About half the time, size within ~±10%.",
    "Fastest (ultrafast 1-pass)": "Several times faster, visibly softer. For quick checks, not keeps.",
    "Draft (NVENC fast 1-pass)": "Needs NVIDIA GPU. Fastest encode, size approximate (~±10%).",
    "Turbo (NVENC ultra-fast 1-pass)": "Needs NVIDIA GPU. Lowest-latency tune, a touch softer than Draft.",
    "Balanced (NVENC 2-pass)": "Needs NVIDIA GPU. GPU two-pass, closer size (~±5–10%).",
}


BURN_COMPARE_ROWS = (
    ("Match size (2-pass)", "x264 CPU", "±1–3% (exact)",
     "1× baseline (slow)", "everything incl. old devices",
     "archival keeps, exact-size needs"),
    ("Fast (1-pass)", "x264 CPU", "~±10%",
     "~2×", "everything incl. old devices",
     "quick CPU burns"),
    ("Fastest (ultrafast 1-pass)", "x264 CPU", "~±10%, visibly softer",
     "~4–6×", "everything incl. old devices",
     "drafts and checks"),
    ("Draft (NVENC 1-pass)", "NVENC GPU", "~±10% (VBR)",
     "very fast (100s of fps)", "modern devices (~2016+)",
     "fast GPU burns"),
    ("Turbo (NVENC ultra-fast)", "NVENC GPU", "~±10%, a touch softer",
     "fastest", "modern devices (~2016+)",
     "speed runs, previews"),
    ("Balanced (NVENC 2-pass)", "NVENC GPU", "~±5–10%",
     "fast", "modern devices (~2016+)",
     "best GPU quality per minute"),
)


BURN_COMPARE_CODEC_NOTE = (
    "Codec switch: HEVC (H.265) ≈30% smaller than H.264 at the same visual "
    "quality - via NVENC at the same GPU speed, or via x265 on CPU "
    "(≈30–40% smaller but much slower). Needs ~2016+ playback hardware; "
    "H.264 plays on everything. Learned estimates, the MB box, and "
    "overshoot compensation track each speed+codec separately."
)


def apply_badge_map(values, status_fn=None):
    mapping = {}
    displays = []
    for v in (values or []):
        mark = ""
        if status_fn is not None:
            try:
                mark = (status_fn(v) or "").strip()
            except Exception:
                mark = ""
        d = f"{v} {mark}" if mark else str(v)
        k, n = d, 2
        while k in mapping:
            k = f"{d} ({n})"
            n += 1
        mapping[k] = v
        displays.append(k)
    return displays, mapping


ENGINE_CHOICES = ["Moonshine v2", "Canary-1B", "Whisper"]


CANARY_TASKS = ["transcribe", "translate"]


CANARY_LANGS = ["auto", "en", "de", "es", "fr"]


WHISPER_TASKS = ["transcribe", "translate"]


WHISPER_LANGS = ["auto", "en", "ja", "zh", "ko", "de", "es", "fr",
                 "it", "pt", "nl", "ru", "ar", "hi", "tr", "id", "uk", "vi", "th"]


SRT_LANGS = ["auto", "en", "ja", "zh", "ko", "de", "es", "fr",
             "it", "pt", "nl", "ru", "ar", "hi", "tr", "id", "uk", "vi", "th"]


SRT_LANG_NAMES = {
    "auto": "Auto-detect", "en": "English", "ja": "Japanese", "zh": "Chinese",
    "ko": "Korean", "de": "German", "es": "Spanish", "fr": "French",
    "it": "Italian", "pt": "Portuguese", "nl": "Dutch", "ru": "Russian",
    "ar": "Arabic", "hi": "Hindi", "tr": "Turkish", "id": "Indonesian",
    "uk": "Ukrainian", "vi": "Vietnamese", "th": "Thai",
}


SRT_LANG_DISPLAY = [SRT_LANG_NAMES[c] for c in SRT_LANGS]


SRT_LANG_CODE_FROM_DISPLAY = {v: k for k, v in SRT_LANG_NAMES.items()}


CANARY_LIVE_SRC_CODES = ["auto", "en", "de", "es", "fr"]


CANARY_SRT_IN_CODES = ["auto", "en", "de", "es", "fr"]


CANARY_SRT_OUT_CODES = ["en", "de", "es", "fr"]


WHISPER_SRT_IN_CODES = list(SRT_LANGS)


WHISPER_SRT_OUT_CODES = [c for c in SRT_LANGS if c != "auto"]
