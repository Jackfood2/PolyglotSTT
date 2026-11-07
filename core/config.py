# core/config.py
# Single source of truth for app config (moved verbatim from
# moonshine_stt.py in v1.4.0 - no behavior change).
import json
import os
import threading

APP_VERSION = "1.4.9"

PROJECT_ROOT = os.path.dirname(os.path.dirname(__file__))
CONFIG_PATH = os.path.join(PROJECT_ROOT, "moonshine_config.json")
NOTES_AUDIO_DIR = os.path.join(PROJECT_ROOT, "notes_audio")
_CONFIG_LOCK = threading.RLock()

DEFAULT_CONFIG = {
    "typing_method": "clipboard",
    "suffix": "none",
    "typing_delay_ms": 0,
    "model_arch": 5,
    "engine": "Moonshine v2",
    "canary_task": "transcribe",
    "canary_src_lang": "auto",
    "whisper_task": "transcribe",
    "whisper_src_lang": "auto",
    "qwen_model": "qwen3-asr-1.7b-q8_0",
    "qwen_src_lang": "auto",
    "dict_terms": [],
    "dict_context": "",
    "dict_engines": {"Whisper": False, "Qwen3-ASR": False},
    "whisper_model": "large-v3-turbo",
    "whisper_device": "auto",
    "compute": "auto",
    "srt_cpu": 0,
    "srt_out_dir": "",
    "srt_input_lang": "auto",
    "srt_output_lang": "en",
    "burn_font_size": 18,
    "burn_sample_start": "0:30",
    "burn_sample_len": 15,
    "burn_vbr_auto": True,
    "burn_vbr_kbps": 2000,
    "burn_speed": "match",
    "burn_codec": "h264",
    "srt_tab": "Live",
    "srt_norm": False,
    "burn_after": False,
    "auto_shutdown": False,
    "completion_alert": True,
    "theme": "dark",
    "windowless": False,
}


def load_local_config():
    try:
        if os.path.exists(CONFIG_PATH):
            with open(CONFIG_PATH, "r", encoding="utf-8") as f:
                cfg = json.load(f)
            out = dict(DEFAULT_CONFIG)
            out.update(cfg)
            if any(k not in cfg for k in DEFAULT_CONFIG):
                # Use the atomic writer (tmp + os.replace) for migration
                # too: a crash mid-write must not truncate the config.
                save_local_config(out)
            return out
    except Exception:
        pass
    return dict(DEFAULT_CONFIG)


def save_local_config(cfg):
    try:
        with _CONFIG_LOCK:
            tmp = CONFIG_PATH + ".tmp"
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump(cfg, f, indent=2)
            os.replace(tmp, CONFIG_PATH)
    except Exception:
        pass


def apply_suffix(text: str, suffix: str) -> str:
    if suffix == "space":
        return text + " "
    if suffix == "newline":
        return text + "\n"
    if suffix == "period_space":
        return text + ". "
    return text
