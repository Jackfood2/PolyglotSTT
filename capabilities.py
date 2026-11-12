"""What each engine and model can actually do.

The UI uses this module to grey out options that the selected model cannot
perform (for example, Whisper large-v3-turbo has no translate task) and to
pick a compatible value when a stored selection is no longer valid.

Keep the tables in sync with the engines themselves:
  * canary_engine.CANARY_SOURCE_LANGS
  * whisper_engine.WHISPER_SOURCE_LANGS / WHISPER_MODEL_CHOICES
  * qwen_engine.QWEN_LANGUAGES
"""

from typing import Dict, Iterable, List, Optional, Tuple

TRANSCRIBE = "transcribe"
TRANSLATE = "translate"
TIMESTAMPS = "timestamps"
LANGUAGE = "language"
SRT = "srt"

ENGINES: Tuple[str, ...] = (
    "Moonshine v2", "Canary-1B", "Whisper", "Qwen3-ASR", "Granite",
)


_PREFERENCE = ("Whisper", "Canary-1B", "Qwen3-ASR", "Moonshine v2",
               "Granite")

_ENGINE_FEATURES: Dict[str, frozenset] = {

    "Moonshine v2": frozenset({TRANSCRIBE, TIMESTAMPS, SRT}),

    "Canary-1B": frozenset({TRANSCRIBE, TRANSLATE, LANGUAGE, SRT}),

    "Whisper": frozenset({TRANSCRIBE, TRANSLATE, LANGUAGE, TIMESTAMPS, SRT}),

    "Qwen3-ASR": frozenset({TRANSCRIBE, LANGUAGE, SRT}),

    # Granite TurboCTC: English only (no LANGUAGE feature, so the picker stays
    # off), transcribe only, no timestamps. Great for Live, fine for SRT via
    # VAD chunking, but punctuation-free output makes it a poor Notes engine.
    "Granite": frozenset({TRANSCRIBE, SRT}),
}


_MODEL_EXCLUSIONS: Dict[Tuple[str, str], frozenset] = {
    ("Whisper", "large-v3-turbo"): frozenset({TRANSLATE}),
}


TRANSLATE_TARGET = "en"

CANARY_LANGS: Tuple[str, ...] = ("auto", "en", "de", "es", "fr")


def _kind(engine: Optional[str]) -> str:
    try:
        value = str(engine or "").strip()
    except Exception:
        return ""
    return value if value in _ENGINE_FEATURES else ""


def engine_supports(engine: Optional[str], feature: str) -> bool:
    """True when the engine itself can perform `feature`."""
    kind = _kind(engine)
    if not kind:
        return False
    return feature in _ENGINE_FEATURES[kind]


def model_supports(engine: Optional[str], model_id: Optional[str],
                   feature: str) -> bool:
    """True when this engine+model pair can perform `feature`."""
    kind = _kind(engine)
    if not kind or feature not in _ENGINE_FEATURES[kind]:
        return False
    try:
        mid = str(model_id or "").strip().lower()
    except Exception:
        mid = ""
    for (ex_engine, ex_model), blocked in _MODEL_EXCLUSIONS.items():
        if kind == ex_engine and ex_model in mid and feature in blocked:
            return False
    return True


def supports(engine: Optional[str], model_id: Optional[str],
             feature: str) -> bool:
    return model_supports(engine, model_id, feature)


def engines_for(feature: str) -> List[str]:
    """Engines that support `feature`, in preference order."""
    return [e for e in _PREFERENCE if engine_supports(e, feature)]


def fallback_engine(feature: str, avoid: Optional[str] = None) -> Optional[str]:
    """Best engine other than `avoid` that supports `feature`."""
    for engine in engines_for(feature):
        if engine != avoid:
            return engine
    return None


def task_options(engine: Optional[str],
                 model_id: Optional[str] = None) -> List[str]:
    """Task values the model can actually run, transcribe first."""
    options = [TRANSCRIBE]
    if model_supports(engine, model_id, TRANSLATE):
        options.append(TRANSLATE)
    return options


def task_supported(engine: Optional[str], model_id: Optional[str],
                   task: Optional[str]) -> bool:
    try:
        value = str(task or TRANSCRIBE).strip().lower()
    except Exception:
        value = TRANSCRIBE
    if value not in (TRANSCRIBE, TRANSLATE):
        return False
    return model_supports(engine, model_id, value)


def coerce_task(engine: Optional[str], model_id: Optional[str],
                task: Optional[str]) -> str:
    """The requested task when valid, otherwise transcribe."""
    return str(task or TRANSCRIBE).strip().lower() \
        if task_supported(engine, model_id, task) else TRANSCRIBE


def translate_models(engine: Optional[str],
                     models: Iterable[str]) -> List[str]:
    """Subset of `models` that can translate with `engine`."""
    return [m for m in models if model_supports(engine, m, TRANSLATE)]


def best_translate_model(engine: Optional[str],
                         models: Iterable[str]) -> Optional[str]:
    """First model (in the given order) that can translate."""
    for model in models:
        if model_supports(engine, model, TRANSLATE):
            return model
    return None


def languages(engine: Optional[str]) -> Tuple[str, ...]:
    """Source language codes the engine accepts, `auto` first."""
    kind = _kind(engine)
    if kind == "Canary-1B":
        return CANARY_LANGS
    if kind == "Qwen3-ASR":
        try:
            from qwen_engine import QWEN_LANGUAGES
            return tuple(QWEN_LANGUAGES)
        except Exception:
            return ("auto",)
    if kind == "Granite":
        try:
            from granite_engine import GRANITE_LANGUAGES
            return tuple(GRANITE_LANGUAGES)
        except Exception:
            return ("auto", "en")
    if kind == "Whisper":
        try:
            from whisper_engine import WHISPER_SOURCE_LANGS
            return tuple(WHISPER_SOURCE_LANGS)
        except Exception:
            return ("auto",)
    return ("auto",)


def language_supported(engine: Optional[str], code: Optional[str]) -> bool:
    """True when the engine can be pinned to this source language."""
    kind = _kind(engine)
    if not kind:
        return False
    if not engine_supports(kind, LANGUAGE):
        return False
    try:
        value = str(code or "auto").strip().lower()
    except Exception:
        value = "auto"
    if value == "auto":
        return True
    return value in languages(kind)


def language_fallback(engine: Optional[str], code: Optional[str]) -> str:
    """The requested code when supported, otherwise auto."""
    return str(code or "auto").strip().lower() \
        if language_supported(engine, code) else "auto"


def output_requires_translate(input_lang: Optional[str],
                              output_lang: Optional[str]) -> bool:
    """True when writing `output_lang` means translating the source."""
    try:
        src = str(input_lang or "auto").strip().lower()
        out = str(output_lang or "").strip().lower()
    except Exception:
        return False
    if not out or out in ("same", "source", "auto"):
        return False
    if src in ("", "auto"):

        return out == TRANSLATE_TARGET
    return out != src


def describe_engine(engine: Optional[str]) -> str:
    """One-line capability summary for tooltips and hints."""
    kind = _kind(engine)
    if not kind:
        return ""
    if kind == "Moonshine v2":
        return "Moonshine v2 transcribes only (no language picker)."
    if kind == "Canary-1B":
        return "Canary-1B: en/de/es/fr, translate always targets English."
    if kind == "Whisper":
        return "Whisper: 99 languages, translate always targets English."
    if kind == "Qwen3-ASR":
        return "Qwen3-ASR transcribes 30 languages; no translation."
    if kind == "Granite":
        return ("Granite TurboCTC: English only, very fast, no punctuation, "
                "no timestamps. Best for Live dictation.")
    return ""
