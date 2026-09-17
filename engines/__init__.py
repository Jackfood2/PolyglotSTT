# engines/__init__.py
# Engine package (v1.4.0): one wrapper per STT backend behind a uniform
# `transcribe(audio, sample_rate)` / `is_ready` surface.
#
#   engines.moonshine  -> Moonshine v2 lightweight English ASR
#   engines.canary     -> NVIDIA Canary-1B ASR + translation
#   engines.whisper    -> Faster-Whisper (native timestamps)
#   engines.note       -> Note-mode chunked recorder + async transcriber
#
# The top-level engine.py / canary_engine.py / whisper_engine.py /
# note_engine.py modules stay as-is (they ARE the package members) and
# are re-exported here, so both `from engine import X` and
# `from engines.moonshine import X` work.
try:
    from engine import (
        TranscriptionEngine,
        MODEL_ARCH_NAMES,
        moonshine_model_dir,
        moonshine_downloaded_map,
        moonshine_cache_info,
        delete_moonshine_model,
    )
except Exception:
    TranscriptionEngine = None
    MODEL_ARCH_NAMES = {}
    moonshine_model_dir = None
    moonshine_downloaded_map = None
    moonshine_cache_info = None
    delete_moonshine_model = None

try:
    from note_engine import (
        NoteRecorder,
        NoteTranscriber,
        cleanup_stale_session_wavs,
        wav_to_mp3,
    )
except Exception:
    NoteRecorder = None
    NoteTranscriber = None
    cleanup_stale_session_wavs = None
    wav_to_mp3 = None

try:
    from services.note_format import (
        format_note_sentences,
        format_note_file_text,
    )
except Exception:
    format_note_sentences = None
    format_note_file_text = None

__all__ = [
    "TranscriptionEngine",
    "MODEL_ARCH_NAMES",
    "moonshine_model_dir",
    "moonshine_downloaded_map",
    "moonshine_cache_info",
    "delete_moonshine_model",
    "NoteRecorder",
    "NoteTranscriber",
    "cleanup_stale_session_wavs",
    "wav_to_mp3",
    "format_note_sentences",
    "format_note_file_text",
]
