# services/__init__.py
# Backend services package (v1.4.0). Sentence formatting for file imports
# lives here; note_engine.py re-exports the same names.
from .note_format import format_note_sentences, format_note_file_text

__all__ = ["format_note_sentences", "format_note_file_text"]
