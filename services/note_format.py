# services/note_format.py
# Sentence formatting for imported audio/video transcriptions: one
# sentence per line so the note reads as sentences, not one wall of text.
# Multilingual by design (Latin . ! ? … + CJK 。 ！ ？ …).
# (Moved verbatim from note_engine.py in v1.4.0 - no behavior change.)
_ABBREV_TAILS = frozenset({
    "mr.", "mrs.", "ms.", "dr.", "prof.", "sr.", "jr.", "st.",
    "vs.", "etc.", "e.g.", "i.e.", "fig.", "no.", "approx.",
})


def format_note_sentences(text: str) -> str:
    """Split raw transcription into one sentence per line.

    - Collapses stray whitespace/newlines first.
    - CJK enders (。！？…) break even without a following space.
    - Latin enders (. ! ? …) break on following whitespace.
    - Common abbreviations (Mr., e.g., …) are glued back together.
    - Returns "" for empty input; otherwise lines joined by "\\n"."""
    import re as _re
    try:
        t = str(text or "")
    except Exception:
        return ""
    try:
        t = t.replace("\r\n", "\n").replace("\r", "\n")
        t = _re.sub(r"[ \t\u00a0]+", " ", t)
        t = _re.sub(r" *\n *", "\n", t)
        t = _re.sub(r"\n{3,}", "\n\n", t).strip()
        if not t:
            return ""
        # Work on a single flowing line; existing newlines are just spaces
        # (chunk joins already separate ideas - re-splitting is cleaner).
        flat = _re.sub(r"\s*\n\s*", " ", t).strip()
        flat = _re.sub(r" {2,}", " ", flat)
        # CJK enders: break even with no following space.
        flat = _re.sub(r"([。！？…]+[\"”’')\]]*)", r"\1\n", flat)
        # Latin enders: break only on following whitespace (protects
        # file.mp3, 3.14, urls... from splitting mid-token).
        flat = _re.sub(r"([.!?…]+[\"”’')\]]*)\s+", r"\1\n", flat)
        cands = [c.strip() for c in flat.split("\n")]
        cands = [c for c in cands if c]
        # Re-glue abbreviations split above ("Mr." + "Smith" -> one line).
        out = []
        for c in cands:
            try:
                tail = c.strip().split()[-1].lower() if c.strip() else ""
            except Exception:
                tail = ""
            if out and tail in _ABBREV_TAILS:
                pass  # current line IS the abbrev tail start - keep, merge next
            if out:
                try:
                    prev_tail = out[-1].strip().split()[-1].lower()
                except Exception:
                    prev_tail = ""
                if prev_tail in _ABBREV_TAILS:
                    out[-1] = (out[-1] + " " + c).strip()
                    continue
            out.append(c)
        return "\n".join(out).strip()
    except Exception:
        try:
            return str(text or "").strip()
        except Exception:
            return ""


def format_note_file_text(filename: str, engine_label: str,
                           sentences: str) -> str:
    """Header + sentence body for an imported file. Never raises."""
    try:
        import time as _t
        name = str(filename or "audio")
        eng = str(engine_label or "").strip()
        body = str(sentences or "").strip()
        stamp = _t.strftime("%Y-%m-%d %H:%M")
        head = f"─── File: {name} ({stamp})"
        if eng:
            head += f" [{eng}]"
        head += " ───"
        return f"{head}\n{body}\n" if body else f"{head}\n(no speech detected)\n"
    except Exception:
        try:
            return str(sentences or "").strip()
        except Exception:
            return ""
