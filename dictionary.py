"""Shared dictionary / vocabulary biasing for the engines that support it.

Two engines can be nudged with a word list:

* **Whisper** (faster-whisper) accepts ``hotwords`` and ``initial_prompt``.
  Both are tokenised into the same 223-token decoder prompt
  (``max_length // 2 - 1`` with the default 448), and anything past that is
  silently clipped - hence the budget maths below.
* **Qwen3-ASR** (transcribe.cpp) accepts ``vocabulary`` (a priority-ordered
  list of terms) and ``prompt`` (free-text context). It has no hard cap in the
  wrapper, but long lists cost accuracy, so we warn past a soft limit.

Moonshine and Canary expose no equivalent knob, so they are not offered.

This module is pure logic (no Tk, no engine imports at module level) so it can
be unit-tested and imported from anywhere.
"""

from typing import Dict, Iterable, List, Optional, Sequence, Tuple

# Whisper's decoder prompt is capped at max_length // 2 - 1 tokens.
WHISPER_TOKEN_LIMIT = 223
# Qwen has no hard cap; these are quality guard rails.
QWEN_SOFT_LIMIT = 200
QWEN_HARD_LIMIT = 500
# A single term longer than this is almost certainly a pasted paragraph.
MAX_TERM_CHARS = 64

DICT_ENGINES: Tuple[str, ...] = ("Whisper", "Qwen3-ASR")

_ENGINE_HELP = {
    "Whisper": (
        "Whisper gets these as hotwords + initial prompt. The decoder only "
        "keeps 223 tokens, so long lists are clipped."
    ),
    "Qwen3-ASR": (
        "Qwen3-ASR gets these as a priority-ordered vocabulary plus your "
        "context text. Around 200 terms is the practical sweet spot."
    ),
}


def engine_help(engine: str) -> str:
    return _ENGINE_HELP.get(engine, "")


def norm_terms(raw: Optional[Iterable[str]]) -> List[str]:
    """Strip, drop blanks, cap length and de-duplicate (order preserved)."""
    out: List[str] = []
    seen = set()
    try:
        items = list(raw or [])
    except Exception:
        return out
    for item in items:
        try:
            term = str(item).strip()
        except Exception:
            continue
        if not term:
            continue
        if len(term) > MAX_TERM_CHARS:
            term = term[:MAX_TERM_CHARS].rstrip()
        key = term.casefold()
        if key in seen:
            continue
        seen.add(key)
        out.append(term)
    return out


def norm_context(raw: Optional[str]) -> str:
    try:
        text = str(raw or "").replace("\r\n", "\n").strip()
    except Exception:
        return ""
    return text


def estimate_tokens(text: str) -> int:
    """Rough token count for English-ish text.

    BPE averages ~4 characters per token, but short proper nouns tokenise
    worse than that, so take the larger of a character and a word estimate.
    Deliberately slightly pessimistic: under-reporting would let Whisper clip
    a term the user believes is active.
    """
    try:
        s = str(text or "")
    except Exception:
        return 0
    if not s.strip():
        return 0
    chars = len(s)
    words = len(s.split())
    by_chars = (chars + 3) // 4
    by_words = (words * 3 + 1) // 2  # ~1.5 tokens per word
    return max(by_chars, by_words, 1)


def _whisper_budget(terms: Sequence[str], context: str) -> Dict:
    """Fit terms + context into Whisper's 223-token decoder prompt.

    Hotwords win the budget; the context text gets whatever is left.
    """
    budget = WHISPER_TOKEN_LIMIT
    used = 0
    kept: List[str] = []
    for term in terms:
        cost = estimate_tokens(term) + 1  # trailing comma separator
        if used + cost > budget:
            # Skip this term but keep scanning: a shorter one later may fit.
            continue
        kept.append(term)
        used += cost
    remaining = max(0, budget - used)
    prompt = ""
    if context and remaining > 0:
        # Trim on word boundaries until the *estimated* count fits. A plain
        # chars/4 cut under-counts wordy text, which used to push the prompt
        # past 223 tokens (Whisper then silently clipped a hotword).
        target = max(0, remaining - 1)  # headroom for the ellipsis
        limit_chars = max(0, target * 4)
        candidate = context[:limit_chars].rstrip() if limit_chars else ""
        while candidate and estimate_tokens(candidate) > target:
            words = candidate.split()
            if len(words) <= 1:
                candidate = ""
                break
            candidate = " ".join(words[:-1]).rstrip()
        prompt = candidate
        if prompt and prompt != context:
            prompt += " …"
        used += estimate_tokens(prompt)
    hotwords = ", ".join(kept)
    return {
        "engine": "Whisper",
        "eligible": True,
        "limit": budget,
        "tokens": used,
        "kept": kept,
        "dropped": len(terms) - len(kept),
        "hotwords": hotwords or None,
        "prompt": prompt or None,
        "warning": "",
    }


def _qwen_budget(terms: Sequence[str], context: str) -> Dict:
    limit = QWEN_SOFT_LIMIT
    kept = list(terms)
    warning = ""
    if len(kept) > QWEN_HARD_LIMIT:
        kept = kept[:QWEN_HARD_LIMIT]
        warning = (
            f"Qwen3-ASR got more than {QWEN_HARD_LIMIT} terms; "
            f"the rest were ignored."
        )
    elif len(kept) > QWEN_SOFT_LIMIT:
        warning = (
            f"{len(kept)} terms is a lot for Qwen3-ASR; accuracy may drop. "
            f"~{QWEN_SOFT_LIMIT} works best."
        )
    return {
        "engine": "Qwen3-ASR",
        "eligible": True,
        "limit": limit,
        "tokens": len(kept),
        "kept": kept,
        "dropped": len(terms) - len(kept),
        "vocabulary": list(kept) or None,
        "prompt": context or None,
        "warning": warning,
    }


def build(engine: str, terms: Sequence[str], context: str) -> Dict:
    """Return the decode payload plus budget info for one engine."""
    clean_terms = norm_terms(terms)
    clean_ctx = norm_context(context)
    if engine == "Whisper":
        return _whisper_budget(clean_terms, clean_ctx)
    if engine == "Qwen3-ASR":
        return _qwen_budget(clean_terms, clean_ctx)
    return {
        "engine": engine,
        "eligible": False,
        "limit": 0,
        "tokens": 0,
        "kept": [],
        "dropped": 0,
        "warning": f"{engine} does not support a custom dictionary.",
    }


def engine_available(engine: str) -> bool:
    """True when the engine's runtime is importable on this install."""
    try:
        if engine == "Whisper":
            import faster_whisper  # noqa: F401
            return True
        if engine == "Qwen3-ASR":
            import transcribe_cpp  # noqa: F401
            return True
    except Exception:
        return False
    return False


def available_map() -> Dict[str, bool]:
    return {e: engine_available(e) for e in DICT_ENGINES}


def enabled_map(raw: Optional[Dict]) -> Dict[str, bool]:
    """Normalise the stored per-engine enable flags."""
    out: Dict[str, bool] = {}
    data = raw if isinstance(raw, dict) else {}
    for engine in DICT_ENGINES:
        out[engine] = bool(data.get(engine, False))
    return out


def describe(engine: str, terms: Sequence[str], context: str) -> str:
    """One-line status for the dialog, e.g. '143 / 223 tokens ✓'."""
    info = build(engine, terms, context)
    if not info.get("eligible"):
        return f"{engine}: not supported"
    if engine == "Whisper":
        text = f"{info['tokens']} / {info['limit']} tokens"
        if info["dropped"]:
            text += f" — {info['dropped']} term(s) over the limit"
        elif info["tokens"] > info["limit"]:
            text += " — over the limit"
        else:
            text += " ✓"
        return text
    count = len(info["kept"])
    text = f"{count} term(s)"
    if count > QWEN_SOFT_LIMIT:
        text += f" — over the ~{QWEN_SOFT_LIMIT} sweet spot"
    else:
        text += " ✓"
    return text


def trim_to_fit(terms: Sequence[str], context: str,
                engines: Sequence[str]) -> List[str]:
    """Cut the list down so every enabled engine keeps all of it."""
    clean = norm_terms(terms)
    keep_n = len(clean)
    for engine in engines:
        info = build(engine, clean, context)
        if info.get("eligible"):
            keep_n = min(keep_n, len(info.get("kept") or []))
    return clean[:keep_n]


def status_text(terms: Sequence[str], context: str,
                engines: Dict[str, bool]) -> str:
    """Compact summary for a status line / log message."""
    active = [e for e, on in (engines or {}).items() if on]
    if not active:
        return "Dictionary off"
    parts = []
    for engine in active:
        info = build(engine, terms, context)
        if engine == "Whisper":
            parts.append(f"Whisper {info['tokens']}/{info['limit']} tok")
        else:
            parts.append(f"Qwen {len(info['kept'])} terms")
    return "Dictionary on — " + ", ".join(parts)
