"""Inverse Text Normalization (ITN) for the Live tab.

Speech recognisers emit verbatim (spoken-form) text: Qwen3-ASR in
particular writes "meet me at three thirty" and "five dollars" instead of
"meet me at 3:30" and "$5". This module rewrites those spoken forms into
written forms after the decode.

Pure Python, no dependencies (no NeMo / torch download, no network).
Deliberately conservative about ambiguous words: "one of them", "no one",
"a second", "first of all" and "one third" are left alone, while clear
numeric uses are converted.

Public API
----------
``normalize(text)``  rewrite a transcript string.
``QWEN_HINT``        prompt line for Qwen3-ASR "native transcription
                     polishing" (digits instead of number words).
"""

import re

# --------------------------------------------------------------------------
# Word tables
# --------------------------------------------------------------------------
_UNITS = {
    "zero": 0, "one": 1, "two": 2, "three": 3, "four": 4,
    "five": 5, "six": 6, "seven": 7, "eight": 8, "nine": 9,
}
_TEENS = {
    "ten": 10, "eleven": 11, "twelve": 12, "thirteen": 13,
    "fourteen": 14, "fifteen": 15, "sixteen": 16,
    "seventeen": 17, "eighteen": 18, "nineteen": 19,
}
_TENS = {
    "twenty": 20, "thirty": 30, "forty": 40, "fifty": 50,
    "sixty": 60, "seventy": 70, "eighty": 80, "ninety": 90,
}
_SCALES = {
    "hundred": 100, "thousand": 1000, "million": 10 ** 6,
    "billion": 10 ** 9, "trillion": 10 ** 12,
}

# Ordinal words (except "second", which is handled with a guard because it
# is far more common as a unit of time: "wait a second").
_ORDINALS = {
    "first": 1, "third": 3, "fourth": 4, "fifth": 5, "sixth": 6,
    "seventh": 7, "eighth": 8, "ninth": 9, "tenth": 10,
    "eleventh": 11, "twelfth": 12, "thirteenth": 13,
    "fourteenth": 14, "fifteenth": 15, "sixteenth": 16,
    "seventeenth": 17, "eighteenth": 18, "nineteenth": 19,
    "twentieth": 20, "thirtieth": 30, "fortieth": 40,
    "fiftieth": 50, "sixtieth": 60, "seventieth": 70,
    "eightieth": 80, "ninetieth": 90,
    "hundredth": 100, "thousandth": 1000, "millionth": 10 ** 6,
}
_ORDINAL_UNITS = {"first", "second", "third", "fourth", "fifth", "sixth",
                  "seventh", "eighth", "ninth"}

_PERCENT = {"percent", "percentage", "pct"}
_CURRENCY = {
    "dollar": "$", "dollars": "$", "buck": "$", "bucks": "$",
    "usd": "$",
    "euro": "\u20ac", "euros": "\u20ac", "eur": "\u20ac",
    "pound": "\u00a3", "pounds": "\u00a3", "quid": "\u00a3",
    "gbp": "\u00a3",
    "yen": "\u00a5", "jpy": "\u00a5",
}
_DEGREES = {"degree", "degrees"}
_MERIDIEM = {"am": "AM", "pm": "PM", "a.m.": "AM", "p.m.": "PM"}

# Single "one" is often a pronoun, not a quantity.
_ONE_NEXT_GUARD = {
    "of", "another", "more", "less", "thing", "things", "day", "days",
    "time", "times", "way", "ways", "side", "hand", "hands", "half",
    "third", "quarter", "moment", "place", "step", "by", "on", "in",
}
_ONE_PREV_GUARD = {
    "no", "any", "some", "every", "each", "this", "that", "the",
    "which", "another", "other", "just", "only",
}

_TOKEN_RE = re.compile(r"[A-Za-z]+(?:'[A-Za-z]+)?")

QWEN_HINT = (
    "Write numbers, times, dates, money and percentages as digits, not as "
    "words, e.g. 3, 3:30 PM, 2024-05-01, $5, 25%."
)


# --------------------------------------------------------------------------
# Small helpers
# --------------------------------------------------------------------------
def _ordinal_suffix(value: int) -> str:
    if 10 <= value % 100 <= 20:
        return "th"
    return {1: "st", 2: "nd", 3: "rd"}.get(value % 10, "th")


def _kind(word: str):
    if word in _UNITS:
        return "U"
    if word in _TEENS:
        return "T"
    if word in _TENS:
        return "N"
    if word == "hundred":
        return "H"
    if word in _SCALES:
        return "S"
    if word == "and":
        return "A"
    return None


# Grammar over token kinds for a well-formed cardinal phrase:
#   chunk = small | [small] hundred [and small] | hundred [and small]
#   full  = [chunk] (scale [and] [chunk])*
_SMALL = r"(?:U|T|NU?)"
_CHUNK = r"(?:(?:{s})?H(?:A(?:{s}))?|{s})".format(s=_SMALL)
_FULL_RE = re.compile(r"(?:{c})?(?:S(?:A)?{c}?)*".format(c=_CHUNK))


def _phrase_value(words):
    total = 0
    current = 0
    for word in words:
        if word == "and":
            continue
        if word in _UNITS:
            current += _UNITS[word]
        elif word in _TEENS:
            current += _TEENS[word]
        elif word in _TENS:
            current += _TENS[word]
        elif word == "hundred":
            current = (current or 1) * 100
        else:
            total += (current or 1) * _SCALES[word]
            current = 0
    return total + current


def _joinable(text: str, left_end: int, right_start: int) -> bool:
    """True when two words are separated only by spaces or one hyphen."""
    sep = text[left_end:right_start]
    if not sep:
        return True
    if sep == "-":
        return True
    return bool(re.fullmatch(r"\s+", sep))


def _collect_phrase(matches, i, text):
    """Return (words, end_index) for the longest valid cardinal run at i."""
    words = []
    ends = []
    j = i
    while j < len(matches):
        word = matches[j].group(0).lower()
        kind = _kind(word)
        if kind is None:
            break
        if words and not _joinable(text, matches[j - 1].end(),
                                   matches[j].start()):
            break
        words.append(word)
        ends.append(j)
        j += 1
    for length in range(len(words), 0, -1):
        candidate = words[:length]
        if candidate[-1] == "and" or candidate[0] == "and":
            continue
        kinds = "".join(_kind(w) for w in candidate)
        if _FULL_RE.fullmatch(kinds):
            return candidate, ends[length - 1] + 1
    return None, i


def _suffix_after(matches, words, j, text):
    """Read a unit word right after a number (percent/currency/degrees)."""
    if j >= len(matches):
        return None, j
    word = matches[j].group(0).lower()
    if not _joinable(text, matches[j - 1].end(), matches[j].start()):
        return None, j
    if word in _PERCENT:
        return "%", j + 1
    if word == "per" and j + 1 < len(matches) and words[j + 1] == "cent":
        if _joinable(text, matches[j].end(), matches[j + 1].start()):
            return "%", j + 2
    if word in _CURRENCY:
        return "\0" + _CURRENCY[word], j + 1
    if word in _DEGREES:
        return "\u00b0", j + 1
    return None, j


def _format_number(value, suffix):
    if isinstance(value, float):
        text = ("%.10f" % value).rstrip("0").rstrip(".")
    else:
        text = str(value)
    if suffix is None:
        return text
    if suffix.startswith("\0"):
        return suffix[1:] + text
    return text + suffix


# --------------------------------------------------------------------------
# Main scan
# --------------------------------------------------------------------------
def normalize(text):
    """Convert spoken-form numbers in ``text`` to written-form digits."""
    if not text:
        return text
    try:
        return _normalize(str(text))
    except Exception:
        return text


def _normalize(text: str) -> str:
    matches = list(_TOKEN_RE.finditer(text))
    if not matches:
        return text
    words = [m.group(0).lower() for m in matches]
    pieces = []
    cursor = 0
    i = 0
    total = len(matches)

    while i < total:
        replaced = _try_at(matches, words, text, i)
        if replaced is None:
            i += 1
            continue
        replacement, j = replaced
        pieces.append(text[cursor:matches[i].start()])
        pieces.append(replacement)
        cursor = matches[j - 1].end()
        i = j

    pieces.append(text[cursor:])
    return "".join(pieces)


def _try_at(matches, words, text, i):
    word = words[i]

    # "a hundred", "an hour" style article before a scale: swallow the
    # article so the result reads "100" rather than "a 100".
    if word in ("a", "an") and i + 1 < len(words) and words[i + 1] in _SCALES:
        nxt = _try_at(matches, words, text, i + 1)
        if nxt is not None:
            return nxt

    # minus / negative prefix
    negative = False
    if word in ("minus", "negative"):
        nxt = _try_at(matches, words, text, i + 1)
        if nxt is None:
            return None
        replacement, j = nxt
        if replacement[:1].isdigit():
            return "-" + replacement, j
        return None

    # ordinal words ("first", "twentieth", ...)
    result = _ordinal_at(matches, words, text, i)
    if result is not None:
        return result

    # cardinal phrase
    phrase, j = _collect_phrase(matches, i, text)
    if not phrase:
        return None
    value = _phrase_value(phrase)

    # "one" / "two" style pronouns and idioms
    if len(phrase) == 1 and value == 1:
        prev = words[i - 1] if i else ""
        nxt = words[j] if j < len(words) else ""
        if prev in _ONE_PREV_GUARD or nxt in _ONE_NEXT_GUARD:
            return None

    # decimal: "three point five" -> 3.5
    decimal = _decimal_at(matches, words, text, j, value)
    if decimal is not None:
        value, j = decimal

    # time: "three thirty pm" -> 3:30 PM
    clock = _time_at(matches, words, text, j, value)
    if clock is not None:
        value, j = clock

    suffix, j2 = _suffix_after(matches, words, j, text)
    if suffix is not None:
        j = j2

    return _format_number(value, suffix), j


def _ordinal_at(matches, words, text, i):
    word = words[i]

    # Compound: "twenty-first", "thirty second" -> 21st / 32nd.
    if word in _TENS and i + 1 < len(words):
        nxt = words[i + 1]
        if nxt in _ORDINAL_UNITS or nxt == "second":
            if _joinable(text, matches[i].end(), matches[i + 1].start()):
                unit = 2 if nxt == "second" else _ORDINALS[nxt]
                value = _TENS[word] + unit
                return (f"{value}{_ordinal_suffix(value)}", i + 2)

    if word == "second":
        prev = words[i - 1] if i else ""
        if prev in ("a", "an", "any", "one", "every", "each"):
            return None
        value = 2
    elif word in _ORDINALS:
        prev = words[i - 1] if i else ""
        nxt = words[i + 1] if i + 1 < len(words) else ""
        # "first of all", "at first", "one third" are not quantities.
        if word == "first" and (nxt == "of" or prev == "at"):
            return None
        if prev in ("a", "an", "one", "two", "three", "four", "five",
                    "six", "seven", "eight", "nine", "ten"):
            return None
        value = _ORDINALS[word]
    else:
        return None

    return (f"{value}{_ordinal_suffix(value)}", i + 1)


def _decimal_at(matches, words, text, j, value):
    if j >= len(words) or words[j] not in ("point",):
        return None
    # digits after the point are spoken one by one
    k = j + 1
    digits = []
    while k < len(words) and words[k] in _UNITS:
        if digits and not _joinable(text, matches[k - 1].end(),
                                    matches[k].start()):
            break
        digits.append(str(_UNITS[words[k]]))
        k += 1
    if not digits:
        return None
    if not _joinable(text, matches[j - 1].end(), matches[j].start()):
        return None
    return float(f"{value}." + "".join(digits)), k


def _time_at(matches, words, text, j, value):
    """Recognise "three thirty", "three oh five", "three pm" as times."""
    if not isinstance(value, int) or not (0 <= value <= 24):
        return None
    if j >= len(words):
        return None
    nxt = words[j]

    # "... three pm" / "three am"
    if nxt in _MERIDIEM:
        if not _joinable(text, matches[j - 1].end(), matches[j].start()):
            return None
        return f"{value}:00 {_MERIDIEM[nxt]}", j + 1

    # "... three oh five" -> 3:05
    if nxt in ("oh", "o") and j + 1 < len(words) and words[j + 1] in _UNITS:
        if not _joinable(text, matches[j - 1].end(), matches[j].start()):
            return None
        if not _joinable(text, matches[j].end(), matches[j + 1].start()):
            return None
        minute = _UNITS[words[j + 1]]
        if minute > 9:
            return None
        end = j + 2
        meridiem = ""
        if end < len(words) and words[end] in _MERIDIEM:
            if _joinable(text, matches[end - 1].end(), matches[end].start()):
                meridiem = " " + _MERIDIEM[words[end]]
                end += 1
        return f"{value}:{minute:02d}{meridiem}", end

    # "... three thirty" -> 3:30
    phrase, k = _collect_phrase(matches, j, text)
    if not phrase or any(w in _SCALES for w in phrase):
        return None
    minute = _phrase_value(phrase)
    if minute > 59:
        return None
    # Only treat it as a time when the minute part is not itself a lone
    # ambiguous "one" (e.g. "three one" is not a time).
    if len(phrase) == 1 and phrase[0] == "one":
        return None
    if not _joinable(text, matches[j - 1].end(), matches[j].start()):
        return None
    meridiem = ""
    end = k
    if end < len(words) and words[end] in _MERIDIEM:
        if _joinable(text, matches[end - 1].end(), matches[end].start()):
            meridiem = " " + _MERIDIEM[words[end]]
            end += 1
    return f"{value}:{minute:02d}{meridiem}", end
