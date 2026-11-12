"""Granite Speech 5.0 2B engine (Hugging Face Transformers).

IBM's Granite Speech 5.0 2B (``ibm-granite/granite-speech-5.0-2b``,
Apache-2.0) is the large sibling of the 470M TurboCTC model. Unlike the
encoder-only CTC model, this is a full speech-LLM (conformer encoder +
BLIP-2 Q-Former projector + Granite LLM), so it brings:

  * six spoken languages: English, French, German, Spanish, Portuguese,
    Japanese (auto-detect or pinned);
  * translation into English (and, upstream, en->it / en->zh - this app
    exposes English only, matching every other engine);
  * punctuation and capitalisation for free;
  * keyword biasing up to 2000 terms via the model-card prompt form
    ``Keywords: <term>, <term>`` plus free-text context;
  * no timestamps, so SRT generation runs it over VAD chunks exactly like
    Canary/Qwen3-ASR/Granite TurboCTC.

The model card recommends decoding segments of at most ~2 minutes, so
longer audio is split internally at low-energy boundaries before decode.

The weights can be a Transformers snapshot kept in ``models_cache/granite-2b``
or one of IBM's official GGUF quantizations (Q8_0 / Q6_K) run by a pinned
llama.cpp build, kept in ``models_cache/granite-2b-gguf``. Both paths stay
portable and fully offline once downloaded. There is no transcribe.cpp
port of the 5.0 2B model; the GGUF files are llama.cpp-format (text model
plus a separate audio projector), so this module drives ``llama-server``
over its local HTTP API for that backend.
"""

import atexit
import base64
import gc
import io
import json
import os
import shutil
import socket
import subprocess
import threading
import time
import wave
import zipfile
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np

PORTABLE_ROOT = Path(__file__).resolve().parent
GRANITE2B_CACHE = PORTABLE_ROOT / "models_cache" / "granite-2b"
GRANITE2B_GGUF_CACHE = PORTABLE_ROOT / "models_cache" / "granite-2b-gguf"
HF_CACHE = PORTABLE_ROOT / "models_cache" / "huggingface"

os.environ.setdefault("HF_HOME", str(HF_CACHE))
os.environ.setdefault("HUGGINGFACE_HUB_CACHE", str(HF_CACHE))

GRANITE2B_REPO = "ibm-granite/granite-speech-5.0-2b"
GRANITE2B_GGUF_REPO = "ibm-granite/granite-speech-5.0-2b-GGUF"
GRANITE2B_GGUF_MMPROJ = "mmproj-model-f16.gguf"

# The GGUF builds are llama.cpp-format (text model + audio projector), so
# they run on a pinned llama.cpp release instead of transformers. The CPU
# build is a 16 MB zip and is downloaded on first use of a GGUF model.
LLAMA_CPP_BUILD = "b9768"
LLAMA_CPP_DIR = PORTABLE_ROOT / "models_cache" / "llama.cpp" / LLAMA_CPP_BUILD
LLAMA_CPP_URL = (
    "https://github.com/ggml-org/llama.cpp/releases/download/"
    f"{LLAMA_CPP_BUILD}/llama-{LLAMA_CPP_BUILD}-bin-win-cpu-x64.zip"
)

# Label -> short id used in config files.
GRANITE2B_MODEL_CHOICES = {
    "Granite Speech 5.0 2B (4.8GB, 6 languages)":
        "granite-speech-5.0-2b",
    "Granite 2B Q8_0 GGUF (3.0GB, llama.cpp, fastest)":
        "granite-speech-5.0-2b-q8_0",
    "Granite 2B Q6_K GGUF (2.6GB, llama.cpp)":
        "granite-speech-5.0-2b-q6_k",
}

GRANITE2B_MODEL_CHOICES_REV = {
    v: k for k, v in GRANITE2B_MODEL_CHOICES.items()
}

# Kept for API symmetry with the other engines (the app reads this to
# validate a stored model id). Values are the main weight file name.
GRANITE2B_MODEL_FILES = {
    "granite-speech-5.0-2b": "model.safetensors",
    "granite-speech-5.0-2b-q8_0": "granite-speech-5.0-2b-Q8_0.gguf",
    "granite-speech-5.0-2b-q6_k": "granite-speech-5.0-2b-Q6_K.gguf",
}

GRANITE2B_GGUF_IDS = (
    "granite-speech-5.0-2b-q8_0",
    "granite-speech-5.0-2b-q6_k",
)

GRANITE2B_DEFAULT_MODEL = "granite-speech-5.0-2b"

# Resident RAM estimate (weights + mel/encoder/LLM scratch), in MB.
# model.safetensors is 4.4 GB and out_llm.safetensors 392 MB; the LLM
# context and activation buffers add roughly another 1-1.5 GB. The GGUF
# builds keep the audio projector at f16 (1.1 GB) plus the quantized LLM;
# llama.cpp reported a 2.4 GB projection for Q8_0 with a 4096 context.
GRANITE2B_RESIDENT_MB = {
    "granite-speech-5.0-2b": 6400,
    "granite-speech-5.0-2b-q8_0": 2900,
    "granite-speech-5.0-2b-q6_k": 2400,
}

# Six spoken languages plus auto-detect. Translation always targets English
# in this app, which every listed language supports.
GRANITE2B_LANGUAGES = ["auto", "en", "fr", "de", "es", "pt", "ja"]

GRANITE2B_TASKS = ["transcribe", "translate"]

# Human names used inside the instruction prompt. The model card's language
# ID prompt is "The speech is in <Language>. ...", with an English name.
_LANGUAGE_NAMES = {
    "en": "English",
    "fr": "French",
    "de": "German",
    "es": "Spanish",
    "pt": "Portuguese",
    "ja": "Japanese",
}

# Model card prompt forms. The keyword marker is what biases decoding; the
# sentence in front of it also drives punctuation, so we always ask for PnC.
_PNC_PROMPT = ("transcribe the speech with proper punctuation and "
               "capitalization.")

# Files that must exist before the snapshot counts as downloaded.
_REQUIRED_FILES = (
    "config.json",
    "generation_config.json",
    "preprocessor_config.json",
    "processor_config.json",
    "tokenizer_config.json",
    "tokenizer.json",
    "vocab.json",
    "merges.txt",
    "chat_template.jinja",
    "model.safetensors",
    "out_llm.safetensors",
)

_MIN_SNAPSHOT_BYTES = 3_000_000_000  # ~4.8 GB complete

# RMS below this is treated as silence and never decoded. The LLM head can
# invent a sentence on non-speech; the recorder's VAD does the real work and
# this is defence in depth (same floor as Granite TurboCTC).
_SILENCE_RMS = 2.0e-4

# The model card recommends segments of up to ~2 minutes.
_MAX_SEGMENT_S = 110.0
_SEGMENT_SEARCH_S = 8.0
_SEGMENT_FRAME_S = 0.10

_LOG_PREFIX = "[granite2b]"


def _log(message: str) -> None:
    try:
        print(f"{_LOG_PREFIX} {message}")
    except Exception:
        pass


# llama.cpp server processes started by this module. They are killed when
# the app exits so a crashed UI cannot leave a 2 GB process behind.
_LIVE_SERVERS: List["subprocess.Popen"] = []


def _kill_process(proc, wait: float = 5.0) -> None:
    try:
        if proc is None or proc.poll() is not None:
            return
        proc.terminate()
        try:
            proc.wait(timeout=wait)
        except Exception:
            proc.kill()
            try:
                proc.wait(timeout=2.0)
            except Exception:
                pass
    except Exception:
        pass
    finally:
        try:
            if proc in _LIVE_SERVERS:
                _LIVE_SERVERS.remove(proc)
        except Exception:
            pass


def _shutdown_servers() -> None:
    for proc in list(_LIVE_SERVERS):
        _kill_process(proc, wait=1.0)


atexit.register(_shutdown_servers)


def granite2b_model_id(model_id: Optional[str]) -> str:
    mid = str(model_id or "").strip()
    return mid if mid in GRANITE2B_MODEL_FILES else GRANITE2B_DEFAULT_MODEL


def granite2b_is_gguf(model_id: Optional[str]) -> bool:
    return granite2b_model_id(model_id) in GRANITE2B_GGUF_IDS


def granite2b_model_dir(model_id: Optional[str] = None) -> Path:
    """Folder holding this variant (one shared folder per backend)."""
    if granite2b_is_gguf(model_id):
        return GRANITE2B_GGUF_CACHE
    return GRANITE2B_CACHE


def granite2b_gguf_paths(model_id: Optional[str] = None) -> Tuple[Path, Path]:
    """(text model, audio projector) paths for a GGUF variant."""
    mid = granite2b_model_id(model_id)
    return (GRANITE2B_GGUF_CACHE / GRANITE2B_MODEL_FILES[mid],
            GRANITE2B_GGUF_CACHE / GRANITE2B_GGUF_MMPROJ)


def llama_cpp_binary(name: str = "llama-server.exe") -> Optional[Path]:
    """Pinned llama.cpp executable, or None when it is not downloaded."""
    path = LLAMA_CPP_DIR / name
    return path if _file_ok(path) else None


def _file_ok(path: Path, minimum: int = 1) -> bool:
    try:
        return path.is_file() and path.stat().st_size >= minimum
    except Exception:
        return False


def _snapshot_complete() -> bool:
    try:
        if not all(_file_ok(GRANITE2B_CACHE / name)
                   for name in _REQUIRED_FILES):
            return False
        total = sum(
            p.stat().st_size
            for p in GRANITE2B_CACHE.rglob("*")
            if p.is_file() and ".cache" not in p.parts
        )
        return total >= _MIN_SNAPSHOT_BYTES
    except Exception:
        return False


def _gguf_complete(model_id: str) -> bool:
    # Q8_0 is 1.96 GB and Q6_K 1.51 GB; anything much smaller is a partial
    # download (the projector is a separate, shared 1.1 GB file).
    minimum = 1_400_000_000
    try:
        model, projector = granite2b_gguf_paths(model_id)
        return _file_ok(model, minimum) and _file_ok(projector, 900_000_000)
    except Exception:
        return False


def granite2b_downloaded_map() -> Dict[str, bool]:
    return {
        mid: (_gguf_complete(mid) if mid in GRANITE2B_GGUF_IDS
              else _snapshot_complete())
        for mid in GRANITE2B_MODEL_FILES
    }


def _dir_bytes(path: Path) -> int:
    try:
        return sum(p.stat().st_size for p in path.rglob("*") if p.is_file())
    except Exception:
        return 0


def granite2b_cache_info() -> Dict[str, int]:
    """Bytes on disk per variant.

    The f16 audio projector is shared by both GGUF variants, so it is
    charged once: to the first variant present in Q8_0 -> Q6_K order (or
    to Q8_0 when none is downloaded). Without that, the Model manager
    total would count 1.1 GB twice.
    """
    hf_size = _dir_bytes(GRANITE2B_CACHE)
    projector = GRANITE2B_GGUF_CACHE / GRANITE2B_GGUF_MMPROJ
    projector_bytes = projector.stat().st_size if _file_ok(projector) else 0
    info: Dict[str, int] = {}
    present = []
    for mid in GRANITE2B_MODEL_FILES:
        if mid in GRANITE2B_GGUF_IDS:
            model = GRANITE2B_GGUF_CACHE / GRANITE2B_MODEL_FILES[mid]
            info[mid] = model.stat().st_size if _file_ok(model) else 0
            if info[mid]:
                present.append(mid)
        else:
            info[mid] = hf_size
    holder = present[0] if present else GRANITE2B_GGUF_IDS[0]
    info[holder] = info.get(holder, 0) + projector_bytes
    return info


def delete_granite2b_model(model_id: Optional[str] = None) -> int:
    """Delete one variant and return the number of bytes reclaimed.

    The GGUF projector and the llama.cpp runtime are shared, so they are
    removed only when the last downloaded GGUF variant goes away."""
    mid = granite2b_model_id(model_id)
    if mid in GRANITE2B_GGUF_IDS:
        model, projector = granite2b_gguf_paths(mid)
        if not _file_ok(model):
            raise FileNotFoundError("This Granite 2B GGUF is not downloaded")
        freed = model.stat().st_size
        try:
            model.unlink()
        except Exception as error:
            raise RuntimeError(f"Could not delete {model.name}: {error}")
        remaining = [p for m in GRANITE2B_GGUF_IDS if m != mid
                     for p in [GRANITE2B_GGUF_CACHE / GRANITE2B_MODEL_FILES[m]]
                     if _file_ok(p)]
        if not remaining:
            for path in (projector,):
                try:
                    if _file_ok(path):
                        freed += path.stat().st_size
                        path.unlink()
                except Exception:
                    pass
            try:
                if LLAMA_CPP_DIR.exists():
                    freed += _dir_bytes(LLAMA_CPP_DIR)
                    shutil.rmtree(LLAMA_CPP_DIR, ignore_errors=True)
            except Exception:
                pass
        return int(freed)

    if not GRANITE2B_CACHE.exists():
        raise FileNotFoundError("Granite 2B is not downloaded")
    freed = _dir_bytes(GRANITE2B_CACHE)
    shutil.rmtree(GRANITE2B_CACHE)
    return int(freed)


def _resident_model_mb(model_id: Optional[str]) -> int:
    return int(GRANITE2B_RESIDENT_MB.get(
        granite2b_model_id(model_id), 6400))


def _to_16k_mono(audio_data: np.ndarray, sample_rate: int) -> np.ndarray:
    """Float32 mono PCM at 16 kHz (the processor's fixed input rate)."""
    raw = np.asarray(audio_data)
    if raw.size == 0:
        return np.empty(0, dtype=np.float32)

    if raw.ndim == 2:
        arr = raw.mean(axis=1)
    else:
        arr = raw.flatten()

    if raw.dtype == np.int16:
        audio = arr.astype(np.float32) / 32768.0
    elif raw.dtype == np.int32:
        audio = arr.astype(np.float32) / 2147483648.0
    else:
        audio = arr.astype(np.float32)

    if audio.size and np.abs(audio).max() > 1.5:
        audio = audio / 32768.0

    np.nan_to_num(audio, copy=False, nan=0.0, posinf=1.0, neginf=-1.0)
    np.clip(audio, -1.0, 1.0, out=audio)

    try:
        rate = int(sample_rate)
    except Exception:
        rate = 16000
    if rate <= 0:
        rate = 16000

    if rate != 16000 and audio.size > 1:
        duration = audio.size / float(rate)
        target = max(1, int(round(duration * 16000)))
        source_index = np.arange(audio.size, dtype=np.float64)
        target_index = np.linspace(0.0, audio.size - 1.0, target)
        audio = np.interp(target_index, source_index, audio).astype(np.float32)

    return np.ascontiguousarray(audio, dtype=np.float32)


def _rms(pcm: np.ndarray) -> float:
    try:
        if pcm.size == 0:
            return 0.0
        return float(np.sqrt(np.mean(np.square(pcm, dtype=np.float64))))
    except Exception:
        return 0.0


def _split_pcm(pcm: np.ndarray, sample_rate: int = 16000,
               max_s: float = _MAX_SEGMENT_S) -> List[np.ndarray]:
    """Split long audio into <= max_s pieces, preferring low-energy cuts.

    The model card recommends <= ~2 minute segments. Cutting on digital
    silence avoids truncating a word at the boundary, which is much cheaper
    than asking the model to recover a half word.
    """
    try:
        total = int(pcm.size)
        limit = int(max(1.0, float(max_s)) * sample_rate)
        if total <= limit:
            return [pcm]

        frame = max(1, int(_SEGMENT_FRAME_S * sample_rate))
        search = int(_SEGMENT_SEARCH_S * sample_rate)
        pieces: List[np.ndarray] = []
        start = 0
        while start < total:
            remaining = total - start
            if remaining <= limit:
                pieces.append(pcm[start:])
                break
            target = start + limit
            lo = max(start + limit // 2, target - search)
            hi = min(total - frame, target)
            cut = target
            if hi > lo:
                # Pick the quietest frame in the window (cheap energy scan).
                best = None
                best_rms = None
                pos = lo
                while pos <= hi:
                    window = pcm[pos:pos + frame]
                    value = _rms(window)
                    if best_rms is None or value < best_rms:
                        best_rms, best = value, pos + frame // 2
                    pos += frame
                if best is not None:
                    cut = best
            # Never emit an empty piece or fail to advance.
            cut = min(total, max(start + frame, cut))
            pieces.append(pcm[start:cut])
            start = cut
        return pieces
    except Exception:
        return [pcm]


def _language_name(code: str) -> str:
    return _LANGUAGE_NAMES.get(str(code or "").strip().lower(), "English")


def _tidy_output(text: str) -> str:
    """Normalise raw decode text from either backend.

    Collapses whitespace and restores the leading capital. The model follows
    the last instruction, and when the instruction ends in a lowercase request
    ("translate the speech to English...") it sometimes mirrors that casing
    and starts the transcript lowercase. Capitalising the first letter here is
    deterministic and costs nothing, instead of depending on prompt wording
    that only holds for some audio.

    A trailing period is deliberately NOT forced: the model omits one when the
    audio is cut mid-sentence, and VAD chunks in SRT are mid-speech by design,
    so inventing a sentence end there would be wrong.
    """
    try:
        out = " ".join(str(text or "").split())
    except Exception:
        return ""
    if not out:
        return ""
    for index, char in enumerate(out):
        if char.isalpha():
            out = out[:index] + char.upper() + out[index + 1:]
            break
        if char.isdigit():
            break
    return out


def _estimate_tokens(text: str) -> int:
    """Cheap BPE estimate (~4 chars per token) for prompt budgeting."""
    s = str(text or "")
    if not s.strip():
        return 0
    return max(1, (len(s) + 3) // 4)


def _select_dtype(torch_module, device: str):
    """Weights dtype for this device, honouring GRANITE2B_DTYPE."""
    table = {
        "float32": torch_module.float32,
        "fp32": torch_module.float32,
        "bfloat16": torch_module.bfloat16,
        "bf16": torch_module.bfloat16,
        "float16": torch_module.float16,
        "fp16": torch_module.float16,
    }
    try:
        override = os.environ.get("GRANITE2B_DTYPE", "").strip().lower()
    except Exception:
        override = ""
    if override in table:
        return table[override]
    return torch_module.bfloat16 if str(device) == "cuda" else torch_module.float32


def build_instruction(task: str, source_lang: str, target_lang: str,
                      keywords: Optional[List[str]] = None,
                      context: Optional[str] = None) -> str:
    """Compose the instruction text from the model card's prompt recipes.

    The audio token is prepended by the caller; this function returns only
    the natural-language instruction.

    Every branch keeps the "proper punctuation and capitalization" request:
    the card's raw keyword template ("transcribe the speech to text.
    Keywords: ...") deliberately produces unpunctuated text, which then
    needed a second pass to be usable in subtitles and notes.
    """
    keywords = [str(k).strip() for k in (keywords or []) if str(k).strip()]
    context_text = str(context or "").strip()
    task = str(task or "transcribe").strip().lower()
    source_lang = str(source_lang or "auto").strip().lower()
    target_name = _language_name(target_lang or "en")

    if task == "translate":
        lang_prefix = ""
        base = (f"translate the speech to {target_name} with proper "
                "punctuation and capitalization.")
    elif source_lang not in ("", "auto"):
        # Card form: "The speech is in <language>. Transcribe it exactly; do
        # not translate." followed by the punctuation request.
        lang_prefix = (f"The speech is in {_language_name(source_lang)}. "
                       "Transcribe it exactly; do not translate. ")
        base = _PNC_PROMPT
    else:
        lang_prefix = ""
        base = _PNC_PROMPT

    # Prompt order is load-bearing. The model follows the LAST instruction,
    # so the punctuation request must come after "Keywords: ...". The
    # language sentence must come after the keywords too: language-first
    # plus a keyword list made the model drop all punctuation in testing,
    # while keywords -> language -> punctuation keeps it.
    #
    # Free-text context goes in the same "before the final request" slot.
    # Appending it AFTER the punctuation request made the model return
    # all-lowercase text and lowercase the keyword it had just been given,
    # so context is placed between keywords and the language/punctuation
    # instruction instead.
    text = ""
    if keywords:
        text += "Keywords: " + ", ".join(keywords) + ". "
    if context_text:
        # Context usually arrives already punctuated; add a sentence break
        # only when it is missing so the prompt never reads "app..".
        if context_text[-1] not in ".!?;:":
            context_text += "."
        text += "Context: " + context_text + " "
    text += lang_prefix + base
    return text


class Granite2BEngine:
    """Drop-in sibling of QwenEngine/GraniteEngine for Granite Speech 5.0 2B.

    Public surface used by the app: load(), is_ready, _loading, _last_error,
    unload(), switch_options(), switch_model(), transcribe(),
    transcribe_file(), transcribe_file_segments(), device, device_info,
    current_arch_name, supported_source_langs, supports_translate,
    supports_timestamps.
    """

    _load_gate = threading.Lock()

    def __init__(
        self,
        task: str = "transcribe",
        source_lang: str = "auto",
        target_lang: str = "en",
        model_id: str = GRANITE2B_DEFAULT_MODEL,
        device: str = "auto",
        on_ready: Optional[Callable] = None,
    ):
        self.task = task if task in GRANITE2B_TASKS else "transcribe"
        self.source_lang = (
            source_lang if source_lang in GRANITE2B_LANGUAGES else "auto"
        )
        # The app translates to English only; the model also supports
        # en->it/zh, but that is intentionally not exposed.
        self.target_lang = "en"
        self.model_id = granite2b_model_id(model_id)
        self.device = (device or "auto").strip().lower() or "auto"
        if self.device == "gpu":
            self.device = "cuda"
        if self.device not in ("auto", "cpu", "cuda"):
            self.device = "auto"

        self._model = None
        self._processor = None
        self._torch = None
        self._torch_dtype = None
        self._ready = False
        self._loading = False
        self._last_error: Optional[str] = None
        self._load_generation = 0
        self._device_used = "cpu"
        self._on_ready = on_ready
        self._lock = threading.Lock()
        self._infer_lock = threading.Lock()
        self._load_lock = threading.Lock()
        self.supported_source_langs = set(GRANITE2B_LANGUAGES) - {"auto"}
        self.supports_translate = True
        self.supports_timestamps = False
        # Optional user dictionary (keyword biasing + context prompt).
        self._dict_terms: List[str] = []
        self._dict_context: str = ""
        self._dict_enabled = False
        self._silence_skips = 0
        # GGUF backend (pinned llama.cpp server) state.
        self._server_proc = None
        self._server_port = 0
        self._server_log_path = None

    # ------------------------------------------------------------------ state
    @property
    def backend(self) -> str:
        return "gguf" if self.model_id in GRANITE2B_GGUF_IDS else "transformers"

    @property
    def is_ready(self) -> bool:
        with self._lock:
            if not self._ready:
                return False
            if self.model_id in GRANITE2B_GGUF_IDS:
                proc = self._server_proc
                return (proc is not None and proc.poll() is None
                        and bool(self._server_port))
            return self._model is not None and self._processor is not None

    @property
    def is_loading(self) -> bool:
        with self._lock:
            return self._loading

    @property
    def last_error(self) -> Optional[str]:
        with self._lock:
            return self._last_error

    @property
    def device_info(self) -> str:
        with self._lock:
            return self._device_used or "cpu"

    @property
    def current_arch_name(self) -> str:
        with self._lock:
            if self.model_id in GRANITE2B_GGUF_IDS:
                return f"{self.model_id} (llama.cpp)"
            return f"granite-speech-5.0-2b ({self._device_used})"

    # -------------------------------------------------------------- model files
    def _ensure_dirs(self) -> None:
        for path in (GRANITE2B_CACHE, HF_CACHE):
            try:
                path.mkdir(parents=True, exist_ok=True)
            except Exception:
                pass

    def _resolve_model_path(
        self,
        generation: int,
        on_progress: Optional[Callable],
    ) -> str:
        self._ensure_dirs()
        if _snapshot_complete():
            return str(GRANITE2B_CACHE)

        if not self._is_current(generation):
            raise RuntimeError("superseded")

        offline = os.environ.get("HF_HUB_OFFLINE", "").strip().upper() in {
            "1", "TRUE", "YES", "ON",
        }
        if offline:
            raise FileNotFoundError(
                "Granite Speech 5.0 2B is not fully downloaded in "
                "models_cache/granite-2b and HF_HUB_OFFLINE is set"
            )

        try:
            from huggingface_hub import snapshot_download
        except Exception as error:
            raise RuntimeError(
                f"huggingface_hub is required to download Granite 2B: {error}"
            )

        self._report(
            on_progress,
            "Downloading Granite Speech 5.0 2B (about 4.8 GB, one time)",
        )
        _log(f"downloading {GRANITE2B_REPO} -> {GRANITE2B_CACHE}")
        snapshot_download(
            repo_id=GRANITE2B_REPO,
            local_dir=str(GRANITE2B_CACHE),
        )
        if not self._is_current(generation):
            raise RuntimeError("superseded")
        if not _snapshot_complete():
            raise RuntimeError(
                "Granite 2B download finished but the snapshot is incomplete; "
                "delete models_cache/granite-2b and retry"
            )
        return str(GRANITE2B_CACHE)

    # ----------------------------------------------------------- gguf backend
    @staticmethod
    def _pick_port() -> int:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        try:
            sock.bind(("127.0.0.1", 0))
            return int(sock.getsockname()[1])
        finally:
            try:
                sock.close()
            except Exception:
                pass

    def _stop_server(self) -> None:
        with self._lock:
            proc = self._server_proc
            self._server_proc = None
            self._server_port = 0
        _kill_process(proc)
        if proc is not None:
            _log("llama.cpp server stopped")

    @staticmethod
    def _download_binary(url: str, dest: Path,
                         on_progress: Optional[Callable],
                         label: str) -> None:
        import urllib.request

        dest.parent.mkdir(parents=True, exist_ok=True)
        part = dest.with_name(dest.name + ".part")
        request = urllib.request.Request(url, headers={"User-Agent": "PolyglotSTT"})
        with urllib.request.urlopen(request, timeout=120) as response:
            total = int(response.headers.get("Content-Length") or 0)
            done = 0
            with open(part, "wb") as handle:
                while True:
                    chunk = response.read(1024 * 512)
                    if not chunk:
                        break
                    handle.write(chunk)
                    done += len(chunk)
                    if total and on_progress is not None:
                        percent = int(done * 100 / total)
                        if percent % 10 == 0:
                            try:
                                on_progress(f"Downloading {label} "
                                            f"({percent}%)")
                            except Exception:
                                pass
        part.replace(dest)

    def _ensure_llama_runtime(self, on_progress: Optional[Callable]) -> Path:
        exe = llama_cpp_binary("llama-server.exe")
        if exe is not None:
            return exe
        if os.name != "nt":
            raise RuntimeError(
                "The GGUF builds run on llama.cpp; automatic download is "
                "Windows-only. Install llama-server and set GRANITE2B_LLAMA_DIR."
            )
        self._report(
            on_progress,
            f"Downloading llama.cpp {LLAMA_CPP_BUILD} runtime (16 MB, one time)",
        )
        zip_path = LLAMA_CPP_DIR.with_suffix(".zip")
        self._download_binary(LLAMA_CPP_URL, zip_path, on_progress,
                              "llama.cpp runtime")
        try:
            with zipfile.ZipFile(zip_path) as archive:
                archive.extractall(LLAMA_CPP_DIR)
        finally:
            try:
                zip_path.unlink()
            except Exception:
                pass
        exe = llama_cpp_binary("llama-server.exe")
        if exe is None:
            raise RuntimeError(
                "llama.cpp download finished but llama-server.exe is missing"
            )
        _log(f"llama.cpp runtime ready: {LLAMA_CPP_DIR}")
        return exe

    def _download_gguf(self, model_id: str, generation: int,
                       on_progress: Optional[Callable]) -> Tuple[Path, Path]:
        model_path, mmproj_path = granite2b_gguf_paths(model_id)
        if _gguf_complete(model_id):
            return model_path, mmproj_path

        offline = os.environ.get("HF_HUB_OFFLINE", "").strip().upper() in {
            "1", "TRUE", "YES", "ON",
        }
        if offline:
            raise FileNotFoundError(
                "This Granite 2B GGUF is not fully downloaded and "
                "HF_HUB_OFFLINE is set"
            )
        try:
            from huggingface_hub import hf_hub_download
        except Exception as error:
            raise RuntimeError(
                f"huggingface_hub is required to download the GGUF: {error}"
            )

        self._report(
            on_progress,
            "Downloading Granite 2B GGUF (about 2.7 GB, one time)",
        )
        GRANITE2B_GGUF_CACHE.mkdir(parents=True, exist_ok=True)
        for name in (GRANITE2B_MODEL_FILES[model_id],
                     GRANITE2B_GGUF_MMPROJ):
            if not self._is_current(generation):
                raise RuntimeError("superseded")
            _log(f"downloading {GRANITE2B_GGUF_REPO}/{name}")
            hf_hub_download(
                repo_id=GRANITE2B_GGUF_REPO,
                filename=name,
                local_dir=str(GRANITE2B_GGUF_CACHE),
            )
        if not _gguf_complete(model_id):
            raise RuntimeError(
                "Granite 2B GGUF download finished but the files are "
                "incomplete; delete models_cache/granite-2b-gguf and retry"
            )
        return model_path, mmproj_path

    def _wait_health(self, port: int, proc, timeout: float = 300.0) -> bool:
        import urllib.error
        import urllib.request

        deadline = time.time() + timeout
        url = f"http://127.0.0.1:{port}/health"
        while time.time() < deadline:
            if proc.poll() is not None:
                return False
            try:
                with urllib.request.urlopen(url, timeout=3) as response:
                    body = json.loads(response.read().decode("utf-8", "replace"))
                if str(body.get("status", "")).lower() == "ok":
                    return True
            except Exception:
                pass
            time.sleep(0.4)
        return False

    def _log_tail(self, lines: int = 12) -> str:
        try:
            path = self._server_log_path
            if path is None or not Path(path).is_file():
                return ""
            data = Path(path).read_text(encoding="utf-8", errors="replace")
            return "\n".join(data.strip().splitlines()[-lines:])
        except Exception:
            return ""

    def _load_gguf(self, generation: int, model_id: str,
                   on_progress: Optional[Callable]) -> None:
        self._stop_server()
        exe = self._ensure_llama_runtime(on_progress)
        if not self._is_current(generation):
            return
        model_path, mmproj_path = self._download_gguf(
            model_id, generation, on_progress)
        if not self._is_current(generation):
            return

        self._report(on_progress, "Loading Granite 2B (llama.cpp)")
        port = self._pick_port()
        GRANITE2B_GGUF_CACHE.mkdir(parents=True, exist_ok=True)
        log_path = GRANITE2B_GGUF_CACHE / "llama-server.log"
        try:
            if log_path.is_file() and log_path.stat().st_size > 8_000_000:
                log_path.unlink()
        except Exception:
            pass
        threads = max(2, min(8, (os.cpu_count() or 4) // 2))
        command = [
            str(exe),
            "-m", str(model_path),
            "--mmproj", str(mmproj_path),
            "--jinja",
            "-c", "4096",
            "--host", "127.0.0.1",
            "--port", str(port),
            "--no-warmup",
            "-t", str(threads),
        ]
        creationflags = 0
        if os.name == "nt":
            creationflags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
        _log(f"starting llama-server on port {port} ({threads} threads)")
        log_handle = open(log_path, "ab", buffering=0)
        try:
            proc = subprocess.Popen(
                command,
                stdin=subprocess.DEVNULL,
                stdout=log_handle,
                stderr=subprocess.STDOUT,
                creationflags=creationflags,
            )
        finally:
            try:
                log_handle.close()
            except Exception:
                pass

        self._server_log_path = str(log_path)
        _LIVE_SERVERS.append(proc)
        if not self._wait_health(port, proc):
            _kill_process(proc)
            tail = self._log_tail()
            raise RuntimeError(
                "llama.cpp server did not start"
                + (f":\n{tail}" if tail else "")
            )
        if not self._is_current(generation):
            _kill_process(proc)
            return

        with self._lock:
            self._server_proc = proc
            self._server_port = port
            self._ready = True
            self._loading = False
            self._last_error = None
            self._device_used = "cpu"
        _log(f"ready: {model_id} on llama.cpp (port {port})")
        self._report(on_progress, "Model ready")
        callback = self._on_ready
        if callback is not None:
            try:
                callback(True, None)
            except Exception:
                pass

    # ------------------------------------------------------------------ load
    def _is_current(self, generation: int) -> bool:
        with self._lock:
            return generation == self._load_generation

    def _report(self, callback: Optional[Callable], message: str) -> None:
        if callback is None:
            return
        try:
            callback(message)
        except Exception:
            pass

    def _target_device(self, torch_module) -> str:
        if self.device == "cpu":
            return "cpu"
        if self.device == "cuda":
            return "cuda" if torch_module.cuda.is_available() else "cpu"
        try:
            return "cuda" if torch_module.cuda.is_available() else "cpu"
        except Exception:
            return "cpu"

    def load(self, on_progress: Optional[Callable] = None) -> None:
        with self._lock:
            self._load_generation += 1
            generation = self._load_generation
            self._loading = True
            self._ready = False
            self._last_error = None
            model_id = self.model_id

        threading.Thread(
            target=self._load_worker,
            args=(generation, model_id, on_progress),
            name=f"Granite2BLoader-{generation}",
            daemon=True,
        ).start()

    def _load_worker(
        self,
        generation: int,
        model_id: str,
        on_progress: Optional[Callable],
    ) -> None:
        model = None
        processor = None
        installed = False

        try:
            with Granite2BEngine._load_gate:
                if not self._is_current(generation):
                    return

                with self._infer_lock:
                    with self._lock:
                        if generation != self._load_generation:
                            return
                        old_model = self._model
                        old_processor = self._processor
                        self._model = None
                        self._processor = None
                        self._ready = False
                    self._close(old_model, old_processor)
                    # A previous GGUF selection keeps a llama.cpp server
                    # running; it must die before another model loads.
                    self._stop_server()

                if model_id in GRANITE2B_GGUF_IDS:
                    self._load_gguf(generation, model_id, on_progress)
                    return

                path = self._resolve_model_path(generation, on_progress)
                if not self._is_current(generation):
                    return

                self._report(on_progress, "Loading Granite 2B (Transformers)")
                try:
                    import torch
                except Exception as error:
                    raise RuntimeError(
                        "torch is required for Granite 2B: "
                        f"{error}. Run setup.bat to install the runtime."
                    )
                try:
                    from transformers import (AutoModelForSpeechSeq2Seq,
                                              AutoProcessor)
                except Exception as error:
                    raise RuntimeError(
                        "transformers is required for Granite 2B: "
                        f"{error}. Run setup.bat to install the runtime."
                    )

                device = self._target_device(torch)
                # Consumer Intel/AMD CPUs have no fast bf16 matmul, so
                # bfloat16 there is emulated and measured ~3x slower than
                # float32 in this project's benchmark. CUDA keeps bf16 (half
                # the weights, native tensor cores). GRANITE2B_DTYPE overrides
                # both for experiments.
                dtype = _select_dtype(torch, device)

                processor = AutoProcessor.from_pretrained(path)
                try:
                    model = AutoModelForSpeechSeq2Seq.from_pretrained(
                        path, dtype=dtype)
                except TypeError:
                    model = AutoModelForSpeechSeq2Seq.from_pretrained(
                        path, torch_dtype=dtype)
                except Exception as error:
                    message = str(error).lower()
                    if "dtype" not in message and "bfloat16" not in message \
                            and "slow and fast" not in message:
                        raise
                    _log(f"bf16 load failed ({error}); retrying float32")
                    model = AutoModelForSpeechSeq2Seq.from_pretrained(
                        path, dtype=torch.float32)
                    dtype = torch.float32

                try:
                    model.to(device)
                    model.eval()
                except Exception:
                    # Float32 is the safe CPU fallback for exotic builds.
                    if dtype is not torch.float32:
                        del model
                        gc.collect()
                        model = AutoModelForSpeechSeq2Seq.from_pretrained(
                            path, dtype=torch.float32)
                        model.to(device)
                        model.eval()
                        dtype = torch.float32
                    else:
                        raise

                if not self._is_current(generation):
                    self._close(model, processor)
                    return

                with self._lock:
                    if generation != self._load_generation:
                        self._close(model, processor)
                        return
                    self._model = model
                    self._processor = processor
                    self._torch = torch
                    self._torch_dtype = dtype
                    self._ready = True
                    self._loading = False
                    self._last_error = None
                    self._device_used = device
                    model = None
                    processor = None
                    installed = True

                _log(
                    "ready: %s on %s (%s)"
                    % (self.model_id, device,
                       str(dtype).replace("torch.", ""))
                )
                self._report(on_progress, "Model ready")

                callback = self._on_ready
                if callback is not None:
                    try:
                        callback(True, None)
                    except Exception:
                        pass

        except Exception as error:
            message = str(error)
            _log(f"load failed: {message}")
            self._close(model, processor)
            with self._lock:
                if generation == self._load_generation:
                    self._ready = False
                    self._loading = False
                    self._model = None
                    self._processor = None
                    self._last_error = message
            callback = self._on_ready
            if callback is not None:
                try:
                    callback(False, message)
                except Exception:
                    pass
        finally:
            if not installed:
                with self._lock:
                    if generation == self._load_generation:
                        self._loading = False

    @staticmethod
    def _close(model, processor) -> None:
        try:
            if model is not None:
                del model
        except Exception:
            pass
        try:
            if processor is not None:
                del processor
        except Exception:
            pass

    def unload(self) -> bool:
        with self._infer_lock:
            with self._lock:
                had = (self._model is not None or self._processor is not None
                       or self._server_proc is not None or self._loading)
                self._load_generation += 1
                model = self._model
                processor = self._processor
                torch_module = self._torch
                self._model = None
                self._processor = None
                self._ready = False
                self._loading = False
                self._last_error = None
            self._stop_server()
            self._close(model, processor)
            try:
                if torch_module is not None:
                    if torch_module.cuda.is_available():
                        torch_module.cuda.empty_cache()
            except Exception:
                pass

        try:
            gc.collect()
        except Exception:
            pass
        return bool(had)

    def switch_model(
        self,
        model_id: str,
        on_ready: Optional[Callable] = None,
    ) -> None:
        mid = granite2b_model_id(model_id)
        with self._lock:
            self.model_id = mid
        if on_ready is not None:
            self._on_ready = on_ready
        self.load()

    def switch_options(
        self,
        task: Optional[str] = None,
        source_lang: Optional[str] = None,
        target_lang: Optional[str] = None,
        on_ready: Optional[Callable] = None,
    ) -> None:
        with self._lock:
            if task in GRANITE2B_TASKS:
                self.task = task
            if source_lang:
                self.source_lang = (
                    source_lang if source_lang in GRANITE2B_LANGUAGES
                    else "auto"
                )
            self.target_lang = "en"
        if on_ready is not None:
            try:
                on_ready(True, None)
            except Exception:
                pass

    def _snapshot_opts(self) -> Tuple[str, str, str]:
        with self._lock:
            return self.task, self.source_lang, self.target_lang

    def _check_lang_support(
        self,
        source_lang: Optional[str],
    ) -> Optional[str]:
        supported = self.supported_source_langs or set()
        src = (source_lang or "auto").strip().lower()
        if src == "auto" or not supported or src in supported:
            return None
        return (
            "Granite 2B does not list language '%s'. It supports: %s. "
            "Use auto-detect or pick a listed language."
            % (src, ", ".join(sorted(supported)))
        )

    # ------------------------------------------------------------- inference
    # Granite 2B's context is 4096 tokens, prompt AND reply together. The
    # model card puts two minutes of audio at ~1200 tokens, so a 2000-term
    # keyword list cannot physically fit next to a long clip; the list is
    # trimmed to what the context can hold instead of letting generate() run
    # off the end of the position table.
    _CONTEXT_TOKENS = 4096
    _AUDIO_TOKENS_PER_S = 10.0

    def _instruction_budget(self, seconds: float) -> int:
        try:
            audio = int(max(0.0, float(seconds)) * self._AUDIO_TOKENS_PER_S)
        except Exception:
            audio = 0
        reserve = self._max_new_tokens(seconds)
        return max(192, self._CONTEXT_TOKENS - audio - reserve - 32)

    def _instruction(self, task: str, language: str,
                     extra_prompt: Optional[str],
                     seconds: float = 0.0) -> str:
        terms, context = self.dictionary_payload()
        if extra_prompt:
            extra = str(extra_prompt).strip()
            if extra:
                context = (context + " " + extra).strip() if context else extra

        budget = self._instruction_budget(seconds)

        if context and _estimate_tokens(context) > max(64, budget // 2):
            keep_chars = max(160, (max(64, budget // 2)) * 4)
            context = context[:keep_chars].rstrip() + " …"

        if not terms:
            return build_instruction(task, language, self.target_lang,
                                     keywords=None, context=context)

        # Longest keyword prefix that fits. Monotonic in length, so binary
        # search instead of rebuilding the prompt once per term.
        lo, hi = 0, len(terms)
        while lo < hi:
            mid = (lo + hi + 1) // 2
            probe = build_instruction(task, language, self.target_lang,
                                      keywords=terms[:mid], context=context)
            if _estimate_tokens(probe) <= budget:
                lo = mid
            else:
                hi = mid - 1
        kept = terms[:lo]
        if len(kept) < len(terms):
            _log("context budget: kept %d of %d keywords for %.0fs audio"
                 % (len(kept), len(terms), seconds))
        return build_instruction(task, language, self.target_lang,
                                 keywords=kept or None, context=context)

    def _max_new_tokens(self, seconds: float) -> int:
        # ~3.5 output tokens/s of speech + instruction headroom; capped so a
        # model that starts repeating cannot run away for minutes.
        try:
            value = int(float(seconds) * 6.0) + 48
        except Exception:
            value = 256
        return max(96, min(1024, value))

    @staticmethod
    def _wav_bytes(pcm: np.ndarray) -> bytes:
        pcm16 = np.clip(np.asarray(pcm, dtype=np.float32), -1.0, 1.0)
        pcm16 = (pcm16 * 32767.0).astype(np.int16)
        buffer = io.BytesIO()
        with wave.open(buffer, "wb") as handle:
            handle.setnchannels(1)
            handle.setsampwidth(2)
            handle.setframerate(16000)
            handle.writeframes(pcm16.tobytes())
        return buffer.getvalue()

    def _decode_gguf(self, instruction: str, pcm: np.ndarray,
                     seconds: float) -> str:
        import urllib.request

        with self._lock:
            port = self._server_port
        if not port:
            return ""
        payload = {
            "model": "granite",
            "temperature": 0,
            "max_tokens": self._max_new_tokens(seconds),
            "messages": [{
                "role": "user",
                "content": [
                    {"type": "text", "text": instruction},
                    {"type": "input_audio", "input_audio": {
                        "data": base64.b64encode(
                            self._wav_bytes(pcm)).decode("ascii"),
                        "format": "wav",
                    }},
                ],
            }],
        }
        request = urllib.request.Request(
            f"http://127.0.0.1:{port}/v1/chat/completions",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(request, timeout=600) as response:
            raw = response.read().decode("utf-8", "replace")
        try:
            body = json.loads(raw)
        except Exception:
            _log(f"llama.cpp returned non-JSON: {raw[:200]!r}")
            return ""
        try:
            return _tidy_output(body["choices"][0]["message"]["content"])
        except Exception:
            return ""

    def _decode_segment(self, pcm: np.ndarray, task: str, language: str,
                        extra_prompt: Optional[str]) -> str:
        seconds = float(pcm.size) / 16000.0
        instruction = self._instruction(task, language, extra_prompt, seconds)
        if self.model_id in GRANITE2B_GGUF_IDS:
            return self._decode_gguf(instruction, pcm, seconds)

        with self._lock:
            model = self._model
            processor = self._processor
            torch_module = self._torch
            device = self._device_used
        if model is None or processor is None or torch_module is None:
            return ""

        prompt = "<|audio|>" + instruction
        try:
            chat = [{"role": "user", "content": prompt}]
            text = processor.tokenizer.apply_chat_template(
                chat, tokenize=False, add_generation_prompt=True)
        except Exception:
            text = prompt

        wav = np.ascontiguousarray(pcm, dtype=np.float32)
        # The Granite feature extractor is fixed at 16 kHz and takes no
        # sampling_rate argument; the PCM is already resampled above.
        inputs = processor(text, wav, device=device, return_tensors="pt")
        inputs = inputs.to(device)
        input_len = int(inputs["input_ids"].shape[-1])

        with torch_module.inference_mode():
            outputs = model.generate(
                **inputs,
                max_new_tokens=self._max_new_tokens(seconds),
                do_sample=False,
                num_beams=1,
            )
        new_tokens = outputs[0, input_len:]
        decoded = processor.tokenizer.batch_decode(
            [new_tokens],
            add_special_tokens=False,
            skip_special_tokens=True,
        )
        text_out = decoded[0] if decoded else ""
        del inputs, outputs, new_tokens
        return _tidy_output(text_out)

    def _run_pcm(self, pcm: np.ndarray, language: Optional[str],
                 task: Optional[str] = None,
                 extra_prompt: Optional[str] = None) -> str:
        with self._infer_lock:
            with self._lock:
                if not self._ready:
                    return ""
                if (self.model_id not in GRANITE2B_GGUF_IDS
                        and self._model is None):
                    return ""

            if _rms(pcm) < _SILENCE_RMS:
                with self._lock:
                    self._silence_skips += 1
                return ""

            snap_task, _, _ = self._snapshot_opts()
            use_task = snap_task if task is None else str(task).lower()
            if use_task not in GRANITE2B_TASKS:
                use_task = "transcribe"
            lang = str(language or "auto").strip().lower()

            pieces = _split_pcm(pcm, 16000, _MAX_SEGMENT_S)
            parts: List[str] = []
            for index, piece in enumerate(pieces):
                if piece.size < 1600:
                    continue
                if len(pieces) > 1:
                    _log(f"segment {index + 1}/{len(pieces)} "
                         f"({piece.size / 16000.0:.0f}s)")
                text = self._decode_segment(piece, use_task, lang,
                                            extra_prompt)
                if text:
                    parts.append(text)
            return " ".join(parts).strip()

    # -------------------------------------------------------------- dictionary
    def set_dictionary(
        self,
        terms: Optional[List[str]] = None,
        context: Optional[str] = None,
        enabled: bool = True,
    ) -> None:
        try:
            import dictionary as _dict
        except Exception:
            _dict = None
        try:
            clean = (_dict.norm_terms(terms) if _dict is not None
                     else [str(t).strip() for t in (terms or []) if str(t).strip()])
            ctx = (_dict.norm_context(context) if _dict is not None
                   else str(context or "").strip())
        except Exception:
            clean, ctx = [], ""
        with self._lock:
            self._dict_terms = list(clean)
            self._dict_context = ctx
            self._dict_enabled = bool(enabled)

    def dictionary_payload(self) -> Tuple[Optional[List[str]], Optional[str]]:
        """(keywords, context prompt) for the current decode, or (None, None)."""
        with self._lock:
            if not self._dict_enabled:
                return None, None
            terms = list(self._dict_terms)
            context = self._dict_context
        if not terms and not context:
            return None, None
        try:
            import dictionary as _dict

            info = _dict.build("Granite 2B", terms, context)
            if not info.get("eligible"):
                return (list(terms) or None), (context or None)
            return info.get("keywords"), (info.get("prompt") or None)
        except Exception:
            return (list(terms) or None), (context or None)

    # ------------------------------------------------------------- transcribe
    def transcribe(
        self,
        audio_data: np.ndarray,
        sample_rate: int = 16000,
        task: Optional[str] = None,
        source_lang: Optional[str] = None,
        extra_prompt: Optional[str] = None,
    ) -> str:
        if not self.is_ready:
            return ""
        try:
            pcm = _to_16k_mono(audio_data, sample_rate)
            if pcm.size < 1600:  # under 0.1 s: nothing to decode
                return ""
            snap_task, snap_src, _ = self._snapshot_opts()
            use_task = snap_task if task is None else str(task).lower()
            lang = source_lang if source_lang is not None else snap_src
            error = self._check_lang_support(lang)
            if error:
                return f"[Granite 2B Error: {error}]"
            return self._run_pcm(pcm, lang, use_task, extra_prompt)
        except Exception as error:
            import traceback

            traceback.print_exc()
            return f"[Granite 2B Error: {error}]"

    def transcribe_file(
        self,
        wav_path: str,
        task: Optional[str] = None,
        source_lang: Optional[str] = None,
        target_lang: Optional[str] = None,
    ) -> str:
        if not self.is_ready:
            return ""
        try:
            import soundfile as sf

            audio, rate = sf.read(str(wav_path), dtype="float32",
                                  always_2d=False)
            snap_task, snap_src, _ = self._snapshot_opts()
            use_task = snap_task if task is None else str(task).lower()
            lang = source_lang if source_lang is not None else snap_src
            error = self._check_lang_support(lang)
            if error:
                return f"[Granite 2B Error: {error}]"
            pcm = _to_16k_mono(audio, rate)
            if pcm.size < 1600:
                return ""
            return self._run_pcm(pcm, lang, use_task)
        except Exception as error:
            import traceback

            traceback.print_exc()
            return f"[Granite 2B Error: {error}]"

    def transcribe_file_segments(
        self, wav_path: str
    ) -> List[Tuple[float, float, str]]:
        """Single-pass transcription with a whole-file cue.

        Granite 2B reports no timestamps, so callers that need cues should
        chunk with VAD (SRT does) rather than rely on this."""
        text = self.transcribe_file(wav_path)
        if not text or text.startswith("[Granite 2B Error"):
            return []
        try:
            import soundfile as sf

            info = sf.info(str(wav_path))
            duration = float(info.frames) / float(info.samplerate or 16000)
        except Exception:
            duration = 0.0
        return [(0.0, max(duration, 0.1), text)]
