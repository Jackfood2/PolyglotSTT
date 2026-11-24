# gpu.py
import shutil
import subprocess
import threading
import time

try:
    from core.no_window import QUIET_FLAGS as _QUIET_FLAGS
except Exception:  # pragma: no cover - helper optional
    _QUIET_FLAGS = 0
_SMI_TIMEOUT = 8
_cache = {}
_cache_lock = threading.Lock()
_nvenc_cache = {}


def refresh_gpu_cache() -> None:
    """Invalidate GPU and NVENC probe caches."""
    with _cache_lock:
        _cache.clear()
        _nvenc_cache.clear()
def _smi_gpus():
    if shutil.which("nvidia-smi") is None:
        return []
    try:
        proc = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total,memory.free,"
             "driver_version",
             "--format=csv,noheader,nounits"],
            stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
            text=True, errors="replace", timeout=_SMI_TIMEOUT,
            creationflags=_QUIET_FLAGS)
    except Exception:
        return []
    out = []
    try:
        for line in (proc.stdout or "").splitlines():
            parts = [p.strip() for p in line.split(",")]
            if len(parts) < 4:
                continue
            try:
                total = int(float(parts[1]))
                free = int(float(parts[2]))
            except Exception:
                continue
            name = parts[0] or "NVIDIA GPU"
            if total < 1500:
                continue
            out.append({"name": name, "total_mb": total, "free_mb": free,
                        "driver": parts[3]})
    except Exception:
        return []
    out.sort(key=lambda g: g["free_mb"], reverse=True)
    return out
def best_gpu():
    now = time.monotonic()
    with _cache_lock:
        hit = _cache.get("smi")
        if hit and now - hit[0] < 60:
            return hit[1]
    gpus = _smi_gpus()
    best = gpus[0] if gpus else None
    with _cache_lock:
        _cache["smi"] = (now, best)
    return best
def torch_cuda():
    try:
        import torch
    except Exception as e:
        return {"ok": False, "reason": f"no torch: {e}"[:120]}
    try:
        ver = getattr(torch, "__version__", "?")
        if not torch.cuda.is_available():
            return {"ok": False, "torch": ver,
                    "reason": "torch has no CUDA (CPU build? run setup.bat)"}
        i = torch.cuda.current_device()
        props = torch.cuda.get_device_properties(i)
        return {"ok": True, "torch": ver, "name": props.name,
                "total_mb": int(props.total_memory // (1024 * 1024))}
    except Exception as e:
        return {"ok": False, "reason": str(e)[:120]}
# CUDA compute types ctranslate2 can offer, in preference order: fastest/most
# accurate first, cheapest last. float16 is NOT required: Pascal (compute
# capability 6.1, e.g. GP107/GTX 1050 Ti) has no fp16 at all and only exposes
# int8 / int8_float32 / float32. Those are still a large win over CPU, so the
# absence of fp16 must never be read as "no CUDA".
_CUDA_COMPUTE_ORDER = (
    "float16",
    "int8_float16",
    "int8_bfloat16",
    "int8_float32",
    "int8",
    "float32",
    "bfloat16",
)


def ct2_cuda():
    try:
        import ctranslate2
    except Exception as e:
        return {"ok": False, "reason": f"no ctranslate2: {e}"[:120]}
    try:
        types = list(ctranslate2.get_supported_compute_types("cuda") or [])
        usable = [t for t in _CUDA_COMPUTE_ORDER if t in types]
        if usable:
            return {"ok": True, "types": types, "usable": usable,
                    "fp16": "float16" in types}
        return {"ok": False,
                "reason": f"no usable CUDA compute types: {types}"[:120]}
    except Exception as e:
        return {"ok": False, "reason": str(e)[:160]}
_WHISPER_FP16_MB = {"tiny": 1200, "base": 1200, "small": 1500,
                    "medium": 2600, "large": 5200, "large-v1": 5200,
                    "large-v2": 5200, "large-v3": 5200,
                    "large-v3-turbo": 3600}
_WHISPER_INT8_MB = {"tiny": 700, "base": 700, "small": 900,
                    "medium": 1400, "large": 2800, "large-v1": 2800,
                    "large-v2": 2800, "large-v3": 2800,
                    "large-v3-turbo": 2000}
_CANARY_CUDA_MIN_MB = 10240
# int8 resident RAM per model, with headroom for MKL scratch buffers. Kept in
# sync with whisper_engine._MODEL_RAM_MB so the load-time budget guard and the
# GUI's "run both engines?" estimate agree.
_WHISPER_RESIDENT_MB = {
    "tiny": 900, "base": 950, "small": 1250,
    "medium": 1900, "large": 3400, "large-v1": 3400,
    "large-v2": 3400, "large-v3": 3400,
    "large-v3-turbo": 2000,
}


def whisper_resident_mb(model_id: str) -> int:
    """Estimated RAM a model holds while resident (int8 on CPU)."""
    return int(_WHISPER_RESIDENT_MB.get((model_id or "").strip(), 3400))
def whisper_cuda_vram_mb(model_id: str, compute: str) -> int:
    """Approximate VRAM a Whisper model needs on CUDA for a compute type."""
    mid = (model_id or "large-v3").strip() or "large-v3"
    need_int8 = _WHISPER_INT8_MB.get(mid, 2800)
    need_fp16 = _WHISPER_FP16_MB.get(mid, 5200)
    compute = str(compute or "").strip().lower()
    if compute in ("float16", "bfloat16"):
        return need_fp16
    if compute in ("int8_float16", "int8_bfloat16"):
        return int(round(need_int8 * 1.15))
    if compute == "int8_float32":
        return int(round(need_int8 * 1.25))
    if compute == "float32":
        return need_fp16 * 2
    return need_int8


def best_cuda_compute(model_id: str, ignore_vram: bool = False):
    """Fastest CUDA compute type ctranslate2 can actually run here.

    Returns (compute, reason). `compute` is None when CUDA is unusable.
    With ignore_vram=True the VRAM table is skipped (an explicit GPU request
    outranks the heuristic; the caller falls back to CPU on load failure).
    """
    cuda = ct2_cuda()
    if not cuda.get("ok"):
        return None, f"ctranslate2 has no CUDA ({cuda.get('reason', '?')})"
    usable = list(cuda.get("usable") or [])
    if not usable:
        return None, "ctranslate2 has no usable CUDA compute type"
    gpu = best_gpu()
    free = int((gpu or {}).get("free_mb") or 0)
    short = []
    for compute in usable:
        if ignore_vram or gpu is None:
            name = gpu["name"] if gpu else "CUDA GPU"
            return compute, f"{name} ({compute})"
        need = whisper_cuda_vram_mb(model_id, compute)
        if free < need:
            short.append(f"{compute} needs ~{need}MB")
            continue
        return compute, f"{gpu['name']} ({free}MB free, {compute})"
    return None, (f"only {free}MB VRAM free on "
                  f"{(gpu or {}).get('name', 'GPU')} "
                  f"({'; '.join(short)})")


def recommend_whisper(model_id: str):
    if best_gpu() is None:
        return "cpu", "int8", "no NVIDIA dGPU detected"
    compute, reason = best_cuda_compute(model_id)
    if compute:
        return "cuda", compute, reason
    return "cpu", "int8", reason
def recommend_canary():
    gpu = best_gpu()
    if gpu is None:
        return False, "no NVIDIA dGPU detected"
    tc = torch_cuda()
    if not tc.get("ok"):
        return False, f"torch CUDA unavailable ({tc.get('reason', '?')})"
    free = int(gpu.get("free_mb") or 0)
    if free >= _CANARY_CUDA_MIN_MB:
        return True, f"{gpu['name']} ({free}MB free)"
    return False, (f"only {free}MB VRAM free on {gpu.get('name', 'GPU')} "
                   f"(Canary wants ~{_CANARY_CUDA_MIN_MB}MB)")
def describe() -> str:
    try:
        gpu = best_gpu()
        if gpu is None:
            return "GPU: none detected (CPU pipeline)"
        tc = torch_cuda()
        ct = ct2_cuda()
        return (f"GPU: {gpu['name']} driver={gpu.get('driver', '?')} "
                f"VRAM {gpu['free_mb']}/{gpu['total_mb']}MB free | "
                f"torch-cuda={'yes' if tc.get('ok') else 'no'} | "
                f"ct2-cuda={'yes' if ct.get('ok') else 'no'}")
    except Exception as e:
        return f"GPU: probe failed ({e})"
def nvenc_available(ffmpeg=None, encoder: str = "h264_nvenc") -> bool:
    try:
        if best_gpu() is None:
            return False
    except Exception:
        return False
    exe = ffmpeg
    if not exe:
        try:
            from srt import get_ffmpeg_exe as _get_ffmpeg
            exe = _get_ffmpeg()
        except Exception:
            return False
    if not exe:
        return False
    try:
        enc = str(encoder or "h264_nvenc").strip().lower() or "h264_nvenc"
    except Exception:
        enc = "h264_nvenc"
    key = (str(exe), enc)
    with _cache_lock:
        if key in _nvenc_cache:
            return bool(_nvenc_cache[key])
    ok = False
    try:
        import re as _re
        proc = subprocess.run(
            [exe, "-hide_banner", "-encoders"],
            stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
            text=True, errors="replace", timeout=30,
            creationflags=_QUIET_FLAGS)
        out = proc.stdout or ""
        ok = bool(_re.search(r"\b" + _re.escape(enc) + r"\b", out))
    except Exception:
        ok = False
    with _cache_lock:
        _nvenc_cache[key] = ok
    return ok