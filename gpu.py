"""Dedicated-GPU detection and VRAM-gated device recommendations.

CPU remains the default everywhere; CUDA is used only when a real NVIDIA
dGPU is present AND free VRAM covers the model. All probes are lazy,
guarded, and cached - importing this module never touches torch.
"""

import shutil
import subprocess
import threading
import time

_SMI_TIMEOUT = 8
_cache = {}
_cache_lock = threading.Lock()


def _smi_gpus():
    """Parse nvidia-smi into [{name, total_mb, free_mb, driver}] (best first).
    [] when no NVIDIA driver/GPU. Never raises, never blocks long."""
    if shutil.which("nvidia-smi") is None:
        return []
    try:
        proc = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total,memory.free,"
                           "driver_version",
             "--format=csv,noheader,nounits"],
            stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
            text=True, errors="replace", timeout=_SMI_TIMEOUT)
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
            # Skip virtualized / tiny adapters that would only OOM.
            if total < 1500:
                continue
            out.append({"name": name, "total_mb": total, "free_mb": free,
                        "driver": parts[3]})
    except Exception:
        return []
    out.sort(key=lambda g: g["free_mb"], reverse=True)
    return out


def best_gpu():
    """Best NVIDIA dGPU dict, or None. Cached for 60s (VRAM changes)."""
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
    """torch CUDA status (lazy import - torch takes seconds to load)."""
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


def ct2_cuda():
    """Whether the installed ctranslate2 can run CUDA kernels (no GPU needed
    to answer - queries supported compute types, not devices)."""
    try:
        import ctranslate2
    except Exception as e:
        return {"ok": False, "reason": f"no ctranslate2: {e}"[:120]}
    try:
        types = list(ctranslate2.get_supported_compute_types("cuda") or [])
        if "float16" in types:
            return {"ok": True, "types": types}
        return {"ok": False, "reason": f"no fp16 CUDA types: {types}"[:120]}
    except Exception as e:
        return {"ok": False, "reason": str(e)[:160]}


# FP16 VRAM appetite per Whisper size class (weights + workspace + margin).
_WHISPER_FP16_MB = {"tiny": 1200, "base": 1200, "small": 1500,
                    "medium": 2600, "large": 5200, "large-v1": 5200,
                    "large-v2": 5200, "large-v3": 5200}
_WHISPER_INT8_MB = {"tiny": 700, "base": 700, "small": 900,
                    "medium": 1400, "large": 2800, "large-v1": 2800,
                    "large-v2": 2800, "large-v3": 2800}
_CANARY_CUDA_MIN_MB = 10240  # 3.9GB fp32 weights + activations + overhead


def recommend_whisper(model_id: str):
    """(device, compute_type, reason) for a Whisper model id.

    device in ("cuda", "cpu"); compute in ("float16", "int8_float16",
    "int8"). CPU stays int8 (fast on AVX); CUDA prefers fp16, int8 on
    smaller cards. Never raises.
    """
    mid = (model_id or "large-v3").strip() or "large-v3"
    need_fp16 = _WHISPER_FP16_MB.get(mid, 5200)
    need_int8 = _WHISPER_INT8_MB.get(mid, 2800)
    gpu = best_gpu()
    if gpu is None:
        return "cpu", "int8", "no NVIDIA dGPU detected"
    cuda = ct2_cuda()
    if not cuda.get("ok"):
        return "cpu", "int8", f"ctranslate2 has no CUDA ({cuda.get('reason', '?')})"
    free = int(gpu.get("free_mb") or 0)
    if free >= need_fp16:
        return "cuda", "float16", f"{gpu['name']} ({free}MB free)"
    if free >= need_int8 and "int8_float16" in (cuda.get("types") or []):
        return "cuda", "int8_float16", f"{gpu['name']} ({free}MB free, int8)"
    return ("cpu", "int8",
            f"only {free}MB VRAM free on {gpu.get('name', 'GPU')} "
            f"(need ~{need_int8}MB)")


def recommend_canary():
    """(use_cuda_bool, reason). Canary is fp32-hungry: 10GB+ free or CPU."""
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
    """One-line human summary for logs / troubleshooting."""
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
