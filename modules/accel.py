"""Accelerator abstraction: run the same tree on NVIDIA (CUDA) or Ascend (NPU).

The code base grew up on CUDA, so a handful of call sites hard-code
``torch.cuda`` / ``'cuda'`` / backend ``"nccl"``.  They now go through this
module, which auto-detects at import time:

* ``ACCEL``   — ``"cuda"``, ``"npu"`` or ``"cpu"``
* ``device(i)`` / ``set_device(i)``
* ``backend()`` — ``"nccl"`` for CUDA, ``"hccl"`` for Ascend
* ``autocast(dtype)`` — context manager with the right ``device_type``
* ``GradScaler()`` — scaler bound to the right device type
* ``manual_seed_all(seed)`` / ``empty_cache()`` / ``is_gpu()``

Override with ``ACCEL=npu`` / ``ACCEL=cuda`` when auto-detection guesses wrong
(e.g. a box with both a GPU and an Ascend card).

Importing this module imports ``torch_npu`` when it is installed; that is what
registers the ``npu`` device with torch, so please import ``modules.accel``
before touching ``torch.npu``.
"""

from __future__ import annotations

import os

import torch

_TORCH_NPU_ERROR = None


def _try_import_npu():
    global _TORCH_NPU_ERROR
    if not hasattr(torch, "npu"):
        try:
            import torch_npu  # noqa: F401  (registers torch.npu)
        except Exception as exc:  # pragma: no cover - depends on the host
            _TORCH_NPU_ERROR = exc
            return False
    try:
        return bool(torch.npu.is_available())
    except Exception as exc:  # pragma: no cover
        _TORCH_NPU_ERROR = exc
        return False


def _detect() -> str:
    forced = os.environ.get("ACCEL", "").strip().lower()
    if forced in ("npu", "cuda", "cpu"):
        return forced
    if _try_import_npu():
        return "npu"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


ACCEL: str = _detect()
IS_NPU: bool = ACCEL == "npu"
IS_CUDA: bool = ACCEL == "cuda"


def is_gpu() -> bool:
    """True on either accelerator (used for fused optimizers, pin_memory, ...)."""
    return ACCEL in ("cuda", "npu")


def available() -> bool:
    return is_gpu()


def device(index=None):
    if index is None:
        return torch.device(ACCEL if ACCEL != "cuda" else "cuda")
    return torch.device(f"{ACCEL}:{index}")


def set_device(index: int) -> None:
    if IS_NPU:
        torch.npu.set_device(index)
    elif IS_CUDA:
        torch.cuda.set_device(index)


def device_count() -> int:
    if IS_NPU:
        return torch.npu.device_count()
    if IS_CUDA:
        return torch.cuda.device_count()
    return 0


def backend() -> str:
    return {"npu": "hccl", "cuda": "nccl"}.get(ACCEL, "gloo")


def manual_seed_all(seed: int) -> None:
    if IS_NPU:
        torch.npu.manual_seed_all(seed)
    elif IS_CUDA:
        torch.cuda.manual_seed_all(seed)


def empty_cache() -> None:
    if IS_NPU:
        torch.npu.empty_cache()
    elif IS_CUDA:
        torch.cuda.empty_cache()


def synchronize() -> None:
    if IS_NPU:
        torch.npu.synchronize()
    elif IS_CUDA:
        torch.cuda.synchronize()


def autocast(dtype=torch.bfloat16, enabled: bool = True):
    """torch.amp.autocast bound to the active accelerator."""
    return torch.amp.autocast(device_type=ACCEL, dtype=dtype, enabled=enabled)


def GradScaler(enabled: bool = True):
    return torch.amp.GradScaler(ACCEL, enabled=enabled)


def amp_supported_dtype():
    """bf16 everywhere we run; Ascend 910B has native bf16."""
    return torch.bfloat16


def cuda_like_optimizations() -> None:
    """TF32 / cuDNN knobs — no-ops (and no crash) on NPU."""
    if not IS_CUDA:
        return
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True



def max_memory_allocated() -> float:
    """Peak bytes allocated on the current accelerator (0 on CPU)."""
    if IS_NPU:
        return float(torch.npu.max_memory_allocated())
    if IS_CUDA:
        return float(torch.cuda.max_memory_allocated())
    return 0.0


def reset_peak_memory_stats() -> None:
    if IS_NPU:
        torch.npu.reset_peak_memory_stats()
    elif IS_CUDA:
        torch.cuda.reset_peak_memory_stats()


def total_memory() -> float:
    """Total HBM of the current device in bytes (0 on CPU)."""
    if IS_NPU:
        return float(torch.npu.get_device_properties(torch.npu.current_device()).total_memory)
    if IS_CUDA:
        return float(torch.cuda.get_device_properties(torch.cuda.current_device()).total_memory)
    return 0.0

def describe() -> str:
    if IS_NPU:
        return f"Ascend NPU x{device_count()} (torch_npu)"
    if IS_CUDA:
        try:
            name = torch.cuda.get_device_name(0)
        except Exception:
            name = "cuda"
        return f"CUDA x{device_count()} ({name})"
    return "CPU"
