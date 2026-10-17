"""Device and dtype selection helpers."""

from __future__ import annotations

import torch


def resolve_device(device: str = "cuda") -> torch.device:
    """Return a :class:`torch.device`, falling back gracefully.

    Priority: requested device → CUDA → MPS → CPU.
    """
    if device == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    if device == "mps" and torch.backends.mps.is_available():
        return torch.device("mps")
    if device not in ("cuda", "mps", "cpu"):
        return torch.device(device)
    return torch.device("cpu")


def resolve_dtype(dtype: torch.dtype, device: torch.device) -> torch.dtype:
    """Ensure *dtype* is compatible with *device*.

    MPS and CPU do not support ``float16`` for all ops — fall back to
    ``bfloat16`` or ``float32`` respectively.
    """
    if dtype == torch.float16:
        if device.type == "cpu":
            return torch.float32
        if device.type == "mps":
            return torch.float32
    return dtype


def gpu_memory_gb() -> float:
    """Return total GPU memory in GB, or ``0.0`` if no GPU."""
    if torch.cuda.is_available():
        return torch.cuda.get_device_properties(0).total_memory / (1024**3)
    return 0.0
