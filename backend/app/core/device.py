"""Pick the inference device: CUDA, then MPS, then CPU.

Torch is imported lazily so the API and DB layers run without the ML stack installed.
"""

from __future__ import annotations

import importlib
from types import ModuleType
from typing import Literal

Device = Literal["cuda", "mps", "cpu"]
VALID_REQUESTS = ("auto", "cuda", "mps", "cpu")


def _import_torch() -> ModuleType | None:
    try:
        return importlib.import_module("torch")
    except ImportError:
        return None


def available_devices(torch: ModuleType | None = None) -> list[Device]:
    """Devices usable in this process, best first. Always ends with "cpu"."""
    torch = torch if torch is not None else _import_torch()
    found: list[Device] = []
    if torch is not None:
        if torch.cuda.is_available():
            found.append("cuda")
        mps = getattr(torch.backends, "mps", None)
        if mps is not None and mps.is_available():
            found.append("mps")
    found.append("cpu")
    return found


def resolve_device(requested: str = "auto", torch: ModuleType | None = None) -> Device:
    """Resolve a config value to a concrete device.

    "auto" picks the best available. An explicit request for an unavailable accelerator
    raises instead of silently running 10x slower on CPU.
    """
    if requested not in VALID_REQUESTS:
        raise ValueError(f"device must be one of {VALID_REQUESTS}, got {requested!r}")
    devices = available_devices(torch)
    if requested == "auto":
        return devices[0]
    if requested not in devices:
        raise RuntimeError(
            f"device {requested!r} requested but not available (available: {devices})"
        )
    return requested  # type: ignore[return-value]
