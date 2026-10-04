from __future__ import annotations

from types import SimpleNamespace

import pytest

from app.core import device as device_mod
from app.core.device import resolve_device


def fake_torch(cuda: bool, mps: bool):
    return SimpleNamespace(
        cuda=SimpleNamespace(is_available=lambda: cuda),
        backends=SimpleNamespace(mps=SimpleNamespace(is_available=lambda: mps)),
    )


@pytest.mark.parametrize(
    ("cuda", "mps", "expected"),
    [(True, True, "cuda"), (False, True, "mps"), (False, False, "cpu")],
)
def test_auto_priority(cuda, mps, expected):
    assert resolve_device("auto", torch=fake_torch(cuda, mps)) == expected


def test_auto_without_torch_is_cpu(monkeypatch):
    monkeypatch.setattr(device_mod, "_import_torch", lambda: None)
    assert resolve_device("auto") == "cpu"


def test_explicit_cpu_always_works():
    assert resolve_device("cpu", torch=fake_torch(True, True)) == "cpu"


def test_explicit_unavailable_accelerator_raises():
    with pytest.raises(RuntimeError, match="cuda"):
        resolve_device("cuda", torch=fake_torch(False, False))


def test_invalid_request_raises():
    with pytest.raises(ValueError):
        resolve_device("tpu")
