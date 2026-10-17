"""Tests for ModelLoader's GPU loading strategy (no models downloaded)."""

from __future__ import annotations

from unittest.mock import patch

import pytest
import torch

from tatvton import TatVTONConfig
from tatvton.models.model_loader import ModelLoader


def _loader(strategy: str, vram_gb: float, cuda: bool = True) -> ModelLoader:
    with patch("torch.cuda.is_available", return_value=cuda):
        loader = ModelLoader(TatVTONConfig(offload_strategy=strategy))  # type: ignore[arg-type]
    patcher = patch("tatvton.models.model_loader.gpu_memory_gb", return_value=vram_gb)
    patcher.start()
    return loader


@pytest.fixture(autouse=True)
def _stop_patches():
    yield
    patch.stopall()


class TestDirectGpuLoading:
    def test_auto_on_t4_loads_direct(self) -> None:
        assert _loader("auto", 14.7)._loads_directly_to_gpu()

    def test_auto_on_small_gpu_offloads(self) -> None:
        assert not _loader("auto", 8.0)._loads_directly_to_gpu()

    def test_none_loads_direct(self) -> None:
        assert _loader("none", 8.0)._loads_directly_to_gpu()

    def test_model_offload_never_direct(self) -> None:
        assert not _loader("model", 40.0)._loads_directly_to_gpu()

    def test_cpu_never_direct(self) -> None:
        loader = _loader("auto", 0.0, cuda=False)
        assert loader.device == torch.device("cpu")
        assert not loader._loads_directly_to_gpu()
