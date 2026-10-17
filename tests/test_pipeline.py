"""Tests for TatVTONPipeline input validation and config."""

from __future__ import annotations

from unittest.mock import patch

import numpy as np
import pytest
from PIL import Image

from tatvton import BBoxPrompt, PointPrompt, TatVTONConfig, TatVTONPipeline


class TestTatVTONConfig:
    def test_frozen(self) -> None:
        cfg = TatVTONConfig()
        with pytest.raises(AttributeError):
            cfg.resolution = 512  # type: ignore[misc]

    def test_replace(self) -> None:
        cfg = TatVTONConfig()
        new = cfg.replace(resolution=768, num_inference_steps=20)
        assert new.resolution == 768
        assert new.num_inference_steps == 20
        assert cfg.resolution == 1024  # original unchanged

    def test_defaults(self) -> None:
        cfg = TatVTONConfig()
        assert cfg.resolution == 1024
        assert cfg.strength == 0.4
        assert cfg.controlnet_model_id is None
        assert cfg.ip_adapter_image_encoder_folder == "models/image_encoder"
        assert cfg.use_densepose is False


class TestPipelineInit:
    def test_default_config(self) -> None:
        pipe = TatVTONPipeline()
        assert pipe.config.resolution == 1024

    def test_custom_config(self) -> None:
        cfg = TatVTONConfig(resolution=768)
        pipe = TatVTONPipeline(cfg)
        assert pipe.config.resolution == 768


class TestPointPrompt:
    def test_valid(self) -> None:
        p = PointPrompt(coords=[(100, 200)])
        assert len(p.coords) == 1

    def test_empty_raises(self) -> None:
        with pytest.raises(ValueError, match="at least one"):
            PointPrompt(coords=[])

    def test_label_mismatch_raises(self) -> None:
        with pytest.raises(ValueError, match="labels length"):
            PointPrompt(coords=[(1, 2), (3, 4)], labels=[1])


class TestBBoxPrompt:
    def test_valid(self) -> None:
        b = BBoxPrompt(bbox=(10, 20, 100, 200))
        assert b.bbox == (10, 20, 100, 200)

    def test_invalid_raises(self) -> None:
        with pytest.raises(ValueError, match="Invalid bbox"):
            BBoxPrompt(bbox=(100, 200, 10, 20))


class TestRefineFlow:
    """Full pipeline flow with SAM skipped (mask given) and SDXL mocked."""

    def test_only_tattoo_area_is_refined(
        self, dummy_body_image: Image.Image, dummy_tattoo_image: Image.Image,
        dummy_mask: np.ndarray,
    ) -> None:
        pipe = TatVTONPipeline(TatVTONConfig(device="cpu"))
        calls: dict = {}

        def fake_generate(image, mask_image, tattoo_image=None, control_image=None, **kw):
            calls.update(image=image, mask=mask_image, tattoo=tattoo_image, control=control_image)
            return Image.new("RGB", image.size, (0, 0, 255))  # obvious "AI" output

        with patch.object(pipe._inpainting_engine, "generate", side_effect=fake_generate):
            result = pipe(
                dummy_body_image, dummy_tattoo_image,
                PointPrompt(coords=[(128, 128)]), mask=dummy_mask, seed=1,
            )

        refine = np.asarray(result.mask) > 127
        assert refine.any() and not refine.all()
        assert calls["control"] is None  # no ControlNet by default
        assert calls["tattoo"].mode == "RGB"
        out = np.asarray(result.image).astype(int)
        assert out[128, 128, 2] > 200  # refined area took the AI output
        assert tuple(out[2, 2]) == (180, 180, 180)  # far outside: untouched
        assert result.metadata["mode"] == "refine"
        assert result.seed == 1
