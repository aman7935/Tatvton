"""Tests for CPU preview ink blending and the preview pipeline (SAM mocked)."""

from __future__ import annotations

import numpy as np
import pytest
from PIL import Image

from tatvton import PointPrompt, TatVTONPreviewPipeline
from tatvton.postprocessing.ink_blend import (
    blend_ink,
    connected_region,
    design_to_rgba,
    has_plain_background,
    mask_geometry,
    place_design,
)


@pytest.fixture()
def white_bg_design() -> Image.Image:
    """100x100 white canvas with a grey 40x20 rectangle (line-art stand-in)."""
    arr = np.full((100, 100, 3), 255, dtype=np.uint8)
    arr[30:50, 20:60] = 128
    return Image.fromarray(arr, mode="RGB")


@pytest.fixture()
def vertical_limb_mask() -> np.ndarray:
    """256x256 mask: a 40px-wide vertical bar, like a forearm."""
    mask = np.zeros((256, 256), dtype=bool)
    mask[20:236, 108:148] = True
    return mask


class TestDesignToRgba:
    def test_background_becomes_transparent(self, white_bg_design: Image.Image) -> None:
        rgba = design_to_rgba(white_bg_design)
        assert rgba.mode == "RGBA"
        # Cropped to the artwork
        assert rgba.size == (40, 20)
        assert np.asarray(rgba)[:, :, 3].min() == 255

    def test_grey_ink_becomes_black(self, white_bg_design: Image.Image) -> None:
        rgb = np.asarray(design_to_rgba(white_bg_design))[:, :, :3]
        assert rgb.max() <= 1

    def test_colour_preserved(self) -> None:
        arr = np.full((50, 50, 3), 255, dtype=np.uint8)
        arr[10:40, 10:40] = (220, 20, 20)
        rgb = np.asarray(design_to_rgba(Image.fromarray(arr)))[:, :, :3]
        assert rgb[..., 0].mean() > 200
        assert rgb[..., 1].mean() < 30

    def test_transparent_design_keeps_alpha(self) -> None:
        arr = np.zeros((60, 60, 4), dtype=np.uint8)
        arr[20:40, 10:50] = (30, 30, 30, 255)
        rgba = design_to_rgba(Image.fromarray(arr, mode="RGBA"))
        assert rgba.size == (40, 20)
        assert np.asarray(rgba)[:, :, :3].max() == 30

    def test_palette_png_with_transparency(self) -> None:
        # Transparent pixels are black underneath, like many exported PNGs
        arr = np.zeros((60, 60, 4), dtype=np.uint8)
        arr[20:40, 10:50] = (200, 30, 30, 255)
        design = Image.fromarray(arr, mode="RGBA").convert("P")
        design.info["transparency"] = design.getpixel((0, 0))
        rgba = design_to_rgba(design)
        assert rgba.size == (40, 20)
        assert np.asarray(rgba)[:, :, 0].mean() > 150

    def test_light_on_dark_design_inverted(self) -> None:
        arr = np.zeros((100, 100, 3), dtype=np.uint8)
        arr[40:60, 30:70] = 255  # white artwork on black
        rgba = np.asarray(design_to_rgba(Image.fromarray(arr)))
        assert rgba.shape[:2] == (20, 40)
        assert rgba[:, :, 3].min() == 255
        assert rgba[:, :, :3].max() <= 1  # becomes black ink

    def test_coloured_paper_background(self) -> None:
        arr = np.full((100, 100, 3), (235, 220, 190), dtype=np.uint8)  # beige
        arr[40:60, 30:70] = (20, 20, 20)
        rgba = np.asarray(design_to_rgba(Image.fromarray(arr)))
        assert rgba.shape[:2] == (20, 40)
        assert rgba[:, :, :3].max() < 40

    def test_colourful_on_dark_keeps_colour(self) -> None:
        arr = np.zeros((100, 100, 3), dtype=np.uint8)
        arr[40:60, 30:70] = (30, 200, 40)  # green art on black
        rgba = np.asarray(design_to_rgba(Image.fromarray(arr)))
        assert rgba.shape[:2] == (20, 40)
        assert rgba[:, :, 1].mean() > 150
        assert rgba[:, :, 0].mean() < 80

    def test_faint_watermark_ignored_for_crop(self) -> None:
        arr = np.zeros((100, 100, 4), dtype=np.uint8)
        arr[:, :, :3] = 255
        arr[:, :, 3] = 40  # watermark everywhere
        arr[40:60, 30:70] = (0, 0, 0, 255)
        assert design_to_rgba(Image.fromarray(arr, mode="RGBA")).size == (40, 20)


class TestConnectedRegion:
    def test_keeps_seed_component(self) -> None:
        mask = np.zeros((64, 64), dtype=bool)
        mask[5:20, 5:20] = True
        mask[40:60, 40:60] = True
        result = connected_region(mask, (10, 10))
        assert result.sum() == 15 * 15
        assert not result[50, 50]

    def test_seed_outside_mask_returns_full(self) -> None:
        mask = np.zeros((64, 64), dtype=bool)
        mask[5:20, 5:20] = True
        assert np.array_equal(connected_region(mask, (50, 50)), mask)


class TestMaskGeometry:
    def test_vertical_limb(self, vertical_limb_mask: np.ndarray) -> None:
        geo = mask_geometry(vertical_limb_mask)
        assert abs(geo.angle) < 1.0
        assert geo.major_extent > geo.minor_extent
        assert 30 < geo.minor_extent <= 40

    def test_local_width(self) -> None:
        # Wide at the top, narrow at the bottom
        mask = np.zeros((200, 200), dtype=bool)
        mask[0:100, 60:140] = True
        mask[100:200, 90:110] = True
        assert mask_geometry(mask, at=(100, 150)).minor_extent < 25

    def test_empty_mask_raises(self) -> None:
        with pytest.raises(ValueError, match="empty"):
            mask_geometry(np.zeros((10, 10), dtype=bool))


class TestBlend:
    def test_place_design_centred(self) -> None:
        design = Image.new("RGBA", (20, 10), (0, 0, 0, 255))
        layer = place_design(design, (100, 100), center=(50, 50), width=20)
        ys, xs = np.nonzero(layer[:, :, 3] > 0.5)
        assert 45 <= xs.mean() <= 55
        assert 45 <= ys.mean() <= 55

    def test_ink_only_inside_mask(
        self, dummy_body_image: Image.Image, dummy_mask: np.ndarray
    ) -> None:
        layer = np.zeros((256, 256, 4), dtype=np.float32)
        layer[:, :, 3] = 1.0  # black ink everywhere
        out = np.asarray(blend_ink(dummy_body_image, layer, dummy_mask))
        assert out[128, 128].mean() < 60  # inside: darkened
        assert out[5, 5].mean() == 180  # outside: untouched


class TestPreviewPipeline:
    def test_end_to_end_with_mask(
        self, dummy_body_image: Image.Image, dummy_tattoo_image: Image.Image,
        dummy_mask: np.ndarray,
    ) -> None:
        pipe = TatVTONPreviewPipeline()
        result = pipe(
            dummy_body_image, dummy_tattoo_image,
            PointPrompt(coords=[(128, 128)]), mask=dummy_mask,
        )
        assert result.image.size == dummy_body_image.size
        assert result.metadata["mode"] == "preview"
        assert np.asarray(result.image)[128, 128].mean() < 180

    def test_invalid_scale(
        self, dummy_body_image: Image.Image, dummy_tattoo_image: Image.Image,
    ) -> None:
        with pytest.raises(ValueError, match="scale"):
            TatVTONPreviewPipeline()(
                dummy_body_image, dummy_tattoo_image,
                PointPrompt(coords=[(10, 10)]), scale=0,
            )


class TestHasPlainBackground:
    def test_white_background(self, white_bg_design: Image.Image) -> None:
        assert has_plain_background(white_bg_design)

    def test_transparent(self, dummy_tattoo_image: Image.Image) -> None:
        arr = np.asarray(dummy_tattoo_image).copy()
        arr[0, 0, 3] = 0
        assert has_plain_background(Image.fromarray(arr, mode="RGBA"))

    def test_photo_like_noise(self) -> None:
        rng = np.random.default_rng(0)
        arr = rng.integers(0, 256, (100, 100, 3), dtype=np.uint8)
        assert not has_plain_background(Image.fromarray(arr))
