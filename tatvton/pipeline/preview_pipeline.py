"""TatVTONPreviewPipeline — CPU-friendly tattoo preview without diffusion."""

from __future__ import annotations

import logging

import numpy as np
from PIL import Image

from tatvton.config import TatVTONConfig
from tatvton.models.model_loader import ModelLoader
from tatvton.postprocessing.ink_blend import (
    blend_ink,
    connected_region,
    design_to_rgba,
    mask_geometry,
    place_design,
)
from tatvton.preprocessing.input_validator import InputValidator
from tatvton.preprocessing.skin_mask_extractor import SkinMaskExtractor
from tatvton.types import BBoxPrompt, PipelineOutput, PointPrompt, RegionPrompt
from tatvton.utils.mask import mask_to_pil

logger = logging.getLogger(__name__)

_PREVIEW_DEFAULTS = TatVTONConfig(
    sam_model_id="facebook/sam2-hiera-tiny",
    device="cpu",
)


class TatVTONPreviewPipeline:
    """Fast tattoo preview: SAM 2 skin mask + warp + multiply blend.

    Runs on CPU in seconds and reproduces the design exactly, at the cost of
    the photorealism a diffusion model would add. Only SAM 2 is downloaded.

    Usage::

        from tatvton import TatVTONPreviewPipeline, PointPrompt

        pipe = TatVTONPreviewPipeline()
        result = pipe(body, tattoo, PointPrompt(coords=[(640, 1550)]))
        result.image.save("preview.png")
    """

    def __init__(self, config: TatVTONConfig | None = None) -> None:
        self._config = config or _PREVIEW_DEFAULTS
        self._model_loader = ModelLoader(self._config)

    @property
    def config(self) -> TatVTONConfig:
        """Return the current pipeline configuration."""
        return self._config

    def extract_mask(self, body_image: Image.Image, region: RegionPrompt) -> np.ndarray:
        """Return the SAM 2 skin mask for *region* as a boolean array.

        The SAM model stays loaded so repeated calls (e.g. from a UI) are fast.
        """
        extractor = SkinMaskExtractor(self._model_loader.load_sam_predictor())
        mask_result = extractor.extract(body_image, region)
        logger.info("Mask extracted (score=%.3f)", mask_result.score)
        return mask_result.mask

    def __call__(
        self,
        body_image: Image.Image,
        tattoo_image: Image.Image,
        region: RegionPrompt,
        *,
        scale: float = 0.8,
        rotation: float = 0.0,
        auto_rotate: bool = True,
        opacity: float = 0.9,
        mask: np.ndarray | None = None,
    ) -> PipelineOutput:
        """Place *tattoo_image* on the skin region selected by *region*.

        Args:
            body_image: Body photo (any resolution, RGB).
            tattoo_image: Tattoo design, transparent or on a white background.
            region: :class:`PointPrompt` or :class:`BBoxPrompt` for placement.
            scale: Design width as a fraction of the limb width.
            rotation: Extra counter-clockwise rotation in degrees.
            auto_rotate: Align the design with the limb's long axis.
            opacity: Ink opacity in ``(0, 1]``.
            mask: Precomputed skin mask; skips SAM when given.

        Returns:
            :class:`PipelineOutput`; ``raw_inpainted`` holds the placed design layer.
        """
        InputValidator.validate_call(body_image, tattoo_image, region)
        if not 0.0 < scale <= 3.0:
            raise ValueError(f"scale must be in (0.0, 3.0], got {scale}")
        if not 0.0 < opacity <= 1.0:
            raise ValueError(f"opacity must be in (0.0, 1.0], got {opacity}")

        body = body_image.convert("RGB")

        # --- Phase 1: Skin mask ---
        if mask is None:
            logger.info("Phase 1: Extracting skin mask with SAM")
            mask = self.extract_mask(body, region)

        # --- Phase 2: Placement ---
        center = _region_center(region)
        mask = connected_region(mask, center)
        geometry = mask_geometry(mask, at=center)
        angle = (geometry.angle if auto_rotate else 0.0) + rotation
        width = int(geometry.minor_extent * scale)
        logger.info(
            "Phase 2: Placing design (width=%dpx, angle=%.1f°)", width, angle
        )
        layer = place_design(
            design_to_rgba(tattoo_image),
            canvas_size=body.size,
            center=center,
            width=width,
            angle=angle,
            max_height=int(geometry.major_extent * 0.9),
        )

        # --- Phase 3: Ink blend ---
        logger.info("Phase 3: Blending ink")
        image = blend_ink(body, layer, mask, opacity=opacity)
        if body_image.mode == "RGBA":
            image.putalpha(body_image.getchannel("A"))

        return PipelineOutput(
            image=image,
            mask=mask_to_pil(mask),
            raw_inpainted=Image.fromarray((layer * 255).astype(np.uint8), mode="RGBA"),
            seed=0,
            metadata={"mode": "preview", "angle": angle, "width": width},
        )

    def unload(self) -> None:
        """Release SAM 2 from memory."""
        self._model_loader.unload_all()


def _region_center(region: RegionPrompt) -> tuple[float, float]:
    """Return the placement centre implied by a region prompt."""
    if isinstance(region, PointPrompt):
        coords = np.asarray(region.coords, dtype=np.float64)
        labels = (
            np.asarray(region.labels)
            if region.labels is not None
            else np.ones(len(coords))
        )
        positive = coords[labels == 1] if (labels == 1).any() else coords
        return float(positive[:, 0].mean()), float(positive[:, 1].mean())
    if isinstance(region, BBoxPrompt):
        x_min, y_min, x_max, y_max = region.bbox
        return (x_min + x_max) / 2, (y_min + y_max) / 2
    raise TypeError(f"Unsupported region type: {type(region).__name__}")
