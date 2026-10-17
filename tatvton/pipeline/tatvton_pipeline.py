"""TatVTONPipeline — main orchestrator for tattoo virtual try-on."""

from __future__ import annotations

import logging

import numpy as np
import torch
from PIL import Image

from tatvton.config import TatVTONConfig
from tatvton.models.model_loader import ModelLoader
from tatvton.pipeline.inpainting_engine import InpaintingEngine
from tatvton.pipeline.preview_pipeline import TatVTONPreviewPipeline
from tatvton.postprocessing.compositing import Compositor
from tatvton.postprocessing.ink_blend import design_to_rgba
from tatvton.preprocessing.input_validator import InputValidator
from tatvton.preprocessing.skin_mask_extractor import SkinMaskExtractor
from tatvton.types import PipelineOutput, RegionPrompt
from tatvton.utils.mask import dilate_mask, mask_to_pil

logger = logging.getLogger(__name__)


class TatVTONPipeline:
    """End-to-end photorealistic tattoo virtual try-on ("composite, then refine").

    1. SAM 2 extracts the skin region at the prompt.
    2. The design is placed exactly on the skin (same as the CPU preview).
    3. SDXL inpainting repaints only the tattoo area at low strength, guided
       by IP-Adapter (the design) and optionally a ControlNet, so the ink
       looks settled into the skin while the design itself is kept.
    4. The refined area is feather-blended back into the original photo.

    Usage::

        from tatvton import TatVTONPipeline, TatVTONConfig, PointPrompt

        pipe = TatVTONPipeline()
        result = pipe(
            body_image=body,
            tattoo_image=tattoo,
            region=PointPrompt(coords=[(300, 400)]),
        )
        result.image.save("output.png")
    """

    def __init__(self, config: TatVTONConfig | None = None) -> None:
        self._config = config or TatVTONConfig()
        self._model_loader = ModelLoader(self._config)
        self._inpainting_engine = InpaintingEngine(self._config, self._model_loader)
        self._placer = TatVTONPreviewPipeline(self._config)
        self._compositor = Compositor(
            dilation_pixels=self._config.mask_dilation_pixels,
            feather_sigma=self._config.mask_feather_sigma,
        )

    @property
    def config(self) -> TatVTONConfig:
        """Return the current pipeline configuration."""
        return self._config

    @staticmethod
    def from_pretrained(
        repo_id: str,
        device: str = "cuda",
        dtype: torch.dtype = torch.float16,
    ) -> TatVTONPipeline:
        """Create a pipeline that uses the ControlNet in a HF Hub repository.

        Args:
            repo_id: Hub repo containing a ControlNet checkpoint.
            device: Target device.
            dtype: Model precision.

        Returns:
            Configured :class:`TatVTONPipeline`.
        """
        config = TatVTONConfig(
            controlnet_model_id=repo_id,
            device=device,
            dtype=dtype,
        )
        return TatVTONPipeline(config)

    def extract_mask(self, body_image: Image.Image, region: RegionPrompt) -> np.ndarray:
        """Return the SAM 2 skin mask for *region*, then unload SAM to free VRAM."""
        extractor = SkinMaskExtractor(self._model_loader.load_sam_predictor())
        mask_result = extractor.extract(body_image.convert("RGB"), region)
        logger.info("Mask extracted (score=%.3f)", mask_result.score)
        extractor.unload()
        self._model_loader.unload_component("sam")
        return mask_result.mask

    def __call__(
        self,
        body_image: Image.Image,
        tattoo_image: Image.Image,
        region: RegionPrompt,
        *,
        strength: float | None = None,
        ip_adapter_scale: float | None = None,
        num_inference_steps: int | None = None,
        seed: int | None = None,
        prompt: str | None = None,
        scale: float = 0.8,
        rotation: float = 0.0,
        auto_rotate: bool = True,
        opacity: float = 0.9,
        mask: np.ndarray | None = None,
    ) -> PipelineOutput:
        """Run the full tattoo virtual try-on pipeline.

        Args:
            body_image: Body photo (any resolution, RGB or RGBA).
            tattoo_image: Tattoo design (plain or transparent background).
            region: :class:`PointPrompt` or :class:`BBoxPrompt` for placement.
            strength: Refinement strength; higher = more AI change, less fidelity.
            ip_adapter_scale: IP-Adapter influence (per-call override).
            num_inference_steps: Denoising steps (per-call override).
            seed: Random seed (per-call override).
            prompt: Optional text prompt.
            scale: Design width as a fraction of the limb width.
            rotation: Extra counter-clockwise rotation in degrees.
            auto_rotate: Align the design with the limb's long axis.
            opacity: Ink opacity of the placed design.
            mask: Precomputed skin mask; skips SAM when given.

        Returns:
            :class:`PipelineOutput`; ``metadata["preview"]`` holds the placed
            design before AI refinement.
        """
        cfg = self._config

        # --- Validate inputs ---
        InputValidator.validate_call(
            body_image, tattoo_image, region, strength, ip_adapter_scale, num_inference_steps
        )

        _seed = seed if seed is not None else cfg.seed
        if _seed is None:
            _seed = int(torch.randint(0, 2**32, (1,)).item())

        body = body_image.convert("RGB")

        # --- Phase 1: Skin mask (SAM 2) ---
        if mask is None:
            logger.info("Phase 1: Extracting skin mask with SAM")
            mask = self.extract_mask(body, region)

        # --- Phase 2: Place the exact design on the skin ---
        logger.info("Phase 2: Placing design")
        placed = self._placer(
            body, tattoo_image, region,
            scale=scale, rotation=rotation, auto_rotate=auto_rotate,
            opacity=opacity, mask=mask,
        )
        ink = np.asarray(placed.raw_inpainted)[:, :, 3] > 12
        skin = np.asarray(placed.mask) > 127
        refine_mask = dilate_mask(ink, cfg.mask_dilation_pixels) & dilate_mask(skin, 2)
        if not refine_mask.any():
            raise ValueError("The tattoo doesn't overlap the selected skin region.")

        # --- Phase 3: ControlNet conditioning (optional) ---
        control_image = None
        if self._model_loader.has_controlnet:
            if cfg.use_densepose:
                logger.info("Phase 3: Extracting DensePose UV map")
                from tatvton.preprocessing.densepose_extractor import DensePoseExtractor

                dp = DensePoseExtractor(
                    config_path="",  # TODO: make configurable
                    weights_url="",
                    device=cfg.device,
                )
                control_image = dp.extract(body)
                dp.unload()
            else:
                control_image = placed.image

        # --- Phase 4: SDXL refinement of the tattoo area ---
        logger.info("Phase 4: Refining tattoo with SDXL")
        design = design_to_rgba(tattoo_image)
        design_on_white = Image.new("RGB", design.size, "white")
        design_on_white.paste(design, mask=design.getchannel("A"))

        raw = self._inpainting_engine.generate(
            image=placed.image,
            mask_image=mask_to_pil(refine_mask),
            tattoo_image=design_on_white,
            control_image=control_image,
            prompt=prompt,
            strength=strength,
            ip_adapter_scale=ip_adapter_scale,
            num_inference_steps=num_inference_steps,
            seed=_seed,
        )
        if raw.size != body.size:
            raw = raw.resize(body.size, Image.Resampling.LANCZOS)

        # --- Phase 5: Feathered blend back into the photo ---
        logger.info("Phase 5: Compositing")
        final = self._compositor.composite(placed.image, raw, refine_mask)
        if body_image.mode == "RGBA":
            final.putalpha(body_image.getchannel("A"))

        logger.info("Done (seed=%d)", _seed)
        return PipelineOutput(
            image=final,
            mask=mask_to_pil(refine_mask),
            raw_inpainted=raw,
            seed=_seed,
            metadata={**placed.metadata, "mode": "refine", "preview": placed.image},
        )

    def unload(self) -> None:
        """Release all loaded models and free GPU memory."""
        self._inpainting_engine.unload()
        self._model_loader.unload_all()
        self._placer.unload()
        logger.info("Pipeline fully unloaded")
