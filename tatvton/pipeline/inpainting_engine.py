"""SDXL inpainting engine with optional ControlNet and IP-Adapter."""

from __future__ import annotations

import logging
from typing import Any

import torch
from PIL import Image

from tatvton.config import TatVTONConfig
from tatvton.models.model_loader import ModelLoader

logger = logging.getLogger(__name__)

DEFAULT_PROMPT = (
    "photo of a healed tattoo on skin, ink settled into the skin, natural skin "
    "texture and pores, realistic lighting, high detail"
)
DEFAULT_NEGATIVE_PROMPT = (
    "sticker, decal, printed, painted on, flat, cartoon, illustration, blurry, "
    "deformed, extra limbs, watermark, text"
)


class InpaintingEngine:
    """Wraps the SDXL inpainting pipeline (optionally with ControlNet / IP-Adapter).

    Images are passed at full resolution: the pipeline crops a padded box
    around the mask, inpaints it at :attr:`TatVTONConfig.resolution`, and
    pastes it back, so detail is kept and the aspect ratio never changes.
    """

    def __init__(self, config: TatVTONConfig, model_loader: ModelLoader) -> None:
        self._config = config
        self._model_loader = model_loader
        self._pipe: Any | None = None

    def _ensure_loaded(self) -> Any:
        """Lazy-load the diffusion pipeline on first call."""
        if self._pipe is None:
            self._pipe = self._model_loader.load_inpainting_pipeline()
        return self._pipe

    def generate(
        self,
        image: Image.Image,
        mask_image: Image.Image,
        tattoo_image: Image.Image | None = None,
        control_image: Image.Image | None = None,
        *,
        prompt: str | None = None,
        negative_prompt: str | None = None,
        strength: float | None = None,
        guidance_scale: float | None = None,
        controlnet_conditioning_scale: float | None = None,
        ip_adapter_scale: float | None = None,
        num_inference_steps: int | None = None,
        seed: int | None = None,
    ) -> Image.Image:
        """Run SDXL inpainting over the masked area.

        Args:
            image: Starting image (e.g. body with the design placed), RGB.
            mask_image: ``'L'`` mask, white where the image may change.
            tattoo_image: Design shown to IP-Adapter (ignored if disabled).
            control_image: ControlNet conditioning (required with ControlNet).
            prompt: Text prompt (default: healed-tattoo photo prompt).
            negative_prompt: Negative prompt.
            strength: Denoising strength override.
            guidance_scale: Classifier-free guidance scale override.
            controlnet_conditioning_scale: ControlNet scale override.
            ip_adapter_scale: IP-Adapter scale override.
            num_inference_steps: Number of denoising steps override.
            seed: Random seed for reproducibility.

        Returns:
            Inpainted PIL image at the size of *image*.
        """
        pipe = self._ensure_loaded()
        cfg = self._config

        kwargs: dict[str, Any] = {
            "prompt": prompt or DEFAULT_PROMPT,
            "negative_prompt": negative_prompt or DEFAULT_NEGATIVE_PROMPT,
            "image": image,
            "mask_image": mask_image,
            "strength": strength if strength is not None else cfg.strength,
            "guidance_scale": guidance_scale if guidance_scale is not None else cfg.guidance_scale,
            "num_inference_steps": (
                num_inference_steps if num_inference_steps is not None else cfg.num_inference_steps
            ),
            "padding_mask_crop": cfg.mask_crop_padding,
            "height": cfg.resolution,
            "width": cfg.resolution,
        }

        if cfg.use_ip_adapter:
            if tattoo_image is None:
                raise ValueError("tattoo_image is required when IP-Adapter is enabled.")
            if ip_adapter_scale is not None:
                pipe.set_ip_adapter_scale(ip_adapter_scale)
            kwargs["ip_adapter_image"] = tattoo_image

        if self._model_loader.has_controlnet:
            if control_image is None:
                raise ValueError("control_image is required when ControlNet is configured.")
            kwargs["control_image"] = control_image
            kwargs["controlnet_conditioning_scale"] = (
                controlnet_conditioning_scale
                if controlnet_conditioning_scale is not None
                else cfg.controlnet_conditioning_scale
            )

        _seed = seed if seed is not None else cfg.seed
        if _seed is not None:
            kwargs["generator"] = torch.Generator(
                device=self._model_loader.device
            ).manual_seed(_seed)

        logger.info(
            "Generating: steps=%d, strength=%.2f, guidance=%.1f, controlnet=%s, ip_adapter=%s",
            kwargs["num_inference_steps"],
            kwargs["strength"],
            kwargs["guidance_scale"],
            self._model_loader.has_controlnet,
            cfg.use_ip_adapter,
        )
        return pipe(**kwargs).images[0]

    def unload(self) -> None:
        """Release the diffusion pipeline from memory."""
        self._pipe = None
        self._model_loader.unload_component("pipeline")
        self._model_loader.unload_component("controlnet")
