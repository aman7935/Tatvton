"""Pipeline configuration — frozen dataclass with sensible defaults."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch


@dataclass(frozen=True)
class TatVTONConfig:
    """Immutable configuration for :class:`~tatvton.TatVTONPipeline`.

    Three-level override strategy:
      1. Defaults here (optimised for 12 GB GPU).
      2. Pass a custom ``TatVTONConfig`` to the pipeline constructor.
      3. Per-call keyword overrides in ``pipeline.__call__(...)``.

    Every default model is public on the Hugging Face Hub (no login needed).
    """

    # -- Model IDs --------------------------------------------------------
    sdxl_model_id: str = "diffusers/stable-diffusion-xl-1.0-inpainting-0.1"
    # SDXL's stock VAE overflows in fp16 (black/NaN images); this one doesn't.
    vae_model_id: str | None = "madebyollin/sdxl-vae-fp16-fix"
    # Optional body-structure guide; ``None`` runs plain SDXL inpainting.
    controlnet_model_id: str | None = None
    ip_adapter_repo_id: str = "h94/IP-Adapter"
    ip_adapter_subfolder: str = "sdxl_models"
    ip_adapter_weight_name: str = "ip-adapter-plus_sdxl_vit-h.safetensors"
    # ViT-H adapters need the ViT-H encoder, not the default bigG one.
    ip_adapter_image_encoder_folder: str = "models/image_encoder"
    sam_model_id: str = "facebook/sam2-hiera-large"

    # -- Inference parameters ---------------------------------------------
    resolution: int = 1024
    num_inference_steps: int = 30
    guidance_scale: float = 7.5
    # Low strength refines the placed design instead of redrawing it.
    strength: float = 0.4
    controlnet_conditioning_scale: float = 0.5
    ip_adapter_scale: float = 0.6
    use_ip_adapter: bool = True

    # -- Mask processing --------------------------------------------------
    mask_dilation_pixels: int = 10
    mask_feather_sigma: float = 5.0
    # Context kept around the tattoo when cropping it for inpainting.
    mask_crop_padding: int = 64

    # -- Device / precision -----------------------------------------------
    device: str = "cuda"
    dtype: torch.dtype = torch.float16
    # "auto": load straight to the GPU when it has >= 14 GB (e.g. Colab T4),
    # otherwise offload idle models to CPU RAM ("model").
    offload_strategy: Literal["auto", "none", "model", "sequential"] = "auto"

    # -- Optional features ------------------------------------------------
    use_densepose: bool = False
    seed: int | None = None

    def replace(self, **overrides: object) -> TatVTONConfig:
        """Return a new config with the given fields replaced."""
        from dataclasses import asdict

        current = asdict(self)
        current.update(overrides)
        return TatVTONConfig(**current)
