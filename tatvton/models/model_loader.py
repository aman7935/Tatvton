"""Lazy model loading with VRAM management."""

from __future__ import annotations

import gc
import logging
from typing import Any

import torch

from tatvton.config import TatVTONConfig
from tatvton.utils.device import gpu_memory_gb, resolve_device, resolve_dtype

logger = logging.getLogger(__name__)

# Enough VRAM to keep SDXL + IP-Adapter resident without CPU offload
_DIRECT_GPU_MIN_GB = 14.0


class ModelLoader:
    """Centralised model loading with lazy initialisation and VRAM management.

    Models are loaded on first access and can be individually unloaded
    to reclaim GPU memory between pipeline phases.
    """

    def __init__(self, config: TatVTONConfig) -> None:
        self._config = config
        self._device = resolve_device(config.device)
        self._dtype = resolve_dtype(config.dtype, self._device)
        self._components: dict[str, Any] = {}

    @property
    def device(self) -> torch.device:
        return self._device

    @property
    def dtype(self) -> torch.dtype:
        return self._dtype

    def load_sam_predictor(self) -> Any:
        """Load and return a SAM 2 image predictor."""
        if "sam" not in self._components:
            logger.info("Loading SAM 2 from %s", self._config.sam_model_id)
            from sam2.sam2_image_predictor import SAM2ImagePredictor

            predictor = SAM2ImagePredictor.from_pretrained(
                self._config.sam_model_id,
                device=self._device,
            )
            self._components["sam"] = predictor
        return self._components["sam"]

    @property
    def has_controlnet(self) -> bool:
        """Whether a ControlNet is configured."""
        return self._config.controlnet_model_id is not None

    def load_controlnet(self) -> Any:
        """Load and return the ControlNet model (requires ``controlnet_model_id``)."""
        if self._config.controlnet_model_id is None:
            raise ValueError("No controlnet_model_id configured.")
        if "controlnet" not in self._components:
            logger.info(
                "Loading ControlNet from %s", self._config.controlnet_model_id
            )
            from diffusers import ControlNetModel

            controlnet = ControlNetModel.from_pretrained(
                self._config.controlnet_model_id,
                torch_dtype=self._dtype,
            )
            self._components["controlnet"] = controlnet
        return self._components["controlnet"]

    def load_inpainting_pipeline(self) -> Any:
        """Load and return the SDXL inpainting pipeline.

        Uses the ControlNet variant when ``controlnet_model_id`` is set, and
        configures IP-Adapter and the chosen offload strategy.
        """
        if "pipeline" not in self._components:
            cfg = self._config
            direct = self._loads_directly_to_gpu()
            kwargs: dict[str, Any] = {"torch_dtype": self._dtype}
            if self._dtype == torch.float16:
                kwargs["variant"] = "fp16"
            if direct:
                # Weights go straight to VRAM instead of being staged in CPU RAM
                kwargs["device_map"] = self._device.type
            if cfg.vae_model_id is not None:
                from diffusers import AutoencoderKL

                logger.info("Loading VAE from %s", cfg.vae_model_id)
                vae = AutoencoderKL.from_pretrained(cfg.vae_model_id, torch_dtype=self._dtype)
                kwargs["vae"] = vae.to(self._device) if direct else vae

            logger.info("Loading SDXL inpainting pipeline from %s", cfg.sdxl_model_id)
            if self.has_controlnet:
                from diffusers import StableDiffusionXLControlNetInpaintPipeline

                controlnet = self.load_controlnet()
                pipe = StableDiffusionXLControlNetInpaintPipeline.from_pretrained(
                    cfg.sdxl_model_id,
                    controlnet=controlnet.to(self._device) if direct else controlnet,
                    **kwargs,
                )
            else:
                from diffusers import StableDiffusionXLInpaintPipeline

                pipe = StableDiffusionXLInpaintPipeline.from_pretrained(
                    cfg.sdxl_model_id, **kwargs
                )

            if cfg.use_ip_adapter:
                logger.info(
                    "Loading IP-Adapter: %s / %s",
                    cfg.ip_adapter_repo_id,
                    cfg.ip_adapter_weight_name,
                )
                pipe.load_ip_adapter(
                    cfg.ip_adapter_repo_id,
                    subfolder=cfg.ip_adapter_subfolder,
                    weight_name=cfg.ip_adapter_weight_name,
                    image_encoder_folder=cfg.ip_adapter_image_encoder_folder,
                )
                pipe.set_ip_adapter_scale(cfg.ip_adapter_scale)

            if direct:
                logger.info("Loaded pipeline directly on %s (no CPU offload)", self._device)
            else:
                self._apply_offload(pipe)
            self._components["pipeline"] = pipe

        return self._components["pipeline"]

    def unload_component(self, name: str) -> None:
        """Unload a named component and free GPU memory."""
        component = self._components.pop(name, None)
        if component is not None:
            del component
            self._flush_gpu()
            logger.info("Unloaded component '%s'", name)

    def unload_all(self) -> None:
        """Unload every loaded component."""
        names = list(self._components.keys())
        self._components.clear()
        self._flush_gpu()
        logger.info("Unloaded all components: %s", names)

    def _loads_directly_to_gpu(self) -> bool:
        """Whether to load the pipeline straight onto the GPU (no CPU staging)."""
        if self._device.type != "cuda":
            return False
        strategy = self._config.offload_strategy
        if strategy == "auto":
            return gpu_memory_gb() >= _DIRECT_GPU_MIN_GB
        return strategy == "none"

    def _apply_offload(self, pipe: Any) -> None:
        """Apply the configured offload strategy to the diffusion pipeline."""
        strategy = self._config.offload_strategy
        if self._device.type != "cuda":
            # CPU offload hooks need a GPU to offload to
            pipe.to(self._device)
        elif strategy == "sequential":
            pipe.enable_sequential_cpu_offload()
            logger.info("Enabled sequential CPU offload")
        elif strategy in ("model", "auto"):
            pipe.enable_model_cpu_offload()
            logger.info("Enabled model CPU offload")
        else:
            pipe.to(self._device)

    @staticmethod
    def _flush_gpu() -> None:
        """Run garbage collection and clear the CUDA cache."""
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
