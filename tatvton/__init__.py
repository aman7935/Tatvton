"""TatVTON — Tattoo Virtual Try-On SDK.

Public API::

    from tatvton import TatVTONPipeline, TatVTONPreviewPipeline, TatVTONConfig
    from tatvton import PointPrompt, BBoxPrompt, PipelineOutput, MaskResult
"""

from tatvton._version import __version__
from tatvton.config import TatVTONConfig
from tatvton.pipeline.preview_pipeline import TatVTONPreviewPipeline
from tatvton.pipeline.tatvton_pipeline import TatVTONPipeline
from tatvton.types import (
    BBoxPrompt,
    MaskResult,
    PipelineOutput,
    PointPrompt,
    RegionPrompt,
)

__all__ = [
    "BBoxPrompt",
    "MaskResult",
    "PipelineOutput",
    "PointPrompt",
    "RegionPrompt",
    "TatVTONConfig",
    "TatVTONPipeline",
    "TatVTONPreviewPipeline",
    "__version__",
]
