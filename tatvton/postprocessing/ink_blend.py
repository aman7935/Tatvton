"""CPU-only tattoo ink blending — design cleanup, placement, multiply blend.

No diffusion model is involved: the tattoo design is warped onto the skin
and blended with a multiply mode so skin texture and lighting show through
the ink.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

from tatvton.utils.image import numpy_to_pil, pil_to_numpy
from tatvton.utils.mask import feather_mask


@dataclass(frozen=True)
class MaskGeometry:
    """Orientation and size of a skin mask.

    Attributes:
        centroid: ``(x, y)`` mask centre of mass.
        angle: Rotation in degrees (PIL convention, counter-clockwise) that
            aligns a design's vertical axis with the mask's long axis.
        major_extent: Mask length along its long axis, in pixels.
        minor_extent: Mask width across its long axis, in pixels.
    """

    centroid: tuple[float, float]
    angle: float
    major_extent: float
    minor_extent: float


def design_to_rgba(design: Image.Image, bg_threshold: float = 0.08) -> Image.Image:
    """Convert a tattoo design on a plain background to RGBA ink.

    Designs that already carry transparency (RGBA, LA, or palette images
    with a transparent colour) keep their own alpha. For opaque designs the
    background colour is estimated from the image border (white, black or
    any plain colour); alpha is the distance from it, and the ink colour is
    recovered by un-compositing over it. Monochrome light-on-dark designs
    (e.g. a white logo on black) are inverted, since tattoo ink is dark.
    The result is cropped to the artwork.

    Args:
        design: Tattoo design image.
        bg_threshold: Distance from the background colour (0-1) treated as
            background.

    Returns:
        RGBA image with a transparent background.
    """
    if design.mode in ("RGBA", "LA", "PA") or "transparency" in design.info:
        rgba_design = design.convert("RGBA")
        if np.asarray(rgba_design)[:, :, 3].min() < 255:
            return _crop_to_artwork(rgba_design)

    rgb = pil_to_numpy(design).astype(np.float32) / 255.0
    bg = _background_colour(rgb)
    if bg.mean() < 0.35 and _is_monochrome(rgb, bg):
        rgb, bg = 1.0 - rgb, 1.0 - bg
    distance = np.abs(rgb - bg).max(axis=2)

    # Recover ink colour: observed = ink * a + bg * (1 - a). Using the raw
    # distance as alpha maps grey line work to black ink, as it is tattooed.
    safe_distance = np.maximum(distance, 1e-3)[:, :, np.newaxis]
    ink = np.clip(bg + (rgb - bg) / safe_distance, 0.0, 1.0)

    # Stretch so the strongest ink is fully opaque (faint line art stays visible)
    peak = max(float(np.percentile(distance, 99.5)), bg_threshold + 1e-3)
    alpha = np.clip((distance - bg_threshold) / (peak - bg_threshold), 0.0, 1.0)

    rgba = np.dstack([ink, alpha]) * 255.0
    return _crop_to_artwork(Image.fromarray(rgba.astype(np.uint8), mode="RGBA"))


def has_plain_background(design: Image.Image) -> bool:
    """Return ``True`` if *design* is transparent or on a plain background.

    Designs that fail this (e.g. a photo of a tattoo on skin) can't be cut
    out reliably; their background will be blended in as ink.
    """
    if design.mode in ("RGBA", "LA", "PA") or "transparency" in design.info:
        if np.asarray(design.convert("RGBA"))[:, :, 3].min() < 255:
            return True
    rgb = pil_to_numpy(design).astype(np.float32) / 255.0
    edges = _border_pixels(rgb)
    near_bg = np.abs(edges - _background_colour(rgb)).max(axis=1) < 0.15
    return float(near_bg.mean()) > 0.8


def _border_pixels(rgb: np.ndarray, border: float = 0.02) -> np.ndarray:
    """Return the outer *border* fraction of pixels as an ``(N, 3)`` array."""
    h, w = rgb.shape[:2]
    bh, bw = max(1, int(h * border)), max(1, int(w * border))
    return np.concatenate([
        rgb[:bh].reshape(-1, 3), rgb[-bh:].reshape(-1, 3),
        rgb[:, :bw].reshape(-1, 3), rgb[:, -bw:].reshape(-1, 3),
    ])


def _background_colour(rgb: np.ndarray) -> np.ndarray:
    """Estimate the background colour as the median of the image border."""
    return np.median(_border_pixels(rgb), axis=0)


def _is_monochrome(rgb: np.ndarray, bg: np.ndarray) -> bool:
    """Return ``True`` if the artwork (pixels away from *bg*) is near-grey."""
    art = rgb[np.abs(rgb - bg).max(axis=2) > 0.2]
    if art.size == 0:
        return True
    return float((art.max(axis=1) - art.min(axis=1)).mean()) < 0.15


def _crop_to_artwork(design: Image.Image, min_alpha: int = 64) -> Image.Image:
    """Trim empty margins so sizing applies to the artwork itself.

    Pixels fainter than *min_alpha* (e.g. stock-site watermarks) don't
    extend the crop.
    """
    bbox = design.getchannel("A").point(lambda v: 255 if v > min_alpha else 0).getbbox()
    return design.crop(bbox) if bbox else design


def connected_region(mask: np.ndarray, seed: tuple[float, float]) -> np.ndarray:
    """Keep only the mask component containing *seed* (e.g. one leg of two).

    Falls back to the full mask when *seed* lies outside it.

    Args:
        mask: ``(H, W)`` boolean array.
        seed: ``(x, y)`` pixel inside the wanted component.

    Returns:
        New boolean mask with a single connected component.
    """
    x, y = round(seed[0]), round(seed[1])
    h, w = mask.shape
    if not (0 <= x < w and 0 <= y < h) or not mask[y, x]:
        return mask.copy()
    # .copy(): fromarray may share a read-only buffer that floodfill can't write
    img = Image.fromarray((mask * 255).astype(np.uint8), mode="L").copy()
    ImageDraw.floodfill(img, (x, y), 128)
    return np.asarray(img) == 128


def mask_geometry(
    mask: np.ndarray, at: tuple[float, float] | None = None
) -> MaskGeometry:
    """Estimate the principal axis and extents of a boolean mask via PCA.

    Args:
        mask: ``(H, W)`` boolean array with at least one ``True`` pixel.
        at: Optional ``(x, y)`` point; the minor extent is then measured in a
            band around it (local limb width) instead of across the whole mask.

    Returns:
        :class:`MaskGeometry` describing the mask.
    """
    ys, xs = np.nonzero(mask)
    if xs.size == 0:
        raise ValueError("Mask is empty; cannot place a tattoo.")

    coords = np.stack([xs, ys]).astype(np.float64)
    centroid = coords.mean(axis=1)
    centred = coords - centroid[:, np.newaxis]

    eigvals, eigvecs = np.linalg.eigh(np.cov(centred) + np.eye(2) * 1e-6)
    major = eigvecs[:, int(np.argmax(eigvals))]
    if major[1] < 0:
        major = -major
    minor = np.array([-major[1], major[0]])

    def _extent(proj: np.ndarray) -> float:
        lo, hi = np.percentile(proj, [5, 95])
        return float(hi - lo)

    major_proj = major @ centred
    major_extent = _extent(major_proj)
    minor_proj = minor @ centred
    if at is not None:
        at_major = major @ (np.asarray(at, dtype=np.float64) - centroid)
        band = np.abs(major_proj - at_major) < max(major_extent * 0.05, 2.0)
        if band.sum() >= 10:
            minor_proj = minor_proj[band]

    return MaskGeometry(
        centroid=(float(centroid[0]), float(centroid[1])),
        angle=float(np.degrees(np.arctan2(major[0], major[1]))),
        major_extent=major_extent,
        minor_extent=_extent(minor_proj),
    )


def place_design(
    design_rgba: Image.Image,
    canvas_size: tuple[int, int],
    center: tuple[float, float],
    width: int,
    angle: float = 0.0,
    max_height: int | None = None,
) -> np.ndarray:
    """Scale, rotate and position a design on a transparent canvas.

    Args:
        design_rgba: RGBA tattoo design.
        canvas_size: ``(W, H)`` of the body image.
        center: ``(x, y)`` placement centre on the canvas.
        width: Target design width in pixels (before rotation).
        angle: Counter-clockwise rotation in degrees.
        max_height: Optional cap on design height (before rotation).

    Returns:
        ``(H, W, 4)`` float32 array in ``[0, 1]``.
    """
    w = max(1, int(width))
    h = max(1, int(w * design_rgba.height / design_rgba.width))
    if max_height is not None and h > max_height:
        h = max(1, int(max_height))
        w = max(1, int(h * design_rgba.width / design_rgba.height))

    resized = design_rgba.resize((w, h), Image.Resampling.LANCZOS)
    rotated = resized.rotate(angle, expand=True, resample=Image.Resampling.BICUBIC)

    canvas = Image.new("RGBA", canvas_size, (0, 0, 0, 0))
    x = round(center[0] - rotated.width / 2)
    y = round(center[1] - rotated.height / 2)
    canvas.paste(rotated, (x, y), rotated)

    return np.asarray(canvas, dtype=np.float32) / 255.0


def blend_ink(
    body: Image.Image,
    layer: np.ndarray,
    skin_mask: np.ndarray,
    opacity: float = 0.9,
    edge_feather: float = 2.0,
    ink_spread: float = 0.6,
) -> Image.Image:
    """Multiply-blend a placed tattoo layer onto skin.

    Args:
        body: Body photo.
        layer: ``(H, W, 4)`` float layer from :func:`place_design`.
        skin_mask: ``(H, W)`` boolean skin mask; ink outside it is dropped.
        opacity: Overall ink opacity.
        edge_feather: Gaussian sigma for softening the skin-mask boundary.
        ink_spread: Gaussian radius simulating ink bleeding into skin.

    Returns:
        Composited RGB image.
    """
    body_arr = pil_to_numpy(body).astype(np.float32)

    alpha = layer[:, :, 3] * feather_mask(skin_mask, edge_feather) * opacity
    ink = layer[:, :, :3]

    if ink_spread > 0:
        alpha_img = Image.fromarray((alpha * 255).astype(np.uint8), mode="L")
        alpha = (
            np.asarray(alpha_img.filter(ImageFilter.GaussianBlur(ink_spread)), dtype=np.float32)
            / 255.0
        )

    # Healed ink is never pure black or fully saturated
    ink = 0.08 + 0.85 * ink

    alpha_3d = alpha[:, :, np.newaxis]
    multiplied = body_arr * ink
    blended = body_arr * (1.0 - alpha_3d) + multiplied * alpha_3d
    return numpy_to_pil(blended)
