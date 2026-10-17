"""TatVTON command-line interface."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def _parse_point(value: str) -> tuple[int, int]:
    """Parse 'X,Y' string into an (x, y) tuple."""
    parts = value.split(",")
    if len(parts) != 2:
        raise argparse.ArgumentTypeError(
            f"Point must be X,Y (got {value!r})"
        )
    try:
        return int(parts[0]), int(parts[1])
    except ValueError:
        raise argparse.ArgumentTypeError(
            f"Point coordinates must be integers (got {value!r})"
        )


def _parse_bbox(value: str) -> tuple[int, int, int, int]:
    """Parse 'X1,Y1,X2,Y2' string into a bounding-box tuple."""
    parts = value.split(",")
    if len(parts) != 4:
        raise argparse.ArgumentTypeError(
            f"Bbox must be X1,Y1,X2,Y2 (got {value!r})"
        )
    try:
        return int(parts[0]), int(parts[1]), int(parts[2]), int(parts[3])
    except ValueError:
        raise argparse.ArgumentTypeError(
            f"Bbox coordinates must be integers (got {value!r})"
        )


def build_parser() -> argparse.ArgumentParser:
    """Build the CLI argument parser."""
    parser = argparse.ArgumentParser(
        prog="tatvton",
        description="TatVTON — Tattoo Virtual Try-On from the command line.",
    )

    # -- Positional --------------------------------------------------------
    parser.add_argument(
        "body",
        type=Path,
        help="Path to the body image.",
    )
    parser.add_argument(
        "tattoo",
        type=Path,
        help="Path to the tattoo design image.",
    )

    # -- Region prompt (mutually exclusive) --------------------------------
    region = parser.add_mutually_exclusive_group(required=True)
    region.add_argument(
        "--point",
        type=_parse_point,
        action="append",
        dest="points",
        metavar="X,Y",
        help="Point prompt coordinate (repeatable for multiple points).",
    )
    region.add_argument(
        "--bbox",
        type=_parse_bbox,
        metavar="X1,Y1,X2,Y2",
        help="Bounding-box prompt.",
    )

    # -- Output ------------------------------------------------------------
    parser.add_argument(
        "-o", "--output",
        type=Path,
        default=Path("output.png"),
        help="Output image path (default: output.png).",
    )
    parser.add_argument(
        "--save-mask",
        action="store_true",
        help="Also save the mask image (<output>_mask.png).",
    )
    parser.add_argument(
        "--save-raw",
        action="store_true",
        help="Also save the raw inpainted image (<output>_raw.png).",
    )

    # -- Preview mode (CPU, no diffusion) ---------------------------------
    parser.add_argument(
        "--preview",
        action="store_true",
        help="Fast CPU preview: SAM 2 tiny mask + ink blend, no diffusion models.",
    )
    parser.add_argument("--scale", type=float, default=0.8, help="Design width vs. limb width (default: 0.8).")
    parser.add_argument("--rotation", type=float, default=0.0, help="Extra design rotation in degrees (default: 0).")
    parser.add_argument("--opacity", type=float, default=0.9, help="Ink opacity of the placed design (default: 0.9).")

    # -- Config overrides --------------------------------------------------
    parser.add_argument("--resolution", type=int, help="Pipeline resolution (default: 1024).")
    parser.add_argument("--steps", type=int, help="Inference steps (default: 30).")
    parser.add_argument("--strength", type=float, help="AI refinement strength (default: 0.4).")
    parser.add_argument("--guidance-scale", type=float, help="Guidance scale (default: 7.5).")
    parser.add_argument("--ip-adapter-scale", type=float, help="IP-Adapter scale (default: 0.6).")
    parser.add_argument("--seed", type=int, help="Random seed for reproducibility.")
    parser.add_argument("--device", choices=["cuda", "cpu", "mps"], help="Device (default: cuda).")
    parser.add_argument(
        "--offload",
        choices=["auto", "none", "model", "sequential"],
        help="Offload strategy (default: auto = direct GPU if >= 14 GB VRAM, else model).",
    )

    return parser


def main(argv: list[str] | None = None) -> None:
    """CLI entry point."""
    parser = build_parser()
    args = parser.parse_args(argv)

    # -- Validate inputs ---------------------------------------------------
    if not args.body.exists():
        parser.error(f"Body image not found: {args.body}")
    if not args.tattoo.exists():
        parser.error(f"Tattoo image not found: {args.tattoo}")

    # -- Late imports (heavy dependencies) ---------------------------------
    from PIL import Image

    from tatvton import (
        BBoxPrompt,
        PointPrompt,
        RegionPrompt,
        TatVTONConfig,
        TatVTONPipeline,
        TatVTONPreviewPipeline,
    )

    # -- Load images -------------------------------------------------------
    print(f"Loading body image:   {args.body}")
    body_image = Image.open(args.body).convert("RGB")

    print(f"Loading tattoo image: {args.tattoo}")
    tattoo_image = Image.open(args.tattoo).convert("RGB")

    # -- Build region prompt -----------------------------------------------
    region: RegionPrompt
    if args.points:
        region = PointPrompt(coords=args.points)
        print(f"Region: PointPrompt with {len(args.points)} point(s)")
    else:
        region = BBoxPrompt(bbox=args.bbox)
        print(f"Region: BBoxPrompt {args.bbox}")

    # -- Build config ------------------------------------------------------
    overrides: dict = {}
    if args.resolution is not None:
        overrides["resolution"] = args.resolution
    if args.steps is not None:
        overrides["num_inference_steps"] = args.steps
    if args.strength is not None:
        overrides["strength"] = args.strength
    if args.guidance_scale is not None:
        overrides["guidance_scale"] = args.guidance_scale
    if args.ip_adapter_scale is not None:
        overrides["ip_adapter_scale"] = args.ip_adapter_scale
    if args.seed is not None:
        overrides["seed"] = args.seed
    if args.device is not None:
        overrides["device"] = args.device
    if args.offload is not None:
        overrides["offload_strategy"] = args.offload

    # -- Run pipeline ------------------------------------------------------
    print("Initializing pipeline...")
    pipeline: TatVTONPipeline | TatVTONPreviewPipeline
    if args.preview:
        pipeline = TatVTONPreviewPipeline()
    else:
        config = TatVTONConfig(**overrides) if overrides else TatVTONConfig()
        pipeline = TatVTONPipeline(config)

    try:
        print("Running inference...")
        result = pipeline(
            body_image=body_image,
            tattoo_image=tattoo_image,
            region=region,
            scale=args.scale,
            rotation=args.rotation,
            opacity=args.opacity,
        )

        # -- Save outputs --------------------------------------------------
        args.output.parent.mkdir(parents=True, exist_ok=True)
        result.image.save(args.output)
        print(f"Saved result: {args.output}")

        if args.save_mask:
            mask_path = args.output.with_name(f"{args.output.stem}_mask{args.output.suffix}")
            result.mask.save(mask_path)
            print(f"Saved mask:   {mask_path}")

        if args.save_raw:
            raw_path = args.output.with_name(f"{args.output.stem}_raw{args.output.suffix}")
            result.raw_inpainted.save(raw_path)
            print(f"Saved raw:    {raw_path}")

        if not args.preview:
            print(f"Seed used: {result.seed}")

    finally:
        pipeline.unload()

    print("Done.")


if __name__ == "__main__":
    main()
