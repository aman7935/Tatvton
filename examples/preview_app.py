"""Interactive CPU tattoo preview.

1. Upload a body photo and a tattoo design.
2. Click the skin in the crosshair view (or type X / Y); a red cross marks
   where the tattoo will go.
3. Press "Generate tattoo". Sliders then update the result live.

Every result is saved as a PNG in ``~/tatvton_results/`` (override with the
``TATVTON_RESULTS_DIR`` environment variable).

Usage:
    pip install gradio
    python examples/preview_app.py   # then open http://127.0.0.1:7860
"""

from __future__ import annotations

import os
from datetime import datetime
from pathlib import Path

import gradio as gr
import numpy as np
from PIL import Image, ImageDraw

from tatvton import PointPrompt, TatVTONPreviewPipeline
from tatvton.postprocessing.ink_blend import has_plain_background

pipe = TatVTONPreviewPipeline()
RESULTS_DIR = Path(os.environ.get("TATVTON_RESULTS_DIR", "~/tatvton_results")).expanduser()


def save_result(image: Image.Image) -> Path:
    """Save *image* as a timestamped PNG in :data:`RESULTS_DIR`."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    path = RESULTS_DIR / f"tattoo_{datetime.now():%Y%m%d_%H%M%S_%f}.png"
    image.save(path)
    return path


def mark_point(body: Image.Image | None, x: float | None, y: float | None) -> Image.Image | None:
    """Return a copy of *body* with a red crosshair where the tattoo will go."""
    if body is None:
        return None
    view = Image.new("RGB", body.size, (128, 128, 128))
    view.paste(body, mask=body.getchannel("A") if body.mode == "RGBA" else None)
    if x is not None and y is not None:
        draw = ImageDraw.Draw(view)
        r = max(body.size) // 40
        w = max(2, r // 4)
        draw.line((x - r, y, x + r, y), fill=(255, 0, 0), width=w)
        draw.line((x, y - r, x, y + r), fill=(255, 0, 0), width=w)
        draw.ellipse((x - r // 2, y - r // 2, x + r // 2, y + r // 2), outline=(255, 0, 0), width=w)
    return view


def on_body_upload(body: Image.Image | None) -> tuple[int | None, int | None, Image.Image | None]:
    """Default the tattoo position to the image centre and show the crosshair."""
    if body is None:
        return None, None, None
    x, y = body.width // 2, body.height // 2
    return x, y, mark_point(body, x, y)


def on_pick(evt: gr.SelectData) -> tuple[int, int]:
    """Fill the X / Y boxes from a click on the crosshair view."""
    return int(evt.index[0]), int(evt.index[1])


def check_design(tattoo: Image.Image | None) -> str:
    """Warn when a design can't be cut out cleanly."""
    if tattoo is not None and not has_plain_background(tattoo):
        return (
            "⚠️ This design doesn't have a plain or transparent background "
            "(e.g. it's a photo), so the background will show up as ink. "
            "For clean results use a design on white, black, a plain colour, "
            "or a transparent PNG."
        )
    return ""


def generate(
    body: Image.Image | None,
    tattoo: Image.Image | None,
    x: float | None,
    y: float | None,
    scale: float,
    rotation: float,
    opacity: float,
    auto_rotate: bool,
) -> tuple[Image.Image, np.ndarray, tuple[int, int], str]:
    """Run SAM 2 on the chosen point, then blend the tattoo."""
    if body is None or tattoo is None:
        raise gr.Error("Upload both a body photo and a tattoo design first.")
    if x is None or y is None:
        raise gr.Error("Click the skin in the crosshair view (or type X and Y) first.")
    point = (int(x), int(y))
    if not (0 <= point[0] < body.width and 0 <= point[1] < body.height):
        raise gr.Error(f"Point {point} is outside the photo ({body.width}x{body.height}).")

    mask = pipe.extract_mask(body.convert("RGB"), PointPrompt(coords=[point]))
    image, saved = render(body, tattoo, mask, point, scale, rotation, opacity, auto_rotate)
    return image, mask, point, f"Done — tattoo placed at {point}. {saved} {check_design(tattoo)}"


def render(
    body: Image.Image | None,
    tattoo: Image.Image | None,
    mask: np.ndarray | None,
    point: tuple[int, int] | None,
    scale: float,
    rotation: float,
    opacity: float,
    auto_rotate: bool,
) -> tuple[Image.Image | None, str]:
    """Blend the tattoo using the cached mask (fast, no model call) and save it."""
    if body is None or tattoo is None or mask is None or point is None:
        return None, ""
    result = pipe(
        body, tattoo, PointPrompt(coords=[point]),
        scale=scale, rotation=rotation, opacity=opacity,
        auto_rotate=auto_rotate, mask=mask,
    )
    return result.image, f"Saved: `{save_result(result.image)}`"


with gr.Blocks(title="TatVTON Preview") as demo:
    gr.Markdown(
        "## TatVTON — tattoo preview (CPU)\n"
        "1. Upload both images  2. **Click the skin** in the crosshair view — "
        "the red cross marks where the tattoo goes  3. Press **Generate tattoo**\n\n"
        f"Every result is saved to `{RESULTS_DIR}`"
    )
    mask_state = gr.State()
    point_state = gr.State()

    with gr.Row():
        body_in = gr.Image(type="pil", image_mode="RGBA", label="1. Body photo")
        tattoo_in = gr.Image(type="pil", image_mode="RGBA", label="1. Tattoo design")

    with gr.Row():
        pick_view = gr.Image(
            type="pil", interactive=False, format="png",
            label="2. Click the skin here (red cross = tattoo position)",
        )
        output = gr.Image(type="pil", image_mode="RGBA", label="Result", format="png")

    with gr.Row():
        x_in = gr.Number(label="X", precision=0)
        y_in = gr.Number(label="Y", precision=0)
        run_btn = gr.Button("3. Generate tattoo", variant="primary", scale=2)
    status = gr.Markdown()

    with gr.Row():
        scale = gr.Slider(0.2, 2.0, value=0.8, step=0.05, label="Size (x limb width)")
        rotation = gr.Slider(-180, 180, value=0, step=1, label="Rotation (°)")
        opacity = gr.Slider(0.3, 1.0, value=0.9, step=0.05, label="Ink opacity")
        auto_rotate = gr.Checkbox(value=True, label="Align with limb")

    body_in.change(on_body_upload, inputs=body_in, outputs=[x_in, y_in, pick_view])
    tattoo_in.change(check_design, inputs=tattoo_in, outputs=status)
    pick_view.select(on_pick, outputs=[x_in, y_in])
    for coord in (x_in, y_in):
        coord.change(mark_point, inputs=[body_in, x_in, y_in], outputs=pick_view)

    run_btn.click(
        lambda: "Running… (first run downloads SAM 2, about 150 MB)", outputs=status
    ).then(
        generate,
        inputs=[body_in, tattoo_in, x_in, y_in, scale, rotation, opacity, auto_rotate],
        outputs=[output, mask_state, point_state, status],
        api_name="generate",
    )

    # Sliders re-blend instantly using the cached mask
    live = [body_in, tattoo_in, mask_state, point_state, scale, rotation, opacity, auto_rotate]
    for slider in (scale, rotation, opacity):
        slider.release(render, inputs=live, outputs=[output, status])
    auto_rotate.change(render, inputs=live, outputs=[output, status])


if __name__ == "__main__":
    demo.launch()
