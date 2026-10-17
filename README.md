# TatVTON — Tattoo Virtual Try-On

Realistic tattoo compositing on body photos using **SAM 2** mask extraction and **SDXL Inpainting + ControlNet + IP-Adapter** synthesis.

Given a body photo, a tattoo design, and a target region, TatVTON generates a photorealistic result with the tattoo naturally blended onto the skin — preserving skin texture, lighting, and body contours.

**[한국어](./README.ko.md)**

## Architecture

### Inference Pipeline

```mermaid
flowchart TB
    subgraph Input
        BODY[Body Image]
        TATTOO[Tattoo Design]
        REGION[Region Prompt<br/>Point / BBox]
    end

    subgraph Phase1["Phase 1 — Mask Extraction"]
        SAM2[SAM 2<br/>Segment Anything 2]
        MASK[Skin Mask]
        BODY --> SAM2
        REGION --> SAM2
        SAM2 --> MASK
    end

    subgraph Phase1b["Phase 1b — Body Structure (Optional)"]
        DP[DensePose]
        IUV[IUV Map]
        BODY --> DP
        DP --> IUV
    end

    subgraph Phase2["Phase 2 — Preprocessing"]
        RESIZE[Resize to 1024×1024]
        DILATE[Mask Dilation + Feathering]
        CLIP_EMB[CLIP Embedding<br/>224×224]
        MASK --> DILATE
        DILATE --> RESIZE
        BODY --> RESIZE
        TATTOO --> CLIP_EMB
    end

    subgraph Phase3["Phase 3 — Diffusion Inpainting"]
        SDXL[SDXL Inpainting]
        CN[ControlNet<br/>Body Structure Guide]
        IPA[IP-Adapter<br/>Tattoo Style Guide]
        RESIZE --> SDXL
        IUV --> CN
        CN --> SDXL
        CLIP_EMB --> IPA
        IPA --> SDXL
        SDXL --> RAW[Raw Inpainted]
    end

    subgraph Phase4["Phase 4 — Compositing"]
        COMP[Alpha Blending<br/>Mask Feathering]
        RAW --> COMP
        RESIZE --> COMP
        DILATE --> COMP
        COMP --> OUTPUT[Output Image]
    end

    style Phase1 fill:#e8f4fd,stroke:#1e88e5
    style Phase1b fill:#fce4ec,stroke:#e53935
    style Phase2 fill:#fff3e0,stroke:#fb8c00
    style Phase3 fill:#e8f5e9,stroke:#43a047
    style Phase4 fill:#f3e5f5,stroke:#8e24aa
```

### Module Structure

```mermaid
graph LR
    subgraph Entry["Entry Points"]
        CLI[cli.py]
        API["__init__.py<br/>(Python API)"]
    end

    subgraph Core["Core Pipeline"]
        PIPE[TatVTONPipeline<br/>pipeline/tatvton_pipeline.py]
        CFG[TatVTONConfig<br/>config.py]
        TYPES[types.py<br/>PointPrompt, BBoxPrompt<br/>MaskResult, PipelineOutput]
    end

    subgraph Models["Model Management"]
        ML[ModelLoader<br/>models/model_loader.py]
        HUB[Hub Download<br/>models/hub.py]
    end

    subgraph Pre["Preprocessing"]
        SME[SkinMaskExtractor<br/>SAM 2]
        DPE[DensePoseExtractor<br/>detectron2]
        VAL[InputValidator]
    end

    subgraph Engine["Inference Engine"]
        IE[InpaintingEngine<br/>SDXL + ControlNet<br/>+ IP-Adapter]
    end

    subgraph Post["Postprocessing"]
        COMP2[Compositor<br/>Alpha Blending]
    end

    subgraph Utils["Utilities"]
        IMG[image.py]
        MSK[mask.py]
        DEV[device.py]
        IMP[imports.py]
    end

    CLI --> PIPE
    API --> PIPE
    CFG --> PIPE
    PIPE --> ML
    ML --> HUB
    PIPE --> SME
    PIPE --> DPE
    PIPE --> VAL
    PIPE --> IE
    PIPE --> COMP2
    IE --> ML
    SME --> ML

    style Entry fill:#e3f2fd
    style Core fill:#e8f5e9
    style Models fill:#fff3e0
    style Pre fill:#fce4ec
    style Engine fill:#f3e5f5
    style Post fill:#e0f7fa
    style Utils fill:#f5f5f5
```

### Training Pipeline (Colab)

```mermaid
flowchart LR
    subgraph Data["Data Collection"]
        CRAWL[Reddit Crawler<br/>+ CLIP Filter]
        RAW_IMG[5,400+ Tattoo Images]
        CRAWL --> RAW_IMG
    end

    subgraph Label["Labeling"]
        DPOSE[DensePose]
        IUV_MAP[IUV Maps]
        BODY_LABEL[Body Part Labels]
        RAW_IMG --> DPOSE
        DPOSE --> IUV_MAP
        DPOSE --> BODY_LABEL
    end

    subgraph Build["Dataset Build"]
        CAPTION[Caption Generation]
        HF_DS[HuggingFace Dataset<br/>image + conditioning + text]
        BODY_LABEL --> CAPTION
        RAW_IMG --> HF_DS
        IUV_MAP --> HF_DS
        CAPTION --> HF_DS
    end

    subgraph Train["ControlNet Training"]
        SDXL_BASE[SDXL Base Model]
        CN_TRAIN[ControlNet SDXL<br/>Fine-tuning]
        CN_MODEL[Trained ControlNet<br/>Weights]
        SDXL_BASE --> CN_TRAIN
        HF_DS --> CN_TRAIN
        CN_TRAIN --> CN_MODEL
    end

    style Data fill:#e8f4fd,stroke:#1e88e5
    style Label fill:#fce4ec,stroke:#e53935
    style Build fill:#fff3e0,stroke:#fb8c00
    style Train fill:#e8f5e9,stroke:#43a047
```

## Dataset

We provide a curated tattoo image dataset on HuggingFace Hub:

**[rlaope/tatvton-tattoo-raw](https://huggingface.co/datasets/rlaope/tatvton-tattoo-raw)** — 5,400+ tattoo images crawled from Reddit, filtered with CLIP for quality (tattoo-on-skin verification). Covers 6 body parts x 12 styles.

## Installation

```bash
pip install -e .
```

SAM 2 (installed separately):

```bash
pip install "sam-2 @ git+https://github.com/facebookresearch/sam2.git"
```

With DensePose support (optional):

```bash
pip install -e ".[densepose]"
```

Development tools:

```bash
pip install -e ".[dev]"
```

## Quick Start

### Python API

```python
from PIL import Image
from tatvton import TatVTONPipeline, PointPrompt

body = Image.open("body.jpg")
tattoo = Image.open("tattoo.png")

pipe = TatVTONPipeline()
result = pipe(
    body_image=body,
    tattoo_image=tattoo,
    region=PointPrompt(coords=[(300, 400)]),
)
result.image.save("output.png")
```

### CLI

```bash
# Point prompt
tatvton body.jpg tattoo.png --point 300,400 -o output.png

# Bounding box prompt
tatvton body.jpg tattoo.png --bbox 100,150,400,600 -o output.png

# Multiple points + options
tatvton body.jpg tattoo.png --point 300,400 --point 350,420 \
    --steps 20 --strength 0.5 --seed 42 -o output.png

# Save mask and raw inpainted result too
tatvton body.jpg tattoo.png --point 300,400 --save-mask --save-raw
```

You can also run via module:

```bash
python -m tatvton body.jpg tattoo.png --point 300,400
```

### CPU Preview Mode (no GPU)

A fast preview that runs on CPU in seconds: SAM 2 *tiny* finds the skin
region, then the design is aligned to the limb and multiply-blended so skin
texture and lighting show through. No diffusion models are downloaded, and
the design is reproduced exactly (less photorealistic than full mode).

```bash
pip install "sam-2 @ git+https://github.com/facebookresearch/sam2.git"  # SAM2_BUILD_CUDA=0 on CPU
tatvton body.jpg tattoo.png --preview --point 300,400 --scale 0.8 --rotation 0 --opacity 0.9

# Interactive app: click the skin to place the tattoo
pip install gradio
python examples/preview_app.py
```

```python
from tatvton import TatVTONPreviewPipeline, PointPrompt

pipe = TatVTONPreviewPipeline()
result = pipe(body, tattoo, PointPrompt(coords=[(300, 400)]), scale=0.8)
```

### CLI Options

| Option | Description | Default |
|--------|-------------|---------|
| `body` | Path to body image (required) | - |
| `tattoo` | Path to tattoo design image (required) | - |
| `--point X,Y` | Point prompt (repeatable) | - |
| `--bbox X1,Y1,X2,Y2` | Bounding box prompt | - |
| `-o, --output` | Output path | `output.png` |
| `--resolution` | Pipeline resolution | 1024 |
| `--steps` | Inference steps | 30 |
| `--strength` | AI refinement strength (lower keeps the design more exact) | 0.4 |
| `--guidance-scale` | Guidance scale | 7.5 |
| `--ip-adapter-scale` | IP-Adapter scale | 0.6 |
| `--seed` | Random seed | random |
| `--device` | cuda / cpu / mps | cuda |
| `--offload` | none / model / sequential | model |
| `--save-mask` | Save mask image | - |
| `--save-raw` | Save raw inpainted result | - |

## Full Realistic Pipeline (GPU)

`TatVTONPipeline` works as **composite, then refine**: the exact design is
placed on the SAM 2 skin mask, then SDXL inpainting repaints only the tattoo
area at low strength (IP-Adapter sees the design), so the ink looks settled
into the skin without the design being redrawn. Every default model is public:

| Model | Hub ID | Role |
|-------|--------|------|
| SAM 2 large | `facebook/sam2-hiera-large` | Skin mask |
| SDXL inpainting | `diffusers/stable-diffusion-xl-1.0-inpainting-0.1` | Realistic refinement |
| SDXL VAE fp16 fix | `madebyollin/sdxl-vae-fp16-fix` | Avoids black/NaN images in fp16 |
| IP-Adapter Plus ViT-H | `h94/IP-Adapter` (`sdxl_models/`, encoder `models/image_encoder`) | Keeps the design style |
| ControlNet (optional) | `controlnet_model_id=None` by default | Body-structure guide |

Needs a GPU (~10 GB VRAM with model offload; a Colab T4 works). The easiest
way to run it is [`notebooks/tatvton_full_pipeline_colab.ipynb`](notebooks/tatvton_full_pipeline_colab.ipynb).

## Configuration

Three levels of configuration:

| Level | Method | Example |
|-------|--------|---------|
| Defaults | `TatVTONPipeline()` | 12 GB GPU optimised |
| Config | `TatVTONPipeline(TatVTONConfig(resolution=768))` | Session-wide |
| Per-call | `pipe(..., strength=0.95, seed=42)` | Call-specific |

## Requirements

- Python 3.9+
- PyTorch 2.0+
- NVIDIA GPU with 12+ GB VRAM (recommended)

## License

MIT
