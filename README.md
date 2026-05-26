# MAVRYA Interior Design Pipeline

Production Replicate pipeline for AI-powered interior room redesign.

## Architecture

```
Room Photo
    │
    ▼
SegFormer (ADE20K)          ← nvidia/segformer-b5-finetuned-ade-640-640
    │
    ▼
Binary Mask                 ← White=replace furniture, Black=preserve architecture
    │
    ▼
Flux Dev + LoRA + ControlNet ← fermatresearch/flux-controlnet-inpaint
    │
    ▼
Redesigned Room (photorealistic)
```

## Room Elements

| Preserved (Black Mask) | Replaced (White Mask) |
|---|---|
| Walls, ceiling, floor | Furniture, sofas, chairs |
| Windows, doors | Cabinets, counters |
| Stairs, columns | Lighting fixtures |
| Structural elements | Decor, rugs, art |

## Setup

```bash
pip install -r requirements.txt
export REPLICATE_API_TOKEN=your_token
```

## Usage

```bash
python main.py \
  --image room.jpg \
  --lora "r8.im/your-org/mavrya-modern-coastal@sha256:..." \
  --prompt "Modern Coastal living room, ocean-inspired palette"
```

## Swapping LoRAs

Replace `--lora` with any trained MAVRYA style LoRA on Replicate:
```bash
--lora "r8.im/mavrya/quiet-luxury@sha256:..."
--lora "r8.im/mavrya/wabi-sabi@sha256:..."
```

## Validated Room Types

- Empty living room
- Furnished living room
- Kitchen
- Bedroom
- Bathroom

## Inputs / Outputs

| Parameter | Type | Description |
|---|---|---|
| `--image` | file path | Room photo (JPG/PNG) |
| `--lora` | Replicate URL | Trained MAVRYA style LoRA |
| `--prompt` | string | Style description |
| `--strength` | float (0-1) | Inpaint strength (default 0.85) |
| `--steps` | int | Inference steps (default 30) |

**Output:** Replicate CDN URL of redesigned room image.

## Built by

[Dr. Sandeep Grover](https://github.com/Sandyyy123) - ML Engineering
