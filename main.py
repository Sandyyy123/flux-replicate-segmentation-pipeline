"""
MAVRYA Interior Design Pipeline - Replicate API entry point.

Usage:
    python main.py --image room.jpg --lora <replicate_lora_id> --prompt "Modern Coastal living room"

Pipeline:
    1. Semantic segmentation (SegFormer, ADE20K)  -> binary mask
    2. Flux Dev + LoRA + ControlNet inpaint        -> redesigned room
    3. Return output image URL
"""
import argparse
import os
import replicate
from PIL import Image
from dotenv import load_dotenv
from segmentation import segment_room, build_inpaint_mask
import tempfile

load_dotenv()

INPAINT_MODEL = "fermatresearch/flux-controlnet-inpaint"

DEFAULT_PROMPT_TEMPLATE = (
    "{style_prompt}, photorealistic interior design, professional photography, "
    "8k resolution, natural lighting, architectural digest quality"
)

NEGATIVE_PROMPT = (
    "blurry, distorted walls, floating furniture, unrealistic proportions, "
    "cartoon, illustration, oversaturated"
)


def run_pipeline(
    image_path: str,
    lora_id: str,
    style_prompt: str,
    strength: float = 0.85,
    guidance_scale: float = 7.5,
    num_inference_steps: int = 30,
    seed: int | None = None,
) -> str:
    """
    Full pipeline: image -> segmentation -> mask -> Flux inpaint -> output URL.
    Returns the output image URL from Replicate.
    """
    # Step 1: Load image
    image = Image.open(image_path).convert("RGB")
    original_size = image.size
    print(f"[1/3] Loaded image: {original_size[0]}x{original_size[1]}px")

    # Step 2: Semantic segmentation + binary mask
    seg_map = segment_room(image)
    mask = build_inpaint_mask(seg_map)
    print(f"[2/3] Segmentation complete. Preserve region covers "
          f"{(mask.convert('L').point(lambda p: 0 if p > 127 else 1)).getdata().count(1)} pixels")

    # Step 3: Write mask to temp file for Replicate upload
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp_mask:
        mask.resize(original_size).save(tmp_mask.name)
        mask_path = tmp_mask.name

    # Step 4: Build prompt
    full_prompt = DEFAULT_PROMPT_TEMPLATE.format(style_prompt=style_prompt)

    # Step 5: Run Flux ControlNet Inpaint on Replicate
    print(f"[3/3] Running Flux inpaint on Replicate...")
    with open(image_path, "rb") as img_f, open(mask_path, "rb") as mask_f:
        input_params = {
            "image": img_f,
            "mask": mask_f,
            "prompt": full_prompt,
            "negative_prompt": NEGATIVE_PROMPT,
            "lora_url": lora_id,
            "strength": strength,
            "guidance_scale": guidance_scale,
            "num_inference_steps": num_inference_steps,
        }
        if seed is not None:
            input_params["seed"] = seed

        output = replicate.run(INPAINT_MODEL, input=input_params)

    os.unlink(mask_path)

    output_url = output[0] if isinstance(output, list) else output
    print(f"Done! Output: {output_url}")
    return str(output_url)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MAVRYA Interior Design Pipeline")
    parser.add_argument("--image", required=True, help="Path to room photo")
    parser.add_argument("--lora", required=True, help="Replicate LoRA model URL/ID")
    parser.add_argument("--prompt", required=True, help="Style prompt (e.g. Modern Coastal)")
    parser.add_argument("--strength", type=float, default=0.85)
    parser.add_argument("--guidance", type=float, default=7.5)
    parser.add_argument("--steps", type=int, default=30)
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()

    result_url = run_pipeline(
        image_path=args.image,
        lora_id=args.lora,
        style_prompt=args.prompt,
        strength=args.strength,
        guidance_scale=args.guidance,
        num_inference_steps=args.steps,
        seed=args.seed,
    )
    print(f"\nOutput image URL:\n{result_url}")
