"""
Semantic segmentation module for interior room photos.
Uses SegFormer-b5-finetuned-ade-640-640 (ADE20K classes).
Generates binary mask: 1=preserve (walls/ceiling/windows/doors), 0=replace (furniture/decor).
"""
from transformers import SegformerImageProcessor, SegformerForSemanticSegmentation
from PIL import Image
import torch
import numpy as np

# ADE20K class IDs to PRESERVE (architectural elements)
PRESERVE_CLASSES = {
    0,   # wall
    2,   # floor
    5,   # ceiling
    8,   # windowpane
    14,  # door
    15,  # stairway
    43,  # column
    48,  # step
    53,  # stairs
    59,  # escalator
    65,  # column
    102, # pillar
}

_processor = None
_model = None

def _load_model():
    global _processor, _model
    if _model is None:
        model_id = "nvidia/segformer-b5-finetuned-ade-640-640"
        _processor = SegformerImageProcessor.from_pretrained(model_id)
        _model = SegformerForSemanticSegmentation.from_pretrained(model_id)
        _model.eval()
    return _processor, _model


def segment_room(image: Image.Image) -> np.ndarray:
    """
    Run semantic segmentation on a room photo.
    Returns per-pixel class ID array (H x W), same size as input image.
    """
    processor, model = _load_model()
    inputs = processor(images=image, return_tensors="pt")
    with torch.no_grad():
        logits = model(**inputs).logits  # (1, num_classes, H/4, W/4)
    upsampled = torch.nn.functional.interpolate(
        logits, size=image.size[::-1], mode="bilinear", align_corners=False
    )
    seg_map = upsampled.argmax(dim=1).squeeze().cpu().numpy()
    return seg_map.astype(np.uint8)


def build_inpaint_mask(seg_map: np.ndarray) -> Image.Image:
    """
    Convert segmentation map to binary inpaint mask.
    White (255) = replace (furniture/decor/unknown).
    Black (0)   = preserve (walls, ceiling, windows, doors, floor).
    """
    mask = np.ones(seg_map.shape, dtype=np.uint8) * 255
    for cls_id in PRESERVE_CLASSES:
        mask[seg_map == cls_id] = 0

    # Morphological cleanup: dilate preserve region slightly to avoid edge bleed
    try:
        from PIL import ImageFilter
        mask_img = Image.fromarray(mask).filter(ImageFilter.MedianFilter(size=5))
    except Exception:
        mask_img = Image.fromarray(mask)
    return mask_img.convert("RGB")
