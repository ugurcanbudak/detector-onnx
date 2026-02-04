from __future__ import annotations

from PIL import Image, ImageDraw, ImageFont
from typing import Sequence
from .infer import Detection


def draw_detections(
    img: Image.Image,
    dets: Sequence[Detection],
    categories: list[str] | None = None,
    color: tuple[int, int, int] = (255, 0, 0),
) -> Image.Image:
    """Draw bounding boxes and labels on image.
    
    Args:
        img: Input PIL image
        dets: List of Detection objects
        categories: Optional list of category names (e.g., COCO classes)
        color: RGB color tuple for boxes and text (default: red)
    
    Returns:
        Annotated image
    """
    out = img.convert("RGB").copy()
    draw = ImageDraw.Draw(out)

    # Default font; avoid font dependency
    try:
        font = ImageFont.load_default()
    except Exception:
        font = None

    for d in dets:
        x1, y1, x2, y2 = d.box_xyxy
        draw.rectangle([x1, y1, x2, y2], outline=color, width=2)

        name = str(d.label)
        if categories is not None and 0 <= d.label < len(categories):
            name = categories[d.label]

        text = f"{name} {d.score:.2f}"
        draw.text((x1 + 3, y1 + 3), text, fill=color, font=font)

    return out
