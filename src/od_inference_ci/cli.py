from __future__ import annotations

import argparse
import os
from PIL import Image, ImageDraw, ImageFont

from od.infer import predict

# COCO class names (91 classes, index 0 is background)
COCO_CLASSES = [
    "__background__", "person", "bicycle", "car", "motorcycle", "airplane", "bus",
    "train", "truck", "boat", "traffic light", "fire hydrant", "N/A", "stop sign",
    "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
    "elephant", "bear", "zebra", "giraffe", "N/A", "backpack", "umbrella", "N/A", "N/A",
    "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball",
    "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket",
    "bottle", "N/A", "wine glass", "cup", "fork", "knife", "spoon", "bowl",
    "banana", "apple", "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza",
    "donut", "cake", "chair", "couch", "potted plant", "bed", "N/A", "dining table",
    "N/A", "N/A", "toilet", "N/A", "tv", "laptop", "mouse", "remote", "keyboard",
    "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator", "N/A", "book",
    "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush"
]


def _get_class_name(label_id: int) -> str:
    """Get COCO class name from label ID."""
    if 0 <= label_id < len(COCO_CLASSES):
        return COCO_CLASSES[label_id]
    return f"class_{label_id}"


def _annotate(img: Image.Image, dets):
    """Draw bounding boxes and labels on the image."""
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.load_default()
    except Exception:
        font = None
    
    for d in dets:
        x1, y1, x2, y2 = d.box_xyxy
        draw.rectangle([(x1, y1), (x2, y2)], outline=(255, 0, 0), width=2)
        class_name = _get_class_name(d.label)
        label_text = f"{class_name}:{d.score:.2f}"
        if font is not None:
            draw.text((x1 + 2, y1 + 2), label_text, fill=(255, 0, 0), font=font)
        else:
            draw.text((x1 + 2, y1 + 2), label_text, fill=(255, 0, 0))
    return img


def main() -> int:
    p = argparse.ArgumentParser(description="OD inference CLI (CI-friendly)")
    p.add_argument("image_path", type=str)
    p.add_argument("--device", type=str, default="cpu")
    p.add_argument("--score", type=float, default=0.5)
    p.add_argument("--topk", type=int, default=20)
    p.add_argument("--save", type=str, default=None, help="Path to save annotated image")
    args = p.parse_args()

    img = Image.open(args.image_path)
    dets = predict(img, device=args.device, score_thresh=args.score, top_k=args.topk)

    if not dets:
        print(f"No detections found (threshold={args.score})")
    else:
        for d in dets:
            class_name = _get_class_name(d.label)
            print(f"{class_name} (id={d.label}) score={d.score:.3f} box={d.box_xyxy}")

    if args.save:
        out_img = _annotate(img.copy(), dets)
        os.makedirs(os.path.dirname(args.save) or ".", exist_ok=True)
        out_img.save(args.save)
        print(f"Saved: {args.save}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
