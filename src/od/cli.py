from __future__ import annotations

import argparse
from PIL import Image

from .infer import predict

def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("image_path", type=str)
    p.add_argument("--device", type=str, default="cpu")
    p.add_argument("--score", type=float, default=0.5)
    p.add_argument("--topk", type=int, default=20)    
    p.add_argument("--save", type=str, default="")
    args = p.parse_args()
    
    img = Image.open(args.image_path)
    dets = predict(img, device=args.device, score_thresh=args.score, top_k=args.topk)

    from .model import get_coco_categories
    categories = get_coco_categories()

    for d in dets:
        name = categories[d.label] if 0 <= d.label < len(categories) else f"id={d.label}"
        print(f"{name} (label={d.label}) score={d.score:.3f} box={d.box_xyxy}")

    if args.save:
        from .visualize import draw_detections
        annotated = draw_detections(img, dets, categories=get_coco_categories())
        annotated.save(args.save)
        print(f"Saved: {args.save}")

    return 0
