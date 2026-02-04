from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
import torch
from PIL import Image

from .model import load_detector


@dataclass(frozen=True)
class Detection:
    box_xyxy: Tuple[float, float, float, float]
    score: float
    label: int


def _pil_to_tensor(img: Image.Image) -> torch.Tensor:
    """
    Converts PIL RGB image to float tensor [3,H,W] in [0,1].
    """
    if img.mode != "RGB":
        img = img.convert("RGB")
    arr = np.asarray(img, dtype=np.uint8).copy()  # HWC
    x = torch.from_numpy(arr).permute(2, 0, 1).float() / 255.0
    return x


@torch.no_grad()
def predict(
    img: Image.Image,
    device: str = "cpu",
    score_thresh: float = 0.5,
    top_k: int = 50,
) -> List[Detection]:
    model = load_detector(device=device)
    x = _pil_to_tensor(img).to(device)

    # torchvision detectors expect: list[tensor]
    outputs = model([x])[0]  # dict: boxes, labels, scores

    boxes = outputs["boxes"].detach().cpu()
    labels = outputs["labels"].detach().cpu()
    scores = outputs["scores"].detach().cpu()

    # Filter + top-k
    keep = scores >= float(score_thresh)
    boxes, labels, scores = boxes[keep], labels[keep], scores[keep]

    if boxes.shape[0] > top_k:
        boxes, labels, scores = boxes[:top_k], labels[:top_k], scores[:top_k]

    dets: List[Detection] = []
    for b, l, s in zip(boxes, labels, scores):
        dets.append(
            Detection(
                box_xyxy=(float(b[0]), float(b[1]), float(b[2]), float(b[3])),
                score=float(s),
                label=int(l),
            )
        )
    return dets
