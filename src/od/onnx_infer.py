from __future__ import annotations
import numpy as np
import onnxruntime as ort
from PIL import Image
from typing import List

from .infer import Detection


def pil_to_nchw_float(img: Image.Image, size=(320, 320)) -> np.ndarray:
    """Convert PIL image to NCHW float tensor for ONNX input."""
    img = img.convert("RGB").resize(size)
    arr = np.asarray(img, dtype=np.uint8)  # HWC
    x = arr.transpose(2, 0, 1).astype(np.float32) / 255.0  # CHW
    return x[None, ...]  # NCHW


def ort_predict(
    onnx_path: str,
    img: Image.Image,
    score_thresh: float = 0.5,
    top_k: int = 50,
) -> List[Detection]:
    """Run ONNX inference and return filtered detections.
    
    Args:
        onnx_path: Path to ONNX model file
        img: Input PIL image
        score_thresh: Minimum confidence threshold
        top_k: Maximum number of detections to return
    
    Returns:
        List of Detection objects
    """
    sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    x = pil_to_nchw_float(img)
    boxes, scores, labels = sess.run(None, {"input": x})
    
    # Filter by score
    keep = scores >= score_thresh
    boxes = boxes[keep]
    scores = scores[keep]
    labels = labels[keep]
    
    # Apply top_k limit
    if len(boxes) > top_k:
        boxes = boxes[:top_k]
        scores = scores[:top_k]
        labels = labels[:top_k]
    
    # Convert to Detection objects
    dets: List[Detection] = []
    for box, score, label in zip(boxes, scores, labels):
        dets.append(
            Detection(
                box_xyxy=(float(box[0]), float(box[1]), float(box[2]), float(box[3])),
                score=float(score),
                label=int(label),
            )
        )
    return dets
