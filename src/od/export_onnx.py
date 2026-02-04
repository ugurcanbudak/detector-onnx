from __future__ import annotations
from pathlib import Path
import torch

from .model import load_detector

class ODWrapper(torch.nn.Module):
    def __init__(self, detector: torch.nn.Module):
        super().__init__()
        self.detector = detector

    def forward(self, x: torch.Tensor):
        # x: [1,3,H,W] float - unbatch for detector
        out = self.detector([x[0]])[0]  # Detector expects list of [3,H,W]
        # Return tensors explicitly (dict can be troublesome in some runtimes)
        return out["boxes"], out["scores"], out["labels"]

def export_detector_onnx(
    out_path: str = "detector.onnx",
    h: int = 320,
    w: int = 320,
    model_type: str = "retinanet",
) -> str:
    """Export detector to ONNX format.
    
    Args:
        out_path: Output ONNX file path
        h: Input height (default 320)
        w: Input width (default 320)
        model_type: Model architecture ('retinanet', 'fcos', 'ssd' recommended)
    
    Returns:
        Path to exported ONNX model
        
    Note:
        Single-stage detectors (RetinaNet, FCOS, SSD) export more reliably than FasterRCNN.
    """
    model = load_detector(device="cpu", model_type=model_type)
    model.eval()
    wrapped = ODWrapper(model)
    print(f"Exporting {model_type} model to ONNX...")

    dummy = torch.randn(1, 3, h, w, dtype=torch.float32)

    out_path = str(Path(out_path))
    
    # Use legacy ONNX exporter for compatibility with torchvision detection models
    with torch.no_grad():
        torch.onnx.export(
            wrapped,
            dummy,
            out_path,
            input_names=["input"],
            output_names=["boxes", "scores", "labels"],
            opset_version=11,
            do_constant_folding=True,
            export_params=True,
            # Force legacy exporter (PyTorch 2.x)
            dynamo=False,
        )
    print(f"Exported ONNX model to: {out_path}")
    return out_path
