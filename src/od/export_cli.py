"""Script to export object detection model to ONNX format."""
from od.export_onnx import export_detector_onnx
import sys

if __name__ == "__main__":
    model_type = sys.argv[1] if len(sys.argv) > 1 else "ssd"
    out_path = sys.argv[2] if len(sys.argv) > 2 else f"{model_type}_detector.onnx"
    
    print(f"Exporting {model_type} to {out_path}")
    export_detector_onnx(out_path, model_type=model_type)
    print("Done!")
