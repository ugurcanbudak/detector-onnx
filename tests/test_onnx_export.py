import os
import numpy as np
from PIL import Image
from od_inference_ci.export_onnx import export_detector_onnx
from od_inference_ci.onnx_infer import ort_predict

def test_onnx_export_and_run(tmp_path):
    # Keep CI lightweight: do NOT download pretrained weights
    os.environ["OD_USE_PRETRAINED"] = "0"

    onnx_path = tmp_path / "detector.onnx"
    export_detector_onnx(str(onnx_path), h=320, w=320)

    img = Image.fromarray((np.random.rand(320, 320, 3) * 255).astype("uint8"), mode="RGB")
    boxes, scores, labels = ort_predict(str(onnx_path), img)

    assert boxes.ndim == 2 and boxes.shape[1] == 4
    assert scores.ndim == 1
    assert labels.ndim == 1
