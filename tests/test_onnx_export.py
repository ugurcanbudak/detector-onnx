import os
import numpy as np
from PIL import Image
from od.export_onnx import export_detector_onnx
from od.onnx_infer import ort_predict

def test_onnx_export_and_run(tmp_path):
    # Keep CI lightweight: do NOT download pretrained weights
    os.environ["OD_USE_PRETRAINED"] = "0"

    onnx_path = tmp_path / "detector.onnx"
    export_detector_onnx(str(onnx_path), h=320, w=320)

    img = Image.fromarray((np.random.rand(320, 320, 3) * 255).astype("uint8"), mode="RGB")
    dets = ort_predict(str(onnx_path), img)

    # Check that we get a list of detections (could be empty)
    assert isinstance(dets, list)
    # If we have detections, verify their structure
    for det in dets:
        assert hasattr(det, 'box_xyxy')
        assert hasattr(det, 'score')
        assert hasattr(det, 'label')
        assert len(det.box_xyxy) == 4
