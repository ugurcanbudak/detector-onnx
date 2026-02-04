from PIL import Image
import numpy as np

from od.infer import predict


def test_predict_runs():
    img = Image.fromarray((np.random.rand(320, 320, 3) * 255).astype("uint8"), mode="RGB")
    dets = predict(img, device="cpu", score_thresh=0.0, top_k=5)
    assert isinstance(dets, list)
    # With random weights we might get 0 or more detections; just validate type.
