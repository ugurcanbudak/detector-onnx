from PIL import Image
import numpy as np

from od.infer import predict, Detection


def test_detection_contract():
    img = Image.fromarray((np.random.rand(320, 320, 3) * 255).astype("uint8"), mode="RGB")
    dets = predict(img, device="cpu", score_thresh=0.0, top_k=3)

    for d in dets:
        assert isinstance(d, Detection)
        x1, y1, x2, y2 = d.box_xyxy
        assert all(isinstance(v, float) for v in (x1, y1, x2, y2))
        assert isinstance(d.score, float)
        assert isinstance(d.label, int)
