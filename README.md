# OD Inference - Object Detection with PyTorch & ONNX

Lightweight object detection inference package using torchvision models, with ONNX export support.

## Features

- 🔥 **PyTorch inference** with multiple detection models
- 📦 **ONNX export** for deployment
- 🎨 **Visual annotations** with bounding boxes and labels
- 🧪 **Test suite** with pytest
- 🐍 **Python 3.10+** required

## Supported Models

| Model | Architecture | ONNX Support | Speed | Accuracy |
|-------|-------------|--------------|-------|----------|
| `fasterrcnn` | FasterRCNN MobileNetV3 | ⚠️ Limited | Medium | High |
| `ssd` | SSDLite MobileNetV3 | ✅ Good | Fast | Medium |
| `retinanet` | RetinaNet ResNet50 FPN | ✅ Good | Medium | High |
| `fcos` | FCOS ResNet50 FPN | ✅ Good | Medium | High |

**Recommended:** Use `retinanet` for best balance of accuracy and ONNX compatibility.

## Setup

### 1. Install Dependencies

```bash
cd test
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
pip install -r requirements.txt
pip install -e .
```

### 2. Fix SSL Certificate Issues (if needed)

If you encounter SSL certificate errors when downloading pretrained weights:

```bash
export SSL_CERT_FILE=/etc/ssl/certs/ca-certificates.crt
```

## Usage

### PyTorch Inference (CLI)

Run detection on an image and save annotated output:

```bash
# Enable pretrained weights
export OD_USE_PRETRAINED=1

# Run inference
python -m od_inference_ci.cli assets/3.jpg --score 0.4 --save assets/3_annotated.jpg
```

**Options:**
- `--device cpu|cuda` - Device to use (default: cpu)
- `--score FLOAT` - Confidence threshold (default: 0.5)
- `--topk INT` - Maximum detections (default: 20)
- `--save PATH` - Save annotated image

### PyTorch Inference (Python API)

```python
from PIL import Image
from od.infer import predict

# Load image
img = Image.open("assets/3.jpg")

# Run detection
dets = predict(img, device="cpu", score_thresh=0.4, top_k=50)

# Print results
for d in dets:
    print(f"Label: {d.label}, Score: {d.score:.3f}, Box: {d.box_xyxy}")
```

### ONNX Export

Export a model to ONNX format:

```bash
# Set environment variables
export OD_USE_PRETRAINED=1
export SSL_CERT_FILE=/etc/ssl/certs/ca-certificates.crt

# Export RetinaNet (recommended)
python -m od.export_cli retinanet retinanet_detector.onnx

# Or export other models
python -m od.export_cli ssd ssd_detector.onnx
python -m od.export_cli fcos fcos_detector.onnx
```

**Python API:**

```python
from od.export_onnx import export_detector_onnx

# Export with specific parameters
export_detector_onnx(
    out_path="my_detector.onnx",
    h=320,
    w=320,
    model_type="retinanet"
)
```

### ONNX Inference

Run inference using the exported ONNX model:

```python
from PIL import Image
from od.onnx_infer import ort_predict

# Load image
img = Image.open("assets/3.jpg")

# Run ONNX inference
dets = ort_predict(
    "retinanet_detector.onnx",
    img,
    score_thresh=0.4,
    top_k=50
)

# Print results
for d in dets:
    print(f"Label: {d.label}, Score: {d.score:.3f}")
```

### Visualization

Draw detections on an image with COCO class names:

```python
from PIL import Image
from od.infer import predict
from od.visualize import draw_detections

# COCO class names
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

# Run detection
img = Image.open("assets/3.jpg")
dets = predict(img, score_thresh=0.4)

# Draw annotations
annotated = draw_detections(img, dets, categories=COCO_CLASSES, color=(255, 0, 0))
annotated.save("output.jpg")
```

## Testing

Run the test suite:

```bash
# Run all tests
pytest -q tests/

# Run specific test
pytest tests/test_infer_smoke.py -v

# With coverage
pytest --cov=od tests/
```

## Environment Variables

| Variable | Values | Description |
|----------|--------|-------------|
| `OD_USE_PRETRAINED` | `0` or `1` | Enable pretrained weights (default: `0`) |
| `SSL_CERT_FILE` | path | Path to SSL certificates (fixes download issues) |

## CI/CD

GitHub Actions workflow is configured in `.github/workflows/ci.yml`:

- ✅ Installs dependencies (CPU-only PyTorch)
- ✅ Runs linting with `ruff`
- ✅ Runs test suite with `pytest`
- ✅ Uses `OD_USE_PRETRAINED=0` to avoid downloads in CI

## Project Structure

```
test/
├── src/
│   ├── od/                      # Main package
│   │   ├── infer.py            # PyTorch inference API
│   │   ├── model.py            # Model loading (4 architectures)
│   │   ├── cli.py              # Original CLI
│   │   ├── export_onnx.py      # ONNX export
│   │   ├── export_cli.py       # Export CLI tool
│   │   ├── onnx_infer.py       # ONNX inference API
│   │   └── visualize.py        # Drawing utilities
│   └── od_inference_ci/        # CI-friendly wrapper
│       ├── cli.py              # CLI with COCO classes
│       └── __init__.py
├── tests/                       # Test suite
├── assets/                      # Sample images
├── pyproject.toml              # Package metadata
├── requirements.txt            # Dependencies
└── README.md                   # This file
```

## Common Issues

### SSL Certificate Error

```bash
export SSL_CERT_FILE=/etc/ssl/certs/ca-certificates.crt
```

### ONNX Runtime Error with FasterRCNN

FasterRCNN has complex control flow that may not work in all ONNX runtimes. Use `retinanet`, `fcos`, or `ssd` instead.

### No Detections Found

- Ensure `OD_USE_PRETRAINED=1` is set
- Lower the `--score` threshold (try 0.3 or 0.2)
- Check that the image loaded correctly

## Quick Reference

```bash
# Setup
pip install -r requirements.txt && pip install -e .

# Inference (PyTorch)
export OD_USE_PRETRAINED=1
python -m od_inference_ci.cli assets/3.jpg --score 0.4 --save output.jpg

# Export to ONNX
export SSL_CERT_FILE=/etc/ssl/certs/ca-certificates.crt
python -m od.export_cli retinanet retinanet.onnx

# Inference (ONNX)
python -c "from od.onnx_infer import ort_predict; from PIL import Image; print(ort_predict('retinanet.onnx', Image.open('assets/3.jpg')))"

# Run tests
pytest -q tests/
```

## License

See project root for license information.
