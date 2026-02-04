from __future__ import annotations

import os
import torch
from torchvision.models.detection import (
    fasterrcnn_mobilenet_v3_large_320_fpn,
    FasterRCNN_MobileNet_V3_Large_320_FPN_Weights,
    ssdlite320_mobilenet_v3_large,
    SSDLite320_MobileNet_V3_Large_Weights,
    retinanet_resnet50_fpn_v2,
    RetinaNet_ResNet50_FPN_V2_Weights,
    fcos_resnet50_fpn,
    FCOS_ResNet50_FPN_Weights,
)

def get_coco_categories() -> list[str]:
    # TorchVision provides category names via the weights metadata
    weights = FasterRCNN_MobileNet_V3_Large_320_FPN_Weights.DEFAULT
    return weights.meta["categories"]

def load_detector(device: str = "cpu", model_type: str = "fasterrcnn") -> torch.nn.Module:
    """Load a torchvision object detector.
    
    Args:
        device: Device to load model on ('cpu' or 'cuda')
        model_type: Model architecture:
            - 'fasterrcnn': FasterRCNN MobileNetV3 320 (complex, may not export well)
            - 'ssd': SSDLite MobileNetV3 320 (simple, ONNX-friendly)
            - 'retinanet': RetinaNet ResNet50 FPN (single-stage, ONNX-friendly)
            - 'fcos': FCOS ResNet50 FPN (anchor-free, ONNX-friendly)
    
    Returns:
        Detection model in eval mode
    """
    use_pretrained = os.getenv("OD_USE_PRETRAINED", "0") == "1"

    if model_type == "ssd":
        if use_pretrained:
            weights = SSDLite320_MobileNet_V3_Large_Weights.DEFAULT
            model = ssdlite320_mobilenet_v3_large(weights=weights)
        else:
            model = ssdlite320_mobilenet_v3_large(weights=None)
    elif model_type == "retinanet":
        if use_pretrained:
            weights = RetinaNet_ResNet50_FPN_V2_Weights.DEFAULT
            model = retinanet_resnet50_fpn_v2(weights=weights)
        else:
            model = retinanet_resnet50_fpn_v2(weights=None)
    elif model_type == "fcos":
        if use_pretrained:
            weights = FCOS_ResNet50_FPN_Weights.DEFAULT
            model = fcos_resnet50_fpn(weights=weights)
        else:
            model = fcos_resnet50_fpn(weights=None)
    else:  # fasterrcnn
        if use_pretrained:
            weights = FasterRCNN_MobileNet_V3_Large_320_FPN_Weights.DEFAULT
            model = fasterrcnn_mobilenet_v3_large_320_fpn(weights=weights)
        else:
            model = fasterrcnn_mobilenet_v3_large_320_fpn(weights=None)

    model.to(device)
    model.eval()
    return model
