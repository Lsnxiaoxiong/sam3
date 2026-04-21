"""
SAM3 ONNX Usage Example

This example shows how to use the exported ONNX model.
"""

import numpy as np
import onnxruntime as ort
from PIL import Image


def preprocess_image(image_path: str) -> np.ndarray:
    """Preprocess image for SAM3."""
    image = Image.open(image_path).convert("RGB")
    image = image.resize((1008, 1008), Image.BILINEAR)
    image_np = np.array(image).astype(np.float32) / 255.0
    mean = np.array([0.5, 0.5, 0.5])
    std = np.array([0.5, 0.5, 0.5])
    image_np = (image_np - mean) / std
    image_np = np.transpose(image_np, (2, 0, 1))
    image_np = np.expand_dims(image_np, axis=0)
    return image_np


def extract_features(onnx_model_path: str, image_path: str):
    """Extract features from image using ONNX model."""
    session = ort.InferenceSession(
        onnx_model_path,
        providers=["CUDAExecutionProvider", "CPUExecutionProvider"]
    )
    
    image = preprocess_image(image_path)
    outputs = session.run(None, {"image": image})
    
    features_p2, features_p3, features_p4, features_p5 = outputs
    print(f"Features extracted:")
    print(f"  P2: {features_p2.shape}")
    print(f"  P3: {features_p3.shape}")
    print(f"  P4: {features_p4.shape}")
    print(f"  P5: {features_p5.shape}")
    
    return outputs


if __name__ == "__main__":
    features = extract_features("sam3_image_encoder.onnx", "test_image.jpg")
