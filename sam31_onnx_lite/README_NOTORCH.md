# SAM3.1 ONNX Lite No-Torch Inference

`infer_sam31_lite_notorch.py` is a lightweight inference-only API for the
ONNX files exported by `sam31_onnx_lite/export_sam31_lite.py`.

It does not import `torch`, `torchvision`, or `sam3`. It uses:

- `onnxruntime`
- `numpy`
- `Pillow`
- `regex`
- `ftfy`

Supported prompts:

- text
- point
- box

Video and cross-image inference are intentionally excluded.

## Example

```python
from sam31_onnx_lite.infer_sam31_lite_notorch import Sam31LiteNoTorchPredictor

predictor = Sam31LiteNoTorchPredictor("output/sam31_onnx_lite")

text_result = predictor.predict_text("assets/images/truck.jpg", "truck")
point_result = predictor.predict_point("assets/images/truck.jpg", (900, 560))
box_result = predictor.predict_box("assets/images/truck.jpg", (80, 300, 1710, 850))

predictor.save_mask(text_result["binary_mask"], "output/notorch_text_mask.png")
predictor.save_mask(point_result["binary_mask"], "output/notorch_point_mask.png")
predictor.save_mask(box_result["binary_mask"], "output/notorch_box_mask.png")
```

Returned result keys:

- `mask`: resized float mask in original image size.
- `binary_mask`: resized boolean mask in original image size.
- `low_res_mask`: decoder mask before resizing.
- `best_score`: selected mask score/logit.
- `best_query_index`: text mode only.
- `raw_outputs`: raw ONNX outputs for debugging.
