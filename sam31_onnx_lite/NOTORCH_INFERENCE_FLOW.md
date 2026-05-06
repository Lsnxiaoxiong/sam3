# SAM3.1 ONNX Lite No-Torch 推理流程详解

本文说明 `sam31_onnx_lite/infer_sam31_lite_notorch.py` 的执行流程。这个版本只做推理，不导出模型，不依赖 `torch`、`torchvision`、`sam3`，只依赖轻量库：

- `onnxruntime`
- `numpy`
- `Pillow`
- `regex`
- `ftfy`

它支持三种图片提示：

- 文本提示：`predict_text(image, "truck")`
- 点提示：`predict_point(image, (x, y))`
- 框提示：`predict_box(image, (x1, y1, x2, y2))`

不包含视频推理和 cross-image 推理。

## 1. 文件定位

核心文件：

```text
sam31_onnx_lite/infer_sam31_lite_notorch.py
```

说明文件：

```text
sam31_onnx_lite/README_NOTORCH.md
sam31_onnx_lite/NOTORCH_INFERENCE_FLOW.md
```

依赖的 ONNX 目录：

```text
output/sam31_onnx_lite/
```

必须存在的文件包括：

```text
output/sam31_onnx_lite/sam31_image_encoder/sam31_image_encoder.onnx
output/sam31_onnx_lite/sam31_text_encoder/sam31_text_encoder.onnx
output/sam31_onnx_lite/sam31_grounding_decoder/sam31_grounding_decoder.onnx
output/sam31_onnx_lite/sam31_point_decoder/sam31_point_decoder.onnx
output/sam31_onnx_lite/sam31_box_decoder/sam31_box_decoder.onnx
output/sam31_onnx_lite/sam31_interactive_dense_pe.npy
```

## 2. 总体执行流程

创建 predictor：

```python
from sam31_onnx_lite.infer_sam31_lite_notorch import Sam31LiteNoTorchPredictor

predictor = Sam31LiteNoTorchPredictor("output/sam31_onnx_lite")
```

初始化时会做四件事：

1. 保存 `model_dir`。
2. 初始化纯 NumPy tokenizer。
3. 选择 ONNX Runtime execution provider。
4. 读取 `sam31_interactive_dense_pe.npy`。

执行推理时，不会一次性加载所有 ONNX。脚本采用懒加载：

```text
第一次用到某个模型 -> 创建 ort.InferenceSession
后续再次用到 -> 复用缓存 session
```

这样避免启动时加载全部大模型。

## 3. 为什么不依赖 torch

原始 `infer_sam31_lite.py` 依赖 `torch` 的地方主要有两个：

1. 图像预处理用 `torchvision.transforms.v2`。
2. 文本 tokenizer 的 `SimpleTokenizer.__call__()` 返回 `torch.LongTensor`。

no-torch 版本替换为：

```text
Pillow + NumPy 完成图像 resize、归一化、转 NCHW
NumpySimpleTokenizer 返回 np.ndarray[int64]
onnxruntime 执行 ONNX
```

因此这个文件不会 import：

```text
torch
torchvision
sam3
```

验证方式：

```python
import sys
from sam31_onnx_lite.infer_sam31_lite_notorch import Sam31LiteNoTorchPredictor

print("torch" in sys.modules)
print("torchvision" in sys.modules)
```

预期均为 `False`。

## 4. ONNX Runtime Session 管理

`Sam31LiteNoTorchPredictor._session(model_name)` 负责加载 ONNX：

```python
path = self.model_dir / model_name / f"{model_name}.onnx"
ort.InferenceSession(str(path), sess_options=options, providers=self.providers)
```

默认 provider 选择逻辑：

```text
如果可用 CUDAExecutionProvider -> 优先 CUDA
始终追加 CPUExecutionProvider -> CUDA 不可用时回退 CPU
```

也可以显式指定：

```python
predictor = Sam31LiteNoTorchPredictor(
    "output/sam31_onnx_lite",
    providers=["CPUExecutionProvider"],
)
```

session options：

```python
options.enable_mem_pattern = False
options.enable_cpu_mem_arena = False
options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
```

这里偏向稳定和可复现，避免某些图优化改变调试行为。

## 5. 图像输入处理

所有模式都会先调用 `_load_rgb_image()`：

支持输入：

```text
str / Path       -> 从磁盘读取
PIL.Image.Image  -> 转 RGB
np.ndarray       -> HxWx3 或 HxWx4
```

统一输出：

```text
PIL RGB image
```

之后 `_preprocess_image()` 会执行：

1. resize 到 `1008 x 1008`。
2. 转 `float32`。
3. 除以 `255.0`，范围变成 `[0, 1]`。
4. 归一化：`(x - 0.5) / 0.5`，范围大致变成 `[-1, 1]`。
5. 从 HWC 转成 NCHW。
6. 增加 batch 维度。

输出 shape：

```text
image: float32 [1, 3, 1008, 1008]
```

这与 ONNX 导出时的输入一致。

## 6. 文本 tokenizer 流程

文本模式使用 `NumpySimpleTokenizer`。

输入：

```text
"truck"
```

处理步骤：

1. `ftfy.fix_text()` 修复文本编码问题。
2. HTML unescape。
3. 去除多余空格。
4. 转小写。
5. 正则切分成 token。
6. UTF-8 byte 转 unicode 字符。
7. 执行 BPE 合并。
8. 添加 `<start_of_text>` 和 `<end_of_text>`。
9. 截断或补零到长度 `32`。

输出：

```text
token_ids: int64 [1, 32]
```

已验证 no-torch tokenizer 对 `"truck"` 的输出与原始 `sam3.model.tokenizer_ve.SimpleTokenizer` 一致：

```text
[49406, 4629, 49407, 0, 0, ...]
```

## 7. 文本提示推理流程

调用：

```python
result = predictor.predict_text("assets/images/truck.jpg", "truck")
```

执行顺序：

```text
image -> sam31_image_encoder
text -> sam31_text_encoder
image features + text features -> sam31_grounding_decoder
select best mask
resize mask to original image size
return result
```

### 7.1 image encoder

输入：

```text
image: [1, 3, 1008, 1008]
```

输出中，文本 grounding 使用这些 key：

```text
vision_pos_enc_0
vision_pos_enc_1
vision_pos_enc_2
backbone_fpn_0
backbone_fpn_1
backbone_fpn_2
```

### 7.2 text encoder

输入：

```text
token_ids: [1, 32]
```

输出：

```text
language_mask
language_features
```

### 7.3 grounding decoder

输入：

```text
vision_pos_enc_0/1/2
backbone_fpn_0/1/2
language_mask
language_features
```

输出：

```text
pred_logits: [1, 200, 1]
pred_boxes_xyxy: [1, 200, 4]
pred_masks: [1, 200, 288, 288]
```

`200` 表示模型给出 200 个候选目标。脚本不会直接取第一个，而是调用 `_select_best_text_mask()`。

### 7.4 文本 mask 选择逻辑

选择逻辑：

1. 取 `pred_logits[0, :, 0]` 作为候选分数。
2. 计算每个 mask 的前景面积比例。
3. 计算每个 box 的面积比例。
4. 过滤掉明显异常候选：

```text
mask_area > 0.001
mask_area < 0.85
box_area > 0.001
box_area < 0.95
```

5. 在有效候选里取 logit 最高的 mask。
6. 如果没有有效候选，退回取全局 logit 最高的 mask。

返回：

```text
best mask
best_query_index
best_score
```

## 8. 点提示推理流程

调用：

```python
result = predictor.predict_point("assets/images/truck.jpg", (900, 560), point_label=1)
```

执行顺序：

```text
image -> sam31_image_encoder
point original coords -> scale to 1008 model coords
image features + point prompt + dense_pe -> sam31_point_decoder
select best multimask
resize mask to original image size
return result
```

### 8.1 点坐标格式

输入点是原图像素坐标：

```text
(x, y)
```

例如：

```text
(900, 560)
```

`point_label`：

```text
1 = 正点，表示目标在这里
0 = 负点，表示这里不是目标
```

### 8.2 坐标缩放

ONNX 模型内部图像大小固定是 `1008 x 1008`。如果原图不是这个尺寸，点坐标必须缩放。

公式：

```python
x_model = x_original * 1008 / original_width
y_model = y_original * 1008 / original_height
```

如果不缩放，点会落到错误位置，mask 可能全黑、全白或分割到别的目标。

输出 shape：

```text
point_coords: float32 [1, 1, 2]
point_labels: int64 [1, 1]
```

### 8.3 interactive decoder 输入

点提示使用：

```text
sam31_point_decoder.onnx
```

它本质上是 `sam31_interactive_decoder.onnx` 的 alias。

输入：

```text
sam2_backbone_fpn_0
sam2_backbone_fpn_1
sam2_backbone_fpn_2
image_pe
point_coords
point_labels
box_xyxy
box_valid_mask
mask_input
```

点模式下：

```text
point_coords = 有效点
point_labels = [1] 或 [0]
box_xyxy = 全 0
box_valid_mask = False
mask_input = 全 0
```

`image_pe` 来自：

```text
output/sam31_onnx_lite/sam31_interactive_dense_pe.npy
```

它是交互式 decoder 需要的固定位置编码。

### 8.4 点模式 mask 选择

interactive decoder 输出：

```text
single_masks
single_scores
multi_masks
multi_scores
```

点提示下使用 multimask：

```python
best_idx = argmax(multi_scores[0])
mask = multi_masks[0, best_idx]
```

原因是点提示歧义更大，一个点可能对应多个合理区域，SAM 会输出多个候选 mask。

## 9. 框提示推理流程

调用：

```python
result = predictor.predict_box(
    "assets/images/truck.jpg",
    (80, 300, 1710, 850),
)
```

执行顺序：

```text
image -> sam31_image_encoder
box original coords -> scale to 1008 model coords
image features + box prompt + dense_pe -> sam31_box_decoder
select single mask
resize mask to original image size
return result
```

### 9.1 框坐标格式

输入框是原图像素坐标：

```text
(x1, y1, x2, y2)
```

含义：

```text
x1, y1 = 左上角
x2, y2 = 右下角
```

### 9.2 框坐标缩放

公式：

```python
x1_model = x1_original * 1008 / original_width
x2_model = x2_original * 1008 / original_width
y1_model = y1_original * 1008 / original_height
y2_model = y2_original * 1008 / original_height
```

输出 shape：

```text
box_xyxy: float32 [1, 1, 4]
box_valid_mask: bool [1, 1]
```

框模式下：

```text
point_coords = [0, 0]
point_labels = [-1]
box_xyxy = 有效框
box_valid_mask = True
mask_input = 全 0
```

`point_labels=-1` 表示没有有效点。

### 9.3 框模式 mask 选择

框提示下使用 single mask：

```python
mask = single_masks[0, 0]
score = single_scores[0, 0]
```

原因是框已经比点提示更明确，通常不需要多候选选择。

## 10. mask 后处理

decoder 输出的 mask 是低分辨率：

```text
low_res_mask: [288, 288]
```

`_resize_mask_to_image()` 会将它 resize 回原图尺寸：

```python
mask_image.resize(image.size, resample=Image.BILINEAR)
```

返回结果里有：

```text
low_res_mask  # float32 [288, 288]
mask          # float32 [original_height, original_width]
binary_mask   # bool [original_height, original_width]
```

二值化阈值：

```python
binary_mask = mask > 0.0
```

保存 mask：

```python
predictor.save_mask(result["binary_mask"], "output/mask.png")
```

保存时会转成：

```text
foreground = 255
background = 0
```

## 11. 返回结果字段

文本模式返回：

```text
low_res_mask
mask
binary_mask
best_score
best_query_index
raw_outputs
```

点/框模式返回：

```text
low_res_mask
mask
binary_mask
best_score
raw_outputs
```

`raw_outputs` 是原始 ONNX 输出，调试时有用。生产环境如果不需要，可以在调用后丢弃，避免长期保存占内存。

## 12. 示例代码

```python
from sam31_onnx_lite.infer_sam31_lite_notorch import Sam31LiteNoTorchPredictor

predictor = Sam31LiteNoTorchPredictor(
    "output/sam31_onnx_lite",
    providers=["CPUExecutionProvider"],
)

text_result = predictor.predict_text("assets/images/truck.jpg", "truck")
point_result = predictor.predict_point("assets/images/truck.jpg", (900, 560))
box_result = predictor.predict_box("assets/images/truck.jpg", (80, 300, 1710, 850))

predictor.save_mask(text_result["binary_mask"], "output/notorch_text_mask.png")
predictor.save_mask(point_result["binary_mask"], "output/notorch_point_mask.png")
predictor.save_mask(box_result["binary_mask"], "output/notorch_box_mask.png")
```

如果想复用 session，提高多次推理效率，应该复用同一个 predictor 对象：

```python
predictor = Sam31LiteNoTorchPredictor("output/sam31_onnx_lite")

for image_path in image_paths:
    result = predictor.predict_text(image_path, "truck")
```

不要在循环里反复创建 predictor，否则会重复加载 ONNX。

## 13. 已验证结果

验证环境：

```text
C:\major\miniconda3\envs\sam3\python.exe
```

验证输入：

```text
assets/images/truck.jpg
```

验证命令使用 Python API 直接调用，不使用 CLI。

结果：

```text
文本提示 truck:
mask shape = (1200, 1800)
best_query_index = 149
best_score ≈ 0.3473
mask area ≈ 0.2933

点提示 (900, 560):
mask shape = (1200, 1800)
best_score ≈ 0.9752
mask area ≈ 0.2974
point_hit = True

框提示 (80, 300, 1710, 850):
mask shape = (1200, 1800)
best_score ≈ 0.9846
mask area ≈ 0.2978
```

输出文件：

```text
output/sam31_notorch_verify/text_mask.png
output/sam31_notorch_verify/point_mask.png
output/sam31_notorch_verify/box_mask.png
```

这些面积与 torch 版 `infer_sam31_lite.py` 的结果基本一致。

## 14. 常见问题

### 14.1 输出 mask 位置不对

优先检查点/框是否使用原图像素坐标。

正确：

```text
point = (900, 560)
box = (80, 300, 1710, 850)
```

不要提前手动缩放到 1008。API 内部已经会缩放。

### 14.2 输出全黑或全白

可能原因：

- 点坐标不在目标上。
- 框坐标顺序错误。
- 输入图像不是 RGB 内容。
- 使用了不匹配的 ONNX 目录。
- `sam31_interactive_dense_pe.npy` 缺失或来自另一批导出。

### 14.3 每次推理都很慢

不要每张图都新建 predictor。

低效：

```python
for path in images:
    predictor = Sam31LiteNoTorchPredictor("output/sam31_onnx_lite")
    predictor.predict_text(path, "truck")
```

高效：

```python
predictor = Sam31LiteNoTorchPredictor("output/sam31_onnx_lite")
for path in images:
    predictor.predict_text(path, "truck")
```

### 14.4 想强制 CPU

```python
predictor = Sam31LiteNoTorchPredictor(
    "output/sam31_onnx_lite",
    providers=["CPUExecutionProvider"],
)
```

### 14.5 想确认没有加载 torch

```python
import sys
from sam31_onnx_lite.infer_sam31_lite_notorch import Sam31LiteNoTorchPredictor

print("torch" in sys.modules)
print("torchvision" in sys.modules)
```

预期：

```text
False
False
```

## 15. 与 torch 版推理脚本的区别

| 项目 | torch 版 | no-torch 版 |
| --- | --- | --- |
| 文件 | `infer_sam31_lite.py` | `infer_sam31_lite_notorch.py` |
| 图像预处理 | `torchvision.transforms.v2` | `Pillow + NumPy` |
| tokenizer 输出 | `torch.LongTensor` | `np.ndarray[int64]` |
| 依赖 `sam3` 包 | 是 | 否 |
| 支持文本 | 是 | 是 |
| 支持点 | 是 | 是 |
| 支持框 | 是 | 是 |
| 支持视频 | 是 | 否 |
| 支持 cross-image | 是 | 否 |
| CLI | 有 argparse | 无 CLI，API 调用 |

no-torch 版的目标是部署轻量推理，不是替代导出脚本。ONNX 仍然需要先由 `export_sam31_lite.py` 生成。

## 16. 最小依赖清单

如果单独部署 no-torch 推理，需要：

```text
onnxruntime
numpy
Pillow
regex
ftfy
```

如果使用 GPU，还需要安装匹配 CUDA 的 `onnxruntime-gpu`。

不需要：

```text
torch
torchvision
sam3
```

但需要保留 BPE 文件：

```text
sam3/assets/bpe_simple_vocab_16e6.txt.gz
```

如果部署时不带整个 `sam3` 包，可以把这个 gzip 文件复制到任意路径，并在初始化时指定：

```python
predictor = Sam31LiteNoTorchPredictor(
    "output/sam31_onnx_lite",
    bpe_path="path/to/bpe_simple_vocab_16e6.txt.gz",
)
```

## 17. 总结

no-torch 推理流程可以概括为：

```text
图片 -> Pillow/NumPy 预处理 -> image_encoder.onnx -> 图像特征

文本 -> NumPy tokenizer -> text_encoder.onnx -> 文本特征
图像特征 + 文本特征 -> grounding_decoder.onnx -> 文本 mask

点/框 -> 坐标缩放到 1008 -> interactive decoder alias -> 交互 mask

低分辨率 mask -> resize 回原图 -> binary mask -> 保存或返回
```

核心原则是：ONNX 只接收 NumPy Tensor，所有 PyTorch 相关逻辑都在导出阶段完成，推理阶段只保留必要的数据预处理、tokenizer、ONNX Runtime 执行和 mask 后处理。
