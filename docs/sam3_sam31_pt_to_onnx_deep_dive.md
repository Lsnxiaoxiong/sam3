# SAM3 / SAM3.1 PT 到 ONNX 结构与导出详解

本文说明本仓库中 SAM3、SAM3.1 的 `.pt` checkpoint 是什么、模型输入输出在哪里定义、如何从源码分析模型、为什么要拆分模块、如何导出 ONNX，以及 SAM3 与 SAM3.1 checkpoint 的主要差异。

读者不需要先懂深度学习。可以把模型理解成一条流水线：图片、文字、点、框先被转换成数字矩阵；模型用这些矩阵计算出“每个像素属于目标的概率”；最后把概率图变成黑白掩码。

## 1. `.pt` 文件到底是什么

`.pt` 不是 ONNX，也不一定是一个能直接调用的完整模型对象。它通常只是 PyTorch 保存的权重文件。权重可以理解成模型里每一层的参数表，例如卷积层、注意力层、文本编码层的矩阵。

本仓库的 SAM3 / SAM3.1 checkpoint 是一个字典，真正的模型权重在 `model` 字段里。源码依据在 `sam3/model_builder.py` 的 `_load_checkpoint()`：

```python
ckpt = torch.load(f, map_location="cpu", weights_only=True)
if "model" in ckpt and isinstance(ckpt["model"], dict):
    ckpt = ckpt["model"]
```

所以导出 ONNX 的第一步不是“直接导出 `.pt`”，而是：

1. 用源码重新搭建同样的网络结构。
2. 从 `.pt` 里取出权重。
3. 把权重加载到网络结构中。
4. 用样例输入跑一遍 `forward`。
5. 把这个可执行路径导出为 ONNX。

最小分析脚本：

```python
import torch
from collections import Counter

ckpt = torch.load("sam3.1_multiplex.pt", map_location="cpu", weights_only=True)
state = ckpt["model"] if "model" in ckpt else ckpt

print(type(ckpt))
print(len(state))
print(Counter(k.split(".")[0] for k in state.keys()))
for k, v in list(state.items())[:20]:
    print(k, tuple(v.shape))
```

这一步能告诉你权重里有哪些大模块，例如 `detector.*` 和 `tracker.*`。

## 2. 从哪里开始分析

不要从 `.pt` 文件本身开始猜。正确起点是官方示例和模型构建函数。

图像推理示例通常类似：

```python
from sam3.model_builder import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor

model = build_sam3_image_model(checkpoint_path="sam3.pt")
processor = Sam3Processor(model)
state = processor.set_image(image)
output = processor.set_text_prompt(state=state, prompt="truck")
```

这段代码透露三个关键信息：

1. 模型通过 `build_sam3_image_model()` 创建，不是从 `.pt` 反序列化出完整对象。
2. 推理入口不是直接 `model(image)`，而是 `Sam3Processor`。
3. 图像和文字是分两步处理的：先 `set_image()`，再 `set_text_prompt()`。

因此分析顺序应该是：

1. `README.md` 和 `examples/*.ipynb`：看官方如何调用。
2. `sam3/model_builder.py`：看模型如何被组装、权重如何加载。
3. `sam3/model/sam3_image_processor.py`：看推理流程如何组织。
4. `sam3/model/sam3_image.py`：看核心模型内部模块。
5. `sam31_onnx_lite/export_sam31_lite.py`：看本仓库如何把 PyTorch 路径封装成 ONNX wrapper。
6. `sam31_onnx_lite/infer_sam31_lite.py`：看 ONNX 运行时如何喂输入、取输出、保存掩码。

## 3. SAM3 图像模型的核心结构

`build_sam3_image_model()` 会创建这些大模块：

| 模块 | 作用 | 类比 |
| --- | --- | --- |
| vision backbone | 把图片变成多层视觉特征 | 看图并提取边缘、纹理、语义 |
| text encoder | 把文本 prompt 变成文本特征 | 把“truck”变成数字语义向量 |
| VL backbone | 统一管理视觉和文本特征 | 图文特征仓库 |
| transformer | 让图像、文本、查询互相交流 | 推理核心 |
| geometry encoder | 编码点、框等几何提示 | 把坐标变成模型能懂的提示 |
| segmentation head | 输出 mask | 画出目标区域 |
| interactive predictor | 点/框交互分割路径 | 类似 SAM1/SAM2 的交互式分割 |

`Sam3Image.__init__()` 中实际挂在模型对象上的关键成员包括：

```text
self.backbone
self.geometry_encoder
self.transformer
self.segmentation_head
self.dot_prod_scoring
self.inst_interactive_predictor
```

理解这些成员很重要，因为 ONNX 导出不是导出“模型名字”，而是导出某个 `nn.Module.forward()` 的计算图。

## 4. SAM3 推理流水线

以文本提示分割图片为例，原始 PyTorch 逻辑可以拆成三步。

第一步：图像编码。

```python
state["backbone_out"] = self.model.backbone.forward_image(image)
```

输入是一张经过预处理的图片，形状通常是：

```text
[batch, 3, 1008, 1008]
```

输出是多层视觉特征，例如：

```text
vision_pos_enc_0/1/2
backbone_fpn_0/1/2
```

这些不是最终掩码，而是模型内部“看懂图片”后的中间结果。

第二步：文本编码。

```python
text_outputs = self.model.backbone.forward_text([prompt], device=self.device)
```

输入是文字，例如 `"truck"`。文字会先经过 BPE tokenizer 变成 token id，长度固定为 `TEXT_CONTEXT = 32`：

```text
token_ids: [batch, 32]
```

输出是：

```text
language_features
language_mask
```

第三步：grounding decoder。

图像特征和文本特征一起进入检测/分割头，输出：

```text
pred_logits      # 每个查询的置信度
pred_boxes_xyxy  # 每个查询的框
pred_masks       # 每个查询的低分辨率 mask
```

本仓库实际测试中，文本输出形状为：

```text
pred_logits: [1, 200, 1]
pred_boxes_xyxy: [1, 200, 4]
pred_masks: [1, 200, 288, 288]
```

其中 `200` 可以理解成模型一次提出 200 个候选目标。推理代码会选择分数最高且掩码面积合理的那个。

## 5. 点、框、文本提示的本质区别

三种提示的目标都一样：告诉模型“我要分割哪个东西”。区别在于提示形式不同。

文本提示：

```text
图片 + "truck" -> 找图中像 truck 的目标
```

点提示：

```text
图片 + 一个正点坐标 -> 分割这个点所在的目标
```

框提示：

```text
图片 + 一个矩形框 -> 分割框内目标
```

点和框提示不依赖文本 encoder，而是走交互式 prompt encoder 和 mask decoder。SAM 系列约定：

```text
point label 1 = 正点，表示目标区域
point label 0 = 负点，表示不要的区域
box label 2/3 = 框的左上角和右下角
```

这就是为什么 `sam31_onnx_lite/export_sam31_lite.py` 里会把 box 转成两组 point-like prompt：

```python
box_coords = box_xyxy.reshape(batch_size, -1, 2, 2)
box_labels = [[2, 3]]
prompt_coords = concat(box_coords, point_coords)
prompt_labels = concat(box_labels, point_labels)
```

从哪里知道要这样做？看 `sam3/sam/prompt_encoder.py` 和 SAM 系列 prompt encoder 设计，框在 prompt encoder 内部本质就是两个特殊点。

## 6. 为什么不能直接导出整个模型

理论上可以尝试导出整个 PyTorch 模型，但工程上通常不可行或不好用，原因包括：

1. 官方推理流程不是单个 `forward(image, prompt)`，而是 Python processor 组织的多步流程。
2. `state` 里有嵌套 dict/list，ONNX 不适合表达这种 Python 对象。
3. 文本、点、框、视频走的路径不同，输入输出差异很大。
4. 视频跟踪有跨帧 memory/state，包含大量动态控制逻辑。
5. 整体导出会非常大，每种提示都重复携带 backbone，加载慢、运行慢。
6. 有些 PyTorch 算子或写法对 ONNX 不友好，需要改写 wrapper。

所以正确思路是拆模块。拆模块不是随便拆，而是沿着官方推理流水线的自然边界拆。

## 7. ONNX 模块划分原则

一个好的 ONNX 划分点应该满足：

1. 输入输出都是 Tensor，不是 Python 对象。
2. 模块边界稳定，多个任务可以复用。
3. 输出张量后续确实会被其他模块使用。
4. 单个 ONNX 不要太大，便于部署和调试。
5. 每个模块可以单独用 ONNX Runtime 验证。

本仓库 SAM3.1 Lite 按如下方式拆分：

| ONNX 模块 | 来源 wrapper | 作用 |
| --- | --- | --- |
| `sam31_image_encoder` | `ImageEncoderWrapper` | 图片 -> 视觉特征 |
| `sam31_text_encoder` | `TextEncoderWrapper` | token ids -> 文本特征 |
| `sam31_grounding_decoder` | `GroundingDecoderWrapper` | 图像特征 + 文本特征 -> masks/boxes/scores |
| `sam31_interactive_decoder` | `InteractiveDecoderWrapper` | 图像特征 + 点/框/mask prompt -> 交互 mask |
| `sam31_point_decoder` | interactive decoder alias | 点提示推理入口 |
| `sam31_box_decoder` | interactive decoder alias | 框提示推理入口 |
| `sam31_reference_feature_encoder` | `ReferenceFeatureEncoderWrapper` | 从参考图框区域提取视觉参考特征 |
| `sam31_grounding_decoder_with_reference` | `GroundingDecoderWithReferenceWrapper` | 文本 + 参考图增强 grounding |
| `sam31_video_prompt_decoder` | `VideoPromptDecoderWrapper` | 实验性直接视频 prompt decoder |

实际稳定视频推理没有使用 `sam31_video_prompt_decoder` 作为最终路径，而是逐帧调用：

```text
image_encoder + interactive_decoder
```

原因是当前直接视频 prompt decoder 和 recurrent state 实验路径生成了异常掩码，如全屏、空白或边缘残影；逐帧交互式路径的输出更稳定、可验证。

## 8. 各 ONNX 模块输入输出

### 8.1 `sam31_image_encoder`

输入：

```text
image: float32 [1, 3, 1008, 1008]
```

图片预处理步骤：

1. RGB。
2. resize 到 `1008 x 1008`。
3. 转 float32。
4. 归一化到模型训练时使用的分布。

输出：

```text
vision_pos_enc_0
vision_pos_enc_1
vision_pos_enc_2
backbone_fpn_0
backbone_fpn_1
backbone_fpn_2
sam2_vision_pos_enc_0
sam2_vision_pos_enc_1
sam2_vision_pos_enc_2
sam2_backbone_fpn_0
sam2_backbone_fpn_1
sam2_backbone_fpn_2
```

前 6 个用于文本 grounding。`sam2_*` 用于点/框交互分割。

### 8.2 `sam31_text_encoder`

输入：

```text
token_ids: int64 [batch, 32]
```

输出：

```text
language_mask
language_features
```

这里的 `32` 来自 `TEXT_CONTEXT = 32`。文字不是直接进模型，而是先通过 `SimpleTokenizer` 转成 token id。

### 8.3 `sam31_grounding_decoder`

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

`pred_masks` 是低分辨率 mask。保存图片时需要 resize 回原图大小，再阈值化。

### 8.4 `sam31_interactive_decoder`

输入：

```text
sam2_backbone_fpn_0
sam2_backbone_fpn_1
sam2_backbone_fpn_2
image_pe
point_coords: [1, N, 2]
point_labels: [1, N]
box_xyxy: [1, M, 4]
box_valid_mask: [1, M]
mask_input: [1, 1, 288, 288]
```

输出：

```text
single_masks
single_scores
multi_masks
multi_scores
```

点提示通常选 `multi_scores` 最高的 mask。框提示通常选 single mask。

注意坐标必须从原图坐标缩放到 `1008 x 1008` 模型坐标：

```python
x_model = x_original * 1008 / original_width
y_model = y_original * 1008 / original_height
```

这是很多 ONNX 推理出错的根源。如果点或框没有缩放，模型看到的提示位置就是错的。

### 8.5 视频输出

稳定视频推理当前是逐帧运行：

```text
for frame in frames:
    image_encoder(frame)
    interactive_decoder(point/box)
    save mask
```

输出：

```text
video_high_res_masks: [num_frames, 1, 288, 288]
video_scores: [num_frames]
```

保存时每帧 resize 回对应原始帧大小。

## 9. 从 PT 到 ONNX 的具体导出流程

导出脚本入口：

```text
sam31_onnx_lite/export_sam31_lite.py
```

核心流程：

1. 解析命令行参数。
2. 创建 SAM3.1 image model。
3. 创建 tracker。
4. 加载 checkpoint。
5. patch 不适合 ONNX 的实现。
6. 包装成多个 `Wrapper(nn.Module)`。
7. 构造 sample input。
8. 调用 `torch.onnx.export()`。
9. 保存 `export_meta.json`。

导出命令：

```powershell
C:\major\miniconda3\envs\sam3\python.exe sam31_onnx_lite\export_sam31_lite.py `
  --checkpoint "C:\Users\lsn\.cache\modelscope\hub\models\facebook\sam3___1\sam3.1_multiplex.pt" `
  --output-dir output\sam31_onnx_lite `
  --device cpu
```

`torch.onnx.export()` 需要一个固定样例输入，因为它要沿着这次前向计算“描图”。例如 image encoder 的样例输入是：

```python
sample_image = torch.zeros((1, 3, 1008, 1008), dtype=torch.float32)
```

文本 encoder 的样例输入是：

```python
sample_tokens = tokenizer(["truck"], context_length=32)
```

交互 decoder 的样例输入包括点、框、mask：

```python
sample_point = torch.tensor([[[320.0, 420.0]]])
sample_label = torch.tensor([[1]])
sample_box = torch.tensor([[[80.0, 300.0, 1710.0, 850.0]]])
sample_mask = torch.zeros((1, 1, 288, 288))
```

样例输入不代表真实只能输入这些值。它的作用是告诉 ONNX：这条计算图里有哪些输入、形状大概是什么。

## 10. 为什么需要 wrapper

PyTorch 原始模型的输入输出经常是 dict、list 或复杂对象，而 ONNX 更喜欢：

```text
Tensor -> Tensor
```

所以 wrapper 的任务是把复杂调用包装成简单接口。

例如原始图像编码输出是：

```python
backbone_out = model.backbone.forward_image(image)
```

里面是嵌套字典。ONNX wrapper 会把它拆成平铺 tuple：

```python
return (
    backbone_out["vision_pos_enc"][0],
    backbone_out["vision_pos_enc"][1],
    backbone_out["vision_pos_enc"][2],
    backbone_out["backbone_fpn"][0],
    backbone_out["backbone_fpn"][1],
    backbone_out["backbone_fpn"][2],
)
```

这样 ONNX Runtime 就能按名字拿到每个输出。

再比如 interactive decoder，原始 SAM decoder 内部有 token 拼接、prompt encoder、mask decoder、多 mask 选择等逻辑。wrapper 把这些逻辑固定成一个可导出的 `forward()`。

## 11. ONNX 推理流程

推理脚本入口：

```text
sam31_onnx_lite/infer_sam31_lite.py
```

文本提示：

```powershell
C:\major\miniconda3\envs\sam3\python.exe sam31_onnx_lite\infer_sam31_lite.py `
  --model-dir output\sam31_onnx_lite `
  --mode text `
  --image assets\images\truck.jpg `
  --text-prompt truck `
  --mask-output output\sam31_verify\text_mask.png
```

点提示：

```powershell
C:\major\miniconda3\envs\sam3\python.exe sam31_onnx_lite\infer_sam31_lite.py `
  --model-dir output\sam31_onnx_lite `
  --mode point `
  --image assets\images\truck.jpg `
  --point 900 560 `
  --point-label 1 `
  --mask-output output\sam31_verify\point_mask.png
```

框提示：

```powershell
C:\major\miniconda3\envs\sam3\python.exe sam31_onnx_lite\infer_sam31_lite.py `
  --model-dir output\sam31_onnx_lite `
  --mode box `
  --image assets\images\truck.jpg `
  --box 80 300 1710 850 `
  --mask-output output\sam31_verify\box_mask.png
```

视频提示：

```powershell
C:\major\miniconda3\envs\sam3\python.exe sam31_onnx_lite\infer_sam31_lite.py `
  --model-dir output\sam31_onnx_lite `
  --mode video `
  --video-dir assets\videos\0001 `
  --point 760 470 `
  --point-label 1 `
  --num-frames 8 `
  --mask-dir output\sam31_verify\video_masks
```

## 12. 如何判断导出结果是否正确

不要只看程序是否运行成功。模型输出必须做视觉和数值检查。

建议检查：

1. 掩码不是全黑。
2. 掩码不是全白。
3. 掩码边界大致贴合目标。
4. 点提示时，点应该落在 mask 内。
5. 框提示时，mask 应该主要位于框内或与框高度重合。
6. 视频时，相邻帧 mask 面积和位置不应剧烈跳变。

本仓库验证过的正常结果：

```text
text mask area ≈ 0.2932, bbox ≈ (84, 282, 1708, 846)
point mask area ≈ 0.2974, point_hit = True
box mask area ≈ 0.2976, box_iou ≈ 0.6992
video frame areas ≈ 0.096 - 0.110, point_hit = True
```

异常结果示例：

```text
mask area 接近 0.0  -> 基本空白
mask area 接近 1.0  -> 基本全屏
bbox = (0, 0, width-1, height-1) -> 通常是全屏错误
视频第一帧全屏、后续空白 -> state/memory 路径有问题
```

`output\sam31_verify\video_state_masks` 就是失败实验路径的结果，不能作为正确视频输出。

## 13. SAM3 与 SAM3.1 checkpoint 差异

本地实际统计结果：

| 项目 | SAM3 | SAM3.1 |
| --- | ---: | ---: |
| checkpoint 路径 | `facebook/sam3/sam3.pt` | `facebook/sam3___1/sam3.1_multiplex.pt` |
| 文件大小 | 约 3.45 GB | 约 3.50 GB |
| tensor 数量 | 1465 | 1623 |
| 公共 key | 1130 | 1130 |
| 仅 SAM3 有 | 335 | - |
| 仅 SAM3.1 有 | - | 493 |

SAM3 顶层前缀：

```text
detector: 1156
tracker: 309
```

SAM3.1 顶层前缀：

```text
detector: 1166
tracker: 457
```

主要差异：

1. SAM3 的 tracker 权重多是 `tracker.sam_mask_decoder.*`、`tracker.transformer.*`。
2. SAM3.1 的 tracker 权重集中到 `tracker.model.*`。
3. SAM3.1 新增或重排了 `interactive_sam_mask_decoder`、`interactive_sam_prompt_encoder`、`interactive_obj_ptr_proj` 等交互式路径。
4. SAM3.1 的 detector 新增 `detector.backbone.vision_backbone.interactive_convs.*`。
5. SAM3.1 发布说明强调 Object Multiplex，目标是多对象视频跟踪时共享 memory、减少重复计算。

可以把差异理解为：

```text
SAM3: 图像 grounding + tracker 是基础结构。
SAM3.1: 在 tracker 和交互式/多对象视频路径上做了重构和增强。
```

图像文本 grounding 的 detector 主体大部分相同，所以 `image_encoder + text_encoder + grounding_decoder` 的拆分思路在 SAM3 和 SAM3.1 上基本一致。真正差异集中在点/框交互、视频跟踪和 state/memory 路径。

## 14. SAM3.1 权重映射为什么需要特殊处理

SAM3.1 checkpoint 中一些权重名字与 `build_sam3_image_model(enable_inst_interactivity=True)` 创建出来的模型成员名字不完全一致。

例如 checkpoint 里有：

```text
detector.backbone.vision_backbone.interactive_convs.*
tracker.model.interactive_sam_prompt_encoder.*
tracker.model.interactive_sam_mask_decoder.*
```

但 image model 里交互式 predictor 期望的名字类似：

```text
backbone.vision_backbone.sam2_convs.*
inst_interactive_predictor.model.sam_prompt_encoder.*
inst_interactive_predictor.model.sam_mask_decoder.*
```

因此 `export_sam31_lite.py` 里实现了 `_load_interactive_predictor_weights()`，做权重名映射：

```text
interactive_convs -> sam2_convs
interactive_sam_prompt_encoder -> inst_interactive_predictor.model.sam_prompt_encoder
interactive_sam_mask_decoder -> inst_interactive_predictor.model.sam_mask_decoder
interactive_mask_downsample -> inst_interactive_predictor.model.mask_downsample
interactive_obj_ptr_proj -> inst_interactive_predictor.model.obj_ptr_proj
```

为什么能这样映射？判断依据是：

1. key 名语义对应。
2. tensor shape 对得上。
3. 加载后实际点/框推理输出正常。

工程上不能只看名字，必须同时检查 shape：

```python
filtered = {
    k: v
    for k, v in mapped.items()
    if k in model_state and tuple(model_state[k].shape) == tuple(v.shape)
}
```

## 15. 如何自己继续分析一个未知 `.pt`

如果以后换了新 checkpoint，建议按这个顺序排查。

第一步：看 checkpoint 包装。

```python
ckpt = torch.load(path, map_location="cpu", weights_only=True)
print(type(ckpt))
print(ckpt.keys() if isinstance(ckpt, dict) else "not dict")
```

第二步：统计 key。

```python
state = ckpt["model"] if "model" in ckpt else ckpt
print(len(state))
print(Counter(k.split(".")[0] for k in state))
print(Counter(".".join(k.split(".")[:2]) for k in state))
```

第三步：找到构建函数。

搜索：

```powershell
rg "def build_.*model|checkpoint_path|load_state_dict" sam3
```

第四步：找到推理入口。

搜索：

```powershell
rg "set_image|set_text_prompt|add_geometric_prompt|predict|track" sam3 examples
```

第五步：跑 PyTorch 原始推理，确认 checkpoint 可用。

第六步：用 hook 或打印输出形状确认中间特征。

```python
def dump(x, name="out"):
    if isinstance(x, torch.Tensor):
        print(name, x.shape, x.dtype)
    elif isinstance(x, dict):
        for k, v in x.items():
            dump(v, f"{name}.{k}")
    elif isinstance(x, (list, tuple)):
        for i, v in enumerate(x):
            dump(v, f"{name}[{i}]")
```

第七步：选模块边界。

优先边界：

```text
图片预处理后 -> image encoder
token ids -> text encoder
图像特征 + 文本特征 -> grounding decoder
图像特征 + 点/框 -> interactive decoder
单帧输出 + memory -> video state step
```

第八步：写 wrapper，把输入输出变成 Tensor。

第九步：导出 ONNX。

第十步：用 ONNX Runtime 跑同一张图，与 PyTorch 输出做数值和视觉对比。

## 16. 常见错误和判断方法

坐标没缩放：

```text
现象：点提示分割到奇怪位置，或完全空白。
原因：原图坐标直接喂给 1008 模型坐标。
解决：x *= 1008 / width, y *= 1008 / height。
```

box 维度错：

```text
现象：ONNX Runtime 报 rank mismatch。
原因：模型期望 [1, M, 4] 或 [1, 4]，实际多套了一维。
解决：用 onnxruntime session.get_inputs() 检查真实输入形状。
```

mask 没 resize：

```text
现象：输出只有 288 x 288，看起来和原图对不上。
原因：decoder 输出是低分辨率 mask。
解决：resize 回原图大小再保存。
```

视频帧排序错：

```text
现象：帧顺序变成 0, 1, 10, 100, 11...
原因：字符串排序。
解决：按文件名数字排序。
```

state 视频路径异常：

```text
现象：第一帧全屏，后续空白。
原因：跨帧 memory/state ONNX 化不完整或 prompt/state 坐标不匹配。
解决：先用逐帧 interactive 路径保证功能正确，再继续单独攻克 recurrent state。
```

## 17. 当前仓库的结论

当前可用、已验证的 SAM3.1 ONNX 路径是：

```text
文本：image_encoder + text_encoder + grounding_decoder
点：image_encoder + point_decoder
框：image_encoder + box_decoder
视频：逐帧 image_encoder + interactive_decoder
```

当前不建议作为最终结果使用的路径：

```text
sam31_video_state_exp
sam31_video_prompt_decoder 直接视频路径
```

它们是探索真正跨帧状态传递的实验路径，但目前掩码质量不稳定。

## 18. 一句话总结

`.pt` 是权重，不是部署模型。要把 SAM3/SAM3.1 转成 ONNX，必须先从官方示例找到真实推理路径，再从 `model_builder.py` 恢复 PyTorch 模型，从 processor 和核心类中找出图像、文本、交互、视频各自的数据流，沿着稳定 Tensor 边界拆成多个 wrapper，最后分别导出并用真实图片/视频验证掩码质量。
