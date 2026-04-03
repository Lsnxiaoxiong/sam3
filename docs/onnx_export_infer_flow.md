# `export_onnx.py` 与 `infer_onnx.py` 执行流程说明

本文说明仓库根目录下两个脚本的职责、执行顺序、输入输出和跨图特征传递链路：

- `export_onnx.py`：把 PyTorch 版 SAM3 模型拆分并导出成多个 ONNX 子图。
- `infer_onnx.py`：加载导出的 ONNX 子图，完成单图、多图、交互式和跨图参考特征推理。

---

## 1. 整体关系

这两个脚本不是“一次导出一个完整大模型，再一次性推理”的设计，而是把能力拆成多个独立模块：

- 图像编码：`sam3_image_encoder.onnx`
- 文本编码：`sam3_text_encoder.onnx`
- 参考框特征提取：`sam3_reference_feature_encoder.onnx`
- 普通 grounding 解码：`sam3_grounding_decoder.onnx`
- 带 reference 的 grounding 解码：`sam3_grounding_decoder_with_reference.onnx`
- 交互式解码：`sam3_interactive_decoder.onnx`
- 视频单步跟踪：`sam3_video_tracking_step.onnx`

`infer_onnx.py` 按推理模式组合调用这些子图，而不是只调用单个 ONNX 文件。

---

## 2. `export_onnx.py` 执行流程

### 2.1 目标

`export_onnx.py` 的目标是把原始 SAM3 PyTorch 模型包装成一组更稳定、边界更清晰的 ONNX 接口，避免直接导出完整原模型时出现：

- RoPE 复数运算不兼容
- 某些 resize/插值行为不稳定
- 交互式和视频跟踪接口过于耦合
- 跨图 reference 特征链路不易单独复用

因此脚本先对模型内部实现做若干“导出友好化 patch”，再构造 wrapper，最后逐个导出。

### 2.2 主入口

入口在 `export_onnx.py:1064` 的 `main()`。

执行顺序如下：

1. 解析参数 `parse_args()`
2. 校验导出约束
3. 对模型做导出前 patch
4. 构建 PyTorch 模型
5. 构建各个 wrapper
6. 准备示例输入张量
7. 预跑部分模块拿到样例中间特征
8. 调用 `_export()` 逐个导出 ONNX
9. 输出导出的文件路径

### 2.3 参数解析

`parse_args()` 位于 `export_onnx.py:957`，关键参数如下：

- `--checkpoint`：SAM3 权重路径，必填
- `--output-dir`：ONNX 输出目录
- `--device`：导出设备，默认 `cpu`
- `--opset`：ONNX opset，默认 `17`
- `--modules`：选择导出哪些模块，默认 `all`

支持的模块名：

- `image_encoder`
- `text_encoder`
- `reference_feature_encoder`
- `grounding_decoder`
- `grounding_decoder_with_reference`
- `video_tracking_step`
- `interactive_decoder`

### 2.4 导出前 patch

`main()` 一开始会调用三个 patch 函数：

- `patch_vitdet_rope_for_export()`：处理 ViTDet 注意力里的 RoPE 导出
- `patch_tracker_rope_for_export()`：处理 tracker 中的 RoPE 导出
- `patch_tracker_resizes_for_export()`：处理 tracker 路径中 resize / SAM head 的导出稳定性

这些 patch 的目的不是改变模型语义，而是把不利于 ONNX 导出的实现改写成 ONNX Runtime 更容易接受的形式。

### 2.5 模型构建

`main()` 中调用：

- `build_sam3_image_model(...)`：构建图像侧模型
- `build_sam3_video_model(...)`：如果需要导出 `video_tracking_step`，额外构建视频模型

之后统一：

- `model.float()`
- `model.eval()`
- 为 RoPE 和 decoder 准备缓存

### 2.6 wrapper 构建

导出不是直接把原始模型整个丢给 `torch.onnx.export()`，而是先包装成职责单一的 wrapper：

- `ImageEncoderWrapper`
  - 输入：`image`
  - 输出：多层 `vision_pos_enc_*`、`backbone_fpn_*`，以及 SAM2 路径对应的特征

- `TextEncoderWrapper`
  - 输入：`tokens`
  - 输出：`language_mask`、`language_features`

- `ReferenceFeatureEncoderWrapper`，定义在 `export_onnx.py:420`
  - 输入：`backbone_fpn_0`、`reference_boxes_xyxy`
  - 内部通过 `roi_align` 从最高分辨率 FPN 特征图提取 RoI 特征
  - 输出：`reference_embedding`

- `GroundingDecoderWrapper`
  - 负责普通文本 grounding 解码

- `GroundingDecoderWithReferenceWrapper`，定义在 `export_onnx.py:450`
  - 在普通 grounding decoder 前额外接收：
    - `reference_features`
    - `reference_valid_mask`
    - `reference_weight`
  - 它会把多个 reference 特征做有效项平均，形成 `reference_bias`
  - 再把该 bias 加到 `language_features` 上
  - 最后走普通 grounding 解码流程

- `InteractiveDecoderWrapper`，定义在 `export_onnx.py:500`
  - 用于点/框/掩码交互式分割

- `VideoTrackingStepWrapper`，定义在 `export_onnx.py:640`
  - 只负责视频跟踪的单步状态更新

### 2.7 示例输入准备

导出 ONNX 需要示例输入，因此脚本在 `main()` 中构造一组样例张量：

- 随机图像：`sample_image`
- 文本 token：`sample_tokens`
- box / point / mask 输入
- reference 相关输入
- video tracking 的 memory / object pointer 相关输入

这些样例只用于导出图的 tracing，不代表真实推理数据。

### 2.8 预跑中间特征

为了减少重复前向，脚本会先根据导出需求预跑一部分模块：

- 若需要图像相关导出，就先运行 `image_encoder(sample_image)`
- 若需要文本相关导出，就先运行 `text_encoder(sample_tokens)`

之后，decoder 类模块直接复用这些中间输出做导出输入。

### 2.9 动态轴定义

`_dynamic_axes()` 位于 `export_onnx.py:983`，为多种输入输出声明动态维度，比如：

- batch 维
- box 数量维
- point 数量维
- reference box 数量维

这样导出的 ONNX 在推理时不要求固定 box 数量或 point 数量。

### 2.10 统一导出函数

`_export()` 位于 `export_onnx.py:1028`，本质上是对 `torch.onnx.export()` 的封装，统一处理：

- 输出路径创建
- `input_names`
- `output_names`
- `dynamic_axes`
- `external_data=True`

这里启用 `external_data=True`，意味着较大的 ONNX 可能拆分出 `.onnx` 以外的外部权重数据文件。

### 2.11 各模块导出顺序

`main()` 中按条件导出：

1. `image_encoder`
2. `text_encoder`
3. `reference_feature_encoder`
4. `grounding_decoder`
5. `grounding_decoder_with_reference`
6. `interactive_decoder`
7. `video_tracking_step`

其中 `interactive_decoder` 导出后，还会额外保存：

- `sam3_interactive_dense_pe.npy`

这个文件在 `infer_onnx.py` 的交互式模式中会直接加载使用。

### 2.12 导出结果

`build_artifacts()` 位于 `export_onnx.py:1051`，统一定义输出文件名：

- `sam3_image_encoder.onnx`
- `sam3_text_encoder.onnx`
- `sam3_reference_feature_encoder.onnx`
- `sam3_grounding_decoder.onnx`
- `sam3_grounding_decoder_with_reference.onnx`
- `sam3_video_tracking_step.onnx`
- `sam3_interactive_decoder.onnx`
- `sam3_interactive_dense_pe.npy`

---

## 3. `infer_onnx.py` 执行流程

### 3.1 目标

`infer_onnx.py` 的目标是基于导出的 ONNX 子图完成推理，支持三类场景：

- grounding：文本驱动检测 + mask 输出
- interactive：点/框/低分辨率 mask 驱动交互式分割
- grounding + reference：跨图参考特征传递

### 3.2 主入口

入口在 `infer_onnx.py:484` 的 `main()`。

逻辑分两条：

- `mode == grounding`：走 `_run_grounding_sequence()`
- `mode == interactive`：只对单张图走 `_interactive_inference()`

### 3.3 参数解析

`parse_args()` 位于 `infer_onnx.py:115`。

关键参数：

- `--model-dir`：ONNX 模型目录
- `--image` / `--images`：单图或多图输入
- `--mode`：`grounding` 或 `interactive`
- `--text-prompt`：grounding 文本提示
- `--grounding-boxes` / `--grounding-box-labels`：辅助 grounding box
- `--reference-image` / `--reference-boxes` / `--reference-weight`：显式参考图
- `--point-coords` / `--point-labels` / `--box-prompt` / `--mask-input`：交互式提示
- `--score-threshold`：grounding 结果筛选阈值

约束：

- `--image` 会被规范化成单元素 `args.images`
- `interactive` 模式只允许单图

### 3.4 ONNX Runtime Session 初始化

`_session()` 位于 `infer_onnx.py:144`。

这里做了两件事：

1. 创建 `SessionOptions`
2. 关闭：
   - `enable_mem_pattern`
   - `enable_cpu_mem_arena`

同时 `_providers()` 会优先选择：

- `CUDAExecutionProvider`
- `CPUExecutionProvider`

并为 CUDA provider 设置：

- `arena_extend_strategy = kSameAsRequested`
- `cudnn_conv_algo_search = DEFAULT`

这些配置的目的，是降低多图顺序推理时显存/内存池持续膨胀的问题。

### 3.5 输入预处理

常见预处理函数：

- `_load_rgb_image()`：安全加载 RGB 图像
- `_resize_for_encoder()`：把图像 resize 到 `1008x1008`
- `_scaled_xyxy_to_model_space()`：把像素框映射到模型输入尺度
- `_scaled_points_to_model_space()`：把点坐标映射到模型输入尺度
- `_xyxy_pixels_to_cxcywh_normalized()`：把像素框转成归一化 `cx, cy, w, h`
- `_upsample_masks()`：把 decoder 输出的 mask logits 放大回原图尺寸

### 3.6 图像编码

`_run_image_encoder()`：

1. 调用 `_resize_for_encoder()`
2. 输入 `sam3_image_encoder.onnx`
3. 拿到字典形式的输出：
   - `vision_pos_enc_*`
   - `backbone_fpn_*`
   - 以及 interactive / tracking 路径需要的 `sam2_*`

### 3.7 文本编码

`_run_text_encoder()`：

1. 通过 `tokenize()` 把文本转成 token
2. 输入 `sam3_text_encoder.onnx`
3. 输出：
   - `language_mask`
   - `language_features`

### 3.8 reference 特征提取

`_extract_reference_state()` 位于 `infer_onnx.py:179`。

它的流程是：

1. 先对参考图跑一次 image encoder
2. 把参考框从原图像素坐标映射到 `1008x1008` 模型坐标
3. 调用 `sam3_reference_feature_encoder.onnx`
4. 取回 `reference_embedding`
5. 组织成 `ReferenceState`

`ReferenceState` 包含：

- `features`
- `valid_mask`
- `weight`
- `source`

其中：

- `features` 的形状是 `[num_reference_boxes, batch, feat_dim]`
- 当前实现里 batch 固定为 `1`

### 3.9 grounding 单步推理

`_grounding_inference()` 位于 `infer_onnx.py:203`。

它是 grounding 模式的核心单步函数，流程如下：

1. 跑 image encoder
2. 跑 text encoder
3. 解析 `--grounding-boxes`
4. 组织 decoder 输入 `feeds`
5. 若存在 `reference_state`，追加：
   - `reference_features`
   - `reference_valid_mask`
   - `reference_weight`
6. 调用 grounding decoder
7. 根据 `score_threshold` 筛选结果
8. 把归一化 box 还原到原图像素坐标
9. 把 mask logits 上采样到原图尺寸

输出：

- `boxes_xyxy`
- `scores`
- `masks`
- `caption_prefix`

说明：

- 如果没有任何结果通过阈值，会返回空数组
- 当前实现里，普通 grounding 和带 reference grounding 只差在 decoder session 不同，以及是否额外注入 reference 输入

### 3.10 多图顺序 grounding

`_run_grounding_sequence()` 位于 `infer_onnx.py:329`。

这是多图和跨图特征传递的主控流程。

#### 初始化阶段

先创建：

- `image_session`
- `text_session`
- `grounding_session`
- `grounding_with_reference_session`（如果文件存在）
- `reference_session`（如果文件存在）

然后解析：

- `reference_boxes`
- `reference_image`

如果用户显式给了 `--reference-image + --reference-boxes`，会先构造一个初始 `reference_state`。

#### 每张图的循环阶段

对 `args.images` 逐张处理：

1. 加载当前图
2. 判断当前图是否应该带 reference 推理
3. 选择使用：
   - `sam3_grounding_decoder.onnx`
   - 或 `sam3_grounding_decoder_with_reference.onnx`
4. 调用 `_grounding_inference()`
5. 可视化并保存结果

#### 首图 reference 自动提取

如果是多图模式，且用户没有显式给 `--reference-image`，脚本会在首图推理后自动提 reference：

- 若提供了 `--reference-boxes`，优先用这些框提特征
- 否则自动选首图得分最高的目标作为 reference

提取出的 `reference_state` 会用于后续图片。

这就是“examples 示例中跨图特征信息传递”的 ONNX 版实现方式。

### 3.11 interactive 推理

`_interactive_inference()` 位于 `infer_onnx.py:293`。

流程如下：

1. 加载 `sam3_image_encoder.onnx`
2. 加载 `sam3_interactive_decoder.onnx`
3. 对输入图跑 image encoder
4. 解析 point / box / mask 输入
5. 从 `sam3_interactive_dense_pe.npy` 加载 `image_pe`
6. 调 interactive decoder
7. 根据 `--multimask-output` 选择单 mask 或多 mask 输出
8. 把低分辨率 mask 上采样回原图

interactive 模式不走 text encoder，也不走 reference 特征链路。

### 3.12 可视化与保存

`_visualize()` 位于 `infer_onnx.py:464`。

主要做三件事：

1. 把 `xyxy` 转成 `imgviz` 需要的 `yxyx`
2. 调用 `imgviz.instances2rgb()`
3. 保存渲染后的结果图

单图默认保存到 `--output`。

多图模式保存到：

- `--output-dir/<stem>_onnx_result.jpg`

---

## 4. 跨图 reference 推理链路

这是当前最容易混淆的部分，单独拆开说明。

### 4.1 导出阶段

`export_onnx.py` 导出两个关键模块：

1. `sam3_reference_feature_encoder.onnx`
   - 负责从参考图的 `backbone_fpn_0` + 参考框中提取 RoI 特征

2. `sam3_grounding_decoder_with_reference.onnx`
   - 负责把 reference 特征融合进语言特征后再做 grounding 解码

### 4.2 推理阶段

`infer_onnx.py` 的 reference 流程：

1. 首图或参考图先提 reference box
2. 用 `sam3_reference_feature_encoder.onnx` 得到 `reference_embedding`
3. 包装成：
   - `reference_features`
   - `reference_valid_mask`
   - `reference_weight`
4. 在后续图的 grounding decoder 输入中加入这些字段
5. `sam3_grounding_decoder_with_reference.onnx` 内部把 reference bias 加到 `language_features`
6. 再执行普通 grounding 解码

### 4.3 与 `infer_onnx_with_reference.py` 的区别

当前 `infer_onnx.py` 走的是“正式 ONNX 导出链路”：

- reference 特征由 `sam3_reference_feature_encoder.onnx` 产生
- feature 融合由 `sam3_grounding_decoder_with_reference.onnx` 完成

不是旧脚本那种“在 Python 里手动改语言特征再送入 decoder”的旁路方式。

---

## 5. 文件依赖关系

### 5.1 `infer_onnx.py` 的最小 grounding 依赖

普通 grounding 至少需要：

- `sam3_image_encoder.onnx`
- `sam3_text_encoder.onnx`
- `sam3_grounding_decoder.onnx`

### 5.2 带 reference grounding 额外依赖

如果要做跨图特征传递，还需要：

- `sam3_reference_feature_encoder.onnx`
- `sam3_grounding_decoder_with_reference.onnx`

### 5.3 interactive 额外依赖

interactive 模式需要：

- `sam3_image_encoder.onnx`
- `sam3_interactive_decoder.onnx`
- `sam3_interactive_dense_pe.npy`

---

## 6. 常见执行示例

### 6.1 导出全部模块

```bash
conda run -n sam3 python export_onnx.py ^
  --checkpoint C:\path\to\sam3.pt ^
  --output-dir output ^
  --device cpu ^
  --modules all
```

### 6.2 单图 grounding

```bash
conda run -n sam3 python infer_onnx.py ^
  --model-dir output ^
  --image assets/images/groceries.jpg ^
  --mode grounding ^
  --text-prompt "red light" ^
  --output output/onnx_result.jpg
```

### 6.3 多图跨图 reference，首图自动提 reference

```bash
conda run -n sam3 python infer_onnx.py ^
  --model-dir output ^
  --images assets/videos/0001/9.jpg assets/videos/0001/14.jpg assets/videos/0001/20.jpg assets/videos/0001/49.jpg ^
  --mode grounding ^
  --text-prompt "white T-shirt" ^
  --output-dir output/onnx_seq
```

### 6.4 多图跨图 reference，首图使用指定参考框

```bash
conda run -n sam3 python infer_onnx.py ^
  --model-dir output ^
  --images assets/videos/0001/9.jpg assets/videos/0001/14.jpg assets/videos/0001/20.jpg assets/videos/0001/49.jpg ^
  --mode grounding ^
  --text-prompt "white T-shirt" ^
  --reference-boxes "600,355,866,510" ^
  --output-dir output/onnx_seq
```

### 6.5 interactive 推理

```bash
conda run -n sam3 python infer_onnx.py ^
  --model-dir output ^
  --image assets/images/groceries.jpg ^
  --mode interactive ^
  --point-coords "320,240" ^
  --point-labels "1" ^
  --box-prompt "200,120,520,460" ^
  --output output/interactive_result.jpg
```

---

## 7. 当前实现的边界

### 7.1 `export_onnx.py`

- `video_tracking_step` 当前要求 `--device cpu`
- ONNX 导出依赖 tracing 样例输入，若未来 wrapper 接口变化，需要同步修改样例张量

### 7.2 `infer_onnx.py`

- `interactive` 模式目前只支持单图
- 多图 reference 只缓存一份当前 reference 特征，不维护更复杂的多 reference 库
- reference 融合方式由导出的 `grounding_decoder_with_reference` 决定，当前是“平均 reference 特征后作为 bias 加到语言特征”

---

## 8. 一句话总结

- `export_onnx.py` 负责把 SAM3 拆成可独立复用的 ONNX 子模块。
- `infer_onnx.py` 负责按任务场景把这些子模块重新串起来。
- 跨图特征传递的核心不是 Python 手工拼特征，而是：
  - 先用 `reference_feature_encoder` 提 reference embedding
  - 再用 `grounding_decoder_with_reference` 在解码时融合该 embedding

