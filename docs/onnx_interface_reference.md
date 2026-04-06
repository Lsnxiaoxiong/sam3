# SAM3 ONNX 接口文档

本文面向直接调用 ONNX 文件的开发者，列出 `export_onnx.py` 导出的各个模型文件的：

- 文件名
- 输入张量
- 输出张量
- 形状约定
- 张量语义
- 典型调用场景

配套流程说明见 `docs/onnx_export_infer_flow.md:1`。

---

## 1. 约定说明

### 1.1 维度记号

文中使用以下记号：

- `B`：batch size
- `T`：文本 token 数
- `N`：box 数量 / query 数量
- `P`：point 数量
- `R`：reference box 数量
- `C`：特征通道数
- `H` / `W`：特征图高宽

### 1.2 当前实现中的常见固定值

虽然 ONNX 中部分维度是动态的，但当前导出与推理脚本默认采用如下配置：

- 输入图像尺寸：`1008 x 1008`
- 文本 token 长度：`32`
- reference 特征维度：`256`
- interactive `mask_input` 尺寸：`256 x 256`

### 1.3 坐标系约定

脚本中同时存在三种框表示：

- 原图像素 `xyxy`
  - 形式：`[x1, y1, x2, y2]`
- 模型输入尺度像素 `xyxy`
  - 原图像素框按比例缩放到 `1008x1008`
- 归一化 `cxcywh`
  - 形式：`[cx, cy, w, h]`
  - 相对于原图宽高归一化到 `0~1`

不同 ONNX 文件使用的坐标格式不同，下面分别说明。

---

## 2. 动态维规则

`export_onnx.py` 在 `_dynamic_axes()` 中为以下维度声明了动态轴：

- `image`：第 `0` 维是 `batch`
- `tokens`：第 `0` 维是 `batch`
- `box_coords` / `box_valid_mask` / `box_labels`：第 `0` 维是 `batch`，第 `1` 维是 `num_boxes`
- `point_coords` / `point_labels`：第 `0` 维是 `batch`，第 `1` 维是 `num_points`
- `reference_boxes_xyxy`：第 `0` 维是 `num_reference_boxes`
- `reference_features` / `reference_valid_mask`：第 `0` 维是 `num_reference_boxes`，第 `1` 维是 `batch`
- `reference_embedding`：第 `0` 维是 `num_reference_boxes`
- `prev_maskmem_features` / `prev_maskmem_pos_enc` / `prev_obj_ptrs`：第 `1` 维是 `batch`

在当前 `infer_onnx.py` 中，虽然 batch 理论上可动态变化，实际基本按 `B=1` 使用。

---

## 3. `sam3_image_encoder.onnx`

### 3.1 用途

把输入 RGB 图像编码为 grounding、interactive 和 video tracking 所需的视觉特征。

### 3.2 输入

- `image`
  - 形状：`[B, 3, 1008, 1008]`
  - 类型：`uint8`
  - 语义：RGB 图像，通道优先

### 3.3 输出

- `vision_pos_enc_0`
- `vision_pos_enc_1`
- `vision_pos_enc_2`
- `backbone_fpn_0`
- `backbone_fpn_1`
- `backbone_fpn_2`
- `sam2_vision_pos_enc_0`
- `sam2_vision_pos_enc_1`
- `sam2_vision_pos_enc_2`
- `sam2_backbone_fpn_0`
- `sam2_backbone_fpn_1`
- `sam2_backbone_fpn_2`

### 3.4 输出语义

- `vision_pos_enc_*`
  - grounding decoder 使用的位置编码
- `backbone_fpn_*`
  - grounding / reference feature encoder 使用的多尺度特征
- `sam2_*`
  - interactive decoder 和 `video_tracking_step` 使用的视觉特征与位置编码

### 3.5 调用方

- `infer_onnx.py` 的 grounding 模式
- `infer_onnx.py` 的 interactive 模式
- 参考特征提取流程
- 视频跟踪单步流程

---

## 4. `sam3_text_encoder.onnx`

### 4.1 用途

把文本 prompt 编码成 grounding decoder 可消费的语言特征。

### 4.2 输入

- `tokens`
  - 形状：`[B, 32]`
  - 类型：通常为整型 token id
  - 语义：文本 tokenizer 输出

### 4.3 输出

- `language_mask`
  - 形状：`[B, 32]`
  - 语义：文本 token 有效位掩码

- `language_features`
  - 形状：`[32, B, D]`
  - 语义：文本特征

### 4.4 说明

- 导出时 `language_features` 的动态轴定义在第 `1` 维，即 batch 维在中间。
- 当前 reference 融合逻辑也直接基于这个张量布局执行。

### 4.5 调用方

- `sam3_grounding_decoder.onnx`
- `sam3_grounding_decoder_with_reference.onnx`

---

## 5. `sam3_reference_feature_encoder.onnx`

### 5.1 用途

从参考图的高分辨率特征图中，按照给定参考框提取 reference embedding。

### 5.2 输入

- `backbone_fpn_0`
  - 形状：`[B, C, H, W]`
  - 语义：来自 `sam3_image_encoder.onnx` 的最高分辨率 FPN 特征

- `reference_boxes_xyxy`
  - 形状：`[R, 4]`
  - 类型：`float32`
  - 语义：参考框，格式为模型输入尺度下的像素 `xyxy`

### 5.3 输出

- `reference_embedding`
  - 形状：`[R, 256]`
  - 语义：每个参考框对应的 reference 特征向量

### 5.4 内部逻辑

该模块内部使用 `roi_align`：

- 基于 `backbone_fpn_0` 裁剪 RoI
- 输出固定大小的 feature patch
- 再做均值池化
- 最终得到每个参考框一个 embedding

### 5.5 调用方

- `infer_onnx.py` 的跨图 reference 推理

---

## 6. `sam3_grounding_decoder.onnx`

### 6.1 用途

执行普通 grounding 推理，不包含 reference 融合。

### 6.2 输入

- `vision_pos_enc_0`
- `vision_pos_enc_1`
- `vision_pos_enc_2`
- `backbone_fpn_0`
- `backbone_fpn_1`
- `backbone_fpn_2`
- `language_mask`
- `language_features`
- `box_coords`
- `box_valid_mask`
- `box_labels`

### 6.3 输入形状与语义

- `vision_pos_enc_*`
  - 形状：`[B, C, H, W]`
  - 语义：视觉位置编码

- `backbone_fpn_*`
  - 形状：`[B, C, H, W]`
  - 语义：视觉特征

- `language_mask`
  - 形状：`[B, 32]`

- `language_features`
  - 形状：`[32, B, D]`

- `box_coords`
  - 形状：`[B, N, 4]`
  - 语义：归一化 `cxcywh`

- `box_valid_mask`
  - 形状：`[B, N]`
  - 语义：哪些 box 输入有效

- `box_labels`
  - 形状：`[B, N]`
  - 语义：box 的正负标记，当前脚本通常传布尔值

### 6.4 输出

- `boxes_xyxy`
  - 形状：`[B, Q, 4]`
  - 语义：归一化 `xyxy` 检测框

- `scores`
  - 形状：`[B, Q]`
  - 语义：每个候选目标的分数

- `masks_logits`
  - 形状：`[B, Q, Hm, Wm]`
  - 语义：低分辨率 mask logits

### 6.5 调用方

- `infer_onnx.py` 的普通 grounding

---

## 7. `sam3_grounding_decoder_with_reference.onnx`

### 7.1 用途

执行带跨图 reference 特征融合的 grounding 推理。

### 7.2 输入

在 `sam3_grounding_decoder.onnx` 的基础上，新增：

- `reference_features`
- `reference_valid_mask`
- `reference_weight`

完整输入列表：

- `vision_pos_enc_0`
- `vision_pos_enc_1`
- `vision_pos_enc_2`
- `backbone_fpn_0`
- `backbone_fpn_1`
- `backbone_fpn_2`
- `language_mask`
- `language_features`
- `box_coords`
- `box_valid_mask`
- `box_labels`
- `reference_features`
- `reference_valid_mask`
- `reference_weight`

### 7.3 reference 相关输入说明

- `reference_features`
  - 形状：`[R, B, 256]`
  - 语义：多个参考框对应的 embedding

- `reference_valid_mask`
  - 形状：`[R, B]`
  - 语义：reference 是否有效

- `reference_weight`
  - 形状：`[1]`
  - 语义：reference 特征融合强度

### 7.4 内部融合逻辑

wrapper 内部大致逻辑是：

1. 根据 `reference_valid_mask` 过滤无效 reference
2. 对 `reference_features` 做有效项平均
3. 形成 `reference_bias`
4. 将 `reference_weight * reference_bias` 加到 `language_features`
5. 再执行普通 grounding decoder

### 7.5 输出

与普通 grounding decoder 相同：

- `boxes_xyxy`
- `scores`
- `masks_logits`

### 7.6 调用方

- `infer_onnx.py` 的多图跨图特征传递模式
- `infer_onnx.py` 的显式 `--reference-image` 模式

---

## 8. `sam3_interactive_decoder.onnx`

### 8.1 用途

执行交互式分割，支持点、框和低分辨率 mask 作为提示。

### 8.2 输入

- `sam2_backbone_fpn_0`
- `sam2_backbone_fpn_1`
- `sam2_backbone_fpn_2`
- `image_pe`
- `point_coords`
- `point_labels`
- `box_xyxy`
- `box_valid_mask`
- `mask_input`

### 8.3 输入形状与语义

- `sam2_backbone_fpn_*`
  - 形状：`[B, C, H, W]`
  - 语义：来自 image encoder 的 interactive 路径特征

- `image_pe`
  - 形状：`[B, C, Hp, Wp]`
  - 语义：交互式 prompt encoder 使用的 dense positional encoding
  - 来源：`sam3_interactive_dense_pe.npy`

- `point_coords`
  - 形状：`[B, P, 2]`
  - 语义：模型输入尺度下的像素点坐标

- `point_labels`
  - 形状：`[B, P]`
  - 语义：点标签，常见值：
    - `1`：正点
    - `0`：负点
    - `-1`：padding / 无效点

- `box_xyxy`
  - 形状：`[B, N, 4]`
  - 语义：模型输入尺度下的像素 `xyxy`

- `box_valid_mask`
  - 形状：`[B, N]`

- `mask_input`
  - 形状：`[B, 1, 256, 256]`
  - 语义：上一次迭代的低分辨率 mask logits，可用于 refinement

### 8.4 输出

- `single_masks`
  - 形状：`[B, 1, Hm, Wm]`

- `single_scores`
  - 形状：`[B, 1]`

- `multi_masks`
  - 形状：`[B, M, Hm, Wm]`

- `multi_scores`
  - 形状：`[B, M]`

### 8.5 调用方

- `infer_onnx.py --mode interactive`

---

## 9. `sam3_interactive_dense_pe.npy`

### 9.1 用途

这是 interactive decoder 所需的固定 dense positional encoding，不是 ONNX 文件，但属于 interactive 接口的一部分。

### 9.2 生成方式

导出 `sam3_interactive_decoder.onnx` 时，由 `export_onnx.py` 同时保存。

### 9.3 调用方

- `infer_onnx.py` 在 interactive 模式下通过 `np.load()` 加载

---

## 10. `sam3_video_tracking_step.onnx`

### 10.1 用途

执行视频跟踪的一步状态更新，用于把当前帧结果编码回 memory，并产出下一步可复用状态。

### 10.2 输入

- `sam2_vision_pos_enc_0`
- `sam2_vision_pos_enc_1`
- `sam2_vision_pos_enc_2`
- `sam2_backbone_fpn_0`
- `sam2_backbone_fpn_1`
- `sam2_backbone_fpn_2`
- `point_coords`
- `point_labels`
- `box_xyxy`
- `box_valid_mask`
- `mask_input`
- `prev_maskmem_features`
- `prev_maskmem_pos_enc`
- `prev_memory_valid`
- `prev_memory_is_cond`
- `prev_memory_tpos`
- `prev_obj_ptrs`
- `prev_obj_ptr_valid`
- `prev_obj_ptr_is_cond`
- `prev_obj_ptr_tpos`

### 10.3 输入语义

前半部分是当前帧视觉与 prompt 输入，后半部分是历史 memory 状态：

- `prev_maskmem_features`
  - 形状：`[Tm, B, C, H, W]`
  - 语义：历史 mask memory 特征

- `prev_maskmem_pos_enc`
  - 形状：`[Tm, B, C, H, W]`
  - 语义：历史 mask memory 位置编码

- `prev_memory_valid`
  - 形状：`[Tm]`
  - 语义：历史 memory 槽位是否有效

- `prev_memory_is_cond`
  - 形状：`[Tm]`
  - 语义：是否为条件帧 memory

- `prev_memory_tpos`
  - 形状：`[Tm]`
  - 语义：历史 memory 的时间位置编码索引

- `prev_obj_ptrs`
  - 形状：`[To, B, 256]`
  - 语义：历史 object pointer

- `prev_obj_ptr_valid`
  - 形状：`[To]`

- `prev_obj_ptr_is_cond`
  - 形状：`[To]`

- `prev_obj_ptr_tpos`
  - 形状：`[To]`

### 10.4 输出

- `tracking_low_res_masks`
  - 当前帧低分辨率 mask logits

- `tracking_high_res_masks`
  - 当前帧高分辨率 mask logits

- `tracking_object_score_logits`
  - 当前帧 object score logits

- `new_maskmem_features`
  - 更新后的 memory feature

- `new_maskmem_pos_enc`
  - 更新后的 memory position encoding

- `new_obj_ptr`
  - 当前帧生成的新 object pointer

### 10.5 调用方

- 当前仓库中的 `infer_onnx.py` 还没有直接串起这个接口
- 这是给后续纯 ONNX 版视频跟踪流程预留的单步模块

---

## 11. `infer_onnx.py` 与这些接口的对应关系

### 11.1 grounding 模式

普通 grounding：

1. `sam3_image_encoder.onnx`
2. `sam3_text_encoder.onnx`
3. `sam3_grounding_decoder.onnx`

带 reference 的 grounding：

1. `sam3_image_encoder.onnx`
2. `sam3_text_encoder.onnx`
3. `sam3_reference_feature_encoder.onnx`
4. `sam3_grounding_decoder_with_reference.onnx`

### 11.2 interactive 模式

1. `sam3_image_encoder.onnx`
2. `sam3_interactive_dense_pe.npy`
3. `sam3_interactive_decoder.onnx`

### 11.3 video tracking

未来若实现纯 ONNX 视频跟踪，接口组合一般会是：

1. `sam3_image_encoder.onnx`
2. `sam3_video_tracking_step.onnx`

---

## 12. 调用时最容易出错的点

### 12.1 `language_features` 的维度顺序

它不是常见的 `[B, T, D]`，而是：

- `[T, B, D]`

如果手工调用 grounding decoder，这一点必须对齐。

### 12.2 `reference_boxes_xyxy` 的坐标系

它要求的是：

- 缩放到 `1008x1008` 模型输入尺度后的像素 `xyxy`

不是原图像素，也不是归一化坐标。

### 12.3 `box_coords` 与 `box_xyxy` 不是一回事

- grounding decoder 用的是 `box_coords = [cx, cy, w, h]`，归一化格式
- interactive decoder 用的是 `box_xyxy = [x1, y1, x2, y2]`，模型尺度像素格式

### 12.4 `mask_input` 是 logits，不是二值 mask

interactive refinement 时，`mask_input` 期望的是低分辨率 mask logits，直接传二值 mask 往往效果不稳定。

---

## 13. 建议的对接顺序

如果你要自己写调用代码，建议按下面顺序接：

### 13.1 只做单图文本分割

先接这三个：

- `sam3_image_encoder.onnx`
- `sam3_text_encoder.onnx`
- `sam3_grounding_decoder.onnx`

### 13.2 再加跨图 reference

补上：

- `sam3_reference_feature_encoder.onnx`
- `sam3_grounding_decoder_with_reference.onnx`

### 13.3 最后再接 interactive 或 tracking

interactive 和 tracking 的输入状态更多，调试成本也更高，建议后接。

