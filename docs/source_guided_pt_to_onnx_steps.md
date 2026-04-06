# 从官方示例出发：如何根据源码把 `.pt` 模型导出为 ONNX

这份文档不是泛泛地讲 ONNX，而是回答一个更实际的问题：

> 如果手里一开始只有官方 `.pt` 模型和官方示例代码，我应该从哪里开始，顺着源码一步一步找到可导出 ONNX 的路径？

本文基于仓库里已经存在的官方入口：

- `test.py:1`
- `README.md:109`
- `sam3/model_builder.py:560`
- `sam3/model_builder.py:526`
- `sam3/model/sam3_image_processor.py:14`
- `sam3/model/sam3_image.py:33`
- `sam3/model/sam3_image.py:439`

---

## 1. 最开始你真正拥有的是什么

如果你是从官方示例开始，通常最先看到的是这种代码：

`test.py:1`

```python
from sam3.model_builder import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor

model = build_sam3_image_model(
    checkpoint_path=".../sam3.pt",
    load_from_HF=False
)

processor = Sam3Processor(model)
state = processor.set_image(image)
output = processor.set_text_prompt(state=state, prompt="fork")
```

或者 README 里的更简化版本：

`README.md:109`

```python
model = build_sam3_image_model()
processor = Sam3Processor(model)
inference_state = processor.set_image(image)
output = processor.set_text_prompt(state=inference_state, prompt="<YOUR_TEXT_PROMPT>")
```

这段代码已经告诉你三件关键事实：

- `.pt` 不是直接被 `torch.load(...); model(...)` 使用的
- 模型是通过 `build_sam3_image_model()` 恢复出来的
- 真正的推理入口不是直接调用 `model.forward()`，而是先经过 `Sam3Processor`

也就是说，**你做 ONNX 导出的起点，不该是 `.pt` 文件本身，而该是 `build_sam3_image_model()` 和 `Sam3Processor` 这两条源码链路。**

---

## 2. 第一步：怎么“先搞清楚 `.pt` 文件是什么”

这个问题在本仓库源码里，其实已经给了答案，不需要猜。

直接看：

- `sam3/model_builder.py:526`

这里有 `_load_checkpoint(model, checkpoint_path)`：

```python
with g_pathmgr.open(checkpoint_path, "rb") as f:
    ckpt = torch.load(f, map_location="cpu", weights_only=True)
if "model" in ckpt and isinstance(ckpt["model"], dict):
    ckpt = ckpt["model"]
```

从这里你能直接得到两个判断：

### 2.1 `.pt` 文件不是简单的单一 `state_dict`

源码显式检查：

- `if "model" in ckpt and isinstance(ckpt["model"], dict)`

说明作者预期 `.pt` 很可能是一个 checkpoint 字典，而不是直接的裸 `state_dict`。

### 2.2 真正权重可能在 `ckpt["model"]`

也就是说，看到这里时，你就已经知道导出前第一步该做什么：

- 先 `torch.load`
- 检查是否有 `model` 字段
- 再取出真正的参数字典

### 2.3 为什么这一步重要

因为你要导出 ONNX，首先得知道：

- `.pt` 是整个模型对象？
- 还是 checkpoint 字典？
- 还是纯权重？

本项目源码已经明确告诉你：**它按 checkpoint 字典处理，并从里面取 `model` 权重。**

---

## 3. 第二步：从哪里恢复模型

官方示例已经给了第一层答案：

- `test.py:6`
- `README.md:112`

都在调用：

- `build_sam3_image_model()`

接下来你应该直接跳转到它的定义：

- `sam3/model_builder.py:560`

### 3.1 `build_sam3_image_model()` 就是恢复模型的总入口

在这个函数里，你能看到完整的恢复路径：

1. 构建视觉组件
2. 构建文本组件
3. 组装 visual-language backbone
4. 构建 transformer
5. 构建 segmentation head
6. 构建 geometry encoder
7. 构建 `Sam3Image`
8. 调 `_load_checkpoint(...)`
9. `model.eval()` / 移动到设备

也就是说：

- **从官方示例跳到 `build_sam3_image_model()`，你就找到了“从哪恢复模型”的正确入口。**

### 3.2 为什么不是直接从 `test.py` 开始导出

因为 `test.py` 只是推理调用示例，它没有告诉你：

- 模型由哪些子模块组成
- checkpoint 如何映射到模型结构
- 哪些模块适合拆出来单独导出

这些信息都在 `model_builder.py` 里。

---

## 4. 第三步：从哪里看到“顶层模块有哪些”

恢复模型后，下一步不是立刻导出，而是先回答：

> 这个模型最上层到底由哪些块组成？

源码里最直接的地方有两个：

### 4.1 在构建函数里看“创建了哪些大模块”

看 `sam3/model_builder.py:560` 附近的 `build_sam3_image_model()`，你能看到这些顶层部件：

- `vision_encoder`
- `text_encoder`
- `backbone`
- `transformer`
- `dot_prod_scoring`
- `segmentation_head`
- `input_geometry_encoder`
- `inst_predictor`

然后这些部件被组装进 `Sam3Image`。

这是**第一层顶层模块视角**：从“模型是怎么拼出来的”看结构。

### 4.2 在 `Sam3Image` 类里看真正挂在对象上的成员

跳到：

- `sam3/model/sam3_image.py:33`

这里的 `Sam3Image.__init__()` 会把真正的顶层成员挂到 `self` 上：

- `self.backbone`
- `self.geometry_encoder`
- `self.transformer`
- `self.segmentation_head`
- `self.dot_prod_scoring`
- `self.inst_interactive_predictor`

这是**第二层顶层模块视角**：从“模型对象实际持有什么成员”看结构。

### 4.3 这一步为什么关键

因为 ONNX 导出不是导 checkpoint，而是导 `nn.Module` 的某个执行路径。

你只有先知道顶层模块有哪些，才能判断：

- 哪些值得拆开导出
- 哪些是公共特征提取部分
- 哪些是 task-specific decoder

---

## 5. 第四步：从哪里看到“推理真正走哪条 forward 路径”

这是很多人最容易绕远路的地方。

你如果只看：

- `model = build_sam3_image_model()`

可能会以为直接导 `model.forward()` 就行。实际上不对。

### 5.1 官方示例暴露的真实入口在 `Sam3Processor`

看：

- `sam3/model/sam3_image_processor.py:14`

官方示例调用是：

1. `processor.set_image(image)`
2. `processor.set_text_prompt(state, prompt)`

这说明真实推理路径是：

- 先处理图片
- 再处理文本
- 最后进入 grounding 推理

### 5.2 `set_image()` 先走 image encoder 路径

看：

- `sam3/model/sam3_image_processor.py:37`

这里核心语句是：

```python
state["backbone_out"] = self.model.backbone.forward_image(image)
```

这一步直接告诉你：

- 图像编码其实是独立路径
- 它的结果被存在 `backbone_out`
- 后续文本和解码会复用这份结果

也正是因为这一步，后面才适合拆出：

- `image_encoder.onnx`

### 5.3 `set_text_prompt()` 先走 text encoder，再走 grounding

看：

- `sam3/model/sam3_image_processor.py:101`

里面先做：

```python
text_outputs = self.model.backbone.forward_text([prompt], device=self.device)
state["backbone_out"].update(text_outputs)
```

然后调用：

```python
return self._forward_grounding(state)
```

这一步说明：

- 文本编码也是相对独立的
- 文本特征被放进 `backbone_out`
- 真正的检测/分割是在 `_forward_grounding()` 里发生的

### 5.4 `_forward_grounding()` 再进入 `model.forward_grounding()`

看：

- `sam3/model/sam3_image_processor.py:164`
- `sam3/model/sam3_image.py:439`

`processor._forward_grounding()` 内部实际调用：

```python
outputs = self.model.forward_grounding(...)
```

到这里，你已经把官方示例的执行链路理出来了：

1. `build_sam3_image_model()`
2. `Sam3Processor.set_image()`
3. `model.backbone.forward_image()`
4. `Sam3Processor.set_text_prompt()`
5. `model.backbone.forward_text()`
6. `processor._forward_grounding()`
7. `model.forward_grounding()`

这条链就是后续拆 ONNX 的真正依据。

---

## 6. 第五步：从哪里看到“中间特征有哪些”

中间特征不是靠猜，是顺着这条调用链看状态字典和函数参数。

### 6.1 第一类中间特征：`backbone_out`

在 `set_image()` 和 `set_text_prompt()` 中，你已经能看到：

- `state["backbone_out"]`

它是最重要的中间容器，里面逐步累积：

- 图像特征
- 视觉位置编码
- 文本特征
- 文本 mask

### 6.2 第二类中间特征：`forward_grounding()` 里的 encoder / decoder 中间量

看 `sam3/model/sam3_image.py:439` 之后的相关方法：

- `_encode_prompt(...)`
- `_run_encoder(...)`
- `_run_decoder(...)`
- `_run_segmentation_heads(...)`

这些函数名本身就在告诉你中间阶段：

- prompt 编码
- encoder 输出
- decoder 输出
- segmentation head 输出

### 6.3 这些中间特征为什么重要

因为导 ONNX 时你要判断：

- 哪些应该成为某个子图的输出
- 哪些应该成为下一个子图的输入

在本仓库最终导出方案里，最关键的中间特征就是：

- image encoder 输出的 `vision_pos_enc_*`
- image encoder 输出的 `backbone_fpn_*`
- text encoder 输出的 `language_mask`
- text encoder 输出的 `language_features`

这不是凭空设计出来的，而是顺着 `Sam3Processor -> forward_grounding` 的链路抽出来的。

---

## 7. 第六步：从哪里判断“哪些部分不适合直接导出”

判断依据也在源码里。

### 7.1 如果一条路径里混了太多 Python 状态管理，就不适合直接导出

比如 `Sam3Processor` 里大量使用：

- `state` 字典
- 条件分支
- 后处理

这类代码适合做“推理编排”，不适合直接导 ONNX。

### 7.2 如果一条路径依赖复杂内部对象，也不适合直接导出

例如 `Sam3Image.forward_grounding()` 的输入并不是简单 Tensor，而是：

- `backbone_out`
- `find_input`
- `geometric_prompt`

这些都不是适合直接暴露给 ONNX Runtime 的接口。

### 7.3 如果有算子 / 实现不稳定，也要先 patch

本仓库最终导出方案中，`export_onnx.py` 在导出前先 patch：

- RoPE
- tracker resize
- 部分 tracker head 路径

这说明作者已经验证过：

- 原始实现并不适合直接原样导出

所以当你顺源码走到这里时，结论应该是：

- **不要直接导 `Sam3Processor`**
- **也不要直接导完整 `Sam3Image.forward_grounding()`**
- **而要进一步拆分出更稳定的 Tensor 边界**

---

## 8. 第七步：从哪里拆分，为什么这么拆

当你已经知道真实推理链路是：

- image encode
- text encode
- grounding decode

拆分逻辑就自然出来了。

### 8.1 第一刀：把 image encoder 拆出来

理由：

- `set_image()` 明确单独调用 `self.model.backbone.forward_image(image)`
- 图像特征会被后续多次复用

所以可以拆成：

- `sam3_image_encoder.onnx`

### 8.2 第二刀：把 text encoder 拆出来

理由：

- `set_text_prompt()` 明确单独调用 `self.model.backbone.forward_text(...)`
- 文本特征也是独立生成的

所以可以拆成：

- `sam3_text_encoder.onnx`

### 8.3 第三刀：把 grounding decoder 拆出来

理由：

- 真正的检测 / 分割发生在 `forward_grounding()` 内部
- 它消费 image/text 中间特征
- 输出 boxes / scores / masks

所以拆成：

- `sam3_grounding_decoder.onnx`

### 8.4 第四刀：把 reference feature encoder 单独拆出来

这个不是从官方 `test.py` 直接能看到的，而是从“跨图推理需求”倒推出来的：

- reference 特征提取本质上只依赖 `backbone_fpn_0 + reference_boxes_xyxy`
- 它和 grounding decoder 可以分离

所以拆成：

- `sam3_reference_feature_encoder.onnx`

### 8.5 第五刀：把带 reference 的 grounding decoder 单独拆出来

原因：

- 跨图推理不是普通 grounding
- 需要额外吃 `reference_features`

所以单独导：

- `sam3_grounding_decoder_with_reference.onnx`

### 8.6 interactive / video 路径为什么也要拆

因为它们走的是不同推理分支，输入状态完全不同：

- interactive 需要点、框、mask prompt
- tracking 需要 memory state / object pointers

所以自然应该拆成独立子图。

---

## 9. 第八步：怎么用 wrapper 重构，为什么必须这样做

到这一步，你已经知道该拆哪些块，但原始函数接口通常不能直接导。

原因是原始接口常常包含：

- dict
- 自定义对象
- 多层状态容器
- Python 控制逻辑

所以必须写 wrapper。

### 9.1 wrapper 的目的

wrapper 只做两件事：

- 把复杂内部调用收口成纯 Tensor 输入
- 只返回部署真正需要的 Tensor 输出

### 9.2 为什么官方源码本身不够直接导

例如 `model.forward_grounding()` 的输入不是部署友好的接口。

而你真正想要给 ONNX 暴露的应该是这种接口：

- 视觉特征
- 文本特征
- box 提示
- 输出 boxes / scores / masks

这就是 wrapper 的职责。

### 9.3 本仓库最终的 wrapper 在哪里

你可以直接看导出实现：

- `export_onnx.py:420`
- `export_onnx.py:450`
- `export_onnx.py:500`
- `export_onnx.py:640`

这些 wrapper 并不是“新模型”，而是：

- 把你前面顺源码分析出来的阶段边界，显式化为可导出接口

---

## 10. 第九步：怎么导出

到这里，才轮到真正的 ONNX 导出。

### 10.1 导出入口在哪里

直接看：

- `export_onnx.py:1064`

这个 `main()` 已经把前面所有分析工作落实成了导出代码。

### 10.2 导出时做了哪些事

顺着 `export_onnx.py:1064` 往下看，执行顺序是：

1. 解析参数
2. patch 不适合直接导出的模块
3. 构建 PyTorch 模型
4. 构建 wrapper
5. 准备样例输入
6. 预跑 image/text encoder 获取中间特征
7. 调 `_export()` 逐个导出子模型

### 10.3 真正导出 ONNX 的函数在哪里

看：

- `export_onnx.py:1028`

这里封装了 `torch.onnx.export(...)`，统一指定：

- `input_names`
- `output_names`
- `dynamic_axes`
- `opset_version`
- `external_data=True`

### 10.4 动态维在哪里定义

看：

- `export_onnx.py:983`

这里统一定义了：

- batch 动态维
- box 数动态维
- point 数动态维
- reference 数动态维

这一步非常关键，因为如果不定义这些，导出的图通常只能吃固定数量的 box / point。

---

## 11. 如果你从“只有官方示例”开始，推荐的实际操作顺序

下面是一条最短可执行路径。

### 第 1 步：从示例确认模型恢复入口

先看：

- `test.py:1`
- `README.md:109`

确认：

- 恢复模型的入口是 `build_sam3_image_model()`
- 推理编排入口是 `Sam3Processor`

### 第 2 步：跳到 `build_sam3_image_model()`

看：

- `sam3/model_builder.py:560`

确认：

- 模型由哪些顶层模块组成
- checkpoint 是什么时候加载进去的

### 第 3 步：跳到 `_load_checkpoint()`

看：

- `sam3/model_builder.py:526`

确认：

- `.pt` 被怎样读取
- 是否包含 `model` 字段
- 权重键如何映射到 image model

### 第 4 步：跳到 `Sam3Processor`

看：

- `sam3/model/sam3_image_processor.py:14`

确认：

- image 和 text 是分两步推理的
- 哪些中间结果存进了 `state`

### 第 5 步：跳到 `Sam3Image.forward_grounding()`

看：

- `sam3/model/sam3_image.py:439`

再顺着内部函数看：

- `_encode_prompt`
- `_run_encoder`
- `_run_decoder`
- `_run_segmentation_heads`

确认：

- 真实的推理阶段边界
- 哪些中间特征最值得抽出来

### 第 6 步：回到 `export_onnx.py`

看：

- `export_onnx.py:1064`

对照前面分析的链路，你就能理解为什么最终拆成：

- image encoder
- text encoder
- grounding decoder
- reference feature encoder
- interactive decoder
- video tracking step

### 第 7 步：再看 wrapper 和 `_export()`

看：

- `export_onnx.py:420`
- `export_onnx.py:450`
- `export_onnx.py:500`
- `export_onnx.py:640`
- `export_onnx.py:1028`

这时你再看 wrapper，会很容易明白：

- 它们不是拍脑袋设计的
- 而是从官方推理链路里抽出来的稳定导出边界

---

## 12. 一句话总结这条源码追踪路径

如果你一开始只有官方 `.pt` 和官方示例，不知道怎么导出 ONNX，正确的源码追踪顺序是：

1. 从 `test.py` / `README` 找到模型恢复入口
2. 跳到 `build_sam3_image_model()` 看模型如何构建
3. 跳到 `_load_checkpoint()` 看 `.pt` 到底是什么
4. 跳到 `Sam3Processor` 看真实推理编排入口
5. 跳到 `Sam3Image.forward_grounding()` 看 image/text/decoder 的真正边界
6. 再回到 `export_onnx.py`，把这些边界重构成 wrapper 并导出 ONNX

也就是说：

- **先从官方示例确认入口**
- **再从 builder 确认模型恢复**
- **再从 processor / forward_grounding 确认推理路径**
- **最后才写 wrapper 和导出 ONNX**

