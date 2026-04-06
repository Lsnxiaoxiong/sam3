# 如何分析 `.pt` 模型结构并导出 ONNX

本文整理一套工程上可落地的方法，说明：

- 如何分析 `.pt` 模型文件里到底是什么
- 如何恢复出可运行的 PyTorch `nn.Module`
- 如何分析模型结构与数据流
- 如何把 PyTorch 模型拆分并导出成 ONNX
- 如何定义输入输出
- 如何验证 ONNX 是否真的可用

本文结合本仓库的实现经验，重点参考：

- `export_onnx.py:1064`
- `export_onnx.py:1028`
- `export_onnx.py:983`
- `docs/onnx_export_infer_flow.md:1`
- `docs/onnx_interface_reference.md:1`

---

## 1. 先搞清楚 `.pt` 文件是什么

`.pt` 文件不一定就是“可直接推理的完整模型”，常见情况有三种：

- 只有 `state_dict`
- 一个 checkpoint 字典，里面包含 `model`、`optimizer`、`epoch` 等字段
- 直接保存的 `nn.Module`

建议第一步先做最小检查：

```python
import torch

ckpt = torch.load("model.pt", map_location="cpu")
print(type(ckpt))
if isinstance(ckpt, dict):
    print(ckpt.keys())
```

### 1.1 常见判断方式

- 如果是 `OrderedDict`
  - 大概率是纯 `state_dict`
- 如果是 `dict`
  - 需要找真正的权重字段，比如 `model`、`state_dict`、`ema`
- 如果是 `nn.Module`
  - 可以直接分析，但这种保存方式相对少见

### 1.2 为什么这一步重要

因为 ONNX 导出不能只靠 `.pt` 文件本身，核心前提是：

- 你必须先得到一个真正可执行的 `nn.Module`

---

## 2. 恢复出 PyTorch 模型对象

拿到 `.pt` 后，下一步不是立刻导出，而是先恢复模型。

典型流程：

```python
model = build_model(...)
state = torch.load("model.pt", map_location="cpu")
model.load_state_dict(state["model"] if "model" in state else state)
model.eval()
```

### 2.1 在本仓库里的对应做法

本仓库不是直接从 `.pt` 反序列化出模型对象，而是：

- 用 `build_sam3_image_model(...)` 重建图像模型
- 用 `build_sam3_video_model(...)` 重建视频模型
- 再加载 checkpoint

这也是更稳的做法，因为：

- 模型结构显式可控
- 不依赖 pickle 反序列化原环境
- 更适合后续导出 ONNX

---

## 3. 分析模型结构：先宏观，再局部

不要一开始就钻进实现细节。先回答 4 个问题：

- 顶层模块有哪些
- 推理真正走哪条 `forward` 路径
- 中间特征有哪些
- 哪些部分不适合直接导出

### 3.1 打印模块树

最粗但很有用：

```python
print(model)
```

如果你要看一层子模块：

```python
for name, module in model.named_children():
    print(name, type(module))
```

如果要看整个模块树：

```python
for name, module in model.named_modules():
    print(name, type(module))
```

分析时重点关注：

- backbone
- neck / FPN
- image encoder
- text encoder
- decoder
- prompt encoder
- memory / tracking 模块
- postprocess 模块

### 3.2 查看 `forward()` 签名

```python
import inspect
print(inspect.signature(model.forward))
```

但不要只看签名，因为很多模型：

- `forward()` 只是总入口
- 推理和训练路径不同
- 真正逻辑可能在别的内部函数里

所以还需要顺着源码看：

- `forward`
- `predict`
- `forward_image`
- `forward_decoder`
- `track_step`

### 3.3 用样例输入实际跑一遍

真正理解结构，必须跑一次前向。

```python
with torch.no_grad():
    out = model(sample_input)
```

建议写一个递归打印工具：

```python
def dump(x, prefix="out"):
    import torch
    if isinstance(x, torch.Tensor):
        print(prefix, x.shape, x.dtype)
    elif isinstance(x, (list, tuple)):
        for i, item in enumerate(x):
            dump(item, f"{prefix}[{i}]")
    elif isinstance(x, dict):
        for k, v in x.items():
            dump(v, f"{prefix}.{k}")
```

目标不是“看数值”，而是确认：

- 输入 shape
- 输出 shape
- 中间哪些张量会被后续模块复用
- 哪些是训练专用，不需要导出

### 3.4 必要时挂 hook 看中间层

如果需要进一步定位 backbone 或 decoder 输出，可以挂 hook：

```python
hooks = []

def save_hook(name):
    def fn(module, inputs, outputs):
        import torch
        if isinstance(outputs, torch.Tensor):
            print(name, outputs.shape)
    return fn

hooks.append(model.backbone.register_forward_hook(save_hook("backbone")))
```

用完记得移除：

```python
for h in hooks:
    h.remove()
```

---

## 4. 找出不适合直接导出的部分

很多 `.pt` 模型不能直接导 ONNX，不是因为模型坏了，而是因为实现方式不适合静态图。

常见问题包括：

- Python 控制流过重
- 输入输出含有自定义对象
- 嵌套 dict/list 太复杂
- 复数运算
- 动态 append / Python 侧拼接
- 某些算子兼容性差
- 混合精度 / BF16 路径不稳定
- 后处理依赖 Python 逻辑

### 4.1 本仓库的实际做法

`export_onnx.py` 在导出前专门做了这些 patch：

- `patch_vitdet_rope_for_export()`
- `patch_tracker_rope_for_export()`
- `patch_tracker_resizes_for_export()`

这类 patch 的目标不是改变模型语义，而是：

- 把 ONNX 不友好的实现改写成更稳定的导出路径

这是非常标准的工程做法。

---

## 5. 不要急着导整个模型，先定义部署边界

实际项目里，最常见的错误是：

- 直接对完整大模型调用 `torch.onnx.export()`

这通常会导致：

- 输入过于复杂
- 输出不可控
- 某些路径不稳定
- 中间特征无法复用
- 导出的 ONNX 体积大且难调试

正确做法是先回答：

- 部署时真正需要哪些功能？
- 哪些中间结果值得复用？
- 哪些模块应该拆开？

### 5.1 本仓库的拆分方式

SAM3 最终拆成：

- `image_encoder`
- `text_encoder`
- `reference_feature_encoder`
- `grounding_decoder`
- `grounding_decoder_with_reference`
- `interactive_decoder`
- `video_tracking_step`

这种拆法的优点：

- 每个模块职责单一
- 接口清晰
- 方便 ONNX Runtime 调试
- 方便做跨图 reference / interactive / tracking 组合

---

## 6. 用 wrapper 重构出可导出的接口

如果原始模型接口不适合导出，最有效的方法是写 wrapper。

示例：

```python
class DecoderWrapper(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.decoder = model.decoder

    def forward(self, image_feats, text_feats, boxes):
        boxes, scores, masks = self.decoder(image_feats, text_feats, boxes)
        return boxes, scores, masks
```

### 6.1 wrapper 的职责

wrapper 只做两件事：

- 把复杂内部调用整理成“纯 Tensor 输入”
- 只返回部署真正需要的 Tensor 输出

### 6.2 本仓库里的 wrapper

关键 wrapper 包括：

- `ReferenceFeatureEncoderWrapper`
- `GroundingDecoderWithReferenceWrapper`
- `InteractiveDecoderWrapper`
- `VideoTrackingStepWrapper`

这些 wrapper 的存在，本质上就是在定义 ONNX 导出的“模块边界”。

---

## 7. 如何定义输入输出

这是导出成败最关键的一步。

原则只有一句：

- **输入输出要面向部署，而不是面向源码内部实现**

### 7.1 输入设计原则

输入应该满足：

- 全部是 Tensor
- 坐标格式清晰
- 尽量少
- 能覆盖真实推理需求
- 便于下游复用

比如在本仓库中：

- `image_encoder` 输入整图
- `grounding_decoder` 不再输入整图，而是输入视觉特征 + 语言特征 + box 提示

这就比导出一个“大而全”的模型更合理。

### 7.2 输出设计原则

输出应该满足：

- 纯 Tensor
- 只保留下游真正需要的结果
- 不返回 Python 对象
- 避免训练专用字段

例如：

- `boxes_xyxy`
- `scores`
- `masks_logits`

就是非常适合部署的输出。

### 7.3 明确命名输入输出

导出时一定要显式指定：

- `input_names`
- `output_names`

否则导出的 ONNX 输入输出名字可能非常难用。

本仓库统一通过 `_export()` 封装这一点，位置见 `export_onnx.py:1028`。

### 7.4 定义动态维

如果 batch、box 数量、point 数量不是固定值，就必须定义动态轴。

示例：

```python
dynamic_axes = {
    "image": {0: "batch"},
    "box_coords": {0: "batch", 1: "num_boxes"},
    "point_coords": {0: "batch", 1: "num_points"},
}
```

本仓库完整实现见 `export_onnx.py:983`。

如果不定义动态轴，常见后果是：

- 模型只能接固定 shape
- box 数量一变就跑不通

---

## 8. 如何准备导出样例输入

ONNX 导出依赖样例输入来 trace 计算图，所以样例输入不能随便写。

### 8.1 样例输入必须满足

- dtype 正确
- shape 合理
- 能触发你要导出的那条路径
- 不要让关键输入在图中被优化掉

### 8.2 本仓库中的样例输入

`export_onnx.py` 里准备了这类样例：

- `sample_image`
- `sample_tokens`
- `sample_box_coords`
- `sample_reference_boxes_xyxy`
- `sample_reference_features`
- `sample_mask_input`
- `sample_prev_maskmem_features`
- `sample_prev_obj_ptrs`

这些样例的作用不是做真实推理，而是保证导出的图结构完整、接口稳定。

### 8.3 为什么样例输入很重要

如果样例输入设计不对，可能导致：

- 某条逻辑分支没被 trace 到
- 图里的某个输入被常量折叠掉
- 输出 shape 锁死
- 导出的图和真实推理路径不一致

---

## 9. 实际导出 ONNX 的标准流程

### 9.1 切到推理态

```python
model.eval()
with torch.inference_mode():
    ...
```

### 9.2 统一 dtype

很多模型在导出前会先转成 `float32`：

```python
model = model.float()
```

原因通常是：

- 避免 BF16 / FP16 导出不稳定
- 避免 ONNX Runtime 对某些精度路径支持不完整

### 9.3 调用 `torch.onnx.export`

标准形式：

```python
torch.onnx.export(
    module,
    args=sample_args,
    f="model.onnx",
    input_names=[...],
    output_names=[...],
    opset_version=17,
    dynamic_axes=dynamic_axes,
    do_constant_folding=True,
    external_data=True,
)
```

### 9.4 关键参数说明

- `opset_version`
  - ONNX 算子版本，过低可能不支持新算子

- `dynamic_axes`
  - 指定 batch、box 数等动态维

- `do_constant_folding=True`
  - 做常量折叠优化

- `external_data=True`
  - 大模型权重可拆成外部数据文件，避免单一 ONNX 文件过大

---

## 10. 导出后必须验模

导出成功不代表模型可用。至少要验证 4 件事。

### 10.1 ONNX 结构检查

```python
import onnx

model = onnx.load("model.onnx")
onnx.checker.check_model(model)
```

### 10.2 检查输入输出接口

```python
import onnxruntime as ort

sess = ort.InferenceSession("model.onnx")
print([i.name for i in sess.get_inputs()])
print([o.name for o in sess.get_outputs()])
```

### 10.3 与 PyTorch 结果对比

用同一组输入分别跑：

- PyTorch 模型
- ONNX Runtime

对比：

- shape 是否一致
- 分数是否接近
- box / mask 是否接近

### 10.4 验证真实业务链路

最容易忽略但最重要的一步，是把导出的多个 ONNX 真正串起来跑业务流程。

本仓库里就是：

- `export_onnx.py` 负责导出
- `infer_onnx.py` 负责实际串联运行

这一步可以发现大量“单模块看起来没问题，但组合起来有问题”的问题。

---

## 11. 结合本仓库的完整思路

### 11.1 导出阶段

`export_onnx.py` 的主流程可以概括为：

1. 解析参数
2. 对模型做 ONNX 友好化 patch
3. 构建 PyTorch 模型
4. 构建 wrapper
5. 准备样例输入
6. 预跑 image/text encoder 获取中间特征
7. 逐个导出 ONNX 子模块
8. 保存 interactive 所需的 `sam3_interactive_dense_pe.npy`

### 11.2 推理阶段

`infer_onnx.py` 则按模式组合这些子模块：

普通 grounding：

1. `sam3_image_encoder.onnx`
2. `sam3_text_encoder.onnx`
3. `sam3_grounding_decoder.onnx`

跨图 reference grounding：

1. `sam3_image_encoder.onnx`
2. `sam3_text_encoder.onnx`
3. `sam3_reference_feature_encoder.onnx`
4. `sam3_grounding_decoder_with_reference.onnx`

interactive：

1. `sam3_image_encoder.onnx`
2. `sam3_interactive_dense_pe.npy`
3. `sam3_interactive_decoder.onnx`

---

## 12. 一个可复用的工作流模板

以后你分析任何 `.pt` 模型，都可以按这个顺序做。

### 第一步：识别 checkpoint 内容

- `torch.load()`
- 看是 `state_dict`、checkpoint 字典还是 `nn.Module`

### 第二步：恢复 `nn.Module`

- 找构建函数
- `load_state_dict`
- `eval()`

### 第三步：看模型结构

- `print(model)`
- `named_modules()`
- `inspect.signature(forward)`

### 第四步：跑样例并打印 shape

- 看输入 shape
- 看输出 shape
- 看中间特征

### 第五步：识别 ONNX 不友好部分

- 控制流
- 复数运算
- 非 Tensor 输入输出
- 自定义后处理

### 第六步：定义部署边界

- 决定是导一个整体，还是拆成多个子模块

### 第七步：编写 wrapper

- 把复杂调用变成稳定的 Tensor 接口

### 第八步：定义 ONNX 接口

- `input_names`
- `output_names`
- `dynamic_axes`

### 第九步：准备样例输入并导出

- 确保样例输入覆盖目标路径

### 第十步：验模与联调

- ONNX checker
- ORT 跑通
- 与 PyTorch 对比
- 串完整业务链路

---

## 13. 经验总结

如果只记三点，建议记这三点：

- 不要直接导整个大模型，先拆出合理的部署边界
- 输入输出定义要服务于部署，而不是服务于源码内部结构
- 导出成功只是开始，必须做 ONNX Runtime 的真实联调

对于本仓库这样的多模态、带 tracking / interactive / reference 分支的模型，拆模块导出是必要而不是可选项。

