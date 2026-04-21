# SAM3 ONNX 示例与 GUI

新增了两个入口：

- `examples/onnx_examples.py`
- `examples/onnx_gui.py`

## 点提示推理

```bash
python examples/onnx_examples.py point_prompt ^
  --model-dir <onnx_dir> ^
  --image assets/images/groceries.jpg ^
  --point-coords "320,240;420,260" ^
  --point-labels "1,0" ^
  --output output/point_prompt.jpg
```

## 框提示推理

```bash
python examples/onnx_examples.py box_prompt ^
  --model-dir <onnx_dir> ^
  --image assets/images/groceries.jpg ^
  --box-prompt "200,120,520,460" ^
  --output output/box_prompt.jpg
```

## 文本提示推理

```bash
python examples/onnx_examples.py text_prompt ^
  --model-dir <onnx_dir> ^
  --image assets/images/groceries.jpg ^
  --text-prompt "red light" ^
  --output output/text_prompt.jpg
```

## 框跨图特征传递

`--target-images` 使用分号分隔多个目标图片路径。

```bash
python examples/onnx_examples.py cross_image_box_transfer ^
  --model-dir <onnx_dir> ^
  --reference-image assets/videos/0001/9.jpg ^
  --reference-boxes "600,355,866,510" ^
  --target-images "assets/videos/0001/14.jpg;assets/videos/0001/20.jpg;assets/videos/0001/49.jpg" ^
  --text-prompt "white T-shirt" ^
  --output-dir output/cross_image_box_transfer
```

## 启动 GUI

```bash
python examples/onnx_gui.py
```

GUI 包含三个页签：

- `点/框提示`
- `文本提示`
- `跨图特征`

如果没有填写 `Model Dir`，程序会自动尝试以下目录：

- 当前目录
- `output/`
- `models/`
- `sam3_onnx_output/`
