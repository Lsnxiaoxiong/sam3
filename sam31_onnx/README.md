# SAM 3.1 ONNX

这个目录是 `sam3.1_multiplex.pt` 的专用 ONNX 导出与推理脚本，不改动仓库根目录现有的 `export_onnx.py` / `infer_onnx.py`。

支持的能力：

- 文本提示推理
- 点提示推理
- 框提示推理
- 视频逐帧提示推理

推荐环境：

```powershell
C:\major\miniconda3\envs\sam3\python.exe -V
```

导出：

```powershell
C:\major\miniconda3\envs\sam3\python.exe sam31_onnx\export_sam31_onnx.py `
  --checkpoint "C:\Users\lsn\.cache\modelscope\hub\models\facebook\sam3___1\sam3.1_multiplex.pt" `
  --output-dir output\sam31_onnx
```

文本提示：

```powershell
C:\major\miniconda3\envs\sam3\python.exe sam31_onnx\infer_sam31_onnx.py `
  --model-dir output\sam31_onnx `
  --mode text `
  --image assets\images\truck.jpg `
  --text-prompt truck
```

点提示：

```powershell
C:\major\miniconda3\envs\sam3\python.exe sam31_onnx\infer_sam31_onnx.py `
  --model-dir output\sam31_onnx `
  --mode point `
  --image assets\images\truck.jpg `
  --point 320 420 `
  --point-label 1
```

框提示：

```powershell
C:\major\miniconda3\envs\sam3\python.exe sam31_onnx\infer_sam31_onnx.py `
  --model-dir output\sam31_onnx `
  --mode box `
  --image assets\images\truck.jpg `
  --box 220 180 860 820
```

视频逐帧提示：

```powershell
C:\major\miniconda3\envs\sam3\python.exe sam31_onnx\infer_sam31_onnx.py `
  --model-dir output\sam31_onnx `
  --mode video `
  --video-dir assets\videos\0001 `
  --point 320 420 `
  --point-label 1 `
  --box 220 180 860 820
```
