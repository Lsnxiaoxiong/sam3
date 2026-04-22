# SAM 3.1 ONNX Lite

`sam31_onnx_lite` 是基于 `sam3.1_multiplex.pt` 的轻量拆分版 ONNX 导出与推理目录。

拆分后的模型：
- `sam31_image_encoder.onnx`
- `sam31_text_encoder.onnx`
- `sam31_grounding_decoder.onnx`
- `sam31_point_decoder.onnx`
- `sam31_box_decoder.onnx`
- `sam31_video_prompt_decoder.onnx`

特点：
- 图像编码器共享，避免 `text / point / box / video` 各自重复携带整套 backbone
- 文本提示走 `image_encoder + text_encoder + grounding_decoder`
- 点提示走 `image_encoder + point_decoder`
- 框提示走 `image_encoder + box_decoder`
- 视频提示走 `image_encoder + video_prompt_decoder`，按帧循环推理

导出：
```powershell
C:\major\miniconda3\envs\sam3\python.exe sam31_onnx_lite\export_sam31_lite.py `
  --checkpoint "C:\Users\lsn\.cache\modelscope\hub\models\facebook\sam3___1\sam3.1_multiplex.pt" `
  --output-dir output\sam31_onnx_lite
```

文本提示：
```powershell
C:\major\miniconda3\envs\sam3\python.exe sam31_onnx_lite\infer_sam31_lite.py `
  --model-dir output\sam31_onnx_lite `
  --mode text `
  --image assets\images\truck.jpg `
  --text-prompt truck
```

点提示：
```powershell
C:\major\miniconda3\envs\sam3\python.exe sam31_onnx_lite\infer_sam31_lite.py `
  --model-dir output\sam31_onnx_lite `
  --mode point `
  --image assets\images\truck.jpg `
  --point 320 420 `
  --point-label 1
```

框提示：
```powershell
C:\major\miniconda3\envs\sam3\python.exe sam31_onnx_lite\infer_sam31_lite.py `
  --model-dir output\sam31_onnx_lite `
  --mode box `
  --image assets\images\truck.jpg `
  --box 220 180 860 820
```

视频提示：
```powershell
C:\major\miniconda3\envs\sam3\python.exe sam31_onnx_lite\infer_sam31_lite.py `
  --model-dir output\sam31_onnx_lite `
  --mode video `
  --video-dir assets\videos\0001 `
  --point 320 420 `
  --point-label 1 `
  --box 220 180 860 820
```
