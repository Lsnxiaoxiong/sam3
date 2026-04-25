# SAM 3.1 Recurrent Video State Experiment

`sam31_video_state_exp` 用于试验真正跨帧状态传递的 ONNX 视频跟踪。

核心思路：
- 导出单步 `sam31_video_tracking_state_step.onnx`
- 每一帧输入当前图像和上一时刻的 `maskmem` / `obj_ptr` 状态
- 当前帧输出新的 `maskmem` / `obj_ptr`
- 推理脚本维护固定长度状态队列并跨帧回灌

导出：
```powershell
C:\major\miniconda3\envs\sam3\python.exe sam31_video_state_exp\export_sam31_video_state.py `
  --checkpoint "C:\Users\lsn\.cache\modelscope\hub\models\facebook\sam3___1\sam3.1_multiplex.pt" `
  --output-dir output\sam31_video_state_exp `
  --device cpu
```

视频递推推理：
```powershell
C:\major\miniconda3\envs\sam3\python.exe sam31_video_state_exp\infer_sam31_video_state.py `
  --model-dir output\sam31_video_state_exp `
  --video-dir assets\videos\0001 `
  --output-dir output\sam31_video_state_exp_run `
  --point 320 420 `
  --point-label 1
```

说明：
- 默认只在首帧注入提示，后续帧依赖跨帧状态递推
- 每帧会导出一个二值 mask 到输出目录
- 这是实验版，目标是验证显式状态回灌链路，而不是完整复刻官方 multiplex demo 的全部多对象调度逻辑
