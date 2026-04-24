import numpy as np
import torch
import imgviz
#################################### For Image ####################################
from PIL import Image
from sam3.model_builder import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor
# Load the model
model = build_sam3_image_model(checkpoint_path=r"C:\Users\lsn\.cache\modelscope\hub\models\facebook\sam3___1\sam3.1_multiplex.pt",
                               load_from_HF=False)
processor = Sam3Processor(model)
image = Image.open(r"D:\source\sam3\assets\images\truck.jpg")
inference_state = processor.set_image(image)
# Prompt the model with text
output = processor.set_text_prompt(state=inference_state, prompt="truck")

# Get the masks, bounding boxes, and scores
masks, boxes, scores = output["masks"], output["boxes"], output["scores"]
print("masks:", masks.shape)
# 如果是 torch tensor，先转 numpy
if isinstance(masks, torch.Tensor):
    masks = masks.detach().cpu().numpy()

print("masks shape:", masks.shape)   # (1, 1, 1200, 1800)

# 取第一个 mask
mask = masks[0, 0]   # shape: (1200, 1800)

# bool -> uint8
mask_img = mask.astype(np.uint8) * 255

# 保存
Image.fromarray(mask_img).save("mask.png")
print("保存成功: mask.png")
#####################