import cv2
import numpy as np
import torch
#################################### For Image ####################################
from PIL import Image

from sam3.model.sam3_image_processor import Sam3Processor
from sam3.model_builder import build_sam3_image_model
# Load the model
model = build_sam3_image_model(
    checkpoint_path=r"C:\Users\lsn\.cache\modelscope\hub\models\facebook\sam3\sam3.pt",
    load_from_HF=False
)

processor = Sam3Processor(model)
# Load an image
image = Image.open(r"D:\temp\coco128\images\000000000092.jpg")
inference_state = processor.set_image(image)
# Prompt the model with text
output = processor.set_text_prompt(state=inference_state, prompt="fork")

# Get the masks, bounding boxes, and scores
masks, boxes, scores = output["masks"], output["boxes"], output["scores"]
print(masks.shape)
if masks.shape[0] > 0:
    mask = masks[0]  # 取第一个

    # (1, H, W) → (H, W)
    mask = mask.squeeze(0)

    # GPU → CPU → numpy
    mask_np = mask.detach().cpu().numpy()

    # bool → uint8 (0/255)
    mask_np = mask_np.astype(np.uint8) * 255

    # 保存
    cv2.imwrite("mask.png", mask_np)