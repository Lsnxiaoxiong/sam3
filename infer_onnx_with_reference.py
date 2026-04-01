#!/usr/bin/env python3
"""
跨图参考的 SAM3 ONNX 推理

使用 RoI 特征传递机制，将第一张图的标注特征作为后续图片的参考。
无需修改 ONNX 模型，通过在语言特征中添加参考区域的 RoI 特征实现。

使用示例:
    # 单图推理（原始模式）
    python infer_onnx_with_reference.py --image input.jpg --text-prompt "dog"

    # 多图推理，使用第一张图作为参考
    python infer_onnx_with_reference.py --images img1.jpg img2.jpg img3.jpg --text-prompt "dog" --use-reference

    # 交互式选择参考框
    python infer_onnx_with_reference.py --images img1.jpg img2.jpg --text-prompt "target object" --use-reference --interactive
"""

import argparse
import pathlib
import sys
import typing
from dataclasses import dataclass

import cv2
import imgviz
import numpy as np
import onnxruntime
import PIL.Image
import torch
from loguru import logger
from numpy.typing import NDArray
from osam._models.yoloworld.clip import tokenize
from torchvision.ops import roi_align


@dataclass
class ReferenceFeature:
    """存储从参考图像提取的特征"""
    roi_features: NDArray  # RoI 池化后的特征 [1, channels, h, w]
    boxes: NDArray  # 归一化框 [cx, cy, w, h]
    masks: NDArray  # 分割掩码 [1, H, W]
    scores: NDArray  # 置信度分数


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # 输入图像
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument(
        "--image",
        type=pathlib.Path,
        help="Path to a single input image.",
    )
    input_group.add_argument(
        "--images",
        type=pathlib.Path,
        nargs="+",
        help="Paths to multiple input images.",
    )

    # 提示
    prompt_group = parser.add_mutually_exclusive_group(required=True)
    prompt_group.add_argument(
        "--text-prompt",
        type=str,
        help="Text prompt for segmentation.",
    )
    prompt_group.add_argument(
        "--box-prompt",
        type=str,
        nargs="?",
        const="0,0,0,0",
        help="Box prompt for segmentation in format: cx,cy,w,h (normalized).",
    )

    # 参考模式
    parser.add_argument(
        "--use-reference",
        action="store_true",
        help="Use first image's annotation as reference for others.",
    )
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="Interactively select the reference box on the first image.",
    )
    parser.add_argument(
        "--output-dir",
        type=pathlib.Path,
        default="output",
        help="Output directory.",
    )
    parser.add_argument(
        "--refine-threshold",
        type=float,
        default=0.5,
        help="Score threshold for using reference features (higher = more confident).",
    )

    args = parser.parse_args()
    logger.debug("input: {}", args.__dict__)

    # 处理图像路径
    if args.image:
        args.images = [args.image]

    # 处理框提示
    if args.box_prompt:
        args.box_prompt = [float(x) for x in args.box_prompt.split(",")]
        if len(args.box_prompt) != 4:
            logger.error("box_prompt must have 4 values: cx, cy, w, h")
            sys.exit(1)

    # 交互式框选择
    if args.interactive and len(args.images) > 0:
        image: NDArray = imgviz.asrgb(imgviz.io.imread(args.images[0]))
        logger.info("Please select reference box on the first image")
        logger.info("Press ENTER or SPACE to confirm, ESC to cancel")

        x, y, w, h = cv2.selectROI(
            "Select reference box",
            image[:, :, ::-1],
            fromCenter=False,
            showCrosshair=True,
        )
        cv2.destroyAllWindows()

        if [x, y, w, h] == [0, 0, 0, 0]:
            logger.warning("No reference box selected, disabling reference mode")
            args.use_reference = False
        else:
            args.box_prompt = [
                (x + w / 2) / image.shape[1],
                (y + h / 2) / image.shape[0],
                w / image.shape[1],
                h / image.shape[0],
            ]
            logger.debug("box_prompt: {!r}", ",".join(f"{x:.3f}" for x in args.box_prompt))

    return args


def extract_roi_features(
    backbone_fpn: list[NDArray],
    boxes_xywh: NDArray,
    image_size: tuple[int, int],
) -> NDArray:
    """
    从 backbone 特征图中提取 RoI 特征

    Args:
        backbone_fpn: FPN 特征图列表 [C, H, W]
        boxes_xywh: 归一化框 [cx, cy, w, h]
        image_size: 原始图像尺寸 (width, height)

    Returns:
        RoI 特征 [1, C, 7, 7]
    """
    # 将 cx,cy,w,h 转换为 xyxy
    cx, cy, w, h = boxes_xywh
    x1 = (cx - w / 2) * image_size[0]
    y1 = (cy - h / 2) * image_size[1]
    x2 = (cx + w / 2) * image_size[0]
    y2 = (cy + h / 2) * image_size[1]
    boxes_xyxy = np.array([[0, x1, y1, x2, y2]], dtype=np.float32)  # [N, 5], N=1

    # 使用最高分辨率的特征图 (backbone_fpn[0])
    # 假设输入图像被 resize 到 1008x1008, stride=14, 所以特征图是 72x72
    feat_map = backbone_fpn[0]  # [C, H, W]
    feat_map = torch.from_numpy(feat_map).unsqueeze(0)  # [1, C, H, W]

    # RoI Align
    # spatial_scale = 原始图像尺寸 / 特征图尺寸 = 1008 / 72 = 14
    spatial_scale = 1008 / feat_map.shape[2]

    roi_features = roi_align(
        feat_map,
        torch.from_numpy(boxes_xyxy),
        output_size=(7, 7),
        spatial_scale=spatial_scale,
        sampling_ratio=2,
        aligned=True,
    )  # [1, C, 7, 7]

    # 全局平均池化得到 [1, C]
    roi_features_pooled = roi_features.mean(dim=(2, 3))  # [1, C]

    return roi_features_pooled.numpy()


def augment_language_features(
    language_features: NDArray,
    reference_features: ReferenceFeature | None,
    augment_type: str = "add",  # 默认使用 add，因为 ONNX 模型期望固定的输入形状
) -> NDArray:
    """
    增强语言特征，加入参考信息

    注意：由于 ONNX decoder 期望固定的输入形状 [batch, 32, D]，
    我们不能使用拼接方式（会改变 token 数量），只能使用相加或替换方式。

    Args:
        language_features: 原始语言特征 [batch, num_tokens, D]
        reference_features: 参考特征（如果有）
        augment_type: 增强方式 ("concat" | "add" | "replace_first" | "none")

    Returns:
        增强后的语言特征（形状与输入相同）
    """
    if reference_features is None:
        return language_features

    if augment_type == "none":
        return language_features

    roi_feat = reference_features.roi_features  # [1, C]

    # 将 roi_feat 从 [1, C] 转换为 [batch, 1, D] 以匹配 language_features
    batch_size = language_features.shape[0]
    num_tokens = language_features.shape[1]
    feat_dim = language_features.shape[2]

    logger.debug(f"augment_language_features: language_features.shape={language_features.shape}, roi_feat.shape={roi_feat.shape}, feat_dim={feat_dim}")

    # 确保 RoI 特征维度匹配
    if roi_feat.shape[1] != feat_dim:
        # 如果维度不匹配，需要使用投影
        logger.warning(
            f"RoI feature dim ({roi_feat.shape[1]}) != language feature dim ({feat_dim}), "
            "using zero-padding projection"
        )
        if roi_feat.shape[1] < feat_dim:
            # 填充 RoI 特征
            padding = np.zeros((roi_feat.shape[0], feat_dim - roi_feat.shape[1]), dtype=roi_feat.dtype)
            roi_feat = np.concatenate([roi_feat, padding], axis=1)
        else:
            # 截断 RoI 特征
            roi_feat = roi_feat[:, :feat_dim]

    roi_feat_reshaped = roi_feat.reshape(1, 1, feat_dim)  # [1, 1, D]
    roi_feat_broadcast = np.tile(roi_feat_reshaped, (batch_size, num_tokens, 1))  # [batch, num_tokens, D]

    if augment_type == "add":
        # 将参考特征加到语言特征上（按元素相加）
        # 这种方式让参考特征作为"偏置"影响所有语言 token
        augmented = language_features + roi_feat_broadcast
        logger.debug(f"Added reference features: {language_features.shape} + {roi_feat_broadcast.shape} -> {augmented.shape}")
        return augmented

    elif augment_type == "replace_first":
        # 替换第一个 token 为参考特征（保留语言特征的分布）
        # 这种方式将参考特征作为"系统提示"token
        augmented = language_features.copy()
        augmented[:, 0:1, :] = roi_feat_broadcast[:, 0:1, :]
        logger.debug(f"Replaced first token with reference: {language_features.shape} -> {augmented.shape}")
        return augmented

    elif augment_type == "concat":
        logger.warning(
            "Concat type is not supported for ONNX decoder (fixed input shape). "
            "Falling back to 'add' type."
        )
        return language_features + roi_feat_broadcast

    return language_features


def run_inference(
    sess_image: onnxruntime.InferenceSession,
    sess_language: onnxruntime.InferenceSession,
    sess_decode: onnxruntime.InferenceSession,
    image_path: pathlib.Path,
    text_prompt: str,
    box_prompt: list[float] | None,
    reference_features: ReferenceFeature | None = None,
) -> tuple[NDArray, NDArray, NDArray, PIL.Image.Image]:
    """运行单图推理"""

    # 1. 图像编码
    image: PIL.Image.Image = PIL.Image.open(image_path).convert("RGB")
    orig_width, orig_height = image.size
    logger.debug(f"original image size: {image.size}")

    logger.debug("running image encoder...")
    output = sess_image.run(
        None,
        {"image": np.asarray(image.resize((1008, 1008))).transpose(2, 0, 1)}
    )
    assert len(output) == 6
    vision_pos_enc: list[NDArray] = output[:3]
    backbone_fpn: list[NDArray] = output[3:]
    logger.debug("finished running image encoder")

    # 2. 语言编码
    logger.debug("running language encoder...")
    output = sess_language.run(
        None,
        {"tokens": tokenize(texts=[text_prompt], context_length=32)}
    )
    assert len(output) == 3
    language_mask: NDArray = output[0]
    language_features: NDArray = output[1]
    logger.debug("finished running language encoder")

    # 3. 增强语言特征（加入参考信息）
    if reference_features is not None:
        language_features = augment_language_features(
            language_features,
            reference_features,
            augment_type="concat",
        )

    # 4. 解码
    logger.debug("running decoder...")
    box_coords: NDArray[np.float32] = np.array(
        box_prompt if box_prompt else [0, 0, 0, 0],
        dtype=np.float32
    ).reshape(1, 1, 4)
    box_labels: NDArray[np.int64] = np.array([[1]], dtype=np.int64)
    box_masks: NDArray[np.bool_] = np.array(
        [False] if box_prompt else [True],
        dtype=np.bool_
    ).reshape(1, 1)

    output = sess_decode.run(
        None,
        {
            "original_height": np.array(orig_height, dtype=np.int64),
            "original_width": np.array(orig_width, dtype=np.int64),
            "backbone_fpn_0": backbone_fpn[0],
            "backbone_fpn_1": backbone_fpn[1],
            "backbone_fpn_2": backbone_fpn[2],
            "vision_pos_enc_2": vision_pos_enc[2],
            "language_mask": language_mask,
            # 使用增强后的语言特征
            "language_features": language_features,
            "box_coords": box_coords,
            "box_labels": box_labels,
            "box_masks": box_masks,
        },
    )
    assert len(output) == 3
    boxes: NDArray = output[0]
    scores: NDArray = output[1]
    masks: NDArray = output[2]
    logger.debug("finished running decoder")

    # 5. 可视化
    viz = imgviz.instances2rgb(
        image=np.asarray(image),
        masks=masks[:, 0, :, :],
        bboxes=boxes[:, [1, 0, 3, 2]],  # 转换为 xyxy
        labels=np.arange(len(masks)) + 1,
        captions=[f"{text_prompt}: {s:.0%}" for s in scores],
        font_size=max(1, min(image.size) // 40),
    )

    return boxes, scores, masks, PIL.Image.fromarray(viz)


def extract_reference_features(
    backbone_fpn: list[NDArray],
    boxes: NDArray,
    masks: NDArray,
    scores: NDArray,
    image_size: tuple[int, int],
    threshold: float = 0.5,
    language_feature_dim: int | None = None,  # 用于维度匹配
) -> ReferenceFeature | None:
    """
    从推理结果中提取参考特征

    选择得分最高的检测结果作为参考
    """
    if len(scores) == 0 or scores.max() < threshold:
        logger.warning(f"No confident detections (max score: {scores.max():.3f} < {threshold})")
        return None

    # 选择得分最高的目标
    best_idx = np.argmax(scores)
    best_box = boxes[best_idx]  # 归一化 xyxy
    best_score = scores[best_idx]
    best_mask = masks[best_idx, 0, :, :]  # [H, W]

    # 将 xyxy 转换为 cx,cy,w,h
    x1, y1, x2, y2 = best_box
    cx = (x1 + x2) / 2
    cy = (y1 + y2) / 2
    w = x2 - x1
    h = y2 - y1
    box_xywh = np.array([cx, cy, w, h])

    # 提取 RoI 特征
    roi_features = extract_roi_features(backbone_fpn, box_xywh, image_size)

    # 如果需要，填充到目标维度
    if language_feature_dim is not None and roi_features.shape[1] != language_feature_dim:
        logger.debug(f"Padding roi_features from {roi_features.shape[1]} to {language_feature_dim}")
        if roi_features.shape[1] < language_feature_dim:
            padding = np.zeros((1, language_feature_dim - roi_features.shape[1]), dtype=roi_features.dtype)
            roi_features = np.concatenate([roi_features, padding], axis=1)
        else:
            roi_features = roi_features[:, :language_feature_dim]

    logger.info(f"Extracted reference feature from best detection (score: {best_score:.3f})")

    return ReferenceFeature(
        roi_features=roi_features,
        boxes=box_xywh,
        masks=best_mask,
        scores=np.array([best_score]),
    )


def main():
    args = parse_args()

    # 初始化 ONNX 会话
    providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']

    logger.info("Loading ONNX models...")
    sess_image = onnxruntime.InferenceSession(
        r"C:\Users\lsn\Downloads\sam3_vit_h\sam3_image_encoder.onnx",
        providers=providers
    )
    sess_language = onnxruntime.InferenceSession(
        r"C:\Users\lsn\Downloads\sam3_vit_h\sam3_language_encoder.onnx",
        providers=providers
    )
    sess_decode = onnxruntime.InferenceSession(
        r"C:\Users\lsn\Downloads\sam3_vit_h\sam3_decoder.onnx",
        providers=providers
    )
    logger.info("ONNX models loaded")

    # 获取 language_features 的维度（用于 RoI 特征维度匹配）
    language_feature_dim = None
    for inp in sess_decode.get_inputs():
        if inp.name == "language_features":
            # language_features shape: [batch, num_tokens, dim]
            # For ONNX, shape might be symbolic (None for some dims)
            if isinstance(inp.shape, (list, tuple)) and len(inp.shape) == 3:
                language_feature_dim = inp.shape[2] if isinstance(inp.shape[2], int) else None
            logger.debug(f"language_features input shape: {inp.shape}, dim={language_feature_dim}")
            break

    # 准备输出目录
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # 参考特征（从第一张图提取）
    reference_features: ReferenceFeature | None = None

    for img_idx, image_path in enumerate(args.images):
        logger.info(f"Processing {img_idx + 1}/{len(args.images)}: {image_path.name}")

        # 确定是否使用参考
        use_ref = args.use_reference and reference_features is not None

        if use_ref:
            logger.info(f"  -> Using reference features (extracted from frame 0)")

        # 运行推理
        boxes, scores, masks, viz_image = run_inference(
            sess_image=sess_image,
            sess_language=sess_language,
            sess_decode=sess_decode,
            image_path=image_path,
            text_prompt=args.text_prompt,
            box_prompt=args.box_prompt if img_idx == 0 else None,  # 只在第一张图使用框提示
            reference_features=reference_features if use_ref else None,
        )

        # 保存结果
        out_path = args.output_dir / f"{image_path.stem}_result.jpg"
        viz_image.save(out_path)
        logger.info(f"  -> Saved result to {out_path}")
        logger.info(f"  -> Detected {len(scores)} objects (scores: {scores.max():.3f} max)")

        # 从第一张图提取参考特征（用于后续图像）
        if args.use_reference and img_idx == 0 and len(scores) > 0:
            # 需要重新运行一次以获取 backbone_fpn
            image_pil = PIL.Image.open(image_path).convert("RGB")
            image_np = np.asarray(image_pil.resize((1008, 1008))).transpose(2, 0, 1)
            output = sess_image.run(None, {"image": image_np})
            backbone_fpn: list[NDArray] = output[3:]

            reference_features = extract_reference_features(
                backbone_fpn=backbone_fpn,
                boxes=boxes,
                masks=masks,
                scores=scores,
                image_size=image_pil.size,
                threshold=args.refine_threshold,
                language_feature_dim=language_feature_dim,
            )

            if reference_features is not None:
                logger.info(f"  -> Reference feature extracted (will be used for subsequent images)")
            else:
                logger.warning(f"  -> Failed to extract reference feature")

    logger.info("Done!")


if __name__ == "__main__":
    main()
