#!/usr/bin/env python3

from __future__ import annotations

import argparse
import pathlib
from importlib import resources
from typing import Sequence

import imgviz
import numpy as np
import onnxruntime as ort
import PIL.Image
from osam._models.yoloworld.clip import tokenize


IMAGE_SIZE = 1008


def _providers() -> list[str]:
    available = ort.get_available_providers()
    ordered = ["CUDAExecutionProvider", "CPUExecutionProvider"]
    return [provider for provider in ordered if provider in available]


def _parse_point_list(value: str | None) -> np.ndarray | None:
    if not value:
        return None
    points = []
    for item in value.split(";"):
        x_str, y_str = item.split(",")
        points.append([float(x_str), float(y_str)])
    return np.asarray(points, dtype=np.float32)


def _parse_label_list(value: str | None) -> np.ndarray | None:
    if not value:
        return None
    return np.asarray([int(item) for item in value.split(",")], dtype=np.int32)


def _parse_box_list(value: str | None) -> np.ndarray | None:
    if not value:
        return None
    boxes = []
    for item in value.split(";"):
        coords = [float(part) for part in item.split(",")]
        if len(coords) != 4:
            raise ValueError(f"invalid box: {item}")
        boxes.append(coords)
    return np.asarray(boxes, dtype=np.float32)


def _resize_for_encoder(image: PIL.Image.Image) -> np.ndarray:
    resized = image.resize((IMAGE_SIZE, IMAGE_SIZE), resample=PIL.Image.BILINEAR)
    return np.asarray(resized, dtype=np.uint8).transpose(2, 0, 1)[None, ...]


def _scaled_xyxy_to_model_space(boxes_xyxy: np.ndarray, width: int, height: int) -> np.ndarray:
    scaled = boxes_xyxy.copy().astype(np.float32)
    scaled[:, [0, 2]] = scaled[:, [0, 2]] * (IMAGE_SIZE / width)
    scaled[:, [1, 3]] = scaled[:, [1, 3]] * (IMAGE_SIZE / height)
    return scaled


def _scaled_points_to_model_space(points_xy: np.ndarray, width: int, height: int) -> np.ndarray:
    scaled = points_xy.copy().astype(np.float32)
    scaled[:, 0] = scaled[:, 0] * (IMAGE_SIZE / width)
    scaled[:, 1] = scaled[:, 1] * (IMAGE_SIZE / height)
    return scaled


def _xyxy_pixels_to_cxcywh_normalized(boxes_xyxy: np.ndarray, width: int, height: int) -> np.ndarray:
    x0, y0, x1, y1 = boxes_xyxy.T
    cx = ((x0 + x1) * 0.5) / width
    cy = ((y0 + y1) * 0.5) / height
    w = (x1 - x0) / width
    h = (y1 - y0) / height
    return np.stack([cx, cy, w, h], axis=1).astype(np.float32)


def _upsample_masks(masks: np.ndarray, width: int, height: int) -> np.ndarray:
    resized = []
    for mask in masks:
        pil = PIL.Image.fromarray(mask.astype(np.float32), mode="F")
        pil = pil.resize((width, height), resample=PIL.Image.BILINEAR)
        resized.append(np.asarray(pil, dtype=np.float32))
    return np.stack(resized, axis=0)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--model-dir", type=pathlib.Path, default=pathlib.Path("models"))
    parser.add_argument("--image", type=pathlib.Path, required=True)
    parser.add_argument("--output", type=pathlib.Path, default=pathlib.Path("output/onnx_result.jpg"))
    parser.add_argument("--mode", choices=["grounding", "interactive"], required=True)
    parser.add_argument("--text-prompt", type=str)
    parser.add_argument("--grounding-boxes", type=str, help="xyxy pixel boxes, separated by ';'")
    parser.add_argument("--grounding-box-labels", type=str, help="1/0 labels for grounding boxes")
    parser.add_argument("--point-coords", type=str, help="x,y;x,y in pixels")
    parser.add_argument("--point-labels", type=str, help="1,0,...")
    parser.add_argument("--box-prompt", type=str, help="xyxy pixel boxes, separated by ';'")
    parser.add_argument("--mask-input", type=pathlib.Path, help="Path to .npy low-res mask logits")
    parser.add_argument("--multimask-output", action="store_true")
    parser.add_argument("--score-threshold", type=float, default=0.5)
    return parser.parse_args()


def _session(model_dir: pathlib.Path, name: str) -> ort.InferenceSession:
    return ort.InferenceSession(str(model_dir / name), providers=_providers())


def _run_image_encoder(session: ort.InferenceSession, image: PIL.Image.Image) -> dict[str, np.ndarray]:
    outputs = session.run(None, {"image": _resize_for_encoder(image)})
    names = [output.name for output in session.get_outputs()]
    return dict(zip(names, outputs))


def _run_text_encoder(session: ort.InferenceSession, prompt: str) -> dict[str, np.ndarray]:
    tokens = tokenize([prompt], context_length=32)
    outputs = session.run(None, {"tokens": tokens})
    names = [output.name for output in session.get_outputs()]
    return dict(zip(names, outputs))


def _grounding_inference(args: argparse.Namespace, image: PIL.Image.Image) -> tuple[np.ndarray, np.ndarray, np.ndarray, str]:
    image_session = _session(args.model_dir, "sam3_image_encoder.onnx")
    text_session = _session(args.model_dir, "sam3_text_encoder.onnx")
    grounding_session = _session(args.model_dir, "sam3_grounding_decoder.onnx")

    image_out = _run_image_encoder(image_session, image)
    prompt = args.text_prompt or "visual"
    text_out = _run_text_encoder(text_session, prompt)

    grounding_boxes = _parse_box_list(args.grounding_boxes)
    if grounding_boxes is None:
        grounding_boxes = np.zeros((1, 4), dtype=np.float32)
        box_valid_mask = np.asarray([[False]], dtype=bool)
        box_labels = np.asarray([[True]], dtype=bool)
    else:
        box_valid_mask = np.ones((1, grounding_boxes.shape[0]), dtype=bool)
        box_labels_raw = _parse_label_list(args.grounding_box_labels)
        if box_labels_raw is None:
            box_labels_raw = np.ones((grounding_boxes.shape[0],), dtype=np.int32)
        if len(box_labels_raw) != grounding_boxes.shape[0]:
            raise ValueError("grounding box labels count must match grounding boxes count")
        box_labels = box_labels_raw.astype(bool)[None, :]
        grounding_boxes = _xyxy_pixels_to_cxcywh_normalized(
            grounding_boxes,
            width=image.width,
            height=image.height,
        )

    feeds = {
        "vision_pos_enc_0": image_out["vision_pos_enc_0"],
        "vision_pos_enc_1": image_out["vision_pos_enc_1"],
        "vision_pos_enc_2": image_out["vision_pos_enc_2"],
        "backbone_fpn_0": image_out["backbone_fpn_0"],
        "backbone_fpn_1": image_out["backbone_fpn_1"],
        "backbone_fpn_2": image_out["backbone_fpn_2"],
        "language_mask": text_out["language_mask"],
        "language_features": text_out["language_features"],
        "box_coords": grounding_boxes[None, ...],
        "box_valid_mask": box_valid_mask,
        "box_labels": box_labels,
    }
    session_input_names = {item.name for item in grounding_session.get_inputs()}
    feeds = {name: value for name, value in feeds.items() if name in session_input_names}
    outputs = grounding_session.run(None, feeds)
    names = [output.name for output in grounding_session.get_outputs()]
    out = dict(zip(names, outputs))

    scores = out["scores"][0]
    keep = scores >= args.score_threshold
    boxes_xyxy = out["boxes_xyxy"][0][keep]
    boxes_xyxy[:, [0, 2]] *= image.width
    boxes_xyxy[:, [1, 3]] *= image.height
    masks_logits = _upsample_masks(out["masks_logits"][0][keep], image.width, image.height)
    masks = masks_logits > 0
    return boxes_xyxy, scores[keep], masks, prompt


def _interactive_inference(args: argparse.Namespace, image: PIL.Image.Image) -> tuple[np.ndarray, np.ndarray, np.ndarray, str]:
    image_session = _session(args.model_dir, "sam3_image_encoder.onnx")
    interactive_session = _session(args.model_dir, "sam3_interactive_decoder.onnx")
    image_out = _run_image_encoder(image_session, image)

    points = _parse_point_list(args.point_coords)
    point_labels = _parse_label_list(args.point_labels)
    if points is None:
        points = np.zeros((1, 2), dtype=np.float32)
        point_labels = np.full((1,), -1, dtype=np.int32)
    else:
        if point_labels is None or len(point_labels) != len(points):
            raise ValueError("point_labels must be provided and match point_coords")
        points = _scaled_points_to_model_space(points, image.width, image.height)

    boxes = _parse_box_list(args.box_prompt)
    if boxes is None:
        boxes = np.zeros((1, 4), dtype=np.float32)
        box_valid_mask = np.asarray([[False]], dtype=bool)
    else:
        boxes = _scaled_xyxy_to_model_space(boxes, image.width, image.height)
        box_valid_mask = np.ones((1, boxes.shape[0]), dtype=bool)

    session_input_names = {item.name for item in interactive_session.get_inputs()}

    if args.mask_input:
        mask_input = np.load(args.mask_input).astype(np.float32)
        if mask_input.ndim == 2:
            mask_input = mask_input[None, None, ...]
        elif mask_input.ndim == 3:
            mask_input = mask_input[None, ...]
    else:
        mask_input = np.zeros((1, 1, 256, 256), dtype=np.float32)

    feeds = {
        "sam2_backbone_fpn_0": image_out["sam2_backbone_fpn_0"],
        "sam2_backbone_fpn_1": image_out["sam2_backbone_fpn_1"],
        "sam2_backbone_fpn_2": image_out["sam2_backbone_fpn_2"],
        "image_pe": np.load(args.model_dir / "sam3_interactive_dense_pe.npy").astype(np.float32),
        "point_coords": points[None, ...],
        "point_labels": point_labels[None, ...],
        "box_xyxy": boxes[None, ...],
        "box_valid_mask": box_valid_mask,
    }
    if "mask_input" in session_input_names:
        feeds["mask_input"] = mask_input
    elif args.mask_input:
        raise ValueError("This exported interactive ONNX model does not expose `mask_input`; re-export is required to use iterative mask refinement.")

    outputs = interactive_session.run(None, feeds)
    names = [output.name for output in interactive_session.get_outputs()]
    out = dict(zip(names, outputs))

    if args.multimask_output:
        masks_logits = out["multi_masks"][0]
        scores = out["multi_scores"][0]
    else:
        masks_logits = out["single_masks"][0]
        scores = out["single_scores"][0]

    masks = _upsample_masks(masks_logits, image.width, image.height) > 0
    dummy_boxes = np.zeros((masks.shape[0], 4), dtype=np.float32)
    return dummy_boxes, scores, masks, "interactive"


def _visualize(
    image: PIL.Image.Image,
    masks: np.ndarray,
    boxes_xyxy: np.ndarray,
    scores: Sequence[float],
    caption_prefix: str,
    output_path: pathlib.Path,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    bboxes_yxyx = boxes_xyxy[:, [1, 0, 3, 2]] if len(boxes_xyxy) else np.zeros((0, 4))
    viz = imgviz.instances2rgb(
        image=np.asarray(image),
        masks=masks,
        bboxes=bboxes_yxyx,
        labels=np.arange(len(masks)) + 1,
        captions=[f"{caption_prefix}: {score:.0%}" for score in scores],
        font_size=max(1, min(image.size) // 40),
    )
    PIL.Image.fromarray(viz).save(output_path)


def main() -> None:
    args = parse_args()
    image = PIL.Image.open(args.image).convert("RGB")

    if args.mode == "grounding":
        boxes_xyxy, scores, masks, caption_prefix = _grounding_inference(args, image)
    else:
        boxes_xyxy, scores, masks, caption_prefix = _interactive_inference(args, image)

    _visualize(image, masks, boxes_xyxy, scores, caption_prefix, args.output)
    print(f"saved {args.output}")


if __name__ == "__main__":
    main()
