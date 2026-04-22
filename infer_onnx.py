#!/usr/bin/env python3

from __future__ import annotations

import argparse
import gc
import importlib.util
import os
import pathlib
from dataclasses import dataclass
from typing import Any, Sequence

import imgviz
import numpy as np
import onnxruntime as ort
import PIL.Image

try:
    from osam._models.yoloworld.clip import tokenize as _external_tokenize
except ImportError:
    _external_tokenize = None


IMAGE_SIZE = 1008
VIDEO_MEMORY_SLOTS = 7
VIDEO_OBJ_PTR_SLOTS = 16
_LOCAL_TOKENIZER: Any | None = None


@dataclass
class ReferenceState:
    features: np.ndarray
    valid_mask: np.ndarray
    weight: np.ndarray
    source: str


@dataclass
class VideoMemoryEntry:
    maskmem_features: np.ndarray
    maskmem_pos_enc: np.ndarray
    obj_ptr: np.ndarray
    is_cond: bool


def _providers() -> list[tuple[str, dict[str, str]] | str]:
    available = ort.get_available_providers()
    providers: list[tuple[str, dict[str, str]] | str] = []
    requested_provider = os.getenv("SAM3_ORT_PROVIDER", "cuda").lower()
    if requested_provider != "cpu" and "CUDAExecutionProvider" in available:
        cuda_options: dict[str, str] = {
            "arena_extend_strategy": "kSameAsRequested",
            "cudnn_conv_algo_search": os.getenv(
                "SAM3_ORT_CUDNN_CONV_ALGO_SEARCH", "HEURISTIC"
            ),
        }
        if os.getenv("SAM3_ORT_CUDNN_MAX_WORKSPACE", "0") == "0":
            cuda_options["cudnn_conv_use_max_workspace"] = "0"
        gpu_mem_limit_mb = os.getenv("SAM3_ORT_GPU_MEM_LIMIT_MB")
        if gpu_mem_limit_mb:
            cuda_options["gpu_mem_limit"] = str(int(gpu_mem_limit_mb) * 1024 * 1024)
        providers.append(("CUDAExecutionProvider", cuda_options))
    if "CPUExecutionProvider" in available:
        providers.append("CPUExecutionProvider")
    return providers


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


def _scaled_xyxy_to_model_space(
    boxes_xyxy: np.ndarray, width: int, height: int
) -> np.ndarray:
    scaled = boxes_xyxy.copy().astype(np.float32)
    scaled[:, [0, 2]] = scaled[:, [0, 2]] * (IMAGE_SIZE / width)
    scaled[:, [1, 3]] = scaled[:, [1, 3]] * (IMAGE_SIZE / height)
    return scaled


def _scaled_points_to_model_space(
    points_xy: np.ndarray, width: int, height: int
) -> np.ndarray:
    scaled = points_xy.copy().astype(np.float32)
    scaled[:, 0] = scaled[:, 0] * (IMAGE_SIZE / width)
    scaled[:, 1] = scaled[:, 1] * (IMAGE_SIZE / height)
    return scaled


def _xyxy_pixels_to_cxcywh_normalized(
    boxes_xyxy: np.ndarray, width: int, height: int
) -> np.ndarray:
    x0, y0, x1, y1 = boxes_xyxy.T
    cx = ((x0 + x1) * 0.5) / width
    cy = ((y0 + y1) * 0.5) / height
    w = (x1 - x0) / width
    h = (y1 - y0) / height
    return np.stack([cx, cy, w, h], axis=1).astype(np.float32)


def _upsample_masks(masks: np.ndarray, width: int, height: int) -> np.ndarray:
    if len(masks) == 0:
        return np.zeros((0, height, width), dtype=np.float32)
    resized = []
    for mask in masks:
        pil = PIL.Image.fromarray(mask.astype(np.float32), mode="F")
        pil = pil.resize((width, height), resample=PIL.Image.BILINEAR)
        resized.append(np.asarray(pil, dtype=np.float32))
    return np.stack(resized, axis=0)


def _bpe_path() -> pathlib.Path:
    return (
        pathlib.Path(__file__).resolve().parent
        / "sam3"
        / "assets"
        / "bpe_simple_vocab_16e6.txt.gz"
    )


def _load_local_tokenizer() -> Any:
    global _LOCAL_TOKENIZER
    if _LOCAL_TOKENIZER is not None:
        return _LOCAL_TOKENIZER
    module_path = pathlib.Path(__file__).resolve().parent / "sam3" / "model" / "tokenizer_ve.py"
    spec = importlib.util.spec_from_file_location("sam3_tokenizer_ve_local", module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"failed to load tokenizer module from {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    _LOCAL_TOKENIZER = module.SimpleTokenizer(_bpe_path(), context_length=32)
    return _LOCAL_TOKENIZER


def _tokenize_text(prompt: str) -> np.ndarray:
    if _external_tokenize is not None:
        tokens = _external_tokenize([prompt], context_length=32)
        if hasattr(tokens, "cpu"):
            tokens = tokens.cpu().numpy()
        return np.asarray(tokens, dtype=np.int64)
    tokenizer = _load_local_tokenizer()
    tokens = tokenizer([prompt], context_length=32)
    return np.asarray(tokens.cpu().numpy(), dtype=np.int64)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--model-dir", type=pathlib.Path, default=pathlib.Path("models"))
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument("--image", type=pathlib.Path)
    input_group.add_argument("--images", type=pathlib.Path, nargs="+")
    parser.add_argument(
        "--output",
        type=pathlib.Path,
        default=pathlib.Path("output/onnx_result.jpg"),
    )
    parser.add_argument(
        "--output-dir",
        type=pathlib.Path,
        default=pathlib.Path("output/onnx_results"),
    )
    parser.add_argument(
        "--mode",
        choices=["grounding", "interactive", "video"],
        required=True,
    )
    parser.add_argument("--text-prompt", type=str)
    parser.add_argument(
        "--grounding-boxes",
        type=str,
        help="xyxy pixel boxes, separated by ';'",
    )
    parser.add_argument(
        "--grounding-box-labels",
        type=str,
        help="1/0 labels for grounding boxes",
    )
    parser.add_argument(
        "--reference-image",
        type=pathlib.Path,
        help="Reference image for cross-image feature transfer",
    )
    parser.add_argument(
        "--reference-boxes",
        type=str,
        help="Reference xyxy pixel boxes on the reference image, separated by ';'",
    )
    parser.add_argument(
        "--reference-weight",
        type=float,
        default=1.0,
        help="Weight applied to transferred reference features",
    )
    parser.add_argument("--point-coords", type=str, help="x,y;x,y in pixels")
    parser.add_argument("--point-labels", type=str, help="1,0,...")
    parser.add_argument(
        "--box-prompt",
        type=str,
        help="xyxy pixel boxes, separated by ';'",
    )
    parser.add_argument(
        "--mask-input",
        type=pathlib.Path,
        help="Path to .npy low-res mask logits",
    )
    parser.add_argument("--multimask-output", action="store_true")
    parser.add_argument("--score-threshold", type=float, default=0.5)
    parser.add_argument(
        "--video-init-frame",
        type=int,
        default=0,
        help="Prompt frame index used to initialize video tracking",
    )
    args = parser.parse_args()
    if args.image is not None:
        args.images = [args.image]
    if args.mode == "interactive" and len(args.images) > 1:
        raise ValueError("interactive mode currently supports a single target image only")
    if args.mode == "video" and len(args.images) < 2:
        raise ValueError("video mode requires at least two frames via --images")
    return args


def _session(model_dir: pathlib.Path, name: str) -> ort.InferenceSession:
    session_options = ort.SessionOptions()
    session_options.enable_mem_pattern = False
    session_options.enable_cpu_mem_arena = False
    return ort.InferenceSession(
        str(model_dir / name),
        sess_options=session_options,
        providers=_providers(),
    )


def _load_rgb_image(path: pathlib.Path) -> PIL.Image.Image:
    with PIL.Image.open(path) as image:
        return image.convert("RGB")


def _run_image_encoder(
    session: ort.InferenceSession, image: PIL.Image.Image
) -> dict[str, np.ndarray]:
    outputs = session.run(None, {"image": _resize_for_encoder(image)})
    names = [output.name for output in session.get_outputs()]
    return dict(zip(names, outputs))


def _run_text_encoder(
    session: ort.InferenceSession, prompt: str
) -> dict[str, np.ndarray]:
    outputs = session.run(None, {"tokens": _tokenize_text(prompt)})
    names = [output.name for output in session.get_outputs()]
    return dict(zip(names, outputs))


def _reference_encoder_session(model_dir: pathlib.Path) -> ort.InferenceSession:
    return _session(model_dir, "sam3_reference_feature_encoder.onnx")


def _extract_reference_state(
    image_session: ort.InferenceSession,
    reference_session: ort.InferenceSession,
    image: PIL.Image.Image,
    boxes_xyxy: np.ndarray,
    weight: float,
    source: str,
) -> ReferenceState | None:
    if boxes_xyxy is None or len(boxes_xyxy) == 0:
        return None
    image_out = _run_image_encoder(image_session, image)
    scaled_boxes = _scaled_xyxy_to_model_space(boxes_xyxy, image.width, image.height)
    reference_embedding = reference_session.run(
        None,
        {
            "backbone_fpn_0": image_out["backbone_fpn_0"],
            "reference_boxes_xyxy": scaled_boxes,
        },
    )[0].astype(np.float32)
    state = ReferenceState(
        features=reference_embedding[:, None, :],
        valid_mask=np.ones((reference_embedding.shape[0], 1), dtype=bool),
        weight=np.asarray([weight], dtype=np.float32),
        source=source,
    )
    del image_out, scaled_boxes, reference_embedding
    gc.collect()
    return state


def _grounding_inference(
    args: argparse.Namespace,
    image: PIL.Image.Image,
    image_session: ort.InferenceSession,
    text_session: ort.InferenceSession,
    grounding_session: ort.InferenceSession,
    reference_state: ReferenceState | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, str]:
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
    if reference_state is not None:
        feeds["reference_features"] = reference_state.features
        feeds["reference_valid_mask"] = reference_state.valid_mask
        feeds["reference_weight"] = reference_state.weight

    session_input_names = {item.name for item in grounding_session.get_inputs()}
    feeds = {name: value for name, value in feeds.items() if name in session_input_names}
    outputs = grounding_session.run(None, feeds)
    names = [output.name for output in grounding_session.get_outputs()]
    out = dict(zip(names, outputs))

    scores = out["scores"][0]
    keep = scores >= args.score_threshold
    if not np.any(keep):
        max_score = float(np.max(scores)) if scores.size else float("nan")
        print(
            f"no detections passed score threshold {args.score_threshold:.3f}; "
            f"max score was {max_score:.3f}"
        )
        del image_out, text_out, feeds, outputs, out, scores, keep
        gc.collect()
        return (
            np.zeros((0, 4), dtype=np.float32),
            np.zeros((0,), dtype=np.float32),
            np.zeros((0, image.height, image.width), dtype=np.float32),
            np.zeros((0, image.height, image.width), dtype=bool),
            prompt,
        )

    boxes_xyxy = out["boxes_xyxy"][0][keep]
    boxes_xyxy[:, [0, 2]] *= image.width
    boxes_xyxy[:, [1, 3]] *= image.height
    masks_logits = _upsample_masks(out["masks_logits"][0][keep], image.width, image.height)
    masks = masks_logits > 0
    scores_kept = scores[keep]
    del image_out, text_out, feeds, outputs, out, scores, keep
    gc.collect()
    return boxes_xyxy, scores_kept, masks_logits, masks, prompt


def _interactive_inference_raw(
    args: argparse.Namespace,
    image: PIL.Image.Image,
    image_session: ort.InferenceSession,
    interactive_session: ort.InferenceSession,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, str]:
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
        "image_pe": np.load(args.model_dir / "sam3_interactive_dense_pe.npy").astype(
            np.float32
        ),
        "point_coords": points[None, ...],
        "point_labels": point_labels[None, ...],
        "box_xyxy": boxes[None, ...],
        "box_valid_mask": box_valid_mask,
    }
    if "mask_input" in session_input_names:
        feeds["mask_input"] = mask_input
    elif args.mask_input:
        raise ValueError(
            "This exported interactive ONNX model does not expose `mask_input`; "
            "re-export is required to use iterative mask refinement."
        )

    outputs = interactive_session.run(None, feeds)
    names = [output.name for output in interactive_session.get_outputs()]
    out = dict(zip(names, outputs))

    if args.multimask_output:
        low_res_logits = out["multi_masks"][0]
        scores = out["multi_scores"][0]
    else:
        low_res_logits = out["single_masks"][0]
        scores = out["single_scores"][0]

    masks_logits = _upsample_masks(low_res_logits, image.width, image.height)
    masks = masks_logits > 0
    dummy_boxes = np.zeros((masks.shape[0], 4), dtype=np.float32)
    return dummy_boxes, scores, low_res_logits, masks, "interactive"


def _interactive_inference(
    args: argparse.Namespace, image: PIL.Image.Image
) -> tuple[np.ndarray, np.ndarray, np.ndarray, str]:
    image_session = _session(args.model_dir, "sam3_image_encoder.onnx")
    interactive_session = _session(args.model_dir, "sam3_interactive_decoder.onnx")
    boxes_xyxy, scores, _low_res_logits, masks, caption_prefix = _interactive_inference_raw(
        args,
        image,
        image_session,
        interactive_session,
    )
    return boxes_xyxy, scores, masks, caption_prefix


def _empty_video_prompt() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    point_coords = np.zeros((1, 1, 2), dtype=np.float32)
    point_labels = np.full((1, 1), -1, dtype=np.int32)
    box_xyxy = np.zeros((1, 1, 4), dtype=np.float32)
    box_valid_mask = np.zeros((1, 1), dtype=bool)
    return point_coords, point_labels, box_xyxy, box_valid_mask


def _empty_video_history() -> list[VideoMemoryEntry]:
    return []


def _pack_video_history(
    history: list[VideoMemoryEntry],
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    if not history:
        prev_maskmem_features = np.zeros(
            (VIDEO_MEMORY_SLOTS, 1, 64, 72, 72),
            dtype=np.float32,
        )
        prev_maskmem_pos_enc = np.zeros_like(prev_maskmem_features)
        prev_memory_valid = np.zeros((VIDEO_MEMORY_SLOTS,), dtype=bool)
        prev_memory_is_cond = np.zeros((VIDEO_MEMORY_SLOTS,), dtype=bool)
        prev_memory_tpos = np.zeros((VIDEO_MEMORY_SLOTS,), dtype=np.int64)
        prev_obj_ptrs = np.zeros((VIDEO_OBJ_PTR_SLOTS, 1, 256), dtype=np.float32)
        prev_obj_ptr_valid = np.zeros((VIDEO_OBJ_PTR_SLOTS,), dtype=bool)
        prev_obj_ptr_is_cond = np.zeros((VIDEO_OBJ_PTR_SLOTS,), dtype=bool)
        prev_obj_ptr_tpos = np.zeros((VIDEO_OBJ_PTR_SLOTS,), dtype=np.int64)
        return (
            prev_maskmem_features,
            prev_maskmem_pos_enc,
            prev_memory_valid,
            prev_memory_is_cond,
            prev_memory_tpos,
            prev_obj_ptrs,
            prev_obj_ptr_valid,
            prev_obj_ptr_is_cond,
            prev_obj_ptr_tpos,
        )

    prev_maskmem_features = np.zeros((VIDEO_MEMORY_SLOTS, 1, 64, 72, 72), dtype=np.float32)
    prev_maskmem_pos_enc = np.zeros_like(prev_maskmem_features)
    prev_memory_valid = np.zeros((VIDEO_MEMORY_SLOTS,), dtype=bool)
    prev_memory_is_cond = np.zeros((VIDEO_MEMORY_SLOTS,), dtype=bool)
    prev_memory_tpos = np.zeros((VIDEO_MEMORY_SLOTS,), dtype=np.int64)

    prev_obj_ptrs = np.zeros((VIDEO_OBJ_PTR_SLOTS, 1, 256), dtype=np.float32)
    prev_obj_ptr_valid = np.zeros((VIDEO_OBJ_PTR_SLOTS,), dtype=bool)
    prev_obj_ptr_is_cond = np.zeros((VIDEO_OBJ_PTR_SLOTS,), dtype=bool)
    prev_obj_ptr_tpos = np.zeros((VIDEO_OBJ_PTR_SLOTS,), dtype=np.int64)

    for index, entry in enumerate(history[:VIDEO_MEMORY_SLOTS]):
        prev_maskmem_features[index] = entry.maskmem_features.astype(np.float32)
        prev_maskmem_pos_enc[index] = entry.maskmem_pos_enc.astype(np.float32)
        prev_memory_valid[index] = True
        prev_memory_is_cond[index] = entry.is_cond
        prev_memory_tpos[index] = index + 1

    for index, entry in enumerate(history[:VIDEO_OBJ_PTR_SLOTS]):
        prev_obj_ptrs[index] = entry.obj_ptr.astype(np.float32)
        prev_obj_ptr_valid[index] = True
        prev_obj_ptr_is_cond[index] = entry.is_cond
        prev_obj_ptr_tpos[index] = index + 1

    return (
        prev_maskmem_features,
        prev_maskmem_pos_enc,
        prev_memory_valid,
        prev_memory_is_cond,
        prev_memory_tpos,
        prev_obj_ptrs,
        prev_obj_ptr_valid,
        prev_obj_ptr_is_cond,
        prev_obj_ptr_tpos,
    )


def _append_video_history(
    history: list[VideoMemoryEntry],
    new_maskmem_features: np.ndarray,
    new_maskmem_pos_enc: np.ndarray,
    new_obj_ptr: np.ndarray,
    *,
    is_cond: bool,
) -> None:
    entry = VideoMemoryEntry(
        maskmem_features=new_maskmem_features.astype(np.float32),
        maskmem_pos_enc=new_maskmem_pos_enc.astype(np.float32),
        obj_ptr=new_obj_ptr.astype(np.float32),
        is_cond=is_cond,
    )
    history.insert(0, entry)
    max_entries = max(VIDEO_MEMORY_SLOTS, VIDEO_OBJ_PTR_SLOTS)
    del history[max_entries:]


def _pick_best_mask(scores: np.ndarray, masks_logits: np.ndarray) -> np.ndarray:
    if len(scores) == 0 or len(masks_logits) == 0:
        raise ValueError("no masks available to initialize video tracking")
    best_index = int(np.argmax(scores))
    best_mask = masks_logits[best_index].astype(np.float32)
    return best_mask[None, None, ...]


def _run_video_tracking_step(
    image_out: dict[str, np.ndarray],
    video_session: ort.InferenceSession,
    point_coords: np.ndarray,
    point_labels: np.ndarray,
    box_xyxy: np.ndarray,
    box_valid_mask: np.ndarray,
    mask_input: np.ndarray,
    history: list[VideoMemoryEntry],
) -> dict[str, np.ndarray]:
    (
        prev_maskmem_features,
        prev_maskmem_pos_enc,
        prev_memory_valid,
        prev_memory_is_cond,
        prev_memory_tpos,
        prev_obj_ptrs,
        prev_obj_ptr_valid,
        prev_obj_ptr_is_cond,
        prev_obj_ptr_tpos,
    ) = _pack_video_history(history)

    feeds = {
        "sam2_vision_pos_enc_0": image_out["sam2_vision_pos_enc_0"],
        "sam2_vision_pos_enc_1": image_out["sam2_vision_pos_enc_1"],
        "sam2_vision_pos_enc_2": image_out["sam2_vision_pos_enc_2"],
        "sam2_backbone_fpn_0": image_out["sam2_backbone_fpn_0"],
        "sam2_backbone_fpn_1": image_out["sam2_backbone_fpn_1"],
        "sam2_backbone_fpn_2": image_out["sam2_backbone_fpn_2"],
        "point_coords": point_coords,
        "point_labels": point_labels,
        "box_xyxy": box_xyxy,
        "box_valid_mask": box_valid_mask,
        "mask_input": mask_input.astype(np.float32),
        "prev_maskmem_features": prev_maskmem_features,
        "prev_maskmem_pos_enc": prev_maskmem_pos_enc,
        "prev_memory_valid": prev_memory_valid,
        "prev_memory_is_cond": prev_memory_is_cond,
        "prev_memory_tpos": prev_memory_tpos,
        "prev_obj_ptrs": prev_obj_ptrs,
        "prev_obj_ptr_valid": prev_obj_ptr_valid,
        "prev_obj_ptr_is_cond": prev_obj_ptr_is_cond,
        "prev_obj_ptr_tpos": prev_obj_ptr_tpos,
    }
    outputs = video_session.run(None, feeds)
    names = [output.name for output in video_session.get_outputs()]
    return dict(zip(names, outputs))


def _run_grounding_sequence(args: argparse.Namespace) -> None:
    image_session = _session(args.model_dir, "sam3_image_encoder.onnx")
    text_session = _session(args.model_dir, "sam3_text_encoder.onnx")
    grounding_session = _session(args.model_dir, "sam3_grounding_decoder.onnx")
    reference_model_path = args.model_dir / "sam3_reference_feature_encoder.onnx"
    with_reference_model_path = args.model_dir / "sam3_grounding_decoder_with_reference.onnx"
    grounding_with_reference_session = None
    if with_reference_model_path.exists():
        grounding_with_reference_session = _session(
            args.model_dir, "sam3_grounding_decoder_with_reference.onnx"
        )
    reference_session = None
    if reference_model_path.exists():
        reference_session = _reference_encoder_session(args.model_dir)

    reference_boxes = _parse_box_list(args.reference_boxes)
    reference_state: ReferenceState | None = None
    if args.reference_image is not None:
        if reference_boxes is None:
            raise ValueError("--reference-image requires --reference-boxes")
        if reference_session is None:
            raise FileNotFoundError(
                f"missing {reference_model_path}; "
                "re-export models with reference_feature_encoder support"
            )
        reference_image = _load_rgb_image(args.reference_image)
        reference_state = _extract_reference_state(
            image_session=image_session,
            reference_session=reference_session,
            image=reference_image,
            boxes_xyxy=reference_boxes,
            weight=args.reference_weight,
            source=f"manual:{args.reference_image.name}",
        )
        print(f"loaded reference features from {args.reference_image}")
        del reference_image
        gc.collect()

    multi_image = len(args.images) > 1
    for index, image_path in enumerate(args.images):
        image = _load_rgb_image(image_path)
        boxes_xyxy = None
        scores = None
        masks = None
        try:
            active_reference = (
                reference_state
                if index > 0 or args.reference_image is not None
                else None
            )
            session_to_use = grounding_session
            if active_reference is not None:
                if grounding_with_reference_session is None:
                    raise FileNotFoundError(
                        f"missing {with_reference_model_path}; "
                        "re-export models with grounding_decoder_with_reference support"
                    )
                session_to_use = grounding_with_reference_session
            boxes_xyxy, scores, _masks_logits, masks, caption_prefix = _grounding_inference(
                args=args,
                image=image,
                image_session=image_session,
                text_session=text_session,
                grounding_session=session_to_use,
                reference_state=active_reference,
            )

            output_path = (
                args.output
                if not multi_image
                else args.output_dir
                / f"{image_path.stem}_onnx_result{args.output.suffix or '.jpg'}"
            )
            _visualize(image, masks, boxes_xyxy, scores, caption_prefix, output_path)
            print(f"saved {output_path}")

            if multi_image and index == 0 and reference_state is None and reference_session is not None:
                if reference_boxes is not None:
                    reference_state = _extract_reference_state(
                        image_session=image_session,
                        reference_session=reference_session,
                        image=image,
                        boxes_xyxy=reference_boxes,
                        weight=args.reference_weight,
                        source=f"first-image-boxes:{image_path.name}",
                    )
                    if reference_state is not None:
                        print(
                            f"extracted cross-image reference from provided boxes on "
                            f"{image_path.name}"
                        )
                elif len(boxes_xyxy) > 0:
                    best_index = int(np.argmax(scores))
                    reference_state = _extract_reference_state(
                        image_session=image_session,
                        reference_session=reference_session,
                        image=image,
                        boxes_xyxy=boxes_xyxy[best_index : best_index + 1],
                        weight=args.reference_weight,
                        source=f"first-image:{image_path.name}",
                    )
                    if reference_state is not None:
                        print(
                            "extracted cross-image reference from "
                            f"{image_path.name} score={float(scores[best_index]):.3f}"
                        )

            if multi_image and active_reference is not None:
                print(
                    f"used reference features from {active_reference.source} "
                    f"for {image_path.name}"
                )
        finally:
            del image, boxes_xyxy, scores, masks
            gc.collect()

    del (
        image_session,
        text_session,
        grounding_session,
        grounding_with_reference_session,
        reference_session,
    )
    gc.collect()


def _run_video_sequence(args: argparse.Namespace) -> None:
    init_frame_idx = args.video_init_frame
    if not 0 <= init_frame_idx < len(args.images):
        raise ValueError(
            f"--video-init-frame must be in [0, {len(args.images) - 1}], "
            f"got {init_frame_idx}"
        )

    image_session = _session(args.model_dir, "sam3_image_encoder.onnx")
    interactive_session = _session(args.model_dir, "sam3_interactive_decoder.onnx")
    video_session = _session(args.model_dir, "sam3_video_tracking_step.onnx")

    text_session = None
    grounding_session = None
    if args.text_prompt:
        text_session = _session(args.model_dir, "sam3_text_encoder.onnx")
        grounding_session = _session(args.model_dir, "sam3_grounding_decoder.onnx")

    history = _empty_video_history()
    previous_mask_input: np.ndarray | None = None

    args.output_dir.mkdir(parents=True, exist_ok=True)
    prompt_frame_path = args.images[init_frame_idx]
    prompt_frame = _load_rgb_image(prompt_frame_path)
    prompt_image_out = _run_image_encoder(image_session, prompt_frame)

    if args.text_prompt:
        assert text_session is not None
        assert grounding_session is not None
        (
            prompt_boxes,
            prompt_scores,
            prompt_masks_logits,
            prompt_masks,
            prompt_caption,
        ) = _grounding_inference(
            args=args,
            image=prompt_frame,
            image_session=image_session,
            text_session=text_session,
            grounding_session=grounding_session,
        )
        previous_mask_input = _pick_best_mask(prompt_scores, prompt_masks_logits)
        prompt_point_coords, prompt_point_labels, prompt_box_xyxy, prompt_box_valid_mask = (
            _empty_video_prompt()
        )
    else:
        (
            prompt_boxes,
            prompt_scores,
            prompt_masks_logits,
            prompt_masks,
            prompt_caption,
        ) = _interactive_inference_raw(
            args,
            prompt_frame,
            image_session,
            interactive_session,
        )
        previous_mask_input = _pick_best_mask(prompt_scores, prompt_masks_logits)
        prompt_points = _parse_point_list(args.point_coords)
        prompt_labels = _parse_label_list(args.point_labels)
        if prompt_points is None:
            prompt_points, prompt_labels, _, _ = _empty_video_prompt()
        else:
            if prompt_labels is None or len(prompt_labels) != len(prompt_points):
                raise ValueError("point_labels must be provided and match point_coords")
            prompt_points = _scaled_points_to_model_space(
                prompt_points, prompt_frame.width, prompt_frame.height
            )
            prompt_points = prompt_points[None, ...].astype(np.float32)
            prompt_labels = prompt_labels[None, ...].astype(np.int32)
        prompt_boxes_xyxy = _parse_box_list(args.box_prompt)
        if prompt_boxes_xyxy is None:
            prompt_box_xyxy = np.zeros((1, 1, 4), dtype=np.float32)
            prompt_box_valid_mask = np.zeros((1, 1), dtype=bool)
        else:
            prompt_box_xyxy = _scaled_xyxy_to_model_space(
                prompt_boxes_xyxy, prompt_frame.width, prompt_frame.height
            )[None, ...].astype(np.float32)
            prompt_box_valid_mask = np.ones(
                (1, prompt_box_xyxy.shape[1]),
                dtype=bool,
            )
        prompt_point_coords = prompt_points
        prompt_point_labels = prompt_labels

    prompt_tracking_out = _run_video_tracking_step(
        prompt_image_out,
        video_session,
        prompt_point_coords,
        prompt_point_labels,
        prompt_box_xyxy,
        prompt_box_valid_mask,
        previous_mask_input,
        history,
    )
    _append_video_history(
        history,
        prompt_tracking_out["new_maskmem_features"][0],
        prompt_tracking_out["new_maskmem_pos_enc"][0],
        prompt_tracking_out["new_obj_ptr"][0],
        is_cond=True,
    )
    previous_mask_input = prompt_tracking_out["tracking_high_res_masks"].astype(np.float32)

    prompt_output_path = args.output_dir / f"{prompt_frame_path.stem}_tracked.jpg"
    _visualize(
        prompt_frame,
        prompt_masks,
        prompt_boxes,
        prompt_scores,
        prompt_caption,
        prompt_output_path,
    )
    print(f"saved {prompt_output_path}")

    for frame_idx, image_path in enumerate(args.images):
        if frame_idx <= init_frame_idx:
            continue

        image = _load_rgb_image(image_path)
        try:
            image_out = _run_image_encoder(image_session, image)
            point_coords, point_labels, box_xyxy, box_valid_mask = _empty_video_prompt()
            track_out = _run_video_tracking_step(
                image_out,
                video_session,
                point_coords,
                point_labels,
                box_xyxy,
                box_valid_mask,
                previous_mask_input,
                history,
            )
            masks_logits = _upsample_masks(
                track_out["tracking_high_res_masks"][0],
                image.width,
                image.height,
            )
            masks = masks_logits > 0
            scores = 1 / (1 + np.exp(-track_out["tracking_object_score_logits"][0].reshape(-1)))
            output_path = args.output_dir / f"{image_path.stem}_tracked.jpg"
            _visualize(image, masks, np.zeros((0, 4), dtype=np.float32), scores, "video", output_path)
            print(f"saved {output_path}")

            _append_video_history(
                history,
                track_out["new_maskmem_features"][0],
                track_out["new_maskmem_pos_enc"][0],
                track_out["new_obj_ptr"][0],
                is_cond=False,
            )
            previous_mask_input = track_out["tracking_high_res_masks"].astype(np.float32)
        finally:
            del image
            gc.collect()

    del image_session, interactive_session, video_session, text_session, grounding_session
    gc.collect()


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
    if args.mode == "grounding":
        _run_grounding_sequence(args)
        return
    if args.mode == "video":
        _run_video_sequence(args)
        return

    image = PIL.Image.open(args.images[0]).convert("RGB")
    boxes_xyxy, scores, masks, caption_prefix = _interactive_inference(args, image)
    _visualize(image, masks, boxes_xyxy, scores, caption_prefix, args.output)
    print(f"saved {args.output}")


if __name__ == "__main__":
    main()
