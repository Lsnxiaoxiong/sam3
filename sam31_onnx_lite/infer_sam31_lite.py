import argparse
import json
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import onnxruntime as ort
from PIL import Image
import torch
from torchvision.transforms import v2

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sam3.model.tokenizer_ve import SimpleTokenizer  # noqa: E402


IMAGE_SIZE = 1008
TEXT_CONTEXT = 32


def _providers() -> List[str]:
    available = ort.get_available_providers()
    providers: List[str] = []
    if "CUDAExecutionProvider" in available:
        providers.append("CUDAExecutionProvider")
    providers.append("CPUExecutionProvider")
    return providers


def _session(path: str) -> ort.InferenceSession:
    options = ort.SessionOptions()
    options.enable_mem_pattern = False
    options.enable_cpu_mem_arena = False
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    return ort.InferenceSession(path, sess_options=options, providers=_providers())


def _model_path(model_dir: Path, name: str) -> Path:
    return model_dir / name / f"{name}.onnx"


def _run_with_available_inputs(
    sess: ort.InferenceSession, feeds: Dict[str, np.ndarray]
) -> Dict[str, np.ndarray]:
    input_names = {i.name for i in sess.get_inputs()}
    filtered = {k: v for k, v in feeds.items() if k in input_names}
    outputs = sess.run(None, filtered)
    return dict(zip([o.name for o in sess.get_outputs()], outputs))


def _load_rgb_image(image_path: str) -> Image.Image:
    return Image.open(image_path).convert("RGB")


def _preprocess_image(image_path: str) -> np.ndarray:
    image = _load_rgb_image(image_path)
    transform = v2.Compose(
        [
            v2.ToImage(),
            v2.ToDtype(torch.uint8, scale=True),
            v2.Resize(size=(IMAGE_SIZE, IMAGE_SIZE)),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ]
    )
    return transform(image).unsqueeze(0).numpy()


def _tokenize(prompt: str, bpe_path: str) -> np.ndarray:
    tokenizer = SimpleTokenizer(bpe_path=bpe_path)
    return tokenizer([prompt], context_length=TEXT_CONTEXT).numpy()


def _default_mask() -> np.ndarray:
    return np.zeros((1, 1, 288, 288), dtype=np.float32)


def _resize_mask_to_image(mask: np.ndarray, image: Image.Image) -> np.ndarray:
    mask_image = Image.fromarray(mask.astype(np.float32), mode="F")
    mask_image = mask_image.resize(image.size, resample=Image.BILINEAR)
    return np.asarray(mask_image, dtype=np.float32)


def _save_mask(mask: np.ndarray, image_path: str, output_path: Path, threshold: float = 0.0) -> str:
    image = _load_rgb_image(image_path)
    resized_mask = _resize_mask_to_image(mask, image)
    binary = resized_mask > threshold
    output_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray((binary.astype(np.uint8) * 255)).save(output_path)
    return str(output_path)


def _run_image_encoder(model_dir: Path, image_path: str) -> Dict[str, np.ndarray]:
    sess = _session(str(_model_path(model_dir, "sam31_image_encoder")))
    return _run_with_available_inputs(sess, {"image": _preprocess_image(image_path)})


def _scaled_xyxy_to_model_space(boxes_xyxy: np.ndarray, width: int, height: int) -> np.ndarray:
    scaled = boxes_xyxy.astype(np.float32).copy()
    scaled[:, [0, 2]] = scaled[:, [0, 2]] * (IMAGE_SIZE / width)
    scaled[:, [1, 3]] = scaled[:, [1, 3]] * (IMAGE_SIZE / height)
    return scaled


def _scaled_points_to_model_space(points_xy: np.ndarray, width: int, height: int) -> np.ndarray:
    scaled = points_xy.astype(np.float32).copy()
    scaled[:, 0] = scaled[:, 0] * (IMAGE_SIZE / width)
    scaled[:, 1] = scaled[:, 1] * (IMAGE_SIZE / height)
    return scaled


def _select_best_text_mask(result: Dict[str, np.ndarray]) -> Tuple[np.ndarray, int, float]:
    logits = result["pred_logits"][0, :, 0]
    boxes = result["pred_boxes_xyxy"][0]
    masks = result["pred_masks"][0]
    mask_area = (masks > 0).reshape(masks.shape[0], -1).mean(axis=1)
    box_area = np.clip(boxes[:, 2] - boxes[:, 0], 0.0, 1.0) * np.clip(
        boxes[:, 3] - boxes[:, 1], 0.0, 1.0
    )
    valid = (
        (mask_area > 0.001)
        & (mask_area < 0.85)
        & (box_area > 0.001)
        & (box_area < 0.95)
    )
    if np.any(valid):
        candidate_indices = np.where(valid)[0]
        best_idx = int(candidate_indices[np.argmax(logits[candidate_indices])])
    else:
        best_idx = int(np.argmax(logits))
    return result["pred_masks"][0, best_idx], best_idx, float(logits[best_idx])


def _select_best_interactive_mask(
    result: Dict[str, np.ndarray], mode: str
) -> Tuple[np.ndarray, float]:
    single_masks = result["single_masks"][0]
    single_scores = result["single_scores"][0]
    multi_masks = result["multi_masks"][0]
    multi_scores = result["multi_scores"][0]
    if mode == "point":
        best_idx = int(np.argmax(multi_scores))
        return multi_masks[best_idx], float(multi_scores[best_idx])
    return single_masks[0], float(single_scores[0])


def run_text(model_dir: Path, image_path: str, prompt: str, bpe_path: str) -> Dict[str, np.ndarray]:
    image_out = _run_image_encoder(model_dir, image_path)
    text_sess = _session(str(_model_path(model_dir, "sam31_text_encoder")))
    text_out = _run_with_available_inputs(text_sess, {"token_ids": _tokenize(prompt, bpe_path)})
    decoder = _session(str(_model_path(model_dir, "sam31_grounding_decoder")))
    return _run_with_available_inputs(
        decoder,
        {
            "vision_pos_enc_0": image_out["vision_pos_enc_0"],
            "vision_pos_enc_1": image_out["vision_pos_enc_1"],
            "vision_pos_enc_2": image_out["vision_pos_enc_2"],
            "backbone_fpn_0": image_out["backbone_fpn_0"],
            "backbone_fpn_1": image_out["backbone_fpn_1"],
            "backbone_fpn_2": image_out["backbone_fpn_2"],
            "language_mask": text_out["language_mask"],
            "language_features": text_out["language_features"],
        },
    )


def run_cross_image(
    model_dir: Path,
    image_path: str,
    reference_image_path: str,
    reference_box_xyxy: np.ndarray,
    prompt: str,
    bpe_path: str,
    reference_weight: float,
) -> Dict[str, np.ndarray]:
    target_image_out = _run_image_encoder(model_dir, image_path)
    reference_image = _load_rgb_image(reference_image_path)
    scaled_reference_box = _scaled_xyxy_to_model_space(
        reference_box_xyxy.astype(np.float32),
        reference_image.width,
        reference_image.height,
    )
    reference_image_out = _run_image_encoder(model_dir, reference_image_path)
    ref_sess = _session(str(_model_path(model_dir, "sam31_reference_feature_encoder")))
    ref_out = _run_with_available_inputs(
        ref_sess,
        {
            "backbone_fpn_0": reference_image_out["backbone_fpn_0"],
            "reference_boxes_xyxy": scaled_reference_box,
        },
    )
    text_sess = _session(str(_model_path(model_dir, "sam31_text_encoder")))
    text_out = _run_with_available_inputs(text_sess, {"token_ids": _tokenize(prompt, bpe_path)})
    decoder = _session(str(_model_path(model_dir, "sam31_grounding_decoder_with_reference")))
    reference_embedding = ref_out["reference_embedding"][:, None, :].astype(np.float32)
    reference_valid = np.ones((reference_embedding.shape[0], 1), dtype=bool)
    return _run_with_available_inputs(
        decoder,
        {
            "vision_pos_enc_0": target_image_out["vision_pos_enc_0"],
            "vision_pos_enc_1": target_image_out["vision_pos_enc_1"],
            "vision_pos_enc_2": target_image_out["vision_pos_enc_2"],
            "backbone_fpn_0": target_image_out["backbone_fpn_0"],
            "backbone_fpn_1": target_image_out["backbone_fpn_1"],
            "backbone_fpn_2": target_image_out["backbone_fpn_2"],
            "language_mask": text_out["language_mask"],
            "language_features": text_out["language_features"],
            "reference_features": reference_embedding,
            "reference_valid_mask": reference_valid,
            "reference_weight": np.asarray([reference_weight], dtype=np.float32),
        },
    )


def run_interactive(
    model_dir: Path,
    image_path: str,
    point_coords: np.ndarray,
    point_labels: np.ndarray,
    box_xyxy: np.ndarray,
    box_valid_mask: np.ndarray,
    decoder_name: str = "sam31_interactive_decoder",
) -> Dict[str, np.ndarray]:
    image = _load_rgb_image(image_path)
    image_out = _run_image_encoder(model_dir, image_path)
    dense_pe = np.load(model_dir / "sam31_interactive_dense_pe.npy").astype(np.float32)
    sess = _session(str(_model_path(model_dir, decoder_name)))
    scaled_points = _scaled_points_to_model_space(point_coords, image.width, image.height)[None, ...]
    scaled_boxes = _scaled_xyxy_to_model_space(box_xyxy, image.width, image.height)[None, ...]
    return _run_with_available_inputs(
        sess,
        {
            "sam2_backbone_fpn_0": image_out["sam2_backbone_fpn_0"],
            "sam2_backbone_fpn_1": image_out["sam2_backbone_fpn_1"],
            "sam2_backbone_fpn_2": image_out["sam2_backbone_fpn_2"],
            "image_pe": dense_pe,
            "point_coords": scaled_points.astype(np.float32),
            "point_labels": point_labels[None, ...].astype(np.int64),
            "box_xyxy": scaled_boxes.astype(np.float32),
            "box_valid_mask": box_valid_mask[None, ...].astype(bool),
            "mask_input": _default_mask(),
        },
    )


def run_video(
    model_dir: Path,
    frame_paths: List[str],
    point_coords: np.ndarray,
    point_labels: np.ndarray,
    box_xyxy: np.ndarray,
    box_valid_mask: np.ndarray,
) -> Dict[str, np.ndarray]:
    video_masks = []
    video_scores = []
    for frame_path in frame_paths:
        outputs = run_interactive(
            model_dir,
            frame_path,
            point_coords,
            point_labels,
            box_xyxy,
            box_valid_mask,
            "sam31_interactive_decoder",
        )
        best_mask, best_score = _select_best_interactive_mask(outputs, "point")
        video_masks.append(best_mask[None, None, ...])
        video_scores.append(best_score)
    return {
        "video_high_res_masks": np.concatenate(video_masks, axis=0),
        "video_scores": np.asarray(video_scores, dtype=np.float32),
    }


def _save_mask_set(
    masks: Iterable[np.ndarray], image_path: str, output_dir: Path, prefix: str
) -> List[str]:
    paths: List[str] = []
    output_dir.mkdir(parents=True, exist_ok=True)
    for idx, mask in enumerate(masks):
        path = output_dir / f"{prefix}_{idx:02d}.png"
        paths.append(_save_mask(mask, image_path, path))
    return paths


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", default="output/sam31_onnx_lite")
    parser.add_argument(
        "--mode",
        required=True,
        choices=["text", "point", "box", "cross_image", "video"],
    )
    parser.add_argument("--image", default="assets/images/truck.jpg")
    parser.add_argument("--video-dir", default="assets/videos/0001")
    parser.add_argument("--text-prompt", default="truck")
    parser.add_argument(
        "--bpe-path",
        default=str(ROOT / "sam3" / "assets" / "bpe_simple_vocab_16e6.txt.gz"),
    )
    parser.add_argument("--point", nargs=2, type=float, default=[900.0, 560.0])
    parser.add_argument("--point-label", type=int, default=1)
    parser.add_argument("--box", nargs=4, type=float, default=[80.0, 300.0, 1710.0, 850.0])
    parser.add_argument("--use-box", action="store_true")
    parser.add_argument("--reference-image", default="assets/images/truck.jpg")
    parser.add_argument(
        "--reference-box",
        nargs=4,
        type=float,
        default=[120.0, 220.0, 1650.0, 940.0],
    )
    parser.add_argument("--reference-weight", type=float, default=1.0)
    parser.add_argument("--output", default="")
    parser.add_argument("--mask-output", default="")
    parser.add_argument("--mask-dir", default="")
    parser.add_argument("--num-frames", type=int, default=8)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    model_dir = Path(args.model_dir)
    point_coords = np.array([args.point], dtype=np.float32)
    point_labels = np.array([args.point_label], dtype=np.int64)
    box_xyxy = np.array([args.box], dtype=np.float32)

    if args.mode == "text":
        result = run_text(model_dir, args.image, args.text_prompt, args.bpe_path)
        best_mask, best_idx, best_logit = _select_best_text_mask(result)
        mask_output = (
            Path(args.mask_output)
            if args.mask_output
            else Path("output") / "sam31_onnx_lite_text_mask.png"
        )
        saved = _save_mask(best_mask, args.image, mask_output)
        summary = {
            "mode": args.mode,
            "pred_logits_shape": list(result["pred_logits"].shape),
            "pred_masks_shape": list(result["pred_masks"].shape),
            "best_query_index": best_idx,
            "best_query_logit": best_logit,
            "mask_output": saved,
        }
    elif args.mode == "cross_image":
        result = run_cross_image(
            model_dir,
            args.image,
            args.reference_image,
            np.array([args.reference_box], dtype=np.float32),
            args.text_prompt,
            args.bpe_path,
            args.reference_weight,
        )
        best_mask, best_idx, best_logit = _select_best_text_mask(result)
        mask_output = (
            Path(args.mask_output)
            if args.mask_output
            else Path("output") / "sam31_onnx_lite_cross_image_mask.png"
        )
        saved = _save_mask(best_mask, args.image, mask_output)
        summary = {
            "mode": args.mode,
            "pred_logits_shape": list(result["pred_logits"].shape),
            "pred_masks_shape": list(result["pred_masks"].shape),
            "best_query_index": best_idx,
            "best_query_logit": best_logit,
            "mask_output": saved,
        }
    elif args.mode == "point":
        result = run_interactive(
            model_dir,
            args.image,
            point_coords,
            point_labels,
            np.zeros((1, 4), dtype=np.float32),
            np.array([False], dtype=bool),
            "sam31_point_decoder",
        )
        best_mask, best_score = _select_best_interactive_mask(result, "point")
        mask_output = (
            Path(args.mask_output)
            if args.mask_output
            else Path("output") / "sam31_onnx_lite_point_mask.png"
        )
        saved = _save_mask(best_mask, args.image, mask_output)
        summary = {
            "mode": args.mode,
            "single_masks_shape": list(result["single_masks"].shape),
            "multi_masks_shape": list(result["multi_masks"].shape),
            "best_score": best_score,
            "mask_output": saved,
        }
    elif args.mode == "box":
        result = run_interactive(
            model_dir,
            args.image,
            np.zeros((1, 2), dtype=np.float32),
            np.array([-1], dtype=np.int64),
            box_xyxy,
            np.array([True], dtype=bool),
            "sam31_box_decoder",
        )
        best_mask, best_score = _select_best_interactive_mask(result, "box")
        mask_output = (
            Path(args.mask_output)
            if args.mask_output
            else Path("output") / "sam31_onnx_lite_box_mask.png"
        )
        saved = _save_mask(best_mask, args.image, mask_output)
        summary = {
            "mode": args.mode,
            "single_masks_shape": list(result["single_masks"].shape),
            "multi_masks_shape": list(result["multi_masks"].shape),
            "best_score": best_score,
            "mask_output": saved,
        }
    else:
        frame_paths = sorted(
            Path(args.video_dir).glob("*.jpg"),
            key=lambda p: int(p.stem) if p.stem.isdigit() else p.stem,
        )[: args.num_frames]
        frame_paths = [str(p) for p in frame_paths]
        video_box = (
            box_xyxy
            if args.use_box
            else np.zeros((1, 4), dtype=np.float32)
        )
        video_box_valid = np.array([args.use_box], dtype=bool)
        result = run_video(
            model_dir,
            frame_paths,
            point_coords,
            point_labels,
            video_box,
            video_box_valid,
        )
        mask_dir = (
            Path(args.mask_dir)
            if args.mask_dir
            else Path("output") / "sam31_onnx_lite_video_masks"
        )
        saved = []
        for idx, frame_path in enumerate(frame_paths):
            saved.append(
                _save_mask(
                    result["video_high_res_masks"][idx, 0],
                    frame_path,
                    mask_dir / f"{Path(frame_path).stem}_mask.png",
                )
            )
        summary = {
            "mode": args.mode,
            "video_high_res_masks_shape": list(result["video_high_res_masks"].shape),
            "mask_outputs": saved[:3] + (["..."] if len(saved) > 3 else []),
        }

    print(json.dumps(summary, indent=2))
    if args.output:
        np.savez(args.output, **result)


if __name__ == "__main__":
    main()
