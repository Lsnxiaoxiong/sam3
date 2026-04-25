import argparse
import json
import sys
from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch
from PIL import Image
from torchvision.transforms import v2

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _session(path: Path) -> ort.InferenceSession:
    options = ort.SessionOptions()
    options.enable_mem_pattern = False
    options.enable_cpu_mem_arena = False
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    return ort.InferenceSession(str(path), sess_options=options, providers=["CPUExecutionProvider"])


def _preprocess_image(image_path: Path, image_size: int) -> np.ndarray:
    image = Image.open(image_path).convert("RGB")
    transform = v2.Compose(
        [
            v2.ToImage(),
            v2.ToDtype(torch.uint8, scale=True),
            v2.Resize(size=(image_size, image_size)),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ]
    )
    return transform(image).unsqueeze(0).numpy()


def _scale_points_to_model_space(
    points_xy: np.ndarray, image_path: Path, image_size: int
) -> np.ndarray:
    image = Image.open(image_path)
    width, height = image.size
    scaled = points_xy.astype(np.float32).copy()
    scaled[..., 0] *= image_size / width
    scaled[..., 1] *= image_size / height
    return scaled


def _scale_boxes_to_model_space(
    boxes_xyxy: np.ndarray, image_path: Path, image_size: int
) -> np.ndarray:
    image = Image.open(image_path)
    width, height = image.size
    scaled = boxes_xyxy.astype(np.float32).copy()
    scaled[..., [0, 2]] *= image_size / width
    scaled[..., [1, 3]] *= image_size / height
    return scaled


def _empty_prompts() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    point_coords = np.zeros((1, 1, 2), dtype=np.float32)
    point_labels = np.full((1, 1), -1, dtype=np.int64)
    box_xyxy = np.zeros((1, 1, 4), dtype=np.float32)
    box_valid_mask = np.zeros((1, 1), dtype=bool)
    return point_coords, point_labels, box_xyxy, box_valid_mask


def _prompt_inputs(
    args: argparse.Namespace, frame_path: Path, image_size: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    point_coords = np.array([[[args.point[0], args.point[1]]]], dtype=np.float32)
    point_labels = np.array([[args.point_label]], dtype=np.int64)
    box_xyxy = np.array([[[args.box[0], args.box[1], args.box[2], args.box[3]]]], dtype=np.float32)
    box_valid_mask = np.array([[args.use_box]], dtype=bool)
    point_coords = _scale_points_to_model_space(point_coords, frame_path, image_size)
    box_xyxy = _scale_boxes_to_model_space(box_xyxy, frame_path, image_size)
    return point_coords, point_labels, box_xyxy, box_valid_mask


def _update_queue(
    old_features: np.ndarray,
    old_pos: np.ndarray,
    old_valid: np.ndarray,
    old_tpos: np.ndarray,
    new_features: np.ndarray,
    new_pos: np.ndarray,
    max_tpos: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    shifted_features = np.concatenate([new_features, old_features[:-1]], axis=0)
    shifted_pos = np.concatenate([new_pos, old_pos[:-1]], axis=0)
    shifted_valid = np.concatenate([np.array([True]), old_valid[:-1]], axis=0)
    shifted_tpos = np.concatenate([np.array([1], dtype=np.int64), np.minimum(old_tpos[:-1] + 1, max_tpos)], axis=0)
    shifted_valid = shifted_valid & (shifted_tpos <= max_tpos)
    return shifted_features, shifted_pos, shifted_valid, shifted_tpos


def _save_mask(mask: np.ndarray, path: Path) -> None:
    binary = (mask > 0).astype(np.uint8) * 255
    Image.fromarray(binary).save(path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", default="output/sam31_video_state_exp")
    parser.add_argument("--video-dir", default="assets/videos/0001")
    parser.add_argument("--output-dir", default="output/sam31_video_state_exp_run")
    parser.add_argument("--point", nargs=2, type=float, default=[760.0, 470.0])
    parser.add_argument("--point-label", type=int, default=1)
    parser.add_argument("--box", nargs=4, type=float, default=[630.0, 250.0, 930.0, 710.0])
    parser.add_argument("--use-box", action="store_true")
    parser.add_argument("--num-frames", type=int, default=8)
    parser.add_argument("--prompt-first-frame-only", action="store_true", default=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    model_dir = Path(args.model_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    meta = json.loads((model_dir / "export_meta.json").read_text(encoding="utf-8"))
    session = _session(model_dir / "sam31_video_tracking_state_step.onnx")
    frame_paths = sorted(Path(args.video_dir).glob("*.jpg"))[: args.num_frames]
    if not frame_paths:
        raise FileNotFoundError(f"no jpg frames under {args.video_dir}")

    num_maskmem = int(meta["num_maskmem"])
    mem_dim = int(meta["mem_dim"])
    mem_h, mem_w = meta["mem_hw"]
    max_obj_ptrs = int(meta["max_obj_ptrs"])
    hidden_dim = int(meta["hidden_dim"])
    image_size = int(meta["image_size"])

    prev_maskmem_features = np.zeros((num_maskmem, 1, mem_dim, mem_h, mem_w), dtype=np.float32)
    prev_maskmem_pos_enc = np.zeros_like(prev_maskmem_features)
    prev_memory_valid = np.zeros((num_maskmem,), dtype=bool)
    prev_memory_tpos = np.arange(1, num_maskmem + 1, dtype=np.int64)
    prev_obj_ptrs = np.zeros((max_obj_ptrs, 1, hidden_dim), dtype=np.float32)
    prev_obj_ptr_valid = np.zeros((max_obj_ptrs,), dtype=bool)
    prev_obj_ptr_tpos = np.arange(1, max_obj_ptrs + 1, dtype=np.int64)
    mask_input = np.zeros((1, 1, 288, 288), dtype=np.float32)

    mask_paths = []
    score_trace = []
    for frame_idx, frame_path in enumerate(frame_paths):
        if frame_idx == 0 or not args.prompt_first_frame_only:
            point_coords, point_labels, box_xyxy, box_valid_mask = _prompt_inputs(
                args, frame_path, image_size
            )
        else:
            point_coords, point_labels, box_xyxy, box_valid_mask = _empty_prompts()

        outputs = session.run(
            None,
            {
                "image": _preprocess_image(frame_path, image_size),
                "point_coords": point_coords,
                "point_labels": point_labels,
                "box_xyxy": box_xyxy,
                "box_valid_mask": box_valid_mask,
                "mask_input": mask_input,
                "prev_maskmem_features": prev_maskmem_features,
                "prev_maskmem_pos_enc": prev_maskmem_pos_enc,
                "prev_memory_valid": prev_memory_valid,
                "prev_memory_tpos": prev_memory_tpos,
                "prev_obj_ptrs": prev_obj_ptrs,
                "prev_obj_ptr_valid": prev_obj_ptr_valid,
                "prev_obj_ptr_tpos": prev_obj_ptr_tpos,
            },
        )

        low_res_masks, high_res_masks, object_score_logits, new_maskmem_features, new_maskmem_pos_enc, new_obj_ptrs = outputs
        prev_maskmem_features, prev_maskmem_pos_enc, prev_memory_valid, prev_memory_tpos = _update_queue(
            prev_maskmem_features,
            prev_maskmem_pos_enc,
            prev_memory_valid,
            prev_memory_tpos,
            new_maskmem_features,
            new_maskmem_pos_enc,
            num_maskmem,
        )
        prev_obj_ptrs, _, prev_obj_ptr_valid, prev_obj_ptr_tpos = _update_queue(
            prev_obj_ptrs,
            np.zeros_like(prev_obj_ptrs),
            prev_obj_ptr_valid,
            prev_obj_ptr_tpos,
            new_obj_ptrs,
            np.zeros_like(new_obj_ptrs),
            max_obj_ptrs,
        )
        mask_path = output_dir / f"{frame_idx:04d}_mask.png"
        _save_mask(high_res_masks[0, 0], mask_path)
        mask_paths.append(str(mask_path))
        score_trace.append(float(object_score_logits.reshape(-1)[0]))

    summary = {
        "frames": len(frame_paths),
        "mask_paths": mask_paths[:3] + (["..."] if len(mask_paths) > 3 else []),
        "final_memory_valid": prev_memory_valid.astype(int).tolist(),
        "final_memory_tpos": prev_memory_tpos.tolist(),
        "final_obj_ptr_valid": prev_obj_ptr_valid.astype(int).tolist(),
        "final_obj_ptr_tpos": prev_obj_ptr_tpos.tolist(),
        "score_trace": score_trace,
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
