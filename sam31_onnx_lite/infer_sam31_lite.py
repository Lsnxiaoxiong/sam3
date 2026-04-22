import argparse
import json
import sys
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import onnxruntime as ort
import torch
from torchvision.transforms import v2

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sam3.model.tokenizer_ve import SimpleTokenizer  # noqa: E402


IMAGE_SIZE = 1008
TEXT_CONTEXT = 32


def _session(path: str) -> ort.InferenceSession:
    options = ort.SessionOptions()
    options.enable_mem_pattern = False
    options.enable_cpu_mem_arena = False
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    return ort.InferenceSession(path, sess_options=options, providers=["CUDAExecutionProvider","CPUExecutionProvider",])


def _model_path(model_dir: Path, name: str) -> Path:
    return model_dir / name / f"{name}.onnx"


def _run_with_available_inputs(
    sess: ort.InferenceSession, feeds: Dict[str, np.ndarray]
) -> list[np.ndarray]:
    input_names = {i.name for i in sess.get_inputs()}
    filtered = {k: v for k, v in feeds.items() if k in input_names}
    return sess.run(None, filtered)


def _preprocess_image(image_path: str) -> np.ndarray:
    import PIL.Image

    image = PIL.Image.open(image_path).convert("RGB")
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


def _run_image_encoder(model_dir: Path, image_path: str) -> Dict[str, np.ndarray]:
    sess = _session(str(_model_path(model_dir, "sam31_image_encoder")))
    outputs = sess.run(None, {"image": _preprocess_image(image_path)})
    names = [o.name for o in sess.get_outputs()]
    return dict(zip(names, outputs))


def run_text(model_dir: Path, image_path: str, prompt: str, bpe_path: str) -> Dict[str, np.ndarray]:
    image_out = _run_image_encoder(model_dir, image_path)
    text_sess = _session(str(_model_path(model_dir, "sam31_text_encoder")))
    language_mask, language_features = _run_with_available_inputs(
        text_sess, {"token_ids": _tokenize(prompt, bpe_path)}
    )
    decoder = _session(str(_model_path(model_dir, "sam31_grounding_decoder")))
    outputs = _run_with_available_inputs(
        decoder,
        {
            "vision_pos_enc_0": image_out["vision_pos_enc_0"],
            "vision_pos_enc_1": image_out["vision_pos_enc_1"],
            "vision_pos_enc_2": image_out["vision_pos_enc_2"],
            "backbone_fpn_0": image_out["backbone_fpn_0"],
            "backbone_fpn_1": image_out["backbone_fpn_1"],
            "backbone_fpn_2": image_out["backbone_fpn_2"],
            "language_mask": language_mask,
            "language_features": language_features,
        },
    )
    return {
        "pred_logits": outputs[0],
        "pred_boxes_xyxy": outputs[1],
        "pred_masks": outputs[2],
    }


def run_point(
    model_dir: Path,
    image_path: str,
    point_coords: np.ndarray,
    point_labels: np.ndarray,
) -> Dict[str, np.ndarray]:
    image_out = _run_image_encoder(model_dir, image_path)
    sess = _session(str(_model_path(model_dir, "sam31_point_decoder")))
    outputs = sess.run(
        None,
        {
            "backbone_fpn_0": image_out["backbone_fpn_0"],
            "backbone_fpn_1": image_out["backbone_fpn_1"],
            "backbone_fpn_2": image_out["backbone_fpn_2"],
            "point_coords": point_coords.astype(np.float32),
            "point_labels": point_labels.astype(np.int64),
            "mask_input": _default_mask(),
        },
    )
    return {
        "low_res_masks": outputs[0],
        "high_res_masks": outputs[1],
        "ious": outputs[2],
        "object_score_logits": outputs[3],
    }


def run_box(
    model_dir: Path,
    image_path: str,
    box_xyxy: np.ndarray,
) -> Dict[str, np.ndarray]:
    image_out = _run_image_encoder(model_dir, image_path)
    sess = _session(str(_model_path(model_dir, "sam31_box_decoder")))
    outputs = sess.run(
        None,
        {
            "backbone_fpn_0": image_out["backbone_fpn_0"],
            "backbone_fpn_1": image_out["backbone_fpn_1"],
            "backbone_fpn_2": image_out["backbone_fpn_2"],
            "box_xyxy": box_xyxy.astype(np.float32),
            "mask_input": _default_mask(),
        },
    )
    return {
        "low_res_masks": outputs[0],
        "high_res_masks": outputs[1],
        "ious": outputs[2],
        "object_score_logits": outputs[3],
    }


def run_video(
    model_dir: Path,
    frame_paths: list[str],
    point_coords: np.ndarray,
    point_labels: np.ndarray,
    box_xyxy: np.ndarray,
) -> Dict[str, np.ndarray]:
    sess = _session(str(_model_path(model_dir, "sam31_video_prompt_decoder")))
    video_low = []
    video_high = []
    last_obj_ptr = None
    last_object_score = None
    for frame_path in frame_paths:
        image_out = _run_image_encoder(model_dir, frame_path)
        outputs = _run_with_available_inputs(
            sess,
            {
                "vision_pos_enc_0": image_out["vision_pos_enc_0"],
                "vision_pos_enc_1": image_out["vision_pos_enc_1"],
                "vision_pos_enc_2": image_out["vision_pos_enc_2"],
                "backbone_fpn_0": image_out["backbone_fpn_0"],
                "backbone_fpn_1": image_out["backbone_fpn_1"],
                "backbone_fpn_2": image_out["backbone_fpn_2"],
                "point_coords": point_coords.astype(np.float32),
                "point_labels": point_labels.astype(np.int64),
                "box_xyxy": box_xyxy.astype(np.float32),
                "mask_input": _default_mask(),
            },
        )
        video_low.append(outputs[0])
        video_high.append(outputs[1])
        last_obj_ptr = outputs[2]
        last_object_score = outputs[3]
    return {
        "video_low_res_masks": np.concatenate(video_low, axis=0),
        "video_high_res_masks": np.concatenate(video_high, axis=0),
        "obj_ptr": last_obj_ptr,
        "object_score_logits": last_object_score,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", default="output/sam31_onnx_lite")
    parser.add_argument("--mode", required=True, choices=["text", "point", "box", "video"])
    parser.add_argument("--image", default="assets/images/truck.jpg")
    parser.add_argument("--video-dir", default="assets/videos/0001")
    parser.add_argument("--text-prompt", default="truck")
    parser.add_argument(
        "--bpe-path",
        default=str(ROOT / "sam3" / "assets" / "bpe_simple_vocab_16e6.txt.gz"),
    )
    parser.add_argument("--point", nargs=2, type=float, default=[320.0, 420.0])
    parser.add_argument("--point-label", type=int, default=1)
    parser.add_argument("--box", nargs=4, type=float, default=[220.0, 180.0, 860.0, 820.0])
    parser.add_argument("--output", default="")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    model_dir = Path(args.model_dir)
    point_coords = np.array([[args.point]], dtype=np.float32)
    point_labels = np.array([[args.point_label]], dtype=np.int64)
    box_xyxy = np.array([args.box], dtype=np.float32)

    if args.mode == "text":
        result = run_text(model_dir, args.image, args.text_prompt, args.bpe_path)
    elif args.mode == "point":
        result = run_point(model_dir, args.image, point_coords, point_labels)
    elif args.mode == "box":
        result = run_box(model_dir, args.image, box_xyxy)
    else:
        frame_paths = sorted(str(p) for p in Path(args.video_dir).glob("*.jpg"))[:8]
        result = run_video(model_dir, frame_paths, point_coords, point_labels, box_xyxy)

    summary = {k: list(v.shape) for k, v in result.items()}
    print(json.dumps(summary, indent=2))
    if args.output:
        np.savez(args.output, **result)


if __name__ == "__main__":
    main()
