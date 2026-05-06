from __future__ import annotations

import gzip
import html
import io
import string
from functools import lru_cache
from pathlib import Path
from typing import Dict, Iterable, List, Literal, Tuple

import ftfy
import numpy as np
import onnxruntime as ort
from PIL import Image
import regex as re


IMAGE_SIZE = 1008
TEXT_CONTEXT = 32
ROOT = Path(__file__).resolve().parent.parent
DEFAULT_BPE_PATH = ROOT / "sam3" / "assets" / "bpe_simple_vocab_16e6.txt.gz"

PromptMode = Literal["text", "point", "box"]


@lru_cache()
def _bytes_to_unicode() -> dict[int, str]:
    bs = (
        list(range(ord("!"), ord("~") + 1))
        + list(range(ord("¡"), ord("¬") + 1))
        + list(range(ord("®"), ord("ÿ") + 1))
    )
    cs = bs[:]
    n = 0
    for b in range(2**8):
        if b not in bs:
            bs.append(b)
            cs.append(2**8 + n)
            n += 1
    return dict(zip(bs, [chr(n) for n in cs]))


def _get_pairs(word: tuple[str, ...]) -> set[tuple[str, str]]:
    pairs = set()
    prev_char = word[0]
    for char in word[1:]:
        pairs.add((prev_char, char))
        prev_char = char
    return pairs


def _clean_text(text: str) -> str:
    text = ftfy.fix_text(text)
    text = html.unescape(html.unescape(text)).strip()
    text = re.sub(r"\s+", " ", text).strip().lower()
    return text


class NumpySimpleTokenizer:
    """CLIP-style BPE tokenizer that returns numpy arrays and does not import torch."""

    def __init__(
        self,
        bpe_path: str | Path = DEFAULT_BPE_PATH,
        context_length: int = TEXT_CONTEXT,
    ) -> None:
        self.byte_encoder = _bytes_to_unicode()
        with Path(bpe_path).open("rb") as fh:
            bpe_bytes = io.BytesIO(fh.read())
            merges = gzip.open(bpe_bytes).read().decode("utf-8").split("\n")
        merges = merges[1 : 49152 - 256 - 2 + 1]
        merges = [tuple(merge.split()) for merge in merges]

        vocab = list(_bytes_to_unicode().values())
        vocab = vocab + [v + "</w>" for v in vocab]
        for merge in merges:
            vocab.append("".join(merge))
        vocab.extend(["<start_of_text>", "<end_of_text>"])

        self.encoder = dict(zip(vocab, range(len(vocab))))
        self.bpe_ranks = dict(zip(merges, range(len(merges))))
        self.cache = {
            "<start_of_text>": "<start_of_text>",
            "<end_of_text>": "<end_of_text>",
        }
        self.sot_token_id = self.encoder["<start_of_text>"]
        self.eot_token_id = self.encoder["<end_of_text>"]
        self.context_length = context_length
        special = "<start_of_text>|<end_of_text>"
        self.pattern = re.compile(
            special + r"""|'s|'t|'re|'ve|'m|'ll|'d|[\p{L}]+|[\p{N}]|[^\s\p{L}\p{N}]+""",
            re.IGNORECASE,
        )

    def bpe(self, token: str) -> str:
        if token in self.cache:
            return self.cache[token]
        word = tuple(token[:-1]) + (token[-1] + "</w>",)
        pairs = _get_pairs(word)
        if not pairs:
            return token + "</w>"

        while True:
            bigram = min(pairs, key=lambda pair: self.bpe_ranks.get(pair, float("inf")))
            if bigram not in self.bpe_ranks:
                break
            first, second = bigram
            new_word = []
            i = 0
            while i < len(word):
                try:
                    j = word.index(first, i)
                    new_word.extend(word[i:j])
                    i = j
                except ValueError:
                    new_word.extend(word[i:])
                    break
                if word[i] == first and i < len(word) - 1 and word[i + 1] == second:
                    new_word.append(first + second)
                    i += 2
                else:
                    new_word.append(word[i])
                    i += 1
            word = tuple(new_word)
            if len(word) == 1:
                break
            pairs = _get_pairs(word)

        word_text = " ".join(word)
        self.cache[token] = word_text
        return word_text

    def encode(self, text: str) -> list[int]:
        bpe_tokens = []
        text = _clean_text(text)
        for token in re.findall(self.pattern, text):
            token = "".join(self.byte_encoder[b] for b in token.encode("utf-8"))
            bpe_tokens.extend(
                self.encoder[bpe_token] for bpe_token in self.bpe(token).split(" ")
            )
        return bpe_tokens

    def __call__(self, texts: str | Iterable[str]) -> np.ndarray:
        if isinstance(texts, str):
            texts = [texts]
        token_lists = [
            [self.sot_token_id] + self.encode(text) + [self.eot_token_id]
            for text in texts
        ]
        result = np.zeros((len(token_lists), self.context_length), dtype=np.int64)
        for i, tokens in enumerate(token_lists):
            if len(tokens) > self.context_length:
                tokens = tokens[: self.context_length]
                tokens[-1] = self.eot_token_id
            result[i, : len(tokens)] = np.asarray(tokens, dtype=np.int64)
        return result


class Sam31LiteNoTorchPredictor:
    """SAM3.1 ONNX image predictor without torch/torchvision imports.

    Supports text, point, and box prompts using the ONNX files produced by
    sam31_onnx_lite/export_sam31_lite.py.
    """

    def __init__(
        self,
        model_dir: str | Path = "output/sam31_onnx_lite",
        *,
        bpe_path: str | Path = DEFAULT_BPE_PATH,
        providers: list[str] | None = None,
    ) -> None:
        self.model_dir = Path(model_dir)
        self.tokenizer = NumpySimpleTokenizer(bpe_path)
        self.providers = providers or _default_providers()
        self.sessions: dict[str, ort.InferenceSession] = {}
        self.dense_pe = np.load(self.model_dir / "sam31_interactive_dense_pe.npy").astype(
            np.float32
        )

    def predict_text(
        self,
        image: str | Path | Image.Image | np.ndarray,
        text_prompt: str,
        *,
        return_resized_mask: bool = True,
    ) -> dict[str, np.ndarray | int | float]:
        original_image = _load_rgb_image(image)
        image_out = self._run_image_encoder(original_image)
        text_out = self._run(
            "sam31_text_encoder",
            {"token_ids": self.tokenizer(text_prompt)},
        )
        result = self._run(
            "sam31_grounding_decoder",
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
        mask, best_idx, best_logit = _select_best_text_mask(result)
        return _format_mask_result(
            mask,
            original_image,
            return_resized_mask=return_resized_mask,
            best_query_index=best_idx,
            best_score=best_logit,
            raw_outputs=result,
        )

    def predict_point(
        self,
        image: str | Path | Image.Image | np.ndarray,
        point_xy: tuple[float, float] | list[float] | np.ndarray,
        *,
        point_label: int = 1,
        return_resized_mask: bool = True,
    ) -> dict[str, np.ndarray | float]:
        original_image = _load_rgb_image(image)
        point_coords = np.asarray([point_xy], dtype=np.float32)
        point_labels = np.asarray([point_label], dtype=np.int64)
        result = self._run_interactive(
            original_image,
            point_coords,
            point_labels,
            np.zeros((1, 4), dtype=np.float32),
            np.asarray([False], dtype=bool),
            "sam31_point_decoder",
        )
        mask, score = _select_best_interactive_mask(result, "point")
        return _format_mask_result(
            mask,
            original_image,
            return_resized_mask=return_resized_mask,
            best_score=score,
            raw_outputs=result,
        )

    def predict_box(
        self,
        image: str | Path | Image.Image | np.ndarray,
        box_xyxy: tuple[float, float, float, float] | list[float] | np.ndarray,
        *,
        return_resized_mask: bool = True,
    ) -> dict[str, np.ndarray | float]:
        original_image = _load_rgb_image(image)
        result = self._run_interactive(
            original_image,
            np.zeros((1, 2), dtype=np.float32),
            np.asarray([-1], dtype=np.int64),
            np.asarray([box_xyxy], dtype=np.float32),
            np.asarray([True], dtype=bool),
            "sam31_box_decoder",
        )
        mask, score = _select_best_interactive_mask(result, "box")
        return _format_mask_result(
            mask,
            original_image,
            return_resized_mask=return_resized_mask,
            best_score=score,
            raw_outputs=result,
        )

    def save_mask(
        self,
        mask: np.ndarray,
        output_path: str | Path,
        *,
        threshold: float = 0.0,
    ) -> str:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(((mask > threshold).astype(np.uint8) * 255)).save(output_path)
        return str(output_path)

    def _run_image_encoder(self, image: Image.Image) -> Dict[str, np.ndarray]:
        return self._run("sam31_image_encoder", {"image": _preprocess_image(image)})

    def _run_interactive(
        self,
        image: Image.Image,
        point_coords: np.ndarray,
        point_labels: np.ndarray,
        box_xyxy: np.ndarray,
        box_valid_mask: np.ndarray,
        decoder_name: str,
    ) -> Dict[str, np.ndarray]:
        image_out = self._run_image_encoder(image)
        scaled_points = _scaled_points_to_model_space(point_coords, image.width, image.height)
        scaled_boxes = _scaled_xyxy_to_model_space(box_xyxy, image.width, image.height)
        return self._run(
            decoder_name,
            {
                "sam2_backbone_fpn_0": image_out["sam2_backbone_fpn_0"],
                "sam2_backbone_fpn_1": image_out["sam2_backbone_fpn_1"],
                "sam2_backbone_fpn_2": image_out["sam2_backbone_fpn_2"],
                "image_pe": self.dense_pe,
                "point_coords": scaled_points[None, ...].astype(np.float32),
                "point_labels": point_labels[None, ...].astype(np.int64),
                "box_xyxy": scaled_boxes[None, ...].astype(np.float32),
                "box_valid_mask": box_valid_mask[None, ...].astype(bool),
                "mask_input": np.zeros((1, 1, 288, 288), dtype=np.float32),
            },
        )

    def _run(self, model_name: str, feeds: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        session = self._session(model_name)
        input_names = {i.name for i in session.get_inputs()}
        filtered = {k: v for k, v in feeds.items() if k in input_names}
        outputs = session.run(None, filtered)
        return dict(zip([o.name for o in session.get_outputs()], outputs))

    def _session(self, model_name: str) -> ort.InferenceSession:
        if model_name not in self.sessions:
            path = self.model_dir / model_name / f"{model_name}.onnx"
            options = ort.SessionOptions()
            options.enable_mem_pattern = False
            options.enable_cpu_mem_arena = False
            options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
            self.sessions[model_name] = ort.InferenceSession(
                str(path),
                sess_options=options,
                providers=self.providers,
            )
        return self.sessions[model_name]


def predict_text(
    image: str | Path | Image.Image | np.ndarray,
    text_prompt: str,
    *,
    model_dir: str | Path = "output/sam31_onnx_lite",
    bpe_path: str | Path = DEFAULT_BPE_PATH,
) -> dict[str, np.ndarray | int | float]:
    return Sam31LiteNoTorchPredictor(model_dir, bpe_path=bpe_path).predict_text(
        image,
        text_prompt,
    )


def predict_point(
    image: str | Path | Image.Image | np.ndarray,
    point_xy: tuple[float, float] | list[float] | np.ndarray,
    *,
    point_label: int = 1,
    model_dir: str | Path = "output/sam31_onnx_lite",
    bpe_path: str | Path = DEFAULT_BPE_PATH,
) -> dict[str, np.ndarray | float]:
    return Sam31LiteNoTorchPredictor(model_dir, bpe_path=bpe_path).predict_point(
        image,
        point_xy,
        point_label=point_label,
    )


def predict_box(
    image: str | Path | Image.Image | np.ndarray,
    box_xyxy: tuple[float, float, float, float] | list[float] | np.ndarray,
    *,
    model_dir: str | Path = "output/sam31_onnx_lite",
    bpe_path: str | Path = DEFAULT_BPE_PATH,
) -> dict[str, np.ndarray | float]:
    return Sam31LiteNoTorchPredictor(model_dir, bpe_path=bpe_path).predict_box(
        image,
        box_xyxy,
    )


def _default_providers() -> list[str]:
    available = ort.get_available_providers()
    providers = []
    if "CUDAExecutionProvider" in available:
        providers.append("CUDAExecutionProvider")
    providers.append("CPUExecutionProvider")
    return providers


def _load_rgb_image(image: str | Path | Image.Image | np.ndarray) -> Image.Image:
    if isinstance(image, (str, Path)):
        return Image.open(image).convert("RGB")
    if isinstance(image, Image.Image):
        return image.convert("RGB")
    array = np.asarray(image)
    if array.ndim != 3 or array.shape[2] not in (3, 4):
        raise ValueError(f"Expected HxWx3 or HxWx4 image array, got {array.shape}")
    if array.shape[2] == 4:
        array = array[:, :, :3]
    if array.dtype != np.uint8:
        array = np.clip(array, 0, 255).astype(np.uint8)
    return Image.fromarray(array, mode="RGB")


def _preprocess_image(image: Image.Image) -> np.ndarray:
    resized = image.resize((IMAGE_SIZE, IMAGE_SIZE), resample=Image.BICUBIC)
    array = np.asarray(resized, dtype=np.float32) / 255.0
    array = (array - 0.5) / 0.5
    array = np.transpose(array, (2, 0, 1))[None, ...]
    return np.ascontiguousarray(array, dtype=np.float32)


def _scaled_xyxy_to_model_space(
    boxes_xyxy: np.ndarray,
    width: int,
    height: int,
) -> np.ndarray:
    scaled = boxes_xyxy.astype(np.float32).copy()
    scaled[:, [0, 2]] *= IMAGE_SIZE / width
    scaled[:, [1, 3]] *= IMAGE_SIZE / height
    return scaled


def _scaled_points_to_model_space(
    points_xy: np.ndarray,
    width: int,
    height: int,
) -> np.ndarray:
    scaled = points_xy.astype(np.float32).copy()
    scaled[:, 0] *= IMAGE_SIZE / width
    scaled[:, 1] *= IMAGE_SIZE / height
    return scaled


def _resize_mask_to_image(mask: np.ndarray, image: Image.Image) -> np.ndarray:
    mask_image = Image.fromarray(mask.astype(np.float32), mode="F")
    mask_image = mask_image.resize(image.size, resample=Image.BILINEAR)
    return np.asarray(mask_image, dtype=np.float32)


def _select_best_text_mask(result: Dict[str, np.ndarray]) -> Tuple[np.ndarray, int, float]:
    logits = result["pred_logits"][0, :, 0]
    boxes = result["pred_boxes_xyxy"][0]
    masks = result["pred_masks"][0]
    mask_area = (masks > 0).reshape(masks.shape[0], -1).mean(axis=1)
    box_area = np.clip(boxes[:, 2] - boxes[:, 0], 0.0, 1.0) * np.clip(
        boxes[:, 3] - boxes[:, 1],
        0.0,
        1.0,
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
    result: Dict[str, np.ndarray],
    mode: PromptMode,
) -> Tuple[np.ndarray, float]:
    if mode == "point":
        best_idx = int(np.argmax(result["multi_scores"][0]))
        return result["multi_masks"][0, best_idx], float(result["multi_scores"][0, best_idx])
    return result["single_masks"][0, 0], float(result["single_scores"][0, 0])


def _format_mask_result(
    mask: np.ndarray,
    image: Image.Image,
    *,
    return_resized_mask: bool,
    best_score: float,
    raw_outputs: Dict[str, np.ndarray],
    best_query_index: int | None = None,
) -> dict[str, np.ndarray | int | float]:
    result: dict[str, np.ndarray | int | float] = {
        "low_res_mask": mask.astype(np.float32, copy=False),
        "best_score": float(best_score),
    }
    if best_query_index is not None:
        result["best_query_index"] = int(best_query_index)
    if return_resized_mask:
        resized = _resize_mask_to_image(mask, image)
        result["mask"] = resized
        result["binary_mask"] = resized > 0.0
    result["raw_outputs"] = raw_outputs  # type: ignore[assignment]
    return result
