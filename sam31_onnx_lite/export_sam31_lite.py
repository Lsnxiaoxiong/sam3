import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Sequence, Tuple

import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sam31_onnx.export_sam31_onnx import (  # noqa: E402
    IMAGE_SIZE,
    TEXT_CONTEXT,
    BoxPromptWrapper,
    GroundingWrapper,
    PointPromptWrapper,
    _load_tracker_checkpoint,
    _make_find_stage,
    _make_stable_prompt,
    _patch_decoder_for_export,
    _patch_mask_downsampler_for_export,
    _patch_tracker_rope_for_export,
    _patch_vitdet_rope_for_export,
    _preprocess_image,
    _token_features,
)
from sam3.model.tokenizer_ve import SimpleTokenizer  # noqa: E402
from sam3.model_builder import build_sam3_image_model, build_tracker  # noqa: E402


class ImageEncoderWrapper(nn.Module):
    def __init__(self, image_model: nn.Module):
        super().__init__()
        self.image_model = image_model

    def forward(self, image: torch.Tensor):
        backbone_out = self.image_model.backbone.forward_image(image)
        return (
            backbone_out["vision_pos_enc"][0],
            backbone_out["vision_pos_enc"][1],
            backbone_out["vision_pos_enc"][2],
            backbone_out["backbone_fpn"][0],
            backbone_out["backbone_fpn"][1],
            backbone_out["backbone_fpn"][2],
        )


class TextEncoderWrapper(nn.Module):
    def __init__(self, image_model: nn.Module):
        super().__init__()
        self.text_encoder = image_model.backbone.language_backbone

    def forward(self, token_ids: torch.Tensor):
        language_mask, language_features, _ = _token_features(self.text_encoder, token_ids)
        return language_mask, language_features


class GroundingDecoderWrapper(nn.Module):
    def __init__(self, image_model: nn.Module):
        super().__init__()
        self.image_model = image_model

    def forward(
        self,
        vision_pos_enc_0: torch.Tensor,
        vision_pos_enc_1: torch.Tensor,
        vision_pos_enc_2: torch.Tensor,
        backbone_fpn_0: torch.Tensor,
        backbone_fpn_1: torch.Tensor,
        backbone_fpn_2: torch.Tensor,
        language_mask: torch.Tensor,
        language_features: torch.Tensor,
    ):
        batch = backbone_fpn_2.shape[0]
        backbone_out = {
            "vision_pos_enc": [vision_pos_enc_0, vision_pos_enc_1, vision_pos_enc_2],
            "backbone_fpn": [backbone_fpn_0, backbone_fpn_1, backbone_fpn_2],
            "language_mask": language_mask,
            "language_features": language_features,
        }
        out = self.image_model.forward_grounding(
            backbone_out=backbone_out,
            find_input=_make_find_stage(batch, backbone_fpn_2.device),
            find_target=None,
            geometric_prompt=_make_stable_prompt(batch, backbone_fpn_2.device),
        )
        return out["pred_logits"], out["pred_boxes_xyxy"], out["pred_masks"]


class PointDecoderWrapper(nn.Module):
    def __init__(self, tracker: nn.Module):
        super().__init__()
        self.tracker = tracker
        self.register_buffer(
            "dense_pe",
            tracker.sam_prompt_encoder.get_dense_pe().detach().clone(),
            persistent=False,
        )

    def forward(
        self,
        backbone_fpn_0: torch.Tensor,
        backbone_fpn_1: torch.Tensor,
        backbone_fpn_2: torch.Tensor,
        point_coords: torch.Tensor,
        point_labels: torch.Tensor,
        mask_input: torch.Tensor,
    ):
        wrapper = PointPromptWrapper.__new__(PointPromptWrapper)
        nn.Module.__init__(wrapper)
        wrapper.image_model = None
        wrapper.tracker = self.tracker
        wrapper.register_buffer("dense_pe", self.dense_pe, persistent=False)
        fpn = [backbone_fpn_0, backbone_fpn_1, backbone_fpn_2]
        high_res = [
            self.tracker.sam_mask_decoder.conv_s0(fpn[0]),
            self.tracker.sam_mask_decoder.conv_s1(fpn[1]),
        ]
        image_embeddings = fpn[2] + self.tracker.no_mem_embed.view(1, -1, 1, 1)
        sparse_embeddings, dense_embeddings = self.tracker.sam_prompt_encoder(
            points=(point_coords, point_labels),
            boxes=None,
            masks=mask_input,
        )
        low_res_masks, ious, _, object_score_logits = self.tracker.sam_mask_decoder(
            image_embeddings=image_embeddings,
            image_pe=self.dense_pe.to(device=image_embeddings.device, dtype=image_embeddings.dtype),
            sparse_prompt_embeddings=sparse_embeddings,
            dense_prompt_embeddings=dense_embeddings,
            multimask_output=True,
            repeat_image=False,
            high_res_features=high_res,
        )
        high_res_masks = torch.nn.functional.interpolate(
            low_res_masks.float(),
            size=(IMAGE_SIZE, IMAGE_SIZE),
            mode="bilinear",
            align_corners=False,
        )
        return low_res_masks, high_res_masks, ious, object_score_logits


class BoxDecoderWrapper(nn.Module):
    def __init__(self, tracker: nn.Module):
        super().__init__()
        self.tracker = tracker
        self.register_buffer(
            "dense_pe",
            tracker.sam_prompt_encoder.get_dense_pe().detach().clone(),
            persistent=False,
        )

    def forward(
        self,
        backbone_fpn_0: torch.Tensor,
        backbone_fpn_1: torch.Tensor,
        backbone_fpn_2: torch.Tensor,
        box_xyxy: torch.Tensor,
        mask_input: torch.Tensor,
    ):
        fpn = [backbone_fpn_0, backbone_fpn_1, backbone_fpn_2]
        high_res = [
            self.tracker.sam_mask_decoder.conv_s0(fpn[0]),
            self.tracker.sam_mask_decoder.conv_s1(fpn[1]),
        ]
        image_embeddings = fpn[2] + self.tracker.no_mem_embed.view(1, -1, 1, 1)
        sparse_embeddings, dense_embeddings = self.tracker.sam_prompt_encoder(
            points=None,
            boxes=box_xyxy,
            masks=mask_input,
        )
        low_res_masks, ious, _, object_score_logits = self.tracker.sam_mask_decoder(
            image_embeddings=image_embeddings,
            image_pe=self.dense_pe.to(device=image_embeddings.device, dtype=image_embeddings.dtype),
            sparse_prompt_embeddings=sparse_embeddings,
            dense_prompt_embeddings=dense_embeddings,
            multimask_output=True,
            repeat_image=False,
            high_res_features=high_res,
        )
        high_res_masks = torch.nn.functional.interpolate(
            low_res_masks.float(),
            size=(IMAGE_SIZE, IMAGE_SIZE),
            mode="bilinear",
            align_corners=False,
        )
        return low_res_masks, high_res_masks, ious, object_score_logits


class VideoPromptDecoderWrapper(nn.Module):
    def __init__(self, tracker: nn.Module):
        super().__init__()
        self.tracker = tracker
        self.register_buffer(
            "dense_pe",
            tracker.sam_prompt_encoder.get_dense_pe().detach().clone(),
            persistent=False,
        )

    def forward(
        self,
        vision_pos_enc_0: torch.Tensor,
        vision_pos_enc_1: torch.Tensor,
        vision_pos_enc_2: torch.Tensor,
        backbone_fpn_0: torch.Tensor,
        backbone_fpn_1: torch.Tensor,
        backbone_fpn_2: torch.Tensor,
        point_coords: torch.Tensor,
        point_labels: torch.Tensor,
        box_xyxy: torch.Tensor,
        mask_input: torch.Tensor,
    ):
        del vision_pos_enc_0, vision_pos_enc_1, vision_pos_enc_2
        fpn = [backbone_fpn_0, backbone_fpn_1, backbone_fpn_2]
        high_res = [
            self.tracker.sam_mask_decoder.conv_s0(fpn[0]),
            self.tracker.sam_mask_decoder.conv_s1(fpn[1]),
        ]
        image_embeddings = fpn[2] + self.tracker.no_mem_embed.view(1, -1, 1, 1)
        sparse_embeddings, dense_embeddings = self.tracker.sam_prompt_encoder(
            points=(point_coords, point_labels),
            boxes=box_xyxy,
            masks=mask_input,
        )
        low_res_masks, _, sam_output_tokens, object_score_logits = self.tracker.sam_mask_decoder(
            image_embeddings=image_embeddings,
            image_pe=self.dense_pe.to(device=image_embeddings.device, dtype=image_embeddings.dtype),
            sparse_prompt_embeddings=sparse_embeddings,
            dense_prompt_embeddings=dense_embeddings,
            multimask_output=False,
            repeat_image=False,
            high_res_features=high_res,
        )
        high_res_masks = torch.nn.functional.interpolate(
            low_res_masks.float(),
            size=(IMAGE_SIZE, IMAGE_SIZE),
            mode="bilinear",
            align_corners=False,
        )
        obj_ptr = self.tracker.obj_ptr_proj(sam_output_tokens[:, 0])
        return low_res_masks, high_res_masks, obj_ptr, object_score_logits


@dataclass
class ExportSpec:
    name: str
    model: nn.Module
    inputs: Tuple[torch.Tensor, ...]
    input_names: Sequence[str]
    output_names: Sequence[str]
    dynamic_axes: Optional[Dict[str, Dict[int, str]]] = None


def _export(spec: ExportSpec, output_dir: Path) -> str:
    model_dir = output_dir / spec.name
    model_dir.mkdir(parents=True, exist_ok=True)
    path = model_dir / f"{spec.name}.onnx"
    torch.onnx.export(
        spec.model,
        spec.inputs,
        str(path),
        input_names=list(spec.input_names),
        output_names=list(spec.output_names),
        dynamic_axes=spec.dynamic_axes,
        opset_version=17,
        do_constant_folding=False,
        external_data=True,
    )
    return str(path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint",
        default=r"C:\Users\lsn\.cache\modelscope\hub\models\facebook\sam3___1\sam3.1_multiplex.pt",
    )
    parser.add_argument("--output-dir", default="output/sam31_onnx_lite")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--sample-image", default="assets/images/truck.jpg")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)

    _patch_vitdet_rope_for_export()
    _patch_mask_downsampler_for_export()
    _patch_decoder_for_export()

    image_model = build_sam3_image_model(
        checkpoint_path=args.checkpoint,
        load_from_HF=False,
        bpe_path=None,
        device=args.device,
        eval_mode=True,
        enable_inst_interactivity=False,
    ).float().eval()
    _patch_vitdet_rope_for_export(image_model)

    tracker = build_tracker(apply_temporal_disambiguation=True).to(device).float().eval()
    _load_tracker_checkpoint(tracker, args.checkpoint)
    _patch_tracker_rope_for_export(tracker)

    image_encoder = ImageEncoderWrapper(image_model).to(device).eval()
    text_encoder = TextEncoderWrapper(image_model).to(device).eval()
    grounding_decoder = GroundingDecoderWrapper(image_model).to(device).eval()
    point_decoder = PointDecoderWrapper(tracker).to(device).eval()
    box_decoder = BoxDecoderWrapper(tracker).to(device).eval()
    video_decoder = VideoPromptDecoderWrapper(tracker).to(device).eval()

    tokenizer = SimpleTokenizer(
        bpe_path=str(ROOT / "sam3" / "assets" / "bpe_simple_vocab_16e6.txt.gz")
    )
    sample_tokens = tokenizer(["truck"], context_length=TEXT_CONTEXT).to(device)
    sample_image = _preprocess_image(args.sample_image, device)
    sample_points = torch.tensor([[[320.0, 420.0]]], dtype=torch.float32, device=device)
    sample_point_labels = torch.tensor([[1]], dtype=torch.int64, device=device)
    sample_box = torch.tensor(
        [[220.0, 180.0, 860.0, 820.0]], dtype=torch.float32, device=device
    )
    sample_mask = torch.zeros((1, 1, 288, 288), dtype=torch.float32, device=device)

    with torch.no_grad():
        image_feats = image_encoder(sample_image)
        language_feats = text_encoder(sample_tokens)

    specs = [
        ExportSpec(
            name="sam31_image_encoder",
            model=image_encoder,
            inputs=(sample_image,),
            input_names=("image",),
            output_names=(
                "vision_pos_enc_0",
                "vision_pos_enc_1",
                "vision_pos_enc_2",
                "backbone_fpn_0",
                "backbone_fpn_1",
                "backbone_fpn_2",
            ),
        ),
        ExportSpec(
            name="sam31_text_encoder",
            model=text_encoder,
            inputs=(sample_tokens,),
            input_names=("token_ids",),
            output_names=("language_mask", "language_features"),
        ),
        ExportSpec(
            name="sam31_grounding_decoder",
            model=grounding_decoder,
            inputs=(
                image_feats[0],
                image_feats[1],
                image_feats[2],
                image_feats[3],
                image_feats[4],
                image_feats[5],
                language_feats[0],
                language_feats[1],
            ),
            input_names=(
                "vision_pos_enc_0",
                "vision_pos_enc_1",
                "vision_pos_enc_2",
                "backbone_fpn_0",
                "backbone_fpn_1",
                "backbone_fpn_2",
                "language_mask",
                "language_features",
            ),
            output_names=("pred_logits", "pred_boxes_xyxy", "pred_masks"),
        ),
        ExportSpec(
            name="sam31_point_decoder",
            model=point_decoder,
            inputs=(
                image_feats[3],
                image_feats[4],
                image_feats[5],
                sample_points,
                sample_point_labels,
                sample_mask,
            ),
            input_names=(
                "backbone_fpn_0",
                "backbone_fpn_1",
                "backbone_fpn_2",
                "point_coords",
                "point_labels",
                "mask_input",
            ),
            output_names=("low_res_masks", "high_res_masks", "ious", "object_score_logits"),
        ),
        ExportSpec(
            name="sam31_box_decoder",
            model=box_decoder,
            inputs=(
                image_feats[3],
                image_feats[4],
                image_feats[5],
                sample_box,
                sample_mask,
            ),
            input_names=(
                "backbone_fpn_0",
                "backbone_fpn_1",
                "backbone_fpn_2",
                "box_xyxy",
                "mask_input",
            ),
            output_names=("low_res_masks", "high_res_masks", "ious", "object_score_logits"),
        ),
        ExportSpec(
            name="sam31_video_prompt_decoder",
            model=video_decoder,
            inputs=(
                image_feats[0],
                image_feats[1],
                image_feats[2],
                image_feats[3],
                image_feats[4],
                image_feats[5],
                sample_points,
                sample_point_labels,
                sample_box,
                sample_mask,
            ),
            input_names=(
                "vision_pos_enc_0",
                "vision_pos_enc_1",
                "vision_pos_enc_2",
                "backbone_fpn_0",
                "backbone_fpn_1",
                "backbone_fpn_2",
                "point_coords",
                "point_labels",
                "box_xyxy",
                "mask_input",
            ),
            output_names=(
                "low_res_masks",
                "high_res_masks",
                "obj_ptr",
                "object_score_logits",
            ),
        ),
    ]

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    exported = {}
    for spec in specs:
        print(f"exporting {spec.name} ...")
        exported[spec.name] = _export(spec, output_dir)

    meta = {
        "checkpoint": args.checkpoint,
        "exports": exported,
        "device": args.device,
    }
    (output_dir / "export_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
