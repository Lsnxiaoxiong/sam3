import argparse
import json
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
from torchvision.ops import roi_align

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sam31_onnx.export_sam31_onnx import (  # noqa: E402
    IMAGE_SIZE,
    TEXT_CONTEXT,
    GroundingWrapper,
    _load_checkpoint_raw,
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


def _load_interactive_predictor_weights(image_model: nn.Module, checkpoint_path: str) -> None:
    ckpt = _load_checkpoint_raw(checkpoint_path)
    mapped = {}
    for key, value in ckpt.items():
        if key.startswith("detector.backbone.vision_backbone.interactive_convs."):
            mapped[
                key.replace(
                    "detector.backbone.vision_backbone.interactive_convs.",
                    "backbone.vision_backbone.sam2_convs.",
                )
            ] = value
            continue
        if key.startswith("tracker.model.interactive_sam_prompt_encoder."):
            mapped[
                key.replace(
                    "tracker.model.interactive_sam_prompt_encoder.",
                    "inst_interactive_predictor.model.sam_prompt_encoder.",
                )
            ] = value
            continue
        if key.startswith("tracker.model.interactive_sam_mask_decoder."):
            mapped[
                key.replace(
                    "tracker.model.interactive_sam_mask_decoder.",
                    "inst_interactive_predictor.model.sam_mask_decoder.",
                )
            ] = value
            continue
        if key.startswith("tracker.model.interactive_mask_downsample."):
            mapped[
                key.replace(
                    "tracker.model.interactive_mask_downsample.",
                    "inst_interactive_predictor.model.mask_downsample.",
                )
            ] = value
            continue
        if key.startswith("tracker.model.interactive_obj_ptr_proj."):
            mapped[
                key.replace(
                    "tracker.model.interactive_obj_ptr_proj.",
                    "inst_interactive_predictor.model.obj_ptr_proj.",
                )
            ] = value
            continue
        if key.startswith("tracker.model."):
            suffix = key.replace("tracker.model.", "", 1)
            common_prefixes = (
                "maskmem_tpos_enc",
                "no_mem_embed",
                "no_mem_pos_enc",
                "no_obj_ptr",
                "no_obj_embed_spatial",
                "transformer.",
                "maskmem_backbone.",
                "obj_ptr_tpos_proj.",
            )
            if suffix.startswith(common_prefixes):
                mapped[f"inst_interactive_predictor.model.{suffix}"] = value

    model_state = image_model.state_dict()
    filtered = {k: v for k, v in mapped.items() if k in model_state and tuple(model_state[k].shape) == tuple(v.shape)}
    image_model.load_state_dict(filtered, strict=False)


class ImageEncoderWrapper(nn.Module):
    def __init__(self, image_model: nn.Module):
        super().__init__()
        self.image_model = image_model

    def forward(self, image: torch.Tensor):
        backbone_out = self.image_model.backbone.forward_image(image)
        outputs = [
            backbone_out["vision_pos_enc"][0],
            backbone_out["vision_pos_enc"][1],
            backbone_out["vision_pos_enc"][2],
            backbone_out["backbone_fpn"][0],
            backbone_out["backbone_fpn"][1],
            backbone_out["backbone_fpn"][2],
        ]
        if getattr(self.image_model, "inst_interactive_predictor", None) is not None and "sam2_backbone_out" in backbone_out:
            sam2_backbone_out = backbone_out["sam2_backbone_out"]
            sam2_backbone_out["backbone_fpn"][0] = (
                self.image_model.inst_interactive_predictor.model.sam_mask_decoder.conv_s0(
                    sam2_backbone_out["backbone_fpn"][0]
                )
            )
            sam2_backbone_out["backbone_fpn"][1] = (
                self.image_model.inst_interactive_predictor.model.sam_mask_decoder.conv_s1(
                    sam2_backbone_out["backbone_fpn"][1]
                )
            )
            outputs.extend(
                [
                    sam2_backbone_out["vision_pos_enc"][0],
                    sam2_backbone_out["vision_pos_enc"][1],
                    sam2_backbone_out["vision_pos_enc"][2],
                    sam2_backbone_out["backbone_fpn"][0],
                    sam2_backbone_out["backbone_fpn"][1],
                    sam2_backbone_out["backbone_fpn"][2],
                ]
            )
        return tuple(outputs)


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

class ReferenceFeatureEncoderWrapper(nn.Module):
    def forward(self, backbone_fpn_0: torch.Tensor, reference_boxes_xyxy: torch.Tensor) -> torch.Tensor:
        if reference_boxes_xyxy.shape[0] == 0:
            return torch.zeros((0, backbone_fpn_0.shape[1]), dtype=backbone_fpn_0.dtype, device=backbone_fpn_0.device)
        batch_indices = torch.zeros((reference_boxes_xyxy.shape[0], 1), dtype=reference_boxes_xyxy.dtype, device=reference_boxes_xyxy.device)
        roi_boxes = torch.cat([batch_indices, reference_boxes_xyxy], dim=1)
        pooled = roi_align(
            backbone_fpn_0,
            roi_boxes,
            output_size=(7, 7),
            spatial_scale=backbone_fpn_0.shape[-1] / float(IMAGE_SIZE),
            sampling_ratio=2,
            aligned=True,
        )
        return pooled.mean(dim=(2, 3))


class GroundingDecoderWithReferenceWrapper(GroundingDecoderWrapper):
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
        reference_features: torch.Tensor,
        reference_valid_mask: torch.Tensor,
        reference_weight: torch.Tensor,
    ):
        seq_len, batch_size, feat_dim = language_features.shape
        if reference_features.shape[0] == 0:
            reference_bias = torch.zeros((batch_size, feat_dim), dtype=language_features.dtype, device=language_features.device)
        else:
            valid = reference_valid_mask.to(language_features.dtype).unsqueeze(-1)
            valid_count = valid.sum(dim=0).clamp_min(1.0)
            weighted_refs = reference_features.to(language_features.dtype) * valid
            reference_bias = weighted_refs.sum(dim=0) / valid_count
        language_features = language_features + (
            reference_weight.reshape(1, 1, 1).to(language_features.dtype)
            * reference_bias.unsqueeze(0).expand(seq_len, batch_size, feat_dim)
        )
        return super().forward(
            vision_pos_enc_0,
            vision_pos_enc_1,
            vision_pos_enc_2,
            backbone_fpn_0,
            backbone_fpn_1,
            backbone_fpn_2,
            language_mask,
            language_features,
        )


class InteractiveDecoderWrapper(nn.Module):
    def __init__(self, image_model: nn.Module):
        super().__init__()
        self.predictor = image_model.inst_interactive_predictor

    def _run_decoder(
        self,
        sam2_backbone_fpn_0: torch.Tensor,
        sam2_backbone_fpn_1: torch.Tensor,
        sam2_backbone_fpn_2: torch.Tensor,
        image_pe: torch.Tensor,
        point_coords: torch.Tensor,
        point_labels: torch.Tensor,
        box_xyxy: torch.Tensor,
        box_valid_mask: torch.Tensor,
        mask_input: torch.Tensor,
        multimask_output: bool,
    ):
        batch_size = sam2_backbone_fpn_2.shape[0]
        device = sam2_backbone_fpn_2.device
        valid_point_labels = torch.where(point_labels >= 0, point_labels, torch.full_like(point_labels, -1)).to(torch.int32)
        concat_points = (point_coords, valid_point_labels)
        if box_xyxy.shape[1] > 0:
            box_coords = box_xyxy.reshape(batch_size, -1, 2, 2)
            box_labels = torch.tensor([[2, 3]], dtype=torch.int32, device=device).repeat(batch_size, box_xyxy.shape[1])
            flat_box_labels = torch.where(
                box_valid_mask.bool().repeat_interleave(2, dim=1),
                box_labels,
                torch.full_like(box_labels, -1),
            )
            concat_coords = torch.cat([box_coords.reshape(batch_size, -1, 2), point_coords], dim=1)
            concat_labels = torch.cat([flat_box_labels, valid_point_labels], dim=1)
            concat_points = (concat_coords, concat_labels)
        sam_mask_input = None if not torch.any(mask_input != 0) else mask_input
        sparse_embeddings, dense_embeddings = self.predictor.model.sam_prompt_encoder(
            points=concat_points,
            boxes=None,
            masks=sam_mask_input,
        )
        high_res_features = [sam2_backbone_fpn_0, sam2_backbone_fpn_1]
        decoder = self.predictor.model.sam_mask_decoder
        s = 1 if decoder.pred_obj_scores else 0
        if decoder.pred_obj_scores:
            output_tokens = torch.cat(
                [decoder.obj_score_token.weight, decoder.iou_token.weight, decoder.mask_tokens.weight], dim=0
            )
        else:
            output_tokens = torch.cat([decoder.iou_token.weight, decoder.mask_tokens.weight], dim=0)
        output_tokens = output_tokens.unsqueeze(0).expand(sparse_embeddings.size(0), -1, -1)
        tokens = torch.cat((output_tokens, sparse_embeddings), dim=1)
        src = sam2_backbone_fpn_2 + dense_embeddings
        pos_src = image_pe.expand(tokens.shape[0], -1, -1, -1)
        b, c, h, w = src.shape
        hs, src = decoder.transformer(src, pos_src, tokens)
        iou_token_out = hs[:, s, :]
        mask_tokens_out = hs[:, s + 1 : (s + 1 + decoder.num_mask_tokens), :]
        src = src.transpose(1, 2).view(b, c, h, w)
        dc1, ln1, act1, dc2, act2 = decoder.output_upscaling
        feat_s0, feat_s1 = high_res_features
        upscaled_embedding = act1(ln1(dc1(src) + feat_s1))
        upscaled_embedding = act2(dc2(upscaled_embedding) + feat_s0)
        hyper_in = torch.stack(
            [decoder.output_hypernetworks_mlps[i](mask_tokens_out[:, i, :]) for i in range(decoder.num_mask_tokens)],
            dim=1,
        )
        b, c, h, w = upscaled_embedding.shape
        all_masks = (hyper_in @ upscaled_embedding.view(b, c, h * w)).view(b, -1, h, w)
        all_iou_predictions = decoder.iou_prediction_head(iou_token_out)
        if multimask_output:
            return all_masks[:, 1:, :, :], all_iou_predictions[:, 1:]
        return all_masks[:, 0:1, :, :], all_iou_predictions[:, 0:1]

    def forward(
        self,
        sam2_backbone_fpn_0: torch.Tensor,
        sam2_backbone_fpn_1: torch.Tensor,
        sam2_backbone_fpn_2: torch.Tensor,
        image_pe: torch.Tensor,
        point_coords: torch.Tensor,
        point_labels: torch.Tensor,
        box_xyxy: torch.Tensor,
        box_valid_mask: torch.Tensor,
        mask_input: torch.Tensor,
    ):
        single_masks, single_scores = self._run_decoder(
            sam2_backbone_fpn_0,
            sam2_backbone_fpn_1,
            sam2_backbone_fpn_2,
            image_pe,
            point_coords,
            point_labels,
            box_xyxy,
            box_valid_mask,
            mask_input,
            multimask_output=False,
        )
        multi_masks, multi_scores = self._run_decoder(
            sam2_backbone_fpn_0,
            sam2_backbone_fpn_1,
            sam2_backbone_fpn_2,
            image_pe,
            point_coords,
            point_labels,
            box_xyxy,
            box_valid_mask,
            mask_input,
            multimask_output=True,
        )
        return single_masks, single_scores, multi_masks, multi_scores


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
        box_valid_mask: torch.Tensor,
        mask_input: torch.Tensor,
    ):
        del vision_pos_enc_0, vision_pos_enc_1, vision_pos_enc_2
        fpn = [backbone_fpn_0, backbone_fpn_1, backbone_fpn_2]
        high_res = [
            self.tracker.sam_mask_decoder.conv_s0(fpn[0]),
            self.tracker.sam_mask_decoder.conv_s1(fpn[1]),
        ]
        image_embeddings = fpn[2] + self.tracker.no_mem_embed.view(1, -1, 1, 1)
        valid_point_labels = torch.where(
            point_labels >= 0,
            point_labels,
            torch.full_like(point_labels, -1),
        ).to(torch.int32)
        box_coords = box_xyxy.reshape(point_coords.shape[0], -1, 2, 2)
        box_labels = torch.tensor([[2, 3]], dtype=torch.int32, device=box_xyxy.device).repeat(
            point_coords.shape[0], box_coords.shape[1]
        )
        box_labels = torch.where(
            box_valid_mask.bool().repeat_interleave(2, dim=1),
            box_labels,
            torch.full_like(box_labels, -1),
        )
        prompt_coords = torch.cat([box_coords.reshape(point_coords.shape[0], -1, 2), point_coords], dim=1)
        prompt_labels = torch.cat([box_labels, valid_point_labels], dim=1)
        sparse_embeddings, dense_embeddings = self.tracker.sam_prompt_encoder(
            points=(prompt_coords, prompt_labels),
            boxes=None,
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
        enable_inst_interactivity=True,
    ).float().eval()
    _patch_vitdet_rope_for_export(image_model)
    _load_interactive_predictor_weights(image_model, args.checkpoint)

    tracker = build_tracker(apply_temporal_disambiguation=True).to(device).float().eval()
    _load_tracker_checkpoint(tracker, args.checkpoint)
    _patch_tracker_rope_for_export(tracker)

    image_encoder = ImageEncoderWrapper(image_model).to(device).eval()
    text_encoder = TextEncoderWrapper(image_model).to(device).eval()
    grounding_decoder = GroundingDecoderWrapper(image_model).to(device).eval()
    reference_feature_encoder = ReferenceFeatureEncoderWrapper().to(device).eval()
    grounding_decoder_with_reference = GroundingDecoderWithReferenceWrapper(image_model).to(device).eval()
    interactive_decoder = InteractiveDecoderWrapper(image_model).to(device).eval()
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
    sample_box_valid = torch.tensor([[True]], dtype=torch.bool, device=device)
    sample_mask = torch.zeros((1, 1, 288, 288), dtype=torch.float32, device=device)
    with torch.no_grad():
        image_feats = image_encoder(sample_image)
        language_feats = text_encoder(sample_tokens)
    sample_reference_features = torch.zeros((1, 1, image_feats[3].shape[1]), dtype=torch.float32, device=device)
    sample_reference_valid = torch.zeros((1, 1), dtype=torch.bool, device=device)
    sample_reference_weight = torch.tensor([1.0], dtype=torch.float32, device=device)
    sample_interactive_pe = (
        image_model.inst_interactive_predictor.model.sam_prompt_encoder.get_dense_pe().to(device)
    )

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
                "sam2_vision_pos_enc_0",
                "sam2_vision_pos_enc_1",
                "sam2_vision_pos_enc_2",
                "sam2_backbone_fpn_0",
                "sam2_backbone_fpn_1",
                "sam2_backbone_fpn_2",
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
            name="sam31_reference_feature_encoder",
            model=reference_feature_encoder,
            inputs=(image_feats[3], sample_box),
            input_names=("backbone_fpn_0", "reference_boxes_xyxy"),
            output_names=("reference_embedding",),
        ),
        ExportSpec(
            name="sam31_grounding_decoder_with_reference",
            model=grounding_decoder_with_reference,
            inputs=(
                image_feats[0],
                image_feats[1],
                image_feats[2],
                image_feats[3],
                image_feats[4],
                image_feats[5],
                language_feats[0],
                language_feats[1],
                sample_reference_features,
                sample_reference_valid,
                sample_reference_weight,
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
                "reference_features",
                "reference_valid_mask",
                "reference_weight",
            ),
            output_names=("pred_logits", "pred_boxes_xyxy", "pred_masks"),
        ),
        ExportSpec(
            name="sam31_interactive_decoder",
            model=interactive_decoder,
            inputs=(
                image_feats[9],
                image_feats[10],
                image_feats[11],
                sample_interactive_pe,
                sample_points,
                sample_point_labels,
                sample_box.unsqueeze(1),
                sample_box_valid,
                sample_mask,
            ),
            input_names=(
                "sam2_backbone_fpn_0",
                "sam2_backbone_fpn_1",
                "sam2_backbone_fpn_2",
                "image_pe",
                "point_coords",
                "point_labels",
                "box_xyxy",
                "box_valid_mask",
                "mask_input",
            ),
            output_names=("single_masks", "single_scores", "multi_masks", "multi_scores"),
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
                sample_box_valid,
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
                "box_valid_mask",
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
    interactive_path = Path(exported["sam31_interactive_decoder"])
    for alias in ("sam31_point_decoder", "sam31_box_decoder"):
        alias_dir = output_dir / alias
        alias_dir.mkdir(parents=True, exist_ok=True)
        alias_path = alias_dir / f"{alias}.onnx"
        shutil.copy2(interactive_path, alias_path)
        exported[alias] = str(alias_path)
    np.save(output_dir / "sam31_interactive_dense_pe.npy", sample_interactive_pe.detach().cpu().numpy())

    meta = {
        "checkpoint": args.checkpoint,
        "exports": exported,
        "device": args.device,
    }
    (output_dir / "export_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
