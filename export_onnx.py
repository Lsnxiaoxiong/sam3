#!/usr/bin/env python3

from __future__ import annotations

import argparse
import pathlib
from dataclasses import dataclass
from importlib import resources

import numpy as np
import torch
from torchvision.transforms import v2

from sam3.model import vitdet
from sam3.model.box_ops import box_cxcywh_to_xyxy
from sam3.model.data_misc import FindStage
from sam3.model.geometry_encoders import Prompt
from sam3.model.sam3_image import Sam3Image
from sam3.model.sam3_image_processor import Sam3Processor
from sam3.model.tokenizer_ve import SimpleTokenizer
from sam3.model_builder import build_sam3_image_model
from sam3.sam.rope import apply_rotary_enc_real


def patch_vitdet_rope_for_export() -> None:
    def _apply_rope(self: vitdet.Attention, q: torch.Tensor, k: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if not self.use_rope:
            return q, k
        return apply_rotary_enc_real(
            q,
            k,
            freqs_cis_real=self.freqs_cis_real,
            freqs_cis_imag=self.freqs_cis_imag,
        )

    vitdet.Attention._apply_rope = _apply_rope


def prepare_vitdet_rope_buffers(model: torch.nn.Module) -> None:
    for module in model.modules():
        if isinstance(module, vitdet.Attention) and getattr(module, "use_rope", False):
            module.register_buffer("freqs_cis_real", module.freqs_cis.real.float())
            module.register_buffer("freqs_cis_imag", module.freqs_cis.imag.float())


def prepare_decoder_coordinate_caches(model: torch.nn.Module, device: torch.device) -> None:
    decoder = model.transformer.decoder
    if getattr(decoder, "compilable_cord_cache", None) is not None:
        coords_h, coords_w = decoder.compilable_cord_cache
        decoder.compilable_cord_cache = (
            coords_h.to(device=device, dtype=torch.float32),
            coords_w.to(device=device, dtype=torch.float32),
        )
    if getattr(decoder, "coord_cache", None):
        decoder.coord_cache = {
            feat_size: (
                cached_h.to(device=device, dtype=torch.float32),
                cached_w.to(device=device, dtype=torch.float32),
            )
            for feat_size, (cached_h, cached_w) in decoder.coord_cache.items()
        }


def _resource_bpe_path() -> str:
    return str(resources.files("sam3").joinpath("assets/bpe_simple_vocab_16e6.txt.gz"))


@dataclass
class ExportArtifacts:
    image_encoder: pathlib.Path
    text_encoder: pathlib.Path
    grounding_decoder: pathlib.Path
    interactive_decoder: pathlib.Path
    interactive_dense_pe: pathlib.Path


class ImageEncoderWrapper(torch.nn.Module):
    def __init__(self, processor: Sam3Processor) -> None:
        super().__init__()
        self.model = processor.model
        self.transform = v2.Compose(
            [
                v2.ToDtype(torch.float32, scale=True),
                v2.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
            ]
        )

    def forward(self, image: torch.Tensor) -> tuple[torch.Tensor, ...]:
        image = self.transform(image)
        backbone_out = self.model.backbone._forward_image_no_act_ckpt(image)
        sam2_backbone_out = backbone_out["sam2_backbone_out"]
        sam2_backbone_out["backbone_fpn"][0] = (
            self.model.inst_interactive_predictor.model.sam_mask_decoder.conv_s0(
                sam2_backbone_out["backbone_fpn"][0]
            )
        )
        sam2_backbone_out["backbone_fpn"][1] = (
            self.model.inst_interactive_predictor.model.sam_mask_decoder.conv_s1(
                sam2_backbone_out["backbone_fpn"][1]
            )
        )

        return (
            backbone_out["vision_pos_enc"][0],
            backbone_out["vision_pos_enc"][1],
            backbone_out["vision_pos_enc"][2],
            backbone_out["backbone_fpn"][0],
            backbone_out["backbone_fpn"][1],
            backbone_out["backbone_fpn"][2],
            sam2_backbone_out["vision_pos_enc"][0],
            sam2_backbone_out["vision_pos_enc"][1],
            sam2_backbone_out["vision_pos_enc"][2],
            sam2_backbone_out["backbone_fpn"][0],
            sam2_backbone_out["backbone_fpn"][1],
            sam2_backbone_out["backbone_fpn"][2],
        )


class TextEncoderWrapper(torch.nn.Module):
    def __init__(self, model: Sam3Image) -> None:
        super().__init__()
        self.model = model

    def forward(self, tokens: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        text_attention_mask = (tokens != 0).bool()
        _, text_memory = self.model.backbone.language_backbone.encoder(tokens)
        text_attention_mask = text_attention_mask.ne(1)
        text_memory = text_memory.transpose(0, 1)
        text_memory_resized = self.model.backbone.language_backbone.resizer(text_memory)
        return text_attention_mask, text_memory_resized


class GroundingDecoderWrapper(torch.nn.Module):
    def __init__(self, model: Sam3Image) -> None:
        super().__init__()
        self.model = model

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
        box_coords: torch.Tensor,
        box_valid_mask: torch.Tensor,
        box_labels: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        batch_size = backbone_fpn_2.shape[0]
        device = backbone_fpn_2.device

        # Keep one geometric token alive during tracing so the exported graph
        # does not bake in empty prompt sequence shapes for text-only grounding.
        dummy_points = torch.full(
            (1, batch_size, 2),
            0.5,
            dtype=torch.float32,
            device=device,
        )
        dummy_point_labels = torch.zeros((1, batch_size), dtype=torch.long, device=device)
        dummy_point_mask = torch.zeros((batch_size, 1), dtype=torch.bool, device=device)

        backbone_out = {
            "vision_features": backbone_fpn_2,
            "vision_pos_enc": [vision_pos_enc_0, vision_pos_enc_1, vision_pos_enc_2],
            "backbone_fpn": [backbone_fpn_0, backbone_fpn_1, backbone_fpn_2],
            "language_features": language_features,
            "language_mask": language_mask,
        }
        find_input = FindStage(
            img_ids=torch.arange(batch_size, device=device, dtype=torch.long),
            text_ids=torch.arange(batch_size, device=device, dtype=torch.long),
            input_boxes=None,
            input_boxes_mask=None,
            input_boxes_label=None,
            input_points=None,
            input_points_mask=None,
        )
        geometric_prompt = Prompt(
            box_embeddings=box_coords.permute(1, 0, 2),
            box_mask=(~box_valid_mask.bool()),
            box_labels=box_labels.permute(1, 0).to(torch.long),
            point_embeddings=dummy_points,
            point_mask=dummy_point_mask,
            point_labels=dummy_point_labels,
        )
        outputs = self.model.forward_grounding(
            backbone_out=backbone_out,
            find_input=find_input,
            geometric_prompt=geometric_prompt,
            find_target=None,
        )

        scores = (
            outputs["pred_logits"].sigmoid()
            * outputs["presence_logit_dec"].sigmoid().unsqueeze(-1)
        ).squeeze(-1)
        boxes_xyxy = box_cxcywh_to_xyxy(outputs["pred_boxes"])
        return boxes_xyxy, scores, outputs["pred_masks"]


class InteractiveDecoderWrapper(torch.nn.Module):
    def __init__(self, model: Sam3Image) -> None:
        super().__init__()
        self.predictor = model.inst_interactive_predictor

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
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size = sam2_backbone_fpn_2.shape[0]
        device = sam2_backbone_fpn_2.device

        valid_point_labels = torch.where(
            point_labels >= 0,
            point_labels,
            torch.full_like(point_labels, -1),
        ).to(torch.int32)
        concat_points: tuple[torch.Tensor, torch.Tensor] | None = (
            point_coords,
            valid_point_labels,
        )

        if box_xyxy.shape[1] > 0:
            box_coords = box_xyxy.reshape(batch_size, -1, 2, 2)
            box_labels = torch.tensor([[2, 3]], dtype=torch.int32, device=device).repeat(
                batch_size, box_xyxy.shape[1]
            )
            flat_box_labels = torch.where(
                box_valid_mask.bool().repeat_interleave(2, dim=1),
                box_labels,
                torch.full_like(box_labels, -1),
            )
            concat_coords = torch.cat([box_coords.reshape(batch_size, -1, 2), point_coords], dim=1)
            concat_labels = torch.cat([flat_box_labels, valid_point_labels], dim=1)
            concat_points = (concat_coords, concat_labels)

        sam_mask_input = None
        if torch.any(mask_input != 0):
            sam_mask_input = mask_input

        sparse_embeddings, dense_embeddings = self.predictor.model.sam_prompt_encoder(
            points=concat_points,
            boxes=None,
            masks=sam_mask_input,
        )
        high_res_features = [sam2_backbone_fpn_0, sam2_backbone_fpn_1]
        decoder = self.predictor.model.sam_mask_decoder
        s = 0
        if decoder.pred_obj_scores:
            output_tokens = torch.cat(
                [
                    decoder.obj_score_token.weight,
                    decoder.iou_token.weight,
                    decoder.mask_tokens.weight,
                ],
                dim=0,
            )
            s = 1
        else:
            output_tokens = torch.cat(
                [decoder.iou_token.weight, decoder.mask_tokens.weight], dim=0
            )
        output_tokens = output_tokens.unsqueeze(0).expand(
            sparse_embeddings.size(0), -1, -1
        )
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
        hyper_in_list = []
        for i in range(decoder.num_mask_tokens):
            hyper_in_list.append(decoder.output_hypernetworks_mlps[i](mask_tokens_out[:, i, :]))
        hyper_in = torch.stack(hyper_in_list, dim=1)
        b, c, h, w = upscaled_embedding.shape
        all_masks = (hyper_in @ upscaled_embedding.view(b, c, h * w)).view(b, -1, h, w)
        all_iou_predictions = decoder.iou_prediction_head(iou_token_out)
        if multimask_output:
            low_res_masks = all_masks[:, 1:, :, :]
            iou_predictions = all_iou_predictions[:, 1:]
        else:
            low_res_masks = all_masks[:, 0:1, :, :]
            iou_predictions = all_iou_predictions[:, 0:1]
        return low_res_masks, iou_predictions

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
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--checkpoint", type=pathlib.Path, required=True)
    parser.add_argument("--output-dir", type=pathlib.Path, default=pathlib.Path("models"))
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--opset", type=int, default=17)
    parser.add_argument(
        "--modules",
        nargs="+",
        choices=["all", "image_encoder", "text_encoder", "grounding_decoder", "interactive_decoder"],
        default=["all"],
    )
    return parser.parse_args()


def _dynamic_axes() -> dict[str, dict[int, str]]:
    return {
        "image": {0: "batch"},
        "tokens": {0: "batch"},
        "vision_pos_enc_0": {0: "batch"},
        "vision_pos_enc_1": {0: "batch"},
        "vision_pos_enc_2": {0: "batch"},
        "backbone_fpn_0": {0: "batch"},
        "backbone_fpn_1": {0: "batch"},
        "backbone_fpn_2": {0: "batch"},
        "sam2_backbone_fpn_0": {0: "batch"},
        "sam2_backbone_fpn_1": {0: "batch"},
        "sam2_backbone_fpn_2": {0: "batch"},
        "image_pe": {0: "batch"},
        "language_mask": {0: "batch"},
        "language_features": {1: "batch"},
        "box_coords": {0: "batch", 1: "num_boxes"},
        "box_valid_mask": {0: "batch", 1: "num_boxes"},
        "box_labels": {0: "batch", 1: "num_boxes"},
        "point_coords": {0: "batch", 1: "num_points"},
        "point_labels": {0: "batch", 1: "num_points"},
        "box_xyxy": {0: "batch", 1: "num_boxes"},
        "single_masks": {0: "batch"},
        "single_scores": {0: "batch"},
        "multi_masks": {0: "batch"},
        "multi_scores": {0: "batch"},
        "boxes_xyxy": {0: "batch"},
        "scores": {0: "batch"},
        "masks_logits": {0: "batch"},
    }


def _export(
    module: torch.nn.Module,
    args: tuple[torch.Tensor, ...],
    output_path: pathlib.Path,
    input_names: list[str],
    output_names: list[str],
    opset: int,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with torch.inference_mode():
        torch.onnx.export(
            module,
            args=args,
            f=output_path,
            input_names=input_names,
            output_names=output_names,
            opset_version=opset,
            do_constant_folding=True,
            dynamic_axes={k: v for k, v in _dynamic_axes().items() if k in input_names + output_names},
            external_data=True,
        )


def build_artifacts(output_dir: pathlib.Path) -> ExportArtifacts:
    return ExportArtifacts(
        image_encoder=output_dir / "sam3_image_encoder.onnx",
        text_encoder=output_dir / "sam3_text_encoder.onnx",
        grounding_decoder=output_dir / "sam3_grounding_decoder.onnx",
        interactive_decoder=output_dir / "sam3_interactive_decoder.onnx",
        interactive_dense_pe=output_dir / "sam3_interactive_dense_pe.npy",
    )


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)

    patch_vitdet_rope_for_export()
    model = build_sam3_image_model(
        checkpoint_path=str(args.checkpoint),
        load_from_HF=False,
        enable_inst_interactivity=True,
        device=args.device,
    )
    model = model.float()
    model.eval()
    prepare_vitdet_rope_buffers(model)
    prepare_decoder_coordinate_caches(model, device)
    processor = Sam3Processor(model, device=args.device)
    tokenizer = SimpleTokenizer(_resource_bpe_path(), context_length=32)
    artifacts = build_artifacts(args.output_dir)

    image_encoder = ImageEncoderWrapper(processor).to(device).eval()
    text_encoder = TextEncoderWrapper(model).to(device).eval()
    grounding_decoder = GroundingDecoderWrapper(model).to(device).eval()
    interactive_decoder = InteractiveDecoderWrapper(model).to(device).eval()

    sample_image = torch.randint(0, 255, (1, 3, 1008, 1008), device=device, dtype=torch.uint8)
    sample_tokens = tokenizer(["person"]).to(device)
    sample_box_coords = torch.tensor([[[0.5, 0.5, 0.2, 0.2]]], device=device, dtype=torch.float32)
    sample_box_valid_mask = torch.tensor([[True]], device=device, dtype=torch.bool)
    sample_box_labels = torch.tensor([[1]], device=device, dtype=torch.bool)
    sample_point_coords = torch.zeros((1, 1, 2), device=device, dtype=torch.float32)
    sample_point_labels = torch.full((1, 1), -1, device=device, dtype=torch.int32)
    sample_box_xyxy = torch.zeros((1, 1, 4), device=device, dtype=torch.float32)
    sample_box_xyxy_valid_mask = torch.tensor([[False]], device=device, dtype=torch.bool)
    sample_mask_input = torch.zeros((1, 1, 256, 256), device=device, dtype=torch.float32)
    sample_interactive_image_pe = (
        interactive_decoder.predictor.model.sam_prompt_encoder.get_dense_pe().to(device)
    )

    requested = set(args.modules)
    image_outputs: tuple[torch.Tensor, ...] | None = None
    language_mask: torch.Tensor | None = None
    language_features: torch.Tensor | None = None

    need_image_outputs = bool({"all", "grounding_decoder", "interactive_decoder"} & requested)
    need_language_outputs = bool({"all", "grounding_decoder"} & requested)
    with torch.no_grad():
        if need_image_outputs:
            image_outputs = image_encoder(sample_image)
        if need_language_outputs:
            language_mask, language_features = text_encoder(sample_tokens)

    if "all" in requested or "image_encoder" in requested:
        _export(
            image_encoder,
            (sample_image,),
            artifacts.image_encoder,
            ["image"],
            [
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
            ],
            args.opset,
        )
    if "all" in requested or "text_encoder" in requested:
        _export(
            text_encoder,
            (sample_tokens,),
            artifacts.text_encoder,
            ["tokens"],
            ["language_mask", "language_features"],
            args.opset,
        )
    if "all" in requested or "grounding_decoder" in requested:
        assert image_outputs is not None
        assert language_mask is not None
        assert language_features is not None
        _export(
            grounding_decoder,
            (
                image_outputs[0],
                image_outputs[1],
                image_outputs[2],
                image_outputs[3],
                image_outputs[4],
                image_outputs[5],
                language_mask,
                language_features,
                sample_box_coords,
                sample_box_valid_mask,
                sample_box_labels,
            ),
            artifacts.grounding_decoder,
            [
                "vision_pos_enc_0",
                "vision_pos_enc_1",
                "vision_pos_enc_2",
                "backbone_fpn_0",
                "backbone_fpn_1",
                "backbone_fpn_2",
                "language_mask",
                "language_features",
                "box_coords",
                "box_valid_mask",
                "box_labels",
            ],
            ["boxes_xyxy", "scores", "masks_logits"],
            args.opset,
        )
    if "all" in requested or "interactive_decoder" in requested:
        assert image_outputs is not None
        _export(
            interactive_decoder,
            (
                image_outputs[9],
                image_outputs[10],
                image_outputs[11],
                sample_interactive_image_pe,
                sample_point_coords,
                sample_point_labels,
                sample_box_xyxy,
                sample_box_xyxy_valid_mask,
                sample_mask_input,
            ),
            artifacts.interactive_decoder,
            [
                "sam2_backbone_fpn_0",
                "sam2_backbone_fpn_1",
                "sam2_backbone_fpn_2",
                "image_pe",
                "point_coords",
                "point_labels",
                "box_xyxy",
                "box_valid_mask",
                "mask_input",
            ],
            ["single_masks", "single_scores", "multi_masks", "multi_scores"],
            args.opset,
        )
        artifacts.interactive_dense_pe.parent.mkdir(parents=True, exist_ok=True)
        np.save(
            artifacts.interactive_dense_pe,
            sample_interactive_image_pe.detach().float().cpu().numpy(),
        )

    print("Exported:")
    for path in [
        artifacts.image_encoder,
        artifacts.text_encoder,
        artifacts.grounding_decoder,
        artifacts.interactive_decoder,
        artifacts.interactive_dense_pe,
    ]:
        if path.exists():
            print(f"  {path}")


if __name__ == "__main__":
    main()

"""
python export_onnx.py --checkpoint C:\Users\lsn\.cache\modelscope\hub\models\facebook\sam3\sam3.pt --output-dir output --device cpu

"""