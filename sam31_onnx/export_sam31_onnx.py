import argparse
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.transforms import v2

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sam3.model.box_ops import box_cxcywh_to_xyxy  # noqa: E402
from sam3.model.data_misc import FindStage  # noqa: E402
from sam3.model.decoder import TransformerDecoder  # noqa: E402
from sam3.model.geometry_encoders import Prompt  # noqa: E402
from sam3.model.memory import SimpleMaskDownSampler  # noqa: E402
from sam3.model.tokenizer_ve import SimpleTokenizer  # noqa: E402
from sam3.model.vitdet import Attention as VitAttention  # noqa: E402
from sam3.model_builder import build_sam3_image_model, build_tracker  # noqa: E402
from sam3.sam.rope import apply_rotary_enc_real  # noqa: E402
from sam3.sam.transformer import RoPEAttention  # noqa: E402


IMAGE_SIZE = 1008
TEXT_CONTEXT = 32


def _load_checkpoint_raw(checkpoint_path: str) -> Dict[str, torch.Tensor]:
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if "model" in ckpt and isinstance(ckpt["model"], dict):
        return ckpt["model"]
    return ckpt


def _load_tracker_checkpoint(model: nn.Module, checkpoint_path: str) -> None:
    ckpt = _load_checkpoint_raw(checkpoint_path)
    tracker_state_raw = {
        k.replace("tracker.model.", ""): v
        for k, v in ckpt.items()
        if k.startswith("tracker.model.")
    }
    model_state = model.state_dict()
    tracker_state = {
        k: v
        for k, v in tracker_state_raw.items()
        if k in model_state and tuple(model_state[k].shape) == tuple(v.shape)
    }
    model.load_state_dict(tracker_state, strict=False)


def _patch_vitdet_rope_for_export(model: Optional[nn.Module] = None) -> None:
    def _apply_rope_real(self: VitAttention, q, k):
        if not self.use_rope:
            return q, k
        return apply_rotary_enc_real(
            q,
            k,
            freqs_cis_real=self.freqs_cis_real,
            freqs_cis_imag=self.freqs_cis_imag,
            repeat_freqs_k=False,
        )

    VitAttention._apply_rope = _apply_rope_real
    if model is not None:
        for module in model.modules():
            if isinstance(module, VitAttention) and getattr(module, "use_rope", False):
                freqs_cis = module.freqs_cis
                module.freqs_cis_real = freqs_cis.real
                module.freqs_cis_imag = freqs_cis.imag


def _patch_tracker_rope_for_export(model: nn.Module) -> None:
    for module in model.modules():
        if isinstance(module, RoPEAttention):
            module.use_rope_real = True
            module.freqs_cis_real = module.freqs_cis.real
            module.freqs_cis_imag = module.freqs_cis.imag
            module.use_fa3 = False


def _patch_mask_downsampler_for_export() -> None:
    def _forward(self: SimpleMaskDownSampler, x: torch.Tensor):
        if self.interpol_size is not None and self.interpol_size != list(x.shape[-2:]):
            x = F.interpolate(
                x.float(),
                size=self.interpol_size,
                align_corners=False,
                mode="bilinear",
                antialias=False,
            )
        return self.encoder(x)

    SimpleMaskDownSampler.forward = _forward


def _patch_decoder_for_export() -> None:
    def _get_rpb_matrix_no_cache(self, reference_boxes, feat_size):
        H, W = feat_size
        boxes_xyxy = box_cxcywh_to_xyxy(reference_boxes).transpose(0, 1)
        bs, num_queries, _ = boxes_xyxy.shape
        coords_h = (
            torch.arange(0, H, device=reference_boxes.device, dtype=torch.float32) / H
        )
        coords_w = (
            torch.arange(0, W, device=reference_boxes.device, dtype=torch.float32) / W
        )
        deltas_y = coords_h.view(1, -1, 1) - boxes_xyxy.reshape(-1, 1, 4)[:, :, 1:4:2]
        deltas_y = deltas_y.view(bs, num_queries, -1, 2)
        deltas_x = coords_w.view(1, -1, 1) - boxes_xyxy.reshape(-1, 1, 4)[:, :, 0:3:2]
        deltas_x = deltas_x.view(bs, num_queries, -1, 2)
        if self.boxRPB in ["log", "both"]:
            deltas_x_log = deltas_x * 8
            deltas_x_log = (
                torch.sign(deltas_x_log)
                * torch.log2(torch.abs(deltas_x_log) + 1.0)
                / math.log2(8)
            )
            deltas_y_log = deltas_y * 8
            deltas_y_log = (
                torch.sign(deltas_y_log)
                * torch.log2(torch.abs(deltas_y_log) + 1.0)
                / math.log2(8)
            )
            if self.boxRPB == "log":
                deltas_x = deltas_x_log
                deltas_y = deltas_y_log
            else:
                deltas_x = torch.cat([deltas_x, deltas_x_log], dim=-1)
                deltas_y = torch.cat([deltas_y, deltas_y_log], dim=-1)
        deltas_x = self.boxRPB_embed_x(deltas_x)
        deltas_y = self.boxRPB_embed_y(deltas_y)
        B = deltas_y.unsqueeze(3) + deltas_x.unsqueeze(2)
        B = B.flatten(2, 3)
        B = B.permute(0, 3, 1, 2).contiguous()
        return B

    TransformerDecoder._get_rpb_matrix = _get_rpb_matrix_no_cache


def _preprocess_image(image_path: str, device: torch.device) -> torch.Tensor:
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
    return transform(image).unsqueeze(0).to(device)


def _make_find_stage(batch_size: int, device: torch.device) -> FindStage:
    return FindStage(
        img_ids=torch.arange(batch_size, device=device, dtype=torch.long),
        text_ids=torch.arange(batch_size, device=device, dtype=torch.long),
        input_boxes=None,
        input_boxes_mask=None,
        input_boxes_label=None,
        input_points=None,
        input_points_mask=None,
    )


def _make_stable_prompt(batch_size: int, device: torch.device) -> Prompt:
    box_embeddings = torch.tensor(
        [[[0.5, 0.5, 1.0, 1.0]]], dtype=torch.float32, device=device
    ).expand(-1, batch_size, -1)
    box_mask = torch.zeros((batch_size, 1), dtype=torch.bool, device=device)
    box_labels = torch.zeros((1, batch_size), dtype=torch.long, device=device)
    point_embeddings = torch.tensor(
        [[[0.5, 0.5]]], dtype=torch.float32, device=device
    ).expand(-1, batch_size, -1)
    point_mask = torch.zeros((batch_size, 1), dtype=torch.bool, device=device)
    point_labels = torch.zeros((1, batch_size), dtype=torch.long, device=device)
    return Prompt(
        box_embeddings=box_embeddings,
        box_mask=box_mask,
        box_labels=box_labels,
        point_embeddings=point_embeddings,
        point_mask=point_mask,
        point_labels=point_labels,
    )


def _token_features(
    text_encoder: nn.Module,
    token_ids: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    text_attention_mask = (token_ids != 0).bool()
    _, text_memory = text_encoder.encoder(token_ids)
    inputs_embeds = text_encoder.encoder.token_embedding(token_ids)
    text_attention_mask = text_attention_mask.ne(1)
    text_memory = text_memory.transpose(0, 1)
    text_memory_resized = text_encoder.resizer(text_memory)
    return text_attention_mask, text_memory_resized, inputs_embeds.transpose(0, 1)


class GroundingWrapper(nn.Module):
    def __init__(self, image_model: nn.Module):
        super().__init__()
        self.image_model = image_model
        self.text_encoder = image_model.backbone.language_backbone

    def forward(self, image: torch.Tensor, token_ids: torch.Tensor):
        backbone_out = self.image_model.backbone.forward_image(image)
        language_mask, language_features, language_embeds = _token_features(
            self.text_encoder, token_ids
        )
        backbone_out.update(
            {
                "language_mask": language_mask,
                "language_features": language_features,
                "language_embeds": language_embeds,
            }
        )
        out = self.image_model.forward_grounding(
            backbone_out=backbone_out,
            find_input=_make_find_stage(image.shape[0], image.device),
            find_target=None,
            geometric_prompt=_make_stable_prompt(image.shape[0], image.device),
        )
        return out["pred_logits"], out["pred_boxes_xyxy"], out["pred_masks"]


class VideoStepWrapper(nn.Module):
    def __init__(self, image_model: nn.Module, tracker: nn.Module):
        super().__init__()
        self.image_model = image_model
        self.tracker = tracker
        self.register_buffer(
            "dense_pe",
            tracker.sam_prompt_encoder.get_dense_pe().detach().clone(),
            persistent=False,
        )

    def forward(
        self,
        image: torch.Tensor,
        point_coords: torch.Tensor,
        point_labels: torch.Tensor,
        box_xyxy: torch.Tensor,
        mask_input: torch.Tensor,
        prev_maskmem_features: torch.Tensor,
        prev_maskmem_pos_enc: torch.Tensor,
        prev_obj_ptr: torch.Tensor,
        prev_valid: torch.Tensor,
    ):
        backbone_out = self.image_model.backbone.forward_image(image)
        fpn = backbone_out["backbone_fpn"]
        feat_sizes = [(x.shape[-2], x.shape[-1]) for x in fpn]
        current_vision_feats = [x.flatten(2).permute(2, 0, 1) for x in fpn]
        current_vision_pos = [
            x.flatten(2).permute(2, 0, 1) for x in backbone_out["vision_pos_enc"]
        ]
        batch = image.shape[0]
        channels = current_vision_feats[-1].shape[-1]
        height, width = feat_sizes[-1]

        if bool(prev_valid.item()):
            mem_feat = prev_maskmem_features.flatten(2).permute(2, 0, 1)
            mem_pos = prev_maskmem_pos_enc.flatten(2).permute(2, 0, 1)
            tpos = self.tracker.maskmem_tpos_enc[self.tracker.num_maskmem - 2]
            mem_pos = mem_pos + tpos
            obj_ptr = prev_obj_ptr.unsqueeze(0)
            obj_pos = self.tracker._get_tpos_enc(
                [1], self.tracker.max_obj_ptrs_in_encoder, image.device
            )
            obj_pos = self.tracker.obj_ptr_tpos_proj(obj_pos).unsqueeze(1).expand(-1, batch, -1)
            prompt = torch.cat([mem_feat, obj_ptr], dim=0)
            prompt_pos = torch.cat([mem_pos, obj_pos], dim=0)
            encoder_out = self.tracker.transformer.encoder(
                src=current_vision_feats[-1:],
                src_key_padding_mask=[None],
                src_pos=current_vision_pos[-1:],
                prompt=prompt,
                prompt_pos=prompt_pos,
                prompt_key_padding_mask=None,
                feat_sizes=feat_sizes[-1:],
                num_obj_ptr_tokens=1,
            )
            pix_feat_with_mem = encoder_out["memory"].permute(1, 2, 0).view(
                batch, channels, height, width
            )
        else:
            pix_feat_with_mem = (current_vision_feats[-1] + self.tracker.no_mem_embed).permute(
                1, 2, 0
            ).view(batch, channels, height, width)

        high_res = [
            self.tracker.sam_mask_decoder.conv_s0(fpn[0]),
            self.tracker.sam_mask_decoder.conv_s1(fpn[1]),
        ]
        sparse_embeddings, dense_embeddings = self.tracker.sam_prompt_encoder(
            points=(point_coords, point_labels),
            boxes=box_xyxy,
            masks=mask_input,
        )
        low_res_masks, ious, sam_output_tokens, object_score_logits = self.tracker.sam_mask_decoder(
            image_embeddings=pix_feat_with_mem,
            image_pe=self.dense_pe.to(
                device=pix_feat_with_mem.device, dtype=pix_feat_with_mem.dtype
            ),
            sparse_prompt_embeddings=sparse_embeddings,
            dense_prompt_embeddings=dense_embeddings,
            multimask_output=False,
            repeat_image=False,
            high_res_features=high_res,
        )
        high_res_masks = F.interpolate(
            low_res_masks.float(),
            size=(IMAGE_SIZE, IMAGE_SIZE),
            mode="bilinear",
            align_corners=False,
        )
        obj_ptr = self.tracker.obj_ptr_proj(sam_output_tokens[:, 0])
        maskmem_features, maskmem_pos_enc = self.tracker._encode_new_memory(
            image=image,
            current_vision_feats=current_vision_feats,
            feat_sizes=feat_sizes,
            pred_masks_high_res=high_res_masks,
            object_score_logits=object_score_logits,
            is_mask_from_pts=bool(point_coords.shape[1] > 0),
            output_dict=None,
            is_init_cond_frame=not bool(prev_valid.item()),
        )
        return (
            low_res_masks,
            high_res_masks,
            obj_ptr,
            object_score_logits,
            maskmem_features,
            maskmem_pos_enc[-1],
        )


class PointPromptWrapper(nn.Module):
    def __init__(self, image_model: nn.Module, tracker: nn.Module):
        super().__init__()
        self.image_model = image_model
        self.tracker = tracker
        self.register_buffer(
            "dense_pe",
            tracker.sam_prompt_encoder.get_dense_pe().detach().clone(),
            persistent=False,
        )

    def forward(
        self,
        image: torch.Tensor,
        point_coords: torch.Tensor,
        point_labels: torch.Tensor,
        mask_input: torch.Tensor,
    ):
        backbone_out = self.image_model.backbone.forward_image(image)
        fpn = backbone_out["backbone_fpn"]
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
            image_pe=self.dense_pe.to(
                device=image_embeddings.device, dtype=image_embeddings.dtype
            ),
            sparse_prompt_embeddings=sparse_embeddings,
            dense_prompt_embeddings=dense_embeddings,
            multimask_output=True,
            repeat_image=False,
            high_res_features=high_res,
        )
        high_res_masks = F.interpolate(
            low_res_masks.float(),
            size=(IMAGE_SIZE, IMAGE_SIZE),
            mode="bilinear",
            align_corners=False,
        )
        return low_res_masks, high_res_masks, ious, object_score_logits


class BoxPromptWrapper(nn.Module):
    def __init__(self, image_model: nn.Module, tracker: nn.Module):
        super().__init__()
        self.image_model = image_model
        self.tracker = tracker
        self.register_buffer(
            "dense_pe",
            tracker.sam_prompt_encoder.get_dense_pe().detach().clone(),
            persistent=False,
        )

    def forward(
        self,
        image: torch.Tensor,
        box_xyxy: torch.Tensor,
        mask_input: torch.Tensor,
    ):
        backbone_out = self.image_model.backbone.forward_image(image)
        fpn = backbone_out["backbone_fpn"]
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
            image_pe=self.dense_pe.to(
                device=image_embeddings.device, dtype=image_embeddings.dtype
            ),
            sparse_prompt_embeddings=sparse_embeddings,
            dense_prompt_embeddings=dense_embeddings,
            multimask_output=True,
            repeat_image=False,
            high_res_features=high_res,
        )
        high_res_masks = F.interpolate(
            low_res_masks.float(),
            size=(IMAGE_SIZE, IMAGE_SIZE),
            mode="bilinear",
            align_corners=False,
        )
        return low_res_masks, high_res_masks, ious, object_score_logits


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
    parser.add_argument("--output-dir", default="output/sam31_onnx")
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

    text_model = GroundingWrapper(image_model).to(device).eval()
    point_model = PointPromptWrapper(image_model, tracker).to(device).eval()
    box_model = BoxPromptWrapper(image_model, tracker).to(device).eval()
    video_model = VideoStepWrapper(image_model, tracker).to(device).eval()

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
    prev_mem = torch.zeros((1, 64, 72, 72), dtype=torch.float32, device=device)
    prev_pos = torch.zeros_like(prev_mem)
    prev_obj_ptr = torch.zeros((1, 256), dtype=torch.float32, device=device)
    prev_valid = torch.tensor([0], dtype=torch.int64, device=device)

    specs = [
        ExportSpec(
            name="sam31_text_prompt",
            model=text_model,
            inputs=(sample_image, sample_tokens),
            input_names=("image", "token_ids"),
            output_names=("pred_logits", "pred_boxes_xyxy", "pred_masks"),
        ),
        ExportSpec(
            name="sam31_point_prompt",
            model=point_model,
            inputs=(sample_image, sample_points, sample_point_labels, sample_mask),
            input_names=("image", "point_coords", "point_labels", "mask_input"),
            output_names=("low_res_masks", "high_res_masks", "ious", "object_score_logits"),
        ),
        ExportSpec(
            name="sam31_box_prompt",
            model=box_model,
            inputs=(sample_image, sample_box, sample_mask),
            input_names=("image", "box_xyxy", "mask_input"),
            output_names=("low_res_masks", "high_res_masks", "ious", "object_score_logits"),
        ),
        ExportSpec(
            name="sam31_video_step",
            model=video_model,
            inputs=(
                sample_image,
                sample_points,
                sample_point_labels,
                sample_box,
                sample_mask,
                prev_mem,
                prev_pos,
                prev_obj_ptr,
                prev_valid,
            ),
            input_names=(
                "image",
                "point_coords",
                "point_labels",
                "box_xyxy",
                "mask_input",
                "prev_maskmem_features",
                "prev_maskmem_pos_enc",
                "prev_obj_ptr",
                "prev_valid",
            ),
            output_names=(
                "low_res_masks",
                "high_res_masks",
                "obj_ptr",
                "object_score_logits",
                "maskmem_features",
                "maskmem_pos_enc",
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
    (output_dir / "export_meta.json").write_text(
        json.dumps(meta, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
