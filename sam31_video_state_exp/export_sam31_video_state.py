import argparse
import json
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sam31_onnx.export_sam31_onnx import (  # noqa: E402
    IMAGE_SIZE,
    _load_tracker_checkpoint,
    _patch_decoder_for_export,
    _patch_mask_downsampler_for_export,
    _patch_vitdet_rope_for_export,
    _preprocess_image,
)
from sam3.model.sam3_tracker_utils import get_1d_sine_pe  # noqa: E402
from sam3.model_builder import build_sam3_image_model, build_tracker  # noqa: E402
from sam3.sam.rope import apply_rotary_enc_real  # noqa: E402
from sam3.sam.transformer import RoPEAttention  # noqa: E402


def _patch_tracker_rope_for_state_export(model: nn.Module) -> None:
    def _forward(
        self: RoPEAttention,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        num_k_exclude_rope: int = 0,
    ) -> torch.Tensor:
        q = self.q_proj(q)
        k = self.k_proj(k)
        v = self.v_proj(v)
        q = self._separate_heads(q, self.num_heads)
        k = self._separate_heads(k, self.num_heads)
        v = self._separate_heads(v, self.num_heads)
        num_k_rope = k.size(-2) - num_k_exclude_rope
        q, k[:, :, :num_k_rope] = apply_rotary_enc_real(
            q,
            k[:, :, :num_k_rope],
            freqs_cis_real=self.freqs_cis_real,
            freqs_cis_imag=self.freqs_cis_imag,
            repeat_freqs_k=self.rope_k_repeat,
        )
        out = F.scaled_dot_product_attention(q, k, v, dropout_p=0.0)
        out = self._recombine_heads(out)
        return self.out_proj(out)

    RoPEAttention.forward = _forward
    for module in model.modules():
        if isinstance(module, RoPEAttention):
            module.use_rope_real = True
            module.freqs_cis_real = module.freqs_cis.real.float().to(next(module.parameters()).device)
            module.freqs_cis_imag = module.freqs_cis.imag.float().to(next(module.parameters()).device)
            module.use_fa3 = False


class RecurrentVideoStepWrapper(nn.Module):
    def __init__(self, image_model: nn.Module, tracker: nn.Module):
        super().__init__()
        self.image_model = image_model
        self.tracker = tracker
        self.register_buffer(
            "image_pe_buffer",
            tracker.sam_prompt_encoder.get_dense_pe().detach().clone(),
            persistent=False,
        )

    def _build_point_inputs(
        self,
        point_coords: torch.Tensor,
        point_labels: torch.Tensor,
        box_xyxy: torch.Tensor,
        box_valid_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size = point_coords.shape[0]
        point_labels = torch.where(
            point_labels >= 0,
            point_labels,
            torch.full_like(point_labels, -1),
        ).to(torch.int32)
        box_coords = box_xyxy.reshape(batch_size, -1, 2, 2)
        num_boxes = box_coords.shape[1]
        box_labels = torch.tensor([[2, 3]], dtype=torch.int32, device=box_xyxy.device).repeat(
            batch_size, num_boxes
        )
        box_labels = torch.where(
            box_valid_mask.bool().repeat_interleave(2, dim=1),
            box_labels,
            torch.full_like(box_labels, -1),
        )
        prompt_coords = torch.cat([box_coords.reshape(batch_size, -1, 2), point_coords], dim=1)
        prompt_labels = torch.cat([box_labels, point_labels], dim=1)
        return prompt_coords, prompt_labels

    def _prepare_memory_conditioned_features(
        self,
        current_vision_feats: list[torch.Tensor],
        current_vision_pos_embeds: list[torch.Tensor],
        feat_sizes: list[tuple[int, int]],
        prev_maskmem_features: torch.Tensor,
        prev_maskmem_pos_enc: torch.Tensor,
        prev_memory_valid: torch.Tensor,
        prev_memory_tpos: torch.Tensor,
        prev_obj_ptrs: torch.Tensor,
        prev_obj_ptr_valid: torch.Tensor,
        prev_obj_ptr_tpos: torch.Tensor,
    ) -> torch.Tensor:
        batch_size = current_vision_feats[-1].size(1)
        hidden_dim = self.tracker.hidden_dim
        mem_dim = self.tracker.mem_dim
        height, width = feat_sizes[-1]

        no_mem_needed = (~torch.any(prev_memory_valid)) & (~torch.any(prev_obj_ptr_valid))

        memory_tpos_index = (
            self.tracker.num_maskmem - prev_memory_tpos.to(torch.long) - 1
        ).clamp(0, self.tracker.num_maskmem - 1)
        memory_tpos_enc = self.tracker.maskmem_tpos_enc.index_select(
            0, memory_tpos_index
        ).squeeze(1).squeeze(1)
        memory_pos = prev_maskmem_pos_enc + memory_tpos_enc[:, None, :, None, None]

        mem_slots, _, _, mem_h, mem_w = prev_maskmem_features.shape
        memory_valid = prev_memory_valid.to(prev_maskmem_features.dtype).view(mem_slots, 1, 1, 1, 1)
        memory_seq = (
            prev_maskmem_features * memory_valid
        ).permute(0, 3, 4, 1, 2).reshape(mem_slots * mem_h * mem_w, batch_size, mem_dim)
        memory_pos_seq = (
            memory_pos * memory_valid
        ).permute(0, 3, 4, 1, 2).reshape(mem_slots * mem_h * mem_w, batch_size, mem_dim)

        max_abs_pos = max(1, self.tracker.max_obj_ptrs_in_encoder)
        ptr_rel_pos = prev_obj_ptr_tpos.to(torch.float32) / max(1, max_abs_pos - 1)
        ptr_pos = get_1d_sine_pe(ptr_rel_pos, dim=hidden_dim)
        ptr_pos = self.tracker.obj_ptr_tpos_proj(ptr_pos).unsqueeze(1).expand(-1, batch_size, -1)
        ptr_seq = prev_obj_ptrs * prev_obj_ptr_valid.to(prev_obj_ptrs.dtype)[:, None, None]
        ptr_pos = ptr_pos * prev_obj_ptr_valid.to(ptr_pos.dtype)[:, None, None]

        if mem_dim < hidden_dim:
            split_factor = hidden_dim // mem_dim
            ptr_seq = ptr_seq.reshape(-1, batch_size, split_factor, mem_dim)
            ptr_seq = ptr_seq.permute(0, 2, 1, 3).reshape(-1, batch_size, mem_dim)
            ptr_pos = ptr_pos.repeat_interleave(split_factor, dim=0)
            ptr_valid = prev_obj_ptr_valid.repeat_interleave(split_factor)
        else:
            ptr_valid = prev_obj_ptr_valid

        prompt = torch.cat([memory_seq, ptr_seq], dim=0)
        prompt_pos = torch.cat([memory_pos_seq, ptr_pos], dim=0)
        encoder_out = self.tracker.transformer.encoder(
            src=current_vision_feats[-1:],
            src_key_padding_mask=[None],
            src_pos=current_vision_pos_embeds[-1:],
            prompt=prompt,
            prompt_pos=prompt_pos,
            feat_sizes=feat_sizes[-1:],
            num_obj_ptr_tokens=ptr_seq.shape[0],
        )
        pix_feat_with_mem = encoder_out["memory"].permute(1, 2, 0).view(
            batch_size, hidden_dim, height, width
        )
        pix_feat_no_mem = (
            current_vision_feats[-1] + self.tracker.no_mem_embed
        ).permute(1, 2, 0).view(batch_size, hidden_dim, height, width)
        return torch.where(no_mem_needed.view(1, 1, 1, 1), pix_feat_no_mem, pix_feat_with_mem)

    def forward(
        self,
        image: torch.Tensor,
        point_coords: torch.Tensor,
        point_labels: torch.Tensor,
        box_xyxy: torch.Tensor,
        box_valid_mask: torch.Tensor,
        mask_input: torch.Tensor,
        prev_maskmem_features: torch.Tensor,
        prev_maskmem_pos_enc: torch.Tensor,
        prev_memory_valid: torch.Tensor,
        prev_memory_tpos: torch.Tensor,
        prev_obj_ptrs: torch.Tensor,
        prev_obj_ptr_valid: torch.Tensor,
        prev_obj_ptr_tpos: torch.Tensor,
    ):
        backbone_out = self.image_model.backbone.forward_image(image)
        fpn = backbone_out["backbone_fpn"]
        feat_sizes = [(x.shape[-2], x.shape[-1]) for x in fpn]
        current_vision_feats = [x.flatten(2).permute(2, 0, 1) for x in fpn]
        current_vision_pos = [x.flatten(2).permute(2, 0, 1) for x in backbone_out["vision_pos_enc"]]

        pix_feat_with_mem = self._prepare_memory_conditioned_features(
            current_vision_feats=current_vision_feats,
            current_vision_pos_embeds=current_vision_pos,
            feat_sizes=feat_sizes,
            prev_maskmem_features=prev_maskmem_features,
            prev_maskmem_pos_enc=prev_maskmem_pos_enc,
            prev_memory_valid=prev_memory_valid,
            prev_memory_tpos=prev_memory_tpos,
            prev_obj_ptrs=prev_obj_ptrs,
            prev_obj_ptr_valid=prev_obj_ptr_valid,
            prev_obj_ptr_tpos=prev_obj_ptr_tpos,
        )

        high_res = [
            self.tracker.sam_mask_decoder.conv_s0(fpn[0]),
            self.tracker.sam_mask_decoder.conv_s1(fpn[1]),
        ]
        prompt_coords, prompt_labels = self._build_point_inputs(
            point_coords=point_coords,
            point_labels=point_labels,
            box_xyxy=box_xyxy,
            box_valid_mask=box_valid_mask,
        )
        sam_mask_input = mask_input
        if sam_mask_input.shape[-2:] != self.tracker.sam_prompt_encoder.mask_input_size:
            sam_mask_input = F.interpolate(
                sam_mask_input.float(),
                size=self.tracker.sam_prompt_encoder.mask_input_size,
                mode="bilinear",
                align_corners=False,
            )
        sparse_embeddings, dense_embeddings = self.tracker.sam_prompt_encoder(
            points=(prompt_coords, prompt_labels),
            boxes=None,
            masks=sam_mask_input,
        )
        low_res_multimasks, _, sam_output_tokens, object_score_logits = self.tracker.sam_mask_decoder(
            image_embeddings=pix_feat_with_mem,
            image_pe=self.image_pe_buffer.to(pix_feat_with_mem.device),
            sparse_prompt_embeddings=sparse_embeddings,
            dense_prompt_embeddings=dense_embeddings,
            multimask_output=False,
            repeat_image=False,
            high_res_features=high_res,
        )
        is_obj_appearing = object_score_logits > 0
        low_res_multimasks = torch.where(
            is_obj_appearing[:, None, None],
            low_res_multimasks,
            low_res_multimasks.new_full((), -1024.0),
        ).float()
        high_res_masks = F.interpolate(
            low_res_multimasks,
            size=(IMAGE_SIZE, IMAGE_SIZE),
            mode="bilinear",
            align_corners=False,
        )
        low_res_masks = low_res_multimasks[:, 0:1]
        high_res_masks = high_res_masks[:, 0:1]

        obj_ptr = self.tracker.obj_ptr_proj(sam_output_tokens[:, 0])
        obj_ptr = is_obj_appearing.float() * obj_ptr + (1 - is_obj_appearing.float()) * self.tracker.no_obj_ptr

        new_maskmem_features, new_maskmem_pos_enc = self.tracker._encode_new_memory(
            image=image,
            current_vision_feats=current_vision_feats,
            feat_sizes=feat_sizes,
            pred_masks_high_res=high_res_masks,
            object_score_logits=object_score_logits,
            is_mask_from_pts=False,
        )
        return (
            low_res_masks,
            high_res_masks,
            object_score_logits,
            new_maskmem_features.unsqueeze(0),
            new_maskmem_pos_enc[-1].unsqueeze(0),
            obj_ptr.unsqueeze(0),
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint",
        default=r"C:\Users\lsn\.cache\modelscope\hub\models\facebook\sam3___1\sam3.1_multiplex.pt",
    )
    parser.add_argument("--output-dir", default="output/sam31_video_state_exp")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--sample-image", default="assets/images/truck.jpg")
    parser.add_argument("--opset", type=int, default=17)
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
    _patch_tracker_rope_for_state_export(tracker)

    model = RecurrentVideoStepWrapper(image_model, tracker).to(device).eval()

    sample_image = _preprocess_image(args.sample_image, device)
    sample_points = torch.tensor([[[320.0, 420.0]]], dtype=torch.float32, device=device)
    sample_point_labels = torch.tensor([[1]], dtype=torch.int64, device=device)
    sample_box = torch.tensor([[[220.0, 180.0, 860.0, 820.0]]], dtype=torch.float32, device=device)
    sample_box_valid_mask = torch.tensor([[True]], dtype=torch.bool, device=device)
    sample_mask = torch.zeros((1, 1, 288, 288), dtype=torch.float32, device=device)

    mem_h = mem_w = 72
    prev_maskmem_features = torch.zeros(
        (tracker.num_maskmem, 1, tracker.mem_dim, mem_h, mem_w), dtype=torch.float32, device=device
    )
    prev_maskmem_pos_enc = torch.zeros_like(prev_maskmem_features)
    prev_memory_valid = torch.zeros((tracker.num_maskmem,), dtype=torch.bool, device=device)
    prev_memory_tpos = torch.arange(1, tracker.num_maskmem + 1, dtype=torch.long, device=device)
    prev_obj_ptrs = torch.zeros(
        (tracker.max_obj_ptrs_in_encoder, 1, tracker.hidden_dim), dtype=torch.float32, device=device
    )
    prev_obj_ptr_valid = torch.zeros((tracker.max_obj_ptrs_in_encoder,), dtype=torch.bool, device=device)
    prev_obj_ptr_tpos = torch.arange(
        1, tracker.max_obj_ptrs_in_encoder + 1, dtype=torch.long, device=device
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    model_path = output_dir / "sam31_video_tracking_state_step.onnx"

    torch.onnx.export(
        model,
        (
            sample_image,
            sample_points,
            sample_point_labels,
            sample_box,
            sample_box_valid_mask,
            sample_mask,
            prev_maskmem_features,
            prev_maskmem_pos_enc,
            prev_memory_valid,
            prev_memory_tpos,
            prev_obj_ptrs,
            prev_obj_ptr_valid,
            prev_obj_ptr_tpos,
        ),
        str(model_path),
        input_names=[
            "image",
            "point_coords",
            "point_labels",
            "box_xyxy",
            "box_valid_mask",
            "mask_input",
            "prev_maskmem_features",
            "prev_maskmem_pos_enc",
            "prev_memory_valid",
            "prev_memory_tpos",
            "prev_obj_ptrs",
            "prev_obj_ptr_valid",
            "prev_obj_ptr_tpos",
        ],
        output_names=[
            "tracking_low_res_masks",
            "tracking_high_res_masks",
            "tracking_object_score_logits",
            "new_maskmem_features",
            "new_maskmem_pos_enc",
            "new_obj_ptrs",
        ],
        opset_version=args.opset,
        do_constant_folding=False,
        external_data=True,
    )

    meta = {
        "checkpoint": args.checkpoint,
        "model": str(model_path),
        "image_size": IMAGE_SIZE,
        "num_maskmem": tracker.num_maskmem,
        "mem_dim": tracker.mem_dim,
        "mem_hw": [mem_h, mem_w],
        "max_obj_ptrs": tracker.max_obj_ptrs_in_encoder,
        "hidden_dim": tracker.hidden_dim,
    }
    (output_dir / "export_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
