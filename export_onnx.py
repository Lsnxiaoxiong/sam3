#!/usr/bin/env python3

from __future__ import annotations

import argparse
import pathlib
from dataclasses import dataclass
from importlib import resources

import numpy as np
import torch
import torch.nn.functional as F
from torchvision.transforms import v2
from torchvision.ops import roi_align

from sam3.model import vitdet
from sam3.model.box_ops import box_cxcywh_to_xyxy
from sam3.model.data_misc import FindStage
from sam3.model.geometry_encoders import Prompt
from sam3.model.memory import SimpleMaskDownSampler
from sam3.model.sam3_image import Sam3Image
from sam3.model.sam3_image_processor import Sam3Processor
from sam3.model.sam3_tracker_base import Sam3TrackerBase
from sam3.model.sam3_tracker_utils import get_1d_sine_pe
from sam3.model.tokenizer_ve import SimpleTokenizer
from sam3.model_builder import build_sam3_image_model, build_sam3_video_model
from sam3.sam.rope import apply_rotary_enc_real
from sam3.sam.transformer import RoPEAttention


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


def patch_tracker_rope_for_export() -> None:
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

        dropout_p = self.dropout_p if self.training else 0.0
        out = F.scaled_dot_product_attention(q, k, v, dropout_p=dropout_p)
        out = self._recombine_heads(out)
        out = self.out_proj(out)
        return out

    RoPEAttention.forward = _forward


def patch_tracker_resizes_for_export() -> None:
    def _mask_downsampler_forward(
        self: SimpleMaskDownSampler, x: torch.Tensor
    ) -> torch.Tensor:
        if self.interpol_size is not None and self.interpol_size != list(x.shape[-2:]):
            x = F.interpolate(
                x.float(),
                size=self.interpol_size,
                align_corners=False,
                mode="bilinear",
                antialias=False,
            )
        return self.encoder(x)

    def _tracker_forward_sam_heads(
        self: Sam3TrackerBase,
        backbone_features: torch.Tensor,
        point_inputs: dict[str, torch.Tensor] | None = None,
        mask_inputs: torch.Tensor | None = None,
        high_res_features: list[torch.Tensor] | None = None,
        multimask_output: bool = False,
        gt_masks: torch.Tensor | None = None,
    ):
        bsz = backbone_features.size(0)
        device = backbone_features.device
        if point_inputs is not None:
            sam_point_coords = point_inputs["point_coords"]
            sam_point_labels = point_inputs["point_labels"]
        else:
            sam_point_coords = torch.zeros(bsz, 1, 2, device=device)
            sam_point_labels = -torch.ones(bsz, 1, dtype=torch.int32, device=device)

        if mask_inputs is not None:
            if mask_inputs.shape[-2:] != self.sam_prompt_encoder.mask_input_size:
                sam_mask_prompt = F.interpolate(
                    mask_inputs.float(),
                    size=self.sam_prompt_encoder.mask_input_size,
                    align_corners=False,
                    mode="bilinear",
                    antialias=False,
                )
            else:
                sam_mask_prompt = mask_inputs
        else:
            sam_mask_prompt = None

        sparse_embeddings, dense_embeddings = self.sam_prompt_encoder(
            points=(sam_point_coords, sam_point_labels),
            boxes=None,
            masks=sam_mask_prompt,
        )
        sparse_embeddings = self._maybe_clone(sparse_embeddings)
        dense_embeddings = self._maybe_clone(dense_embeddings)
        image_pe = self._maybe_clone(self.sam_prompt_encoder.get_dense_pe())
        (
            low_res_multimasks,
            ious,
            sam_output_tokens,
            object_score_logits,
        ) = self.sam_mask_decoder(
            image_embeddings=backbone_features,
            image_pe=image_pe,
            sparse_prompt_embeddings=sparse_embeddings,
            dense_prompt_embeddings=dense_embeddings,
            multimask_output=multimask_output,
            repeat_image=False,
            high_res_features=high_res_features,
        )
        low_res_multimasks = self._maybe_clone(low_res_multimasks)
        ious = self._maybe_clone(ious)
        sam_output_tokens = self._maybe_clone(sam_output_tokens)
        object_score_logits = self._maybe_clone(object_score_logits)

        if self.training and self.teacher_force_obj_scores_for_mem:
            is_obj_appearing = torch.any(gt_masks.float().flatten(1) > 0, dim=1)[..., None]
        else:
            is_obj_appearing = object_score_logits > 0

        low_res_multimasks = torch.where(
            is_obj_appearing[:, None, None],
            low_res_multimasks,
            low_res_multimasks.new_full((), -1024.0),
        )
        low_res_multimasks = low_res_multimasks.float()
        high_res_multimasks = F.interpolate(
            low_res_multimasks,
            size=(self.image_size, self.image_size),
            mode="bilinear",
            align_corners=False,
        )

        sam_output_token = sam_output_tokens[:, 0]
        if multimask_output:
            best_iou_inds = torch.argmax(ious, dim=-1)
            batch_inds = torch.arange(bsz, device=device)
            low_res_masks = low_res_multimasks[batch_inds, best_iou_inds].unsqueeze(1)
            high_res_masks = high_res_multimasks[batch_inds, best_iou_inds].unsqueeze(1)
            if sam_output_tokens.size(1) > 1:
                sam_output_token = sam_output_tokens[batch_inds, best_iou_inds]
        else:
            low_res_masks, high_res_masks = low_res_multimasks, high_res_multimasks

        obj_ptr = self.obj_ptr_proj(sam_output_token)
        lambda_is_obj_appearing = is_obj_appearing.float()
        obj_ptr = lambda_is_obj_appearing * obj_ptr
        obj_ptr = obj_ptr + (1 - lambda_is_obj_appearing) * self.no_obj_ptr

        return (
            low_res_multimasks,
            high_res_multimasks,
            ious,
            low_res_masks,
            high_res_masks,
            obj_ptr,
            object_score_logits,
        )

    def _tracker_use_mask_as_output(
        self: Sam3TrackerBase,
        backbone_features: torch.Tensor,
        high_res_features: list[torch.Tensor],
        mask_inputs: torch.Tensor,
    ):
        out_scale, out_bias = 20.0, -10.0
        mask_inputs_float = mask_inputs.float()
        high_res_masks = mask_inputs_float * out_scale + out_bias
        low_res_masks = F.interpolate(
            high_res_masks,
            size=(
                high_res_masks.size(-2) // self.backbone_stride * 4,
                high_res_masks.size(-1) // self.backbone_stride * 4,
            ),
            align_corners=False,
            mode="bilinear",
            antialias=False,
        )
        ious = mask_inputs.new_ones(mask_inputs.size(0), 1).float()
        _, _, _, _, _, obj_ptr, _ = self._forward_sam_heads(
            backbone_features=backbone_features,
            mask_inputs=self.mask_downsample(mask_inputs_float),
            high_res_features=high_res_features,
            gt_masks=mask_inputs,
        )
        is_obj_appearing = torch.any(mask_inputs.flatten(1).float() > 0.0, dim=1)[..., None]
        lambda_is_obj_appearing = is_obj_appearing.float()
        object_score_logits = out_scale * lambda_is_obj_appearing + out_bias
        obj_ptr = lambda_is_obj_appearing * obj_ptr
        obj_ptr = obj_ptr + (1 - lambda_is_obj_appearing) * self.no_obj_ptr

        return (
            low_res_masks,
            high_res_masks,
            ious,
            low_res_masks,
            high_res_masks,
            obj_ptr,
            object_score_logits,
        )

    SimpleMaskDownSampler.forward = _mask_downsampler_forward
    Sam3TrackerBase._forward_sam_heads = _tracker_forward_sam_heads
    Sam3TrackerBase._use_mask_as_output = _tracker_use_mask_as_output


def prepare_vitdet_rope_buffers(model: torch.nn.Module) -> None:
    for module in model.modules():
        if isinstance(module, vitdet.Attention) and getattr(module, "use_rope", False):
            module.register_buffer("freqs_cis_real", module.freqs_cis.real.float())
            module.register_buffer("freqs_cis_imag", module.freqs_cis.imag.float())


def prepare_tracker_rope_buffers(model: torch.nn.Module) -> None:
    for module in model.modules():
        if isinstance(module, RoPEAttention):
            module.use_rope_real = True
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
    reference_feature_encoder: pathlib.Path
    grounding_decoder: pathlib.Path
    grounding_decoder_with_reference: pathlib.Path
    video_tracking_step: pathlib.Path
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


class ReferenceFeatureEncoderWrapper(torch.nn.Module):
    def forward(
        self,
        backbone_fpn_0: torch.Tensor,
        reference_boxes_xyxy: torch.Tensor,
    ) -> torch.Tensor:
        if reference_boxes_xyxy.shape[0] == 0:
            return torch.zeros(
                (0, backbone_fpn_0.shape[1]),
                dtype=backbone_fpn_0.dtype,
                device=backbone_fpn_0.device,
            )

        batch_indices = torch.zeros(
            (reference_boxes_xyxy.shape[0], 1),
            dtype=reference_boxes_xyxy.dtype,
            device=reference_boxes_xyxy.device,
        )
        roi_boxes = torch.cat([batch_indices, reference_boxes_xyxy], dim=1)
        pooled = roi_align(
            backbone_fpn_0,
            roi_boxes,
            output_size=(7, 7),
            spatial_scale=backbone_fpn_0.shape[-1] / 1008.0,
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
        box_coords: torch.Tensor,
        box_valid_mask: torch.Tensor,
        box_labels: torch.Tensor,
        reference_features: torch.Tensor,
        reference_valid_mask: torch.Tensor,
        reference_weight: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        seq_len, batch_size, feat_dim = language_features.shape
        if reference_features.shape[0] == 0:
            reference_bias = torch.zeros(
                (batch_size, feat_dim),
                dtype=language_features.dtype,
                device=language_features.device,
            )
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
            box_coords,
            box_valid_mask,
            box_labels,
        )


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


class VideoTrackingStepWrapper(torch.nn.Module):
    def __init__(self, tracker: torch.nn.Module) -> None:
        super().__init__()
        self.tracker = tracker
        self.register_buffer(
            "image_pe_buffer",
            tracker.sam_prompt_encoder.get_dense_pe().detach().clone(),
        )

    def _build_point_inputs(
        self,
        point_coords: torch.Tensor,
        point_labels: torch.Tensor,
        box_xyxy: torch.Tensor,
        box_valid_mask: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        batch_size = point_coords.shape[0]
        valid_point_labels = torch.where(
            point_labels >= 0,
            point_labels,
            torch.full_like(point_labels, -1),
        ).to(torch.int32)
        concat_coords = point_coords
        concat_labels = valid_point_labels

        box_coords = box_xyxy.reshape(batch_size, -1, 2, 2)
        box_labels = torch.tensor([[2, 3]], dtype=torch.int32, device=box_xyxy.device).repeat(
            batch_size, box_xyxy.shape[1]
        )
        flat_box_labels = torch.where(
            box_valid_mask.bool().repeat_interleave(2, dim=1),
            box_labels,
            torch.full_like(box_labels, -1),
        )
        concat_coords = torch.cat([box_coords.reshape(batch_size, -1, 2), concat_coords], dim=1)
        concat_labels = torch.cat([flat_box_labels, concat_labels], dim=1)

        return {
            "point_coords": concat_coords,
            "point_labels": concat_labels,
        }

    def _prepare_explicit_memory_conditioned_features(
        self,
        current_vision_feats: list[torch.Tensor],
        current_vision_pos_embeds: list[torch.Tensor],
        feat_sizes: list[tuple[int, int]],
        prev_maskmem_features: torch.Tensor,
        prev_maskmem_pos_enc: torch.Tensor,
        prev_memory_valid: torch.Tensor,
        prev_memory_is_cond: torch.Tensor,
        prev_memory_tpos: torch.Tensor,
        prev_obj_ptrs: torch.Tensor,
        prev_obj_ptr_valid: torch.Tensor,
        prev_obj_ptr_is_cond: torch.Tensor,
        prev_obj_ptr_tpos: torch.Tensor,
    ) -> tuple[torch.Tensor, int]:
        batch_size = current_vision_feats[-1].size(1)
        hidden_dim = self.tracker.hidden_dim
        mem_dim = self.tracker.mem_dim
        height, width = feat_sizes[-1]
        device = current_vision_feats[-1].device

        no_mem_needed = (~torch.any(prev_memory_valid)) & (~torch.any(prev_obj_ptr_valid))

        memory_tpos_index = torch.where(
            prev_memory_is_cond,
            torch.zeros_like(prev_memory_tpos),
            prev_memory_tpos,
        )
        memory_tpos_index = (
            self.tracker.num_maskmem - memory_tpos_index.to(torch.long) - 1
        ).clamp(0, self.tracker.num_maskmem - 1)
        memory_tpos_enc = self.tracker.maskmem_tpos_enc.index_select(
            0, memory_tpos_index
        ).squeeze(1).squeeze(1)

        memory_pos = prev_maskmem_pos_enc + memory_tpos_enc[:, None, :, None, None]
        if getattr(self.tracker, "cond_frame_spatial_embedding", None) is not None:
            memory_pos = memory_pos + (
                prev_memory_is_cond.to(memory_pos.dtype)[:, None, None, None, None]
                * self.tracker.cond_frame_spatial_embedding.view(1, 1, mem_dim, 1, 1)
            )

        mem_slots, _, _, mem_h, mem_w = prev_maskmem_features.shape
        memory_seq = prev_maskmem_features.permute(0, 3, 4, 1, 2).reshape(
            mem_slots * mem_h * mem_w, batch_size, mem_dim
        )
        memory_pos_seq = memory_pos.permute(0, 3, 4, 1, 2).reshape(
            mem_slots * mem_h * mem_w, batch_size, mem_dim
        )
        memory_valid_seq = prev_memory_valid.to(memory_seq.dtype).view(mem_slots, 1, 1, 1, 1)
        memory_seq = (
            prev_maskmem_features * memory_valid_seq
        ).permute(0, 3, 4, 1, 2).reshape(mem_slots * mem_h * mem_w, batch_size, mem_dim)
        memory_pos_seq = (
            memory_pos * memory_valid_seq
        ).permute(0, 3, 4, 1, 2).reshape(mem_slots * mem_h * mem_w, batch_size, mem_dim)

        max_abs_pos = max(1, self.tracker.max_obj_ptrs_in_encoder)
        ptr_rel_pos = prev_obj_ptr_tpos.to(torch.float32) / max(1, max_abs_pos - 1)
        ptr_pos = get_1d_sine_pe(ptr_rel_pos, dim=hidden_dim)
        ptr_pos = self.tracker.obj_ptr_tpos_proj(ptr_pos).unsqueeze(1).expand(
            -1, batch_size, -1
        )

        ptr_seq = prev_obj_ptrs * prev_obj_ptr_valid.to(prev_obj_ptrs.dtype)[:, None, None]
        if getattr(self.tracker, "cond_frame_obj_ptr_embedding", None) is not None:
            ptr_seq = ptr_seq + (
                prev_obj_ptr_is_cond.to(ptr_seq.dtype)[:, None, None]
                * self.tracker.cond_frame_obj_ptr_embedding
            )
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
        pix_feat_with_mem = torch.where(no_mem_needed, pix_feat_no_mem, pix_feat_with_mem)
        return pix_feat_with_mem, ptr_seq.shape[0]

    def forward(
        self,
        sam2_vision_pos_enc_0: torch.Tensor,
        sam2_vision_pos_enc_1: torch.Tensor,
        sam2_vision_pos_enc_2: torch.Tensor,
        sam2_backbone_fpn_0: torch.Tensor,
        sam2_backbone_fpn_1: torch.Tensor,
        sam2_backbone_fpn_2: torch.Tensor,
        point_coords: torch.Tensor,
        point_labels: torch.Tensor,
        box_xyxy: torch.Tensor,
        box_valid_mask: torch.Tensor,
        mask_input: torch.Tensor,
        prev_maskmem_features: torch.Tensor,
        prev_maskmem_pos_enc: torch.Tensor,
        prev_memory_valid: torch.Tensor,
        prev_memory_is_cond: torch.Tensor,
        prev_memory_tpos: torch.Tensor,
        prev_obj_ptrs: torch.Tensor,
        prev_obj_ptr_valid: torch.Tensor,
        prev_obj_ptr_is_cond: torch.Tensor,
        prev_obj_ptr_tpos: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        backbone_out = {
            "backbone_fpn": [
                sam2_backbone_fpn_0,
                sam2_backbone_fpn_1,
                sam2_backbone_fpn_2,
            ],
            "vision_pos_enc": [
                sam2_vision_pos_enc_0,
                sam2_vision_pos_enc_1,
                sam2_vision_pos_enc_2,
            ],
        }
        (
            _,
            current_vision_feats,
            current_vision_pos_embeds,
            feat_sizes,
        ) = self.tracker._prepare_backbone_features(backbone_out)

        high_res_features = [
            x.permute(1, 2, 0).view(x.size(1), x.size(2), *s)
            for x, s in zip(current_vision_feats[:-1], feat_sizes[:-1])
        ]

        pix_feat_with_mem, _ = self._prepare_explicit_memory_conditioned_features(
            current_vision_feats=current_vision_feats,
            current_vision_pos_embeds=current_vision_pos_embeds,
            feat_sizes=feat_sizes,
            prev_maskmem_features=prev_maskmem_features,
            prev_maskmem_pos_enc=prev_maskmem_pos_enc,
            prev_memory_valid=prev_memory_valid,
            prev_memory_is_cond=prev_memory_is_cond,
            prev_memory_tpos=prev_memory_tpos,
            prev_obj_ptrs=prev_obj_ptrs,
            prev_obj_ptr_valid=prev_obj_ptr_valid,
            prev_obj_ptr_is_cond=prev_obj_ptr_is_cond,
            prev_obj_ptr_tpos=prev_obj_ptr_tpos,
        )

        point_inputs = self._build_point_inputs(
            point_coords=point_coords,
            point_labels=point_labels,
            box_xyxy=box_xyxy,
            box_valid_mask=box_valid_mask,
        )
        sam_mask_input = mask_input.to(pix_feat_with_mem.device)
        image_pe = self.image_pe_buffer.to(pix_feat_with_mem.device)

        sam_point_coords = point_inputs["point_coords"]
        sam_point_labels = point_inputs["point_labels"]
        if sam_mask_input.shape[-2:] != self.tracker.sam_prompt_encoder.mask_input_size:
            sam_mask_input = F.interpolate(
                sam_mask_input.float(),
                size=self.tracker.sam_prompt_encoder.mask_input_size,
                align_corners=False,
                mode="bilinear",
                antialias=False,
            )

        sparse_embeddings, dense_embeddings = self.tracker.sam_prompt_encoder(
            points=(sam_point_coords, sam_point_labels),
            boxes=None,
            masks=sam_mask_input,
        )
        decoder = self.tracker.sam_mask_decoder
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
        src = pix_feat_with_mem + dense_embeddings
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
        low_res_multimasks = (hyper_in @ upscaled_embedding.view(b, c, h * w)).view(
            b, -1, h, w
        )
        ious = decoder.iou_prediction_head(iou_token_out)
        if decoder.pred_obj_scores:
            object_score_logits = decoder.pred_obj_score_head(hs[:, 0, :])
        else:
            object_score_logits = 10.0 * ious.new_ones(ious.shape[0], 1)
        sam_output_tokens = mask_tokens_out[:, 0:1]
        is_obj_appearing = object_score_logits > 0
        low_res_multimasks = torch.where(
            is_obj_appearing[:, None, None],
            low_res_multimasks,
            low_res_multimasks.new_full((), -1024.0),
        )
        low_res_multimasks = low_res_multimasks.float()
        high_res_multimasks = F.interpolate(
            low_res_multimasks,
            size=(self.tracker.image_size, self.tracker.image_size),
            mode="bilinear",
            align_corners=False,
        )
        low_res_masks = low_res_multimasks[:, 0:1]
        high_res_masks = high_res_multimasks[:, 0:1]
        sam_output_token = sam_output_tokens[:, 0]
        obj_ptr = self.tracker.obj_ptr_proj(sam_output_token)
        lambda_is_obj_appearing = is_obj_appearing.float()
        obj_ptr = lambda_is_obj_appearing * obj_ptr + (
            1 - lambda_is_obj_appearing
        ) * self.tracker.no_obj_ptr

        new_maskmem_features, new_maskmem_pos_enc = self.tracker._encode_new_memory(
            image=None,
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
            new_maskmem_features,
            new_maskmem_pos_enc[-1],
            obj_ptr,
        )


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
        choices=[
            "all",
            "image_encoder",
            "text_encoder",
            "reference_feature_encoder",
            "grounding_decoder",
            "grounding_decoder_with_reference",
            "video_tracking_step",
            "interactive_decoder",
        ],
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
        "reference_boxes_xyxy": {0: "num_reference_boxes"},
        "reference_features": {0: "num_reference_boxes", 1: "batch"},
        "reference_valid_mask": {0: "num_reference_boxes", 1: "batch"},
        "reference_embedding": {0: "num_reference_boxes"},
        "prev_maskmem_features": {1: "batch"},
        "prev_maskmem_pos_enc": {1: "batch"},
        "prev_obj_ptrs": {1: "batch"},
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
        "tracking_low_res_masks": {0: "batch"},
        "tracking_high_res_masks": {0: "batch"},
        "tracking_object_score_logits": {0: "batch"},
        "new_maskmem_features": {0: "batch"},
        "new_maskmem_pos_enc": {0: "batch"},
        "new_obj_ptr": {0: "batch"},
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
        reference_feature_encoder=output_dir / "sam3_reference_feature_encoder.onnx",
        grounding_decoder=output_dir / "sam3_grounding_decoder.onnx",
        grounding_decoder_with_reference=output_dir / "sam3_grounding_decoder_with_reference.onnx",
        video_tracking_step=output_dir / "sam3_video_tracking_step.onnx",
        interactive_decoder=output_dir / "sam3_interactive_decoder.onnx",
        interactive_dense_pe=output_dir / "sam3_interactive_dense_pe.npy",
    )


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    requested = set(args.modules)
    if ("all" in requested or "video_tracking_step" in requested) and device.type != "cpu":
        raise ValueError(
            "video_tracking_step export currently requires --device cpu; "
            "CUDA export produces invalid bfloat16 ONNX for ONNX Runtime"
        )

    patch_vitdet_rope_for_export()
    patch_tracker_rope_for_export()
    patch_tracker_resizes_for_export()
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
    reference_feature_encoder = ReferenceFeatureEncoderWrapper().to(device).eval()
    grounding_decoder = GroundingDecoderWrapper(model).to(device).eval()
    grounding_decoder_with_reference = GroundingDecoderWithReferenceWrapper(model).to(
        device
    ).eval()
    interactive_decoder = InteractiveDecoderWrapper(model).to(device).eval()
    video_tracking_step = None
    video_image_outputs: tuple[torch.Tensor, ...] | None = None
    if "all" in requested or "video_tracking_step" in requested:
        video_model = build_sam3_video_model(
            checkpoint_path=str(args.checkpoint),
            load_from_HF=False,
            device=str(device),
        )
        video_model = video_model.float().eval()
        prepare_tracker_rope_buffers(video_model)
        video_tracking_step = VideoTrackingStepWrapper(video_model.tracker).to(device).eval()

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
    sample_reference_boxes_xyxy = torch.tensor(
        [[100.0, 100.0, 300.0, 300.0]], device=device, dtype=torch.float32
    )
    sample_reference_features = torch.zeros((1, 1, 256), device=device, dtype=torch.float32)
    sample_reference_valid_mask = torch.ones((1, 1), device=device, dtype=torch.bool)
    sample_reference_weight = torch.tensor([1.0], device=device, dtype=torch.float32)
    sample_prev_maskmem_features = torch.zeros((7, 1, 64, 72, 72), device=device, dtype=torch.float32)
    sample_prev_maskmem_pos_enc = torch.zeros((7, 1, 64, 72, 72), device=device, dtype=torch.float32)
    sample_prev_memory_valid = torch.zeros((7,), device=device, dtype=torch.bool)
    sample_prev_memory_is_cond = torch.zeros((7,), device=device, dtype=torch.bool)
    sample_prev_memory_tpos = torch.arange(1, 8, device=device, dtype=torch.long).clamp_max(6)
    sample_prev_obj_ptrs = torch.zeros((16, 1, 256), device=device, dtype=torch.float32)
    sample_prev_obj_ptr_valid = torch.zeros((16,), device=device, dtype=torch.bool)
    sample_prev_obj_ptr_is_cond = torch.zeros((16,), device=device, dtype=torch.bool)
    sample_prev_obj_ptr_tpos = torch.arange(1, 17, device=device, dtype=torch.long).clamp_max(15)
    sample_interactive_image_pe = (
        interactive_decoder.predictor.model.sam_prompt_encoder.get_dense_pe().to(device)
    )

    image_outputs: tuple[torch.Tensor, ...] | None = None
    language_mask: torch.Tensor | None = None
    language_features: torch.Tensor | None = None

    need_image_outputs = bool(
        {
            "all",
            "reference_feature_encoder",
            "grounding_decoder",
            "grounding_decoder_with_reference",
            "video_tracking_step",
            "interactive_decoder",
        }
        & requested
    )
    need_language_outputs = bool(
        {"all", "grounding_decoder", "grounding_decoder_with_reference"} & requested
    )
    with torch.no_grad():
        if need_image_outputs:
            image_outputs = image_encoder(sample_image)
        if "all" in requested or "video_tracking_step" in requested:
            assert image_outputs is not None
            video_image_outputs = image_outputs
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
    if "all" in requested or "reference_feature_encoder" in requested:
        assert image_outputs is not None
        _export(
            reference_feature_encoder,
            (
                image_outputs[3],
                sample_reference_boxes_xyxy,
            ),
            artifacts.reference_feature_encoder,
            ["backbone_fpn_0", "reference_boxes_xyxy"],
            ["reference_embedding"],
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
    if "all" in requested or "grounding_decoder_with_reference" in requested:
        assert image_outputs is not None
        assert language_mask is not None
        assert language_features is not None
        _export(
            grounding_decoder_with_reference,
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
                sample_reference_features,
                sample_reference_valid_mask,
                sample_reference_weight,
            ),
            artifacts.grounding_decoder_with_reference,
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
                "reference_features",
                "reference_valid_mask",
                "reference_weight",
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
    if "all" in requested or "video_tracking_step" in requested:
        assert video_image_outputs is not None
        assert video_tracking_step is not None
        _export(
            video_tracking_step,
            (
                video_image_outputs[6],
                video_image_outputs[7],
                video_image_outputs[8],
                video_image_outputs[9],
                video_image_outputs[10],
                video_image_outputs[11],
                sample_point_coords.to(video_tracking_step.image_pe_buffer.device),
                sample_point_labels.to(video_tracking_step.image_pe_buffer.device),
                sample_box_xyxy.to(video_tracking_step.image_pe_buffer.device),
                sample_box_xyxy_valid_mask.to(video_tracking_step.image_pe_buffer.device),
                sample_mask_input.to(video_tracking_step.image_pe_buffer.device),
                sample_prev_maskmem_features.to(video_tracking_step.image_pe_buffer.device),
                sample_prev_maskmem_pos_enc.to(video_tracking_step.image_pe_buffer.device),
                sample_prev_memory_valid.to(video_tracking_step.image_pe_buffer.device),
                sample_prev_memory_is_cond.to(video_tracking_step.image_pe_buffer.device),
                sample_prev_memory_tpos.to(video_tracking_step.image_pe_buffer.device),
                sample_prev_obj_ptrs.to(video_tracking_step.image_pe_buffer.device),
                sample_prev_obj_ptr_valid.to(video_tracking_step.image_pe_buffer.device),
                sample_prev_obj_ptr_is_cond.to(video_tracking_step.image_pe_buffer.device),
                sample_prev_obj_ptr_tpos.to(video_tracking_step.image_pe_buffer.device),
            ),
            artifacts.video_tracking_step,
            [
                "sam2_vision_pos_enc_0",
                "sam2_vision_pos_enc_1",
                "sam2_vision_pos_enc_2",
                "sam2_backbone_fpn_0",
                "sam2_backbone_fpn_1",
                "sam2_backbone_fpn_2",
                "point_coords",
                "point_labels",
                "box_xyxy",
                "box_valid_mask",
                "mask_input",
                "prev_maskmem_features",
                "prev_maskmem_pos_enc",
                "prev_memory_valid",
                "prev_memory_is_cond",
                "prev_memory_tpos",
                "prev_obj_ptrs",
                "prev_obj_ptr_valid",
                "prev_obj_ptr_is_cond",
                "prev_obj_ptr_tpos",
            ],
            [
                "tracking_low_res_masks",
                "tracking_high_res_masks",
                "tracking_object_score_logits",
                "new_maskmem_features",
                "new_maskmem_pos_enc",
                "new_obj_ptr",
            ],
            args.opset,
        )

    print("Exported:")
    for path in [
        artifacts.image_encoder,
        artifacts.text_encoder,
        artifacts.reference_feature_encoder,
        artifacts.grounding_decoder,
        artifacts.grounding_decoder_with_reference,
        artifacts.video_tracking_step,
        artifacts.interactive_decoder,
        artifacts.interactive_dense_pe,
    ]:
        if path.exists():
            print(f"  {path}")


if __name__ == "__main__":
    main()


r"""
 python export_onnx.py --checkpoint C:\Users\lsn\.cache\modelscope\hub\models\facebook\sam3\sam3.pt --output-dir output --device cpu --modules all
"""