#!/usr/bin/env python
#
# Copyright 2024 Amazon.com, Inc. or its affiliates. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License"). You may not use this file
# except in compliance with the License. A copy of the License is located at
#
# http://aws.amazon.com/apache2.0/
#
# or in the "LICENSE.txt" file accompanying this file. This file is distributed on an "AS IS"
# BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, express or implied. See the License for
# the specific language governing permissions and limitations under the License.
import os
import sys
from typing import Any

import torch
from sam2.modeling.sam2_base import SAM2Base
from sam2.sam2_image_predictor import SAM2ImagePredictor
from torch import nn


class Sam2Wrapper(nn.Module):

    def __init__(self, sam_model: SAM2Base, multimask_output: bool) -> None:
        super().__init__()
        self.model = sam_model
        self.image_encoder = sam_model.image_encoder
        self.no_mem_embed = sam_model.no_mem_embed
        self.mask_decoder = sam_model.sam_mask_decoder
        self.prompt_encoder = sam_model.sam_prompt_encoder
        self.img_size = sam_model.image_size
        self.multimask_output = multimask_output
        self.sparse_embedding = None

    @torch.no_grad()
    def encode(self, x: torch.Tensor) -> tuple[Any, Any, Any]:
        backbone_out = self.image_encoder(x)
        backbone_out["backbone_fpn"][0] = self.model.sam_mask_decoder.conv_s0(
            backbone_out["backbone_fpn"][0])
        backbone_out["backbone_fpn"][1] = self.model.sam_mask_decoder.conv_s1(
            backbone_out["backbone_fpn"][1])

        feature_maps = backbone_out["backbone_fpn"][-self.model.
                                                    num_feature_levels:]
        vision_pos_embeds = backbone_out["vision_pos_enc"][-self.model.
                                                           num_feature_levels:]

        feat_sizes = [(x.shape[-2], x.shape[-1]) for x in vision_pos_embeds]

        # flatten NxCxHxW to HWxNxC
        vision_feats = [x.flatten(2).permute(2, 0, 1) for x in feature_maps]
        vision_feats[-1] = vision_feats[-1] + self.no_mem_embed

        feats = [
            feat.permute(1, 2, 0).reshape(1, -1, *feat_size)
            for feat, feat_size in zip(vision_feats[::-1], feat_sizes[::-1])
        ][::-1]

        return feats[0], feats[1], feats[2]

    @torch.no_grad()
    def forward(
        self,
        image_embed: torch.Tensor,
        high_res_feats_0: torch.Tensor,
        high_res_feats_1: torch.Tensor,
        point_coords: torch.Tensor,
        point_labels: torch.Tensor,
        mask_input: torch.Tensor,
        has_mask_input: torch.Tensor,
    ):
        sparse_embedding = self._embed_points(point_coords, point_labels)
        self.sparse_embedding = sparse_embedding
        dense_embedding = self._embed_masks(mask_input, has_mask_input)

        high_res_feats = [high_res_feats_0, high_res_feats_1]
        image_embed = image_embed

        masks, iou_predictions, _, _ = self.mask_decoder.predict_masks(
            image_embeddings=image_embed,
            image_pe=self.prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=sparse_embedding,
            dense_prompt_embeddings=dense_embedding,
            repeat_image=False,
            high_res_features=high_res_feats,
        )

        if self.multimask_output:
            masks = masks[:, 1:, :, :]
            iou_predictions = iou_predictions[:, 1:]
        else:
            masks, iou_pred = (
                self.mask_decoder._dynamic_multimask_via_stability(
                    masks, iou_predictions))

        masks = torch.clamp(masks, -32.0, 32.0)

        return masks, iou_predictions

    def _embed_points(self, point_coords: torch.Tensor,
                      point_labels: torch.Tensor) -> torch.Tensor:

        point_coords = point_coords + 0.5

        padding_point = torch.zeros((point_coords.shape[0], 1, 2),
                                    device=point_coords.device)
        padding_label = -torch.ones(
            (point_labels.shape[0], 1), device=point_labels.device)
        point_coords = torch.cat([point_coords, padding_point], dim=1)
        point_labels = torch.cat([point_labels, padding_label], dim=1)

        point_coords[:, :, 0] = point_coords[:, :, 0] / self.model.image_size
        point_coords[:, :, 1] = point_coords[:, :, 1] / self.model.image_size

        point_embedding = self.prompt_encoder.pe_layer._pe_encoding(
            point_coords)
        point_labels = point_labels.unsqueeze(-1).expand_as(point_embedding)

        point_embedding = point_embedding * (point_labels != -1)
        point_embedding = (point_embedding +
                           self.prompt_encoder.not_a_point_embed.weight *
                           (point_labels == -1))

        for i in range(self.prompt_encoder.num_point_embeddings):
            point_embedding = (point_embedding +
                               self.prompt_encoder.point_embeddings[i].weight *
                               (point_labels == i))

        return point_embedding

    def _embed_masks(self, input_mask: torch.Tensor,
                     has_mask_input: torch.Tensor) -> torch.Tensor:
        mask_embedding = has_mask_input * self.prompt_encoder.mask_downscaling(
            input_mask)
        mask_embedding = mask_embedding + (
            1 - has_mask_input
        ) * self.prompt_encoder.no_mask_embed.weight.reshape(1, -1, 1, 1)
        return mask_embedding


def trace_model(model_id: str):
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    model_name = f"{model_id[9:]}"
    os.makedirs(model_name, exist_ok=True)

    predictor = SAM2ImagePredictor.from_pretrained(model_id, device=device)
    model = Sam2Wrapper(predictor.model, True)

    input_image = torch.ones(1, 3, 1024, 1024).to(device)
    high_res_feats_0, high_res_feats_1, image_embed = model.encode(input_image)

    # trace decoder model
    embed_size = (
        predictor.model.image_size // predictor.model.backbone_stride,
        predictor.model.image_size // predictor.model.backbone_stride,
    )
    mask_input_size = [4 * x for x in embed_size]

    point_coords = torch.randint(low=0,
                                 high=1024,
                                 size=(1, 5, 2),
                                 dtype=torch.float)
    point_labels = torch.randint(low=0, high=1, size=(1, 5), dtype=torch.float)
    mask_input = torch.randn(1, 1, *mask_input_size, dtype=torch.float)
    has_mask_input = torch.tensor([1], dtype=torch.float)

    converted = torch.jit.trace_module(
        model, {
            "encode":
            input_image,
            "forward": (image_embed, high_res_feats_0, high_res_feats_1,
                        point_coords, point_labels, mask_input, has_mask_input)
        })
    torch.jit.save(converted, f"{model_name}/{model_name}.pt")

    # save serving.properties
    serving_file = os.path.join(model_name, "serving.properties")
    with open(serving_file, "w") as f:
        f.write(
            f"engine=PyTorch\n"
            f"option.modelName={model_name}\n"
            f"translatorFactory=ai.djl.modality.cv.translator.Sam2TranslatorFactory\n"
            f"encode_method=encode\n")


if __name__ == '__main__':
    hf_model_id = sys.argv[1] if len(
        sys.argv) > 1 else "facebook/sam2-hiera-tiny"
    trace_model(hf_model_id)
