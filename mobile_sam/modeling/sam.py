# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
from torch import nn
from torch.nn import functional as F

from typing import Any, Dict, List, Tuple, Union

from tiny_vit_sam import TinyViT
from image_encoder import ImageEncoderViT
from mask_decoder import MaskDecoder
from transformer import TwoWayTransformer
from prompt_encoder import PromptEncoder

from huggingface_hub import PyTorchModelHubMixin, hf_hub_download


class Sam(nn.Module):
    mask_threshold: float = 0.0
    image_format: str = "RGB"

    def __init__(
        self,
        image_encoder: Union[ImageEncoderViT, TinyViT],
        prompt_encoder: PromptEncoder,
        mask_decoder: MaskDecoder,
        pixel_mean: List[float] = [123.675, 116.28, 103.53],
        pixel_std: List[float] = [58.395, 57.12, 57.375],
    ) -> None:
        """
        SAM predicts object masks from an image and input prompts.

        Arguments:
          image_encoder (ImageEncoderViT): The backbone used to encode the
            image into image embeddings that allow for efficient mask prediction.
          prompt_encoder (PromptEncoder): Encodes various types of input prompts.
          mask_decoder (MaskDecoder): Predicts masks from the image embeddings
            and encoded prompts.
          pixel_mean (list(float)): Mean values for normalizing pixels in the input image.
          pixel_std (list(float)): Std values for normalizing pixels in the input image.
        """
        super().__init__()
        self.image_encoder = image_encoder
        self.prompt_encoder = prompt_encoder
        self.mask_decoder = mask_decoder
        self.register_buffer("pixel_mean", torch.Tensor(pixel_mean).view(-1, 1, 1), False)
        self.register_buffer("pixel_std", torch.Tensor(pixel_std).view(-1, 1, 1), False)

    @property
    def device(self) -> Any:
        return self.pixel_mean.device

    @torch.no_grad()
    def forward(
        self,
        batched_input: List[Dict[str, Any]],
        multimask_output: bool,
    ) -> List[Dict[str, torch.Tensor]]:
        """
        Predicts masks end-to-end from provided images and prompts.
        If prompts are not known in advance, using SamPredictor is
        recommended over calling the model directly.

        Arguments:
          batched_input (list(dict)): A list over input images, each a
            dictionary with the following keys. A prompt key can be
            excluded if it is not present.
              'image': The image as a torch tensor in 3xHxW format,
                already transformed for input to the model.
              'original_size': (tuple(int, int)) The original size of
                the image before transformation, as (H, W).
              'point_coords': (torch.Tensor) Batched point prompts for
                this image, with shape BxNx2. Already transformed to the
                input frame of the model.
              'point_labels': (torch.Tensor) Batched labels for point prompts,
                with shape BxN.
              'boxes': (torch.Tensor) Batched box inputs, with shape Bx4.
                Already transformed to the input frame of the model.
              'mask_inputs': (torch.Tensor) Batched mask inputs to the model,
                in the form Bx1xHxW.
          multimask_output (bool): Whether the model should predict multiple
            disambiguating masks, or return a single mask.

        Returns:
          (list(dict)): A list over input images, where each element is
            as dictionary with the following keys.
              'masks': (torch.Tensor) Batched binary mask predictions,
                with shape BxCxHxW, where B is the number of input prompts,
                C is determined by multimask_output, and (H, W) is the
                original size of the image.
              'iou_predictions': (torch.Tensor) The model's predictions
                of mask quality, in shape BxC.
              'low_res_logits': (torch.Tensor) Low resolution logits with
                shape BxCxHxW, where H=W=256. Can be passed as mask input
                to subsequent iterations of prediction.
        """
        input_images = torch.stack([self.preprocess(x["image"]) for x in batched_input], dim=0)
        image_embeddings = self.image_encoder(input_images)

        outputs = []
        for image_record, curr_embedding in zip(batched_input, image_embeddings):
            if "point_coords" in image_record:
                points = (image_record["point_coords"], image_record["point_labels"])
            else:
                points = None
            sparse_embeddings, dense_embeddings = self.prompt_encoder(
                points=points,
                boxes=image_record.get("boxes", None),
                masks=image_record.get("mask_inputs", None),
            )
            low_res_masks, iou_predictions = self.mask_decoder(
                image_embeddings=curr_embedding.unsqueeze(0),
                image_pe=self.prompt_encoder.get_dense_pe(),
                sparse_prompt_embeddings=sparse_embeddings,
                dense_prompt_embeddings=dense_embeddings,
                multimask_output=multimask_output,
            )
            masks = self.postprocess_masks(
                low_res_masks,
                input_size=image_record["image"].shape[-2:],
                original_size=image_record["original_size"],
            )
            masks = masks > self.mask_threshold
            outputs.append(
                {
                    "masks": masks,
                    "iou_predictions": iou_predictions,
                    "low_res_logits": low_res_masks,
                }
            )
        return outputs

    def postprocess_masks(
        self,
        masks: torch.Tensor,
        input_size: Tuple[int, ...],
        original_size: Tuple[int, ...],
    ) -> torch.Tensor:
        """
        Remove padding and upscale masks to the original image size.

        Arguments:
          masks (torch.Tensor): Batched masks from the mask_decoder,
            in BxCxHxW format.
          input_size (tuple(int, int)): The size of the image input to the
            model, in (H, W) format. Used to remove padding.
          original_size (tuple(int, int)): The original size of the image
            before resizing for input to the model, in (H, W) format.

        Returns:
          (torch.Tensor): Batched masks in BxCxHxW format, where (H, W)
            is given by original_size.
        """
        masks = F.interpolate(
            masks,
            (self.image_encoder.img_size, self.image_encoder.img_size),
            mode="bilinear",
            align_corners=False,
        )
        masks = masks[..., : input_size[0], : input_size[1]]
        masks = F.interpolate(masks, original_size, mode="bilinear", align_corners=False)
        return masks

    def preprocess(self, x: torch.Tensor) -> torch.Tensor:
        """Normalize pixel values and pad to a square input."""
        # Normalize colors
        x = (x - self.pixel_mean) / self.pixel_std

        # Pad
        h, w = x.shape[-2:]
        padh = self.image_encoder.img_size - h
        padw = self.image_encoder.img_size - w
        x = F.pad(x, (0, padw, 0, padh))
        return x


class MobileSAM(Sam, PyTorchModelHubMixin):
    def __init__(self, config):
        
        image_encoder = TinyViT(**config["image_encoder"])
        prompt_encoder = PromptEncoder(**config["prompt_encoder"])
        mask_decoder = MaskDecoder(num_multimask_outputs=config["mask_decoder"]["num_multimask_outputs"],
                                   transformer_dim=config["mask_decoder"]["transformer_dim"],
                                   iou_head_depth=config["mask_decoder"]["iou_head_depth"],
                                   iou_head_hidden_dim=config["mask_decoder"]["iou_head_hidden_dim"],
                                      transformer=TwoWayTransformer(
                                              depth=config["mask_decoder"]["transformer"]["depth"],
                                              embedding_dim=config["mask_decoder"]["transformer"]["embedding_dim"],
                                              mlp_dim=config["mask_decoder"]["transformer"]["mlp_dim"],
                                              num_heads=8,
                            ))
        
        super().__init__(image_encoder, prompt_encoder, mask_decoder, config["pixel_mean"], config["pixel_std"])


prompt_embed_dim = 256
image_size = 1024
vit_patch_size = 16
image_embedding_size = image_size // vit_patch_size   

config = {
    "image_encoder": dict(img_size=1024,
                      in_chans=3,
                      num_classes=1000,
                      embed_dims=[64, 128, 160, 320],
                      depths=[2, 2, 6, 2],
                      num_heads=[2, 4, 5, 10],
                      window_sizes=[7, 7, 14, 7],
                      mlp_ratio=4.,
                      drop_rate=0.,
                      drop_path_rate=0.0,
                      use_checkpoint=False,
                      mbconv_expand_ratio=4.0,
                      local_conv_size=3,
                      layer_lr_decay=0.8),
    "prompt_encoder": dict(
                      embed_dim=prompt_embed_dim,
                      image_embedding_size=(image_embedding_size, image_embedding_size),
                      input_image_size=(image_size, image_size),
                      mask_in_chans=16),
    "mask_decoder": dict(
                      transformer_dim=prompt_embed_dim,
                      iou_head_depth=3,
                      iou_head_hidden_dim=256,
                      num_multimask_outputs=3,
                      transformer=dict(depth=2,
                                     embedding_dim=prompt_embed_dim,
                                      mlp_dim=2048,
                                      num_heads=8,)
                      ),
    "pixel_mean": [123.675, 116.28, 103.53],
    "pixel_std": [58.395, 57.12, 57.375],
}
    
model = MobileSAM(config)

# load weights
filepath = hf_hub_download(repo_id="dhkim2810/MobileSAM", filename="mobile_sam.pt", repo_type="space")
state_dict = torch.load(filepath, map_location="cpu")
model.load_state_dict(state_dict)
            
# save locally
# model.save_pretrained("tiny-vit-sam-224", config=config)

# push to HF hub
model.push_to_hub("nielsr/mobilesam", config=config)

# reload
model = MobileSAM.from_pretrained("nielsr/mobilesam")