# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn as nn

from functools import partial

from .modeling import ImageEncoderViT, MaskDecoder, PromptEncoder, Sam, TwoWayTransformer
from tinyvit.tiny_vit import TinyViT#11000 
from efficientvit.models.efficientvit.backbone import EfficientViTLargeBackbone
from efficientvit.models.efficientvit.sam import SamNeck, EfficientViTSamImageEncoder
from efficientvit.models.nn.norm import set_norm_eps


def _unwrap_checkpoint_state_dict(
    checkpoint,
    preferred_keys=("model", "state_dict"),
    module_prefix=None,
):
    if not isinstance(checkpoint, dict):
        return checkpoint

    # try common wrapper keys first
    for key in preferred_keys:
        if key in checkpoint:
            value = checkpoint[key]
            if isinstance(value, dict):
                # Recursively unwrap if nested
                tensor_count = sum(1 for v in value.values() if torch.is_tensor(v))
                if tensor_count > 0:
                    checkpoint = value
                    break
                # If no tensors found, try unwrapping further
                checkpoint = _unwrap_checkpoint_state_dict(value, preferred_keys)
                break
    else:
        # checkpoint itself may be state_dict - check if it contains tensors
        tensor_count = sum(1 for v in checkpoint.values() if torch.is_tensor(v))
        if not (tensor_count > 0 and tensor_count >= len(checkpoint) * 0.8):
            # find nested state_dict in values
            for key, value in checkpoint.items():
                if isinstance(value, dict):
                    tensor_count = sum(1 for v in value.values() if torch.is_tensor(v))
                    if tensor_count > 0 and tensor_count >= len(value) * 0.8:
                        checkpoint = value
                        break
            else:
                # If still nothing found, return the checkpoint as-is
                available_keys = ", ".join(checkpoint.keys())
                print(f"Warning: Unable to find a model state dict in checkpoint. Available keys: {available_keys}")
                return checkpoint

    if not module_prefix:
        return checkpoint

    module_state_dict = {}
    prefix = f"{module_prefix}."
    for key, value in checkpoint.items():
        if key.startswith(prefix):
            module_state_dict[key[len(prefix):]] = value

    if module_state_dict:
        return module_state_dict

    return checkpoint

def build_sam_vit_h(checkpoint=None):
    return _build_sam(
        encoder_embed_dim=1280,
        encoder_depth=32,
        encoder_num_heads=16,
        encoder_global_attn_indexes=[7, 15, 23, 31],
        checkpoint=checkpoint,
    )

def build_sam_vit_l(checkpoint=None):
    return _build_sam(
        encoder_embed_dim=1024,
        encoder_depth=24,
        encoder_num_heads=16,
        encoder_global_attn_indexes=[5, 11, 17, 23],
        checkpoint=checkpoint,
    )


def build_sam_vit_b(checkpoint=None):
    return _build_sam(
        encoder_embed_dim=768,
        encoder_depth=12,
        encoder_num_heads=12,
        encoder_global_attn_indexes=[2, 5, 8, 11],
        checkpoint=checkpoint,
    )

def _build_sam(
    encoder_embed_dim,
    encoder_depth,
    encoder_num_heads,
    encoder_global_attn_indexes,
    checkpoint=None,
):
    prompt_embed_dim = 256
    image_size = 1024
    vit_patch_size = 16
    image_embedding_size = image_size // vit_patch_size
    sam = Sam(
        image_encoder=ImageEncoderViT(
            depth=encoder_depth,
            embed_dim=encoder_embed_dim,
            img_size=image_size,
            mlp_ratio=4,
            norm_layer=partial(torch.nn.LayerNorm, eps=1e-6),
            num_heads=encoder_num_heads,
            patch_size=vit_patch_size,
            qkv_bias=True,
            use_rel_pos=True,
            global_attn_indexes=encoder_global_attn_indexes,
            window_size=14,
            out_chans=prompt_embed_dim,
        ),
        prompt_encoder=PromptEncoder(
            embed_dim=prompt_embed_dim,
            image_embedding_size=(image_embedding_size, image_embedding_size),
            input_image_size=(image_size, image_size),
            mask_in_chans=16,
        ),
        mask_decoder=MaskDecoder(
            num_multimask_outputs=3,
            transformer=TwoWayTransformer(
                depth=2,
                embedding_dim=prompt_embed_dim,
                mlp_dim=2048,
                num_heads=8,
            ),
            transformer_dim=prompt_embed_dim,
            iou_head_depth=3,
            iou_head_hidden_dim=256,
        ),
        pixel_mean=[123.675, 116.28, 103.53],
        pixel_std=[58.395, 57.12, 57.375],
    )
    sam.eval()
    if checkpoint is not None:
        with open(checkpoint, "rb") as f:
            state_dict = torch.load(f)
        sam.load_state_dict(state_dict,strict=False)
    return sam
class EncoderWrapper(nn.Module):
    """Wraps encoders to expose necessary attributes"""
    def __init__(self, encoder, in_channels=None, out_channels=256):
        super().__init__()
        self.encoder = encoder
        self.out_channels = out_channels
        
        # Copy important attributes from wrapped encoder
        if hasattr(encoder, 'img_size'):
            self.img_size = encoder.img_size
        if hasattr(encoder, 'patch_embed'):
            self.patch_embed = encoder.patch_embed
    
    def forward(self, x):
        return self.encoder(x)
    
    def __getattr__(self, name):
        """Forward attribute access to the wrapped encoder if not found"""
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.encoder, name)


def build_sam_vit_t_encoder(checkpoint=None):
    mobile_sam = TinyViT(img_size=1024, in_chans=3, num_classes=1000,
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
                layer_lr_decay=0.8)
    if checkpoint is not None:
        with open(checkpoint, "rb") as f:
            checkpoint_data = torch.load(f, map_location="cpu", weights_only=False)
        state_dict = _unwrap_checkpoint_state_dict(checkpoint_data, module_prefix="image_encoder")
        mobile_sam.load_state_dict(state_dict, strict=False)
    
    # Wrap to expose img_size attribute
    return EncoderWrapper(mobile_sam)
    
def build_efficientvit_l2_encoder(checkpoint=None):
    backbone = EfficientViTLargeBackbone(
                width_list=[32, 64, 128, 256, 512],
                depth_list=[1, 2, 2, 8, 8],
                in_channels=3,
                qkv_dim=32,
                norm="bn2d",
                act_func="gelu",
                )
    neck = SamNeck(
        fid_list=["stage4", "stage3", "stage2"],
        in_channel_list=[512, 256, 128],
        head_width=256,
        head_depth=12,
        expand_ratio=1,
        middle_op="fmbconv",
        out_dim=256,
    )
    image_encoder = EfficientViTSamImageEncoder(backbone, neck)
    set_norm_eps(image_encoder, 1e-6)
    checkpoints = torch.load(checkpoint)
    checkpoint = checkpoints["state_dict"]
    new_state_dict = {}
    if checkpoint!=None:
        for key, value in checkpoint.items():
            index = key.find("image_encoder.")
            if index != -1:
                new_key = key[index + len("image_encoder."):]
                new_state_dict[new_key] = value
            else:
                continue
        image_encoder.load_state_dict(new_state_dict,strict=True)  # origin
        print('checkpoint_load_scucess')
    return image_encoder


def build_sam_vit_h_encoder(checkpoint=None):
    prompt_embed_dim = 256
    image_size = 1024
    vit_patch_size = 16
    encoder_embed_dim=1280
    encoder_depth=32
    encoder_num_heads=16
    encoder_global_attn_indexes=[7, 15, 23, 31]
    image_encoder=ImageEncoderViT(
            depth=encoder_depth,
            embed_dim=encoder_embed_dim,
            img_size=image_size,
            mlp_ratio=4,
            norm_layer=partial(torch.nn.LayerNorm, eps=1e-6),
            num_heads=encoder_num_heads,
            patch_size=vit_patch_size,
            qkv_bias=True,
            use_rel_pos=True,
            global_attn_indexes=encoder_global_attn_indexes,
            window_size=14,
            out_chans=prompt_embed_dim,
        )
    if checkpoint is not None:
        with open(checkpoint, "rb") as f:
            state_dict = torch.load(f)
        image_encoder.load_state_dict(state_dict,strict=True)
    return image_encoder

def build_PromptGuidedDecoder(checkpoint=None):
    prompt_embed_dim = 256
    image_size = 1024
    vit_patch_size = 16
    image_embedding_size = image_size // vit_patch_size
    prompt_encoder=PromptEncoder(
        embed_dim=prompt_embed_dim,
        image_embedding_size=(image_embedding_size, image_embedding_size),
        input_image_size=(image_size, image_size),
        mask_in_chans=16,
    )
    mask_decoder=MaskDecoder(
        num_multimask_outputs=3,
        transformer=TwoWayTransformer(
            depth=2,
            embedding_dim=prompt_embed_dim,
            mlp_dim=2048,
            num_heads=8,
        ),
        transformer_dim=prompt_embed_dim,
        iou_head_depth=3,
        iou_head_hidden_dim=256,
    )
    if checkpoint is not None:
        with open(checkpoint, "rb") as f:
            state_dict = torch.load(f)
        promt_dict=state_dict['PromtEncoder']
        mask_dict=state_dict['MaskDecoder']
        prompt_encoder.load_state_dict(promt_dict)
        mask_decoder.load_state_dict(mask_dict)
    return {'PromtEncoder':prompt_encoder,'MaskDecoder':mask_decoder}

sam_model_registry = {
    "default": build_sam_vit_h,
    "vit_h": build_sam_vit_h,
    "vit_l": build_sam_vit_l,
    "vit_b": build_sam_vit_b,
    "tiny_vit": build_sam_vit_t_encoder,
    "efficientvit_l2": build_efficientvit_l2_encoder,
    "PromptGuidedDecoder": build_PromptGuidedDecoder,
    "sam_vit_h": build_sam_vit_h_encoder,
}
