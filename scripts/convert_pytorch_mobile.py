# import sys
# sys.path.append(".")
import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt
import os
from torch import nn
from torch.utils.mobile_optimizer import optimize_for_mobile
from mobile_sam import sam_model_registry, SamPredictor
from mobile_sam.utils.onnx import SamOnnxModel

if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog='Convert', description='Convert SAM model to Torchscript or CoreML')#
    parser.add_argument("--model_type", default="vit_t", help="registered model type")
    parser.add_argument("--checkpoint", default="./weights/mobile_sam.pt", help="model file")
    parser.add_argument('output', help="Output directory.")
    args = parser.parse_args()
    
    os.makedirs(args.output, exist_ok=True)

    print("Loading model...")
    sam = sam_model_registry[args.model_type](checkpoint=args.checkpoint)
    model = SamOnnxModel(sam, return_single_mask=True)

    embed_dim = sam.prompt_encoder.embed_dim
    embed_size = sam.prompt_encoder.image_embedding_size
    enc = sam.image_encoder.eval()
    ex = torch.randn(1, 3, 1024, 1024, dtype=torch.float32)
    mask_input_size = [4 * x for x in embed_size]
    out = enc(ex)

    # def replace_gelu_with_tanh(model):
    #     for child_name, child_module in model.named_children():
    #         if isinstance(child_module, nn.GELU):
    #             print("replacing gelu with tanh")
    #             setattr(model, child_name, nn.Tanh())
    #         else:
    #             replace_gelu_with_tanh(child_module)
    # replace_gelu_with_tanh(enc)

    embedding_model_ts = torch.jit.script(
        enc,
        example_inputs=[ex.unsqueeze(0)], # Why the hell is this unsqueeze necessary?
    )

    decoder_inputs = {
        "image_embeddings": torch.randn(1, embed_dim, *embed_size, dtype=torch.float),
        "point_coords": torch.randint(low=0, high=1024, size=(1, 5, 2), dtype=torch.float),
        "point_labels": torch.randint(low=0, high=4, size=(1, 5), dtype=torch.float),
        "mask_input": torch.randn(1, 1, *mask_input_size, dtype=torch.float),
        "has_mask_input": torch.tensor([1], dtype=torch.float),
        "orig_im_size": torch.tensor([1500, 2250], dtype=torch.float),
    }

    predictor_model_ts = torch.jit.script(
        model,
        example_inputs=[
            decoder_inputs.values()
        ],
    )
    
    def save_pt(model, model_filename: str):
        print("Optimizing for Pytorch Mobile")
        torch.jit.save(model, os.path.join(args.output, f"{model_filename}.pt"))
        print("before optimize", torch.jit.export_opnames(model))
        # torch.quantization.fuse_models...
        model_cpu = optimize_for_mobile(model, backend="cpu")
        print("after optimize for cpu: ", torch.jit.export_opnames(model_cpu))
        model_cpu._save_for_lite_interpreter(os.path.join(args.output, f"cpu_{model_filename}.ptl"))
        model_metal = optimize_for_mobile(model, backend="metal")
        print("after optimize for metal: ", torch.jit.export_opnames(model_metal))
        print(model_metal.code)
        model_metal._save_for_lite_interpreter(os.path.join(args.output, f"metal_{model_filename}.ptl"))

    save_pt(embedding_model_ts, "vit_image_embedding")
    save_pt(predictor_model_ts, "mobilesam_predictor")

    print("Done")

