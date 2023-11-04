# mamba install timm onnxruntime onnx monkeytype
import sys
sys.path.append(".")
import torch
import numpy as np
import matplotlib.pyplot as plt
import os
from mobile_sam import sam_model_registry, SamPredictor
from mobile_sam.utils.onnx import SamOnnxModel
from torch import nn
import onnxruntime
from onnxruntime.quantization import QuantType
from onnxruntime.quantization.quantize import quantize_dynamic
# sam_checkpoint = "notebooks/sam_vit_h_4b8939.pth"
# model_type = "vit_h"

# device = "cuda"

# class Encoder(torch.nn.Module):
#     def __init__(self, sam):
#         super().__init__()
#         self.encoder = sam.image_encoder
# 
#     def forward(self, x):
#         return self.encoder(x)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(prog='Convert', description='Convert SAM model to Torchscript or CoreML')#
    parser.add_argument("--jit_method", default="script", help="one of script, trace[, dynamo].")
    parser.add_argument("--convert_method", default="pt_mobile", help="convert method. pt_mobile / onnx,ct / onnx,tf")
    parser.add_argument("--model_type", default="vit_t", help="registered model type")
    parser.add_argument("--checkpoint", default="./weights/mobile_sam.pt", help="model file")
    parser.add_argument('output', help="Output directory.")
    print("Loading model...")
    args = parser.parse_args()
    
    os.makedirs(args.output, exist_ok=True)
    sam = sam_model_registry[args.model_type](checkpoint=args.checkpoint) # .cuda().eval()
    # sam.to(device=device)
    model = SamOnnxModel(sam, return_single_mask=True) # .cuda().eval() # s.SamForCoreML(sam)

    jit_method = args.jit_method # Trace: "ValueError: Incompatible dim 3 in shapes (1, 3, 1024, 1024) vs. (1, 1, 1, 3)". Also, if statements

    fn = sam.image_encoder.eval() # Encoder(sam).eval() # sam.image_encoder.layers[0].blocks[0] # sam.image_encoder
    ex = torch.randn(1, 3, 1024, 1024, dtype=torch.float32)
    embed_dim = sam.prompt_encoder.embed_dim
    embed_size = sam.prompt_encoder.image_embedding_size
    mask_input_size = [4 * x for x in embed_size]

    out = fn(ex)
    print("test ok")

    if jit_method == "script":
        print("Script...")

        def replace_gelu_with_tanh(model):
            for child_name, child_module in model.named_children():
                if isinstance(child_module, nn.GELU):
                    print("replacing gelu with tanh")
                    setattr(model, child_name, nn.Tanh())
                else:
                    replace_gelu_with_tanh(child_module)
        replace_gelu_with_tanh(fn)
        embedding_model_ts = torch.jit.script(
            fn, # .cuda(),
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
    # elif jit_method == "trace":
    #     print("Trace...")
    #     embedding_model_ts = torch.jit.trace(
    #         fn, 
    #         example_inputs=[ ex ],
    #     )
    # elif jit_method == "dynamo":
    #     embedding_model_ts = torch.compile(fn)
    #     print("compiled...")
    else:
        raise Exception()
    
    convert_method = args.convert_method.split(",") # ["pt_mobile"] # ["onnx", "ct"] # ["onnx", "ct"]
    onnx_out = "embedding_model_ts.onnx"

    if "pt_mobile" in convert_method:
        def save_pt(model, model_filename: str):
            print("Optimizing for Pytorch Mobile")
            from torch.utils.mobile_optimizer import optimize_for_mobile
            torch.jit.save(model, os.path.join(args.output, f"{model_filename}.pt"))
            # embedding_model_ts = torch.jit.load(out1)
            print("before optimize", torch.jit.export_opnames(model))
            # torch.quantization.fuse_models...
            # embedding_model_metal_ts = optimize_for_mobile(model) # , backend="metal")
            # print("after optimize", torch.jit.export_opnames(model))
            # output_file = os.path.join(args.output, "vit_image_embedding_metal.ptl")
            # print(f"Saving to {output_file}")
            # embedding_model_metal_ts._save_for_lite_interpreter(model_filename_opt)
            model_cpu = optimize_for_mobile(model, backend="cpu")
            print("after optimize for cpu: ", torch.jit.export_opnames(model_cpu))
            model_cpu._save_for_lite_interpreter(os.path.join(args.output, f"cpu_{model_filename}.ptl"))
            model_metal = optimize_for_mobile(model, backend="metal")
            print("after optimize for metal: ", torch.jit.export_opnames(model_metal))
            print(model_metal.code)
            model_metal._save_for_lite_interpreter(os.path.join(args.output, f"metal_{model_filename}.ptl"))


        save_pt(embedding_model_ts, "vit_image_embedding")
        save_pt(predictor_model_ts, "mobilesam_predictor")
    else:
        raise Exception()
    # elif "onnx" in convert_method:
    #     # Note, onnx support has been removed in coremltools after 5.2. This means last tested pytorch version is 1.10.2
    #     inputs = {
    #         "x": ex,
    #     }
    # 
    #     torch.onnx.export(
    #         embedding_model_ts.cpu().eval(),
    #         args=inputs, # list(inputs.values()),
    #         f=onnx_out,
    #         verbose=True,
    #         input_names=["x"],
    #         output_names=["out"],
    #         dynamic_axes={'x': {0: 'batch_size', 2: 'height', 3: 'width'}}
    #     )
    #     print("succeeded to onnx")
    # 
    # elif "dynamo_onnx" in convert_method:
    #     export_output = torch.onnx.dynamo_export(embedding_model_ts, ex)
    #     export_output.save(onnx_out)
    #     print("dynamo onnx output successful")
    # 
    # if "ct" in convert_method:
    #     print("Convert to coreml")
    #     onnx_model = ct.utils.load_spec(onnx_out)
    # 
    #     image_input_shape = ct.Shape((1, 3, 1024, 1024))
    #     pixel_bias = [-x for x in sam.pixel_mean]
    #     pixel_scale = [1./x for x in sam.pixel_std]
    # 
    #     image_input = ct.ImageType(name="x",
    #                                shape=image_input_shape,
    #                                scale=pixel_scale, bias=pixel_bias)
    #     print("Convert...")
    #     embedding_model_coreml = ct.convert(
    #         onnx_model, # "embedding_model_ts.onnx",
    #         source='pytorch',
    #         convert_to="mlprogram",
    #         inputs=[
    #             image_input,
    #         ],
    #         debug=True
    #      )
    # 
    #     
    # elif "torchscript_ct" in convert_method:
    #     # The following does not work with the current version.
    #     image_input_shape = ct.Shape((1, 3, 1024, 1024))
    #     pixel_bias = [-x for x in sam.pixel_mean]
    #     pixel_scale = [1./x for x in sam.pixel_std]
    # 
    #     image_input = ct.ImageType(name="input_1",
    #                                shape=image_input_shape,
    #                                scale=pixel_scale, bias=pixel_bias)
    #     print("Convert...")
    #     embedding_model_coreml = ct.convert(
    #         embedding_model_ts.eval(),
    #         source='pytorch',
    #         convert_to="mlprogram",
    #         inputs=[
    #             image_input,
    #         ],
    #         debug=True
    #      )
    # elif "tf" in convert_method:
    #     # https://github.com/sithu31296/PyTorch-ONNX-TFLite
    #     # git clone https://github.com/onnx/onnx-tensorflow.git && cd onnx-tensorflow
    #     # pip install -e .
    #     import onnx
    # 
    #     onnx_model = onnx.load(onnx_out)
    # 
    #     breakpoint()
    #     print("Loaded onnx model")
    #     
    #     from onnx_tf.backend import prepare
    #     tf_rep = prepare(onnx_model)
    #     tf_rep.export_graph("embedding_model_tf") # Must not have an extension...?
    #     breakpoint()
    #      print("Exported to TF") 
    # 
    #     import tensorflow as tf
    #     model = tf.saved_model.load("embedding_model_tf")
    #     model.trainable = False
    # 
    #     input_tensor = tf.random.uniform([1, 3, 1024, 1024])
    #     out = model(**{'x': input_tensor})
    #     
    #     breakpoint()
    #     print("TF test ok")
    #     converter = tf.lite.TFLiteConverter.from_saved_model("embedding_model_tf")
    #     converter.target_spec.supported_ops = [
    #         tf.lite.OpsSet.TFLITE_BUILTINS, # enable TensorFlow Lite ops.
    #         tf.lite.OpsSet.SELECT_TF_OPS # enable TensorFlow ops.
    #      ]
    #     tflite_model = converter.convert()
    # 
    #     breakpoint()
    #     print("TF-lite conversion ok")
    #     # Save the model
    #     with open("embedding_model.tflite", 'wb') as f:
    #         f.write(tflite_model)
    #     print("Exported to TF-Lite")



    print("Done")
    # predictor = SamPredictor(sam)
    #print(torch.jit.script(sam))
    
    
