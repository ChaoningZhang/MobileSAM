import argparse
import ast
import torch
from PIL import Image
import cv2
import os
import sys
from mobilesamv2.promt_mobilesamv2 import ObjectAwareModel
from mobilesamv2 import sam_model_registry, SamPredictor
from typing import Any, Dict, Generator,List
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ObjectAwareModel_path", type=str, default='./PromptGuidedDecoder/ObjectAwareModel.pt', help="ObjectAwareModel path")
    parser.add_argument("--Prompt_guided_Mask_Decoder_path", type=str, default='./PromptGuidedDecoder/Prompt_guided_Mask_Decoder.pt', help="Prompt_guided_Mask_Decoder path")
    parser.add_argument("--encoder_path", type=str, default="./", help="select your own path")
    parser.add_argument("--img_path", type=str, default="./test_images/", help="path to image file")
    parser.add_argument("--imgsz", type=int, default=1024, help="image size")
    parser.add_argument("--iou",type=float,default=0.9,help="yolo iou")
    parser.add_argument("--conf", type=float, default=0.4, help="yolo object confidence threshold")
    parser.add_argument("--retina",type=bool,default=True,help="draw segmentation masks",)
    parser.add_argument("--output_dir", type=str, default="./", help="image save path")
    parser.add_argument("--decoder_batch_size", type=int, default=64, help="boxes per SAM decoder batch")
    parser.add_argument("--max_box_area_ratio", type=float, default=0.5, help="drop detections covering too much of the image")
    parser.add_argument("--min_box_area_ratio", type=float, default=0.0005, help="drop detections that are too small to be useful")
    parser.add_argument("--encoder_type", choices=['tiny_vit','sam_vit_h','mobile_sam','efficientvit_l2','efficientvit_l1','efficientvit_l0'], help="choose the model type")
    return parser.parse_args()
def create_model():
    Prompt_guided_path='./PromptGuidedDecoder/Prompt_guided_Mask_Decoder.pt'
    obj_model_path='./weight/ObjectAwareModel.pt'
    ObjAwareModel = ObjectAwareModel(obj_model_path)
    PromptGuidedDecoder=sam_model_registry['PromptGuidedDecoder'](Prompt_guided_path)
    mobilesamv2 = sam_model_registry['vit_h']()
    mobilesamv2.prompt_encoder=PromptGuidedDecoder['PromtEncoder']
    mobilesamv2.mask_decoder=PromptGuidedDecoder['MaskDecoder']
    return mobilesamv2,ObjAwareModel
    
def show_anns(anns, image):
    """Blend segmentation masks with the image using NumPy"""
    if len(anns) == 0:
        return image
    
    # Convert to numpy if needed
    if isinstance(image, torch.Tensor):
        image = image.cpu().numpy()
    
    # Start with the original image
    output = image.copy().astype(np.float32)
    h, w = output.shape[:2]
    
    # Layer masks on top with blending
    for ann_idx in range(anns.shape[0]):
        # Get mask and threshold at 0.5
        if isinstance(anns, torch.Tensor):
            m = (anns[ann_idx] > 0.5).cpu().numpy()
        else:
            m = anns[ann_idx] > 0.5
        
        # Only apply if mask has any pixels
        if m.any():
            # Generate random color for this mask
            color = np.random.randint(0, 255, 3)
            
            # Blend: 70% image + 30% mask color
            output[m] = output[m] * 0.7 + color * 0.3
    
    return output.astype(np.uint8)

def save_boxes(image, boxes, output_path):
    fig, ax = plt.subplots(figsize=(20, 20))
    ax.imshow(image)
    for box in boxes:
        x0, y0, x1, y1 = box.tolist()
        ax.add_patch(Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False, edgecolor='red', linewidth=2))
    ax.axis('off')
    fig.savefig(output_path, bbox_inches='tight', pad_inches=0.0)
    plt.close(fig)

def save_union_mask_overlay(image, masks, output_path):
    union_mask = masks.any(dim=0).cpu().numpy()
    overlay = image.copy()
    overlay[union_mask] = (0.35 * overlay[union_mask] + 0.65 * np.array([255, 0, 0])).astype(np.uint8)
    plt.figure(figsize=(20, 20))
    plt.imshow(overlay)
    plt.axis('off')
    plt.savefig(output_path, bbox_inches='tight', pad_inches=0.0)
    plt.close()

def filter_boxes(boxes, image_shape, min_area_ratio, max_area_ratio):
    image_h, image_w = image_shape[:2]
    image_area = image_h * image_w
    box_areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    area_ratios = box_areas / image_area
    keep = (area_ratios >= min_area_ratio) & (area_ratios <= max_area_ratio)
    return boxes[keep], keep

def batch_iterator(batch_size: int, *args) -> Generator[List[Any], None, None]:
    assert len(args) > 0 and all(
        len(a) == len(args[0]) for a in args
    ), "Batched iteration must have inputs of all the same size."
    n_batches = len(args[0]) // batch_size + int(len(args[0]) % batch_size != 0)
    for b in range(n_batches):
        yield [arg[b * batch_size : (b + 1) * batch_size] for arg in args]

encoder_path={'efficientvit_l2':'./weight/l2.pt',
            'tiny_vit':'./weight/mobile_sam.pt',
            'sam_vit_h':'./weight/sam_vit_h.pt',}

def main(args):
    # import pdb;pdb.set_trace()
    output_dir=args.output_dir  
    os.makedirs(output_dir, exist_ok=True)
    mobilesamv2, ObjAwareModel=create_model()
    image_encoder=sam_model_registry[args.encoder_type](encoder_path[args.encoder_type])
    mobilesamv2.image_encoder=image_encoder
    device = "cuda" if torch.cuda.is_available() else "cpu"
    mobilesamv2.to(device=device)
    mobilesamv2.eval()
    predictor = SamPredictor(mobilesamv2)
    image_files= os.listdir(args.img_path)
    for image_name in image_files:
        print(image_name)
        image = cv2.imread(os.path.join(args.img_path, image_name))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        obj_results = ObjAwareModel(image,device=device,retina_masks=args.retina,imgsz=args.imgsz,conf=args.conf,iou=args.iou)
        predictor.set_image(image)
        input_boxes1 = obj_results[0].boxes.xyxy
        if input_boxes1.numel() == 0:
            print(f"No objects detected in {image_name}, skipping.")
            continue
        filtered_boxes, keep = filter_boxes(input_boxes1, image.shape, args.min_box_area_ratio, args.max_box_area_ratio)
        print(f"Detected {input_boxes1.shape[0]} boxes, kept {filtered_boxes.shape[0]} after area filtering.")
        if filtered_boxes.numel() == 0:
            print(f"No boxes remained after filtering for {image_name}, skipping.")
            continue
        image_stem, _ = os.path.splitext(image_name)
        save_boxes(image, filtered_boxes.detach().cpu(), os.path.join(output_dir, f"{image_stem}_boxes.png"))
        input_boxes = predictor.transform.apply_boxes_torch(filtered_boxes, predictor.original_size).to(device=device)
        sam_mask=[]
        for (boxes,) in batch_iterator(args.decoder_batch_size, input_boxes):
            with torch.no_grad():
                masks, _, _ = predictor.predict_torch(
                    point_coords=None,
                    point_labels=None,
                    boxes=boxes,
                    mask_input=None,
                    multimask_output=False,
                    return_logits=False,
                )
                sam_mask.append(masks.squeeze(1).float())
                del masks
                if device == "cuda":
                    torch.cuda.empty_cache()
        sam_mask=torch.cat(sam_mask)
        annotation = sam_mask
        
        areas = torch.sum(annotation, dim=(1, 2))
        sorted_indices = torch.argsort(areas, descending=True)
        show_img = annotation[sorted_indices]
        image_stem, _ = os.path.splitext(image_name)
        save_union_mask_overlay(image, show_img.bool(), os.path.join(output_dir, f"{image_stem}_mask_union.png"))
        
        # Blend masks with image and save directly
        blended_image = show_anns(show_img, image)
        blended_pil = Image.fromarray(blended_image)
        blended_pil.save(os.path.join(output_dir, f"{image_stem}_mask_overlay.png"))
        del sam_mask, annotation, show_img, predictor.features
        if device == "cuda":
            torch.cuda.empty_cache()

if __name__ == "__main__":
    args = parse_args()
    main(args)
