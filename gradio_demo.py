import os
import cv2
import gradio as gr
import numpy as np
import torch
from mobile_encoder.setup_mobile_sam import setup_model
from segment_anything import SamPredictor


def get_point(img, sel_pix, point_type, evt: gr.SelectData):
    if point_type == 'foreground_point':
        sel_pix.append((evt.index, 1))   
    elif point_type == 'background_point':
        sel_pix.append((evt.index, 0))    
    else:
        sel_pix.append((evt.index, 1))    
    # draw points
    for point, label in sel_pix:
        cv2.drawMarker(img, point, colors[label], markerType=markers[label], markerSize=20, thickness=5)
    if img[..., 0][0, 0] == img[..., 2][0, 0]:  # BGR to RGB
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img if isinstance(img, np.ndarray) else np.array(img)

def undo_points(orig_img, sel_pix):
    temp = orig_img.copy()
    # draw points
    if len(sel_pix) != 0:
        sel_pix.pop()
        for point, label in sel_pix:
            cv2.drawMarker(temp, point, colors[label], markerType=markers[label], markerSize=20, thickness=5)
    if temp[..., 0][0, 0] == temp[..., 2][0, 0]:  
        temp = cv2.cvtColor(temp, cv2.COLOR_BGR2RGB)
    return temp if isinstance(temp, np.ndarray) else np.array(temp)

def store_img(img):
    return img, [] 


if __name__ == "__main__":
    
    print('Initializing MobileSAM models... ')

    sam_checkpoint = "./weights/mobile_sam.pt"
    checkpoint = torch.load(sam_checkpoint)
    mobile_sam = setup_model()
    mobile_sam.load_state_dict(checkpoint,strict=True)
    device = "cuda"
    mobile_sam.to(device=device)
    mobile_sam.eval()
    predictor = SamPredictor(mobile_sam)
    
    
    colors = [(255, 0, 0), (0, 255, 0)]
    markers = [1, 5]

    def run_inference(input_x, selected_points):
        with torch.no_grad():
            input_x = cv2.cvtColor(input_x, cv2.COLOR_BGR2RGB)
            predictor.set_image(input_x)
            
            if len(selected_points) != 0:
                points = np.array([p for p, _ in selected_points])
                labels = np.array([int(l) for _, l in selected_points])
                masks, scores, logits = predictor.predict(
                    point_coords=points,
                    point_labels=labels,
                    box = None,
                    multimask_output=False,
                )
            else:
                points, labels = None, None
                        
            mask = masks[0]
            mask = np.stack([mask]*3, axis=-1) * 255
            mask[:, :, 0] = 0
            mask[:, :, 1] = 0
            mask = mask.astype(np.float64)
            input_x = input_x.astype(np.float64)

            img = cv2.addWeighted(input_x, 0.6, mask, 0.4, 0)
            
            torch.cuda.empty_cache()

            mask = mask.astype(np.uint8)
            img = img.astype(np.uint8)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
        return  mask, img

    with gr.Blocks() as demo:
        gr.Markdown(
            """
            # <center>MobileSAM
            """
        )
        with gr.Row().style(equal_height=True):
            with gr.Column():
               
                original_image = gr.State(value=None)  
                input_image = gr.Image(type="numpy")
                
                with gr.Column():
                    selected_points = gr.State([])     
                    with gr.Row():
                        undo_button = gr.Button('Remove Points')
                    radio = gr.Radio(['foreground_point', 'background_point'], label='point labels')
                
                button = gr.Button("Start!")

            # show the image with mask
            with gr.Tab(label='Mobile SAM Mask'):
                mask = gr.Image(type='numpy')
            with gr.Tab(label='Mobile SAM Vis'):
                img = gr.Image(type='numpy')

        input_image.upload(
            store_img,
            [input_image],
            [original_image, selected_points]
        )
        input_image.select(
            get_point,
            [input_image, selected_points, radio],
            [input_image],
        )
        undo_button.click(
            undo_points,
            [original_image, selected_points],
            [input_image]
        )
        button.click(run_inference, inputs=[original_image, selected_points], outputs=[mask, img])

        with gr.Row():
            with gr.Column():
                background_image = gr.State(value=None)

    demo.launch()