import gradio as gr
import numpy as np

from mobile_sam import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor
import torch

# Load the model
model_type = "vit_t"
sam_checkpoint = "./weights/mobile_sam.pt"
device = "cuda" if torch.cuda.is_available() else "cpu"
mobile_sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
mobile_sam.to(device=device)
mobile_sam.eval()
predictor = SamPredictor(mobile_sam)

# Define the example directory
EXAMPLE_DIR = 'example_images/'

# Create the demo
with gr.Blocks(title='MobileSAM demo') as demo:
    gr.Markdown(' # MobileSAM demo.')
    gr.Markdown(' ## Hover over the uploaded image and select a pixel.')
    with gr.Row():
        input_img = gr.Image(label="Input")
        output_img = gr.Image(label="Selected Segment")

    gr.Examples([f'{EXAMPLE_DIR}/orange.jpeg',
                f'{EXAMPLE_DIR}/lion.jpeg'], [input_img])

    def get_select_coords(img: np.array, event: gr.SelectData, default_label: int = 1) -> np.array:
        # The event stores the coordinates of the selected region
        predictor.set_image(img)
        points_coords = np.array([[event.index[0], event.index[1]]])
        points_labels = np.array([default_label])
        masks, _, _ = predictor.predict(points_coords, points_labels)

        # overlay the mask on the image with alpha blending
        mask = masks[0]
        mask = np.stack([mask, mask, mask], axis=2)
        output_img = (img.copy()/255).astype(np.float32)
        output_img[mask == 0] *= 0.5

        return output_img

    input_img.select(get_select_coords, [input_img], output_img)

if __name__ == "__main__":
    demo.launch()
