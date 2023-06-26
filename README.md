# Faster Segment Anything (MobileSAM)

![SAM design](assets/model_diagram.jpg?raw=true)

**MobileSAM** keeps exactly the same pipeline as the original SAM, but replaces the original heavyweight encoder (632M) with a much smaller Tiny-ViT (5M). It performs on par with the original SAM. For inference speed, MobileSAM runs around 10ms per image: 8ms on the image encoder and 2ms on the mask decoder. It is 7 times smaller and 4 times faster than the concurrent FastSAM. Performacne-wise, MobileSAM outperforms FastSAM in all aspects.

**How to Adapt from SAM to MobileSAM?** Since MobileSAM keeps exactly the same pipeline as the original SAM, we inherit pre-processing, post-processing, and all other interfaces from the original SAM. The users who use the original SAM can adapt to MobileSAM with zero effort, by assuing everything is exactly the same except for a smaller image encoder in the SAM. The only problem you might encounter is to load our pretrained model weights.

**How is MobileSAM trained?** MobileSAM is trained on a single GPU with 100k datasets (1% of the original images) for less than a day. The training code will be available soon.

<p float="left">
  <img src="assets/mask1.png?raw=true" width="49.1%" />
  <img src="assets/mask2.jpg?raw=true" width="49.1%" /> 
</p>

## Installation

The code requires `python>=3.8`, as well as `pytorch>=1.7` and `torchvision>=0.8`. Please follow the instructions [here](https://pytorch.org/get-started/locally/) to install both PyTorch and TorchVision dependencies. Installing both PyTorch and TorchVision with CUDA support is strongly recommended.

Install Mobile Segment Anything:

```
pip install git+https://github.com/ChaoningZhang/MobileSAM.git
```

or clone the repository locally and install with

```
git clone git@github.com:ChaoningZhang/MobileSAM.git
cd MobileSAM; pip install -e .
```

The following optional dependencies are necessary for mask post-processing, saving masks in COCO format, the example notebooks, and exporting the model in ONNX format. `jupyter` is also required to run the example notebooks.

```
pip install opencv-python pycocotools matplotlib onnxruntime onnx
```

## <a name="GettingStarted"></a>Getting Started

First download a [model checkpoint](#model-checkpoints). Then the model can be used in just a few lines to get masks from a given prompt:

```
from segment_anything import SamPredictor, sam_model_registry
sam = sam_model_registry["<model_type>"](checkpoint="<path/to/checkpoint>")
predictor = SamPredictor(sam)
predictor.set_image(<your_image>)
masks, _, _ = predictor.predict(<input_prompts>)
```

or generate masks for an entire image:

```
from segment_anything import SamAutomaticMaskGenerator, sam_model_registry
sam = sam_model_registry["<model_type>"](checkpoint="<path/to/checkpoint>")
mask_generator = SamAutomaticMaskGenerator(sam)
masks = mask_generator.generate(<your_image>)
```

<p float="left">
  <img src="assets/notebook1.png?raw=true" width="49.1%" />
  <img src="assets/notebook2.png?raw=true" width="48.9%" />
</p>

## Citing Segment Anything

If you use MobileSAM in your research, please use the following BibTeX entry.

