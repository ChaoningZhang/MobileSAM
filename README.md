# Faster Segment Anything (MobileSAM)

![SAM design](assets/model_diagram.jpg?raw=true)

**MobileSAM** performs on par with the original SAM and keeps exactly the same pipeline as the original SAM except for a change on the image encoder. Specifically, we replace the original heavyweight ViT-H encoder (632M) with a much smaller Tiny-ViT (5M). On a single GPU, MobileSAM runs around 10ms per image: 8ms on the image encoder and 2ms on the mask decoder. 

The comparison of ViT-based image encoder is summarzed as follows: 

Image Encoder                                      | Original SAM | MobileSAM 
:-----------------------------------------:|:---------|:-----:
Paramters      |  611M   | 5M
Speed      |  452ms  | 8ms

Original SAM and MobileSAM have exactly the same prompt-guided mask decoder: 

Mask Decoder                                      | Original SAM | MobileSAM 
:-----------------------------------------:|:---------|:-----:
Paramters      |  3.876M   | 3.876M
Speed      |  2ms  | 2ms

The comparison of the whole pipeline is summarzed as follows: 
Whole Pipeline                                      | Original SAM | MobileSAM 
:-----------------------------------------:|:---------|:-----:
Paramters      |  615M   | 9.66M
Speed      |  454ms  | 10ms

**Is MobileSAM better than FastSAM? To our best knowldege, yes!** MobileSAM is 7 times smaller and 4 times faster than the concurrent FastSAM. Performacne-wise, MobileSAM outperforms FastSAM in all aspects.
The comparison of the whole pipeline is summarzed as follows: 
Whole Pipeline                                      | Original SAM | MobileSAM 
:-----------------------------------------:|:---------|:-----:
Paramters      |  68M   | 9.66M
Speed      |  40  |10ms

**How to Adapt from SAM to MobileSAM?** Since MobileSAM keeps exactly the same pipeline as the original SAM, we inherit pre-processing, post-processing, and all other interfaces from the original SAM. The users who use the original SAM can adapt to MobileSAM with zero effort, by assuing everything is exactly the same except for a smaller image encoder in the SAM.

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


## <a name="GettingStarted"></a>Getting Started

First download a [model checkpoint](#model-checkpoints). Then the model can be used in just a few lines to get masks from a given prompt:

```
from segment_anything import SamPredictor, sam_model_registry

predictor = SamPredictor(sam)
predictor.set_image(<your_image>)
masks, _, _ = predictor.predict(<input_prompts>)
```

or generate masks for an entire image:

```
from segment_anything import SamAutomaticMaskGenerator, sam_model_registry

mask_generator = SamAutomaticMaskGenerator(sam)
masks = mask_generator.generate(<your_image>)
```

<p float="left">
  <img src="assets/notebook1.png?raw=true" width="49.1%" />
  <img src="assets/notebook2.png?raw=true" width="48.9%" />
</p>

## Citing our MobileSAM
If you use MobileSAM in your research, please use the following BibTeX entry. :mega: Thank you!

```bibtex
@article{mobile_sam,
  title={Faster Segment Anything: Towards Lightweight SAM for Mobile Applications},
  author={Zhang, Chaoning and Han, Dongshen and Qiao, Yu and Kim, Jung Uk and Bae, Sung Ho and Lee, Seungkyu and Hong, Choong Seon},
  journal={arXiv preprint arXiv:2306.14289},
  year={2023}
}
```

## Acknowledgement

1. The code of MobileSAM project is heavily dependent on the original [Segment Anything](https://github.com/facebookresearch/segment-anything) project.

```bibtex
@article{kirillov2023segany,
  title={Segment Anything}, 
  author={Kirillov, Alexander and Mintun, Eric and Ravi, Nikhila and Mao, Hanzi and Rolland, Chloe and Gustafson, Laura and Xiao, Tete and Whitehead, Spencer and Berg, Alexander C. and Lo, Wan-Yen and Doll{\'a}r, Piotr and Girshick, Ross},
  journal={arXiv:2304.02643},
  year={2023}
}
```
2. We also provide acknowledgement to [TinyViT](https://github.com/microsoft/Cream/tree/main/TinyViT).   

```bibtex
@InProceedings{tiny_vit,
  title={TinyViT: Fast Pretraining Distillation for Small Vision Transformers},
  author={Wu, Kan and Zhang, Jinnian and Peng, Houwen and Liu, Mengchen and Xiao, Bin and Fu, Jianlong and Yuan, Lu},
  booktitle={European conference on computer vision (ECCV)},
  year={2022}
}
```

