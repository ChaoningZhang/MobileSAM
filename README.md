<p float="center">
  <img src="assets/logo2.png?raw=true" width="99.1%" />
</p>

# Faster Segment Anything (MobileSAM)
:pushpin: MobileSAM paper is available at [paper link](https://arxiv.org/pdf/2306.14289.pdf).

:grapes: Meida coverage and Projects that adapt from SAM to MobileSAM (Updates)

* **2023/06/29**: [AnyLabeling](https://github.com/vietanhdev/anylabeling) supports MobileSAM for Image encoder full-finetuing. Thanks for their effort.
* **2023/06/29**: [SonarSAM](https://github.com/wangsssky/SonarSAM) supports MobileSAM for auto-labeling. Thanks for their effort.
* **2023/06/29**: [Stable Diffusion WebUIv](https://github.com/continue-revolution/sd-webui-segment-anything) supports MobileSAM. Thanks for their effort.

* **2023/06/28**: [Grounding-SAM](https://github.com/IDEA-Research/Grounded-Segment-Anything) supports MobileSAM with [Grounded-MobileSAM](https://github.com/IDEA-Research/Grounded-Segment-Anything/tree/main/EfficientSAM). Thanks for their effort.

* **2023/06/27**: MobileSAM has been featured by [AK](https://twitter.com/_akhaliq?lang=en), see the link [AK's MobileSAM tweet](https://twitter.com/_akhaliq/status/1673585099097636864). Thanks for their effort.
![MobileSAM](assets/model_diagram.jpg?raw=true)

<p float="left">
  <img src="assets/mask_comparision.jpg?raw=true" width="99.1%" />
</p>

 
:star: **MobileSAM** performs on par with the original SAM (at least visually) and keeps exactly the same pipeline as the original SAM except for a change on the image encoder. Specifically, we replace the original heavyweight ViT-H encoder (632M) with a much smaller Tiny-ViT (5M). On a single GPU, MobileSAM runs around 12ms per image: 8ms on the image encoder and 4ms on the mask decoder. 

* The comparison of ViT-based image encoder is summarzed as follows: 

    Image Encoder                                      | Original SAM | MobileSAM 
    :-----------------------------------------:|:---------|:-----:
    Paramters      |  611M   | 5M
    Speed      |  452ms  | 8ms

* Original SAM and MobileSAM have exactly the same prompt-guided mask decoder: 

    Mask Decoder                                      | Original SAM | MobileSAM 
    :-----------------------------------------:|:---------|:-----:
    Paramters      |  3.876M   | 3.876M
    Speed      |  4ms  | 4ms

* The comparison of the whole pipeline is summarized as follows:

    Whole Pipeline (Enc+Dec)                                      | Original SAM | MobileSAM 
    :-----------------------------------------:|:---------|:-----:
    Paramters      |  615M   | 9.66M
    Speed      |  456ms  | 12ms

:star: **Original SAM and MobileSAM with a (single) point as the prompt.**  

<p float="left">
  <img src="assets/mask_point.jpg?raw=true" width="99.1%" />
</p>

:star: **Original SAM and MobileSAM with a box as the prompt.** 
<p float="left">
  <img src="assets/mask_box.jpg?raw=true" width="99.1%" />
</p>

:heart: **Is MobileSAM faster and smaller than FastSAM? Yes, to our knowledge!** 
MobileSAM is around 7 times smaller and around 5 times faster than the concurrent FastSAM. 
The comparison of the whole pipeline is summarzed as follows: 
Whole Pipeline (Enc+Dec)                                      | FastSAM | MobileSAM 
:-----------------------------------------:|:---------|:-----:
Paramters      |  68M   | 9.66M
Speed      |  64ms  |12ms

:heart: **Is MobileSAM better than FastSAM for performance? Yes, to our knowledge!** 
FastSAM is suggested to work with multiple points, thus we compare the mIoU with two prompt points (with different pixel distances) and show the resutls as follows. Our MobileSAM is better than FastSAM under this setup to align with the original SAM. 
mIoU                                     | FastSAM | MobileSAM 
:-----------------------------------------:|:---------|:-----:
100      |  0.27   | 0.73
200      |  0.33  |0.71
300      |  0.37  |0.74
400      |  0.41  |0.73
500      |  0.41  |0.73



:heart: **How to Adapt from SAM to MobileSAM?** Since MobileSAM keeps exactly the same pipeline as the original SAM, we inherit pre-processing, post-processing, and all other interfaces from the original SAM. The users who use the original SAM can adapt to MobileSAM with zero effort, by assuming everything is exactly the same except for a smaller image encoder in the SAM.

:heart: **How is MobileSAM trained?** MobileSAM is trained on a single GPU with 100k datasets (1% of the original images) for less than a day. The training code will be available soon.




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
The MobileSAM can be loaded in the following ways:

```
from mobile_encoder.setup_mobile_sam import setup_model
checkpoint = torch.load('../weights/mobile_sam.pt')
mobile_sam = setup_model()
mobile_sam.load_state_dict(checkpoint,strict=True)
```

Then the model can be easily used in just a few lines to get masks from a given prompt:

```
from segment_anything import SamPredictor
device = "cuda"
mobile_sam.to(device=device)
mobile_sam.eval()
predictor = SamPredictor(mobile_sam)
predictor.set_image(<your_image>)
masks, _, _ = predictor.predict(<input_prompts>)
```

or generate masks for an entire image:

```
from segment_anything import SamAutomaticMaskGenerator

mask_generator = SamAutomaticMaskGenerator(mobile_sam)
masks = mask_generator.generate(<your_image>)
```


## BibTex of our MobileSAM
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

<details>
<summary>
<a href="https://github.com/facebookresearch/segment-anything">SAM</a> (Segment Anything) [<b>bib</b>]
</summary>

```bibtex
@article{kirillov2023segany,
  title={Segment Anything}, 
  author={Kirillov, Alexander and Mintun, Eric and Ravi, Nikhila and Mao, Hanzi and Rolland, Chloe and Gustafson, Laura and Xiao, Tete and Whitehead, Spencer and Berg, Alexander C. and Lo, Wan-Yen and Doll{\'a}r, Piotr and Girshick, Ross},
  journal={arXiv:2304.02643},
  year={2023}
}
```
</details>



<details>
<summary>
<a href="https://github.com/microsoft/Cream/tree/main/TinyViT">TinyViT</a> (TinyViT: Fast Pretraining Distillation for Small Vision Transformers) [<b>bib</b>]
</summary>

```bibtex
@InProceedings{tiny_vit,
  title={TinyViT: Fast Pretraining Distillation for Small Vision Transformers},
  author={Wu, Kan and Zhang, Jinnian and Peng, Houwen and Liu, Mengchen and Xiao, Bin and Fu, Jianlong and Yuan, Lu},
  booktitle={European conference on computer vision (ECCV)},
  year={2022}
```
</details>




