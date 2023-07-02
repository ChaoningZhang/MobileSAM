---
title: MobileSAM
emoji: 🐠
colorFrom: indigo
colorTo: yellow
sdk: gradio
python_version: 3.8.10
sdk_version: 3.35.2
app_file: app.py
pinned: false
license: apache-2.0
---

# Faster Segment Anything(MobileSAM)

Demo of official PyTorch implementation of [MobileSAM](https://github.com/ChaoningZhang/MobileSAM).


**MobileSAM** performs on par with the original SAM (at least visually) and keeps exactly the same pipeline as the original SAM except for a change on the image encoder.
Specifically, we replace the original heavyweight ViT-H encoder (632M) with a much smaller Tiny-ViT (5M). On a single GPU, MobileSAM runs around 12ms per image: 8ms on the image encoder and 4ms on the mask decoder. 

## To run on local PC
First, mobile_sam must be installed to run on pc. Refer to [Installation Instruction](https://github.com/dhkim2810/MobileSAM/tree/master#installation)

Then run the following

```
python app.py
```

## License

The model is licensed under the [Apache 2.0 license](LICENSE).


## Acknowledgement

- [Segment Anything](https://segment-anything.com/) provides the SA-1B dataset and the base codes.
- [TinyViT](https://github.com/microsoft/Cream/tree/main/TinyViT) provides codes and pre-trained models.

## Citing MobileSAM

If you find this project useful for your research, please consider citing the following BibTeX entry.

```bibtex
@article{mobile_sam,
  title={Faster Segment Anything: Towards Lightweight SAM for Mobile Applications},
  author={Zhang, Chaoning and Han, Dongshen and Qiao, Yu and Kim, Jung Uk and Bae, Sung Ho and Lee, Seungkyu and Hong, Choong Seon},
  journal={arXiv preprint arXiv:2306.14289},
  year={2023}
}
```
