# EfficientViT: Multi-Scale Linear Attention for High-Resolution Dense Prediction
# Han Cai, Junyan Li, Muyan Hu, Chuang Gan, Song Han
# International Conference on Computer Vision (ICCV), 2023

import torch

__all__ = ["label_smooth"]


def label_smooth(target: torch.Tensor, n_classes: int, smooth_factor=0.1) -> torch.Tensor:
    # convert to one-hot
    batch_size = target.shape[0]
    target = torch.unsqueeze(target, 1)
    soft_target = torch.zeros((batch_size, n_classes), device=target.device)
    soft_target.scatter_(1, target, 1)
    # label smoothing
    soft_target = torch.add(soft_target * (1 - smooth_factor), smooth_factor / n_classes)
    return soft_target
