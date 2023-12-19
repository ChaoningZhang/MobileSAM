# EfficientViT: Multi-Scale Linear Attention for High-Resolution Dense Prediction
# Han Cai, Junyan Li, Muyan Hu, Chuang Gan, Song Han
# International Conference on Computer Vision (ICCV), 2023

import torch

__all__ = ["REGISTERED_OPTIMIZER_DICT", "build_optimizer"]

# register optimizer here
#   name: optimizer, kwargs with default values
REGISTERED_OPTIMIZER_DICT: dict[str, tuple[type, dict[str, any]]] = {
    "sgd": (torch.optim.SGD, {"momentum": 0.9, "nesterov": True}),
    "adam": (torch.optim.Adam, {"betas": (0.9, 0.999), "eps": 1e-8, "amsgrad": False}),
    "adamw": (torch.optim.AdamW, {"betas": (0.9, 0.999), "eps": 1e-8, "amsgrad": False}),
}


def build_optimizer(
    net_params, optimizer_name: str, optimizer_params: dict or None, init_lr: float
) -> torch.optim.Optimizer:
    optimizer_class, default_params = REGISTERED_OPTIMIZER_DICT[optimizer_name]
    optimizer_params = optimizer_params or {}

    for key in default_params:
        if key in optimizer_params:
            default_params[key] = optimizer_params[key]
    optimizer = optimizer_class(net_params, init_lr, **default_params)
    return optimizer
