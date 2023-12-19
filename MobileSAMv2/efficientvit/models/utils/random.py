# EfficientViT: Multi-Scale Linear Attention for High-Resolution Dense Prediction
# Han Cai, Junyan Li, Muyan Hu, Chuang Gan, Song Han
# International Conference on Computer Vision (ICCV), 2023

import numpy as np
import torch

__all__ = [
    "torch_randint",
    "torch_random",
    "torch_shuffle",
    "torch_uniform",
    "torch_random_choices",
]


def torch_randint(low: int, high: int, generator: torch.Generator or None = None) -> int:
    """uniform: [low, high)"""
    if low == high:
        return low
    else:
        assert low < high
        return int(torch.randint(low=low, high=high, generator=generator, size=(1,)))


def torch_random(generator: torch.Generator or None = None) -> float:
    """uniform distribution on the interval [0, 1)"""
    return float(torch.rand(1, generator=generator))


def torch_shuffle(src_list: list[any], generator: torch.Generator or None = None) -> list[any]:
    rand_indexes = torch.randperm(len(src_list), generator=generator).tolist()
    return [src_list[i] for i in rand_indexes]


def torch_uniform(low: float, high: float, generator: torch.Generator or None = None) -> float:
    """uniform distribution on the interval [low, high)"""
    rand_val = torch_random(generator)
    return (high - low) * rand_val + low


def torch_random_choices(
    src_list: list[any],
    generator: torch.Generator or None = None,
    k=1,
    weight_list: list[float] or None = None,
) -> any or list:
    if weight_list is None:
        rand_idx = torch.randint(low=0, high=len(src_list), generator=generator, size=(k,))
        out_list = [src_list[i] for i in rand_idx]
    else:
        assert len(weight_list) == len(src_list)
        accumulate_weight_list = np.cumsum(weight_list)

        out_list = []
        for _ in range(k):
            val = torch_uniform(0, accumulate_weight_list[-1], generator)
            active_id = 0
            for i, weight_val in enumerate(accumulate_weight_list):
                active_id = i
                if weight_val > val:
                    break
            out_list.append(src_list[active_id])

    return out_list[0] if k == 1 else out_list
