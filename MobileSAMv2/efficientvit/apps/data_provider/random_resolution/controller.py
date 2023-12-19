# EfficientViT: Multi-Scale Linear Attention for High-Resolution Dense Prediction
# Han Cai, Junyan Li, Muyan Hu, Chuang Gan, Song Han
# International Conference on Computer Vision (ICCV), 2023

import copy

import torch
import torchvision.transforms as transforms
import torchvision.transforms.functional as F

from efficientvit.models.utils import torch_random_choices

__all__ = [
    "RRSController",
    "get_interpolate",
    "MyRandomResizedCrop",
]


class RRSController:
    ACTIVE_SIZE = (224, 224)
    IMAGE_SIZE_LIST = [(224, 224)]

    CHOICE_LIST = None

    @staticmethod
    def get_candidates() -> list[tuple[int, int]]:
        return copy.deepcopy(RRSController.IMAGE_SIZE_LIST)

    @staticmethod
    def sample_resolution(batch_id: int) -> None:
        RRSController.ACTIVE_SIZE = RRSController.CHOICE_LIST[batch_id]

    @staticmethod
    def set_epoch(epoch: int, batch_per_epoch: int) -> None:
        g = torch.Generator()
        g.manual_seed(epoch)
        RRSController.CHOICE_LIST = torch_random_choices(
            RRSController.get_candidates(),
            g,
            batch_per_epoch,
        )


def get_interpolate(name: str) -> F.InterpolationMode:
    mapping = {
        "nearest": F.InterpolationMode.NEAREST,
        "bilinear": F.InterpolationMode.BILINEAR,
        "bicubic": F.InterpolationMode.BICUBIC,
        "box": F.InterpolationMode.BOX,
        "hamming": F.InterpolationMode.HAMMING,
        "lanczos": F.InterpolationMode.LANCZOS,
    }
    if name in mapping:
        return mapping[name]
    elif name == "random":
        return torch_random_choices(
            [
                F.InterpolationMode.NEAREST,
                F.InterpolationMode.BILINEAR,
                F.InterpolationMode.BICUBIC,
                F.InterpolationMode.BOX,
                F.InterpolationMode.HAMMING,
                F.InterpolationMode.LANCZOS,
            ],
        )
    else:
        raise NotImplementedError


class MyRandomResizedCrop(transforms.RandomResizedCrop):
    def __init__(
        self,
        scale=(0.08, 1.0),
        ratio=(3.0 / 4.0, 4.0 / 3.0),
        interpolation: str = "random",
    ):
        super(MyRandomResizedCrop, self).__init__(224, scale, ratio)
        self.interpolation = interpolation

    def forward(self, img: torch.Tensor) -> torch.Tensor:
        i, j, h, w = self.get_params(img, list(self.scale), list(self.ratio))
        target_size = RRSController.ACTIVE_SIZE
        return F.resized_crop(img, i, j, h, w, list(target_size), get_interpolate(self.interpolation))

    def __repr__(self) -> str:
        format_string = self.__class__.__name__
        format_string += f"(\n\tsize={RRSController.get_candidates()},\n"
        format_string += f"\tscale={tuple(round(s, 4) for s in self.scale)},\n"
        format_string += f"\tratio={tuple(round(r, 4) for r in self.ratio)},\n"
        format_string += f"\tinterpolation={self.interpolation})"
        return format_string
