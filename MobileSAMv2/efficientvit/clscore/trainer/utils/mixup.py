# EfficientViT: Multi-Scale Linear Attention for High-Resolution Dense Prediction
# Han Cai, Junyan Li, Muyan Hu, Chuang Gan, Song Han
# International Conference on Computer Vision (ICCV), 2023

import torch.distributions

from efficientvit.apps.data_provider.augment import rand_bbox
from efficientvit.models.utils.random import torch_randint, torch_shuffle

__all__ = ["apply_mixup", "mixup", "cutmix"]


def apply_mixup(
    images: torch.Tensor,
    labels: torch.Tensor,
    lam: float,
    mix_type="mixup",
) -> tuple[torch.Tensor, torch.Tensor]:
    if mix_type == "mixup":
        return mixup(images, labels, lam)
    elif mix_type == "cutmix":
        return cutmix(images, labels, lam)
    else:
        raise NotImplementedError


def mixup(
    images: torch.Tensor,
    target: torch.Tensor,
    lam: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    rand_index = torch_shuffle(list(range(0, images.shape[0])))

    flipped_images = images[rand_index]
    flipped_target = target[rand_index]

    return (
        lam * images + (1 - lam) * flipped_images,
        lam * target + (1 - lam) * flipped_target,
    )


def cutmix(
    images: torch.Tensor,
    target: torch.Tensor,
    lam: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    rand_index = torch_shuffle(list(range(0, images.shape[0])))

    flipped_images = images[rand_index]
    flipped_target = target[rand_index]

    b, _, h, w = images.shape
    lam_list = []
    for i in range(b):
        bbx1, bby1, bbx2, bby2 = rand_bbox(
            h=h,
            w=w,
            lam=lam,
            rand_func=torch_randint,
        )
        images[i, :, bby1:bby2, bbx1:bbx2] = flipped_images[i, :, bby1:bby2, bbx1:bbx2]
        lam_list.append(1 - ((bbx2 - bbx1) * (bby2 - bby1) / (h * w)))
    lam = torch.Tensor(lam_list).to(images.device).view(b, 1)
    return images, lam * target + (1 - lam) * flipped_target
