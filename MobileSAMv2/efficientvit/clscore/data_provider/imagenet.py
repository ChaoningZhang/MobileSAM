# EfficientViT: Multi-Scale Linear Attention for High-Resolution Dense Prediction
# Han Cai, Junyan Li, Muyan Hu, Chuang Gan, Song Han
# International Conference on Computer Vision (ICCV), 2023

import copy
import math
import os

import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder

from efficientvit.apps.data_provider import DataProvider
from efficientvit.apps.data_provider.augment import RandAug
from efficientvit.apps.data_provider.random_resolution import MyRandomResizedCrop, get_interpolate
from efficientvit.apps.utils import partial_update_config
from efficientvit.models.utils import val2list

__all__ = ["ImageNetDataProvider"]


class ImageNetDataProvider(DataProvider):
    name = "imagenet"

    data_dir = "/dataset/imagenet"
    n_classes = 1000
    _DEFAULT_RRC_CONFIG = {
        "train_interpolate": "random",
        "test_interpolate": "bicubic",
        "test_crop_ratio": 1.0,
    }

    def __init__(
        self,
        data_dir: str or None = None,
        rrc_config: dict or None = None,
        data_aug: dict or list[dict] or None = None,
        ###########################################
        train_batch_size=128,
        test_batch_size=128,
        valid_size: int or float or None = None,
        n_worker=8,
        image_size: int or list[int] = 224,
        num_replicas: int or None = None,
        rank: int or None = None,
        train_ratio: float or None = None,
        drop_last: bool = False,
    ):
        self.data_dir = data_dir or self.data_dir
        self.rrc_config = partial_update_config(
            copy.deepcopy(self._DEFAULT_RRC_CONFIG),
            rrc_config or {},
        )
        self.data_aug = data_aug

        super().__init__(
            train_batch_size,
            test_batch_size,
            valid_size,
            n_worker,
            image_size,
            num_replicas,
            rank,
            train_ratio,
            drop_last,
        )

    def build_valid_transform(self, image_size: tuple[int, int] or None = None) -> any:
        image_size = (image_size or self.active_image_size)[0]
        crop_size = int(math.ceil(image_size / self.rrc_config["test_crop_ratio"]))
        return transforms.Compose(
            [
                transforms.Resize(
                    crop_size,
                    interpolation=get_interpolate(self.rrc_config["test_interpolate"]),
                ),
                transforms.CenterCrop(image_size),
                transforms.ToTensor(),
                transforms.Normalize(**self.mean_std),
            ]
        )

    def build_train_transform(self, image_size: tuple[int, int] or None = None) -> any:
        image_size = image_size or self.image_size

        # random_resize_crop -> random_horizontal_flip
        train_transforms = [
            MyRandomResizedCrop(interpolation=self.rrc_config["train_interpolate"]),
            transforms.RandomHorizontalFlip(),
        ]

        # data augmentation
        post_aug = []
        if self.data_aug is not None:
            for aug_op in val2list(self.data_aug):
                if aug_op["name"] == "randaug":
                    data_aug = RandAug(aug_op, mean=self.mean_std["mean"])
                elif aug_op["name"] == "erase":
                    from timm.data.random_erasing import RandomErasing

                    random_erase = RandomErasing(aug_op["p"], device="cpu")
                    post_aug.append(random_erase)
                    data_aug = None
                else:
                    raise NotImplementedError
                if data_aug is not None:
                    train_transforms.append(data_aug)
        train_transforms = [
            *train_transforms,
            transforms.ToTensor(),
            transforms.Normalize(**self.mean_std),
            *post_aug,
        ]
        return transforms.Compose(train_transforms)

    def build_datasets(self) -> tuple[any, any, any]:
        train_transform = self.build_train_transform()
        valid_transform = self.build_valid_transform()

        train_dataset = ImageFolder(os.path.join(self.data_dir, "train"), train_transform)
        test_dataset = ImageFolder(os.path.join(self.data_dir, "val"), valid_transform)

        train_dataset, val_dataset = self.sample_val_dataset(train_dataset, valid_transform)
        return train_dataset, val_dataset, test_dataset
