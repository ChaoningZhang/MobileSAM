# EfficientViT: Multi-Scale Linear Attention for High-Resolution Dense Prediction
# Han Cai, Junyan Li, Muyan Hu, Chuang Gan, Song Han
# International Conference on Computer Vision (ICCV), 2023

import copy
import warnings

import torch.utils.data
from torch.utils.data.distributed import DistributedSampler

from efficientvit.apps.data_provider.random_resolution import RRSController
from efficientvit.models.utils import val2tuple

__all__ = ["parse_image_size", "random_drop_data", "DataProvider"]


def parse_image_size(size: int or str) -> tuple[int, int]:
    if isinstance(size, str):
        size = [int(val) for val in size.split("-")]
        return size[0], size[1]
    else:
        return val2tuple(size, 2)


def random_drop_data(dataset, drop_size: int, seed: int, keys=("samples",)):
    g = torch.Generator()
    g.manual_seed(seed)  # set random seed before sampling validation set
    rand_indexes = torch.randperm(len(dataset), generator=g).tolist()

    dropped_indexes = rand_indexes[:drop_size]
    remaining_indexes = rand_indexes[drop_size:]

    dropped_dataset = copy.deepcopy(dataset)
    for key in keys:
        setattr(dropped_dataset, key, [getattr(dropped_dataset, key)[idx] for idx in dropped_indexes])
        setattr(dataset, key, [getattr(dataset, key)[idx] for idx in remaining_indexes])
    return dataset, dropped_dataset


class DataProvider:
    data_keys = ("samples",)
    mean_std = {"mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225]}
    SUB_SEED = 937162211  # random seed for sampling subset
    VALID_SEED = 2147483647  # random seed for the validation set

    name: str

    def __init__(
        self,
        train_batch_size: int,
        test_batch_size: int or None,
        valid_size: int or float or None,
        n_worker: int,
        image_size: int or list[int] or str or list[str],
        num_replicas: int or None = None,
        rank: int or None = None,
        train_ratio: float or None = None,
        drop_last: bool = False,
    ):
        warnings.filterwarnings("ignore")
        super().__init__()

        # batch_size & valid_size
        self.train_batch_size = train_batch_size
        self.test_batch_size = test_batch_size or self.train_batch_size
        self.valid_size = valid_size

        # image size
        if isinstance(image_size, list):
            self.image_size = [parse_image_size(size) for size in image_size]
            self.image_size.sort()  # e.g., 160 -> 224
            RRSController.IMAGE_SIZE_LIST = copy.deepcopy(self.image_size)
            self.active_image_size = RRSController.ACTIVE_SIZE = self.image_size[-1]
        else:
            self.image_size = parse_image_size(image_size)
            RRSController.IMAGE_SIZE_LIST = [self.image_size]
            self.active_image_size = RRSController.ACTIVE_SIZE = self.image_size

        # distributed configs
        self.num_replicas = num_replicas
        self.rank = rank

        # build datasets
        train_dataset, val_dataset, test_dataset = self.build_datasets()

        if train_ratio is not None and train_ratio < 1.0:
            assert 0 < train_ratio < 1
            _, train_dataset = random_drop_data(
                train_dataset,
                int(train_ratio * len(train_dataset)),
                self.SUB_SEED,
                self.data_keys,
            )

        # build data loader
        self.train = self.build_dataloader(train_dataset, train_batch_size, n_worker, drop_last=drop_last, train=True)
        self.valid = self.build_dataloader(val_dataset, test_batch_size, n_worker, drop_last=False, train=False)
        self.test = self.build_dataloader(test_dataset, test_batch_size, n_worker, drop_last=False, train=False)
        if self.valid is None:
            self.valid = self.test
        self.sub_train = None

    @property
    def data_shape(self) -> tuple[int, ...]:
        return 3, self.active_image_size[0], self.active_image_size[1]

    def build_valid_transform(self, image_size: tuple[int, int] or None = None) -> any:
        raise NotImplementedError

    def build_train_transform(self, image_size: tuple[int, int] or None = None) -> any:
        raise NotImplementedError

    def build_datasets(self) -> tuple[any, any, any]:
        raise NotImplementedError

    def build_dataloader(self, dataset: any or None, batch_size: int, n_worker: int, drop_last: bool, train: bool):
        if dataset is None:
            return None
        if isinstance(self.image_size, list) and train:
            from efficientvit.apps.data_provider.random_resolution._data_loader import RRSDataLoader

            dataloader_class = RRSDataLoader
        else:
            dataloader_class = torch.utils.data.DataLoader
        if self.num_replicas is None:
            return dataloader_class(
                dataset=dataset,
                batch_size=batch_size,
                shuffle=True,
                num_workers=n_worker,
                pin_memory=True,
                drop_last=drop_last,
            )
        else:
            sampler = DistributedSampler(dataset, self.num_replicas, self.rank)
            return dataloader_class(
                dataset=dataset,
                batch_size=batch_size,
                sampler=sampler,
                num_workers=n_worker,
                pin_memory=True,
                drop_last=drop_last,
            )

    def set_epoch(self, epoch: int) -> None:
        RRSController.set_epoch(epoch, len(self.train))
        if isinstance(self.train.sampler, DistributedSampler):
            self.train.sampler.set_epoch(epoch)

    def assign_active_image_size(self, new_size: int or tuple[int, int]) -> None:
        self.active_image_size = val2tuple(new_size, 2)
        new_transform = self.build_valid_transform(self.active_image_size)
        # change the transform of the valid and test set
        self.valid.dataset.transform = self.test.dataset.transform = new_transform

    def sample_val_dataset(self, train_dataset, valid_transform) -> tuple[any, any]:
        if self.valid_size is not None:
            if 0 < self.valid_size < 1:
                valid_size = int(self.valid_size * len(train_dataset))
            else:
                assert self.valid_size >= 1
                valid_size = int(self.valid_size)
            train_dataset, val_dataset = random_drop_data(
                train_dataset,
                valid_size,
                self.VALID_SEED,
                self.data_keys,
            )
            val_dataset.transform = valid_transform
        else:
            val_dataset = None
        return train_dataset, val_dataset

    def build_sub_train_loader(self, n_samples: int, batch_size: int) -> any:
        # used for resetting BN running statistics
        if self.sub_train is None:
            self.sub_train = {}
        if self.active_image_size in self.sub_train:
            return self.sub_train[self.active_image_size]

        # construct dataset and dataloader
        train_dataset = copy.deepcopy(self.train.dataset)
        if n_samples < len(train_dataset):
            _, train_dataset = random_drop_data(
                train_dataset,
                n_samples,
                self.SUB_SEED,
                self.data_keys,
            )
        RRSController.ACTIVE_SIZE = self.active_image_size
        train_dataset.transform = self.build_train_transform(image_size=self.active_image_size)
        data_loader = self.build_dataloader(train_dataset, batch_size, self.train.num_workers, True, False)

        # pre-fetch data
        self.sub_train[self.active_image_size] = [
            data for data in data_loader for _ in range(max(1, n_samples // len(train_dataset)))
        ]

        return self.sub_train[self.active_image_size]
