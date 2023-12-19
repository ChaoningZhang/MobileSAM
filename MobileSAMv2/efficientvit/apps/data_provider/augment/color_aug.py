# EfficientViT: Multi-Scale Linear Attention for High-Resolution Dense Prediction
# Han Cai, Junyan Li, Muyan Hu, Chuang Gan, Song Han
# International Conference on Computer Vision (ICCV), 2023

import numpy as np
import torchvision.transforms as transforms
from PIL import Image
from timm.data.auto_augment import rand_augment_transform

__all__ = ["ColorAug", "RandAug"]


class ImageAug:
    def aug_image(self, image: Image.Image) -> Image.Image:
        raise NotImplementedError

    def __call__(self, feed_dict: dict or np.ndarray or Image.Image) -> dict or np.ndarray or Image.Image:
        if isinstance(feed_dict, dict):
            output_dict = feed_dict
            image = feed_dict[self.key]
        else:
            output_dict = None
            image = feed_dict
        is_ndarray = isinstance(image, np.ndarray)
        if is_ndarray:
            image = Image.fromarray(image)

        image = self.aug_image(image)

        if is_ndarray:
            image = np.array(image)

        if output_dict is None:
            return image
        else:
            output_dict[self.key] = image
            return output_dict


class ColorAug(transforms.ColorJitter, ImageAug):
    def __init__(self, brightness=0, contrast=0, saturation=0, hue=0, key="data"):
        super().__init__(
            brightness=brightness,
            contrast=contrast,
            saturation=saturation,
            hue=hue,
        )
        self.key = key

    def aug_image(self, image: Image.Image) -> Image.Image:
        return transforms.ColorJitter.forward(self, image)

    def forward(self, feed_dict: dict or np.ndarray or Image.Image) -> dict or np.ndarray or Image.Image:
        return ImageAug.__call__(self, feed_dict)


class RandAug(ImageAug):
    def __init__(self, config: dict[str, any], mean: tuple[float, float, float], key="data"):
        n = config.get("n", 2)
        m = config.get("m", 9)
        mstd = config.get("mstd", 1.0)
        inc = config.get("inc", 1)
        tpct = config.get("tpct", 0.45)
        config_str = f"rand-n{n}-m{m}-mstd{mstd}-inc{inc}"

        aa_params = dict(
            translate_pct=tpct,
            img_mean=tuple([min(255, round(255 * x)) for x in mean]),
            interpolation=Image.BICUBIC,
        )
        self.aug_op = rand_augment_transform(config_str, aa_params)
        self.key = key

    def aug_image(self, image: Image.Image) -> Image.Image:
        return self.aug_op(image)

    def __repr__(self):
        return self.aug_op.__repr__()
