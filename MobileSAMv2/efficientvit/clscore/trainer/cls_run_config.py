# EfficientViT: Multi-Scale Linear Attention for High-Resolution Dense Prediction
# Han Cai, Junyan Li, Muyan Hu, Chuang Gan, Song Han
# International Conference on Computer Vision (ICCV), 2023

from efficientvit.apps.trainer.run_config import RunConfig

__all__ = ["ClsRunConfig"]


class ClsRunConfig(RunConfig):
    label_smooth: float
    mixup_config: dict  # allow none to turn off mixup
    bce: bool
    mesa: dict

    @property
    def none_allowed(self):
        return ["mixup_config", "mesa"] + super().none_allowed
