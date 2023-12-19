# EfficientViT: Multi-Scale Linear Attention for High-Resolution Dense Prediction
# Han Cai, Junyan Li, Muyan Hu, Chuang Gan, Song Han
# International Conference on Computer Vision (ICCV), 2023

from efficientvit.models.efficientvit import (
    EfficientViTSam,
    efficientvit_sam_l0,
    efficientvit_sam_l1,
    efficientvit_sam_l2,
)
from efficientvit.models.nn.norm import set_norm_eps
from efficientvit.models.utils import load_state_dict_from_file

__all__ = ["create_sam_model"]


REGISTERED_SAM_MODEL: dict[str, str] = {
    "l0": "assets/checkpoints/sam/l0.pt",
    "l1": "assets/checkpoints/sam/l1.pt",
    "l2": "assets/checkpoints/sam/l2.pt",
}


def create_sam_model(name: str, pretrained=True, weight_url: str or None = None, **kwargs) -> EfficientViTSam:
    model_dict = {
        "l0": efficientvit_sam_l0,
        "l1": efficientvit_sam_l1,
        "l2": efficientvit_sam_l2,
    }

    model_id = name.split("-")[0]
    if model_id not in model_dict:
        raise ValueError(f"Do not find {name} in the model zoo. List of models: {list(model_dict.keys())}")
    else:
        model = model_dict[model_id](**kwargs)
    set_norm_eps(model, 1e-6)

    if pretrained:
        weight_url = weight_url or REGISTERED_SAM_MODEL.get(name, None)
        if weight_url is None:
            raise ValueError(f"Do not find the pretrained weight of {name}.")
        else:
            weight = load_state_dict_from_file(weight_url)
            model.load_state_dict(weight)
    return model
