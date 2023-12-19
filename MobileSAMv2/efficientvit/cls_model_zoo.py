# EfficientViT: Multi-Scale Linear Attention for High-Resolution Dense Prediction
# Han Cai, Junyan Li, Muyan Hu, Chuang Gan, Song Han
# International Conference on Computer Vision (ICCV), 2023

from efficientvit.models.efficientvit import (
    EfficientViTCls,
    efficientvit_cls_b0,
    efficientvit_cls_b1,
    efficientvit_cls_b2,
    efficientvit_cls_b3,
    efficientvit_cls_l1,
    efficientvit_cls_l2,
    efficientvit_cls_l3,
)
from efficientvit.models.nn.norm import set_norm_eps
from efficientvit.models.utils import load_state_dict_from_file

__all__ = ["create_cls_model"]


REGISTERED_CLS_MODEL: dict[str, str] = {
    "b0-r224": "assets/checkpoints/cls/b0-r224.pt",
    ###############################################
    "b1-r224": "assets/checkpoints/cls/b1-r224.pt",
    "b1-r256": "assets/checkpoints/cls/b1-r256.pt",
    "b1-r288": "assets/checkpoints/cls/b1-r288.pt",
    ###############################################
    "b2-r224": "assets/checkpoints/cls/b2-r224.pt",
    "b2-r256": "assets/checkpoints/cls/b2-r256.pt",
    "b2-r288": "assets/checkpoints/cls/b2-r288.pt",
    ###############################################
    "b3-r224": "assets/checkpoints/cls/b3-r224.pt",
    "b3-r256": "assets/checkpoints/cls/b3-r256.pt",
    "b3-r288": "assets/checkpoints/cls/b3-r288.pt",
    ###############################################
    "l1-r224": "assets/checkpoints/cls/l1-r224.pt",
    ###############################################
    "l2-r224": "assets/checkpoints/cls/l2-r224.pt",
    "l2-r256": "assets/checkpoints/cls/l2-r256.pt",
    "l2-r288": "assets/checkpoints/cls/l2-r288.pt",
    "l2-r320": "assets/checkpoints/cls/l2-r320.pt",
    "l2-r384": "assets/checkpoints/cls/l2-r384.pt",
    ###############################################
    "l3-r224": "assets/checkpoints/cls/l3-r224.pt",
    "l3-r256": "assets/checkpoints/cls/l3-r256.pt",
    "l3-r288": "assets/checkpoints/cls/l3-r288.pt",
    "l3-r320": "assets/checkpoints/cls/l3-r320.pt",
    "l3-r384": "assets/checkpoints/cls/l3-r384.pt",
}


def create_cls_model(name: str, pretrained=True, weight_url: str or None = None, **kwargs) -> EfficientViTCls:
    model_dict = {
        "b0": efficientvit_cls_b0,
        "b1": efficientvit_cls_b1,
        "b2": efficientvit_cls_b2,
        "b3": efficientvit_cls_b3,
        #########################
        "l1": efficientvit_cls_l1,
        "l2": efficientvit_cls_l2,
        "l3": efficientvit_cls_l3,
    }

    model_id = name.split("-")[0]
    if model_id not in model_dict:
        raise ValueError(f"Do not find {name} in the model zoo. List of models: {list(model_dict.keys())}")
    else:
        model = model_dict[model_id](**kwargs)
    if model_id in ["l1", "l2", "l3"]:
        set_norm_eps(model, 1e-7)

    if pretrained:
        weight_url = weight_url or REGISTERED_CLS_MODEL.get(name, None)
        if weight_url is None:
            raise ValueError(f"Do not find the pretrained weight of {name}.")
        else:
            weight = load_state_dict_from_file(weight_url)
            model.load_state_dict(weight)
    return model
