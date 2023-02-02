import torch
import torchvision
from torchvision.models import (
    resnet50,
    efficientnet_b0,
    vit_b_16,
    convnext_base,
    densenet161,
)


class ModelsModule:
    def __init__(self, cfg):
        modality = cfg.data.modality

        if modality == "Image":
            self.model_1 = convnext_base(weights=cfg.data.weights_cnext)
            self.model_2 = efficientnet_b0(weights=cfg.data.weights_effnet)
            self.model_3 = torch.hub.load(
                "facebookresearch/deit:main", cfg.data.weights_vit, pretrained=True
            )

        self.model_1.eval()
        self.model_2.eval()
        self.model_3.eval()

