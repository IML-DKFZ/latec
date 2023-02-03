import torch
import torchvision
from torchvision.models import (
    efficientnet_b0,
    vit_b_16,
    convnext_base,
    densenet161,
)

from modules.components.resnet import resnet50


class ModelsModule:
    def __init__(self, cfg):
        modality = cfg.data.modality

        if modality == "Image":
            self.model_1 = resnet50(weights=cfg.data.weights_resnet)
            self.model_2 = efficientnet_b0(weights=cfg.data.weights_effnet)
            self.model_3 = torch.hub.load(
                "facebookresearch/deit:main", cfg.data.weights_vit, pretrained=True
            )

        self.model_1.eval()
        self.model_2.eval()
        self.model_3.eval()

