import torch
import torchvision
from torchvision.models import efficientnet_b0, vit_b_16, convnext_base, densenet161

from modules.components.resnet import resnet50
from modules.components.deit_vit import deit_small_patch16_224


class ModelsModule:
    def __init__(self, cfg):
        modality = cfg.data.modality
        self.models = []

        if modality == "Image":

            self.models.append(resnet50(weights=cfg.data.weights_resnet))
            self.models.append(efficientnet_b0(weights=cfg.data.weights_effnet))
            self.models.append(
                deit_small_patch16_224(pretrained=cfg.data.weights_effnet)
            )

        for model in self.models:
            model.eval()
