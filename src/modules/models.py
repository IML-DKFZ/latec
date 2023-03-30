import torch
import torchvision
from torch import nn
from torchvision.models import efficientnet_b0

from modules.components.resnet import resnet50
from modules.components.deit_vit import deit_small_patch16_224


def load_from_lightning(model, model_filepath):
    # load the checkpoint
    pretrained_dict = torch.load(model_filepath)
    if "state_dict" in pretrained_dict.keys():
        pretrained_dict = pretrained_dict["state_dict"]

    pretrained_dict = {
        k.replace("model.", "").replace("module.", ""): v
        for k, v in pretrained_dict.items()
    }

    model.load_state_dict(pretrained_dict)

    del pretrained_dict


class ModelsModule:
    def __init__(self, cfg):
        modality = cfg.data.modality
        self.models = []

        if modality == "Image":
            if isinstance(cfg.data.weights_vit, bool):
                self.models.append(resnet50(weights=cfg.data.weights_resnet))
                self.models.append(efficientnet_b0(weights=cfg.data.weights_effnet))
                self.models.append(
                    deit_small_patch16_224(pretrained=cfg.data.weights_vit)
                )
            else:
                model_1 = resnet50()
                model_1.fc = nn.Linear(2048, cfg.data.num_classes, bias=True)
                load_from_lightning(model_1, cfg.data.weights_resnet)
                self.models.append(model_1)

                model_2 = efficientnet_b0()
                model_2.classifier[1] = nn.Linear(1280, cfg.data.num_classes, bias=True)
                load_from_lightning(model_1, cfg.data.weights_resnet)
                self.models.append(model_2)

                model_3 = deit_small_patch16_224(num_classes=cfg.data.num_classes)
                load_from_lightning(model_3, cfg.data.weights_vit)
                self.models.append(model_3)

        for model in self.models:
            model.eval()
