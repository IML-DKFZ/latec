import torch
import torchvision
from torch import nn

from torchvision.models import efficientnet_b0
from modules.components.resnet import resnet50
from modules.components.deit_vit import deit_small_patch16_224, VoxelEmbed

from efficientnet_pytorch_3d import EfficientNet3D
from torchvision.models.video import r3d_18
from timm.models.layers import trunc_normal_

from modules.components.pointnet import PointNet, PointNet2
from modules.components.dgcnn import DGCNN
from modules.components.pc_transformer import PCT


def load_from_lightning(model, model_filepath):
    # load the checkpoint
    pretrained_dict = torch.load(model_filepath)
    if "state_dict" in pretrained_dict.keys():
        pretrained_dict = pretrained_dict["state_dict"]

    pretrained_dict = {
        k.replace("model.", "").replace("module.", ""): v
        for k, v in pretrained_dict.items()
    }

    pretrained_dict = {
        k.replace("patch_embed.proj.conv3d_1.", "patch_embed.proj."): v
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
                #### ResNet 50 ####
                model_1 = resnet50()
                model_1.fc = nn.Linear(2048, cfg.data.num_classes, bias=True)
                load_from_lightning(model_1, cfg.data.weights_resnet)
                self.models.append(model_1)

                #### EfficientNet B0 ####
                model_2 = efficientnet_b0()
                model_2.classifier[1] = nn.Linear(1280, cfg.data.num_classes, bias=True)
                load_from_lightning(model_2, cfg.data.weights_effnet)
                self.models.append(model_2)

                #### ViT (Small) ####
                model_3 = deit_small_patch16_224(num_classes=cfg.data.num_classes)
                load_from_lightning(model_3, cfg.data.weights_vit)
                self.models.append(model_3)

        if modality == "Voxel":
            #### 3D ResNet 18 ####
            model_1 = r3d_18()
            model_1.stem[0] = nn.Conv3d(
                1,
                64,
                kernel_size=(3, 7, 7),
                stride=(1, 2, 2),
                padding=(1, 3, 3),
                bias=False,
            )
            model_1.fc = nn.Linear(512, cfg.data.num_classes, bias=True)

            load_from_lightning(model_1, cfg.data.weights_3dresnet)
            self.models.append(model_1)

            #### 3D EfficientNet B0 ####
            model_2 = EfficientNet3D.from_name(
                "efficientnet-b0",
                override_params={"num_classes": cfg.data.num_classes},
                in_channels=1,
            )
            model_2._conv_stem = nn.Conv3d(
                1,
                32,
                kernel_size=(3, 7, 7),
                stride=(1, 2, 2),
                padding=(22, 22, 22),
                bias=False,
            )
            load_from_lightning(model_2, cfg.data.weights_3deffnet)
            self.models.append(model_2)

            #### Simple3DFormer ####
            model_3 = deit_small_patch16_224(
                num_classes=cfg.data.num_classes, pretrained=False
            )

            # replace patch_embed layer
            model_3.patch_embed = VoxelEmbed(
                voxel_size=28,
                cell_size=4,
                patch_size=7,
                in_chans=1,
                embed_dim=model_3.embed_dim,
            )
            # change positional encoding
            model_3.num_patches = model_3.patch_embed.num_patches
            model_3.pos_embed = nn.Parameter(
                torch.zeros(1, model_3.patch_embed.num_patches + 1, model_3.embed_dim)
            )
            trunc_normal_(model_3.pos_embed, std=0.02)

            load_from_lightning(model_3, cfg.data.weights_s3dformer)
            self.models.append(model_3)

        if modality == "Point_Cloud":
            #### PointNet(++) ####
            model_1 = PointNet(classes=cfg.data.num_classes)
            load_from_lightning(model_1, cfg.data.weights_pointnet)
            self.models.append(model_1)

            #### DGCNN ####
            model_2 = DGCNN(output_channels=cfg.data.num_classes)
            load_from_lightning(model_2, cfg.data.weights_dgcnn)
            self.models.append(model_2)

            #### PCT ####
            model_3 = PCT(output_channels=cfg.data.num_classes)
            load_from_lightning(model_3, cfg.data.weights_pct)
            self.models.append(model_3)

        for model in self.models:
            model.eval()
