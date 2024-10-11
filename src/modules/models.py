import torch
import torchvision
from torch import nn
from torchvision.models import efficientnet_b0
from src.modules.components.resnet import resnet50
from src.modules.components.deit_vit import deit_small_patch16_224, VolumeEmbed

from efficientnet_pytorch_3d import EfficientNet3D
from torchvision.models.video import r3d_18
from timm.models.layers import trunc_normal_

from src.modules.components.pointnet import PointNet, PointNet2
from src.modules.components.dgcnn import DGCNN
from src.modules.components.pc_transformer import PCT


def load_from_lightning(model, model_filepath):
    """Load model weights from a Lightning checkpoint."""
    pretrained_dict = torch.load(
        model_filepath,
        map_location="cpu" if not torch.cuda.is_available() else "cuda:0",
    )

    # Extract state_dict if necessary and clean key names
    pretrained_dict = pretrained_dict.get("state_dict", pretrained_dict)
    pretrained_dict = {
        k.replace("model.", "")
        .replace("module.", "")
        .replace("patch_embed.proj.conv3d_1.", "patch_embed.proj."): v
        for k, v in pretrained_dict.items()
    }

    model.load_state_dict(pretrained_dict, strict=False)
    del pretrained_dict


class ModelsModule:
    def __init__(self, cfg):
        self.models = []
        modality = cfg.data.modality

        if modality == "image":
            self._initialize_image_models(cfg)
        elif modality == "volume":
            self._initialize_volume_models(cfg)
        elif modality == "point_cloud":
            self._initialize_point_cloud_models(cfg)

        for model in self.models:
            model.eval()

    def _initialize_image_models(self, cfg):
        """Initialize image-based models."""
        if isinstance(cfg.data.weights_vit, bool):  # For ImageNet
            self.models.append(resnet50(weights=cfg.data.weights_resnet))
            self.models.append(efficientnet_b0(weights=cfg.data.weights_effnet))
            self.models.append(deit_small_patch16_224(pretrained=cfg.data.weights_vit))
        else:  # For all other Image datasets
            self.models.append(self._create_resnet50(cfg))
            self.models.append(self._create_efficientnet_b0(cfg))
            self.models.append(self._create_vit(cfg))

    def _initialize_volume_models(self, cfg):
        """Initialize volume-based models."""
        self.models.append(self._create_3d_resnet(cfg))
        self.models.append(self._create_3d_efficientnet_b0(cfg))
        self.models.append(self._create_simple3dformer(cfg))

    def _initialize_point_cloud_models(self, cfg):
        """Initialize point cloud-based models."""
        self.models.append(self._create_pointnet(cfg))
        self.models.append(self._create_dgcnn(cfg))
        self.models.append(self._create_pct(cfg))

    def _create_resnet50(self, cfg):
        model = resnet50()
        model.fc = nn.Linear(2048, cfg.data.num_classes, bias=True)
        load_from_lightning(model, cfg.data.weights_resnet)
        return model

    def _create_efficientnet_b0(self, cfg):
        model = efficientnet_b0()
        model.classifier[1] = nn.Linear(1280, cfg.data.num_classes, bias=True)
        load_from_lightning(model, cfg.data.weights_effnet)
        return model

    def _create_vit(self, cfg):
        model = deit_small_patch16_224(num_classes=cfg.data.num_classes)
        load_from_lightning(model, cfg.data.weights_vit)
        return model

    def _create_3d_resnet(self, cfg):
        model = r3d_18()
        model.stem[0] = nn.Conv3d(
            1,
            64,
            kernel_size=(3, 7, 7),
            stride=(1, 2, 2),
            padding=(1, 3, 3),
            bias=False,
        )
        model.fc = nn.Linear(512, cfg.data.num_classes, bias=True)
        load_from_lightning(model, cfg.data.weights_3dresnet)
        return model

    def _create_3d_efficientnet_b0(self, cfg):
        model = EfficientNet3D.from_name(
            "efficientnet-b0",
            override_params={"num_classes": cfg.data.num_classes},
            in_channels=1,
        )
        model._conv_stem = nn.Conv3d(
            1,
            32,
            kernel_size=(3, 7, 7),
            stride=(1, 2, 2),
            padding=(22, 22, 22),
            bias=False,
        )
        load_from_lightning(model, cfg.data.weights_3deffnet)
        return model

    def _create_simple3dformer(self, cfg):
        model = deit_small_patch16_224(
            num_classes=cfg.data.num_classes, pretrained=False
        )
        model.patch_embed = VolumeEmbed(
            volume_size=28,
            cell_size=4,
            patch_size=7,
            in_chans=1,
            embed_dim=model.embed_dim,
        )
        model.num_patches = model.patch_embed.num_patches
        model.pos_embed = nn.Parameter(
            torch.zeros(1, model.patch_embed.num_patches + 1, model.embed_dim)
        )
        trunc_normal_(model.pos_embed, std=0.02)
        load_from_lightning(model, cfg.data.weights_s3dformer)
        return model

    def _create_pointnet(self, cfg):
        model = PointNet(classes=cfg.data.num_classes)
        load_from_lightning(model, cfg.data.weights_pointnet)
        return model

    def _create_dgcnn(self, cfg):
        model = DGCNN(output_channels=cfg.data.num_classes)
        load_from_lightning(model, cfg.data.weights_dgcnn)
        return model

    def _create_pct(self, cfg):
        model = PCT(output_channels=cfg.data.num_classes)
        load_from_lightning(model, cfg.data.weights_pct)
        return model
