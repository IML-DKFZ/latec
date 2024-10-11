from src.utils.reshape_transforms import *


def get_hidden_layer(model, modality):
    if model.__class__.__name__ == "ResNet":
        layer = [model.layer4[-1]]
        reshape = None
        include_negative = False
    elif model.__class__.__name__ == "EfficientNet":
        layer = [model.features[-1]]
        reshape = None
        include_negative = False
    elif model.__class__.__name__ == "VisionTransformer":
        layer = [model.blocks[-1].norm1]
        reshape = reshape_transform_2D if modality == "image" else reshape_transform_3D
        include_negative = False
    elif model.__class__.__name__ == "EfficientNet3D":
        layer = [model._blocks[-13]]
        reshape = None
        include_negative = False
    elif model.__class__.__name__ == "VideoResNet":
        layer = [model.layer3]
        reshape = None
        include_negative = False
    elif model.__class__.__name__ == "PointNet":
        layer = [model.transform.bn1]
        reshape = None
        include_negative = False
    elif model.__class__.__name__ == "DGCNN":
        layer = [model.conv5]
        reshape = None
        include_negative = False
    elif model.__class__.__name__ == "PCT":
        layer = [model.pt_last.sa4.after_norm]
        reshape = reshape_transform_2D
        include_negative = True

    return layer, reshape, include_negative


def get_hidden_layer_eval(model):
    if model.__class__.__name__ == "ResNet":
        layer = ["layer4.1.conv2"]
    elif model.__class__.__name__ == "EfficientNet":
        layer = ["features.8.0"]
    elif model.__class__.__name__ == "VisionTransformer":
        layer = ["blocks.11.norm1"]
    elif model.__class__.__name__ == "EfficientNet3D":
        layer = ["_blocks.15._expand_conv"]
    elif model.__class__.__name__ == "VideoResNet":
        layer = ["layer4.0.conv1.1"]
    elif model.__class__.__name__ == "PointNet":
        layer = ["transform.bn1"]
    elif model.__class__.__name__ == "DGCNN":
        layer = ["linear1"]
    elif model.__class__.__name__ == "PCT":
        layer = ["linear1"]

    return layer
