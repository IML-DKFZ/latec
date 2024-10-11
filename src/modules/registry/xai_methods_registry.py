import numpy as np
import torch
from captum.attr import (
    Occlusion,
    KernelShap,
    Saliency,
    InputXGradient,
    GuidedBackprop,
    IntegratedGradients,
    GradientShap,
    DeepLift,
    DeepLiftShap,
    Lime,
    LRP,
    NoiseTunnel,
)
from captum._utils.models.linear_model.model import (
    SGDLasso,
    SkLearnRidge,
)
from omegaconf.omegaconf import open_dict

from src.modules.components.score_cam import ScoreCAM
from src.modules.components.grad_cam import GradCAM
from src.modules.components.grad_cam_plusplus import GradCAMPlusPlus
from src.modules.components.attention import AttentionLRP
from src.utils.reshape_transforms import *


class XAIMethodRegistry:
    def __init__(self):
        self.__method_registry = {}

    def register_method(self, name, hparams=None):
        """
        Registers a method along with its hyperparameters.

        Parameters:
        - name (str): The name of the method to register.
        - hparams (dict, optional): The hyperparameters associated with the method.
        """

        def inner(func):
            if name in self.__method_registry:
                raise ValueError(f"Method '{name}' is already registered.")
            self.__method_registry[name] = {
                "initializer": func,
                "hparams": hparams or {},
            }
            return func

        return inner

    def get_method(self, name):
        if name not in self.__method_registry:
            raise KeyError(f"Method '{name}' is not registered.")
        return self.__method_registry[name]["initializer"]

    def get_hparams(self, name):
        if name not in self.__method_registry:
            raise KeyError(f"Method '{name}' is not registered.")
        return self.__method_registry[name]["hparams"]

    def list_methods(self):
        return list(self.__method_registry.keys())


xai_method_registry = XAIMethodRegistry()


@xai_method_registry.register_method("occlusion")
def config(hparams, model, modality, x_batch, **kwargs):
    method = Occlusion(model)

    if modality == "image":
        hparams["sliding_window_shapes"] = (
            x_batch.shape[1],
            hparams["sliding_window_shapes"],
            hparams["sliding_window_shapes"],
        )
    elif modality == "volume":
        hparams["sliding_window_shapes"] = (
            1,
            hparams["sliding_window_shapes"],
            hparams["sliding_window_shapes"],
            hparams["sliding_window_shapes"],
        )
    elif modality == "point_cloud":
        hparams["sliding_window_shapes"] = (
            hparams["sliding_window_shapes"],
            1,
        )

    return method, hparams


@xai_method_registry.register_method("lime")
def config(hparams, model, modality, x_batch, **kwargs):
    method = Lime(
        model,
        interpretable_model=SkLearnRidge(alpha=hparams["alpha"])
        if modality == "point_cloud"
        else SGDLasso(alpha=hparams["alpha"]),
    )

    del hparams["alpha"]

    hparams["feature_mask"] = feature_mask(modality)

    return method, hparams


@xai_method_registry.register_method("kernelshap")
def config(hparams, model, modality, x_batch, **kwargs):
    method = KernelShap(model)

    hparams["feature_mask"] = feature_mask(modality)

    return method, hparams


@xai_method_registry.register_method("saliency")
def config(hparams, model, modality, x_batch, **kwargs):
    method = Saliency(model)

    return method, hparams


@xai_method_registry.register_method("inputxgradient")
def config(hparams, model, modality, x_batch, **kwargs):
    method = InputXGradient(model)

    return method, hparams


@xai_method_registry.register_method("guidedbackprop")
def config(hparams, model, modality, x_batch, **kwargs):
    method = GuidedBackprop(model)

    return method, hparams


@xai_method_registry.register_method("gradcam")
def config(hparams, model, modality, x_batch, **kwargs):
    method = GradCAM(
        model,
        kwargs["layer"],
        reshape_transform=kwargs["reshape"],
        include_negative=kwargs["include_negative"],
    )

    return method, hparams


@xai_method_registry.register_method("scorecam")
def config(hparams, model, modality, x_batch, **kwargs):
    method = ScoreCAM(
        model,
        kwargs["layer"],
        reshape_transform=kwargs["reshape"],
    )

    method.batch_size = hparams["batch_size"]
    del hparams["batch_size"]

    return method, hparams


@xai_method_registry.register_method("gradcamplusplus")
def config(hparams, model, modality, x_batch, **kwargs):
    method = GradCAMPlusPlus(
        model,
        kwargs["layer"],
        reshape_transform=kwargs["reshape"],
        include_negative=kwargs["include_negative"],
    )

    return method, hparams


@xai_method_registry.register_method("integratedgradients")
def config(hparams, model, modality, x_batch, **kwargs):
    method = IntegratedGradients(model)

    return method, hparams


@xai_method_registry.register_method("expectedgradients")
def config(hparams, model, modality, x_batch, **kwargs):
    method = GradientShap(model)

    hparams["baselines"] = x_batch

    return method, hparams


@xai_method_registry.register_method("deeplift")
def config(hparams, model, modality, x_batch, **kwargs):
    method = DeepLift(model, eps=hparams["eps"])

    del hparams["eps"]

    return method, hparams


@xai_method_registry.register_method("deepliftshap")
def config(hparams, model, modality, x_batch, **kwargs):
    method = DeepLiftShap(model)

    hparams["baselines"] = x_batch if x_batch.shape[0] < 16 else x_batch[0:16]

    return method, hparams


@xai_method_registry.register_method("lrp")
def config(hparams, model, modality, x_batch, **kwargs):
    if (
        model.__class__.__name__ == "VisionTransformer"
        or model.__class__.__name__ == "PCT"
    ):
        method = AttentionLRP(model, modality=modality)
        hparams = {}
        hparams["method"] = "full"
    else:
        method = LRP(model, epsilon=hparams["eps"], gamma=hparams["gamma"])
        hparams = {}

    return method, hparams


@xai_method_registry.register_method("attention")
def config(hparams, model, modality, x_batch, **kwargs):
    if (
        model.__class__.__name__ == "VisionTransformer"
        or model.__class__.__name__ == "PCT"
    ):
        method = AttentionLRP(model, modality=modality)
        hparams["method"] = "last_layer_attn"

    else:
        method = {}

    return method, hparams


@xai_method_registry.register_method("attentionrollout")
def config(hparams, model, modality, x_batch, **kwargs):
    if (
        model.__class__.__name__ == "VisionTransformer"
        or model.__class__.__name__ == "PCT"
    ):
        method = AttentionLRP(model, modality=modality)
        hparams["method"] = "rollout"

    else:
        method = {}

    return method, hparams


@xai_method_registry.register_method("attentionlrp")
def config(hparams, model, modality, x_batch, **kwargs):
    if (
        model.__class__.__name__ == "VisionTransformer"
        or model.__class__.__name__ == "PCT"
    ):
        method = AttentionLRP(model, modality=modality)
        hparams["method"] = "transformer_attribution"

    else:
        method = {}

    return method, hparams
