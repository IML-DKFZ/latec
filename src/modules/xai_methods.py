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

from src.modules.components.score_cam import ScoreCAM
from src.modules.components.grad_cam import GradCAM
from src.modules.components.grad_cam_plusplus import GradCAMPlusPlus
from src.modules.components.attention import AttentionLRP
from src.utils.reshape_transforms import *
from src.utils.hidden_layer_selection import get_hidden_layer
from src.modules.registry.xai_methods_registry import xai_method_registry


class XAIMethodsModule:
    def __init__(self, cfg, model, x_batch):
        modality = cfg.data.modality
        xai_cfg = cfg.explain_method

        self.xai_methods = []
        self.xai_hparams = []

        # Select layer, reshape, and include_negative parameters
        layer, reshape, include_negative = get_hidden_layer(model, modality)

        # Initialize XAI methods and their hyperparameters
        self._initialize_xai_methods(
            xai_cfg, model, modality, x_batch, layer, reshape, include_negative
        )

    def _initialize_xai_methods(
        self, xai_cfg, model, modality, x_batch, layer, reshape, include_negative
    ):
        """Initialize XAI methods and their hyperparameters."""
        for xai_method in xai_cfg.keys():
            method, hparams = self._create_xai_method(
                xai_method,
                xai_cfg,
                model,
                modality,
                x_batch,
                layer,
                reshape,
                include_negative,
            )

            if method:
                self.xai_methods.append(method)
                self.xai_hparams.append(hparams)

    def _create_xai_method(
        self,
        xai_method,
        xai_cfg,
        model,
        modality,
        x_batch,
        layer,
        reshape,
        include_negative,
    ):
        """Create an individual XAI method and its hyperparameters."""
        method_creator = xai_method_registry.get_method(xai_method)
        return method_creator(
            dict(xai_cfg.get(xai_method, {})).copy(),
            model,
            modality,
            x_batch,
            layer=layer,
            reshape=reshape,
            include_negative=include_negative,
        )

    def attribute(self, x, y):
        # Generate the attributes using a list comprehension
        attr = [
            method.attribute(inputs=x, target=y, **params)
            for method, params in zip(self.xai_methods, self.xai_hparams)
        ]

        # Convert attributes to numpy arrays if they are tensors
        attr_total = np.asarray(
            [
                item.detach().cpu().numpy() if torch.is_tensor(item) else item
                for item in attr
            ]
        )

        # Reorder the axes of the numpy array
        return np.moveaxis(attr_total, 0, 1)
