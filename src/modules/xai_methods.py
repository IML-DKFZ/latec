import numpy as np
import torch
from captum.attr import (
    FeaturePermutation,
    FeatureAblation,
    Occlusion,
    Lime,
    KernelShap,
    Saliency,
    InputXGradient,
    GuidedBackprop,
    IntegratedGradients,
    GradientShap,
    DeepLift,
    DeepLiftShap,
)
from captum._utils.models.linear_model import SkLearnLinearModel
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

from modules.components.lrp import LRP
from modules.components.score_cam import ScoreCAM
from modules.components.grad_cam import GradCAM
from modules.components.grad_cam_plusplus import GradCAMPlusPlus
from modules.components.attention import AttentionLRP


def reshape_transform(tensor, height=14, width=14):
    result = tensor[:, 1:, :].reshape(tensor.size(0), height, width, tensor.size(2))

    # Bring the channels to the first dimension,
    # like in CNNs.
    result = result.transpose(2, 3).transpose(1, 2)
    return result


def rescale_attention(tensor, height=14, width=14):
    atten = tensor.reshape(1, 1, height, width)
    atten = torch.nn.functional.interpolate(atten, scale_factor=16, mode="bilinear")
    atten = atten.reshape(224, 224).detach().numpy()
    atten = (atten - atten.min()) / (atten.max() - atten.min())
    return atten


class XAIMethodsModule:
    def __init__(self, cfg, model, x_batch):
        self.modality = cfg.data.modality
        self.xai_cfg = cfg.xai_method
        self.x_batch = x_batch
        self.model = model

        if model.__class__.__name__ == "ResNet":
            layer = [model.layer4[-1]]
            reshap = None
            include_negative = False
        elif model.__class__.__name__ == "EfficientNet":
            layer = [model.features[-1]]  # find optimal layers
            reshap = None
            include_negative = False
        else:
            layer = [model.blocks[-1].norm1]
            reshap = reshape_transform
            include_negative = True

        if self.modality == "Image":
            if self.xai_cfg.feature_permutation:
                self.fp = FeaturePermutation(model)
            if self.xai_cfg.feature_ablatation:
                self.fa = FeatureAblation(model)
            if self.xai_cfg.occlusion:
                self.occ = Occlusion(model)
            if self.xai_cfg.lime:
                self.lime = Lime(
                    model,
                    interpretable_model=SkLearnLinearModel(
                        "linear_model.Lasso", alpha=self.xai_cfg.lime_alpha
                    ),
                )
                self.ks = KernelShap(model)
            if self.xai_cfg.saliency:
                self.sa = Saliency(model)
            if self.xai_cfg.input_x_gradient:
                self.ixg = InputXGradient(model)
            if self.xai_cfg.guided_backprob:
                self.gb = GuidedBackprop(model)
            if self.xai_cfg.gcam:
                self.gcam = GradCAM(
                    model,
                    layer,
                    reshape_transform=reshap,
                    include_negative=include_negative,
                )
            if self.xai_cfg.scam:
                self.scam = ScoreCAM(model, layer, reshape_transform=reshap)
                self.scam.batch_size = self.xai_cfg.scam_batch_size
            if self.xai_cfg.gcampp:
                self.gcampp = GradCAMPlusPlus(
                    model,
                    layer,
                    reshape_transform=reshap,
                    include_negative=include_negative,
                )
            if self.xai_cfg.ig:
                self.ig = IntegratedGradients(model)
            if self.xai_cfg.eg:
                self.eg = GradientShap(model)
            if self.xai_cfg.deeplift:
                self.dl = DeepLift(model, eps=self.xai_cfg.dl_eps)
            if self.xai_cfg.deeplift_shap:
                self.dlshap = DeepLiftShap(model)
            if self.xai_cfg.lrp or self.xai_cfg.attention:
                if model.__class__.__name__ == "VisionTransformer":
                    self.lrp = AttentionLRP(model)
                else:
                    self.lrp = LRP(model, epsilon=self.xai_cfg.lrp_eps)


    def attribute_batch(self, x_batch, y_batch):
        "Attribution methods working on the full batch of observations"
        attr = []

        if self.modality == "Image":
            if self.xai_cfg.feature_permutation:
                attr.append(self.fp.attribute(x_batch, target=y_batch))
            if self.xai_cfg.feature_ablatation:
                attr.append(
                    self.fa.attribute(
                        x_batch, target=y_batch, baselines=self.xai_cfg.fa_baselines
                    )
                )
            if self.xai_cfg.gcam:
                attr_gcam = self.gcam(
                    input_tensor=x_batch,
                    targets=[ClassifierOutputTarget(y) for y in y_batch],
                )
                attr.append(np.repeat(np.expand_dims(attr_gcam, 1), 3, axis=1))  # /3 ?
            if self.xai_cfg.scam:
                attr_scam = self.scam(
                    input_tensor=x_batch,
                    targets=[ClassifierOutputTarget(y) for y in y_batch],
                )
                attr.append(np.repeat(np.expand_dims(attr_scam, 1), 3, axis=1))  # /3 ?
            if self.xai_cfg.gcampp:
                attr_gcampp = self.gcampp(
                    input_tensor=x_batch,
                    targets=[ClassifierOutputTarget(y) for y in y_batch],
                )
                attr.append(
                    np.repeat(np.expand_dims(attr_gcampp, 1), 3, axis=1)
                )  # /3 ?

        if attr is not None:
            attr_total = np.asarray(
                [i.detach().numpy() if torch.is_tensor(i) else i for i in attr]
            )
            attr_total = np.swapaxes(attr_total, 0, 1)
        else:
            attr_total = attr

        return attr_total

    def attribute_single(self, x, y):
        "Attribution methods working on single observations"
        attr = []

        if self.modality == "Image":
            if self.xai_cfg.occlusion:
                attr.append(
                    self.occ.attribute(
                        x,
                        target=y,
                        strides=self.xai_cfg.occ_strides,
                        sliding_window_shapes=(
                            x.shape[1],
                            self.xai_cfg.occ_sliding_window_shapes,
                            self.xai_cfg.occ_sliding_window_shapes,
                        ),
                        baselines=self.xai_cfg.occ_baselines,
                    )
                )
            if self.xai_cfg.lime:
                attr.append(
                    self.lime.attribute(
                        x,
                        target=y,
                        n_samples=self.xai_cfg.lime_n_samples,
                        perturbations_per_eval=self.xai_cfg.lime_perturbations_per_eval,
                    )
                )
            if self.xai_cfg.kernel_shap:
                attr.append(
                    self.ks.attribute(
                        x,
                        target=y,
                        baselines=self.xai_cfg.ks_baselines,
                        n_samples=self.xai_cfg.ks_n_samples,
                        perturbations_per_eval=self.xai_cfg.ks_perturbations_per_eval,
                    )
                )
            if self.xai_cfg.saliency:
                attr.append(self.sa.attribute(x, target=y))
            if self.xai_cfg.input_x_gradient:
                attr.append(self.ixg.attribute(x, target=y))
            if self.xai_cfg.guided_backprob:
                attr.append(self.gb.attribute(x, target=y))
            if self.xai_cfg.ig:
                attr.append(
                    self.ig.attribute(
                        x,
                        target=y,
                        baselines=self.xai_cfg.ig_baselines,
                        n_steps=self.xai_cfg.ig_n_steps,
                    )
                )
            if self.xai_cfg.eg:
                attr.append(
                    self.eg.attribute(
                        x,
                        target=y,
                        baselines=self.x_batch,
                        n_samples=self.xai_cfg.eg_n_samples,
                        stdevs=self.xai_cfg.eg_stdevs,
                    )
                )
            if self.xai_cfg.deeplift:
                attr.append(
                    self.dl.attribute(x, target=y, baselines=self.xai_cfg.dl_baselines)
                )
            if self.xai_cfg.deeplift_shap:
                attr.append(self.dlshap.attribute(x, target=y, baselines=self.x_batch))

            if self.xai_cfg.lrp:
                if self.model.__class__.__name__ == "VisionTransformer":
                    attr_lrp = self.lrp.generate_LRP(x, method="full", index=y)
                    attr.append(
                        np.repeat(
                            np.expand_dims(attr_lrp.detach().numpy(), 1), 3, axis=1
                        )
                    )
                else:
                    attr.append(self.lrp.attribute(x, target=y))

            if self.xai_cfg.attention:
                if self.model.__class__.__name__ == "VisionTransformer":
                    atten_raw = self.lrp.generate_LRP(x, method="last_layer_attn", index=y)
                    attr.append(
                        np.repeat(
                            np.expand_dims(rescale_attention(atten_raw), (0,1)), 3, axis=1
                        )
                    )
                else:
                    attr.append(np.zeros((1, 3, 224, 224)))

            if self.xai_cfg.attention:
                if self.model.__class__.__name__ == "VisionTransformer":
                    atten_roll = self.lrp.generate_LRP(x, method="rollout", index=y)
                    attr.append(
                        np.repeat(
                            np.expand_dims(rescale_attention(atten_roll), (0,1)), 3, axis=1
                        )
                    )
                else:
                    attr.append(np.zeros((1, 3, 224, 224)))

            if self.xai_cfg.attention:
                if self.model.__class__.__name__ == "VisionTransformer":
                    atten_lrp = self.lrp.generate_LRP(x, method="transformer_attribution", index=y)
                    attr.append(
                        np.repeat(
                            np.expand_dims(rescale_attention(atten_lrp), (0,1)), 3, axis=1
                        )
                    )
                else:
                    attr.append(np.zeros((1, 3, 224, 224)))

        if attr is not None:
            attr_total = np.asarray(
                [i.detach().numpy() if torch.is_tensor(i) else i for i in attr]
            )
            attr_total = np.expand_dims(attr_total.squeeze(), 0)
        else:
            attr_total = attr

        return attr_total
