import quantus
import numpy as np
import torch

from quantus import (
    FaithfulnessCorrelation,
    FaithfulnessEstimate,
    MonotonicityCorrelation,
    PixelFlipping,
    RegionPerturbation,
    IROF,
    ROAD,
    Sufficiency,
    LocalLipschitzEstimate,
    MaxSensitivity,
    Continuity,
    RelativeInputStability,
    RelativeOutputStability,
    RelativeRepresentationStability,
    Infidelity,
    Sparseness,
    Complexity,
    EffectiveComplexity,
)
from modules.components.insertion_deletion import InsertionDeletion
from quantus.functions.perturb_func import gaussian_noise
from captum.metrics import infidelity
from copy import deepcopy

# disable_warnings = True


def perturb_fn(inputs):
    noise = (
        torch.tensor(np.random.normal(0, 0.005, inputs.shape)).float().to(inputs.device)
    )
    return noise, inputs - noise


class EvalModule:
    def __init__(self, cfg, model):
        self.modality = cfg.data.modality
        self.eval_cfg = cfg.eval_method

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

        # Faithfulness
        if self.eval_cfg.FaithfulnessCorrelation:
            self.FaithfulnessCorrelation = FaithfulnessCorrelation(
                nr_runs=self.eval_cfg.fc_nr_runs,
                subset_size=self.eval_cfg.fc_subset_size,
                perturb_baseline=self.eval_cfg.fc_perturb_baseline,
                perturb_func=quantus.perturb_func.baseline_replacement_by_indices,
                similarity_func=quantus.similarity_func.correlation_pearson,
                abs=False,
                return_aggregate=False,
                disable_warnings=True,
                normalise=self.eval_cfg.normalise,
            )

        if self.eval_cfg.FaithfulnessEstimate:
            self.FaithfulnessEstimate = FaithfulnessEstimate(
                perturb_func=quantus.perturb_func.baseline_replacement_by_indices,
                similarity_func=quantus.similarity_func.correlation_pearson,
                features_in_step=self.eval_cfg.fe_features_in_step,
                perturb_baseline=self.eval_cfg.fe_perturb_baseline,
                disable_warnings=True,
                normalise=self.eval_cfg.normalise,
            )

        if self.eval_cfg.MonotonicityCorrelation:
            self.MonotonicityCorrelation = MonotonicityCorrelation(
                nr_samples=self.eval_cfg.mc_nr_samples,
                features_in_step=self.eval_cfg.mc_features_in_step,
                perturb_baseline=self.eval_cfg.mc_perturb_baseline,
                perturb_func=quantus.perturb_func.baseline_replacement_by_indices,
                similarity_func=quantus.similarity_func.correlation_spearman,
                disable_warnings=True,
                normalise=self.eval_cfg.normalise,
            )

        if self.eval_cfg.PixelFlipping:
            self.PixelFlipping = PixelFlipping(
                features_in_step=self.eval_cfg.pf_features_in_step,
                perturb_baseline=self.eval_cfg.pf_perturb_baseline,
                perturb_func=quantus.perturb_func.baseline_replacement_by_indices,
                disable_warnings=True,
                normalise=self.eval_cfg.normalise,
            )

        if self.eval_cfg.RegionPerturbation:
            self.RegionPerturbation = RegionPerturbation(
                patch_size=self.eval_cfg.rp_patch_size,
                regions_evaluation=self.eval_cfg.rp_regions_evaluation,
                perturb_baseline=self.eval_cfg.rp_perturb_baseline,
                normalise=self.eval_cfg.normalise,
                disable_warnings=True,
            )

        if self.eval_cfg.InsertionDeletion:
            self.InsertionDeletion = InsertionDeletion(
                pixel_batch_size=self.eval_cfg.id_pixel_batch_size,
                sigma=self.eval_cfg.id_sigma,
            )

        if self.eval_cfg.IROF:
            self.IROF = IROF(
                segmentation_method=self.eval_cfg.irof_segmentation_method,
                perturb_baseline=self.eval_cfg.irof_perturb_baseline,
                perturb_func=quantus.perturb_func.baseline_replacement_by_indices,
                return_aggregate=False,
                disable_warnings=True,
                normalise=self.eval_cfg.normalise,
            )

        if self.eval_cfg.ROAD:
            self.ROAD = ROAD(
                noise=self.eval_cfg.road_noise,
                perturb_func=quantus.perturb_func.gaussian_noise
                if self.modality == "Point_Cloud"
                else quantus.perturb_func.noisy_linear_imputation,
                percentages=list(range(1, self.eval_cfg.road_percentages_max, 2)),
                perturb_func_kwargs={"indexed_axes": (0, 1)}
                if self.modality == "Point_Cloud"
                else None,
                display_progressbar=False,
                disable_warnings=True,
                normalise=self.eval_cfg.normalise,
            )

        if self.eval_cfg.Sufficiency:
            self.Sufficiency = Sufficiency(
                threshold=self.eval_cfg.su_threshold,
                return_aggregate=False,
                disable_warnings=True,
                normalise=self.eval_cfg.normalise,
            )

        # Robustness
        if self.eval_cfg.LocalLipschitzEstimate:
            self.LocalLipschitzEstimate = LocalLipschitzEstimate(
                nr_samples=self.eval_cfg.lle_nr_samples,
                perturb_std=self.eval_cfg.lle_perturb_std,
                perturb_mean=self.eval_cfg.lle_perturb_mean,
                norm_numerator=quantus.similarity_func.distance_euclidean,
                norm_denominator=quantus.similarity_func.distance_euclidean,
                perturb_func=quantus.perturb_func.gaussian_noise,
                similarity_func=quantus.similarity_func.lipschitz_constant,
                disable_warnings=True,
                normalise=self.eval_cfg.normalise,
            )

        if self.eval_cfg.MaxSensitivity:
            self.MaxSensitivity = MaxSensitivity(
                nr_samples=self.eval_cfg.ms_nr_samples,
                lower_bound=self.eval_cfg.ms_lower_bound,
                norm_numerator=quantus.norm_func.fro_norm,
                norm_denominator=quantus.norm_func.fro_norm,
                perturb_func=quantus.perturb_func.uniform_noise,
                similarity_func=quantus.similarity_func.difference,
                disable_warnings=True,
                normalise=self.eval_cfg.normalise,
            )

        if self.eval_cfg.Continuity:
            self.Continuity = Continuity(
                patch_size=self.eval_cfg.co_patch_size,
                nr_steps=self.eval_cfg.co_nr_steps,
                perturb_baseline=self.eval_cfg.co_perturb_baseline,
                similarity_func=quantus.similarity_func.correlation_spearman,
                disable_warnings=True,
                normalise=self.eval_cfg.normalise,
            )

        if self.eval_cfg.RelativeInputStability:
            self.RelativeInputStability = RelativeInputStability(
                nr_samples=self.eval_cfg.ris_nr_samples,
                disable_warnings=True,
                normalise=self.eval_cfg.normalise,
            )

        if self.eval_cfg.RelativeOutputStability:
            self.RelativeOutputStability = RelativeOutputStability(
                nr_samples=self.eval_cfg.ros_nr_samples,
                disable_warnings=True,
                normalise=self.eval_cfg.normalise,
            )

        if self.eval_cfg.RelativeRepresentationStability:
            self.RelativeRepresentationStability = RelativeRepresentationStability(
                nr_samples=self.eval_cfg.rrs_nr_samples,
                layer_names=layer,
                disable_warnings=True,
                normalise=self.eval_cfg.normalise,
            )

        # if self.eval_cfg.Infidelity:
        #     self.Infidelity = Infidelity(
        #         perturb_baseline=self.eval_cfg.in_perturb_baseline,
        #         perturb_func=quantus.perturb_func.baseline_replacement_by_indices,
        #         n_perturb_samples=self.eval_cfg.in_n_perturb_samples,
        #         perturb_patch_sizes=[self.eval_cfg.in_perturb_patch_sizes],
        #         display_progressbar=False,
        #         disable_warnings=True,
        #         normalise=self.eval_cfg.normalise,
        #     )

        # Usability
        if self.eval_cfg.Sparseness:
            self.Sparseness = Sparseness(
                disable_warnings=True, normalise=self.eval_cfg.normalise
            )

        if self.eval_cfg.Complexity:
            self.Complexity = Complexity(
                disable_warnings=True, normalise=self.eval_cfg.normalise
            )

        if self.eval_cfg.EffectiveComplexity:
            self.EffectiveComplexity = EffectiveComplexity(
                eps=self.eval_cfg.ec_eps,
                disable_warnings=True,
                normalise=self.eval_cfg.normalise,
            )

    def evaluate(self, model, x_batch, y_batch, a_batch, xai_methods, count_xai):
        eval_scores = []
        # Faithfulness
        if self.eval_cfg.FaithfulnessCorrelation:
            score = self.FaithfulnessCorrelation(
                    model=model,
                    x_batch=x_batch,
                    y_batch=y_batch,
                    a_batch=a_batch,
                    device=self.eval_cfg.device,
                )
            
            eval_scores.append(deepcopy(score))

        if self.eval_cfg.FaithfulnessEstimate:
            score = self.FaithfulnessEstimate(
                    model=model,
                    x_batch=x_batch,
                    y_batch=y_batch,
                    a_batch=a_batch,
                    device=self.eval_cfg.device,
                )
            eval_scores.append(deepcopy(score))
        if self.eval_cfg.MonotonicityCorrelation:
            eval_scores.append(
                self.MonotonicityCorrelation(
                    model=model,
                    x_batch=x_batch,
                    y_batch=y_batch,
                    a_batch=a_batch,
                    device=self.eval_cfg.device,
                )
            )
        if self.eval_cfg.PixelFlipping:
            eval_scores.append(
                self.PixelFlipping(
                    model=model,
                    x_batch=x_batch,
                    y_batch=y_batch,
                    a_batch=a_batch,
                    device=self.eval_cfg.device,
                )
            )
        if self.eval_cfg.RegionPerturbation:
            eval_scores.append(
                self.RegionPerturbation(
                    model=model,
                    x_batch=x_batch,
                    y_batch=y_batch,
                    a_batch=a_batch,
                    device=self.eval_cfg.device,
                )
            )

        if self.eval_cfg.InsertionDeletion:
            ins_auc, del_auc = self.InsertionDeletion.evaluate(
                model=model, x_batch=x_batch, y_batch=y_batch, a_batch=a_batch
            )
            eval_scores.append(ins_auc)
            eval_scores.append(del_auc)

        if self.eval_cfg.IROF:
            eval_scores.append(
                self.IROF(
                    model=model,
                    x_batch=x_batch,
                    y_batch=y_batch,
                    a_batch=a_batch,
                    device=self.eval_cfg.device,
                )
            )

        if self.eval_cfg.ROAD:
            eval_scores.append(
                self.ROAD(
                    model=model,
                    x_batch=x_batch,
                    y_batch=y_batch,
                    a_batch=np.expand_dims(np.mean(a_batch, 1), 1)
                    if self.modality == "Image"
                    else a_batch,  # does not work with 3 channel attribution!
                    device=self.eval_cfg.device,
                )
            )

        if self.eval_cfg.Sufficiency:
            eval_scores.append(
                self.Sufficiency(
                    model=model,
                    x_batch=x_batch,
                    y_batch=y_batch,
                    a_batch=a_batch,
                    device=self.eval_cfg.device,
                )
            )

        # Robustness
        if self.eval_cfg.LocalLipschitzEstimate:
            eval_scores.append(
                self.LocalLipschitzEstimate(
                    model=model,
                    x_batch=x_batch,
                    y_batch=y_batch,
                    a_batch=a_batch,
                    explain_func=xai_methods.xai_methods[count_xai].attribute,
                    explain_func_kwargs=xai_methods.xai_hparams[count_xai],
                    device=self.eval_cfg.device,
                )
            )

        if self.eval_cfg.MaxSensitivity:
            eval_scores.append(
                self.MaxSensitivity(
                    model=model,
                    x_batch=x_batch,
                    y_batch=y_batch,
                    a_batch=a_batch,
                    explain_func=xai_methods.xai_methods[count_xai].attribute,
                    explain_func_kwargs=xai_methods.xai_hparams[count_xai],
                    device=self.eval_cfg.device,
                )
            )

        if self.eval_cfg.Continuity:
            eval_scores.append(
                self.Continuity(
                    model=model,
                    x_batch=x_batch,
                    y_batch=y_batch,
                    a_batch=a_batch,
                    explain_func=xai_methods.xai_methods[count_xai].attribute,
                    explain_func_kwargs=xai_methods.xai_hparams[count_xai],
                    device=self.eval_cfg.device,
                )
            )

        if self.eval_cfg.RelativeInputStability:
            eval_scores.append(
                self.RelativeInputStability(
                    model=model,
                    x_batch=x_batch,
                    y_batch=y_batch,
                    a_batch=a_batch,
                    explain_func=xai_methods.xai_methods[count_xai].attribute,
                    explain_func_kwargs=xai_methods.xai_hparams[count_xai],
                    device=self.eval_cfg.device,
                )
            )

        if self.eval_cfg.RelativeOutputStability:
            eval_scores.append(
                self.RelativeOutputStability(
                    model=model,
                    x_batch=x_batch,
                    y_batch=y_batch,
                    a_batch=a_batch,
                    explain_func=xai_methods.xai_methods[count_xai].attribute,
                    explain_func_kwargs=xai_methods.xai_hparams[count_xai],
                    device=self.eval_cfg.device,
                )
            )

        if self.eval_cfg.RelativeRepresentationStability:
            eval_scores.append(
                self.RelativeRepresentationStability(
                    model=model,
                    x_batch=x_batch,
                    y_batch=y_batch,
                    a_batch=a_batch,
                    explain_func=xai_methods.xai_methods[count_xai].attribute,
                    explain_func_kwargs=xai_methods.xai_hparams[count_xai],
                    device=self.eval_cfg.device,
                )
            )

        if self.eval_cfg.Infidelity:
            score = infidelity(
                model,
                perturb_fn,
                torch.from_numpy(x_batch.copy()).to(next(model.parameters()).device),
                torch.from_numpy(a_batch.copy()).to(next(model.parameters()).device),
                target=torch.from_numpy(y_batch.copy()).to(next(model.parameters()).device),
                n_perturb_samples=self.eval_cfg.in_n_perturb_samples,
            )
            eval_scores.append(score.detach().cpu().numpy())

        # Usability
        if self.eval_cfg.Sparseness:
            eval_scores.append(
                self.Sparseness(
                    model=model,
                    x_batch=x_batch,
                    y_batch=y_batch,
                    a_batch=a_batch,
                    device=self.eval_cfg.device,
                )
            )

        if self.eval_cfg.Complexity:
            eval_scores.append(
                self.Complexity(
                    model=model,
                    x_batch=x_batch,
                    y_batch=y_batch,
                    a_batch=a_batch,
                    device=self.eval_cfg.device,
                )
            )

        if self.eval_cfg.EffectiveComplexity:
            eval_scores.append(
                self.EffectiveComplexity(
                    model=model,
                    x_batch=x_batch,
                    y_batch=y_batch,
                    a_batch=a_batch,
                    device=self.eval_cfg.device,
                )
            )

        return eval_scores
