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
from src.modules.components.insertion_deletion import Insertion, Deletion
from src.modules.components.infidelity import Infidelity
from src.utils.hidden_layer_selection import get_hidden_layer_eval


class MetricRegistry:
    def __init__(self):
        self.__metric_registry = {}

    def register_metric(self, name, hparams=None):
        """
        Registers a metric along with its hyperparameters.

        Parameters:
        - name (str): The name of the metric to register.
        """

        def inner(func):
            if name in self.__metric_registry:
                raise ValueError(f"Metric '{name}' is already registered.")
            self.__metric_registry[name] = {
                "initializer": func,
            }
            return func

        return inner

    def get_metric(self, name):
        if name not in self.__metric_registry:
            raise KeyError(f"Metric '{name}' is not registered.")
        return self.__metric_registry[name]["initializer"]

    def list_metrics(self):
        return list(self.__metric_registry.keys())


# Create an instance of the metric registry
eval_metric_registry = MetricRegistry()


# Faithfulness Metrics


@eval_metric_registry.register_metric("faithfulnesscorrelation")
def config(hparams, modality, **kwargs):
    metric = FaithfulnessCorrelation(
        nr_runs=hparams["nr_runs"],
        subset_size=hparams["subset_size"],
        perturb_baseline=hparams["perturb_baseline"],
        perturb_func=quantus.perturb_func.baseline_replacement_by_indices,
        similarity_func=quantus.similarity_func.correlation_pearson,
        abs=False,
        return_aggregate=False,
        disable_warnings=True,
        normalise=True,
    )

    return metric


@eval_metric_registry.register_metric("faithfulnessestimate")
def config(hparams, modality, **kwargs):
    metric = FaithfulnessEstimate(
        perturb_func=quantus.perturb_func.baseline_replacement_by_indices,
        similarity_func=quantus.similarity_func.correlation_pearson,
        features_in_step=hparams["features_in_step"],
        perturb_baseline=hparams["perturb_baseline"],
        disable_warnings=True,
        normalise=True,
    )

    return metric


@eval_metric_registry.register_metric("monotonicitycorrelation")
def config(hparams, modality, **kwargs):
    metric = MonotonicityCorrelation(
        nr_samples=hparams["nr_samples"],
        features_in_step=hparams["features_in_step"],
        perturb_baseline=hparams["perturb_baseline"],
        perturb_func=quantus.perturb_func.baseline_replacement_by_indices,
        similarity_func=quantus.similarity_func.correlation_spearman,
        disable_warnings=True,
        normalise=True,
    )

    return metric


@eval_metric_registry.register_metric("pixelflipping")
def config(hparams, modality, **kwargs):
    metric = PixelFlipping(
        features_in_step=hparams["features_in_step"],
        perturb_baseline=hparams["perturb_baseline"],
        perturb_func=quantus.perturb_func.baseline_replacement_by_indices,
        disable_warnings=True,
        normalise=True,
    )

    return metric


@eval_metric_registry.register_metric("regionperturbation")
def config(hparams, modality, **kwargs):
    metric = RegionPerturbation(
        patch_size=hparams["patch_size"],
        regions_evaluation=hparams["regions_evaluation"],
        perturb_baseline=hparams["perturb_baseline"],
        normalise=True,
        disable_warnings=True,
    )

    return metric


@eval_metric_registry.register_metric("insertion")
def config(hparams, modality, **kwargs):
    metric = Insertion(
        pixel_batch_size=hparams["pixel_batch_size"],
        sigma=hparams["sigma"],
        kernel_size=hparams["kernel_size"],
        modality=modality,
    )

    return metric


@eval_metric_registry.register_metric("deletion")
def config(hparams, modality, **kwargs):
    metric = Deletion(
        pixel_batch_size=hparams["pixel_batch_size"],
        modality=modality,
    )

    return metric


@eval_metric_registry.register_metric("irof")
def config(hparams, modality, **kwargs):
    metric = IROF(
        segmentation_method=hparams["segmentation_method"],
        perturb_baseline=hparams["perturb_baseline"],
        perturb_func=quantus.perturb_func.baseline_replacement_by_indices,
        return_aggregate=False,
        disable_warnings=True,
        normalise=True,
        modality=modality,
    )

    return metric


@eval_metric_registry.register_metric("road")
def config(hparams, modality, **kwargs):
    metric = ROAD(
        noise=hparams["noise"],
        perturb_func=quantus.perturb_func.noisy_linear_imputation
        if modality == "image"
        else quantus.perturb_func.gaussian_noise,
        percentages=list(range(1, hparams["percentages_max"], 2)),
        perturb_func_kwargs={"indexed_axes": (0, 1)}
        if modality == "point_cloud"
        else {"indexed_axes": (0, 1, 2)}
        if modality == "volume"
        else None,
        display_progressbar=False,
        disable_warnings=True,
        normalise=True,
    )

    return metric


@eval_metric_registry.register_metric("sufficiency")
def config(hparams, modality, **kwargs):
    metric = Sufficiency(
        threshold=hparams["threshold"],
        return_aggregate=False,
        disable_warnings=True,
        normalise=True,
        modality=modality,
    )

    return metric


@eval_metric_registry.register_metric("infidelity")
def config(hparams, modality, **kwargs):
    metric = Infidelity(
        n_perturb_samples=hparams["n_perturb_samples"],
        modality=modality,
    )

    return metric


# Robustness Metrics


@eval_metric_registry.register_metric("locallipschitzestimate")
def config(hparams, modality, **kwargs):
    metric = LocalLipschitzEstimate(
        nr_samples=hparams["nr_samples"],
        perturb_std=hparams["perturb_std"],
        perturb_mean=hparams["perturb_mean"],
        norm_numerator=quantus.similarity_func.distance_euclidean,
        norm_denominator=quantus.similarity_func.distance_euclidean,
        perturb_func=quantus.perturb_func.gaussian_noise,
        similarity_func=quantus.similarity_func.lipschitz_constant,
        disable_warnings=True,
        normalise=True,
    )

    return metric


@eval_metric_registry.register_metric("maxsensitivity")
def config(hparams, modality, **kwargs):
    metric = MaxSensitivity(
        nr_samples=hparams["nr_samples"],
        lower_bound=hparams["lower_bound"],
        norm_numerator=quantus.norm_func.fro_norm,
        norm_denominator=quantus.norm_func.fro_norm,
        perturb_func=quantus.perturb_func.uniform_noise,
        similarity_func=quantus.similarity_func.difference,
        disable_warnings=True,
        normalise=True,
    )

    return metric


@eval_metric_registry.register_metric("continuity")
def config(hparams, modality, **kwargs):
    metric = Continuity(
        patch_size=hparams["patch_size"],
        nr_steps=hparams["nr_steps"],
        perturb_baseline=hparams["perturb_baseline"],
        similarity_func=quantus.similarity_func.correlation_spearman,
        disable_warnings=True,
        normalise=True,
        modality=modality,
    )

    return metric


@eval_metric_registry.register_metric("relativeinputstability")
def config(hparams, modality, **kwargs):
    metric = RelativeInputStability(
        nr_samples=hparams["nr_samples"],
        disable_warnings=True,
        normalise=True,
        return_nan_when_prediction_changes=False,
    )

    return metric


@eval_metric_registry.register_metric("relativeoutputstability")
def config(hparams, modality, **kwargs):
    metric = RelativeOutputStability(
        nr_samples=hparams["nr_samples"],
        disable_warnings=True,
        normalise=True,
        return_nan_when_prediction_changes=False,
    )

    return metric


@eval_metric_registry.register_metric("relativerepresentationstability")
def config(hparams, modality, **kwargs):
    metric = RelativeRepresentationStability(
        nr_samples=hparams["nr_samples"],
        layer_names=kwargs["layer"],
        disable_warnings=True,
        normalise=True,
        return_nan_when_prediction_changes=False,
    )

    return metric


# Complexity Metrics


@eval_metric_registry.register_metric("sparseness")
def config(hparams, modality, **kwargs):
    metric = Sparseness(disable_warnings=True, normalise=True)

    return metric


@eval_metric_registry.register_metric("complexity")
def config(hparams, modality, **kwargs):
    metric = Complexity(disable_warnings=True, normalise=True)

    return metric


@eval_metric_registry.register_metric("effectivecomplexity")
def config(hparams, modality, **kwargs):
    metric = EffectiveComplexity(
        eps=hparams["eps"],
        disable_warnings=True,
        normalise=True,
    )

    return metric
