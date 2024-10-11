import torch
import numpy as np
from captum.metrics import infidelity


def perturb_fn(inputs):
    noise = (
        torch.tensor(np.random.normal(0, 0.005, inputs.shape)).float().to(inputs.device)
    )
    return noise, inputs - noise


class Infidelity:
    def __init__(self, n_perturb_samples, modality):
        """
        Initializes the InfidelityEvaluator class.

        Parameters:
        - model: The model to evaluate.
        - modality (str): The data modality (e.g., 'volume', 'image', etc.).
        - n_perturb_samples (int): Number of perturbation samples to use in the infidelity metric.
        """
        self.modality = modality
        self.n_perturb_samples = n_perturb_samples
        self.perturb_fn = perturb_fn

    def preprocess_inputs(self, x_batch, a_batch, y_batch, model):
        """
        Prepares the input data for the infidelity metric based on the modality.

        Parameters:
        - x_batch (np.ndarray): The batch of input data.
        - a_batch (np.ndarray): The batch of attribution data.
        - y_batch (np.ndarray): The batch of target labels.

        Returns:
        - Tuple[torch.Tensor, torch.Tensor, torch.Tensor]: Processed input tensors.
        """
        device = next(model.parameters()).device
        x_tensor = torch.from_numpy(x_batch.copy()).to(device)
        a_tensor = torch.from_numpy(a_batch.copy()).to(device)
        y_tensor = torch.from_numpy(y_batch.copy()).to(device)

        if self.modality == "volume":
            x_tensor = x_tensor.unsqueeze(1)  # Add channel dimension for 3D data
            a_tensor = a_tensor.unsqueeze(1)

        return x_tensor, a_tensor, y_tensor

    def __call__(self, model, x_batch, a_batch, y_batch, **kwargs):
        """
        Evaluates the infidelity score using the Captum library.

        Parameters:
        - x_batch (np.ndarray): The batch of input data.
        - a_batch (np.ndarray): The batch of attribution data.
        - y_batch (np.ndarray): The batch of target labels.
        - perturb_fn (function): The perturbation function used by the infidelity metric.

        Returns:
        - float: The infidelity score.
        """
        x_tensor, a_tensor, y_tensor = self.preprocess_inputs(
            x_batch, a_batch, y_batch, model
        )

        # Compute the infidelity score
        score = infidelity(
            model,
            self.perturb_fn,
            x_tensor,
            a_tensor,
            target=y_tensor,
            n_perturb_samples=self.n_perturb_samples,
        )

        return score.detach().cpu().numpy()
