import numpy as np
from src.modules.components.base_cam import BaseCAM

# https://arxiv.org/abs/1710.11063


class GradCAMPlusPlus(BaseCAM):
    def __init__(
        self,
        model,
        target_layers,
        use_cuda=False,
        reshape_transform=None,
        include_negative=False,
    ):
        super(GradCAMPlusPlus, self).__init__(
            model,
            target_layers,
            use_cuda,
            reshape_transform,
            include_negative=include_negative,
        )

    def get_cam_weights(
        self, inputs, target_layers, target_category, activations, grads
    ):
        grads_power_2 = grads**2
        grads_power_3 = grads_power_2 * grads
        # Equation 19 in https://arxiv.org/abs/1710.11063
        if len(activations.shape[2:]) == 1:
            sum_activations = np.sum(activations, axis=2)
        else:
            sum_activations = np.sum(activations, axis=(2, 3))

        if len(activations.shape[2:]) == 1:
            sel = sum_activations[:, :, None]
        else:
            sel = sum_activations[:, :, None, None]

        eps = 0.000001
        aij = grads_power_2 / (2 * grads_power_2 + sel * grads_power_3 + eps)
        # Now bring back the ReLU from eq.7 in the paper,
        # And zero out aijs where the activations are 0
        aij = np.where(grads != 0, aij, 0)

        weights = np.maximum(grads, 0) * aij

        if len(activations.shape[2:]) == 2:
            return np.sum(weights, axis=(2, 3))
        elif len(activations.shape[2:]) == 1:
            return np.sum(weights, axis=2)
        else:
            return np.sum(weights, axis=(2, 3, 4))
