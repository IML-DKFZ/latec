import numpy as np
from src.modules.components.base_cam import BaseCAM


class GradCAM(BaseCAM):
    def __init__(
        self,
        model,
        target_layers,
        use_cuda=False,
        reshape_transform=None,
        include_negative=False,
    ):
        super(GradCAM, self).__init__(
            model,
            target_layers,
            use_cuda,
            reshape_transform,
            include_negative=include_negative,
        )

    def get_cam_weights(
        self, inputs, target_layer, target_category, activations, grads
    ):
        if len(activations.shape[2:]) == 2:
            return np.mean(grads, axis=(2, 3))
        elif len(activations.shape[2:]) == 1:
            return np.mean(grads, axis=2)
        else:
            return np.mean(grads, axis=(2, 3, 4))
