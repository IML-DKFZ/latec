import torch
from src.modules.components.base_cam import BaseCAM
import numpy as np


class ScoreCAM(BaseCAM):
    def __init__(self, model, target_layers, use_cuda=False, reshape_transform=None):
        super(ScoreCAM, self).__init__(
            model,
            target_layers,
            use_cuda,
            reshape_transform=reshape_transform,
            uses_gradients=False,
        )

    def get_cam_weights(self, inputs, target_layer, targets, activations, grads):
        with torch.no_grad():
            activation_tensor = torch.from_numpy(activations).to(inputs.device)

            upsampled = torch.nn.functional.interpolate(
                activation_tensor, size=inputs.shape[2:], mode="nearest"
            )

            maxs = upsampled.view(upsampled.size(0), upsampled.size(1), -1).max(dim=-1)[
                0
            ]
            mins = upsampled.view(upsampled.size(0), upsampled.size(1), -1).min(dim=-1)[
                0
            ]

            if len(inputs.shape[2:]) == 2:
                maxs, mins = maxs[:, :, None, None], mins[:, :, None, None]
            elif len(inputs.shape[2:]) == 1:
                maxs, mins = maxs[:, :, None], mins[:, :, None]
            else:
                maxs, mins = maxs[:, :, None, None, None], mins[:, :, None, None, None]

            upsampled = (upsampled - mins) / ((maxs - mins) + 1e-7)

            if len(inputs.shape[2:]) == 1:
                input_tensors = inputs[:, None, :] * upsampled[:, :, None, :]
            else:
                input_tensors = inputs[:, None, :, :] * upsampled[:, :, None, :, :]

            if hasattr(self, "batch_size"):
                BATCH_SIZE = self.batch_size
            else:
                BATCH_SIZE = 16

            scores = []
            for target, tensor in zip(targets, input_tensors):
                for i in range(0, tensor.size(0), BATCH_SIZE):
                    batch = tensor[i : i + BATCH_SIZE, :]
                    outputs = [target(o).cpu().item() for o in self.model(batch)]
                    scores.extend(outputs)
            scores = torch.Tensor(scores)
            scores = scores.view(activations.shape[0], activations.shape[1])
            weights = np.copy(torch.nn.Softmax(dim=-1)(scores).numpy())
            return weights
