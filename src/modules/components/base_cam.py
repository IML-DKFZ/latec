import numpy as np
import torch
import ttach as tta
import cv2
from typing import Callable, List, Tuple
from pytorch_grad_cam.activations_and_gradients import ActivationsAndGradients
from pytorch_grad_cam.utils.svd_on_activations import get_2d_projection
from pytorch_grad_cam.utils.image import scale_cam_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget


class BaseCAM:
    def __init__(
        self,
        model: torch.nn.Module,
        target_layers: List[torch.nn.Module],
        use_cuda: bool = False,
        reshape_transform: Callable = None,
        compute_input_gradient: bool = False,
        uses_gradients: bool = True,
        include_negative: bool = False,
    ) -> None:
        self.model = model.eval()
        self.target_layers = target_layers
        self.include_negative = include_negative
        self.cuda = use_cuda
        if self.cuda:
            self.model = model.cuda()
        self.reshape_transform = reshape_transform
        self.compute_input_gradient = compute_input_gradient
        self.uses_gradients = uses_gradients
        self.activations_and_grads = ActivationsAndGradients(
            self.model, target_layers, reshape_transform
        )

    """ Get a vector of weights for every channel in the target layer.
        Methods that return weights channels,
        will typically need to only implement this function. """

    def get_cam_weights(
        self,
        inputs: torch.Tensor,
        target_layers: List[torch.nn.Module],
        targets: List[torch.nn.Module],
        activations: torch.Tensor,
        grads: torch.Tensor,
    ) -> np.ndarray:
        raise Exception("Not Implemented")

    def get_cam_image(
        self,
        inputs: torch.Tensor,
        target_layer: torch.nn.Module,
        targets: List[torch.nn.Module],
        activations: torch.Tensor,
        grads: torch.Tensor,
        eigen_smooth: bool = False,
    ) -> np.ndarray:
        weights = self.get_cam_weights(
            inputs, target_layer, targets, activations, grads
        )

        if len(activations.shape[2:]) == 2:
            weighted_activations = weights[:, :, None, None] * activations
        elif len(activations.shape[2:]) == 1:
            weighted_activations = weights[:, :, None] * activations
        else:
            weighted_activations = weights[:, :, None, None, None] * activations

        if eigen_smooth:
            cam = get_2d_projection(weighted_activations)
        else:
            cam = weighted_activations.sum(axis=1)
        return cam

    def attribute(
        self,
        inputs: torch.Tensor,
        target: List[torch.nn.Module],
        eigen_smooth: bool = False,
    ) -> np.ndarray:
        if self.cuda:
            inputs = inputs.cuda()

        if self.compute_input_gradient:
            inputs = torch.autograd.Variable(inputs, requires_grad=True)

        outputs = self.activations_and_grads(inputs)
        if target is None:
            target_categories = np.argmax(outputs.cpu().data.numpy(), axis=-1)
            target = [ClassifierOutputTarget(category) for category in target]
        elif len(target.shape) == 0:
            target = [ClassifierOutputTarget(target)]
        else:
            target = [ClassifierOutputTarget(category) for category in target]

        if self.uses_gradients:
            self.model.zero_grad()
            loss = sum([target(output) for target, output in zip(target, outputs)])
            loss.backward(retain_graph=True)

        # In most of the saliency attribution papers, the saliency is
        # computed with a single target layer.
        # Commonly it is the last convolutional layer.
        # Here we support passing a list with multiple target layers.
        # It will compute the saliency image for every image,
        # and then aggregate them (with a default mean aggregation).
        # This gives you more flexibility in case you just want to
        # use all conv layers for example, all Batchnorm layers,
        # or something else.
        cam_per_layer = self.compute_cam_per_layer(inputs, target, eigen_smooth)
        output = self.aggregate_multi_layers(cam_per_layer)

        if len(inputs.shape[2:]) == 2 or len(inputs.shape[2:]) == 1:
            return np.repeat(np.expand_dims(output, 1), 3, axis=1)
        else:
            return np.expand_dims(output, 1)

    def get_target_width_height(self, inputs: torch.Tensor) -> Tuple[int, int]:
        if len(inputs.shape[2:]) == 2:
            return inputs.size(-1), inputs.size(-2)
        elif len(inputs.shape[2:]) == 1:
            return (inputs.size(-1),)
        else:
            return inputs.size(-1), inputs.size(-2), inputs.size(-3)

    def compute_cam_per_layer(
        self, inputs: torch.Tensor, targets: List[torch.nn.Module], eigen_smooth: bool
    ) -> np.ndarray:
        activations_list = [
            a.cpu().data.numpy() for a in self.activations_and_grads.activations
        ]
        grads_list = [
            g.cpu().data.numpy() for g in self.activations_and_grads.gradients
        ]
        target_size = self.get_target_width_height(inputs)

        cam_per_target_layer = []
        # Loop over the saliency image from every layer
        for i in range(len(self.target_layers)):
            target_layer = self.target_layers[i]
            layer_activations = None
            layer_grads = None
            if i < len(activations_list):
                layer_activations = activations_list[i]
            if i < len(grads_list):
                layer_grads = grads_list[i]

            cam = self.get_cam_image(
                inputs,
                target_layer,
                targets,
                layer_activations,
                layer_grads,
                eigen_smooth,
            )

            if self.include_negative == True:
                cam = np.abs(cam)

            cam = np.maximum(cam, 0)

            result = []
            for img in cam:  # Removed min max rescaling
                if target_size is not None:
                    if len(target_size) == 2:
                        img = cv2.resize(img, target_size)
                    else:
                        img = torch.nn.functional.interpolate(
                            torch.Tensor(img).unsqueeze(0).unsqueeze(0),
                            size=target_size,
                            mode="nearest",
                        )
                        img = img.squeeze().numpy()
                result.append(img)

            scaled = np.float32(result)

            cam_per_target_layer.append(scaled[:, None, :])

        return cam_per_target_layer

    def aggregate_multi_layers(self, cam_per_target_layer: np.ndarray) -> np.ndarray:
        cam_per_target_layer = np.concatenate(cam_per_target_layer, axis=1)
        if self.include_negative == False:
            cam_per_target_layer = np.maximum(cam_per_target_layer, 0)
        result = np.mean(cam_per_target_layer, axis=1)
        return scale_cam_image(result)

    def forward_augmentation_smoothing(
        self,
        inputs: torch.Tensor,
        targets: List[torch.nn.Module],
        eigen_smooth: bool = False,
    ) -> np.ndarray:
        transforms = tta.Compose(
            [tta.HorizontalFlip(), tta.Multiply(factors=[0.9, 1, 1.1])]
        )
        cams = []
        for transform in transforms:
            augmented_tensor = transform.augment_image(inputs)
            cam = self.attribute(augmented_tensor, targets, eigen_smooth)

            # The ttach library expects a tensor of size BxCxHxW
            cam = cam[:, None, :, :]
            cam = torch.from_numpy(cam)
            cam = transform.deaugment_mask(cam)

            # Back to numpy float32, HxW
            cam = cam.numpy()
            cam = cam[:, 0, :, :]
            cams.append(cam)

        cam = np.mean(np.float32(cams), axis=0)
        return cam

    def __call__(
        self,
        inputs: torch.Tensor,
        targets: List[torch.nn.Module] = None,
        aug_smooth: bool = False,
        eigen_smooth: bool = False,
    ) -> np.ndarray:
        # Smooth the CAM result with test time augmentation
        if aug_smooth is True:
            return self.forward_augmentation_smoothing(inputs, targets, eigen_smooth)

        return self.attribute(inputs, targets, eigen_smooth)

    def __del__(self):
        self.activations_and_grads.release()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, exc_tb):
        self.activations_and_grads.release()
        if isinstance(exc_value, IndexError):
            # Handle IndexError here...
            print(
                f"An exception occurred in CAM with block: {exc_type}. Message: {exc_value}"
            )
            return True
