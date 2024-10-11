import torch
import numpy as np


def reshape_transform_3D(tensor, height=7, width=7, z=7):
    """
    Reshapes a 3D tensor into a format suitable for processing in CNNs.

    Parameters:
    - tensor (torch.Tensor): The input tensor to be reshaped, typically from a model's attention or feature map.
    - height (int, optional): The height dimension for the reshaped tensor. Default is 7.
    - width (int, optional): The width dimension for the reshaped tensor. Default is 7.
    - z (int, optional): The depth dimension for the reshaped tensor. Default is 7.

    Returns:
    - torch.Tensor: The reshaped tensor with channels moved to the first dimension, ready for CNN processing.
    """
    result = tensor[:, 1:, :].reshape(tensor.size(0), height, width, z, tensor.size(2))

    # Bring the channels to the first dimension like in CNNs.
    result = result.transpose(3, 4).transpose(2, 3).transpose(1, 2)
    return result


def reshape_transform_2D(tensor, height=14, width=14):
    """
    Reshapes a 2D tensor into a format suitable for processing in CNNs.

    Parameters:
    - tensor (torch.Tensor): The input tensor to be reshaped, typically from a model's attention or feature map.
    - height (int, optional): The height dimension for the reshaped tensor. Default is 14.
    - width (int, optional): The width dimension for the reshaped tensor. Default is 14.

    Returns:
    - torch.Tensor: The reshaped tensor with channels moved to the first dimension, ready for CNN processing.
    """
    result = tensor[:, 1:, :].reshape(tensor.size(0), height, width, tensor.size(2))

    # Bring the channels to the first dimension like in CNNs.
    result = result.transpose(2, 3).transpose(1, 2)
    return result


def feature_mask(modality="image"):
    """
    Creates a feature mask for different modalities (image, volume, point cloud) to be used in feature extraction or visualization.

    Parameters:
    - modality (str, optional): The data modality for which to create the feature mask.
                                Supported values are "image", "volume", and "point_cloud". Default is "image".

    Returns:
    - torch.Tensor or None: The feature mask as a tensor of integers for the specified modality.
                            Returns None for unsupported modalities like "point_cloud".
    """
    if modality == "image":
        x = np.arange(0, 224 / 16, 1)

        x = np.repeat(x, 16, axis=0)

        row = np.vstack([x, x, x, x, x, x, x, x, x, x, x, x, x, x, x, x])

        rows = []

        for i in range(int(224 / 16)):
            rows.append(row + ((224 / 16) * i))

        mask = np.vstack(rows)

        return torch.from_numpy(mask).type(torch.int64)

    elif modality == "volume":
        x = np.arange(0, 28 / 7, 1)

        x = np.repeat(x, 7, axis=0)

        row = np.vstack([x, x, x, x, x, x, x])

        rows = []

        for i in range(int(28 / 7)):
            rows.append(row + ((28 / 7) * i))

        slice = np.vstack(rows)

        slice = np.repeat(np.expand_dims(slice, -1), 7, axis=-1)

        slices = []
        for i in range(int(28 / 7)):
            slices.append(slice + (16 * i))

        mask = np.concatenate(slices, axis=-1)

        return torch.from_numpy(mask).type(torch.int64)

    elif modality == "point_cloud":
        return None
