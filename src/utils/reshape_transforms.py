import torch
import numpy as np


def reshape_transform_3D(tensor, height=7, width=7, z=7):
    result = tensor[:, 1:, :].reshape(tensor.size(0), height, width, z, tensor.size(2))

    # Bring the channels to the first dimension,
    # like in CNNs.
    result = result.transpose(3, 4).transpose(2, 3).transpose(1, 2)
    return result


def reshape_transform_2D(tensor, height=14, width=14):
    result = tensor[:, 1:, :].reshape(tensor.size(0), height, width, tensor.size(2))

    # Bring the channels to the first dimension,
    # like in CNNs.
    result = result.transpose(2, 3).transpose(1, 2)
    return result


def feature_mask(modality="Image"):
    if modality == "Image":
        x = np.arange(0, 224 / 16, 1)

        x = np.repeat(x, 16, axis=0)

        row = np.vstack([x, x, x, x, x, x, x, x, x, x, x, x, x, x, x, x])

        rows = []

        for i in range(int(224 / 16)):
            rows.append(row + ((224 / 16) * i))

        mask = np.vstack(rows)

        return torch.from_numpy(mask).type(torch.int64)

    elif modality == "Volume":
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

    elif modality == "Point_Cloud":
        return None
