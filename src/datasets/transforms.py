import numpy as np
import torch


def normalize(image):
    """
    Z-score normalization.

    Only normalize non-zero voxels.
    """

    mask = image > 0

    if np.sum(mask) == 0:
        return image.astype(np.float32)

    image = image.astype(np.float32)

    mean = image[mask].mean()
    std = image[mask].std()

    image[mask] = (image[mask] - mean) / (std + 1e-8)

    return image


def to_tensor(image):

    return torch.from_numpy(image).float()
