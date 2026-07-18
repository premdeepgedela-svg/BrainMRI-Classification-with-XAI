import numpy as np


def has_brain_tissue(image, threshold=0.05):
    """
    Check whether an MRI slice contains sufficient brain tissue.

    Parameters
    ----------
    image : np.ndarray
        2D MRI slice.

    threshold : float
        Minimum fraction of non-zero pixels.

    Returns
    -------
    bool
    """

    non_zero_ratio = np.count_nonzero(image) / image.size

    return non_zero_ratio >= threshold