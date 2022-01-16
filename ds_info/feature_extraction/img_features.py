from os import X_OK
import numpy as np

def resolution(x):
    """Image resolution. The order is different depending on input type.

    Parameters:
    x_sitk (SimpleITK.SimpleITK.Image or numpy.ndarray): image

    Returns:
    tuple(int): resolution
    """
    if isinstance(x, np.ndarray):
        return np.shape(x)
    else:
        return x.GetSize()

def voxel_spacing(x_sitk):
    """Voxel spacing

    Parameters:
    x_sitk (SimpleITK.SimpleITK.Image): image

    Returns:
    lst(int): spacing
    """
    return [round(x_sp, 2) for x_sp in x_sitk.GetSpacing()]

def intensity_mean_median(x):
    flattened_x = x.flatten()
    return np.mean(flattened_x), np.median(flattened_x)