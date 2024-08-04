# import the necessary packages
import numpy as np


def shape_to_np(shape, dtype="int"):
    """
    Convert a list of facial landmarks to a NumPy array of (x, y, z) coordinates.

    This method takes a list of facial landmark points and converts each landmark to a 3D coordinate
    with the z-coordinate set to the same value as the x-coordinate. It initializes an array of shape
    `(len(shape), 3)` where each row corresponds to a landmark point represented by `(x, y, z)`.

    Args:
        shape (list or np.ndarray): A list or array of facial landmarks where each landmark is a 2D
                                    coordinate (x, y).
        dtype (str): Data type of the output NumPy array. Default is `"int"`. Can be changed to
                     `"float"` if needed.

    Returns:
        np.ndarray: A NumPy array of shape `(len(shape), 3)` where each entry is a 3D coordinate
                    (x, y, z) with `x`, `y` from the input `shape` and `z` set equal to `x`.
    """
    # initialize the list of (x, y, z)-coordinates
    coords = np.zeros((len(shape), 3), dtype=dtype)

    # loop over all facial landmarks and convert them
    # to a 3-tuple of (x, y, z)-coordinates
    for i in range(0, len(shape)):
        coords[i] = (shape[i][0], shape[i][0], shape[i][0])

    # return the list of (x, y, z)-coordinates
    return coords

def _normalize(arr):
    """ Normalizes the features of the array to [-1, 1].

        Args:
            arr (numpy array): array with features to normalize

        Returns:
            arr_norm (numpy array): numpy array with features normalized to [-1, 1]
    """
    arrMax = np.max(arr)
    arrMin = np.min(arr)
    absMax = np.max([np.abs(arrMax), np.abs(arrMin)])
    return arr / absMax

def shift_to_positive_range(p_cloud):
    """
    Shifts all values of a multidimensional point cloud into the positive range of
    values

    Args:
          p_cloud (np array): multidimensional point cloud
    Returns:
          shifted_cloud (np array): Point cloud shifted to positive value range along
          axis = 0
    """
    # Find the minimum values along each axis
    min_values = np.min(p_cloud, axis=0)

    # Subtract the minimum values from all points
    shifted_pcloud = p_cloud - min_values

    return shifted_pcloud

