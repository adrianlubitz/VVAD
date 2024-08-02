# import the necessary packages
from collections import OrderedDict
import numpy as np

# define a dictionary that maps the indexes of the facial
# landmarks to specific face regions

# For dlib’s 68-point facial landmark detector:
FACIAL_LANDMARKS_68 = OrderedDict([
    ("mouth", (48, 68)),
    ("inner_mouth", (60, 68)),
    ("right_eyebrow", (17, 22)),
    ("left_eyebrow", (22, 27)),
    ("right_eye", (36, 42)),
    ("left_eye", (42, 48)),
    ("nose", (27, 36)),
    ("jaw", (0, 17))
])

# in order to support legacy code, we'll default the indexes to the
# 68-point model
FACIAL_LANDMARKS = FACIAL_LANDMARKS_68


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
