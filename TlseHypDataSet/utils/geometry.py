from typing import List, Tuple
import geopandas as gpd
import numpy as np

import os


def is_polygon_in_rectangle(bounds: np.ndarray, rectangle: Tuple[int]) -> bool:
    """
    Calculates if a rectangle contains a polygon.
    :param bounds: (left, top, right, bottom) bounds of a polygon
    :param rectangle: (left, top, right, bottom) bounds of a rectangle
    :return: True if the polygon is in the rectangle
    """
    in_rectangle = (rectangle[0] < bounds[0]) * (bounds[2] < rectangle[2]) * \
                   (bounds[1] < rectangle[1]) * (rectangle[3] < bounds[3])

    return in_rectangle


def flip_array(*arrays):
        horizontal = np.random.random() > 0.5
        vertical = np.random.random() > 0.5
        if horizontal:
            arrays = [np.fliplr(arr) for arr in arrays]
        if vertical:
            arrays = [np.flipud(arr) for arr in arrays]
        return arrays