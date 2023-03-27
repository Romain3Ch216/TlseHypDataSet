from typing import List, Tuple
import geopandas as gpd
from geopandas import GeoDataFrame
import numpy as np
import rasterio
from rasterio.features import rasterize
from utils.utils import make_dirs
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

def rasterize_gt_shapefile(dataset):
    """
    Rasterize the ground truth shapefile.
    """
    gt = dataset.ground_truth # gpd.read_file(os.path.join(dataset.root_path, 'GT', dataset.gt_path))
    make_dirs([os.path.join(dataset.root_path, 'rasters')])
    paths = {}

    def shapes(gt: GeoDataFrame, attribute: str):
        indices = gt.index
        for i in range(len(gt)):
            if np.isnan(gt.loc[indices[i], attribute]):
                yield gt.loc[indices[i], 'geometry'], 0
            else:
                yield gt.loc[indices[i], 'geometry'], int(gt.loc[indices[i], attribute])

    for attribute in ['Material', 'Class_2', 'Class_1']:
        paths[attribute] = []
        for id, img_path in enumerate(dataset.images_path):
            path = os.path.join(dataset.root_path, 'rasters', 'gt_{}_{}.bsq'.format(attribute, id))
            img = rasterio.open(os.path.join(dataset.root_path, 'images', img_path))
            shape = img.shape
            data = rasterize(shapes(gt.groupby(by='Image').get_group(id + 1), attribute), shape[:2], dtype='uint8',
                             transform=img.transform)
            data = data.reshape(1, data.shape[0], data.shape[1]).astype(int)
            with rasterio.Env():
                profile = img.profile
                profile.update(
                    dtype=rasterio.uint8,
                    count=1,
                    compress='lzw')
                with rasterio.open(path, 'w', **profile) as dst:
                    dst.write(data)
            paths[attribute].append(path)
    return paths
