import pdb

import torch
from torch.utils.data import Dataset
import os
import geopandas as gpd
from geopandas import GeoDataFrame
import numpy as np
import pickle as pkl
from osgeo import gdal
import rasterio
from rasterio.features import rasterize
from utils.utils import make_dirs
from utils.geometry import is_polygon_in_rectangle
from utils.dataset import spatial_disjoint_split
from torchvision import transforms



class TlseHypDataSet(Dataset):
    """

    """
    def __init__(self, root_path: str,
                 patch_size: int,
                 padding: int):
        self.root_path = root_path
        self.patch_size = patch_size
        self.padding = padding
        self.transform = None

        self.images_path = [
            'TLS_3d_2021-06-15_11-10-12_reflectance_rect.bsq',
            'TLS_1c_2021-06-15_10-41-20_reflectance_rect.bsq',
            'TLS_3a_2021-06-15_11-10-12_reflectance_rect.bsq',
            'TLS_5c_2021-06-15_11-40-13_reflectance_rect.bsq',
            'TLS_1d_2021-06-15_10-41-20_reflectance_rect.bsq',
            'TLS_9c_2021-06-15_12-56-29_reflectance_rect.bsq',
            'TLS_1b_2021-06-15_10-41-20_reflectance_rect.bsq',
            'TLS_1e_2021-06-15_10-41-20_reflectance_rect.bsq'
        ]

        self.gt_path = 'ground_truth.shp'

        dirs_in_root = os.listdir(root_path)
        assert ('images' in dirs_in_root) and ('gt' in dirs_in_root), \
            "Root directory should include an 'images' and a 'gt' folder."

        for image in self.images_path:
            assert image in os.listdir(os.path.join(root_path, 'images')), "Image {} misses".format(image)
            header = image[:-3] + 'hdr'
            assert header in os.listdir(os.path.join(root_path, 'images')), "Header {} misses".format(header)

        for ext in ['cpg', 'dbf', 'shp', 'prj', 'shx']:
            gt_file = self.gt_path[:-3] + ext
            assert gt_file in os.listdir(os.path.join(root_path, 'GT'))

        self.ground_truth = gpd.read_file(os.path.join(self.root_path, 'GT', self.gt_path))

        self.wv = None
        self.bbl = None
        self.E_dir = None
        self.E_dif = None
        self.patch_coordinates = [] # dict((k, []) for k in self.ground_truth.index)

        self.read_metadata()
        self.compute_patches()

        self.image_rasters = [gdal.Open(os.path.join(self.root_path, 'images', image_path), gdal.GA_ReadOnly)
                              for image_path in self.images_path]
        self.gts_path = self.rasterize_gt_shapefile()
        self.gt_rasters = dict((att, [gdal.Open(gt_path, gdal.GA_ReadOnly) for gt_path in self.gts_path[att]])
                               for att in self.gts_path)


    def read_metadata(self):
        self.wv = []
        self.bbl = []
        self.E_dir = []
        self.E_dif = []

        with open('../TlseHypDataSet/metadata/tlse_metadata.txt', 'r') as f:
            for i, line in enumerate(f):
                if i > 0:
                    line = line.split(' ')
                    self.wv.append(float(line[0]))
                    self.bbl.append(line[1] == 'True')
                    self.E_dir.append(float(line[2]))
                    self.E_dif.append(float(line[3]))

        self.bbl = np.array(self.bbl)
        self.E_dir = np.array(self.E_dir)
        self.E_dif = np.array(self.E_dif)
        self.wv = np.array(self.wv)
        self.wv = self.wv[self.bbl]

    @property
    def classes(self):
        gt = gpd.read_file(self.path_gt)
        classes = np.unique(gt['Material'])
        classes = classes[~np.isnan(classes)]
        return classes

    @property
    def bands(self):
        """
        Extract the band numbers and the bad band list from the header of the first image.
        """
        bands = tuple(np.where(self.bbl.astype(int) != 0)[0].astype(int) + 1)
        bands = [int(b) for b in bands]
        return bands

    @property
    def labels(self):
        labels_ = {
            1: 'Orange tile',
            2: 'Dark tile',
            3: 'Slate',
            4: 'Clear fiber cement',
            5: 'Dark fiber cement',
            6: 'Sheet metal - dark painted sheet metal',
            7: 'Clear painted sheet metal',
            8: 'Dark asphalt',
            9: 'Red asphalt',
            10: 'Beige asphalt',
            11: 'Green painted asphalt',
            12: 'White road marking',
            13: 'Cement',
            14: 'Brown paving stone',
            15: 'Clear paving stone',
            16: 'Pink concrete paving stone',
            17: 'Clear concrete paving stone',
            18: 'Running track',
            19: 'Synthetic grass',
            20: 'Healthy grass',
            21: 'Stressed grass',
            22: 'Tree',
            23: 'Bare soil',
            24: 'Bare soil with vegetation',
            25: 'Cement with gravels',
            26: 'Gravels',
            27: 'Rocks',
            28: 'Green porous concrete',
            29: 'Red porous concrete',
            30: 'Seaweed',
            31: 'Water - swimming pool',
            32: 'Water',
            33: 'Sorgho',
            34: 'Wheat',
            35: 'Field bean',
            36: 'Clear plastic cover'
        }
        return labels_

    @property
    def areas(self):
        """

        :return:
        """
        groups = np.unique(self.ground_truth['Group'])
        n_groups = len(groups)
        classes = np.unique(self.ground_truth['Material'])
        classes = classes[1:]
        n_classes = len(classes)
        areas = np.zeros((n_groups, n_classes))

        for i in range(len(groups)):
            for j in range(n_classes):
                group = groups[i]
                class_id = classes[j]
                polygons = self.ground_truth[self.ground_truth['Group'] == group]
                polygons = polygons[polygons['Material'] == class_id]
                areas[i, j] = polygons['geometry'].area.sum()
        return areas.astype(int)

    def rasterize_gt_shapefile(self):
        """
        Rasterize the ground truth shapefile.
        """
        gt = self.ground_truth # gpd.read_file(os.path.join(dataset.root_path, 'GT', dataset.gt_path))
        make_dirs([os.path.join(self.root_path, 'rasters')])
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
            for id, img_path in enumerate(self.images_path):
                path = os.path.join(self.root_path, 'rasters', 'gt_{}_{}.bsq'.format(attribute, id))
                img = rasterio.open(os.path.join(self.root_path, 'images', img_path))
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

    def split_already_computed(self, p_labeled, p_test):
        file = 'ground_truth_split_p_labeled_{}_p_test_{}.pkl'.format(p_labeled, p_test)
        return file in os.listdir(self.root_path)

    def load_splits(self, p_labeled, p_test):
        file = os.path.join(self.root_path, 'ground_truth_split_p_labeled_{}_p_test_{}.pkl'.format(p_labeled, p_test))
        with open(os.path.join(self.root_path, file), 'rb') as f:
            data = pkl.load(f)
        return data

    def save_splits(self, solutions, p_labeled, p_test):
        file = os.path.join(self.root_path, 'ground_truth_split_p_labeled_{}_p_test_{}.pkl'.format(p_labeled, p_test))
        with open(os.path.join(self.root_path, file), 'wb') as f:
            pkl.dump(solutions, f)

    def compute_patches(self):
        polygons_by_image = self.ground_truth.groupby(by='Image')
        groups, images, patch_coordinates = [], [], []
        for img_id in polygons_by_image.groups:
            image_path = os.path.join(self.root_path, 'images', self.images_path[img_id-1])
            raster = gdal.Open(image_path, gdal.GA_ReadOnly)
            transform = raster.GetGeoTransform()
            xOrigin = transform[0]
            yOrigin = transform[3]
            pixelWidth = transform[1]
            pixelHeight = -transform[5]
            # Xgeo = xOrigin + Xpixel*pixelWidth
            # Xpixel = (Xgeo - xOrigin) / pixelWidth
            # Ygeo = yOrigin - Yline * pixelHeight
            # Yline = (yOrigin - Ygeo) / pixelHeight
            image_bounds = (xOrigin,
                            yOrigin,
                            xOrigin + raster.RasterXSize * pixelWidth,
                            yOrigin - raster.RasterYSize * pixelHeight)

            polygons = polygons_by_image.get_group(img_id)
            for id, polygon in polygons.iterrows():
                patches = []
                bounds = polygon['geometry'].bounds  # min x, min y, max x, max y
                assert is_polygon_in_rectangle(bounds, image_bounds), "Polygon is not in image"

                left_col = int((bounds[0] - xOrigin) / pixelWidth) - self.padding // 2
                right_col = int((bounds[2] - xOrigin) / pixelWidth) + self.padding // 2
                top_row = int((yOrigin - bounds[3]) / pixelHeight) - self.padding // 2
                bottom_row = int((yOrigin - bounds[1]) / pixelHeight) + self.padding // 2

                width = right_col - left_col
                height = bottom_row - top_row
                n_x_patches = int(np.ceil(width / self.patch_size))
                n_y_patches = int(np.ceil(height / self.patch_size))

                for i in range(n_x_patches):
                    for j in range(n_y_patches):
                        patches.append(
                            tuple((left_col + i * self.patch_size, top_row + j * self.patch_size, self.patch_size, self.patch_size))
                        )

                groups.extend([polygon['Group']] * len(patches))
                images.extend([img_id] * len(patches))
                patch_coordinates.extend(patches)
        self.patch_coordinates = np.zeros((len(groups), 6), dtype=int)
        self.patch_coordinates[:, 0] = images
        self.patch_coordinates[:, 1] = groups
        self.patch_coordinates[:, 2:] = patch_coordinates

    def __len__(self):
        return len(self.patch_coordinates)

    def __getitem__(self, i):
        i = int(i)
        image_id = self.patch_coordinates[i, 0]
        coordinates = self.patch_coordinates[i, 2:]
        col_offset, row_offset, col_size, row_size = [int(x) for x in coordinates]

        sample = self.image_rasters[image_id-1].ReadAsArray(col_offset, row_offset, col_size, row_size,
                                                          band_list=self.bands)
        gt = [self.gt_rasters[att][image_id-1].ReadAsArray(col_offset, row_offset, col_size, row_size)
              for att in self.gt_rasters]
        gt = [x.reshape(x.shape[0], x.shape[1], -1) for x in gt]
        gt = np.concatenate(gt, axis=-1)

        sample = np.transpose(sample, (1, 2, 0))
        sample = sample / 10**4

        sample = np.asarray(np.copy(sample), dtype="float32")
        gt = np.asarray(np.copy(gt), dtype="int64")

        sample = torch.from_numpy(sample)
        gt = torch.from_numpy(gt)

        if self.transform is not None:
            sample, gt = self.transform((sample, gt))

        return sample, gt
