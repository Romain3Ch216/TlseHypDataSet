import pdb
from typing import List
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
from TlseHypDataSet.utils.geometry import is_polygon_in_rectangle
from TlseHypDataSet.utils.utils import make_dirs, data_in_folder, tile_raster
import pkgutil
import csv
import seaborn as sns
import h5py


__all__ = [
    'TlseHypDataSet'
]


class TlseHypDataSet(Dataset):
    """

    """

    def __init__(self, root_path: str,
                 pred_mode: str,
                 patch_size: int,
                 padding: int = 0,
                 low_level_only: bool = False,
                 images: List = None,
                 subset: float = 1,
                 in_h5py: bool = False,
                 data_on_gpu: bool = False):

        self.name = 'Toulouse'
        self.root_path = root_path
        self.pred_mode = pred_mode
        self.patch_size = patch_size
        self.padding = padding
        self.low_level_only = low_level_only
        self.images = images
        self.subset = subset
        self.h5py = in_h5py
        self.transform = None
        self.saved_h5py = in_h5py
        self.data_on_gpu = data_on_gpu

        make_dirs([os.path.join(self.root_path, 'inputs')])
        make_dirs([os.path.join(self.root_path, 'outputs')])

        self.images_path = [
            'TLS_3d_2021-06-15_11-10-12_reflectance_rect',
            'TLS_1c_2021-06-15_10-41-20_reflectance_rect',
            'TLS_3a_2021-06-15_11-10-12_reflectance_rect',
            'TLS_5c_2021-06-15_11-40-13_reflectance_rect',
            'TLS_1d_2021-06-15_10-41-20_reflectance_rect',
            'TLS_9c_2021-06-15_12-56-29_reflectance_rect',
            'TLS_1b_2021-06-15_10-41-20_reflectance_rect',
            'TLS_1e_2021-06-15_10-41-20_reflectance_rect'
        ]

        if self.images is not None:
            self.images_path = [self.images_path[img_id] for img_id in self.images]

        self.gt_path = 'ground_truth.shp'

        dirs_in_root = os.listdir(root_path)
        assert ('images' in dirs_in_root) and ('GT' in dirs_in_root), \
            "Root directory should include an 'images' and a 'GT' folder."

        if data_in_folder([image + '.tif' for image in self.images_path], os.path.join(root_path, 'images')) is False:
            for image in self.images_path:
                assert image + '.bsq' in os.listdir(os.path.join(root_path, 'images')), "Image {} misses".format(image)
                assert image + '.hdr' in os.listdir(os.path.join(root_path, 'images')), "Header {} misses".format(image)
                tile_raster(os.path.join(self.root_path, 'images', image + '.bsq'))

        for ext in ['cpg', 'dbf', 'shp', 'prj', 'shx']:
            gt_file = self.gt_path[:-3] + ext
            assert gt_file in os.listdir(os.path.join(root_path, 'GT'))

        self.ground_truth = gpd.read_file(os.path.join(self.root_path, 'GT', self.gt_path))

        self.wv = None
        self.bbl = None
        self.E_dir = None
        self.E_dif = None
        self.n_bands = None
        self.samples = None

        print('Read metadata...')
        self.read_metadata()

        print('Open images...')
        self.image_rasters = [gdal.Open(os.path.join(self.root_path, 'images', image_path + '.tif'), gdal.GA_ReadOnly)
                              for image_path in self.images_path]
        print('Rasterize ground truth...')
        self.gts_path = self.rasterize_gt_shapefile()
        print('Open ground truth rasters...')
        self.gt_rasters = dict(
            (att, [gdal.Open(gt_path[:-3] + 'tif', gdal.GA_ReadOnly) for gt_path in self.gts_path[att]])
            for att in self.gts_path)

        if pred_mode == 'patch':
            self.compute_patches()
        elif pred_mode == 'pixel':
            self.compute_pixels()
        else:
            raise ValueError("pred_mode is either patch or pixel.")

        if self.h5py:
            print('Saving data set in h5py files...')
            h5py_data, h5py_labels = self.save_data_set()
            self.h5py_data, self.h5py_labels = h5py.File(h5py_data, 'r')['data'], h5py.File(h5py_labels, 'r')['data']
            if self.data_on_gpu:
                print('Loading whole data on device...')
                self.h5py_data = self.h5py_data[()]
                self.h5py_labels = self.h5py_labels[()]

    def read_metadata(self):
        self.wv = []
        self.bbl = []
        self.E_dir = []
        self.E_dif = []

        metadata = pkgutil.get_data(__name__, "metadata/tlse_metadata.txt")
        data_reader = csv.reader(metadata.decode('utf-8').splitlines(), delimiter=' ')

        for i, line in enumerate(data_reader):
            if i > 0:
                self.wv.append(float(line[0]))
                self.bbl.append(line[1] == 'True')
                self.E_dir.append(float(line[2]))
                self.E_dif.append(float(line[3]))

        self.bbl = np.array(self.bbl)
        self.E_dir = np.array(self.E_dir)
        self.E_dif = np.array(self.E_dif)
        self.wv = np.array(self.wv)
        self.n_bands = self.bbl.sum()

    @property
    def classes(self):
        gt = gpd.read_file(self.path_gt)
        classes = np.unique(gt['Material'])
        classes = classes[~np.isnan(classes)]
        return classes

    @property
    def n_classes(self):
        return len(self.labels)

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
            24: 'Cement with gravels',
            25: 'Gravels',
            26: 'Rocks',
            27: 'Green porous concrete',
            28: 'Red porous concrete',
            29: 'Seaweed',
            30: 'Water - swimming pool',
            31: 'Water',
            32: 'Sorgho',
            33: 'Wheat',
            34: 'Field bean',
            35: 'Clear plastic cover'
        }
        return labels_

    @property
    def colors(self):
        colors_ = sns.color_palette("hls", self.n_classes)
        return colors_

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
        gt = self.ground_truth  # gpd.read_file(os.path.join(dataset.root_path, 'GT', dataset.gt_path))
        make_dirs([os.path.join(self.root_path, 'rasters')])
        paths = {}

        def shapes(gt: GeoDataFrame, attribute: str):
            indices = gt.index
            for i in range(len(gt)):
                if np.isnan(gt.loc[indices[i], attribute]):
                    yield gt.loc[indices[i], 'geometry'], 0
                else:
                    yield gt.loc[indices[i], 'geometry'], int(gt.loc[indices[i], attribute])

        for attribute in ['Material', 'Class_2', 'Class_1', 'Group']:
            paths[attribute] = []
            dtype = 'uint16' if attribute == 'Group' else 'uint8'
            rasterio_dtype = rasterio.uint16 if dtype == 'uint16' else rasterio.uint8
            for id, img_path in enumerate(self.images_path):
                paths[attribute].append(os.path.join(self.root_path, 'rasters', 'gt_{}_{}.tif'.format(attribute, id)))
                if 'gt_{}_{}.tif'.format(attribute, id) in os.listdir(os.path.join(self.root_path, 'rasters')):
                    continue
                else:
                    path = os.path.join(self.root_path, 'rasters', 'gt_{}_{}.bsq'.format(attribute, id))
                    if 'gt_{}_{}.bsq'.format(attribute, id) in os.listdir(os.path.join(self.root_path, 'rasters')):
                        pass
                    else:
                        img = rasterio.open(os.path.join(self.root_path, 'images', img_path + '.bsq'))
                        shape = img.shape
                        data = rasterize(shapes(gt.groupby(by='Image').get_group(id + 1), attribute),
                                         shape[:2],
                                         dtype=dtype,
                                         transform=img.transform)
                        data = data.reshape(1, data.shape[0], data.shape[1]).astype(int)
                        with rasterio.Env():
                            profile = img.profile
                            profile.update(
                                dtype=rasterio_dtype,
                                count=1,
                                compress='lzw')
                            with rasterio.open(path, 'w', **profile) as dst:
                                dst.write(data)

                    tile_raster(path)
        return paths

    def split_already_computed(self, p_labeled, p_val, p_test):
        file = 'ground_truth_split_p_labeled_{}_p_val_{}_p_test_{}.pkl'.format(p_labeled, p_val, p_test)
        already_computed = file in os.listdir(os.path.join(self.root_path, 'outputs'))
        if already_computed:
            print('Data sets split is already computed')
        else:
            print('Computing data sets split...')
        return already_computed

    def load_splits(self, p_labeled, p_val, p_test):
        file = os.path.join(self.root_path, 'outputs', 'ground_truth_split_p_labeled_{}_p_val_{}_p_test_{}.pkl'.format(
            p_labeled, p_val, p_test))
        with open(os.path.join(self.root_path, file), 'rb') as f:
            data = pkl.load(f)
        return data

    def save_splits(self, solutions, p_labeled, p_val, p_test):
        file = os.path.join(self.root_path, 'outputs', 'ground_truth_split_p_labeled_{}_p_val_{}_p_test_{}.pkl'.format(
            p_labeled, p_val, p_test))
        with open(os.path.join(self.root_path, file), 'wb') as f:
            pkl.dump(solutions, f)

    def compute_patches(self):
        polygons_by_image = self.ground_truth.groupby(by='Image')
        groups, images, patch_coordinates = [], [], []
        for img_id in polygons_by_image.groups:
            image_path = os.path.join(self.root_path, 'images', self.images_path[img_id - 1])
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
                if (self.low_level_only is False) or (polygon['Material'] != 0):
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
                                tuple((left_col + i * self.patch_size, top_row + j * self.patch_size, self.patch_size,
                                       self.patch_size))
                            )

                    groups.extend([polygon['Group']] * len(patches))
                    images.extend([img_id] * len(patches))
                    patch_coordinates.extend(patches)

        self.samples = np.zeros((len(groups), 6), dtype=int)
        self.samples[:, 0] = images
        self.samples[:, 1] = groups
        self.samples[:, 2:] = patch_coordinates

    def compute_pixels(self):
        group_list, col_list, row_list, img_list = [], [], [], []
        for img_id, (gt, groups) in enumerate(zip(self.gt_rasters['Material'], self.gt_rasters['Group'])):
            gt = gt.ReadAsArray(gdal.GA_ReadOnly)
            groups = groups.ReadAsArray(gdal.GA_ReadOnly)
            coords = np.where(gt != 0)
            groups = groups[coords]
            img_list.extend([img_id] * len(groups))
            group_list.extend(groups)
            col_offset = coords[1] - self.patch_size // 2
            row_offset = coords[0] - self.patch_size // 2
            col_list.extend(col_offset)
            row_list.extend(row_offset)

        self.samples = np.zeros((len(group_list), 6), dtype=int)
        self.samples[:, 0] = img_list
        self.samples[:, 1] = group_list
        self.samples[:, 2] = col_list
        self.samples[:, 3] = row_list
        self.samples[:, 4] = self.patch_size
        self.samples[:, 5] = self.patch_size

        if self.subset < 1:
            n_samples = int(self.subset * self.samples.shape[0])
            subset = np.random.choice(np.arange(self.samples.shape[0]), size=n_samples, replace=False)
            self.samples = self.samples[subset]

    def save_data_set(self):
        images = 'images_' + '_'.join([str(img_id) for img_id in self.images]) if self.images is not None else 'all_images'
        data_file_path = os.path.join(self.root_path, 'inputs', 'data_{}_{}_{}.hdf5'.format(self.pred_mode, self.patch_size, images))
        labels_file_path = os.path.join(self.root_path, 'inputs', 'labels_{}_{}_{}.hdf5'.format(self.pred_mode, self.patch_size, images))
        if 'data_{}_{}_{}.hdf5'.format(self.pred_mode, self.patch_size, images) in os.listdir(os.path.join(self.root_path, 'inputs')) and\
                'labels_{}_{}_{}.hdf5'.format(self.pred_mode, self.patch_size, images) in os.listdir(os.path.join(self.root_path, 'inputs')):
            self.saved_h5py = True
            print("Data already saved in .h5py files.")
        else:

            data_file = h5py.File(data_file_path, "w")
            labels_file = h5py.File(labels_file_path, "w")
            if self.pred_mode == 'pixel':
                batch_size = 1024
            else:
                batch_size = 16

            sample, gt = self.__getitem__(0)
            data = data_file.create_dataset("data", tuple((len(self),)) + sample.shape, dtype='float32')
            labels = labels_file.create_dataset("data", tuple((len(self),)) + gt.shape, dtype='int8')

            loader = torch.utils.data.DataLoader(self, shuffle=False, batch_size=batch_size)
            i = 0
            for sample, gt in loader:
                b = sample.shape[0]
                data[i: i + b] = sample
                labels[i: i + b] = gt
                i += b
            self.saved_h5py = True

        return data_file_path, labels_file_path

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i):
        if self.h5py and self.saved_h5py:
            sample = self.h5py_data[i]
            gt = self.h5py_labels[i]
        else:
            image_id = self.samples[i, 0]
            if self.pred_mode == 'patch':
                image_id = image_id - 1

            coordinates = self.samples[i, 2:]
            col_offset, row_offset, col_size, row_size = [int(x) for x in coordinates]

            sample = self.image_rasters[image_id].ReadAsArray(col_offset, row_offset, col_size, row_size,
                                                              band_list=self.bands)
            gt = [self.gt_rasters[att][image_id].ReadAsArray(col_offset, row_offset, col_size, row_size)
                  for att in ['Material', 'Class_2', 'Class_1']]
            gt = [x.reshape(x.shape[0], x.shape[1], -1) for x in gt]
            gt = np.concatenate(gt, axis=-1)

            if self.low_level_only:
                gt = gt[:, :, 0]

            sample = np.transpose(sample, (1, 2, 0))
            sample = sample / 10 ** 4

            sample = np.asarray(np.copy(sample), dtype="float32")
            gt = np.asarray(np.copy(gt), dtype="int64")

            sample = torch.from_numpy(sample)
            gt = torch.from_numpy(gt)

            if self.patch_size == 1:
                sample = sample.squeeze(1)
                gt = gt.squeeze(1)

        if self.transform is not None and (self.saved_h5py and self.h5py):
            sample, gt = self.transform((sample, gt))

        return sample, gt
