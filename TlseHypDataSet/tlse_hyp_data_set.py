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
from TlseHypDataSet.utils.utils import make_dirs, data_in_folder, tile_raster
import pkgutil
import csv
import seaborn as sns
import h5py
import importlib_resources
import pkg_resources


__all__ = [
    'TlseHypDataSet'
]


class TlseHypDataSet(Dataset):
    """
    A torch.utils.data.Dataset object to process the Toulouse Hyperspectral Data Set
    """
    def __init__(self, root_path: str,
                 pred_mode: str,
                 patch_size: int,
                 annotations: str = 'land_cover',
                 images: List = None,
                 in_h5py: bool = False,
                 data_on_gpu: bool = False
                 ):
        """
        :param root_path: path to the folder where the data is stored
        :param pred_mode: 'pixel' for pixel-wise classification or 'patch' for patch segmentation
        :param patch_size: size of the patch, i.e. gives (batch_size x patch_size x patch_size x n_bands) dimensional samples
        :param annotations: 'land_cover', 'land_use' or 'both'
        :param images: select a subset of image tiles by specifying tile index (in the following order [3d, 1c, 3a, 5c, 1d, 9c, 1b, 1e, 3e])
        :param in_h5py: if True, save the data samples and labels in h5py files to speed up data reading. At first use, \
        writing the data in the h5py file may take a very long time, but it might considerably decrease the reading \
        time afterwards
        :param data_on_gpu: if True, store the whole data on the device (e.g. on the gpu)
        """

        self.name = 'Toulouse'
        self.root_path = root_path
        self.pred_mode = pred_mode
        self.patch_size = patch_size
        self.annotations = annotations
        self.images = images
        self.h5py = in_h5py
        self.transform = None
        self.saved_h5py = in_h5py
        self.data_on_gpu = data_on_gpu
        self.rgb_bands = np.array([77, 32, 0])

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
            'TLS_1e_2021-06-15_10-41-20_reflectance_rect',
            'TLS_3e_2021-06-15_11-10-12_reflectance_rect'
        ]

        if self.images is None:
            self.images = np.arange(len(self.images_path))

        dirs_in_root = os.listdir(root_path)
        assert ('images' in dirs_in_root), "Root directory should include an 'images' folder."

        for image in np.array(self.images_path)[np.array(self.images)]:
            if data_in_folder([image + '.tif'], os.path.join(root_path, 'images')) is False:
                assert image + '.bsq' in os.listdir(os.path.join(root_path, 'images')), "Image {} misses".format(image)
                assert image + '.hdr' in os.listdir(os.path.join(root_path, 'images')), "Header {} misses".format(image)
                tile_raster(os.path.join(self.root_path, 'images', image + '.bsq'))

        gt_rsrc = importlib_resources.files('TlseHypDataSet.ground_truth').joinpath('ground_truth.shp')
        self.ground_truth = gpd.read_file(gt_rsrc)

        self.wv_ = None
        self.bbl_ = None
        self.E_dir_ = None
        self.E_dif_ = None
        self.theta_ = 22.12 * np.pi / 180
        self.n_bands_ = None
        self.samples = None


        print('Read metadata...')
        self.read_metadata()

        print('Open images...')
        self.image_rasters = [gdal.Open(os.path.join(self.root_path, 'images', image_path + '.tif'), gdal.GA_ReadOnly)
                              for image_path in np.array(self.images_path)[np.array(self.images)]]

        if 'zero_masks.pkl' in os.listdir(os.path.join(self.root_path, 'inputs')):
            with open(os.path.join(self.root_path, 'inputs', 'zero_masks.pkl'), 'rb') as f:
                self.zero_masks = pkl.load(f)
        else:
            self.zero_masks = [np.ones((img.RasterYSize, img.RasterXSize)) for img in self.image_rasters]

        print('Rasterize ground truth...')
        self.gts_path = self.rasterize_gt_shapefile()

        print('Open ground truth rasters...')
        self.gt_rasters = dict(
            (att, [(self.images[i], gdal.Open(gt_path[:-3] + 'tif', gdal.GA_ReadOnly)) for i, gt_path in enumerate(self.gts_path[att])])
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

        self.test_patches = [
            {
                'img_id': 1,
                'coords': (900, 600, 300, 300)
            },
            {
                'img_id': 1,
                'coords': (1500, 1470, 300, 300)
            },
            {
                'img_id': 1,
                'coords': (2700, 3950, 300, 300)
            },
            {
                'img_id': 6,
                'coords': (2500, 3600, 300, 300)
            }
        ]

    def read_metadata(self):
        wv = []
        bbl = []
        E_dir = []
        E_dif = []

        metadata = pkgutil.get_data(__name__, "metadata/tlse_metadata.txt")
        data_reader = csv.reader(metadata.decode('utf-8').splitlines(), delimiter=' ')

        for i, line in enumerate(data_reader):
            if i > 0:
                wv.append(float(line[0]))
                bbl.append(line[1] == 'True')
                E_dir.append(float(line[2]))
                E_dif.append(float(line[3]))

        self.bbl_ = np.array(bbl)
        E_dir = np.array(E_dir)
        E_dif = np.array(E_dif)
        self.wv_ = np.array(wv)
        self.n_bands_ = self.bbl_.sum()
        self.E_dir_ = torch.from_numpy(E_dir[self.bbl_] / np.cos(self.theta_)).float()
        self.E_dif_ = torch.from_numpy(E_dif[self.bbl_]).float()

    @property
    def standard_splits(self):
        """
        :return: A list of 8 spatially disjoint splits of the ground truth
        """
        standard_splits = []
        for split_id in range(1, 9):
            split = pkl.load(pkg_resources.resource_stream('TlseHypDataSet.default_splits', 'split_{}.pkl'.format(split_id)))
            standard_splits.append(split)
        return standard_splits

    @property
    def theta(self):
        """
        :return: The solar zenith angle
        """
        return torch.tensor([self.theta_]).float()

    @property
    def E_dir(self):
        """
        :return: The direct irradiance at ground level
        """
        return self.E_dir_

    @property
    def E_dif(self):
        """
        :return: The diffuse irradiance at ground level
        """
        return self.E_dif_

    @property
    def wv(self):
        """
        :return: A list of the spectral channels wavelengths
        """
        return self.wv_

    @property
    def n_bands(self):
        """
        :return: The number of usable spectral channels
        """
        return self.n_bands_

    @property
    def bbl(self):
        """
        :return: A list of bad bands (i.e. unusable spectral channels) encoded by zeros
        """
        return self.bbl_

    @property
    def classes(self):
        gt_rsrc = importlib_resources.files('TlseHypDataSet.ground_truth').joinpath('ground_truth.shp')
        gt = gpd.read_file(gt_rsrc)
        classes = np.unique(gt['Material'])
        classes = classes[~np.isnan(classes)]
        return classes

    @property
    def n_classes(self):
        return len(self.land_cover_nomenclature)

    @property
    def bands(self):
        """
        :return: A list of usable band indices
        """
        bands = tuple(np.where(self.bbl.astype(int) != 0)[0].astype(int) + 1)
        bands = [int(b) for b in bands]
        return bands

    @property
    def land_cover_nomenclature(self):
        """
        :return: A dict with the children land cover classes (bottom of the hierarchy)
        """
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
            16: 'Clear plastic cover',
            17: 'Synthetic track',
            18: 'Synthetic grass',
            19: 'Healthy grass',
            20: 'Stressed grass',
            21: 'Tree',
            22: 'Bare soil',
            23: 'Gravels',
            24: 'Rocks',
            25: 'Green porous concrete',
            26: 'Red porous concrete',
            27: 'Seaweed',
            28: 'Water - swimming pool',
            29: 'Water',
            30: 'Sorgho',
            31: 'Wheat',
            32: 'Field bean',

        }
        return labels_

    @property
    def land_cover_nomenclature_top(self):
        """
        :return: A dict with the parent land cover classes (top of the hierarchy)
        """
        labels_ = {
            1: 'Tile',
            2: 'Slate',
            3: 'Fiber cement',
            4: 'Sheet metal',
            5: 'Asphalt',
            6: 'Cement',
            7: 'Paving stone',
            8: 'Clear plastic cover',
            9: 'Synthetic material',
            10: 'Vegetation',
            11: 'Bare soil',
            12: 'Gravels and rocks',
            13: 'Porous concrete',
            14: 'Water',
            15: 'Crops'
        }
        return labels_

    @property
    def bottom_to_top(self):
        """
        :return: A dict whose keys are bottom land cover classes and values are top land cover classes\
         in the hierarchical nomenclature
        """
        b2t = {
            1: 1, 2: 1, 3: 2, 4: 3, 5: 3, 6: 4, 7: 4, 8: 5, 9: 5, 10: 5, 11: 5, 12: 5, 13: 6, 14: 7, 15: 7, 16: 8,
            17: 9, 18: 9, 19: 10, 20: 10, 21: 10, 22: 11, 23: 12, 24: 12, 25: 13, 26: 13, 27: 14, 28: 14, 29: 14,
            30: 15, 31: 15, 32: 15
        }
        return b2t



    @property
    def land_use_nomenclature(self):
        """
        :return: A dict with the land use classes
        """
        labels_ = {
            1: 'Roads',
            2: 'Railways',
            3: 'Roofs',
            4: 'Parking lots',
            5: 'Building sites',
            6: 'Sport facilities',
            7: 'Lakes / rivers / harbors',
            8: 'Swimming pools',
            9: 'Forests',
            10: 'Cultivated fields',
            11: 'Boats',
            12: 'Open areas'
        }
        return labels_


    @property
    def permeability(self):
        """
        :return: A dict with the permeability of the land cover (0 = impermeable, 1 = permeable)
        """
        permeability_ = {}
        for class_id in self.land_cover_nomenclature:
            if class_id <= 16:
                permeability_[class_id] = 0
            else:
                permeability_[class_id] = 1
        return permeability_

    @property
    def colors(self):
        """
        :return: A dict of class colors for land cover maps
        """
        colors_ = sns.color_palette("hls", self.n_classes)
        return colors_

    @property
    def areas(self):
        groups = np.unique(self.ground_truth['Group'])
        n_groups = len(groups)
        classes = np.unique(self.ground_truth['Material'])
        classes = classes[1:]
        n_classes = len(classes)
        areas = np.zeros((n_groups, n_classes))

        for i in range(n_classes):
            class_id = classes[i]
            polygons = self.ground_truth[self.ground_truth['Material'] == class_id]
            polygons_by_groups = polygons.groupby(by='Group')
            for group in polygons_by_groups.groups.keys():
                group_indice = np.where(groups == group)[0][0]
                areas[group_indice, i] = polygons_by_groups.get_group(group).area.sum()
        return areas.astype(int)

    @property
    def n_samples(self):
        n_samples_ = np.zeros(self.n_classes)
        for gt_id in self.images:
            path = os.path.join(self.root_path, 'rasters', 'gt_Material_{}.bsq'.format(gt_id))
            gt = rasterio.open(path)
            gt = gt.read()
            for class_id in np.unique(gt):
                if class_id != 0 and class_id != 40:
                    n_samples_[class_id-1] += np.sum(gt == class_id)
        return n_samples_

    def rasterize_gt_shapefile(self):
        """
        Rasterize the ground truth shapefile.
        """
        gt = self.ground_truth
        make_dirs([os.path.join(self.root_path, 'rasters')])
        paths = {}

        def shapes(gt: GeoDataFrame, attribute: str):
            indices = gt.index
            for i in range(len(gt)):
                if np.isnan(gt.loc[indices[i], attribute]):
                    yield gt.loc[indices[i], 'geometry'], 0
                else:
                    yield gt.loc[indices[i], 'geometry'], int(gt.loc[indices[i], attribute])

        for attribute in ['Material', 'Abstract', 'Group']:
            paths[attribute] = []
            dtype = 'uint16' if attribute == 'Group' else 'uint8'
            rasterio_dtype = rasterio.uint16 if dtype == 'uint16' else rasterio.uint8
            for id, img_path in zip(self.images, np.array(self.images_path)[np.array(self.images)]):
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

    def split_already_computed(self, p_labeled, p_val, p_test, timestamp):
        images = 'images_' + '_'.join([str(img_id) for img_id in self.images]) if self.images is not None else 'all_images'
        file = 'ground_truth_split_{}_{}_p_labeled_{}_p_val_{}_p_test_{}.pkl'.format(timestamp, images, p_labeled, p_val, p_test)
        already_computed = file in os.listdir(os.path.join(self.root_path, 'outputs'))
        if already_computed:
            print('Data sets split is already computed')
        else:
            print('Computing data sets split...')
        return already_computed

    def load_splits(self, path=None, p_labeled=None, p_val=None, p_test=None, fold=None):
        if path is None:
            assert (p_labeled is not None) and (p_val is not None) and (p_test is not None) and (fold is not None), \
            "If a path is not given, then the proportions p_labeled, p_val and p_test must be provided, as well as the fold number"

            images = 'images_' + '_'.join([str(img_id) for img_id in self.images]) if self.images is not None else 'all_images'
            file = os.path.join(self.root_path, 'outputs', 'ground_truth_split_{}_p_labeled_{}_p_val_{}_p_test_{}.pkl'.format(
                images, p_labeled, p_val, p_test))
            with open(file, 'rb') as f:
                splits = pkl.load(f)
            splits = splits[fold]
        else:
            with open(os.path.join(self.root_path, 'outputs'), 'rb') as f:
                splits = pkl.load(f)
        return splits

    def save_splits(self, solutions, p_labeled, p_val, p_test, timestamp):
        images = 'images_' + '_'.join([str(img_id) for img_id in self.images]) if self.images is not None else 'all_images'
        file = os.path.join(self.root_path, 'outputs', 'ground_truth_split_{}_{}_p_labeled_{}_p_val_{}_p_test_{}.pkl'.format(
            timestamp, images, p_labeled, p_val, p_test))
        with open(os.path.join(self.root_path, file), 'wb') as f:
            pkl.dump(solutions, f)

    @property
    def proj_data(self):
        file = os.path.join(self.root_path, 'outputs', 'data_proj.npy')
        if 'data_proj.npy' in os.listdir(os.path.join(self.root_path, 'outputs')):
            proj_data = np.load(file)
            return proj_data
        else:
            raise ValueError('Data has not been projected yet.')

    def compute_patches(self):
        group_list, patches, img_list = [], [], []
        for i, ((img_id, gt), (_, groups)) in enumerate(zip(self.gt_rasters['Material'], self.gt_rasters['Group'])):
            gt = gt.ReadAsArray(gdal.GA_ReadOnly)
            zero_mask_ = self.zero_masks[i]
            groups = groups.ReadAsArray(gdal.GA_ReadOnly)
            nx_patches = gt.shape[0] // self.patch_size
            ny_patches = gt.shape[1] // self.patch_size
            for k in range(nx_patches):
                for l in range(ny_patches):
                    top = k * self.patch_size
                    left = l * self.patch_size
                    labels = gt[top: min(gt.shape[0], top + self.patch_size),
                             left: min(gt.shape[1], left + self.patch_size)]
                    group = groups[top: min(gt.shape[0], top + self.patch_size),
                             left: min(gt.shape[1], left + self.patch_size)]
                    zero_mask = zero_mask_[top: min(gt.shape[0], top + self.patch_size),
                             left: min(gt.shape[1], left + self.patch_size)]
                    if labels.sum() > 10 and np.sum(zero_mask) > (0.1 * self.patch_size ** 2):
                        labels_in_patch, counts = np.unique(labels, return_counts=True)
                        labels_in_patch = [x for x in labels_in_patch if x !=0]
                        if 40 in labels_in_patch and len(labels_in_patch) == 1: # labels_in_patch[np.argmax(counts)] == 40:
                            group = -1
                        else:
                            list_groups = np.unique(group[group != 0])
                            n_px_groups = [(group == group_id).sum() for group_id in list_groups]
                            group = list_groups[np.argmax(n_px_groups)]
                        patches.append(tuple((left, top, self.patch_size, self.patch_size)))
                        img_list.append(i)
                        group_list.append(group)

        self.samples = np.zeros((len(group_list), 6), dtype=int)
        self.samples[:, 0] = img_list
        self.samples[:, 1] = group_list
        self.samples[:, 2:] = np.array(patches)

    def compute_pixels(self):
        group_list, col_list, row_list, img_list = [], [], [], []
        for i, ((img_id, gt), (_, groups)) in enumerate(zip(self.gt_rasters['Material'], self.gt_rasters['Group'])):
            gt = gt.ReadAsArray(gdal.GA_ReadOnly)
            groups = groups.ReadAsArray(gdal.GA_ReadOnly)
            coords = np.where((gt != 0) * (gt != 40))
            groups = groups[coords]
            img_list.extend([i] * len(groups))
            group_list.extend(groups)
            col_offset = coords[1] - self.patch_size // 2
            row_offset = coords[0] - self.patch_size // 2
            col_list.extend(col_offset)
            row_list.extend(row_offset)

            unlabeled_coords = np.where(gt == 40)
            group_list.extend(-1 * np.ones(len(unlabeled_coords[0])))
            img_list.extend([i] * len(unlabeled_coords[0]))
            col_offset = unlabeled_coords[1] - self.patch_size // 2
            row_offset = unlabeled_coords[0] - self.patch_size // 2
            col_list.extend(col_offset)
            row_list.extend(row_offset)

        self.samples = np.zeros((len(group_list), 6), dtype=int)
        self.samples[:, 0] = img_list
        self.samples[:, 1] = group_list
        self.samples[:, 2] = col_list
        self.samples[:, 3] = row_list
        self.samples[:, 4] = self.patch_size
        self.samples[:, 5] = self.patch_size

    def save_data_set(self):
        images = 'images_' + '_'.join([str(img_id) for img_id in self.images]) if self.images is not None else 'all_images'
        option = '_'
        data_file_path = os.path.join(self.root_path, 'inputs', 'data_{}_{}_{}_{}.hdf5'.format(self.pred_mode, self.patch_size, images, option))
        labels_file_path = os.path.join(self.root_path, 'inputs', 'labels_{}_{}_{}_{}.hdf5'.format(self.pred_mode, self.patch_size, images, option))
        if 'data_{}_{}_{}_{}.hdf5'.format(self.pred_mode, self.patch_size, images, option) in os.listdir(os.path.join(self.root_path, 'inputs')):
            self.saved_h5py = True
            print("Data already saved in .h5py files.")
        else:
            self.saved_h5py = False
            data_file = h5py.File(data_file_path, "w")
            labels_file = h5py.File(labels_file_path, "w")
            if self.pred_mode == 'pixel':
                batch_size = 2**13
            else:
                batch_size = 16

            sample, gt = self.__getitem__(0)
            labels = labels_file.create_dataset("data", tuple((len(self),)) + gt.shape, dtype='int8')

            data = data_file.create_dataset("data", tuple((len(self),)) + sample.shape, dtype='float32')

            loader = torch.utils.data.DataLoader(self, shuffle=False, batch_size=batch_size)
            i = 0
            for batch in loader:
                sample, gt = batch
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

            coordinates = self.samples[i, 2:]
            col_offset, row_offset, col_size, row_size = [int(x) for x in coordinates]

            sample = self.image_rasters[image_id].ReadAsArray(col_offset, row_offset, col_size, row_size,
                                                              band_list=self.bands)

            sample = np.transpose(sample, (1, 2, 0))
            sample = sample / 10 ** 4
            sample = np.asarray(np.copy(sample), dtype="float32")
            sample = torch.from_numpy(sample)

            gt = [self.gt_rasters[att][image_id][1].ReadAsArray(col_offset, row_offset, col_size, row_size)
                  for att in ['Material', 'Abstract']]
            gt = [x.reshape(x.shape[0], x.shape[1], -1) for x in gt]
            gt = np.concatenate(gt, axis=-1)

            if self.annotations == 'land_cover':
                gt = gt[:, :, 0]
            elif self.annotations == 'land_use':
                gt = gt[:, :, 1]
            elif self.annotations == 'both':
                pass
            else:
                raise NotImplementedError("annotations should be 'land_cover', 'land_use' or 'both'")

            gt = np.asarray(np.copy(gt), dtype="int64")
            gt = torch.from_numpy(gt)

            if self.patch_size == 1:
                sample = sample.squeeze(1)
                gt = gt.squeeze(1)
            if self.pred_mode == 'pixel' and self.patch_size > 1:
                gt = gt[self.patch_size // 2, self.patch_size // 2]

        if self.transform is not None and ((self.saved_h5py and self.h5py) or (self.h5py is False)):
            if isinstance(sample, np.ndarray):
                sample = torch.from_numpy(sample)
            if isinstance(gt, np.ndarray):
                gt = torch.from_numpy(gt)
            sample, gt = self.transform((sample, gt))

        return sample, gt


class Split:
    def __init__(self, path):
        with open(path, 'rb') as f:
            self.sets = pkl.load(f)


class SubsetSampler(torch.utils.data.Sampler):
    def __init__(self, indices):
        self.indices = indices

    def __iter__(self):
        return (i for i in self.indices)

    def __len__(self):
        return len(self.indices)

