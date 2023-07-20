import pdb

import torch
from torch.utils.data import Dataset
from TlseHypDataSet.utils.geometry import _compute_number_of_tiles, _compute_float_overlapping, ceil_int
from typing import Union, Sequence, List
import numpy as np
from scipy import io
import os
import rasterio
from rasterio.warp import reproject, Resampling

class HyperspectralDataSet(Dataset):
    def __init__(self, root_path: str, patch_size: int, min_overlapping: int):
        self.root_path = root_path
        self.patch_size = patch_size
        self.min_overlapping = min_overlapping
        self.transform = None

        self.image_path = None
        self.gt_path = None
        self.bbl = None
        self.E_dir = None
        self.E_dif = None

        self.img, self.gt = self.load_data()
        self.compute_patches()

    @property
    def labels(self):
        raise NotImplementedError

    @property
    def n_classes(self):
        return len(self.labels)

    def load_data(self):
        raise NotImplementedError

    def compute_patches(self):
        H, W, B = self.img.shape
        self.nx = _compute_number_of_tiles(self.patch_size, H, self.min_overlapping)
        self.ny = _compute_number_of_tiles(self.patch_size, W, self.min_overlapping)
        self.float_overlapping_x = _compute_float_overlapping(self.patch_size, H, self.nx)
        self.float_overlapping_y = _compute_float_overlapping(self.patch_size, W, self.ny)
        self._max_index = self.nx * self.ny
        self.indices = self._valid_indices

    @property
    def _valid_indices(self) -> List:
        """
        Computes indices from patches that admit at least 4 neighbours
        """
        indices: List[int] = []
        for idx in range(self._max_index):
            extent, _, _ = self._get_tile_extent(idx)
            gt = self.gt[extent[0]: extent[0] + extent[2], extent[1]: extent[1] + extent[3]]
            if gt.sum() > 0:
                indices.append(idx)
        return indices

    def _get_tile_extent(self, idx: int) -> Union[Sequence[int], int, int]:
        """
        offset(i) = round(i * (tileSize - float_overlapping_x))
          size(i) = tileSize

        :return: current tile extent as
            [x_tile_offset, y_tile_offset, x_tile_size, y_tile_size], x_tile_index, y_tile_index
        """
        x_tile_index = idx % self.nx
        y_tile_index = int(idx * 1.0 / self.nx)
        x_tile_offset = int(np.round(x_tile_index * (self.patch_size - self.float_overlapping_x)))
        y_tile_offset = int(np.round(y_tile_index * (self.patch_size - self.float_overlapping_y)))

        return [x_tile_offset, y_tile_offset, self.patch_size, self.patch_size], x_tile_index, y_tile_index

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx: int) -> Union[np.ndarray, Sequence[int]]:
        # if idx < 0 or idx >= len(self):
        #     raise IndexError("Index is out of range")
        idx = self.indices[idx]

        # Define ROI to extract
        extent, x_tile_index, y_tile_index = self._get_tile_extent(idx)
        # extent = [xoffset, yoffset, tile_size_x, tile_size_y]

        sample = self.img[extent[0]: extent[0] + extent[2], extent[1]: extent[1] + extent[3]]
        gt = self.gt[extent[0]: extent[0] + extent[2], extent[1]: extent[1] + extent[3]]

        sample = np.asarray(np.copy(sample), dtype="float32")
        gt = np.asarray(np.copy(gt), dtype="int64")

        sample = torch.from_numpy(sample)
        sample = sample / 10**4

        if self.transform is not None:
            sample, gt = self.transform((sample, gt))
        return sample, gt


class PaviaU(HyperspectralDataSet):
    def __init__(self, root_path: str, patch_size: int, min_overlapping: int = 0):
        super().__init__(root_path, patch_size, min_overlapping)
        self.name = 'PaviaU'
        self.wv = np.linspace(0.43, 0.86, 103)
        self.rgb_bands = np.array([65, 28, 2])

    @property
    def labels(self):
        labels_ = {
            1: "Asphalt",
            2: "Meadows",
            3: "Gravel",
            4: "Trees",
            5: "Painted metal sheets",
            6: "Bare Soil",
            7: "Bitumen",
            8: "Self-Blocking Bricks",
            9: "Shadows"
        }

        return labels_

    def load_data(self):
        img = io.loadmat(os.path.join(self.root_path, 'PaviaU.mat'))["paviaU"]
        gt = io.loadmat(os.path.join(self.root_path, 'PaviaU_gt.mat'))["paviaU_gt"]
        return img, gt


class Houston(HyperspectralDataSet):
    def __init__(self, root_path: str, patch_size: int, min_overlapping: int = 0):
        self.name = 'Houston'
        self.bbl = np.ones(48)
        self.bands = np.arange(1, 49)
        self.rgb_bands = np.array([23, 12, 6])
        self.wv = np.array([374.399994,  388.700012,  403.100006,  417.399994,  431.700012,  446.100006,
  460.399994,  474.700012,  489.000000,  503.399994,  517.700012,  532.000000,
  546.299988,  560.599976,  574.900024,  589.200012,  603.599976,  617.900024,
  632.200012,  646.500000,  660.799988,  675.099976,  689.400024,  703.700012,
  718.000000,  732.299988,  746.599976,  760.900024,  775.200012,  789.500000,
  803.799988,  818.099976,  832.400024,  846.700012,  861.099976,  875.400024,
  889.700012,  904.000000,  918.299988,  932.599976,  946.900024,  961.200012,
  975.500000,  989.799988, 1004.200012, 1018.500000, 1032.800049, 1047.099976])
        self.wv = self.wv /1000
        super().__init__(root_path, patch_size, min_overlapping)

    @property
    def labels(self):
        labels_ = {
            0: "Unclassified",
            1: "Healthy grass",
            2: "Stressed grass",
            3: "Artificial turf",
            4: "Evergreen trees",
            5: "Deciduous trees",
            6: "Bare earth",
            7: "Water",
            8: "Residential buildings",
            9: "Non-residential buildings",
            10: "Roads",
            11: "Sidewalks",
            12: "Crosswalks",
            13: "Major thoroughfares",
            14: "Highways",
            15: "Railways",
            16: "Paved parking lots",
            17: "Unpaved parking lots",
            18: "Cars",
            19: "Trains",
            20: "Stadium seats"
        }

        return labels_

    def load_data(self):
        img_path = os.path.join(self.root_path, '20170218_UH_CASI_S4_NAD83.tiff')
        gt_path = os.path.join(self.root_path, '2018_IEEE_GRSS_DFC_GT_TR.tif')

        with rasterio.open(img_path) as src:
            #get data image transform crs shape metadata
            dst_transform = src.transform
            dst_crs = src.crs
            dst_shape = src.height, src.width

            img = src.read()
            # tmp
            img = img[:48, :, :]
            img = img.transpose(1, 2, 0)

        with rasterio.open(gt_path) as src:
            #update metadata of label image with image metadata
            kwargs = src.profile.copy()
            kwargs.update({
                'crs': dst_crs,
                'transform': dst_transform,
                'width': dst_shape[1],
                'height': dst_shape[0]
            })
            #get label header tags
            dst_tags = src.tags(ns=src.driver)

            with rasterio.open(gt_path, 'w', **kwargs) as dst:
                #reproject label image and write it
                reproject(
                    source=rasterio.band(src, 1),
                    destination=rasterio.band(dst, 1),
                    src_transform=src.transform,
                    src_crs=src.crs,
                    dst_transform=dst_transform,
                    dst_crs=dst_crs,
                    resampling=Resampling.nearest)

        with rasterio.open(gt_path) as src:
            gt = src.read()
            gt = gt.reshape(gt.shape[1], gt.shape[2])

        return img, gt