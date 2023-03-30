import torch
from torch.utils.data import Dataset
from TlseHypDataSet.utils.geometry import _compute_number_of_tiles, _compute_float_overlapping, ceil_int
from typing import Union, Sequence, List
import numpy as np
from scipy import io
import os


class HyperspectralDataSet(Dataset):
    def __init__(self, root_path: str, patch_size: int, min_overlapping: int):
        self.root_path = root_path
        self.patch_size = patch_size
        self.min_overlapping = min_overlapping
        self.transform = None

        self.image_path = None
        self.gt_path = None
        self.wv = np.linspace(0.43, 0.86, 103)
        self.bbl = None
        self.E_dir = None
        self.E_dif = None

        self.img, self.gt = self.load_data()
        self.compute_patches()

    @property
    def labels(self):
        raise NotImplementedError

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
    def __init__(self, root_path: str, patch_size: int, min_overlapping: int):
        super().__init__(root_path, patch_size, min_overlapping)
        self.name = 'PaviaU'

    @property
    def labels(self):
        labels_ = {
            0: "Undefined",
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
