import pdb

import torch
import numpy as np
from scipy.ndimage import gaussian_filter1d
from TlseHypDataSet.utils.spectral import SpectralWrapper, get_continuous_bands
from TlseHypDataSet.utils.spectral_indices import *
from skimage.filters import gabor
from torchvision import transforms

__all__ = [
    'RandomFlip',
    'GaussianFilter',
    'SpectralIndices'
]


class GaussianConvolution(torch.nn.Module):
    def __init__(self, sigma, n_channels):
        super(GaussianConvolution, self).__init__()
        self.sigma = sigma
        self.n_channels = n_channels

    def forward(self, x):
        x = gaussian_filter1d(x, sigma=self.sigma)
        return x


class GaussianFilter(object):
    """
    Like-wise torchvision.transforms class to apply 1D Gaussian filters on the spectral dimension
    """

    def __init__(self, bbl: np.ndarray, sigma: float):
        n_bands = get_continuous_bands(bbl)
        filters = {}
        for i in range(len(n_bands)):
            filters[f'conv-{i}'] = GaussianConvolution(sigma=sigma, n_channels=n_bands[i])
        self.filter = SpectralWrapper(filters)

    def __call__(self, data) -> torch.Tensor:
        sample, gt = data
        return self.filter(sample), gt


class RandomFlip(object):
    """

    """

    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, data):
        """
        :param sample: input image [height, width, n_bands], extent, neighbours_extent
        :return: features [height, width, n_spectral_indices], extent, neighbours_extent
        """
        sample, gt = data
        tensors = [sample, gt]
        horizontal = torch.rand(1).item() > self.p
        vertical = torch.rand(1).item() > self.p
        if horizontal:
            tensors = [torch.fliplr(T) for T in tensors]
        if vertical:
            tensors = [torch.flipud(T) for T in tensors]
        return tensors[0], tensors[1]


class SpectralIndices(object):
    """
    Like-wise torchvision.transforms class to compute spectral indices
    """

    def __init__(self, wv: np.ndarray):
        self.spectral_indices = (
            NDVI(wv),
            ANVI(wv),
            CI(wv),
            NDVI_RE(wv),
            VgNIR_BI(wv),
            SAVI(wv),
            Identity(wv, wv_max=0.86)
        )

    def __call__(self, data):
        """
        :param data: input image [height, width, n_bands]
        :return: features [height, width, n_spectral_indices]
        """
        sample, gt = data
        features = torch.zeros((sample.shape[0], sample.shape[1], sum([index.dim for index in self.spectral_indices])))
        for i, spectral_index in enumerate(self.spectral_indices):
            index = spectral_index(sample)
            if len(index.shape) > 2:
                features[:, :, i:i + spectral_index.dim] = index
            else:
                features[:, :, i:i + spectral_index.dim] = index.unsqueeze(-1)
        return features, gt


class GaborFilters(object):
    """
    Like-wise torchvision.transforms class to compute Gabor filters
    """

    def __init__(self, n_frequencies: int = 4, n_thetas: int = 6):
        self.n_frequencies = n_frequencies
        self.n_thetas = n_thetas
        self.frequencies = np.linspace(0.05, 0.5, n_frequencies)
        self.thetas = np.linspace(0, np.pi / 2, n_thetas)

    def __call__(self, data):
        """
        :param data: input image [height, width, n_bands]
        :return: features [height, width, n_frequencies x n_thetas]
        """
        sample, gt = data
        sample = sample.numpy()
        sample = np.mean(sample, axis=-1)
        features = torch.zeros((sample.shape[0], sample.shape[1], self.n_frequencies * self.n_thetas))
        k = 0
        for freq in self.frequencies:
            for theta in self.thetas:
                real, imag = gabor(sample, frequency=freq, theta=theta)
                feature = real + imag
                feature = (feature - np.min(feature, axis=(0, 1))) / (
                            np.max(feature, axis=(0, 1)) - np.min(feature, axis=(0, 1)))
                features[:, :, k] = torch.from_numpy(feature)
                k += 1
        return features, gt


class Stats(object):
    """
    Like-wise torchvision.transforms class to compute statistics (mean, standard deviation, first and third quartiles, first and last deciles, minimum and maximum) over a 2D image
    """

    def __init__(self):
        self.statistics = (
            lambda x: torch.mean(x, dim=(0, 1)),
            lambda x: torch.std(x, dim=(0, 1)),
            lambda x: torch.quantile(x.reshape(-1, x.shape[-1]), 0.10, dim=0),
            lambda x: torch.quantile(x.reshape(-1, x.shape[-1]), 0.25, dim=0),
            lambda x: torch.quantile(x.reshape(-1, x.shape[-1]), 0.75, dim=0),
            lambda x: torch.quantile(x.reshape(-1, x.shape[-1]), 0.90, dim=0),
            lambda x: torch.min(x),
            lambda x: torch.max(x)
        )
        self.out_channels = len(self.statistics)

    def __call__(self, data):
        sample, gt = data
        statistics = torch.zeros(sample.shape[-1], self.out_channels)
        # statistics.shape = [number_of_input_channels, number_of_statistics]

        for i, stat in enumerate(self.statistics):
            statistics[:, i] = stat(sample)

        statistics = statistics.view(1, -1)
        return statistics, gt


class Concat(transforms.Compose):
    """
    Concatenate several data transformations
    """

    def __init__(self, transforms_):
        super().__init__(transforms_)

    def __call__(self, data):
        sample, gt = data
        out = []
        for t in self.transforms:
            sample, _ = t((sample, gt))
            out.append(sample)
        return torch.cat(out, dim=-1), gt

