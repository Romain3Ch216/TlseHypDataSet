import torch
from scipy.ndimage import gaussian_filter1d
from TlseHypDataSet.utils.spectral import SpectralWrapper, get_continuous_bands
import numpy as np


__all__ = [
    'GaussianFilter'
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

    """
    def __init__(self, bbl: np.ndarray, sigma: float):
        n_bands = get_continuous_bands(bbl)
        filters = {}
        for i in range(len(n_bands)):
            filters[f'conv-{i}'] = GaussianConvolution(sigma=1.5, n_channels=n_bands[i])
        self.filter = SpectralWrapper(filters)

    def __call__(self, sample: torch.Tensor) -> torch.Tensor:
        """
        :param sample:
        :return:
        """
        return self.filter(sample)

