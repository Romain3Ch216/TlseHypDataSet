import pdb

import torch
import torch.nn as nn
import numpy as np
from typing import List


class SpectralWrapper(nn.Module):
    """
    Converts a dict of CNNs (one for each continous spectral domain)
    into a single CNN.
    """
    def __init__(self, models):
        super(SpectralWrapper, self).__init__()
        self.models = nn.ModuleDict(models)

    @property
    def out_channels(self):
        with torch.no_grad():
            n_channels = sum([model.n_channels for model in self.models.values()])
            x = torch.ones((2, n_channels))
            x = self.forward(x)
        return x.numel()//2

    def forward(self, x):
        z, B = {}, 0
        if len(x.shape) < 3:
            x = x.unsqueeze(1)
        for model_id, model in self.models.items():
            out = model(x[:, :, B:B+model.n_channels])
            if isinstance(out, np.ndarray):
                out = torch.from_numpy(out)
            z[model_id] = out
            B += model.n_channels

        keys = list(z.keys())
        out = torch.cat([z[keys[i]] for i in range(len(z))], dim=-1)

        return out


def get_continuous_bands(bbl: np.ndarray) -> List[int]:
    n_bands = []
    good_bands = np.where(bbl == True)[0]
    s = 1
    for i in range(len(good_bands)-1):
        if good_bands[i] == good_bands[i+1]-1:
            s += 1
        else:
            n_bands.append(s)
            s = 1

    n_bands.append(s)
    return n_bands


class SpectralIndex:
    """
    Generic class for spectral indices
    """
    def __init__(self, wv: np.ndarray, epsilon: float = 1e-4):
        self.wv = torch.from_numpy(wv).view(1, -1)
        self.epsilon = epsilon
        self.dim = 1

    def _get_bands(self, lambdas: List[float]) -> List[int]:
        """
        :args: wavelengths of the spectral index
        :return: closest bands to the given wavelengths
        """
        wv = self.wv.repeat(len(lambdas), 1)
        lambdas = torch.Tensor(lambdas).view(-1, 1)
        diff = torch.abs(lambdas - wv)
        bands = torch.argmin(diff, dim=1)
        return bands

    def ceil(self, index: torch.Tensor) -> torch.Tensor:
        return torch.clamp(index, min=-6, max=6)

    def __call__(self, data):
        raise NotImplementedError


def full_spectra(spectra, bbl):
    """
    Args:
    - spectra: npy array, HS cube
    - bbl: npy boolean array, masked bands
    Output:
    HS cube with NaN at masked band locations
    """
    bbl = np.array(bbl).astype(bool)
    res = np.zeros((spectra.shape[0],len(bbl)))
    res[:, bbl] = spectra
    res[:, bbl == False] = np.nan
    return res
