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

        for model_id, model in self.models.items():
            z[model_id] = model(x[:, B:B+model.n_channels])
            B += model.n_channels

        keys = list(z.keys())
        out = torch.cat([z[keys[i]] for i in range(len(z))], dim=1)

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