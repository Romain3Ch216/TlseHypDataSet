import pdb

import torch
import numpy as np
from TlseHypDataSet.utils.spectral import SpectralIndex


class Identity(SpectralIndex):
    """
    Returns identity for given spectral bands
    """
    def __init__(self, wv: np.ndarray, chunk: int = 20, wv_max=None):
        super().__init__(wv)
        if wv_max is not None:
            n = np.sum(wv < wv_max)
        else:
            n = len(wv) - 1
        self.bands = np.linspace(0, n, chunk).astype(int)
        self.dim = len(self.bands)

    def __call__(self, data: torch.Tensor) -> torch.Tensor:
        out = data[:, :, self.bands]
        return out


class NDVI(SpectralIndex):
    """
    Normalized Difference Vegetation Index
    NDVI := \frac{R-NIR}{R+NIR}
    """

    def __init__(self, wv: np.ndarray):
        super().__init__(wv)
        self.lambdas = [0.65, 0.80]

    def __call__(self, data: torch.Tensor) -> torch.Tensor:
        bands = self._get_bands(self.lambdas)
        R = data[:, :, bands[0]]
        NIR = data[:, :, bands[1]]
        ndvi = (NIR - R) / (R + NIR + self.epsilon)
        return self.ceil(ndvi)


class ANVI(SpectralIndex):
    """
    Advanced Normalized Vegetation Index
    ANVI := \frac{NIR-B}{B+NIR}

    Peña-Barragán J.M., López-Granados F., Jurado-Expósito M., García-Torres L. (2006).
    Mapping Ridolfia segetum patches in sunflower crop using remote sensing. Weed
    Research 47, 164–172.
    """

    def __init__(self, wv: np.ndarray):
        super().__init__(wv)
        self.lambdas = [0.45, 0.80]

    def __call__(self, data: torch.Tensor) -> torch.Tensor:
        bands = self._get_bands(self.lambdas)
        B = data[:, :, bands[0]]
        NIR = data[:, :, bands[1]]
        anvi = (NIR - B) / (B + NIR + self.epsilon)
        return self.ceil(anvi)


class CI(SpectralIndex):
    """
    Chlorophyll Index
    CI := \frac{NIR}{G}-1

    Bausch W.C., Khosla R. (2010). QuickBird satellite versus ground-based multi-spectral data
    for estimating nitrogen status of irrigated maize. Precision Agriculture 11, 274–290.
    """

    def __init__(self, wv: np.ndarray):
        super().__init__(wv)
        self.lambdas = [0.55, 0.80]

    def __call__(self, data: torch.Tensor) -> torch.Tensor:
        bands = self._get_bands(self.lambdas)
        G = data[:, :, bands[0]]
        NIR = data[:, :, bands[1]]
        ci = NIR / (G + self.epsilon) - 1
        return self.ceil(ci)


class NDVI_RE(SpectralIndex):
    """
    Normalized Difference Red-Edge Vegetation Index
    NDVI_RE := \frac{NIR - RE}{NIR + RE}

    Gitelson, A.; Merzlyak, M.N. Spectral Reflectance Changes Associated with Autumn Senescence of
    Aesculus hippocastanum L. and Acer platanoides L. Leaves. Spectral Features and Relation to Chlorophyll Estimation.
    J. Plant Physiol. 1994, 143, 286–292.
    """

    def __init__(self, wv: np.ndarray):
        super().__init__(wv)
        self.lambdas = [0.73, 0.84]

    def __call__(self, data: torch.Tensor) -> torch.Tensor:
        bands = self._get_bands(self.lambdas)
        RE = data[:, :, bands[0]]
        NIR = data[:, :, bands[1]]
        ndvi_re = (NIR - RE) / (NIR + RE + self.epsilon)
        return self.ceil(ndvi_re)


class NDII(SpectralIndex):
    """
    Normalized Difference Infrared Index
    NDII := \frac{NIR-SWIR}{NIR+SWIR}

    Cheng T., Riaño D., Koltunov A., Whiting M.L., Ustin S.L., Rodriguez J. (2013). Detection of
    diurnal variation in orchard canopy water content using MODIS/ASTER airborne
    simulator (MASTER) data. Remote Sensing of Environment 132, 1–12.
    """

    def __init__(self, wv: np.ndarray):
        super().__init__(wv)
        self.lambdas = [0.86, 1.67]

    def __call__(self, data: torch.Tensor) -> torch.Tensor:
        bands = self._get_bands(self.lambdas)
        NIR = data[:, :, bands[0]]
        SWIR = data[:, :, bands[1]]
        ndii = (NIR - SWIR) / (NIR + SWIR + self.epsilon)
        return ndii


class NDBI(SpectralIndex):
    """
    Normalized Difference Built-up Index
    NDBI := \frac{MIR-NIR}{MIR+NIR}

    Zha, Y., Gao, Y., & Nt, S. (2003). Use of normalized difference built-
    up index in automatically mapping urban areas from TM imagery.
    International Journal of Remote Sensing, 24(3), 583–594.
    """

    def __init__(self, wv: np.ndarray):
        super().__init__(wv)
        self.lambdas = [0.80, 2.2]

    def __call__(self, data: torch.Tensor) -> torch.Tensor:
        bands = self._get_bands(self.lambdas)
        NIR = data[:, :, bands[0]]
        MIR = data[:, :, bands[1]]
        ndbi = (MIR - NIR) / (MIR + NIR + self.epsilon)
        return self.ceil(ndbi)


class VgNIR_BI(SpectralIndex):
    """
    Visible green-based built-up index
    VgNIR_BI := \frac{G-NIR}{G+NIR}

    Estoque CR, Murayama Y. 2015. Classification and change detection of built-up lands from Landsat-7
    ETM þ and Landsat-8 OLI/TIRS imageries: a comparative assessment of various spectral indices. Ecol
    Indic. 56:205–217.
    """

    def __init__(self, wv: np.ndarray):
        super().__init__(wv)
        self.lambdas = [0.55, 0.80]

    def __call__(self, data: torch.Tensor) -> torch.Tensor:
        bands = self._get_bands(self.lambdas)
        G = data[:, :, bands[0]]
        NIR = data[:, :, bands[1]]
        vgnir_bi = (G - NIR) / (G + NIR + self.epsilon)
        return self.ceil(vgnir_bi)


class BAEI(SpectralIndex):
    """
    Built-up Area Extraction Method
    BAEI := \frac{R+0.3}{G+SWIR1}

    Bhatti SS, Tripathi NK. 2014. Built-up area extraction using Landsat 8 OLI imagery. GISc Remote Sens.
    51(4):445–467.
    """

    def __init__(self, wv: np.ndarray):
        super().__init__(wv)
        self.lambdas = [0.55, 0.65, 1.6]

    def __call__(self, data: torch.Tensor) -> torch.Tensor:
        bands = self._get_bands(self.lambdas)
        G = data[:, :, bands[0]]
        R = data[:, :, bands[1]]
        SWIR1 = data[:, :, bands[2]]
        baei = (R+0.3) / (G + SWIR1 + self.epsilon)
        return self.ceil(baei)


class SAVI(SpectralIndex):
    """
    Soil Adjusted Vegetation Index
    SAVI := \frac{(NIR - R) * (1 + l) }{NIR + R + l}

    Huete, A. R. (1988). A soil-adjusted vegetation index (SAVI). Remote sensing of environment, 25(3), 295-309.
    """

    def __init__(self, wv: np.ndarray, l: float = 0.5):
        super().__init__(wv)
        self.lambdas = [0.65, 0.8]
        self.l = l

    def __call__(self, data: torch.Tensor) -> torch.Tensor:
        bands = self._get_bands(self.lambdas)
        R = data[:, :, bands[0]]
        NIR = data[:, :, bands[1]]
        savi = (NIR - R)*(1 + self.l) / (NIR + R + self.l)
        return self.ceil(savi)


class MNDWI(SpectralIndex):
    """
    Modified Normalized Difference Water Index
    MNDWI := \frac{G - MIR}{G + MIR}

     Hanqiu Xu (2006) Modification of normalised difference water index (NDWI) to enhance open water features in
     remotely sensed imagery, International Journal of Remote Sensing, 27:14, 3025-3033, DOI: 10.1080/01431160600589179
    """

    def __init__(self, wv: np.ndarray):
        super().__init__(wv)
        self.lambdas = [0.55, 2.2]

    def __call__(self, data: torch.Tensor) -> torch.Tensor:
        bands = self._get_bands(self.lambdas)
        G = data[:, :, bands[0]]
        MIR = data[:, :, bands[1]]
        mndwi = (G - MIR) / (G + MIR + self.epsilon)
        return self.ceil(mndwi)


class IBI(SpectralIndex):
    """
    Index-based Built-up Index
    IBI := \frac{NDBI - (SAVI + MNDWI)/2}{NDBI + (SAVI + MNDWI)/2}

    Xu H. 2008. A new index for delineating built-up land features in satellite imagery. Int J Remote Sens.
    29(14):4269–4276.
    """

    def __init__(self, wv: np.ndarray):
        super().__init__(wv)
        self.ndbi = NDBI(wv)
        self.savi = SAVI(wv)
        self.mndwi = MNDWI(wv)

    def __call__(self, data: torch.Tensor) -> torch.Tensor:
        num = self.ndbi(data) - (self.savi(data)+self.mndwi(data))/2
        den = self.ndbi(data) + (self.savi(data)+self.mndwi(data))/2 + self.epsilon
        return self.ceil(num/den)


class ClayIndex(SpectralIndex):
    """
    """

    def __init__(self, wv: np.ndarray):
        super().__init__(wv)
        self.lambdas = [1.19, 2.22, 2.25]

    def __call__(self, data: torch.Tensor) -> torch.Tensor:
        bands = self._get_bands(self.lambdas)
        indice = (data[:, :, bands[0]] + data[:, :, bands[2]] - data[:, :, bands[1]]) / \
                 (data[:, :, bands[0]] + data[:, :, bands[1]] + data[:, :, bands[2]])
        return indice


class GravelIndex(SpectralIndex):
    """
    """

    def __init__(self, wv: np.ndarray):
        super().__init__(wv)
        self.lambdas = [2.25, 2.3, 2.35]

    def __call__(self, data: torch.Tensor) -> torch.Tensor:
        bands = self._get_bands(self.lambdas)
        indice = (data[:, :, bands[0]] + data[:, :, bands[2]] - data[:, :, bands[1]]) / \
                 (data[:, :, bands[0]] + data[:, :, bands[1]] + data[:, :, bands[2]])
        return indice


class PlasticIndex(SpectralIndex):
    """
    """

    def __init__(self, wv: np.ndarray):
        super().__init__(wv)
        self.lambdas = [1.69, 1.72, 1.75]

    def __call__(self, data: torch.Tensor) -> torch.Tensor:
        bands = self._get_bands(self.lambdas)
        indice = (data[:, :, bands[0]] + data[:, :, bands[2]] - data[:, :, bands[1]]) / \
                 (data[:, :, bands[0]] + data[:, :, bands[1]] + data[:, :, bands[2]])
        return indice