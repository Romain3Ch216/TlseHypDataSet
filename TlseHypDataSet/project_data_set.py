import pdb
import torch
from torchvision import transforms
from utils.transforms import RandomFlip, GaussianFilter, SpectralIndices, GaborFilters, Concat, Stats
from tlse_hyp_data_set import TlseHypDataSet
import matplotlib.pyplot as plt
from other_data_sets import PaviaU
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt


compute_features = False
if compute_features:
    dataset = TlseHypDataSet(
        '/home/rothor/Documents/ONERA/Datasets/Toulouse',
        patch_size=64,
        padding=4)

    dataset.transform = transforms.Compose([
        GaussianFilter(dataset.bbl, sigma=1.5),
        Concat([
            SpectralIndices(dataset.wv),
            GaborFilters()
            ]),
        Stats()
    ])

    features = []

    for i in tqdm(range(len(dataset)), total=len(dataset)):
        sample, gt = dataset.__getitem__(i)
        features.append(sample)

    features = torch.cat(features)
    features = features.numpy()
    np.save('/home/rothor/Documents/ONERA/Datasets/Toulouse/DataSetComparison/tlse_features.npy', features)

    dataset = PaviaU('/home/rothor/Documents/ONERA/Datasets/PaviaU',
                     patch_size=64,
                     min_overlapping=0)


    dataset.transform = transforms.Compose([
        Concat([
            SpectralIndices(dataset.wv),
            GaborFilters()
            ]),
        Stats()
    ])

    features = []

    for i in tqdm(range(len(dataset)), total=len(dataset)):
        sample, gt = dataset.__getitem__(i)
        features.append(sample)

    features = torch.cat(features)
    features = features.numpy()
    np.save('/home/rothor/Documents/ONERA/Datasets/Toulouse/DataSetComparison/paviau_features.npy', features)

tlse_features = np.load('/home/rothor/Documents/ONERA/Datasets/Toulouse/DataSetComparison/tlse_features.npy')
pavia_features = np.load('/home/rothor/Documents/ONERA/Datasets/Toulouse/DataSetComparison/paviau_features.npy')

features = np.concatenate((tlse_features, pavia_features), axis=0)
data_sets = np.concatenate((np.zeros(tlse_features.shape[0]), np.ones(pavia_features.shape[0])), axis=0)

TSNE = True
if TSNE:
    from sklearn.manifold import TSNE
    proj = TSNE(n_components=2).fit_transform(features)

else:
    from sklearn.decomposition import PCA
    proj = PCA(n_components=2).fit_transform(features)

tlse_proj = proj[data_sets == 0]
pavia_proj = proj[data_sets == 1]

fig = plt.figure()
plt.scatter(tlse_proj[:, 0], tlse_proj[:, 1], color='green', alpha=0.3)
plt.scatter(pavia_proj[:, 0], pavia_proj[:, 1], color='blue', alpha=0.3)
plt.show()

