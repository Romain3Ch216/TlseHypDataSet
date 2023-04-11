import pdb

import torch
from torchvision import transforms
from TlseHypDataSet.utils.transforms import RandomFlip, GaussianFilter, SpectralIndices, GaborFilters, Concat, Stats
from TlseHypDataSet.other_data_sets import PaviaU
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from TlseHypDataSet.utils.utils import make_dirs
import os
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA

__all__ = [
    'project_features'
]


def compute_dataset_features(dataset):
    path = os.path.join(dataset.root_path, 'outputs', 'data_set_features.npy')
    if ('outputs' in os.listdir(dataset.root_path)) and ('data_set_features.npy' in os.listdir(os.path.join(dataset.root_path, 'outputs'))):
        features = np.load(path)
    else:
        if dataset.name == 'Toulouse':
            dataset.transform = transforms.Compose([
                GaussianFilter(dataset.bbl, sigma=1.5),
                Concat([
                    SpectralIndices(dataset.wv),
                    GaborFilters()
                    ]),
                Stats()
            ])

        elif dataset.name == 'PaviaU':
            dataset.transform = transforms.Compose([
                Concat([
                    SpectralIndices(dataset.wv),
                    GaborFilters()
                    ]),
                Stats()
            ])

        elif dataset.name == 'Houston':
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
        make_dirs([os.path.join(dataset.root_path, 'outputs')])
        np.save(path, features)
    return features


# def compute_spectra_features(dataset):
#     path = os.path.join(dataset.root_path, 'outputs', 'spectra_features.npy')
#     if ('outputs' in os.listdir(dataset.root_path)) \
#             and ('spectra_features.npy' in os.listdir(os.path.join(dataset.root_path, 'outputs'))):
#         features = np.load(path)
#     else:
#         if dataset.name == 'Toulouse':
#             dataset.transform = GaussianFilter(dataset.bbl, sigma=1.5)
#         features = []
#         for i in tqdm(range(len(dataset)), total=len(dataset)):
#             sample, gt = dataset.__getitem__(i)
#             features.append(sample)


def project_features(datasets, proj='TSNE', save_to=None):
    colors = ['green', 'blue', 'orange']
    features = []
    set_id = []
    for i, dataset in enumerate(datasets):
        data_features = compute_dataset_features(dataset)
        nan_mask = np.sum(np.isnan(data_features), axis=1) == 0
        data_features = data_features[nan_mask]
        features.extend(data_features)
        set_id.extend([i] * len(data_features))
    set_id = np.array(set_id)

    if proj == 'TSNE':
        proj = TSNE(n_components=2).fit_transform(features)
    elif proj == 'PCA':
        proj = PCA(n_components=2).fit_transform(features)

    fig = plt.figure(figsize=(15, 15))
    for i in range(len(datasets)):
        plt.scatter(proj[set_id == i, 0], proj[set_id == i, 1], color=colors[i], alpha=0.3)
    if save_to is None:
        plt.show()
    else:
        plt.savefig(save_to, dpi=150, bbox_inches='tight', pad_inches=0.05)




