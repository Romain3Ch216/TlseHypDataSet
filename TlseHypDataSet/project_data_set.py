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
    'project_features',
    'project_features_class'
]


def compute_dataset_features(dataset, ftype='spectral'):
    path = os.path.join(dataset.root_path, 'outputs')
    if ('outputs' in os.listdir(dataset.root_path)) and (f'{dataset.name}_{ftype}_features.npy' in os.listdir(os.path.join(dataset.root_path, 'outputs'))):
        features = np.load(os.path.join(path, f'{dataset.name}_{ftype}_features.npy'))
        labels = np.load(os.path.join(path, f'{dataset.name}_{ftype}_labels.npy'))
    else:
        if dataset.name == 'Toulouse':
            if ftype == 'spectral':
                dataset.transform = transforms.Compose([
                    GaussianFilter(dataset.bbl, sigma=1.5),
                    SpectralIndices(dataset.wv),
                    Stats()
                ])
            if ftype == 'spatial':
                dataset.transform = transforms.Compose([
                    GaussianFilter(dataset.bbl, sigma=1.5),
                    GaborFilters(),
                    Stats()
                ])
            elif ftype == 'both':
                dataset.transform = transforms.Compose([
                    GaussianFilter(dataset.bbl, sigma=1.5),
                    Concat([
                        SpectralIndices(dataset.wv),
                        GaborFilters()
                        ]),
                    Stats()
                ])

        else:
            if ftype == 'spectral':
                dataset.transform = transforms.Compose([
                    SpectralIndices(dataset.wv),
                    Stats()
                ])
            elif ftype =='spatial':
                dataset.transform = transforms.Compose([
                    GaborFilters(),
                    Stats()
                ])
            elif ftype == 'both':
                dataset.transform = transforms.Compose([
                    Concat([
                        SpectralIndices(dataset.wv),
                        GaborFilters()
                        ]),
                    Stats()
                ])


        features = []
        labels = []
        for i in tqdm(range(len(dataset)), total=len(dataset)):
            sample, gt = dataset.__getitem__(i)
            n_pixels = gt.shape[0] * gt.shape[1]
            gt = gt[gt != 0] - 1
            y, counts = np.unique(gt, return_counts=True)
            proportions = np.zeros(dataset.n_classes)
            proportions[y] = counts
            proportions = proportions / n_pixels
            features.append(sample)
            labels.append(proportions)

        features = torch.cat(features).numpy()
        labels = np.stack(labels)
        make_dirs([os.path.join(dataset.root_path, 'outputs')])
        np.save(os.path.join(path, f'{dataset.name}_{ftype}_features.npy'), features)
        np.save(os.path.join(path, f'{dataset.name}_{ftype}_labels.npy'), labels)
    return features, labels


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
    colors = ['#FF697F', '#6D88E1', '#595753']
    features = []
    set_id = []
    for i, dataset in enumerate(datasets):
        data_features, _ = compute_dataset_features(dataset)
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


def project_features_class(dataset, class_id, save_to=None, proj='TSNE'):
    data_features, labels = compute_dataset_features(dataset)
    nan_mask = np.sum(np.isnan(data_features), axis=1) == 0
    data_features = data_features[nan_mask]
    labels = labels[nan_mask]

    if proj == 'TSNE':
        proj = TSNE(n_components=2).fit_transform(data_features)
    elif proj == 'PCA':
        proj = PCA(n_components=2).fit_transform(data_features)

    fig = plt.figure(figsize=(15, 15))
    plt.scatter(proj[:, 0], proj[:, 1], c=labels[:, class_id], cmap='Blues')
    if save_to is None:
        plt.show()
    else:
        plt.savefig(save_to, dpi=150, bbox_inches='tight', pad_inches=0.05)




