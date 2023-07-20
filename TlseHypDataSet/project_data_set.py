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
        labels = np.load(os.path.join(path, f'{dataset.name}_labels.npy'))
    else:
        if dataset.name == 'Toulouse':
            if ftype == 'spectral':
                dataset.transform = transforms.Compose([
                    GaussianFilter(dataset.bbl, sigma=1.5),
                    SpectralIndices(dataset.wv[dataset.bbl]),
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
                        SpectralIndices(dataset.wv[dataset.bbl]),
                        GaborFilters()
                        ]),
                    Stats()
                ])

        else:
            if ftype == 'spectral':
                dataset.transform = transforms.Compose([
                    SpectralIndices(dataset.wv[dataset.bbl]),
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
                        SpectralIndices(dataset.wv[dataset.bbl]),
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
        np.save(os.path.join(path, f'{dataset.name}_labels.npy'), labels)
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


def project_features(datasets, ftype, proj='TSNE', restore_proj=None, save_to=None, images=False):
    colors = ['#FF697F', '#6D88E1', '#595753']


    features = []
    set_id = []
    labels = [[], [], []]
    for i, dataset in enumerate(datasets):
        data_features, labels_ = compute_dataset_features(dataset, ftype)
        nan_mask = np.sum(np.isnan(data_features), axis=1) == 0
        data_features = data_features[nan_mask]
        features.extend(data_features)
        labels[i].extend(labels_)
        set_id.extend([i] * len(data_features))
    set_id = np.array(set_id)
    features = np.stack(features)
    labels = [np.stack(labels[i]) for i in range(len(labels))]

    if restore_proj is None:
        if proj == 'TSNE':
            proj = TSNE(n_components=2).fit_transform(features)
        elif proj == 'PCA':
            proj = PCA(n_components=2).fit_transform(features)
    else:
        proj = np.load(restore_proj)

    fig, ax = plt.subplots(figsize=(15, 15))
    cmaps = ['Reds', 'Blues', 'Greens']
    for i in range(len(datasets)):
        X = proj[set_id == i, 0]
        Y = proj[set_id == i, 1]

        # for j in range(np.sum(set_id == i)):
        #     ax.annotate(j, (X[j], Y[j]))
        # plt.scatter(proj[set_id == i, 0], proj[set_id == i, 1], c=np.sum(labels[i], axis=1), cmap=cmaps[i])
        if images:
            if i == 2:
                id = 111
                plt.arrow(X[id], Y[id], 15, +10)
                plt.arrow(X[id], Y[id], 35, +10)
                img, _ = datasets[i].__getitem__(id)
                plt.imshow(img[:, :, datasets[i].rgb_bands], extent=(X[id]+15, X[id]+35, Y[id]+10, Y[id]+30))
                #
                id = 21
                plt.arrow(X[id], Y[id], 15, 5)
                plt.arrow(X[id], Y[id], 15, 25)
                img, _ = datasets[i].__getitem__(id)
                plt.imshow(img[:, :, datasets[i].rgb_bands], extent=(X[id]+15, X[id]+35, Y[id]+5, Y[id]+25))

                id = 177
                plt.arrow(X[id], Y[id], 5, -10)
                plt.arrow(X[id], Y[id], 25, -10)
                img, _ = datasets[i].__getitem__(id)
                plt.imshow(img[:, :, datasets[i].rgb_bands].transpose(0, 1), extent=(X[id]+5, X[id]+25, Y[id]-10, Y[id]-30))
                #
                # id = 5
                # plt.arrow(X[id], Y[id], 10, 10)
                # plt.arrow(X[id], Y[id], 10, -10)
                # img, _ = datasets[i].__getitem__(id)
                # plt.imshow(img[:, :, datasets[i].rgb_bands], extent=(X[id]+10, X[id]+30, Y[id]-10, Y[id]+10))

            if i == 1:
                # id = 2
                # plt.arrow(X[id], Y[id], -20, -10)
                # plt.arrow(X[id], Y[id], -20, 10)
                # img, _ = datasets[i].__getitem__(id)
                # plt.imshow(2 * img[:, :, datasets[i].rgb_bands], extent=(X[id]-40, X[id]-20, Y[id]+10, Y[id]-10))

                id = 30
                plt.arrow(X[id], Y[id], -10, 20)
                plt.arrow(X[id], Y[id], -10, 0)
                img, _ = datasets[i].__getitem__(id)
                plt.imshow(2 * img[:, :, datasets[i].rgb_bands], extent=(X[id]-30, X[id]-10, Y[id], Y[id]+20))

                id = 29
                plt.arrow(X[id], Y[id], -10, 25)
                plt.arrow(X[id], Y[id], -10, 5)
                img, _ = datasets[i].__getitem__(id)
                plt.imshow(3 * img[:, :, datasets[i].rgb_bands], extent=(X[id]-30, X[id]-10, Y[id]+5, Y[id]+25))

                id = 47
                plt.arrow(X[id], Y[id], 5, -5)
                plt.arrow(X[id], Y[id], 5, -25)
                img, _ = datasets[i].__getitem__(id)
                plt.imshow(2 * img[:, :, datasets[i].rgb_bands], extent=(X[id]+5, X[id]+25, Y[id]-5, Y[id]-25))

                # id = 5
                # plt.arrow(X[id], Y[id], +20, +10)
                # plt.arrow(X[id], Y[id], +20, -10)
                # img, _ = datasets[i].__getitem__(id)
                # plt.imshow(2 * img[:, :, datasets[i].rgb_bands], extent=(X[id]+20, X[id]+40, Y[id]+10, Y[id]-10))

            if i == 0:
                id = 512
                plt.arrow(X[id], Y[id], 15, +20)
                plt.arrow(X[id], Y[id], 15, +0)
                img, _ = datasets[i].__getitem__(id)
                plt.imshow(2 * img[:, :, datasets[i].rgb_bands], extent=(X[id]+15, X[id]+35, Y[id]+20, Y[id]+0))

                id = 590
                plt.arrow(X[id], Y[id], -5, +10)
                plt.arrow(X[id], Y[id], +15, +10)
                img, _ = datasets[i].__getitem__(id)
                plt.imshow(4 * img[:, :, datasets[i].rgb_bands].transpose(0, 1), extent=(X[id]-5, X[id]+15, Y[id]+30, Y[id]+10))

                id = 471
                plt.arrow(X[id], Y[id], -10, 20)
                plt.arrow(X[id], Y[id], 10, 20)
                img, _ = datasets[i].__getitem__(id)
                plt.imshow(4 * img[:, :, datasets[i].rgb_bands], extent=(X[id]-10, X[id]+10, Y[id]+20, Y[id]+40))


                id = 607
                plt.arrow(X[id], Y[id], -10, -25)
                plt.arrow(X[id], Y[id], -10, -5)
                img, _ = datasets[i].__getitem__(id)
                plt.imshow(2 * img[:, :, datasets[i].rgb_bands], extent=(X[id]-10, X[id]-30, Y[id]-25, Y[id]-5))

                id = 681
                plt.arrow(X[id], Y[id], 10, -20)
                plt.arrow(X[id], Y[id], 10, 0)
                img, _ = datasets[i].__getitem__(id)
                plt.imshow(2 * img[:, :, datasets[i].rgb_bands], extent=(X[id]+10, X[id]+30, Y[id], Y[id]-20))

                id = 669
                plt.arrow(X[id], Y[id], -5, -10)
                plt.arrow(X[id], Y[id], -25, -10)
                img, _ = datasets[i].__getitem__(id)
                plt.imshow(2 * img[:, :, datasets[i].rgb_bands], extent=(X[id]-25, X[id]-5, Y[id]-10, Y[id]-30))

        plt.scatter(X, Y, color=colors[i], alpha=0.8)

    ax.text(-65, -60, 'Toulouse', color=colors[0], fontsize=25)
    ax.text(-65, -55, 'Pavia University', color=colors[1], fontsize=25)
    ax.text(-65, -50, 'Houston University', color=colors[2], fontsize=25)
    plt.xticks([])
    plt.yticks([])
    plt.axis('off')
    # img_id = 0
    # X = proj[set_id == img_id, 0]
    # Y = proj[set_id == img_id, 1]
    # id = 383
    # plt.arrow(X[id], Y[id], -20, 10)
    # plt.arrow(X[id], Y[id], -20, -10)
    # img, _ = datasets[img_id].__getitem__(id)
    # plt.imshow(2 * img[:, :, datasets[img_id].rgb_bands], extent=(X[id]-40, X[id]-20, Y[id]-10, Y[id]+10))


    if save_to is None:
        plt.show()
    else:
        plt.savefig(os.path.join(save_to, f'{ftype}_proj.pdf'), dpi=150, bbox_inches='tight', pad_inches=0.05)
        np.save(os.path.join(save_to, f'{ftype}_proj.npy'), proj)
        np.save(os.path.join(save_to, f'{ftype}_sets.npy'), set_id)


def project_features_class(dataset, class_id, ftype, restore_proj=None, save_to=None, proj='TSNE'):
    data_features, labels = compute_dataset_features(dataset, ftype)
    nan_mask = np.sum(np.isnan(data_features), axis=1) == 0
    data_features = data_features[nan_mask]
    labels = labels[nan_mask]

    if restore_proj is None:
        if proj == 'TSNE':
            proj = TSNE(n_components=2).fit_transform(data_features)
        elif proj == 'PCA':
            proj = PCA(n_components=2).fit_transform(data_features)
    else:
        proj = np.load(f'{ftype}_proj.npy')
        sets = np.load(f'{ftype}_sets.npy')
        if restore_proj == 'toulouse':
            proj = proj[sets == 0]
        elif restore_proj == 'pavia':
            proj = proj[sets == 1]
        elif restore_proj == 'houston':
            proj = proj[sets == 2]

    if class_id == 'all':
        for class_id in range(labels.shape[-1]):
            fig = plt.figure(figsize=(15, 15))
            plt.scatter(proj[:, 0], proj[:, 1], c=labels[:, class_id], cmap='Blues')
            plt.show()
    else:
        fig = plt.figure(figsize=(15, 15))
        plt.scatter(proj[:, 0], proj[:, 1], c=np.sum(labels[:, np.array(class_id)], axis=1), cmap='Blues')
        if save_to is None:
            plt.show()
        else:
            plt.savefig(os.path.join(save_to, f'class_{class_id}.pdf'), dpi=150, bbox_inches='tight', pad_inches=0.05)




