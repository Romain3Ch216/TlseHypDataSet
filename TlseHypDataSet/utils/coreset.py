import pdb

import torch
import numpy as np
from typing import List
from sklearn.decomposition import PCA
from TlseHypDataSet.dimension_reduction.autoencoders import pretrained_encoder, Encoding
import pickle as pkl
import os


__all__ = [
    'CoreSet',
    'core_set_selection',
    'random_selection'
]


def euclidean_dist(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """
    Compute the euclidean distance between two tensors.

    :param x: a (NxD) tensor
    :param y: a (MxD) tensor
    :return: a (NxM) distance matrix
    """
    m, n = x.size(0), y.size(0)
    xx = torch.pow(x, 2).sum(1, keepdim=True).expand(m, n)
    yy = torch.pow(y, 2).sum(1, keepdim=True).expand(n, m).t()
    dist = xx + yy
    dist.addmm_(x, y.t(), beta=1, alpha=-2)
    dist = dist.clamp(min=1e-12).sqrt()
    return dist


def dataset_to_tensors(dataset):
    sample, gt = dataset.__getitem__(0)
    assert sample.shape[0] == 1, "Samples must be in 1D."
    samples = torch.zeros((len(dataset), sample.shape[-1]))
    labels = torch.zeros(len(dataset))
    loader = torch.utils.data.DataLoader(dataset, shuffle=False, batch_size=10000)
    i = 0
    for sample, gt in loader:
        batch_size = sample.shape[0]
        sample = sample.view(batch_size, -1)
        gt = gt.view(batch_size)
        samples[i: i+batch_size, :] = sample
        labels[i: i+batch_size] = gt
        i += batch_size
    return samples, labels


def dataset_to_labels(dataset):
    sample, gt = dataset.__getitem__(0)
    assert sample.shape[0] == 1, "Samples must be in 1D."
    labels = torch.zeros(len(dataset))
    loader = torch.utils.data.DataLoader(dataset, shuffle=False, batch_size=10000)
    i = 0
    for _, gt in loader:
        batch_size = gt.shape[0]
        labels[i: i+batch_size] = gt.view(batch_size)
        i += batch_size
    return labels


def core_set_selection(dataset, budget=10, metric=None, dim_reduction='autoencoder', n_components=8):
    """
    Performs a class-wise core-set selection with a k-center greedy algorithm as implemented in https://github.com/PatrickZH/DeepCore

    :param dataset: a torch.utils.data.Dataset object
    :param budget: number of samples to select by class
    :param metric: distance metric to use (a callable)
    :param dim_reduction: method to represent the data in a lower dimensional space ('pca' or 'autoencoder')
    :param n_components: dimension of the representation space
    :return: A core-set selection of the data
    """
    core_set = CoreSet(budget, metric, dim_reduction, n_components)
    file = 'core_set_selection.pkl'
    if file in os.listdir(os.path.join(dataset.root_path, 'outputs')):
        with open(os.path.join(dataset.root_path, 'outputs', file), 'rb') as f:
            selection = pkl.load(f)
    else:
        selection = core_set(dataset)
        with open(os.path.join(dataset.root_path, 'outputs', file), 'wb') as f:
            pkl.dump(selection, f)
    return selection


def random_selection(dataset, budget=10):
    """
    Performs a class-wise random selection

    :param dataset: a torch.utils.data.Dataset object
    :param budget: number of samples to select by class
    :return: A random selection of the data
    """

    random_selection_ = RandomSelection(budget)
    file = 'random_selection.pkl'
    if file in os.listdir(os.path.join(dataset.root_path, 'outputs')):
        with open(os.path.join(dataset.root_path, 'outputs', file), 'rb') as f:
            selection = pkl.load(f)
    else:
        selection = random_selection_(dataset)
        with open(os.path.join(dataset.root_path, 'outputs', file), 'wb') as f:
            pkl.dump(selection, f)
    return selection


class CoreSet:
    def __init__(self, budget, metric=None, dim_reduction='pca', n_components=8):
        self.budget = budget
        self.metric = euclidean_dist if metric is None else metric
        self.dim_reduction = dim_reduction
        self.n_components = n_components

    def transform(self, dataset):
        if self.dim_reduction == 'pca':
            data, labels = dataset_to_tensors(dataset)
            proj = PCA(n_components=self.n_components).fit_transform(data)
        elif self.dim_reduction == 'autoencoder':
            labels = dataset_to_labels(dataset)
            loader = torch.utils.data.DataLoader(dataset, batch_size=1024, shuffle=False, pin_memory=True)
            encoder = Encoding(pretrained_encoder('cpu'))
            proj = encoder.transform(loader)
            proj = proj.numpy()
            labels = labels.numpy()
        else:
            raise NotImplementedError("Only pca reduction available.")
        return proj, labels

    def __call__(self, dataset):
        selection = {}
        print("Project data in lower space...")
        proj, labels = self.transform(dataset)
        indices = np.arange(proj.shape[0])
        print("Compute core set selection...")
        for class_id in np.unique(labels):
            proj_samples = proj[class_id == labels]
            indices_samples = indices[class_id == labels]
            core_set_selection = k_center_greedy(proj_samples, self.budget, self.metric)
            selection[class_id] = indices_samples[core_set_selection]
        return selection


def k_center_greedy(data: np.ndarray,
                    budget: int,
                    metric: callable,
                    already_selected: List = [],
                    print_freq: int = 20,
                    device='cpu') -> np.ndarray:
    """
    Select a subset of the data according to the k-center greedy algorithm.
    Modified from https://github.com/PatrickZH/DeepCore

    :param data: a (NxD) array where N is the number of samples and D is the data dimension
    :param budget: number of samples to select
    :param metric: distance metric
    :param already_selected: indices of the already selected samples
    :param print_freq: verbose argument
    :param device: cpu or gpu
    :return: indices of selected samples
    """

    sample_num = data.shape[0]
    assert sample_num >= 1

    data = torch.from_numpy(data.astype(np.float32))

    if budget < 0:
        raise ValueError("Illegal budget size.")
    elif budget > sample_num:
        budget = sample_num

    index = np.arange(sample_num)

    assert callable(metric)

    already_selected = np.array(already_selected)

    if len(already_selected) == 0:
        select_result = np.zeros(sample_num, dtype=bool)
        # Randomly select one initial point.
        already_selected = [np.random.randint(0, sample_num)]
        budget -= 1
        select_result[already_selected] = True
    else:
        select_result = np.in1d(index, already_selected)

    num_of_already_selected = np.sum(select_result)

    # Initialize a (num_of_already_selected+budget-1)*sample_num data storing distances of pool points from
    # each clustering center.None
    dis_data = -1 * torch.ones([num_of_already_selected + budget - 1, sample_num], requires_grad=False).to(device)
    dis_data[:num_of_already_selected, ~select_result] = metric(data[select_result], data[~select_result])
    mins = torch.min(dis_data[:num_of_already_selected, :], dim=0).values

    for i in range(budget):
        if i % print_freq == 0:
            print("| Selecting [%3d/%3d]" % (i + 1, budget))
        p = torch.argmax(mins).item()
        select_result[p] = True

        if i == budget - 1:
            break
        mins[p] = -1
        dis_data[num_of_already_selected + i, ~select_result] = metric(data[[p]], data[~select_result])
        mins = torch.min(mins, dis_data[num_of_already_selected + i])
    return index[select_result]


class RandomSelection:
    def __init__(self, budget):
        self.budget = budget

    def transform(self, dataset):
        labels = dataset_to_labels(dataset)
        return labels

    def __call__(self, dataset):
        selection = {}
        labels = self.transform(dataset)
        indices = np.arange(labels.shape[0])
        print("Random sampling...")
        for class_id in np.unique(labels):
            indices_samples = indices[class_id == labels]
            random_selection = np.random.choice(np.arange(len(indices_samples)), size=self.budget, replace=False)
            selection[class_id] = indices_samples[random_selection]
        return selection
