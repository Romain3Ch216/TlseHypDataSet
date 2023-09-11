# Toulouse Hyperspectral Data Set


```
import torch
from tlse_hyp_data_set import TlseHypDataSet
from tlse_hyp_data_set.utils.dataset import spatial_disjoint_split
from tlse_hyp_data_set.utils.preprocess import GaussianFilter


dataset = TlseHypDataSet(
    '/path/to/dataset/root', 
    patch_size=32, 
    padding=4, 
    flip_augmentation=True, 
    preprocess=GaussianFilter(sigma=1.5)
    )
    
labeled_set, unlabeled_set, test_set = spatial_disjoint_split(dataset, p_labeled=0.05, p_test=0.5)

labeled_data_loader = torch.utils.data.DataLoader(labeled_set)

for sample, labels in labeled_set:
    ...
```


