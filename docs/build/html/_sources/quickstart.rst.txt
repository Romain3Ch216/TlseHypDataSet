Getting started
===============

Install **TlseHypDataSet** with pip:

.. code-block:: console

   $ pip install TlseHypDataSet
   
Download the hyperspectral images in an :code:`images` folder: 

.. code-block:: 

   /path/to/dataset/
   ├── images
       ├── TLS_1b_2021-06-15_10-41-20_reflectance_rect.bsq
       ├── TLS_1b_2021-06-15_10-41-20_reflectance_rect.hdr


Load training data from a standard train / test split in a Pytorch data loader and start training:

.. code-block:: python

    import torch
    from tlse_hyp_data_set import TlseHypDataSet
    from tlse_hyp_data_set.utils.dataset import DisjointDataSplit

    dataset = TlseHypDataSet('/path/to/dataset/')
    
    # Load the first standard ground truth split
    ground_truth_split = DisjointDataSplit(dataset.standard_splits[0])
    
    train_loader = torch.utils.data.DataLoader(
        ground_truth_split.sets_['train'], 
        shuffle=True, 
        batch_size=1024
        )

    for epoch in range(100):
        for samples, labels in train_loader:
            ...

