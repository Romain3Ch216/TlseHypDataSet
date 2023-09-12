Getting started
===============

Install **TlseHypDataSet** with pip:

.. code-block:: console

   $ pip install TlseHypDataSet
   
Download the hyperspectral images at `www.toulouse-hyperspectral-data-set.com <https://www.toulouse-hyperspectral-data-set.com>`_ in an :code:`images` folder: 

.. code-block:: 

   /path/to/dataset/
   ├── images
       ├── TLS_1b_2021-06-15_10-41-20_reflectance_rect.bsq
       ├── TLS_1b_2021-06-15_10-41-20_reflectance_rect.hdr
       ├── ...


The :code:`TlseHypDataSet` class has a :code:`standard_splits` attribute that contains 8 standard splits of the ground truth in a :code:`'train'` set, a :code:`'labeled_pool'`, an :code:`'unlabeled_pool'`, a :code:`'validation'` set and a :code:`'test'` set, as explained `here <#>`_. The following example shows how to load the training set of the first standard train / test split in a Pytorch data loader (see the :doc:`dataset` section for more details):

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

