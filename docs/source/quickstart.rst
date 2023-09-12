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


The :code:`TlseHypDataSet` class has a :code:`standard_splits` attribute that contains 8 standard splits of the ground truth in a :code:`'train'` set, a :code:`'labeled_pool'`, an :code:`'unlabeled_pool'`, a :code:`'validation'` set and a :code:`'test'` set, as explained `here <#>`_. The following example shows how to load the training set of the first standard train / test split in a Pytorch data loader with the :code:`DisjointDataSplit` class (see the :doc:`dataset` and the :doc:`split` sections for more details):

.. code-block:: python

    import torch
    from TlseHypDataSet.tlse_hyp_data_set import TlseHypDataSet
    from TlseHypDataSet.utils.dataset import DisjointDataSplit

    dataset = TlseHypDataSet('/path/to/dataset/', pred_mode='pixel', patch_size=1)
    
    # Load the first standard ground truth split
    ground_truth_split = DisjointDataSplit(dataset, split=dataset.standard_splits[0])
    
    train_loader = torch.utils.data.DataLoader(
        ground_truth_split.sets_['train'], 
        shuffle=True, 
        batch_size=1024
        )

    for epoch in range(100):
        for samples, labels in train_loader:
            ...


NB: at first use, the images and the ground truth will be processed and additional data will be saved in a :code:`rasters` folder.
