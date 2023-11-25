Toulouse Hyperspectral Data Set
===============================

**TlseHypDataSet** is a Python library to flexibly load `PyTorch <https://pytorch.org/>`_ datasets and run machine learning experiments on the `Toulouse Hyperspectral Data Set <https://www.toulouse-hyperspectral-data-set.com/>`_. 

Getting started
===============

The **TlseHypDataSet** is compatible with Python=3.8 and depends on `GDAL <https://pypi.org/project/GDAL/>`_ which is recommended to be installed with conda:

.. code-block:: console

   $ conda create --name tlse python=3.8
   $ conda activate tlse
   $ conda install -c conda-forge gdal
   $ pip install TlseHypDataSet
   
Download the hyperspectral images from the `data catalogue <https://camcatt.sedoo.fr/catalogue/>`_ in an `images` folder: 

.. code-block:: 

   /path/to/dataset/
   ├── images
       ├── TLS_1b_2021-06-15_10-41-20_reflectance_rect.bsq
       ├── TLS_1b_2021-06-15_10-41-20_reflectance_rect.hdr
       ├── ...


Further documentation is going to be available soon. Here is a first example for a quick start.

The :code:`TlseHypDataSet` class has a :code:`standard_splits` attribute that contains 8 standard splits of the ground truth in a :code:`train` set, a :code:`labeled_pool`, an :code:`unlabeled_pool`, a :code:`validation` set and a :code:`test` set, as explained in our `paper <https://arxiv.org/pdf/2311.08863.pdf>`_. The following example shows how to load the training set of the first standard train / test split in a Pytorch data loader with the :code:`DisjointDataSplit` class:

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

Citation
========

If you use the **TlseHypDataSet** library, please cite the following two articles:

.. code-block:: bibtex

    @article{ROUPIOZ2023109109,
    title = {Multi-source datasets acquired over Toulouse (France) in 2021 for urban microclimate studies during the CAMCATT/AI4GEO field campaign},
    journal = {Data in Brief},
    volume = {48},
    pages = {109109},
    year = {2023},
    issn = {2352-3409},
    doi = {https://doi.org/10.1016/j.dib.2023.109109},
    url = {https://www.sciencedirect.com/science/article/pii/S2352340923002287},
    author = {L. Roupioz and X. Briottet and K. Adeline and A. {Al Bitar} and D. Barbon-Dubosc and R. Barda-Chatain and P. Barillot and S. Bridier and E. Carroll and C. Cassante and A. Cerbelaud and P. Déliot and P. Doublet and P.E. Dupouy and S. Gadal and S. Guernouti and A. {De Guilhem De Lataillade} and A. Lemonsu and R. Llorens and R. Luhahe and A. Michel and A. Moussous and M. Musy and F. Nerry and L. Poutier and A. Rodler and N. Riviere and T. Riviere and J.L. Roujean and A. Roy and A. Schilling and D. Skokovic and J. Sobrino},
    keywords = {Land surface temperature, Spectral emissivity, Spectral reflectance, Air temperature, Airborne LiDAR, Atmospheric data, Urban area},
    }

    @misc{thoreau2023toulouse,
      title={Toulouse Hyperspectral Data Set: a benchmark data set to assess semi-supervised spectral representation learning and pixel-wise classification techniques}, 
      author={Romain Thoreau and Laurent Risser and Véronique Achard and Béatrice Berthelot and Xavier Briottet},
      year={2023},
      eprint={2311.08863},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
     }
