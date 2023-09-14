Transforms
==========

.. autoclass:: TlseHypDataSet.utils.transforms.GaborFilters
.. autoclass:: TlseHypDataSet.utils.transforms.SpectralIndices
.. autoclass:: TlseHypDataSet.utils.transforms.GaussianFilter
.. autoclass:: TlseHypDataSet.utils.transforms.Concat
.. autoclass:: TlseHypDataSet.utils.transforms.Stats


The example below shows how data transformations were combined to qualitatively compare the Toulouse Hyperspectral Data Set to other data sets (see comparison at `www.toulouse-hyperspectral-data-set.com <https://www.toulouse-hyperspectral-data-set.com>`_).


.. code-block:: python

    import torch
    from torchvision import transforms
    from TlseHypDataSet.tlse_hyp_data_set import TlseHypDataSet
    from TlseHypDataSet.utils.transforms import GaussianFilter, SpectralIndices,\
                                                GaborFilters, Concat, Stats

    dataset = TlseHypDataSet('/path/to/dataset/', pred_mode='patch', patch_size=64)
    dataset.transform = transforms.Compose([
                    GaussianFilter(dataset.bbl, sigma=1.5),
                    Concat([
                        SpectralIndices(dataset.wv[dataset.bbl]),
                        GaborFilters()
                        ]),
                    Stats()
                ])
                
                
