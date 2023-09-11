Transforms
==========

.. autoclass:: TlseHypDataSet.utils.transforms.GaborFilters
.. autoclass:: TlseHypDataSet.utils.transforms.SpectralIndices
.. autoclass:: TlseHypDataSet.utils.transforms.GaussianFilter
.. autoclass:: TlseHypDataSet.utils.transforms.Concat
.. autoclass:: TlseHypDataSet.utils.transforms.Stats


The example below shows how data transformations were combined to produce Fig. 5 in `Toulouse Hyperspectral Data Set: analysis and study of self-supervision for spectral representation learning #>`_


.. code-block:: python

    import torch
    from tlse_hyp_data_set import TlseHypDataSet
    from tlse_hyp_data_set.utils.transforms import GaussianFilter, SpectralIndices, 
                                                   GaborFilters, Concat

    dataset = TlseHypDataSet('/path/to/dataset/')
    dataset.transform = transforms.Compose([
                    GaussianFilter(dataset.bbl, sigma=1.5),
                    Concat([
                        SpectralIndices(dataset.wv[dataset.bbl]),
                        GaborFilters()
                        ]),
                    Stats()
                ])
                
                
