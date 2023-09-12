Utils
=====

The following example shows how to convert land cover labels in permeable / impermeable labels or to land cover classes of the higher level of the hierarchy.

.. code-block:: 

    >>> from TlseHypDataSet.utils.utils import labels_to_labels
    >>> from TlseHypDataSet.tlse_hyp_data_set import TlseHypDataSet
    >>> import numpy as np
    >>> dataset = TlseHypDataSet('/path/to/dataset', pred_mode='pixel', patch_size=1)
    >>> labels = np.array([[3, 6], [17, 20]])
    >>> labels_to_labels(labels, dataset.permeability)
    array([[0, 0],
           [1, 1]])
    >>> labels_to_labels(labels, dataset.bottom_to_top)
    array([[ 2,  4],
           [ 9, 10]])
