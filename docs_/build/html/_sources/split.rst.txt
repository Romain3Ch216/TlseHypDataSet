Spatially disjoint ground truth splits
======================================

The ground truth of the Toulouse Hyperspectral Data Set is a shapefile that contains polygons associated to a land cover class. Spatially close polygons are grouped together, resulting in a few hundreds of groups. The split of the ground truth consists in assigning each group to a set (among the labeled training set, the unlabeled training set, the validation set and the test set) such that the proportions of pixels in every sets respect some conditions. Standard splits of the ground truth are provided with the TlseHypDataSet class described in the :doc:`dataset` section.

.. autoclass:: TlseHypDataSet.utils.dataset.DisjointDataSplit
    :members:
