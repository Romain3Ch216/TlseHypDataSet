Dataset
=======

.. autoclass:: TlseHypDataSet.tlse_hyp_data_set.TlseHypDataSet
    :members:
    
    .. code-block:: 
        
        >>> from TlseHypDataSet.tlse_hyp_data_set import TlseHypDataSet 
        >>> dataset = TlseHypDataSet('/path/to/dataset', pred_mode='pixel', patch_size=5)
        >>> sample, label = next(iter(dataset))
        >>> sample.shape
        torch.Size([5, 5, 310]) # 310 is the number of spectral channels
        >>> label.shape
        torch.Size([1])
        >>> dataset = TlseHypDataSet('/path/to/dataset', pred_mode='patch', patch_size=5)
        >>> sample, labels = next(iter(dataset))
        >>> sample.shape
        torch.Size([5, 5, 310]) # 310 is the number of spectral channels
        >>> labels.shape
        torch.Size([5, 5])
        
    

