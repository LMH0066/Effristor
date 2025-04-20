from typing import Optional, Sequence

import numpy as np
from anndata import AnnData
from scvi import settings
from scvi.data import AnnDataManager
from scvi.dataloaders import DataSplitter
from scvi.model._utils import parse_device_args


def setup_data(
    matrix: np.array,
    target: Sequence,
) -> AnnData:
    """Setup function.

    Parameters
    ----------
    matrix
        # TODO
    target
        # TODO

    Returns
    -------
    An :class:`~anndata.AnnData` object containing the data required for model training.
    """

    target = np.array(target)
    target = target[:, np.newaxis] if target.ndim == 1 else target
    datas = AnnData(X=matrix, obsm={"target": target})

    return datas


class AnnDataSplitter(DataSplitter):
    """Creates data loaders ``train_set``, ``validation_set``, ``test_set`` using given indices."""

    def __init__(
        self,
        adata_manager: AnnDataManager,
        train_indices,
        valid_indices,
        test_indices,
        accelerator: str = "auto",
        pin_memory: bool = False,
        **kwargs,
    ):
        super().__init__(adata_manager=adata_manager, pin_memory=pin_memory)
        self.data_loader_kwargs = kwargs
        self.train_idx = train_indices
        self.val_idx = valid_indices
        self.test_idx = test_indices
        self.accelerator = accelerator

    def setup(self, stage: Optional[str] = None):
        """Over-ride parent's setup to preserve split idx."""
        return
