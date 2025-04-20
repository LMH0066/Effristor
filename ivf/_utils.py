from enum import Enum

import torch
from scipy import stats
from scvi import scvi_logger, settings
from torchmetrics import Metric
from torchmetrics.utilities import dim_zero_cat


"""
metric
"""


class LOSS_KEYS(str, Enum):
    """Module loss keys."""

    FocalLoss = "Focal_loss"


class DefaultMetric(Metric):
    higher_is_better = False

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.add_state("preds", default=[], dist_reduce_fx="cat")
        self.add_state("target", default=[], dist_reduce_fx="cat")

    def update(self, outputs: list) -> None:
        _predicts, _targets = None, None
        for output in outputs:
            _predict = output["predicts"]
            _predicts = (
                _predict
                if _predicts is None
                else torch.cat((_predicts, _predict), dim=0)
            )

            _target = output["targets"]
            _target = _target[0] if _target.dim() == 3 else _target
            _targets = (
                _target if _targets is None else torch.cat((_targets, _target), dim=0)
            )

        self.preds.append(_predicts)
        self.target.append(_targets)

    def compute(self):
        preds = dim_zero_cat(self.preds)
        target = dim_zero_cat(self.target)
        return 1 - stats.spearmanr(preds.flatten(), target.flatten())[0]


"""
log
"""

_logger = scvi_logger
settings.logging_dir = "./.ivf/"
