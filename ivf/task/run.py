import anndata
import click
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from lightning.pytorch.loggers import NeptuneLogger
from scipy import stats
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc
from torchmetrics import Metric
from torchmetrics.utilities import dim_zero_cat
from tqdm import tqdm

import ivf
import ivf._utils


@click.group()
def cli():
    pass


@cli.command(short_help="Generate dataset for training.")
@click.option("--input_file", type=str)
@click.option("--split_key", type=str)
@click.option("--output_file", type=str, default="dataset")
@click.option("--random_seed", type=int, default=42)
def create_dataset(input_file, split_key, output_file, random_seed):
    np.random.seed(42)

    data = pd.read_excel(input_file)
    # dataset = ivf.setup_data(data.iloc[:, 2:].values, data.iloc[:, 1].values)
    dataset = ivf.setup_data(data.iloc[:, 2:].values, (~data.iloc[:, 1].values.astype(bool)).astype(int))

    index = dataset.obs.index.to_numpy().astype(int)
    train_index = np.random.choice(index, size=int(len(index) * 0.8), replace=False)
    validate_index = np.setdiff1d(index, train_index)
    split = pd.Series(["0"] * len(index))
    split[train_index], split[validate_index] = "train", "validate"
    dataset.obs[split_key] = split.values

    dataset.write(filename=output_file, compression="gzip")


class MyMetric(Metric):
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
        # parse inputs
        preds = dim_zero_cat(self.preds)
        preds = torch.sigmoid(preds)
        target = dim_zero_cat(self.target)
        print(preds.unique(return_counts=True))
        # compute auroc
        # return 1 - roc_auc_score(target[:, 0], preds[:, 0])
        # compute auprc
        precision, recall, _ = precision_recall_curve(target[:, 0], preds[:, 0], pos_label=1)
        auprc = auc(recall, precision)
        return 1 - auprc


@cli.command()
@click.option("--dataset_path", type=str, default="dataset")
@click.option("--split_key", type=str)
def train(dataset_path, split_key):
    dataset = anndata.read_h5ad(dataset_path)
    ivf.IVF.setup_anndata(dataset)
    model = ivf.IVF(
        dataset,
        module_params={
            "d_model": 512,
            "num_encoder_layers": 6,
            "nhead": 8,
            "dim_feedforward": 1024,
            "dropout": 0.1,
        },
        split_key=split_key,
    )

    model.train(
        max_epochs=100,
        save_ckpt_every_n_epoch=1,
        plan_kwargs={
            "metric": MyMetric(),
            "lr": 5e-5,
            "weight_decay": 0.5,
        },
        batch_size=32,
        save_top_k=10,
        # num_workers=18,
        # precision="16-mixed",  # 16bit is easy to cause nan
        device=[7],
        # logger=NeptuneLogger(log_model_checkpoints=False),
    )


if __name__ == "__main__":
    cli()
