import os
from pathlib import Path

import anndata
import click
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from lightning.pytorch.loggers import NeptuneLogger
from scipy import stats
from sklearn.metrics import auc, precision_recall_curve, roc_auc_score
from sklearn.model_selection import train_test_split
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
    np.random.seed(random_seed)

    data = pd.read_excel(input_file)
    # dataset = ivf.setup_data(data.iloc[:, 2:].values, data.iloc[:, 1].values)
    dataset = ivf.setup_data(
        data.iloc[:, 2:].values, (~data.iloc[:, 1].values.astype(bool)).astype(int)
    )

    index = dataset.obs.index.to_numpy().astype(int)

    target = dataset.obsm["target"].flatten() if len(dataset.obsm["target"].shape) > 1 else dataset.obsm["target"]
    train_index, validate_index = train_test_split(
        index, 
        test_size=0.2, 
        random_state=random_seed,
        stratify=target
    )

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
        # print(preds.unique(return_counts=True))
        # compute auroc
        return 1 - roc_auc_score(target[:, 0], preds[:, 0])
        # compute auprc
        # precision, recall, _ = precision_recall_curve(
        #     target[:, 0], preds[:, 0], pos_label=1)
        # auprc = auc(recall, precision)
        # return 1 - auprc


@cli.command()
@click.option("--dataset_dir_path", type=str, required=True)
@click.option("--split_key", type=str, required=True)
def train_multi(dataset_dir_path, split_key):
    dataset_dir = Path(dataset_dir_path)
    dataset_files = sorted(
        [p for p in dataset_dir.rglob("*") if p.is_file()],
        key=lambda p: int(p.stem.split("_")[-1])
    )

    for dataset_file in dataset_files:
        print(f"Start training with dataset: {dataset_file}")

        dataset = anndata.read_h5ad(dataset_file)

        if split_key not in dataset.obs.columns:
            print(f"Skip {dataset_file}: split key '{split_key}' not found.")
            continue

        train_indices = np.where(dataset.obs[split_key] == "train")[0]
        if len(train_indices) == 0:
            print(f"Skip {dataset_file}: no train samples found.")
            continue

        ivf.IVF.setup_anndata(dataset)
        model = ivf.IVF(
            dataset,
            module_params={
                "d_model": 128,
                "num_encoder_layers": 2,
                "nhead": 32,
                "dim_feedforward": 128,
                "dropout": 0.01,
            },
            split_key=split_key,
        )

        model.train(
            max_epochs=150,
            save_ckpt_every_n_epoch=1,
            plan_kwargs={
                "metric": MyMetric(),
                "lr": 5e-5,
                "weight_decay": 0.1,
                "n_epochs_warmup": 0,
                "one_cycle_scheduler": True,
                "one_cycle_total_steps": 26000,
                "one_cycle_pct_start": 0.3,
                "gclip": 0,
            },
            batch_size=32,
            save_top_k=1,
            # num_workers=18,
            # precision="16-mixed",
            device=[5],
            # logger=NeptuneLogger(log_model_checkpoints=False),
        )

        print(f"Finished training: {dataset_file}")


if __name__ == "__main__":
    cli()
