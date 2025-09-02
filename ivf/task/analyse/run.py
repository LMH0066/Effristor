import os

import anndata
import click
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import torch
from tqdm import tqdm

import ivf
from ivf import IVF


@click.command()
@click.option("--model_path", type=str)
@click.option("--adata_path", type=str)
@click.option("--xlsx_path", type=str, default="")
@click.option("--output_path", type=str, default="predictions.csv")
def predict_and_evaluate(
    model_path: str,
    adata_path: str,
    xlsx_path: str = "",
    output_path: str = "predictions.csv",
):
    adata = anndata.read_h5ad(adata_path)
    model = IVF.load(model_path, adata)

    if xlsx_path == "":
        val_indices = np.where(adata.obs["split_key"] == "validate")[0]
        predictions = model.predict(adata, val_indices, batch_size=64)
    else:
        data = pd.read_excel(xlsx_path)
        dataset = ivf.setup_data(
            data.iloc[:, 2:].values,
            (~data.iloc[:, 1].values.astype(bool)).astype(int)
        )
        ivf.IVF.setup_anndata(dataset)
        predictions = model.predict(dataset, batch_size=64)

    predictions.X = torch.sigmoid(torch.tensor(predictions.X)).numpy()
    pd.DataFrame(
        {
            "predictions": predictions.X.flatten(),
            "targets": predictions.obsm["target"].flatten()
        }
    ).to_csv(output_path, index=False)


@click.command()
@click.option("--model_path", type=str)
@click.option("--adata_path", type=str)
@click.option("--split_key", type=str, default="split_key")
@click.option("--output_path", type=str, default="interpret_result.csv")
def interpret(
    model_path: str,
    adata_path: str,
    split_key: str = "split_key",
    output_path: str = "interpret_result.csv"
):
    adata = anndata.read_h5ad(adata_path)

    # Load the trained model
    model = IVF.load(model_path, adata)

    # Load the validation data
    val_indices = np.where(adata.obs[split_key] == "validate")[0]

    # Make interpretations
    interpretations = []
    for i in tqdm(val_indices, desc="Generating interpretations", unit="sample"):
        interpretations.append(model.interpret(adata[i].X))
    interpretations = torch.stack(interpretations).squeeze()

    pd.DataFrame(interpretations.numpy()).to_csv(output_path, index=False)


@click.command()
@click.option("--model_path", type=str)
@click.option("--xlsx_path", type=str)
@click.option("--output_path", type=str, default="suggest.csv")
def suggest(model_path: str, xlsx_path: str, output_path: str = "suggest.csv"):
    # Load the trained model
    adata = anndata.read_h5ad(os.path.join(model_path, "dataset82"))
    model = IVF.load(model_path, adata)

    # Loading data to be analyzed
    raw_data = pd.read_excel(xlsx_path)
    df = raw_data[raw_data["Normal fertilization"] == 0].drop(columns=["ID", "Normal fertilization"])  # Chose abnormal case
    learning_rates = {
        19: 1e4,   # E2 at hCG day
        17: 1e4,   # Gonadotropin (Gn) dosage
        27: 1e-1,  # Abstinence days
        38: 5e-2,  # Concentration after semen optimization treatment
        # 18: 1,     # Gn stimulation days
        # 23: 2,     # The number of follicles ≥14mm at hCG day
    }
    result = {}
    for i in tqdm(range(df.values.shape[0])):
        result[i] = model.suggest(df.values[i], learning_rates, n_steps=10000)
    for i in result.keys():
        result[i]["E2 at hCG day"] = result[i]["changes"][19]
        result[i]["Gonadotropin (Gn) dosage"] = result[i]["changes"][17]
        result[i]["Abstinence days"] = result[i]["changes"][27]
        result[i]["Concentration after semen optimization treatment"] = result[i]["changes"][38]
        # result[i]["Gn stimulation days"] = result[i]["changes"][18]
        # result[i]["The number of follicles ≥14mm at hCG day"] = result[i]["changes"][23]
        del result[i]["changes"]
    results = pd.DataFrame(result)
    results.columns = raw_data[raw_data["Normal fertilization"] == 0]["ID"]
    results.to_csv(output_path, index=False)


@click.group()
def cli():
    pass


cli.add_command(predict_and_evaluate)
cli.add_command(interpret)
cli.add_command(suggest)

if __name__ == "__main__":
    cli()