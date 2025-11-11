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
@click.option("--output_path", type=str, default="interpret_result.csv")
def interpret(
    model_path: str,
    adata_path: str,
    output_path: str = "interpret_result.csv"
):
    adata = anndata.read_h5ad(adata_path)

    # Load the trained model
    model = IVF.load(model_path, adata)

    # Make interpretations
    interpretations = []
    for i in tqdm(adata.obs_names, desc="Generating interpretations", unit="sample"):
        interpretations.append(model.interpret(adata[i].X))
    interpretations = torch.stack(interpretations).squeeze()

    pd.DataFrame(interpretations.numpy()).to_csv(output_path, index=False)


@click.command()
@click.option("--model_path", type=str)
@click.option("--xlsx_path", type=str)
@click.option("--output_path", type=str, default="suggest_4.csv")
def suggest_4(model_path: str, xlsx_path: str, output_path: str = "suggest_4.csv"):
    # Load the trained model
    adata = anndata.read_h5ad(os.path.join(model_path, "dataset82"))
    model = IVF.load(model_path, adata)

    # Loading data to be analyzed
    raw_data = pd.read_excel(xlsx_path)
    df = raw_data[raw_data["Normal fertilization"] == 0].drop(columns=["ID", "Normal fertilization"])  # Chose abnormal case
    learning_rates = {
        19: 2e4,   # E2 at hCG day
        17: 5e3,   # Gonadotropin (Gn) dosage
        27: 5,     # Abstinence days
        38: 10,    # Concentration after semen optimization treatment
    }
    clamp_range = {
        19: (19, 18176),
        17: (75, 11650),
        27: (0, 18),
        38: (1, 5),
    }
    need_round = {
        19: False,
        17: True,
        27: True,
        38: True,
    }
    
    result = {}
    batch_size = 64
    for i in tqdm(range(0, df.values.shape[0], batch_size)):
        batch_data = df.values[i:i+batch_size]
        batch_results = model.suggest_batch(
            batch_data, learning_rates, clamp_range, need_round, n_steps=10000
        )
        for j, res in enumerate(batch_results):
            result[i + j] = res
    for i in result.keys():
        result[i]["E2 at hCG day"] = result[i]["changes"][19]
        result[i]["Gonadotropin (Gn) dosage"] = result[i]["changes"][17]
        result[i]["Abstinence days"] = result[i]["changes"][27]
        result[i]["Concentration after semen optimization treatment"] = result[i]["changes"][38]
        # result[i]["Gn stimulation days"] = result[i]["changes"][18]
        # result[i]["The number of follicles ≥14mm at hCG day"] = result[i]["changes"][23]
        del result[i]["changes"]
    results = pd.DataFrame(result)

    failed_ratio = results.T["score"].apply(lambda x: x[0] == x[1]).mean()
    mean_diff = results.T["score"].apply(lambda x: abs(x[0] - x[1])).mean()
    print(f"Failed ratio: {failed_ratio}; Mean difference: {mean_diff}")

    results.columns = raw_data[raw_data["Normal fertilization"] == 0]["ID"]
    results.to_csv(output_path)


@click.command()
@click.option("--model_path", type=str)
@click.option("--xlsx_path", type=str)
@click.option("--feature_count", type=int)
@click.option("--output_path", type=str, default="suggest.csv")
def suggest(model_path: str, xlsx_path: str, feature_count: int, output_path: str = "suggest.csv"):
    # Load the trained model
    adata = anndata.read_h5ad(os.path.join(model_path, "dataset82"))
    model = IVF.load(model_path, adata)

    # Loading data to be analyzed
    raw_data = pd.read_excel(xlsx_path)
    df = raw_data[raw_data["Normal fertilization"] == 0].drop(columns=["ID", "Normal fertilization"])  # Chose abnormal case
    column_names = {
        19: "E2 at hCG day",
        17: "Gonadotropin (Gn) dosage",
        27: "Abstinence days",
        38: "Concentration after semen optimization treatment",
        7:  "BMI",
        18: "Gn stimulation days",
        23: "The number of follicles ≥14mm at hCG day",
        22: "Endometrial thickness at hCG day",
        24: "Number of punctured follicles",
        11: "Estradiol (E2)",
        42: "Number of oocyte for IVF",
        43: "Number of oocyte for ICSI",
        26: "Type of semen",
        37: "Volume after semen optimization treatment",
    }
    learning_rates = {
        19: 2e4,   # E2 at hCG day
        17: 5e3,   # Gonadotropin (Gn) dosage
        27: 5,     # Abstinence days
        38: 10,    # Concentration after semen optimization treatment
        7:  10,    # BMI
        18: 15,    # Gn stimulation days
        23: 25,    # The number of follicles ≥14mm at hCG day
        22: 5,     # Endometrial thickness at hCG day
        24: 1,     # Number of punctured follicles
        11: 20,    # Estradiol (E2)
        42: 5,     # Number of oocyte for IVF
        43: 10,    # Number of oocyte for ICSI
        26: 1e-1,  # Type of semen
        37: 15,     # Volume after semen optimization treatment
    }
    clamp_range = {
        19: (19, 18176),
        17: (75, 11650),
        27: (0, 18),
        38: (1, 5),
        7:  (15, 40),
        18: (1, 30),
        23: (0, 51),
        22: (2, 86),
        24: (1, 54),
        11: (3.36, 4768),
        42: (0, 54),
        43: (0, 39),
        26: (1, 2),
        37: (0.2, 3),
    }
    need_round = {
        19: False,
        17: True,
        27: True,
        38: True,
        7:  False,
        18: True,
        23: True,
        22: False,
        24: True,
        11: False,
        42: True,
        43: True,
        26: True,
        37: False,
    }
    
    result = {}
    batch_size = 64
    for i in tqdm(range(0, df.values.shape[0], batch_size)):
        batch_data = df.values[i:i+batch_size]
        batch_results = model.suggest_batch(
            batch_data,
            dict(list(learning_rates.items())[:feature_count]),
            dict(list(clamp_range.items())[:feature_count]),
            dict(list(need_round.items())[:feature_count]),
            n_steps=10000
        )
        for j, res in enumerate(batch_results):
            result[i + j] = res
    for i in result.keys():
        for k, v in dict(list(column_names.items())[:feature_count]).items():
            if k in result[i]["changes"].keys():
                result[i][v] = result[i]["changes"][k]
        del result[i]["changes"]
    results = pd.DataFrame(result)

    failed_ratio = results.T["score"].apply(lambda x: x[0] == x[1]).mean()
    mean_diff = results.T["score"].apply(lambda x: abs(x[0] - x[1])).mean()
    print(f"Failed ratio: {failed_ratio}; Mean difference: {mean_diff}")

    results.columns = raw_data[raw_data["Normal fertilization"] == 0]["ID"]
    results.to_csv(output_path)


@click.group()
def cli():
    pass


cli.add_command(predict_and_evaluate)
cli.add_command(interpret)
cli.add_command(suggest_4)
cli.add_command(suggest)

if __name__ == "__main__":
    cli()