import os

import anndata
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from tqdm import tqdm

from ivf import IVF


def main(model_path: str, xlsx_path: str):
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
    results.to_csv("suggest.csv")


if __name__ == "__main__":
    import sys
    if len(sys.argv) != 3:
        print("Usage: python suggest.py <model_path> <xlsx_path>")
    else:
        main(sys.argv[1], sys.argv[2])
