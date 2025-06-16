import anndata
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from tqdm import tqdm

from ivf import IVF


def main(model_path: str, adata_path: str, split_key: str = "split_key"):
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

    pd.DataFrame(interpretations.numpy()).to_csv('interpret_result.csv', index=False)

    sns.heatmap(interpretations, cmap='bwr', center=0)
    plt.xlabel('Score')
    plt.ylabel('Sample')
    plt.savefig('interpret_result.png', dpi=300, bbox_inches='tight')


if __name__ == "__main__":
    import sys
    if len(sys.argv) != 3:
        print("Usage: python predict_and_evaluate.py <model_path> <adata_path>")
    else:
        main(sys.argv[1], sys.argv[2])
