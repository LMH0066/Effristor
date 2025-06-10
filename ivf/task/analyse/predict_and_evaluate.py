import anndata
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score, roc_curve
from ivf import IVF


def main(model_path: str, adata_path: str, split_key: str = "split_key"):
    adata = anndata.read_h5ad(adata_path)

    # Load the trained model
    model = IVF.load(model_path, adata)

    # Load the validation data
    val_indices = np.where(adata.obs[split_key] == "validate")[0]

    # Make predictions
    predictions = model.predict(adata, val_indices, batch_size=64)

    predictions.X = torch.sigmoid(torch.tensor(predictions.X)).numpy()
    pd.DataFrame(
        {
            "predictions": predictions.X.flatten(),
            "targets": predictions.obsm["target"].flatten()
        }
    ).to_csv('predictions.csv', index=False)

    # Calculate ROC AUC score
    roc_auc = roc_auc_score(predictions.obsm["target"], predictions.X)

    print(f"ROC AUC Score: {roc_auc}")

    # Plot ROC curve
    fpr, tpr, _ = roc_curve(predictions.obsm["target"], predictions.X)

    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2,
             label=f'ROC curve (area = {roc_auc:.3f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc="lower right")

    plt.savefig('roc_curve.png')


if __name__ == "__main__":
    import sys
    if len(sys.argv) != 3:
        print("Usage: python predict_and_evaluate.py <model_path> <adata_path>")
    else:
        main(sys.argv[1], sys.argv[2])
