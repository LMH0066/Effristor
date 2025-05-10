import anndata
import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from ivf import IVF

def main(model_path: str, adata_path: str, split_key: str = "validate"):
    # Load the trained model
    model = IVF.load(model_path)

    # Load the validation data
    adata = anndata.read_h5ad(adata_path)
    validation_indices = np.where(adata.obs[split_key] == "validate")[0]
    validation_data = adata[validation_indices]

    # Prepare the data for prediction
    inputs = model.get_dataset(validation_data)

    # Make predictions
    predictions = model.predict(inputs)

    # Calculate ROC AUC score
    targets = validation_data.obsm["target"]
    roc_auc = roc_auc_score(targets, predictions)

    print(f"ROC AUC Score: {roc_auc}")

if __name__ == "__main__":
    import sys
    if len(sys.argv) != 3:
        print("Usage: python predict_and_evaluate.py <model_path> <adata_path>")
    else:
        main(sys.argv[1], sys.argv[2])
