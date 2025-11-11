import anndata
import numpy as np
import torch
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import HistGradientBoostingRegressor

adata_path = "../dataset82"

adata = anndata.read_h5ad(adata_path)
train_indices = np.where(adata.obs["split_key"] == "train")[0]
val_indices = np.where(adata.obs["split_key"] == "validate")[0]

X_train = adata.X[train_indices]
X_val = adata.X[val_indices]
y_train = adata.obsm["target"][train_indices].flatten()
y_val = adata.obsm["target"][val_indices].flatten()

imputer = SimpleImputer(strategy='mean')
X_train_clean = imputer.fit_transform(X_train)
X_val_clean = imputer.transform(X_val)
y_train_clean = np.nan_to_num(y_train, nan=np.nanmean(y_train))
y_val_clean = np.nan_to_num(y_val, nan=np.nanmean(y_train))

lr_model = LinearRegression(fit_intercept=False, positive=True)
lr_model.fit(X_train_clean, y_train_clean)
lr_predictions = lr_model.predict(X_val_clean)

hgb_model = HistGradientBoostingRegressor()
hgb_model.fit(X_train_clean, y_train_clean)
hgb_predictions = hgb_model.predict(X_val_clean)

pd.DataFrame(
    {
        "predictions": lr_predictions,
        "targets": y_val_clean
    }
).to_csv("predictions.lr.csv", index=False)
pd.DataFrame(
    {
        "predictions": hgb_predictions,
        "targets": y_val_clean
    }
).to_csv("predictions.hgb.csv", index=False)