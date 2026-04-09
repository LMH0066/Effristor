import anndata
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from tabpfn import TabPFNRegressor

import ivf

adata_path = "../dataset/dataset_0"
test_data = pd.read_excel("../data.2025.xlsx")
test_dataset = ivf.setup_data(
    test_data.iloc[:, 2:].values,
    (~test_data.iloc[:, 1].values.astype(bool)).astype(int),
)

adata = anndata.read_h5ad(adata_path)
train_indices = np.where(adata.obs["split_key"] == "train")[0]
val_indices = np.where(adata.obs["split_key"] == "validate")[0]

X_train = adata.X[train_indices]
X_val = adata.X[val_indices]
X_test = test_dataset.X
y_train = adata.obsm["target"][train_indices].flatten()
y_val = adata.obsm["target"][val_indices].flatten()
y_test = test_dataset.obsm["target"].flatten()

imputer = SimpleImputer(strategy="mean")
X_train_clean = imputer.fit_transform(X_train)
X_val_clean = imputer.transform(X_val)
X_test_clean = imputer.transform(X_test)
y_train_clean = np.nan_to_num(y_train, nan=np.nanmean(y_train))
y_val_clean = np.nan_to_num(y_val, nan=np.nanmean(y_train))
y_test_clean = np.nan_to_num(y_test, nan=np.nanmean(y_train))


lr_model = LinearRegression(fit_intercept=False, positive=True)
lr_model.fit(X_train_clean, y_train_clean)
lr_predictions = lr_model.predict(X_val_clean)
lr_test_predictions = lr_model.predict(X_test_clean)

hgb_model = HistGradientBoostingRegressor()
hgb_model.fit(X_train_clean, y_train_clean)
hgb_predictions = hgb_model.predict(X_val_clean)
hgb_test_predictions = hgb_model.predict(X_test_clean)

rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train_clean, y_train_clean)
rf_predictions = rf_model.predict(X_val_clean)
rf_test_predictions = rf_model.predict(X_test_clean)

tab_model = TabPFNRegressor()
tab_model.fit(X_train, y_train)
tab_predictions = tab_model.predict(X_val)
tab_test_predictions = tab_model.predict(X_test)

pd.DataFrame({"predictions": lr_predictions, "targets": y_val_clean}).to_csv(
    "benchmark/predictions.{}.lr.csv".format(adata_path.split("_")[-1]), index=False
)
pd.DataFrame({"predictions": lr_test_predictions, "targets": y_test_clean}).to_csv(
    "benchmark/predictions.{}.lr.2025.csv".format(adata_path.split("_")[-1]), index=False
)
pd.DataFrame({"predictions": hgb_predictions, "targets": y_val_clean}).to_csv(
    "benchmark/predictions.{}.hgb.csv".format(adata_path.split("_")[-1]), index=False
)
pd.DataFrame({"predictions": hgb_test_predictions, "targets": y_test_clean}).to_csv(
    "benchmark/predictions.{}.hgb.2025.csv".format(adata_path.split("_")[-1]), index=False
)
pd.DataFrame({"predictions": rf_predictions, "targets": y_val_clean}).to_csv(
    "benchmark/predictions.{}.rf.csv".format(adata_path.split("_")[-1]), index=False
)
pd.DataFrame({"predictions": rf_test_predictions, "targets": y_test_clean}).to_csv(
    "benchmark/predictions.{}.rf.2025.csv".format(adata_path.split("_")[-1]), index=False
)
pd.DataFrame({"predictions": tab_predictions, "targets": y_val}).to_csv(
    "benchmark/predictions.{}.tab.csv".format(adata_path.split("_")[-1]), index=False
)
pd.DataFrame({"predictions": tab_test_predictions, "targets": y_test}).to_csv(
    "benchmark/predictions.{}.tab.2025.csv".format(adata_path.split("_")[-1]), index=False
)