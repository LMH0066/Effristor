python run.py predict-and-evaluate --model_path ../.ivf/2025-04-20_21-15-12_val_metric/epoch=223-step=34720-val_metric=0.15028613805770874 --adata_path ../dataset82
python run.py predict-and-evaluate --model_path ../.ivf/2025-04-20_21-15-12_val_metric/epoch=223-step=34720-val_metric=0.15028613805770874 --adata_path ../dataset82 --xlsx_path ../data.20250829.xlsx --output_path predictions.20250829.csv
python run.py interpret --model_path ../.ivf/2025-04-20_21-15-12_val_metric/epoch=223-step=34720-val_metric=0.15028613805770874 --adata_path ../dataset82
python run.py suggest --model_path ../.ivf/2025-04-20_21-15-12_val_metric/epoch=223-step=34720-val_metric=0.15028613805770874 --xlsx_path ../data.xlsx
