# Develop Guidelines
## environment prepare
```bash
conda create -n Effristor python=3.10
conda activate Effristor
pip install poetry
poetry install
```
## pre commit
```bash
# Format python code
poe format
# Run test code
poe test
```
