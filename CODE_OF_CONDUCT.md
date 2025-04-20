# Develop Guidelines
## environment prepare
```bash
conda create -n ivf python=3.10
conda activate ivf
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
