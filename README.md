# ETAI-Project
This repository contains the group project made for the course of Exploratory Topics in Artificial Intelligence (ETAI) that is part of the MDSAA @NOVAIMS


## Creating the virtual environment

### MacOS/ Linux
python3.11 -m venv .venv

source .venv/bin/activate

### Windows
py -3.11 -m venv .venv

.\.venv\Scripts\Activate.ps1

## Installing the requirements
Python 3.11.

pip install -r requirements.txt

## Running the pipeline
Put the four CSVs in `data/raw/`, then from the repo root:

```bash
python main.py train --task classification   # CV scores, then saves models/classification.joblib
python main.py predict                        # writes results/submission_classification.csv (Kaggle format)
```

Add `--config configs/dev.yaml` to either command for a fast smoke run (writes to `models/dev`, `results/dev`).
Settings live in `config.yaml`; files in `configs/` only override the keys that differ.
