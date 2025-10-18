# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a DVC-managed machine learning project with two distinct components:

1. **ML Pipeline (Iris Classification)**: A DVC pipeline that trains a RandomForest classifier on the Iris dataset
2. **Flask Application (House Price Prediction)**: A separate Flask web app for predicting house prices using pre-trained models

These are independent workflows - the pipeline trains iris models, while the Flask app serves a different house price model.

## Architecture

### DVC Pipeline Architecture
The ML pipeline follows a sequential 4-stage workflow orchestrated by DVC:
- **prepare** → **features** → **train** → **evaluate**

Each stage is defined in `dvc.yaml` with explicit dependencies (deps) and outputs (outs). All hyperparameters are externalized in `params.yaml`. The pipeline uses DVC's parameterization syntax (`${features.test_size}`) to inject params at runtime.

**Key Dependency Note**: There's a mismatch in `dvc.yaml` line 12 - the `features` stage expects `data/iris_dataset.csv` as a dependency but the `prepare` stage outputs `data/iris.csv`. The actual file in data/ is `iris_dataset.csv`. To run the pipeline successfully, either:
- Change line 12 in `dvc.yaml` from `data/iris_dataset.csv` to `data/iris.csv`
- Or update `src/prepare.py:14` to output `iris_dataset.csv` instead of `iris.csv`

### Flask Application Architecture
The Flask app (`housepk_app.py`) loads 4 artifacts from `models/`:
- `house_price_model.pkl`: trained model
- `model_features.pkl`: ordered feature list
- `label_encoders.pkl`: LabelEncoders for categorical features
- `feature_field_map.pkl`: maps feature names to form field names

The app dynamically generates forms based on these artifacts, handling both categorical (dropdown) and numeric (text input) features.

**Note**: The Flask app requires HTML templates (`templates/index.html`, `templates/result.html`) and the `models/` directory with all 4 artifacts to run. These are not included in the repository.

### Data Flow
```
prepare.py → data/iris.csv
features.py → data/{X_train,X_test,y_train,y_test}.npy
train.py → model.pkl
evaluate.py → metrics/eval.json
```

## Commands

### Environment Setup
**Windows PowerShell**:
```powershell
Set-ExecutionPolicy -ExecutionPolicy Bypass -Scope Process
.venv\Scripts\activate
```

**Command Prompt**:
```cmd
.venv\Scripts\activate.bat
```

**Install dependencies**:
```bash
pip install -r requriements.txt
```
Note: File is named `requriements.txt` (typo in original).

### DVC Pipeline

**Run entire pipeline**:
```bash
dvc repro
```

**Run specific stage**:
```bash
dvc repro <stage_name>  # prepare, features, train, or evaluate
```

**Run single stage without DVC**:
```bash
python src/prepare.py --out_dir data
python src/features.py --in_csv data/iris.csv --out_dir data --test_size 0.2 --random_state 42
python src/train.py --data_dir data --model_out model.pkl
python src/evaluate.py --data_dir data --model model.pkl --out metrics/eval.json
```

**View pipeline DAG**:
```bash
dvc dag
```

**View metrics**:
```bash
dvc metrics show
```

### Flask Application

**Run web server**:
```bash
python housepk_app.py
```
Starts server on http://localhost:5000

**Test API endpoint**:
```bash
curl -X POST http://localhost:5000/api/predict -H "Content-Type: application/json" -d '{"feature1": value1, ...}'
```

## Parameter Management

All ML hyperparameters live in `params.yaml`. To modify training:
1. Edit `params.yaml` (e.g., change `train.n_estimators: 100` to `150`)
2. Run `dvc repro` - DVC will detect param changes and re-run affected stages

**Important**: The `train.py` script reads `params.yaml` directly (line 14-15), while DVC injects other parameters via command-line args (e.g., `${features.test_size}` in the `features` stage). The `evaluate.py` script hardcodes `average="macro"` (line 22) instead of using the `evaluate.average` parameter from `params.yaml`.

## File Locations

- **Source code**: `src/` (prepare.py, features.py, train.py, evaluate.py)
- **Pipeline config**: `dvc.yaml`, `params.yaml`
- **Data artifacts**: `data/` (ignored by git, tracked by DVC)
- **Models**: `model.pkl` for iris, `models/` for house price app
- **Metrics**: `metrics/eval.json`
- **Flask app**: `housepk_app.py` (requires `models/` directory with artifacts)

## Known Issues

1. **Pipeline dependency mismatch**: `dvc.yaml` line 12 expects `data/iris_dataset.csv` but `prepare` stage outputs `data/iris.csv`. The actual file in data/ is `iris_dataset.csv`. Fix by updating line 12 to match `prepare` output, or change prepare output to match the dependency.
2. **Requirements typo**: Dependencies file is named `requriements.txt` instead of `requirements.txt`.
3. **Unused parameter**: `params.yaml` defines `evaluate.average: macro` but `evaluate.py:22` hardcodes this value instead of reading from params.
4. **Missing Flask dependencies**: `requriements.txt` doesn't include `flask`, which is required for `housepk_app.py`.
