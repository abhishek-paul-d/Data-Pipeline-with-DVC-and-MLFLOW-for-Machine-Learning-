### Data Pipeline with DVC and MLFLOW for Machine Learning

An end-to-end machine learning pipeline demonstrating data versioning with DVC and experiment tracking with MLflow. The pipeline trains a Random Forest Classifier on the Pima Indians Diabetes Dataset.

## Key Features

- **Data Version Control (DVC)**
  - Tracks and versions datasets, models, and pipeline stages
  - Ensures reproducibility across environments
  - Supports remote storage for large datasets and models

- **Experiment Tracking (MLflow)**
  - Logs hyperparameters, metrics, and artifacts
  - Enables comparison of different model runs
  - Tracks model performance over time

## Pipeline Stages

<img width="1361" height="769" alt="Screenshot 2025-07-23 100124" src="https://github.com/user-attachments/assets/2f8c9158-f1a9-435e-aea3-d51dca0e575d" />

### 1. Preprocessing
`src/preprocess.py`:
- Reads raw data from `data/raw/data.csv`
- Performs basic preprocessing (column renaming, etc.)
- Outputs processed data to `data/processed/data.csv`

### 2. Training
`src/train.py`:
- Trains a Random Forest Classifier
- Saves model to `models/random_forest.pkl`
- Logs hyperparameters and model to MLflow

### 3. Evaluation
`src/evaluate.py`:
- Loads trained model and evaluates performance
- Logs evaluation metrics to MLflow

## Project Goals

- **Reproducibility**: Consistent results across runs and environments
- **Experimentation**: Easy comparison of different model configurations
- **Collaboration**: Team-friendly workflow for data science projects

## About DVC and MLflow

### Data Version Control (DVC)
DVC is an open-source version control system for machine learning projects that:
- Extends Git to handle large files (datasets, models)
- Provides data and model versioning capabilities
- Creates reproducible pipelines through dependency tracking
- Supports remote storage (S3, GCS, SSH, etc.) for large files
- Works alongside Git to version both code and data

Key DVC Features:
- Data and model versioning
- Pipeline reproducibility
- Experiment management
- Data registry capabilities
- Cross-team collaboration tools

### MLflow
MLflow is an open-source platform for managing the machine learning lifecycle that provides:
- Experiment tracking for logging parameters, metrics, and artifacts
- Model packaging and deployment capabilities
- Model registry for versioning and staging models
- Project packaging for reproducible runs

Key MLflow Components:
- Tracking Server: Records and queries experiments
- Projects: Packaging format for reproducible runs
- Models: Standard format for packaging ML models
- Model Registry: Centralized model store

## Technology Stack

- Python (data processing, modeling)
- DVC (data version control)
- MLflow (experiment tracking)
- Scikit-learn (machine learning)

## Setup and Usage

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Run the pipeline:
```bash
dvc repro
```

## DVC Pipeline Commands

### Add Preprocessing Stage
```bash
dvc stage add -n preprocess \
    -p preprocess.input,preprocess.output \
    -d src/preprocess.py -d data/raw/data.csv \
    -o data/processed/data.csv \
    python src/preprocess.py
```

### Add Training Stage
```bash
dvc stage add -n train \
    -p train.data,train.model,train.random_state,train.n_estimators,train.max_depth \
    -d src/train.py -d data/raw/data.csv \
    -o models/model.pkl \
    python src/train.py
```

### Add Evaluation Stage
```bash
dvc stage add -n evaluate \
    -d src/evaluate.py -d models/model.pkl -d data/raw/data.csv \
    python src/evaluate.py
