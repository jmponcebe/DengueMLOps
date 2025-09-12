# MLflow Configuration for Dengue Prediction Project

import os
import mlflow
from pathlib import Path

# Project configuration
PROJECT_NAME = "dengue-prediction-brazil"
EXPERIMENT_NAME = "dengue-ml-experiments"

# MLflow tracking configuration
MLFLOW_TRACKING_URI = "file:./mlruns"  # Local file store
# MLFLOW_TRACKING_URI = "http://localhost:5000"  # For MLflow server

# Artifact storage
ARTIFACT_LOCATION = "./mlflow-artifacts"

# Model registry
MODEL_REGISTRY_URI = MLFLOW_TRACKING_URI

# Tags for experiments
DEFAULT_TAGS = {
    "project": "tfm-mlops-dengue",
    "team": "data-science",
    "stage": "development",
    "country": "brazil",
    "disease": "dengue"
}

# Experiment configuration
EXPERIMENT_CONFIG = {
    "name": EXPERIMENT_NAME,
    "artifact_location": ARTIFACT_LOCATION,
    "tags": DEFAULT_TAGS
}

def setup_mlflow():
    """
    Initialize MLflow tracking and create experiment if it doesn't exist
    """
    # Set tracking URI
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    
    # Create directories if needed
    os.makedirs(ARTIFACT_LOCATION, exist_ok=True)
    
    # Create or get experiment
    try:
        experiment_id = mlflow.create_experiment(
            name=EXPERIMENT_NAME,
            artifact_location=ARTIFACT_LOCATION,
            tags=DEFAULT_TAGS
        )
        print(f"✅ Created new experiment: {EXPERIMENT_NAME} (ID: {experiment_id})")
    except mlflow.exceptions.MlflowException:
        experiment = mlflow.get_experiment_by_name(EXPERIMENT_NAME)
        experiment_id = experiment.experiment_id
        print(f"📋 Using existing experiment: {EXPERIMENT_NAME} (ID: {experiment_id})")
    
    # Set experiment as active
    mlflow.set_experiment(EXPERIMENT_NAME)
    
    return experiment_id

def get_mlflow_config():
    """
    Return MLflow configuration dictionary
    """
    return {
        "tracking_uri": MLFLOW_TRACKING_URI,
        "experiment_name": EXPERIMENT_NAME,
        "artifact_location": ARTIFACT_LOCATION,
        "project_name": PROJECT_NAME,
        "tags": DEFAULT_TAGS
    }

if __name__ == "__main__":
    setup_mlflow()
    print("🎯 MLflow configuration completed!")
    print(f"Tracking URI: {MLFLOW_TRACKING_URI}")
    print(f"Experiment: {EXPERIMENT_NAME}")
    print(f"Artifacts: {ARTIFACT_LOCATION}")
