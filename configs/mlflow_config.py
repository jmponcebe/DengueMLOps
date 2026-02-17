# MLflow Configuration for Dengue Prediction Project

import os
import mlflow
from mlflow.exceptions import MlflowException
from pathlib import Path

# Project root — resuelve paths absolutos independientemente del CWD
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Project configuration
PROJECT_NAME = "dengue-prediction-brazil"
REGISTERED_MODEL_NAME = "dengue-alertlevel-classifier"

# MLflow tracking — paths absolutos para que funcione desde notebooks/ o raíz
MLFLOW_TRACKING_URI = (PROJECT_ROOT / "mlruns").as_uri()
# MLFLOW_TRACKING_URI = "http://localhost:5000"  # For MLflow server

# Artifact storage
ARTIFACT_LOCATION = str(PROJECT_ROOT / "mlflow-artifacts")

# Directorio temporal para artefactos antes de log_artifact()
ARTIFACT_TEMP_DIR = str(PROJECT_ROOT / "models" / "tmp_artifacts")

# Experiments por fase de experimentación
EXPERIMENTS = {
    "baselines": "01-baselines",
    "model_selection": "02-model-selection",
    "hyperparameter_tuning": "03-hyperparameter-tuning",
    "class_imbalance": "04-class-imbalance",
    "final_evaluation": "05-final-evaluation",
}

# Tags comunes a todos los experiments
DEFAULT_TAGS = {
    "project": "tfm-mlops-dengue",
    "target": "nivel",
    "country": "brazil",
    "disease": "dengue",
}

# Métricas principales para comparación
PRIMARY_METRIC = "macro_f1"
METRICS_TO_LOG = [
    "accuracy", "macro_f1", "weighted_f1", "cohen_kappa",
    "log_loss", "roc_auc_ovr",
]

# Temporal split boundaries
SPLIT_CONFIG = {
    "train_end": 2021,
    "val_end": 2023,
    "test_start": 2024,
}


def setup_experiment(phase: str) -> str:
    """
    Configura MLflow para una fase específica de experimentación.
    
    Args:
        phase: clave de EXPERIMENTS ('baselines', 'model_selection', etc.)
    Returns:
        experiment_id
    """
    if phase not in EXPERIMENTS:
        raise ValueError(f"Phase '{phase}' no válida. Opciones: {list(EXPERIMENTS.keys())}")

    experiment_name = EXPERIMENTS[phase]
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    os.makedirs(ARTIFACT_LOCATION, exist_ok=True)

    try:
        experiment_id = mlflow.create_experiment(
            name=experiment_name,
            artifact_location=ARTIFACT_LOCATION,
            tags={**DEFAULT_TAGS, "phase": phase},
        )
        print(f"Created experiment: {experiment_name} (ID: {experiment_id})")
    except MlflowException:
        experiment = mlflow.get_experiment_by_name(experiment_name)
        assert experiment is not None, f"Experiment '{experiment_name}' not found"
        experiment_id = experiment.experiment_id
        print(f"Using experiment: {experiment_name} (ID: {experiment_id})")

    mlflow.set_experiment(experiment_name)
    return experiment_id


def setup_all_experiments():
    """Crea todos los experiments del proyecto de una vez."""
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    os.makedirs(ARTIFACT_LOCATION, exist_ok=True)

    ids = {}
    for phase, name in EXPERIMENTS.items():
        try:
            eid = mlflow.create_experiment(
                name=name,
                artifact_location=ARTIFACT_LOCATION,
                tags={**DEFAULT_TAGS, "phase": phase},
            )
        except MlflowException:
            exp = mlflow.get_experiment_by_name(name)
            assert exp is not None, f"Experiment '{name}' not found"
            eid = exp.experiment_id
        ids[phase] = eid
        print(f"  {name} -> ID {eid}")

    return ids


def get_mlflow_config():
    """Return MLflow configuration dictionary."""
    return {
        "tracking_uri": MLFLOW_TRACKING_URI,
        "experiments": EXPERIMENTS,
        "artifact_location": ARTIFACT_LOCATION,
        "project_name": PROJECT_NAME,
        "registered_model_name": REGISTERED_MODEL_NAME,
        "primary_metric": PRIMARY_METRIC,
        "tags": DEFAULT_TAGS,
        "split_config": SPLIT_CONFIG,
    }


if __name__ == "__main__":
    print("Setting up MLflow experiments...")
    ids = setup_all_experiments()
    print(f"\nTracking URI: {MLFLOW_TRACKING_URI}")
    print(f"Artifacts: {ARTIFACT_LOCATION}")
    print(f"Experiments: {len(ids)}")
