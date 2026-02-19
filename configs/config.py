# Project Configuration
PROJECT_CONFIG = {
    "name": "tfm-mlops-dengue",
    "version": "0.1.0",
    "description": "MLOps pipeline for dengue prediction in Brazil",
    "author": "Jose María Ponce Bernabé"
}

# Data configuration
DATA_CONFIG = {
    "raw_data_path": "data/raw",
    "interim_data_path": "data/interim", 
    "processed_data_path": "data/processed",
    "external_data_path": "data/external",
    
    # API configuration
    "mosqlimate_api": {
        "base_url": "https://api.mosqlimate.org",
        "endpoints": {
            "dengue": "/api/dengue",
            "climate": "/api/climate"
        }
    },
    
    # Date ranges
    "date_range": {
        "start_date": "2010-01-01",
        "end_date": "2025-08-31"
    }
}

# Model configuration
MODEL_CONFIG = {
    "target_variable": "nivel",
    "target_classes": [1, 2, 3, 4],
    "class_labels": ["Verde", "Amarelo", "Laranja", "Vermelho"],

    "algorithms": {
        "random_forest": {
            "n_estimators": [100, 200, 500],
            "max_depth": [10, 20, None],
            "min_samples_split": [2, 5, 10],
        },
        "xgboost": {
            "n_estimators": [100, 300, 500],
            "max_depth": [3, 6, 10],
            "learning_rate": [0.01, 0.1, 0.2],
        },
        "lightgbm": {
            "n_estimators": [100, 300, 500],
            "max_depth": [5, 10, -1],
            "learning_rate": [0.01, 0.1, 0.2],
        },
        "catboost": {
            "iterations": [200, 500],
            "depth": [4, 6, 8],
            "learning_rate": [0.03, 0.1],
        },
    },

    "validation": {
        "method": "temporal_split",
        "train_years": (2010, 2021),
        "val_years": (2022, 2023),
        "test_years": (2024, 2024),
    },

    "metrics": [
        "accuracy", "macro_f1", "weighted_f1",
        "cohen_kappa", "log_loss", "roc_auc_ovr",
    ],
    "primary_metric": "macro_f1",
}

# Feature engineering configuration
FEATURE_CONFIG = {
    "temporal_features": {
        "cycles": [5, 6],  # years
        "seasonality": True,
        "lag_periods": [1, 3, 6, 12],  # months
        "rolling_windows": [3, 6, 12]  # months
    },
    
    "climate_features": {
        "interactions": True,
        "polynomial_degree": 2,
        "normalize": True
    },
    
    "spatial_features": {
        "municipality_encoding": "region",  # 5 macro-regions, not target encoding
        "regional_aggregation": True
    }
}

# Monitoring configuration
MONITORING_CONFIG = {
    "data_drift": {
        "reference_window": "1Y",  # 1 year
        "detection_window": "1M",  # 1 month
        "threshold": 0.05
    },
    
    "model_performance": {
        "metrics": ["mae", "mape"],
        "thresholds": {
            "mae": 50,  # cases
            "mape": 0.3  # 30%
        }
    },
    
    "alerts": {
        "email": ["admin@project.com"],
        "slack_webhook": None
    }
}

# Deployment configuration  
DEPLOYMENT_CONFIG = {
    "model_registry": {
        "staging_alias": "staging",
        "production_alias": "champion"
    },
    
    "api": {
        "host": "0.0.0.0",
        "port": 8000,
        "workers": 4
    },
    
    "streamlit": {
        "host": "0.0.0.0", 
        "port": 8501,
        "title": "Dengue Prediction Brazil"
    },
    
    "docker": {
        "base_image": "python:3.11-slim",
        "requirements_file": "requirements-prod.txt"
    }
}

# Logging configuration
LOGGING_CONFIG = {
    "level": "INFO",
    "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    "handlers": {
        "file": "logs/app.log",
        "console": True
    }
}
