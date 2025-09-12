# Project Configuration
PROJECT_CONFIG = {
    "name": "tfm-mlops-dengue",
    "version": "0.1.0",
    "description": "MLOps pipeline for dengue prediction in Brazil",
    "author": "TFM Student"
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
    "target_variable": "casos_dengue",
    "algorithms": {
        "xgboost": {
            "n_estimators": [100, 200, 500],
            "max_depth": [3, 6, 10],
            "learning_rate": [0.01, 0.1, 0.2]
        },
        "random_forest": {
            "n_estimators": [100, 200],
            "max_depth": [10, 20, None],
            "min_samples_split": [2, 5, 10]
        },
        "lightgbm": {
            "n_estimators": [100, 200],
            "max_depth": [5, 10],
            "learning_rate": [0.01, 0.1]
        }
    },
    
    "validation": {
        "method": "time_series_split",
        "n_splits": 5,
        "test_size": 0.2
    },
    
    "metrics": [
        "mae", "mse", "rmse", "mape", "r2"
    ]
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
        "municipality_encoding": "target",
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
        "production_alias": "production"
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
        "base_image": "python:3.9-slim",
        "requirements_file": "requirements.txt"
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
