"""
Data module for dengue prediction project.

Main components:
- MosqlimateAPIClient: Fetch data from mosqlimate API
- MosqlimateDataLoader: Load data from local parquet files
- DataProcessor: Clean and standardize data

Usage:
    from data import sync_dataset, MosqlimateDataLoader

    # Sincronizar datos (descarga completo o actualiza faltantes)
    sync_dataset()

    # Cargar datos
    loader = MosqlimateDataLoader()
    df = loader.load_all_states_data()
"""

from .api_client import MosqlimateAPIClient, sync_dataset
from .mosqlimate_loader import MosqlimateDataLoader, DataProcessor

__all__ = [
    'MosqlimateAPIClient',
    'MosqlimateDataLoader',
    'DataProcessor',
    'sync_dataset'
]
