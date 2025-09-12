# Data ingestion and processing utilities
"""
Data module for dengue prediction project.
Handles data loading, cleaning, and initial processing.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import requests
import json
from typing import Optional, Dict, List
import logging

logger = logging.getLogger(__name__)

class DataLoader:
    """
    Data loader for dengue and climate data from various sources
    """
    
    def __init__(self, data_path: str = "data/raw"):
        self.data_path = Path(data_path)
        self.data_path.mkdir(parents=True, exist_ok=True)
        
    def load_historical_dengue(self, filepath: Optional[str] = None) -> pd.DataFrame:
        """
        Load historical dengue data
        
        Args:
            filepath: Optional path to dengue data file
            
        Returns:
            DataFrame with dengue cases data
        """
        if filepath is None:
            filepath = self.data_path / "historical_api_data" / "dengue_cases.csv"
            
        try:
            df = pd.read_csv(filepath)
            df['date'] = pd.to_datetime(df['date'])
            logger.info(f"Loaded dengue data: {len(df)} records")
            return df
        except FileNotFoundError:
            logger.warning(f"File not found: {filepath}")
            return pd.DataFrame()
            
    def load_climate_data(self, filepath: Optional[str] = None) -> pd.DataFrame:
        """
        Load climate data
        
        Args:
            filepath: Optional path to climate data file
            
        Returns:
            DataFrame with climate variables
        """
        if filepath is None:
            filepath = self.data_path / "climate" / "climate_data.csv"
            
        try:
            df = pd.read_csv(filepath)
            df['date'] = pd.to_datetime(df['date'])
            logger.info(f"Loaded climate data: {len(df)} records")
            return df
        except FileNotFoundError:
            logger.warning(f"File not found: {filepath}")
            return pd.DataFrame()
            
    def load_openapi_spec(self, filepath: Optional[str] = None) -> Dict:
        """
        Load OpenAPI specification for mosqlimate API
        
        Args:
            filepath: Path to openapi.json file
            
        Returns:
            Dictionary with API specification
        """
        if filepath is None:
            filepath = self.data_path / "openapi.json"
            
        try:
            with open(filepath, 'r') as f:
                spec = json.load(f)
            logger.info("Loaded OpenAPI specification")
            return spec
        except FileNotFoundError:
            logger.warning(f"OpenAPI spec not found: {filepath}")
            return {}

class DataCleaner:
    """
    Data cleaning and validation utilities
    """
    
    @staticmethod
    def clean_dengue_data(df: pd.DataFrame) -> pd.DataFrame:
        """
        Clean dengue data
        
        Args:
            df: Raw dengue dataframe
            
        Returns:
            Cleaned dataframe
        """
        df_clean = df.copy()
        
        # Remove negative cases
        df_clean = df_clean[df_clean['casos_dengue'] >= 0]
        
        # Handle missing values
        df_clean['casos_dengue'] = df_clean['casos_dengue'].fillna(0)
        
        # Remove outliers (cases > 99.9 percentile)
        upper_bound = df_clean['casos_dengue'].quantile(0.999)
        df_clean = df_clean[df_clean['casos_dengue'] <= upper_bound]
        
        logger.info(f"Cleaned dengue data: {len(df_clean)} records retained")
        return df_clean
        
    @staticmethod
    def clean_climate_data(df: pd.DataFrame) -> pd.DataFrame:
        """
        Clean climate data
        
        Args:
            df: Raw climate dataframe
            
        Returns:
            Cleaned dataframe
        """
        df_clean = df.copy()
        
        # Temperature bounds (Brazil: -10°C to 50°C)
        df_clean = df_clean[
            (df_clean['temperatura_media'] >= -10) & 
            (df_clean['temperatura_media'] <= 50)
        ]
        
        # Precipitation bounds (0 to 1000mm/month)
        df_clean = df_clean[
            (df_clean['precipitacion'] >= 0) & 
            (df_clean['precipitacion'] <= 1000)
        ]
        
        # Humidity bounds (0 to 100%)
        df_clean = df_clean[
            (df_clean['humedad'] >= 0) & 
            (df_clean['humedad'] <= 100)
        ]
        
        # Fill missing values with median by municipality
        climate_vars = ['temperatura_media', 'precipitacion', 'humedad']
        for var in climate_vars:
            df_clean[var] = df_clean.groupby('municipio')[var].transform(
                lambda x: x.fillna(x.median())
            )
        
        logger.info(f"Cleaned climate data: {len(df_clean)} records retained")
        return df_clean

class DataMerger:
    """
    Utilities for merging different data sources
    """
    
    @staticmethod
    def merge_dengue_climate(
        dengue_df: pd.DataFrame, 
        climate_df: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Merge dengue and climate data
        
        Args:
            dengue_df: Dengue cases dataframe
            climate_df: Climate variables dataframe
            
        Returns:
            Merged dataframe
        """
        # Ensure date columns are datetime
        dengue_df['date'] = pd.to_datetime(dengue_df['date'])
        climate_df['date'] = pd.to_datetime(climate_df['date'])
        
        # Merge on municipality and date
        merged_df = pd.merge(
            dengue_df,
            climate_df,
            on=['municipio', 'date'],
            how='inner'
        )
        
        logger.info(f"Merged data: {len(merged_df)} records")
        return merged_df

def load_sample_data() -> pd.DataFrame:
    """
    Load or generate sample data for development/testing
    
    Returns:
        Sample dataframe with dengue and climate data
    """
    logger.info("Generating sample data for development")
    
    np.random.seed(42)
    municipios = ['São Paulo', 'Rio de Janeiro', 'Salvador', 'Brasília', 'Fortaleza']
    dates = pd.date_range('2010-01-01', '2025-08-31', freq='M')
    
    data = []
    for municipio in municipios:
        for date in dates:
            # Simulate 5-year cycles
            year_cycle = np.sin(2 * np.pi * (date.year - 2010) / 5.5)
            seasonal = np.sin(2 * np.pi * date.month / 12)
            
            base_cases = 50 + 30 * year_cycle + 20 * seasonal
            cases = max(0, base_cases + np.random.normal(0, 10))
            
            data.append({
                'municipio': municipio,
                'date': date,
                'casos_dengue': int(cases),
                'temperatura_media': 25 + 5 * np.sin(2 * np.pi * date.month / 12) + np.random.normal(0, 2),
                'precipitacion': max(0, 100 + 50 * np.sin(2 * np.pi * (date.month - 3) / 12) + np.random.normal(0, 30)),
                'humedad': 60 + 15 * np.sin(2 * np.pi * date.month / 12) + np.random.normal(0, 5)
            })
    
    df = pd.DataFrame(data)
    logger.info(f"Generated sample data: {len(df)} records")
    return df
