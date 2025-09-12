"""
Real data loader for dengue historical data from mosqlimate API
Handles the hierarchical parquet structure: uf={state}/year={year}/month={month}/data.parquet
"""

import pandas as pd
import numpy as np
from pathlib import Path
import json
from typing import List, Optional, Dict, Tuple
import logging
from tqdm import tqdm

logger = logging.getLogger(__name__)

class MosqlimateDataLoader:
    """
    Loader for historical dengue data from mosqlimate API stored in parquet format
    """
    
    def __init__(self, data_path: Optional[str] = None):
        # Usar ruta simple por defecto (funciona desde notebooks)
        if data_path is None:
            data_path = "../data/raw/historical_api_data"
                
        self.data_path = Path(data_path)
        self.states_mapping = {
            'AC': 'Acre', 'AL': 'Alagoas', 'AP': 'Amapá', 'AM': 'Amazonas',
            'BA': 'Bahia', 'CE': 'Ceará', 'DF': 'Distrito Federal', 'ES': 'Espírito Santo',
            'GO': 'Goiás', 'MA': 'Maranhão', 'MT': 'Mato Grosso', 'MS': 'Mato Grosso do Sul',
            'MG': 'Minas Gerais', 'PA': 'Pará', 'PB': 'Paraíba', 'PR': 'Paraná',
            'PE': 'Pernambuco', 'PI': 'Piauí', 'RJ': 'Rio de Janeiro', 'RN': 'Rio Grande do Norte',
            'RS': 'Rio Grande do Sul', 'RO': 'Rondônia', 'RR': 'Roraima', 'SC': 'Santa Catarina',
            'SP': 'São Paulo', 'SE': 'Sergipe', 'TO': 'Tocantins'
        }
        
    def get_available_states(self) -> List[str]:
        """Get list of available states in the data"""
        states = []
        for path in self.data_path.iterdir():
            if path.is_dir() and path.name.startswith('uf='):
                state = path.name.replace('uf=', '')
                states.append(state)
        return sorted(states)
    
    def get_available_years(self, state: str) -> List[int]:
        """Get available years for a specific state"""
        state_path = self.data_path / f"uf={state}"
        if not state_path.exists():
            return []
            
        years = []
        for path in state_path.iterdir():
            if path.is_dir() and path.name.startswith('year='):
                year = int(path.name.replace('year=', ''))
                years.append(year)
        return sorted(years)
    
    def get_data_summary(self) -> Dict:
        """Get summary of available data"""
        summary = {
            'states': self.get_available_states(),
            'total_states': 0,
            'date_range': {'start': None, 'end': None},
            'total_files': 0
        }
        
        all_years = []
        total_files = 0
        
        for state in summary['states']:
            years = self.get_available_years(state)
            all_years.extend(years)
            
            # Count files for this state
            for year in years:
                year_path = self.data_path / f"uf={state}" / f"year={year}"
                if year_path.exists():
                    for month_path in year_path.iterdir():
                        if month_path.is_dir() and (month_path / "data.parquet").exists():
                            total_files += 1
        
        summary['total_states'] = len(summary['states'])
        summary['total_files'] = total_files
        
        if all_years:
            summary['date_range']['start'] = f"{min(all_years)}-01-01"
            summary['date_range']['end'] = f"{max(all_years)}-12-31"
            
        return summary
        
    def load_state_data(
        self, 
        state: str, 
        year_start: int = 2010, 
        year_end: int = 2025
    ) -> pd.DataFrame:
        """
        Load data for a specific state and date range
        
        Args:
            state: State code (e.g., 'SP', 'RJ')
            year_start: Start year
            year_end: End year
            
        Returns:
            DataFrame with dengue data for the state
        """
        state_path = self.data_path / f"uf={state}"
        if not state_path.exists():
            logger.warning(f"State {state} not found in data")
            return pd.DataFrame()
            
        dataframes = []
        years = range(year_start, year_end + 1)
        
        for year in tqdm(years, desc=f"Loading {state} data"):
            year_path = state_path / f"year={year}"
            if not year_path.exists():
                continue
                
            for month in range(1, 13):
                month_path = year_path / f"month={month:02d}"
                data_file = month_path / "data.parquet"
                
                if data_file.exists():
                    try:
                        df_month = pd.read_parquet(data_file)
                        df_month['uf'] = state
                        df_month['state_name'] = self.states_mapping.get(state, state)
                        df_month['year'] = year
                        df_month['month'] = month
                        dataframes.append(df_month)
                    except Exception as e:
                        logger.warning(f"Error loading {data_file}: {e}")
                        
        if dataframes:
            df = pd.concat(dataframes, ignore_index=True)
            logger.info(f"Loaded {len(df)} records for state {state}")
            return df
        else:
            logger.warning(f"No data found for state {state}")
            return pd.DataFrame()
    
    def load_sample_file(self, state: str = 'SP', year: int = 2024, month: int = 1) -> pd.DataFrame:
        """Load a sample file to inspect data structure"""
        file_path = self.data_path / f"uf={state}" / f"year={year}" / f"month={month:02d}" / "data.parquet"
        
        if file_path.exists():
            df = pd.read_parquet(file_path)
            logger.info(f"Sample file loaded: {len(df)} records from {file_path}")
            return df
        else:
            logger.warning(f"Sample file not found: {file_path}")
            return pd.DataFrame()
    
    def load_all_states_data(
        self, 
        states: Optional[List[str]] = None,
        year_start: int = 2010,
        year_end: int = 2025,
        sample_only: bool = False
    ) -> pd.DataFrame:
        """
        Load data for multiple states
        
        Args:
            states: List of state codes. If None, loads all available
            year_start: Start year
            year_end: End year
            sample_only: If True, loads only 2023-2024 data for faster processing
            
        Returns:
            Combined DataFrame with all states data
        """
        if states is None:
            states = self.get_available_states()
            
        if sample_only:
            year_start = max(year_start, 2023)
            year_end = min(year_end, 2024)
            logger.info(f"Sample mode: loading {year_start}-{year_end} only")
            
        all_dataframes = []
        
        for state in tqdm(states, desc="Loading states"):
            df_state = self.load_state_data(state, year_start, year_end)
            if not df_state.empty:
                all_dataframes.append(df_state)
                
        if all_dataframes:
            df_combined = pd.concat(all_dataframes, ignore_index=True)
            logger.info(f"Combined dataset: {len(df_combined)} records from {len(all_dataframes)} states")
            return df_combined
        else:
            logger.warning("No data loaded")
            return pd.DataFrame()

class DataProcessor:
    """
    Process and clean the loaded dengue data
    """
    
    @staticmethod
    def inspect_data_structure(df: pd.DataFrame) -> Dict:
        """Inspect the structure of loaded mosqlimate data"""
        if df.empty:
            return {"error": "DataFrame is empty"}
            
        info = {
            "shape": df.shape,
            "columns": list(df.columns),
            "dtypes": df.dtypes.to_dict(),
            "missing_values": df.isnull().sum().to_dict(),
            "date_range": None,
            "unique_values": {}
        }
        
        # Check for mosqlimate date column
        if 'data_iniSE' in df.columns:
            try:
                date_series = pd.to_datetime(df['data_iniSE'])
                info["date_range"] = {
                    "start": str(date_series.min()),
                    "end": str(date_series.max())
                }
            except:
                pass
        
        # Get unique values for all categorical columns
        categorical_dtypes = ['object', 'category', 'string']
        for col in df.columns:
            if df[col].dtype.name in categorical_dtypes or df[col].dtype == 'object':
                unique_count = df[col].nunique()
                if unique_count <= 100:  # Show unique values if reasonable number
                    info["unique_values"][col] = sorted(df[col].dropna().unique().tolist())
                else:
                    info["unique_values"][col] = f"{unique_count} unique values"
        
        # Add specific mosqlimate insights
        if 'nivel' in df.columns:
            info["risk_levels"] = df['nivel'].value_counts().to_dict()
            
        if 'casos' in df.columns:
            info["cases_stats"] = {
                "total": int(df['casos'].sum()),
                "mean": float(df['casos'].mean()),
                "max": int(df['casos'].max()),
                "zero_cases": int((df['casos'] == 0).sum())
            }
                
        return info
    
    @staticmethod
    def standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
        """Standardize column names and data types based on real mosqlimate structure"""
        df_clean = df.copy()
        
        # Mapeo completo de columnas basado en datos reales de mosqlimate
        column_mappings = {
            # Temporal
            'data_iniSE': 'date',
            'SE': 'semana_epidemiologica',
            
            # Casos y estimaciones
            'casos_est': 'casos_estimados',
            'casos_est_min': 'casos_estimados_min', 
            'casos_est_max': 'casos_estimados_max',
            'casos': 'casos_dengue',  # Ya está en español
            'casprov': 'casos_probables',
            'casprov_est': 'casos_probables_estimados',
            'casprov_est_min': 'casos_probables_min',
            'casprov_est_max': 'casos_probables_max',
            'casconf': 'casos_confirmados',
            
            # Geografía
            'municipio_geocodigo': 'geocodigo_municipio',  # Ya está en español
            'municipio_nome': 'nombre_municipio',  # Ya está en español
            'Localidade_id': 'localidad_id',
            'uf': 'uf',  # Ya está en español
            'state_name': 'nombre_estado',
            
            # Epidemiológicos
            'nivel': 'nivel_alerta',
            'p_inc100k': 'incidencia_100k',
            'nivel_inc': 'nivel_incidencia',
            'p_rt1': 'prob_rt_mayor_1',
            'Rt': 'factor_reproduccion',
            'receptivo': 'receptividad',
            'transmissao': 'transmision',
            
            # Climáticos
            'tempmed': 'temperatura_media',
            'tempmin': 'temperatura_min',
            'tempmax': 'temperatura_max',
            'umidmed': 'humedad_media',
            'umidmin': 'humedad_min',
            'umidmax': 'humedad_max',
            
            # Demografía y metadatos
            'pop': 'poblacion',
            'id': 'id_registro',
            'versao_modelo': 'version_modelo',
            'year': 'year',
            'month': 'month'
        }
        
        # Aplicar mapeos si las columnas existen
        for old_col, new_col in column_mappings.items():
            if old_col in df_clean.columns:
                df_clean = df_clean.rename(columns={old_col: new_col})
        
        # Asegurar que date sea datetime
        if 'date' in df_clean.columns:
            df_clean['date'] = pd.to_datetime(df_clean['date'], errors='coerce')
            
        return df_clean

def load_openapi_spec(file_path: Optional[str] = None) -> Dict:
    """Load the OpenAPI specification"""
    if file_path is None:
        file_path = "../data/raw/openapi.json"
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            spec = json.load(f)
        logger.info("OpenAPI specification loaded successfully")
        return spec
    except Exception as e:
        logger.error(f"Error loading OpenAPI spec: {e}")
        return {}

# Example usage functions
def quick_data_exploration():
    """Quick exploration of the available data"""
    loader = MosqlimateDataLoader()
    
    # Get data summary
    summary = loader.get_data_summary()
    print("📊 Data Summary:")
    print(f"States available: {summary['total_states']}")
    print(f"Date range: {summary['date_range']['start']} to {summary['date_range']['end']}")
    print(f"Total files: {summary['total_files']}")
    print(f"States: {summary['states'][:10]}...")  # Show first 10
    
    # Load sample file
    print("\n🔍 Sample file structure:")
    df_sample = loader.load_sample_file('SP', 2024, 1)
    if not df_sample.empty:
        processor = DataProcessor()
        structure = processor.inspect_data_structure(df_sample)
        print(f"Shape: {structure['shape']}")
        print(f"Columns: {structure['columns']}")
        print("First few rows:")
        print(df_sample.head())
    
    return summary, df_sample

if __name__ == "__main__":
    # Run quick exploration
    summary, sample_df = quick_data_exploration()
