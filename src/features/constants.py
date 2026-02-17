"""
Variable classification and constants for dengue feature engineering.
Based on analysis from EDA and production constraints.
"""

# Target variable
TARGET = 'nivel'

# Variables temporales (disponibles en predicción)
TEMPORAL_VARS = ['data_iniSE', 'SE', 'year', 'month']

# Variables geográficas (estáticas, siempre disponibles)
GEO_VARS = ['municipio_geocodigo', 'municipio_nome', 'pop', 'uf', 'state_name']

# Variables climáticas (solo disponibles con lag 4-8 semanas)
CLIMATE_VARS = ['tempmin', 'tempmed', 'tempmax', 'umidmin', 'umidmed', 'umidmax']

# Variables con target leakage (epidemiológicas actuales, IDs sin valor)
LEAKAGE_VARS = [
    'casos_est', 'casos_est_min', 'casos_est_max', 'casos',
    'p_rt1', 'p_inc100k', 'Rt', 'receptivo', 'transmissao', 'nivel_inc',
    'casprov', 'casprov_est', 'casprov_est_min', 'casprov_est_max', 'casconf',
    'versao_modelo', 'Localidade_id', 'id', 'notif_accum_year'
]

# Macro-regiones de Brasil (conocimiento de dominio, sin leakage)
REGION_MAP = {
    'AC': 'Norte', 'AP': 'Norte', 'AM': 'Norte', 'PA': 'Norte',
    'RO': 'Norte', 'RR': 'Norte', 'TO': 'Norte',
    'AL': 'Nordeste', 'BA': 'Nordeste', 'CE': 'Nordeste', 'MA': 'Nordeste',
    'PB': 'Nordeste', 'PE': 'Nordeste', 'PI': 'Nordeste', 'RN': 'Nordeste', 'SE': 'Nordeste',
    'DF': 'Centro-Oeste', 'GO': 'Centro-Oeste', 'MT': 'Centro-Oeste', 'MS': 'Centro-Oeste',
    'ES': 'Sudeste', 'MG': 'Sudeste', 'RJ': 'Sudeste', 'SP': 'Sudeste',
    'PR': 'Sul', 'RS': 'Sul', 'SC': 'Sul'
}
REGIONS = ['Norte', 'Nordeste', 'Centro-Oeste', 'Sudeste', 'Sul']

# Features engineered seleccionadas (validadas en notebook 02)
TEMPORAL_FEATURES = ['month_sin', 'month_cos', 'se_sin', 'se_cos', 'is_peak_season', 'quarter']

CLIMATE_FEATURES = ['tempmed_lag8w', 'tempmed_roll12w', 'umidmed_roll4w', 'temp_x_humid_lag4w']

GEO_FEATURES = ['pop_log', 'region_Nordeste', 'region_Centro-Oeste', 'region_Sudeste', 'region_Sul']

ALL_ENGINEERED_FEATURES = TEMPORAL_FEATURES + CLIMATE_FEATURES + GEO_FEATURES

# Columnas de identificación para producción
PRODUCTION_ID_COLS = [
    'data_iniSE', 'SE', 'year', 'month',
    'municipio_geocodigo', 'municipio_nome', 'uf', 'state_name', 'pop'
]

# Columnas del dataset de producción (sin leakage)
PRODUCTION_COLUMNS = PRODUCTION_ID_COLS + ALL_ENGINEERED_FEATURES + [TARGET]

# Variables base para lags climáticos
CLIMATE_BASE_VARS = ['tempmed', 'umidmed']
LAG_PERIODS = [4, 6, 8]
ROLLING_WINDOWS = [4, 8, 12]
