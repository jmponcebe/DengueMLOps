# Cliente API Mosqlimate - Guía de Uso

## Requisitos

**API Key requerido**. Ver [api_authentication.md](api_authentication.md) para obtenerlo.

## Uso Básico

```python
from src.data.api_client import MosqlimateAPIClient, sync_dataset

# Con API key
client = MosqlimateAPIClient(api_key='tu_x_uid_key')

# Descargar datos de un estado
df = client.fetch_dengue_data(
    uf='SP',
    start='2024-01-01',
    end='2024-12-31',
    disease='dengue'
)

# O sincronizar todo el dataset
sync_dataset(year_start=2010)  # Requiere MOSQLIMATE_API_KEY en env
```

## Endpoints Disponibles

El cliente soporta el endpoint principal de Infodengue:

| Parámetro | Requerido | Descripción |
| ----------- | ----------- | ------------- |
| `disease` | Sí | 'dengue', 'zika', 'chik' |
| `start` | Sí | Fecha inicio (YYYY-MM-DD) |
| `end` | Sí | Fecha fin (YYYY-MM-DD) |
| `uf` | No | Estado (ej: 'SP', 'RJ') |
| `geocode` | No | Código IBGE municipio |

## Sincronización Inteligente

```python
from src.data.api_client import sync_dataset

# Detecta automáticamente qué falta y solo descarga eso
sync_dataset(year_start=2010)
```

Funcionamiento:

1. Detecta archivos existentes en `data/raw/historical_api_data/`
2. Calcula meses faltantes entre year_start y fecha actual
3. Descarga solo lo necesario

## Pipeline Completo

```python
# 1. Sincronizar desde API (requiere API key)
from src.data.api_client import sync_dataset
sync_dataset()

# 2. Cargar datos locales (no requiere API key)
from src.data.mosqlimate_loader import MosqlimateDataLoader
loader = MosqlimateDataLoader()
df = loader.load_all_states_data()
```

## Características

- **Paginación automática**: Maneja respuestas paginadas
- **Retry con backoff**: Reintenta peticiones fallidas
- **Rate limiting**: 0.5s entre peticiones
- **Detección de faltantes**: No descarga archivos existentes

## Testing

```bash
# Ejecutar tests
pytest tests/test_data_api.py -v

# Test completo con cobertura
pytest tests/test_data_api.py -v --cov=src/data
```

## Estructura de Datos

**Entrada**: API mosqlimate  
**Salida**: Parquet files organizados por estado/año/mes  
**Formato**: DataFrame con columnas epidemiológicas y climáticas

Ver `src/data/mosqlimate_loader.py` para detalles de columnas disponibles.

## Referencias

- **API Docs**: <https://api.mosqlimate.org/docs>
- **Código fuente**: `src/data/api_client.py`
- **Tests**: `tests/test_data_api.py`
