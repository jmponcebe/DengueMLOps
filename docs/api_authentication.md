# Autenticación API Mosqlimate

## Resumen

La API de Mosqlimate requiere un **X-UID-Key** para descargar datos epidemiológicos.

> **Nota**: El modo por defecto de `scripts/setup_data.py` (modelo champion + GeoJSON)
> **no** requiere API key. Solo los modos que descargan datos de Mosqlimate
> (`--latest`, `--data`, `--all`) necesitan la clave configurada.

## Cómo Obtener el API Key

1. **Ir a** <https://api.mosqlimate.org/>
2. **Autenticarse con GitHub** (botón de login)
3. **Ir a tu perfil**: `<https://api.mosqlimate.org/<tu_usuario>/`
4. **Copiar el UID-Key** que aparece en tu perfil

## Uso en Python

```python
from src.data.api_client import MosqlimateAPIClient

# Opción 1: Pasar directamente
client = MosqlimateAPIClient(api_key='tu_uid_key_aqui')

# Opción 2: Variable de entorno
import os
os.environ['MOSQLIMATE_API_KEY'] = 'tu_uid_key_aqui'
client = MosqlimateAPIClient(api_key=os.environ.get('MOSQLIMATE_API_KEY'))

# Obtener datos
df = client.fetch_dengue_data(
    uf='SP',
    start='2024-01-01',
    end='2024-12-31',
    disease='dengue'
)
```

## Header de Autenticación

El cliente envía automáticamente:

```text
X-UID-Key: <tu_api_key>
```

## Endpoints Disponibles

| Endpoint | Descripción | Frecuencia |
| ---------- | ------------- | ------------ |
| `/api/datastore/infodengue/` | Datos epidemiológicos | Semanal |
| `/api/datastore/climate/` | Series climáticas | Diaria |
| `/api/datastore/climate/weekly/` | Clima agregado | Semanal |
| `/api/datastore/mosquito/` | Abundancia mosquitos | Variable |
| `/api/datastore/episcanner/` | Parámetros epidémicos | Anual |

## Mientras no tengas API Key

Los datos existentes (2010-ago 2025) son suficientes para el TFM:

- 96.9% de cobertura
- 5,022 archivos parquet
- 27 estados completos
