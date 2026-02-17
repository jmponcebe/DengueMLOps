# Feature Engineering — Documentación Técnica

## Contexto y Motivación

El dengue es una enfermedad vectorial transmitida por el mosquito *Aedes aegypti*. La predicción a nivel municipal requiere features que capturen la **estacionalidad**, las **condiciones climáticas previas** y la **heterogeneidad geográfica** de Brasil, sin usar información que no estaría disponible en el momento real de predicción.

### Restricción Fundamental: No Target Leakage

En el dataset de mosqlimate, las variables epidemiológicas (`casos`, `Rt`, `p_inc100k`, etc.) y climáticas actuales (`tempmed`, `umidmed`) corresponden al **mismo período** que el target `nivel`. Usar estas variables como predictores sería circular — equivale a predecir la severidad del brote usando indicadores derivados del propio brote.

En producción, al hacer predicción para la semana N:

- **Sí disponemos**: Datos temporales (es abril, semana 16), población, ubicación geográfica, clima de semanas anteriores
- **No disponemos**: Clima de la semana objetivo, casos actuales, indicadores epidemiológicos

## Pipeline de Feature Engineering

### 1. Features Temporales (6)

|Feature|Descripción|Correlación con target|
|---------|-------------|----------------------|
|`month_sin`|Seno del mes (ciclo anual)|+0.202|
|`month_cos`|Coseno del mes (ciclo anual)|-0.011|
|`se_sin`|Seno de semana epidemiológica|+0.036|
|`se_cos`|Coseno de semana epidemiológica|+0.018|
|`is_peak_season`|1 si mes ∈ {Ene, Feb, Mar, Abr}|+0.178|
|`quarter`|Trimestre del año (1-4)|-0.166|

**Justificación**: El dengue presenta estacionalidad marcada con pico en marzo-abril y valle en septiembre (ratio 9.9x). El encoding cíclico sin/cos captura la naturaleza circular del tiempo; `is_peak_season` es un indicador directo del período de alta transmisión.

Los pares cíclicos (sin+cos) se mantienen **completos** incluso cuando un componente tiene baja correlación individual, porque juntos definen un punto único en el ciclo.

### 2. Features Climáticas (4)

|Feature|Descripción|Correlación|
|---------|-------------|-------------|
|`tempmed_lag8w`|Temperatura media hace 8 semanas|+0.089|
|`tempmed_roll12w`|Media móvil 12 semanas (con shift)|+0.089|
|`umidmed_roll4w`|Media móvil humedad 4 semanas (con shift)|+0.067|
|`temp_x_humid_lag4w`|Interacción temp×humedad lag 4w|+0.114|

**Justificación epidemiológica**: El ciclo de vida del vector (huevo → larva → pupa → adulto: 8-12 días) más la incubación extrínseca (8-12 días) implica que las condiciones climáticas afectan la transmisión con **4-8 semanas de retraso**. Las medias móviles capturan tendencias persistentes.

**Detalle técnico**: Se usa `shift(1)` antes de `rolling()` para excluir la observación actual de la ventana móvil. El lag se aplica con `groupby('municipio_geocodigo').shift(n)` para evitar contaminación entre municipios.

**Interacción temp×humedad**: La interacción captura condiciones tropicales favorables para el vector (calor húmedo). Tiene la mayor correlación individual entre features climáticas (+0.114), superando a los lags simples.

#### Proceso de selección

Se evalúan 14 candidatas (6 lags × 2 variables + 6 rolling × 2 variables + 2 interacciones) y se selecciona:

- El mejor lag por variable base (por correlación absoluta con target)
- El mejor rolling por variable base
- Las interacciones con correlación > 30% del baseline

### 3. Features Geográficas (5)

|Feature|Descripción|Correlación|
|---------|-------------|-------------|
|`pop_log`|Log de población municipal|+0.224|
|`region_Nordeste`|Dummy macro-región Nordeste|-0.037|
|`region_Centro-Oeste`|Dummy macro-región Centro-Oeste|+0.033|
|`region_Sudeste`|Dummy macro-región Sudeste|+0.079|
|`region_Sul`|Dummy macro-región Sul|-0.067|

**`pop_log`** es la feature individual más potente (+0.224). Las ciudades grandes concentran más casos por mayor densidad vectorial, movilidad humana y capacidad de detección. El logaritmo normaliza la distribución fuertemente sesgada.

**Macro-regiones**: Las 5 regiones de Brasil (Norte, Nordeste, Centro-Oeste, Sudeste, Sul) capturan diferencias climáticas y epidemiológicas estructurales. Se usa one-hot encoding en vez de target encoding para evitar completamente el riesgo de leakage. `region_Norte` se omite como categoría de referencia (se incluyen N-1 dummies).

### Ranking Final de Features

```text
  1. pop_log                   | +0.224
  2. month_sin                 | +0.202
  3. is_peak_season            | +0.178
  4. quarter                   | -0.166
  5. temp_x_humid_lag4w        | +0.114
  6. tempmed_roll12w           | +0.089
  7. tempmed_lag8w             | +0.089
  8. region_Sudeste            | +0.079
  9. umidmed_roll4w            | +0.067
 10. region_Sul                | -0.067
 11. region_Nordeste           | -0.037
 12. se_sin                    | +0.036
 13. region_Centro-Oeste       | +0.033
 14. se_cos                    | +0.018
 15. month_cos                 | -0.011
```

## Validación con Modelo

Se compara un RandomForest baseline (year + month + pop) vs RF con las 15 features adicionales:

|Métrica|Baseline|Con FE|Mejora|
|---------|----------|--------|--------|
|Accuracy|0.877|0.877|+0.000|
|F1 weighted|0.821|0.822|+0.001|

La mejora marginal es **esperada** por tres razones:

1. **Desbalance extremo** (93.1% nivel 1): ambos modelos aciertan prediciendo la clase dominante
2. **RF rápido** (100 árboles, 150K subsample) no es una evaluación final
3. Feature importance muestra que las features engineered **dominan** las importancias relativas

La evaluación definitiva será en el notebook de modeling con class weighting, oversampling y métricas per-class.

## Datasets Generados

### Comprehensive (60 columnas)

Todas las variables originales + 15 features engineered + variables intermedias de lag. Para investigación y análisis.

### Producción (25 columnas)

- 9 columnas de identificación: `data_iniSE`, `SE`, `year`, `month`, `municipio_geocodigo`, `municipio_nome`, `uf`, `state_name`, `pop`
- 15 features engineered
- 1 target: `nivel`

Sin variables climáticas del mismo período, sin variables epidemiológicas, sin metadatos.

## Módulo de Producción

```python
from src.features import DengueFeatureEngineer

fe = DengueFeatureEngineer()
df_features = fe.transform(df_raw)       # Añade 15 features
df_prod = fe.get_production_dataset(df_features)  # Filtra a 25 cols seguras
```

Constantes y clasificación de variables en `src/features/constants.py`. Tests en `tests/test_features.py` (26 tests, 97% cobertura).
