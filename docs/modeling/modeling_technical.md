# Modelado - Documentación Técnica

## Resumen

Clasificación multiclase del nivel de alerta de dengue (1-4) para municipios de Brasil. Pipeline completo desde baselines hasta un modelo champion registrado en MLflow Model Registry.

## Experimental Setup

### Datos

| Split | Periodo | Registros |
| ------- | --------- | ----------- |
| Train | 2010-2021 | 3,482,380 |
| Validation | 2022-2023 | 584,013 |
| Test | 2024 | 289,640 |

**15 features de producción** sin target leakage:

- Temporales (6): `month_sin`, `month_cos`, `se_sin`, `se_cos`, `is_peak_season`, `quarter`
- Climáticas (4): `tempmed_lag8w`, `tempmed_roll12w`, `umidmed_roll4w`, `temp_x_humid_lag4w`
- Geográficas (5): `pop_log`, `region_Nordeste`, `region_Centro-Oeste`, `region_Sudeste`, `region_Sul`

**NaN handling**: 14.1% de nulls en features climáticos (lag). XGBoost/LightGBM/CatBoost los manejan nativamente. Para sklearn se usa `SimpleImputer(strategy='median')`.

### Distribución del Target

| Nivel | Descripción | Proporción |
| ------- | ------------- | ------------ |
| 1 (verde) | Sin alerta | 93.1% |
| 2 (amarillo) | Atención | 3.9% |
| 3 (naranja) | Alerta | 0.2% |
| 4 (rojo) | Alerta crítica | 2.8% |

Desbalance extremo de 460:1 entre la clase mayoritaria y la minoritaria (nivel 3).

### Métricas

- **Primary**: `macro_f1` — trata todas las clases por igual, penaliza ignorar clases minoritarias
- **Secondary**: `cohen_kappa` — concordancia ajustada por azar, relevante con desbalance
- **Monitoring**: `accuracy`, `weighted_f1`, `log_loss`, `roc_auc_ovr`, `f1_class_{1-4}`

## Estructura de Experimentos en MLflow

5 experiments organizados secuencialmente:

| # | Experiment | Runs | Propósito |
| --- | ----------- | ------ | ----------- |
| 01 | baselines | 4 | Líneas base (dummy, logistic, decision tree) |
| 02 | model-selection | 5 | RF, XGB, LGBM, CatBoost por defecto + dataset registry |
| 03 | hyperparameter-tuning | 2 parent + 80 child | Optuna para XGB y LGBM con nested runs |
| 04 | class-imbalance | 2 | Balanced weights vs custom weights |
| 05 | final-evaluation | 3 | Champion en test + mlflow.evaluate() + registry |

### Funcionalidades MLflow utilizadas

- **Nested runs**: Cada trial de Optuna como child run del study parent
- **Dataset lineage**: `mlflow.log_input()` para trazabilidad de datos
- **Model signatures**: Input/output schema para validación en producción
- **Model Registry**: Registro formal con alias "champion"
- **mlflow.evaluate()**: Evaluación automática con artefactos (confusion matrix, métricas)
- **Artifacts**: Confusion matrices, feature importance, classification reports, SHAP plots, Optuna plots

## Resultados

### Tabla Consolidada

| Fase | Modelo | macro_f1 | cohen_kappa | accuracy |
| ------ | -------- | ---------- | ------------- | ---------- |
| **Champion (test)** | XGB balanced weights | **0.3903** | **0.3319** | 0.7333 |
| Imbalance (val) | XGB balanced weights | 0.3831 | 0.2631 | 0.7962 |
| Imbalance (val) | XGB custom weights | 0.3648 | 0.2519 | 0.8660 |
| Baseline | Decision tree | 0.3169 | 0.1544 | 0.8472 |
| Tuning (val) | LGBM Optuna best | 0.2778 | 0.0769 | 0.8610 |
| Tuning (val) | XGB Optuna best | 0.2751 | 0.0842 | 0.8827 |
| Selection | CatBoost default | 0.2676 | 0.0707 | 0.8823 |
| Selection | XGB default | 0.2668 | 0.0658 | 0.8825 |
| Selection | RF default | 0.2668 | 0.0696 | 0.8816 |
| Baseline | Dummy most frequent | 0.2343 | 0.0000 | 0.8819 |

### Observaciones Clave

1. **El desbalance es el factor dominante**: El HP tuning sin balanced weights apenas mejora (~0.27→0.28). Con balanced weights salta a 0.38. El manejo de clases importa más que el algoritmo.

2. **Paradoja accuracy-vs-F1**: Los modelos sin balanceo muestran accuracy >0.88, pero macro_f1 ~0.27 (predicen casi todo como clase 1). El champion tiene accuracy "peor" (0.73) pero macro_f1 mucho mejor (0.39).

3. **XGB ≈ LGBM**: Ambos rinden prácticamente igual en tuning (0.2751 vs 0.2778). Se selecciona XGBoost porque es ligeramente mejor con balanced weights y tiene ecosistema más maduro con MLflow.

4. **Generalización temporal**: El champion generaliza bien al test 2024 (macro_f1 0.39 test vs 0.38 val), lo que indica que los patrones temporales y climáticos capturados son estables.

5. **Techo del modelo**: macro_f1=0.39 refleja una limitación real: predecir nivel de alerta sin variables epidemiológicas directas (para evitar leakage) es inherentemente difícil. Este techo está bien documentado y justificado.

## Interpretabilidad (SHAP)

SHAP TreeExplainer en muestra de 5,000 observaciones del test:

### Top Features por Importancia Global

1. **`pop_log`** — Domina para todas las clases, especialmente nivel 3 y 4. Ciudades más grandes tienen más alertas.
2. **`month_sin`/`month_cos`** — Estacionalidad fuerte. Captura el pico Mar-Abr.
3. **`tempmed_roll12w`** — Temperatura media rolling de 12 semanas. Proxy de temporada cálida.
4. **`umidmed_roll4w`** — Humedad media rolling. Contribuye a condiciones de transmisión.
5. **`temp_x_humid_lag4w`** — Interacción temperatura×humedad con lag.

### SHAP para Nivel 4 (Alerta Roja)

- `pop_log` alto → aumenta probabilidad de alerta roja
- `region_Sul` → reduce probabilidad (menor incidencia histórica)
- `tempmed_roll12w` alto → contribuye positivamente a alertas
- Patrón coherente con la epidemiología: ciudades grandes + clima cálido húmedo + fuera de la región Sur

## Módulo de Producción (src/models/)

### constants.py

- `LABEL_MAP` / `LABEL_MAP_INV`: Mapeo 1-4 ↔ 0-3
- `encode_labels()` / `decode_labels()`: Conversión transparente
- `compute_balanced_weights()`: Pesos inversamente proporcionales por clase
- `CLASSES`, `NEEDS_ZERO_INDEX`, `SEED`

### evaluation.py

- `compute_metrics()`: Todas las métricas con labels originales (1-4)
- `plot_confusion_matrix()`: Counts + normalized side by side
- `plot_feature_importance()`: Bar chart horizontal
- `save_classification_report()`: JSON serialization

### trainer.py

- `DengueModelTrainer`: Clase que encapsula train/predict/log con MLflow
  - `train_and_log()`: Entrena, evalúa, loguea params/metrics/artifacts/model
  - `predict()`: Con decode automático para 0-indexed models
  - `predict_proba()`: Probabilidades de predicción

## Champion Model

| Propiedad | Valor |
| ----------- | ------- |
| Algoritmo | XGBClassifier |
| HP Tuning | Optuna, 40 trials |
| Imbalance | Balanced sample weights |
| Train set | Train+Val 2010-2023 (4.07M rows) |
| Test set | 2024 (290K rows) |
| macro_f1 | 0.3903 |
| cohen_kappa | 0.3319 |
| roc_auc_ovr | 0.7993 |
| Registry | `dengue-alertlevel-classifier`, alias "champion" |
