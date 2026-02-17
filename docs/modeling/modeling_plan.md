# Plan de Modelado y Experimentación con MLflow

## Visión General

La fase de modelado sigue una progresión natural de experimentación que cualquier ingeniero ML seguiría: desde baselines simples hasta modelos optimizados con técnicas avanzadas. MLflow no es solo una herramienta auxiliar, sino el **eje central** que documenta cada decisión, permitiendo reproducibilidad total y trazabilidad de extremo a extremo.

---

## 1. Estructura de Experimentos en MLflow

### Jerarquía de Experiments

Usaremos **múltiples experiments** en MLflow para organizar las fases lógicas del proceso. Esto genera una vista limpia en la UI y facilita la comparación entre fases.

```text
dengue-prediction/
├── 01-baselines              ← Modelos de referencia (DummyClassifier, logistic)
├── 02-model-selection         ← Comparación de algoritmos con config por defecto
├── 03-hyperparameter-tuning   ← Optuna + MLflow para los 2-3 mejores modelos
├── 04-class-imbalance         ← Técnicas para el desbalanceo (93% clase 1)
└── 05-final-evaluation        ← Modelo champion en test set
```

### Naming Convention para Runs

```text
{algorithm}_{strategy}_{version}
```

Ejemplos:

- `dummy_stratified_v1`
- `xgb_default_v1`
- `lgbm_optuna_v3`
- `xgb_smote_focal_v2`

---

## 2. Fases de Experimentación

### Fase 1: Baselines (Experiment: `01-baselines`)

**Objetivo**: Establecer referencias mínimas para evaluar mejora real.

| Run | Modelo | Descripción |
| --- | --- | --- |
| `dummy_most_frequent` | DummyClassifier(strategy='most_frequent') | Predice siempre clase 1 |
| `dummy_stratified` | DummyClassifier(strategy='stratified') | Aleatorio proporcional |
| `logistic_default` | LogisticRegression | Línea base con modelo simple |
| `decision_tree_default` | DecisionTreeClassifier | Árbol simple para interpretabilidad |

**Qué loguear**:

- Parámetros: estrategia, random_seed
- Métricas: accuracy, macro_f1, weighted_f1, cohen_kappa, log_loss
- Artefactos: classification_report.txt, confusion_matrix.png

**Por qué importa**: Con 93.1% de clase 1, el dummy ya logra ~0.87 accuracy. Necesitamos macro_f1 y cohen_kappa para medir capacidad real de discriminación entre las 4 clases.

---

### Fase 2: Selección de Modelos (Experiment: `02-model-selection`)

**Objetivo**: Comparar familias de algoritmos con configuración por defecto.

| Run | Modelo | Justificación |
| --- | --- | --- |
| `rf_default` | RandomForestClassifier | Ensemble robusto, buen baseline |
| `xgb_default` | XGBClassifier | Gradient boosting, manejo nativo de desbalanceo |
| `lgbm_default` | LGBMClassifier | Rápido, eficiente en memoria, manejo de categorías |
| `catboost_default` | CatBoostClassifier | Bueno con categorías, robusto a overfitting |

**Qué loguear**:

- Parámetros: todos los hiperparámetros del modelo
- Métricas: accuracy, macro_f1, weighted_f1, f1 por clase, cohen_kappa, log_loss, roc_auc_ovr
- Artefactos: confusion_matrix.png, classification_report.json, feature_importance.png, roc_curves.png
- Tags: `algorithm_family`, `handles_imbalance`, `training_time`
- **MLflow Dataset**: Registrar el dataset de train/val con `mlflow.log_input()`

**Funcionalidad avanzada**: Usar `mlflow.log_input()` para registrar los datasets de entrada y tener trazabilidad dato→modelo.

---

### Fase 3: Optimización de Hiperparámetros (Experiment: `03-hyperparameter-tuning`)

**Objetivo**: Tuning exhaustivo de los 2-3 mejores modelos de la Fase 2, usando Optuna integrado con MLflow.

**Integración Optuna-MLflow**:

```python
# Cada trial de Optuna = 1 run en MLflow (nested runs)
with mlflow.start_run(run_name="xgb_optuna_study"):
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=100, callbacks=[mlflow_callback])
```

**Espacios de búsqueda** (ejemplo XGBoost):

```python
{
    "n_estimators": (100, 1000),
    "max_depth": (3, 12),
    "learning_rate": (0.01, 0.3),
    "subsample": (0.6, 1.0),
    "colsample_bytree": (0.6, 1.0),
    "min_child_weight": (1, 10),
    "scale_pos_weight": calculado dinámicamente,
    "reg_alpha": (1e-8, 10.0),
    "reg_lambda": (1e-8, 10.0),
}
```

**Qué loguear (por trial)**:

- Parámetros: hiperparámetros del trial
- Métricas: objective_value (macro_f1), accuracy, weighted_f1
- Tags: `trial_number`, `pruned` (si Optuna prunea el trial)

**Qué loguear (run padre)**:

- Artefactos: optuna_optimization_history.png, param_importances.png, contour_plots.png
- Mejores parámetros como JSON
- Estudio completo serializado

**Funcionalidad avanzada**:

- **Nested Runs**: Cada trial como child run del estudio principal
- **MLflowCallback de Optuna**: Integración nativa `optuna.integration.MLflowCallback`
- **Pruning**: MedianPruner para detener trials poco prometedores

---

### Fase 4: Manejo del Desbalanceo (Experiment: `04-class-imbalance`)

**Objetivo**: Evaluar estrategias específicas para el desbalanceo severo (93.1% nivel 1).

| Estrategia | Descripción |
| --- | --- |
| `class_weight_balanced` | Pesos inversamente proporcionales a frecuencia |
| `sample_weight_custom` | Pesos manuales enfatizando clases 3 y 4 (alerta) |
| `smote_oversampling` | SMOTE para clases minoritarias |
| `focal_loss` | Loss function que penaliza más samples "fáciles" |
| `threshold_tuning` | Ajustar umbrales de decisión post-hoc |

**Qué loguear**:

- Métricas adicionales: recall por clase (especialmente clases 3 y 4), precision-recall curves
- Artefactos: distribución de predicciones, calibration curve
- Tags: `imbalance_strategy`, `minority_class_recall`

---

### Fase 5: Evaluación Final (Experiment: `05-final-evaluation`)

**Objetivo**: Evaluación rigurosa del modelo champion en test set (2024).

**Proceso**:

1. Seleccionar el mejor modelo de las fases 3-4
2. Re-entrenar en train+val (2010-2023)
3. Evaluar en test (2024) — **una sola vez**
4. Registrar en Model Registry

**Qué loguear**:

- Métricas completas en test
- Artefactos: informe completo de evaluación, SHAP plots, predicciones vs real por estado
- **Model Signature**: Input/output schema con `mlflow.models.infer_signature()`
- **Input Example**: Ejemplo real para la UI de MLflow

---

## 3. Funcionalidades Avanzadas de MLflow

### 3.1 Model Registry y Lifecycle Management

```python
# Registrar modelo con alias
model_uri = f"runs:/{run_id}/model"
mv = mlflow.register_model(model_uri, "dengue-alertlevel-classifier")

# Asignar alias (MLflow 2.x)
client = mlflow.tracking.MlflowClient()
client.set_registered_model_alias("dengue-alertlevel-classifier", "champion", mv.version)
```

**Ciclo de vida del modelo**:

- Version 1: Baseline (LogisticRegression) → alias `baseline`
- Version 2: XGBoost default → sin alias
- Version 3: XGBoost tuned → alias `challenger`
- Version 4: XGBoost tuned + SMOTE → alias `champion`

### 3.2 Custom Metrics (Evaluación con mlflow.evaluate)

```python
# Evaluar con métricas estándar + custom
results = mlflow.evaluate(
    model=model_uri,
    data=eval_data,
    targets="nivel",
    model_type="classifier",
    extra_metrics=[
        make_metric(eval_fn=cohen_kappa_score_fn, name="cohen_kappa", greater_is_better=True),
        make_metric(eval_fn=minority_recall_fn, name="minority_recall", greater_is_better=True),
    ],
    evaluators="default",
)
```

### 3.3 Model Signature + Input Example

```python
from mlflow.models import infer_signature

signature = infer_signature(X_train, model.predict(X_train))
input_example = X_val.iloc[:3].to_dict(orient="records")

mlflow.sklearn.log_model(
    model, "model",
    signature=signature,
    input_example=input_example,
    registered_model_name="dengue-alertlevel-classifier"
)
```

### 3.4 MLflow Datasets (Trazabilidad de Datos)

```python
import mlflow.data

# Registrar dataset de entrenamiento
train_dataset = mlflow.data.from_pandas(
    df_train, source="data/processed/production_features.parquet",
    name="dengue_train", targets="nivel"
)
mlflow.log_input(train_dataset, context="training")

# Registrar dataset de validación
val_dataset = mlflow.data.from_pandas(
    df_val, source="data/processed/production_features.parquet",
    name="dengue_val", targets="nivel"
)
mlflow.log_input(val_dataset, context="validation")
```

### 3.5 Artefactos Ricos

Para cada run relevante, loguear:

- **Plots**: confusion matrix, ROC curves, feature importance, SHAP summary
- **Reports**: classification report como JSON y como texto
- **Data snapshots**: predicciones en validación para análisis posterior
- **Config**: diccionario completo de configuración del experimento

---

## 4. Capturas de Pantalla Recomendadas para la Memoria

### Imprescindibles (incluir seguro)

| # | Pantalla | Qué muestra | Dónde en la memoria |
| --- | --- | --- | --- |
| 1 | **Experiments List** | Vista general de los 5 experiments con nº de runs | Metodología - diseño experimental |
| 2 | **Runs Table con métricas** | Tabla comparativa de runs ordenados por macro_f1 | Resultados - selección de modelos |
| 3 | **Run Detail - Parámetros** | Detalle de un run mostrando всех hiperparámetros | Resultados - configuración del modelo |
| 4 | **Run Detail - Métricas** | Métricas del mejor modelo | Resultados - evaluación |
| 5 | **Run Detail - Artefactos** | Confusion matrix, feature importance desde la UI | Resultados - análisis del modelo |
| 6 | **Chart: Compare Runs** | Gráfico de barras comparando modelos (metric plot) | Resultados - comparación |
| 7 | **Model Registry** | Lista de versiones con aliases champion/challenger | Metodología - MLOps lifecycle |
| 8 | **Model Version Detail** | Detalle de versión con signature, input example, source run | Despliegue - modelo en producción |

### Muy recomendables (alto valor)

| # | Pantalla | Qué muestra | Dónde en la memoria |
| --- | --- | --- | --- |
| 9 | **Nested Runs (Optuna)** | Run padre con nested trials colapsables | Resultados - optimización HP |
| 10 | **Optuna plots como artefactos** | Optimization history y param importances en la UI | Resultados - optimización HP |
| 11 | **Dataset Lineage** | Tab "Datasets" mostrando train/val datasets vinculados al run | Metodología - trazabilidad |
| 12 | **Compare 2-3 Runs** | Side-by-side de modelos con diff de parámetros | Resultados - análisis comparativo |
| 13 | **Model Signature** | Input/output schema en la UI de MLflow | Despliegue - contrato del modelo |
| 14 | **Evaluation Artifacts** | Tab de evaluación con métricas y tablas autogeneradas | Resultados - evaluación avanzada |

### Bonus (si caben)

| # | Pantalla | Qué muestra |
| --- | --- | --- |
| 15 | **SHAP plot como artefacto** | Interpretabilidad del modelo desde MLflow UI |
| 16 | **Runs search/filter** | Búsqueda avanzada con filtros (`metrics.macro_f1 > 0.5`) |
| 17 | **Tags en runs** | Tags como `imbalance_strategy=smote` para organización |

---

## 5. Estructura del Notebook 03-modeling.ipynb

1. Setup y configuración
   - Imports, MLflow setup, carga de datos
   - Split temporal: train (2010-2021), val (2022-2023), test (2024)

2. Exploración pre-modelado
   - Distribución del target en cada split
   - Verificación de features y shapes

3. Baselines (Experiment: 01-baselines)
   - DummyClassifier (most_frequent, stratified)
   - LogisticRegression, DecisionTree
   - Función helper para logging uniforme

4. Selección de modelos (Experiment: 02-model-selection)
   - RF, XGBoost, LightGBM, CatBoost con defaults
   - Comparación visual
   - Dataset logging con mlflow.log_input()

5. Optimización con Optuna (Experiment: 03-hyperparameter-tuning)
   - Nested runs: cada trial como child run
   - Optuna MLflowCallback
   - Visualización de optimization history

6. Manejo de desbalanceo (Experiment: 04-class-imbalance)
   - class_weight, sample_weight, SMOTE
   - Comparación de recall en clases minoritarias

7. Evaluación final (Experiment: 05-final-evaluation)
   - Mejor modelo en test set
   - mlflow.evaluate() con custom metrics
   - Model Registry: registrar con alias "champion"
   - Model signature + input example
   - SHAP para interpretabilidad

8. Resumen y conclusiones
   - Tabla final de resultados
   - Selección justificada del modelo champion

---

## 6. Métricas Clave

Dado el desbalanceo severo, las métricas prioritarias son:

| Métrica | Por qué | Uso |
| --- | --- | --- |
| **Macro F1** | Promedio no ponderado entre clases; penaliza fallar en clases raras | **Métrica principal de optimización** |
| **Weighted F1** | Ponderado por soporte; visión general | Métrica secundaria |
| **Cohen's Kappa** | Mide acuerdo más allá del azar; ideal para clases desbalanceadas | Métrica complementaria |
| **Accuracy** | Solo referencial, dominada por clase mayoritaria | Comparación con baseline |
| **Recall clase 3-4** | Detectar alertas altas es crítico en salud pública | Evaluación de impacto |
| **Log Loss** | Calidad de las probabilidades predichas | Calibración |

---

## 7. Timeline Estimado

| Fase | Duración estimada | Dependencias |
| --- | --- | --- |
| 1. Baselines | 1-2 horas | Datos procesados |
| 2. Model Selection | 2-3 horas | Fase 1 |
| 3. HP Tuning (Optuna) | 3-4 horas (compute) | Fase 2 |
| 4. Class Imbalance | 2-3 horas | Fase 3 |
| 5. Final Evaluation | 1-2 horas | Fase 4 |
| **Total** | **~10-14 horas** | |

---

## 8. Notas para la Memoria

### Narrativa del flujo experimental

La memoria debería contar la historia así:

1. "Empezamos con baselines para entender el problema" → accuracy de 87% no significa nada
2. "Comparamos familias de modelos" → gradient boosting domina
3. "Optimizamos hiperparámetros" → mejora de X% en macro_f1
4. "Atacamos el desbalanceo" → mejora en recall de clases críticas
5. "Evaluamos rigurosamente en datos no vistos" → resultados finales

### Puntos de diferenciación MLOps

Para el TFM, enfatizar:

- **Trazabilidad completa**: De dato a modelo desplegado, todo registrado
- **Reproducibilidad**: Cualquier run se puede recrear con sus parámetros
- **Comparabilidad**: Decisiones basadas en métricas, no intuición
- **Lifecycle management**: Model Registry con aliases, no archivos sueltos
- **Evaluación avanzada**: `mlflow.evaluate()` con custom metrics
