# models/

Este directorio es un placeholder del scaffolding inicial del proyecto.

Los modelos entrenados se gestionan a través de **MLflow**:

- **MLflow Tracking**: `mlruns/` (metadatos de experimentos)
- **MLflow Artifacts**: `mlflow-artifacts/models/` (artefactos de modelos serializados)
- **Producción (AWS)**: S3 bucket `dengue-mlops-data-{ACCOUNT_ID}/models/champion/`
- **Model Registry**: Modelo registrado como `dengue-alertlevel-classifier` con alias `champion`

Para cargar el modelo champion en código:

```python
import mlflow
model = mlflow.pyfunc.load_model("models:/dengue-alertlevel-classifier@champion")
```
