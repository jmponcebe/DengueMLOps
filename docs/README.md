# Documentación Técnica - TFM MLOps Dengue

Esta carpeta contiene documentación técnica detallada, análisis metodológico y contenido académico para el desarrollo del TFM (Trabajo de Fin de Máster).

## Estructura

- `exploratory_analysis/` - Análisis exploratorio detallado
  - `production_constraints.md` - Restricciones temporales para predicción realista
- `feature_engineering/` - Documentación técnica de ingeniería de features
  - `feature_engineering_technical.md` - Pipeline completo, justificaciones y resultados
- `modeling/` - Documentación de la fase de modelado
  - `modeling_plan.md` - Plan de experimentación y fases
  - `modeling_technical.md` - Resultados, decisiones y análisis detallado
- `deployment/` - Documentación de despliegue y monitoreo
  - `architecture.md` - Arquitectura del sistema, diagramas de componentes
  - `aws_deployment.md` - Guía paso a paso para despliegue en AWS (EC2, ECS/Fargate)
  - `deployment_technical.md` - Contenido técnico para el capítulo de deployment del TFM
- `academic_contributions/` - Contribuciones académicas y teóricas
  - `impact_and_contributions.md` - Impacto y contribuciones del proyecto
- `api_client_usage.md` - Uso del cliente de la API mosqlimate
- `api_authentication.md` - Autenticación con la API

> **Quick start**: Para obtener el modelo champion y los datos necesarios, ejecutar
> `python scripts/setup_data.py` (modelo + GeoJSON) o `python scripts/setup_data.py --latest`
> (modelo + GeoJSON + datos del año actual). El modelo se descarga desde
> [GitHub Releases v1.0.0](https://github.com/jmponcebe/DengueMLOps/releases/tag/v1.0.0).

## Propósito

Los notebooks mantienen el código conciso y práctico, mientras que esta documentación proporciona:

- **Análisis profundo**: Metodología y fundamentos teóricos
- **Contexto académico**: Literatura científica y contribuciones
- **Validación técnica**: Restricciones epidemiológicas y temporales
- **Preparación TFM**: Material de base para la redacción de la memoria

> **Nota**: La memoria final del TFM se encuentra en `docs/memoria/` (LaTeX, plantilla CIDaeN UCLM). El resto de esta carpeta sirvió como material preparatorio y referencia técnica durante el desarrollo.

## Estado Actual

| Fase | Notebook | Documentación | Módulo src/ | Memoria |
| --- | --- | --- | --- | --- |
| EDA | `01-exploratory-data-analysis.ipynb` | `exploratory_analysis/`, `academic_contributions/` | `src/data/` | Cap 2 |
| Feature Engineering | `02-feature-engineering.ipynb` | `feature_engineering/` | `src/features/` | Cap 2 |
| Modelado | `03-modeling.ipynb` | `modeling/` | `src/models/` | Cap 2-3 |
| Deployment | - | `deployment/` | `app/`, `src/monitoring/` | Cap 4-5 |
| TFM (Memoria) | - | - | - | `docs/memoria/` (6 caps + apéndice) |
