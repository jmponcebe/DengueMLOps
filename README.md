# Predicción de Dengue en Brasil - Proyecto MLOps para TFM

## 📋 Descripción del Proyecto

Este proyecto implementa un pipeline completo de MLOps para la predicción del riesgo de dengue por municipio en Brasil, desarrollado como parte de un Trabajo de Fin de Máster (TFM). El sistema utiliza datos históricos de dengue (2010-2025) y variables climáticas para generar predicciones de riesgo a nivel municipal.

## 🎯 Objetivos

- **Pipeline MLOps completo**: Desde ingesta de datos hasta monitoreo en producción
- **Predicción de dengue**: Modelo que considera ciclos epidemiológicos de 5-6 años
- **Visualización interactiva**: App Streamlit con mapa de Brasil y predicciones
- **Monitoreo de calidad**: Detección de data drift con Evidently
- **Experimentación**: Tracking completo con MLflow

## 🏗️ Arquitectura del Proyecto

```text
tfm-mlops/
├── data/                   # Gestión de datos
│   ├── raw/               # Datos originales (dengue, clima)
│   ├── interim/           # Datos en procesamiento
│   ├── processed/         # Datos finales para modelado
│   └── external/          # Datos de terceros (shapefiles, etc.)
├── notebooks/             # Jupyter notebooks para exploración
├── src/                   # Código de producción
│   ├── data/             # Ingesta y procesamiento de datos
│   ├── features/         # Feature engineering
│   ├── models/           # Entrenamiento y evaluación
│   └── monitoring/       # Monitoreo y alertas
├── app/                   # Aplicación Streamlit y API
├── models/               # Modelos entrenados y artefactos
├── monitoring/           # Reportes de monitoreo
├── docs/                 # Documentación del proyecto
├── tests/                # Tests unitarios e integración
└── configs/              # Archivos de configuración
```

## 🚀 Componentes Principales

### 1. **Ingesta de Datos**

- API mosqlimate para datos históricos de dengue
- Datos climáticos (temperatura, precipitación, humedad)
- Procesamiento ETL automatizado

### 2. **Feature Engineering**

- Detección de patrones cíclicos (5-6 años)
- Variables epidemiológicas derivadas
- Agregaciones temporales y espaciales

### 3. **Modelado y Experimentación**

- Múltiples algoritmos (XGBoost, Random Forest, etc.)
- Tracking completo con MLflow
- Validación temporal y espacial

### 4. **Despliegue**

- **App Streamlit**: Mapa interactivo de Brasil con predicciones
- **API REST**: Endpoint para predicciones en tiempo real
- **Containerización**: Docker para reproducibilidad

### 5. **Monitoreo**

- **Data Drift**: Evidently para detectar cambios en distribución
- **Model Performance**: Métricas de calidad en producción
- **Alertas**: Sistema de notificaciones automáticas

## 📊 Datos Utilizados

### Fuentes de Datos

- **Dengue**: API mosqlimate (2010-2025)
- **Clima**: Variables meteorológicas por municipio
- **Geográficos**: Shapefiles de municipios brasileños

### Variables Principales

- Casos confirmados de dengue
- Temperatura media, máxima, mínima
- Precipitación acumulada
- Humedad relativa
- Índice de vegetación (NDVI)

## 🛠️ Tecnologías

### Desarrollo y Experimentación

- **Python**: Lenguaje principal
- **Jupyter**: Notebooks para exploración
- **Pandas/NumPy**: Manipulación de datos
- **Scikit-learn**: Modelado ML

### MLOps y Deployment

- **MLflow**: Experiment tracking y model registry
- **Streamlit**: Aplicación web interactiva
- **FastAPI**: API REST para serving
- **Docker**: Containerización

### Monitoreo y Calidad

- **Evidently**: Data drift y model monitoring
- **Pytest**: Testing automatizado
- **GitHub Actions**: CI/CD pipeline

### Visualización

- **Plotly**: Gráficos interactivos
- **Folium**: Mapas dinámicos
- **Seaborn/Matplotlib**: Análisis exploratorio

## 📈 Roadmap de Desarrollo

### Fase 1: Exploración y Análisis (Días 1-2)

- [x] Setup del proyecto y estructura
- [ ] Análisis exploratorio de datos
- [ ] Identificación de patrones cíclicos
- [ ] Documentación inicial

### Fase 2: Feature Engineering y Modelado (Días 3-4)

- [ ] Feature engineering avanzado
- [ ] Experimentación con modelos
- [ ] Configuración MLflow
- [ ] Validación y selección de modelo

### Fase 3: Desarrollo de Aplicación (Días 5-6)

- [ ] Desarrollo app Streamlit
- [ ] API REST para serving
- [ ] Integración con mapas de Brasil
- [ ] Testing e2e

### Fase 4: Monitoreo y Producción (Días 7-8)

- [ ] Implementación monitoring
- [ ] Setup Evidently
- [ ] Containerización
- [ ] CI/CD pipeline (opcional)

## 📚 Para el TFM

Este proyecto está diseñado para generar contenido rico para la memoria del TFM:

- **Metodología**: Pipeline MLOps completo documentado
- **Experimentación**: Múltiples modelos y métricas en MLflow
- **Innovación**: Enfoque en ciclos epidemiológicos del dengue
- **Impacto**: Aplicación práctica para salud pública
- **Calidad**: Monitoreo y testing automatizado

## 🚦 Estado Actual

🔄 **En desarrollo activo** - Configuración inicial completada

## 📞 Contacto

Proyecto desarrollado para TFM en MLOps - Predicción de Dengue en Brasil

---

***Última actualización: Septiembre 2025***
