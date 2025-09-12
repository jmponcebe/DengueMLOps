# Makefile para automatizar tareas del proyecto MLOps

.PHONY: help setup install clean data train serve test lint format docs

# Variables
PYTHON = C:/Users/ponce/CIDaeN/tfm-mlops/.venv/Scripts/python.exe
PIP = C:/Users/ponce/CIDaeN/tfm-mlops/.venv/Scripts/pip.exe
VENV_NAME = .venv

# Comandos de ayuda
help:
	@echo "🎯 Comandos disponibles para el proyecto MLOps Dengue:"
	@echo ""
	@echo "📦 Setup y Dependencies:"
	@echo "  make setup          - Configurar entorno virtual completo"
	@echo "  make install        - Instalar dependencias"
	@echo "  make requirements   - Generar requirements.txt actualizado"
	@echo ""
	@echo "🔧 Data Pipeline:"
	@echo "  make data           - Ejecutar pipeline completo de datos"
	@echo "  make clean-data     - Limpiar datos intermedios"
	@echo ""
	@echo "🤖 ML Pipeline:"
	@echo "  make train          - Entrenar modelos con MLflow"
	@echo "  make evaluate       - Evaluar modelos"
	@echo "  make serve          - Iniciar API de serving"
	@echo ""
	@echo "🌐 Applications:"
	@echo "  make streamlit      - Ejecutar app Streamlit"
	@echo "  make api            - Ejecutar API FastAPI"
	@echo "  make mlflow-ui      - Abrir interfaz MLflow"
	@echo ""
	@echo "🧪 Quality & Testing:"
	@echo "  make test           - Ejecutar tests"
	@echo "  make lint           - Linter código"
	@echo "  make format         - Formatear código"
	@echo "  make monitoring     - Generar reportes monitoring"
	@echo ""
	@echo "📚 Documentation:"
	@echo "  make docs           - Generar documentación"
	@echo "  make clean          - Limpiar archivos temporales"

# Setup del proyecto
setup: $(VENV_NAME)/Scripts/activate
	@echo "✅ Proyecto configurado correctamente"

$(VENV_NAME)/Scripts/activate:
	python -m venv $(VENV_NAME)
	$(PIP) install --upgrade pip
	$(PIP) install -r requirements.txt
	@echo "🔧 Entorno virtual creado y dependencias instaladas"

# Instalar dependencias
install:
	$(PIP) install -r requirements.txt
	$(PIP) install -e .
	@echo "📦 Dependencias instaladas"

# Generar requirements
requirements:
	$(PIP) freeze > requirements.txt
	@echo "📝 Requirements.txt actualizado"

# Pipeline de datos
data:
	$(PYTHON) src/data/make_dataset.py
	@echo "🔄 Pipeline de datos ejecutado"

clean-data:
	rm -rf data/interim/*
	rm -rf data/processed/*
	@echo "🧹 Datos intermedios limpiados"

# Entrenamiento de modelos
train:
	$(PYTHON) src/models/train_model.py
	@echo "🎯 Modelos entrenados"

evaluate:
	$(PYTHON) src/models/evaluate_model.py
	@echo "📊 Modelos evaluados"

# Serving
serve:
	$(PYTHON) app/api.py
	@echo "🚀 API de serving iniciada en http://localhost:8000"

# Aplicaciones
streamlit:
	$(PYTHON) -m streamlit run app/streamlit_app.py --server.port 8501
	@echo "🌐 App Streamlit ejecutándose en http://localhost:8501"

api:
	$(PYTHON) -m uvicorn app.api:app --host 0.0.0.0 --port 8000 --reload
	@echo "⚡ API FastAPI ejecutándose en http://localhost:8000"

mlflow-ui:
	$(PYTHON) -m mlflow ui --host 0.0.0.0 --port 5000
	@echo "🔬 MLflow UI ejecutándose en http://localhost:5000"

# Testing y calidad
test:
	$(PYTHON) -m pytest tests/ -v --cov=src
	@echo "🧪 Tests ejecutados"

lint:
	$(PYTHON) -m flake8 src/
	$(PYTHON) -m flake8 app/
	@echo "🔍 Linting completado"

format:
	$(PYTHON) -m black src/ app/ tests/
	$(PYTHON) -m isort src/ app/ tests/
	@echo "✨ Código formateado"

# Monitoreo
monitoring:
	$(PYTHON) src/monitoring/generate_reports.py
	@echo "📈 Reportes de monitoreo generados"

# Documentación
docs:
	$(PYTHON) -m mkdocs build
	@echo "📚 Documentación generada"

# Limpieza
clean:
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type d -name ".pytest_cache" -exec rm -rf {} +
	find . -type d -name "*.egg-info" -exec rm -rf {} +
	rm -rf build/
	rm -rf dist/
	@echo "🧹 Archivos temporales limpiados"

# Docker
docker-build:
	docker build -t dengue-prediction .
	@echo "🐳 Imagen Docker construida"

docker-run:
	docker run -p 8000:8000 -p 8501:8501 dengue-prediction
	@echo "🚢 Contenedor Docker ejecutándose"

# Pipelines completos
all: setup data train evaluate
	@echo "🎉 Pipeline completo ejecutado"

dev-setup: setup install
	$(PIP) install -r requirements-dev.txt
	pre-commit install
	@echo "🛠️ Entorno de desarrollo configurado"
