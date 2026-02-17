"""
Pydantic schemas para la API de predicción de dengue.
Validación de inputs y serialización de responses.
"""

from pydantic import BaseModel, Field
from typing import List, Optional, Dict
from enum import IntEnum


class AlertLevel(IntEnum):
    """Niveles de alerta epidemiológica del InfoDengue."""
    VERDE = 1
    AMARELO = 2
    LARANJA = 3
    VERMELHO = 4


# 15 features de producción (sin leakage)
FEATURE_NAMES = [
    "month_sin", "month_cos", "se_sin", "se_cos",
    "is_peak_season", "quarter",
    "tempmed_lag8w", "tempmed_roll12w", "umidmed_roll4w", "temp_x_humid_lag4w",
    "pop_log",
    "region_Nordeste", "region_Centro-Oeste", "region_Sudeste", "region_Sul",
]

ALERT_LABELS = {1: "Verde", 2: "Amarelo", 3: "Laranja", 4: "Vermelho"}
ALERT_COLORS = {1: "#00cc00", 2: "#ffcc00", 3: "#ff6600", 4: "#cc0000"}


class PredictionInput(BaseModel):
    """Input para predicción individual."""
    month_sin: float = Field(..., ge=-1, le=1, description="Seno del mes (encoding cíclico)")
    month_cos: float = Field(..., ge=-1, le=1, description="Coseno del mes")
    se_sin: float = Field(..., ge=-1, le=1, description="Seno de la semana epidemiológica")
    se_cos: float = Field(..., ge=-1, le=1, description="Coseno de la semana epidemiológica")
    is_peak_season: int = Field(..., ge=0, le=1, description="1 si es temporada pico (ene-abr)")
    quarter: int = Field(..., ge=1, le=4, description="Trimestre del año")

    # Climáticas — pueden ser null si no hay datos con lag suficiente
    tempmed_lag8w: Optional[float] = Field(None, description="Temperatura media lag 8 semanas")
    tempmed_roll12w: Optional[float] = Field(None, description="Media móvil temperatura 12 semanas")
    umidmed_roll4w: Optional[float] = Field(None, description="Media móvil humedad 4 semanas")
    temp_x_humid_lag4w: Optional[float] = Field(None, description="Interacción temp*humedad lag 4w")

    # Geográficas
    pop_log: float = Field(..., gt=0, description="Log de la población del municipio")
    region_Nordeste: int = Field(0, ge=0, le=1)
    region_Centro_Oeste: int = Field(0, ge=0, le=1, alias="region_Centro-Oeste")
    region_Sudeste: int = Field(0, ge=0, le=1)
    region_Sul: int = Field(0, ge=0, le=1)

    model_config = {"populate_by_name": True}

    def to_feature_dict(self) -> dict:
        """Convierte a dict con los nombres exactos de features del modelo."""
        return {
            "month_sin": self.month_sin,
            "month_cos": self.month_cos,
            "se_sin": self.se_sin,
            "se_cos": self.se_cos,
            "is_peak_season": self.is_peak_season,
            "quarter": self.quarter,
            "tempmed_lag8w": self.tempmed_lag8w,
            "tempmed_roll12w": self.tempmed_roll12w,
            "umidmed_roll4w": self.umidmed_roll4w,
            "temp_x_humid_lag4w": self.temp_x_humid_lag4w,
            "pop_log": self.pop_log,
            "region_Nordeste": self.region_Nordeste,
            "region_Centro-Oeste": self.region_Centro_Oeste,
            "region_Sudeste": self.region_Sudeste,
            "region_Sul": self.region_Sul,
        }


class PredictionResult(BaseModel):
    """Resultado de una predicción."""
    nivel: int = Field(..., ge=1, le=4, description="Nivel de alerta predicho (1-4)")
    label: str = Field(..., description="Etiqueta del nivel (Verde/Amarelo/Laranja/Vermelho)")
    color: str = Field(..., description="Color hex del nivel de alerta")
    probabilities: Dict[str, float] = Field(
        ..., description="Probabilidad por clase"
    )


class PredictionResponse(BaseModel):
    """Response de predicción individual."""
    prediction: PredictionResult
    model_version: str
    features_used: int


class BatchInput(BaseModel):
    """Input para predicción por lotes."""
    instances: List[PredictionInput] = Field(
        ..., min_length=1, max_length=1000,
        description="Lista de instancias a predecir (máx 1000)"
    )


class BatchResponse(BaseModel):
    """Response de predicción por lotes."""
    predictions: List[PredictionResult]
    count: int
    model_version: str


class ModelInfo(BaseModel):
    """Información del modelo en producción."""
    name: str
    version: str
    alias: str
    algorithm: str
    n_features: int
    features: List[str]
    target: str
    classes: Dict[int, str]
    metrics: Dict[str, float]
    training_period: str
    description: str


class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    model_loaded: bool
    version: str
