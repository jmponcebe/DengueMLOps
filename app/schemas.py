"""
Pydantic schemas para la API de predicción de dengue.
Validación de inputs y serialización de responses.
"""

from pydantic import BaseModel, Field
from typing import List, Optional, Dict
from enum import IntEnum


class AlertLevel(IntEnum):
    """InfoDengue epidemiological alert levels."""
    GREEN = 1
    YELLOW = 2
    ORANGE = 3
    RED = 4


# 15 features de producción (sin leakage)
FEATURE_NAMES = [
    "month_sin", "month_cos", "se_sin", "se_cos",
    "is_peak_season", "quarter",
    "tempmed_lag8w", "tempmed_roll12w", "umidmed_roll4w", "temp_x_humid_lag4w",
    "pop_log",
    "region_Nordeste", "region_Centro-Oeste", "region_Sudeste", "region_Sul",
]

ALERT_LABELS = {1: "Green", 2: "Yellow", 3: "Orange", 4: "Red"}
ALERT_COLORS = {1: "#00cc00", 2: "#ffcc00", 3: "#ff6600", 4: "#cc0000"}


class PredictionInput(BaseModel):
    """Input for individual prediction."""
    month_sin: float = Field(..., ge=-1, le=1, description="Month sine (cyclic encoding)")
    month_cos: float = Field(..., ge=-1, le=1, description="Month cosine")
    se_sin: float = Field(..., ge=-1, le=1, description="Epidemiological week sine")
    se_cos: float = Field(..., ge=-1, le=1, description="Epidemiological week cosine")
    is_peak_season: int = Field(..., ge=0, le=1, description="1 if peak season (Jan-Apr)")
    quarter: int = Field(..., ge=1, le=4, description="Year quarter")

    # Climate — can be null if no data with sufficient lag
    tempmed_lag8w: Optional[float] = Field(None, description="Mean temperature 8-week lag")
    tempmed_roll12w: Optional[float] = Field(None, description="Temperature 12-week rolling mean")
    umidmed_roll4w: Optional[float] = Field(None, description="Humidity 4-week rolling mean")
    temp_x_humid_lag4w: Optional[float] = Field(None, description="Temp*humidity interaction 4w lag")

    # Geographic
    pop_log: float = Field(..., gt=0, description="Log of municipality population")
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
    """Prediction result."""
    nivel: int = Field(..., ge=1, le=4, description="Predicted alert level (1-4)")
    label: str = Field(..., description="Level label (Green/Yellow/Orange/Red)")
    color: str = Field(..., description="Alert level hex color")
    probabilities: Dict[str, float] = Field(
        ..., description="Per-class probability"
    )


class PredictionResponse(BaseModel):
    """Individual prediction response."""
    prediction: PredictionResult
    model_version: str
    features_used: int


class BatchInput(BaseModel):
    """Batch prediction input."""
    instances: List[PredictionInput] = Field(
        ..., min_length=1, max_length=1000,
        description="List of instances to predict (max 1000)"
    )


class BatchResponse(BaseModel):
    """Batch prediction response."""
    predictions: List[PredictionResult]
    count: int
    model_version: str


class ModelInfo(BaseModel):
    """Production model information."""
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
