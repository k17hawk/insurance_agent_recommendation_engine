from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
from datetime import datetime
from enum import Enum

class InsuranceType(str, Enum):
    auto = "Auto"
    home = "Home"
    bundle = "Bundle"

class Language(str, Enum):
    english = "English"
    french = "French"
    other = "Other"

class ClaimsRisk(str, Enum):
    none = "None"
    minor = "Minor"
    major = "Major"

class PredictionRequest(BaseModel):
    lead_id: str
    broker_id: str
    # Simple frontend fields
    insurance_type: InsuranceType
    language: Language
    claims_risk: ClaimsRisk
    quote_value: float
    lead_difficulty: float = Field(ge=0, le=1)
    sophistication: float = Field(ge=0, le=1)
    patience_hours: float
    digital_engagement_score: float
    tenure_years: float
    month: int = Field(ge=1, le=12)
    hour_of_day: int = Field(ge=0, le=23)
    lead_dayofweek: int = Field(ge=0, le=6)
    lead_quarter: int = Field(ge=1, le=4)
    is_weekend: bool
    multi_product_intent: bool
    # Optional: override A/B test version
    model_version: Optional[str] = None   # e.g., "v2.1.0" or "latest"

class PredictionResponse(BaseModel):
    lead_id: str
    broker_id: str
    conversion_probability: float
    prediction: int   # 0 or 1 after threshold
    model_version: str
    inference_time_ms: Optional[float] = Field(None, description="Inference time in milliseconds")
    timestamp: datetime = Field(default_factory=datetime.now, description="Prediction timestamp")

    

class ConversionLog(BaseModel):
    lead_id: str
    broker_id: str
    converted: bool

class ABTestConfig(BaseModel):
    experiment_id: str
    default_model: str
    variants: List[Dict[str, Any]]
    enabled: bool

class ABTestResults(BaseModel):
    experiment_id: str
    variants: List[Dict[str, Any]]
    lift: Optional[str] = None
    p_value: Optional[float] = None

class HealthResponse(BaseModel):
    status: str
    models_loaded: List[str]
    default_model: str

