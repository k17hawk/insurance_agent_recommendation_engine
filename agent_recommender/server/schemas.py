from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from datetime import datetime


class LeadFeatures(BaseModel):
    """Client/Lead features"""
    quote_value: float = Field(..., description="Quote value for the lead")
    lead_difficulty: float = Field(..., description="Lead difficulty score")
    sophistication: float = Field(..., description="Lead sophistication level")
    patience_hours: float = Field(..., description="Lead patience in hours")
    digital_engagement_score: float = Field(..., description="Digital engagement score")
    tenure_years: float = Field(..., description="Customer tenure in years")
    log_quote_value: float = Field(..., description="Log transformed quote value")
    log_patience_hours: float = Field(..., description="Log transformed patience hours")
    month: int = Field(..., description="Month of lead creation")
    hour_of_day: int = Field(..., description="Hour of day lead was created")
    lead_dayofweek: int = Field(..., description="Day of week (0-6)")
    lead_quarter: int = Field(..., description="Quarter of year")
    is_weekend: int = Field(..., description="Is weekend (0/1)")
    insurance_type_enc: int = Field(..., description="Encoded insurance type")
    claims_risk: int = Field(..., description="Claims risk level (0-2)")
    multi_product_intent: int = Field(..., description="Multi-product intent (0/1)")
    insurance_type_missing: int = Field(..., description="Insurance type missing flag")
    language_missing: int = Field(..., description="Language missing flag")
    tenure_years_missing: int = Field(..., description="Tenure years missing flag")
    digital_engagement_score_missing: int = Field(..., description="Digital engagement score missing flag")


class BrokerFeatures(BaseModel):
    """Broker/Agent features"""
    skill_level: float = Field(..., description="Broker skill level")
    conversion_rate: float = Field(..., description="Broker conversion rate")
    csat_score: float = Field(..., description="Broker CSAT score")
    reliability: float = Field(..., description="Broker reliability score")
    efficiency: float = Field(..., description="Broker efficiency score")
    avg_response_time: float = Field(..., description="Average response time in hours")
    burnout_risk: float = Field(..., description="Burnout risk score")
    commission_rate: float = Field(..., description="Commission rate")
    cost_per_lead: float = Field(..., description="Cost per lead")
    utilization: float = Field(..., description="Broker utilization rate")
    ribo_licensed: int = Field(..., description="RIBO licensed (0/1)")
    is_new_broker: int = Field(..., description="Is new broker (0/1)")
    expertise_auto: int = Field(..., description="Auto expertise (0/1)")
    expertise_home: int = Field(..., description="Home expertise (0/1)")
    expertise_bundle: int = Field(..., description="Bundle expertise (0/1)")
    broker_quality_score: float = Field(..., description="Broker quality score")
    lang_Bilingual: int = Field(..., description="Bilingual (0/1)")
    lang_English: int = Field(..., description="English speaker (0/1)")
    lang_French: int = Field(..., description="French speaker (0/1)")


class InteractionFeatures(BaseModel):
    """Interaction features between lead and broker"""
    expertise_match: float = Field(..., description="Expertise match score")
    language_match: float = Field(..., description="Language match score")
    workload_ratio: float = Field(..., description="Broker workload ratio")
    quality_x_value: float = Field(..., description="Quality multiplied by value")
    position_bias: float = Field(..., description="Position bias")
    interaction_number: int = Field(..., description="Interaction number")
    responded: int = Field(..., description="Responded flag (0/1)")
    response_time_bucket_ord: int = Field(..., description="Response time bucket ordinal")
    log_response_time_hours: float = Field(..., description="Log response time")
    ribo_x_expertise: float = Field(..., description="RIBO times expertise")
    claims_x_skill: float = Field(..., description="Claims risk times skill")
    tenure_x_quality: float = Field(..., description="Tenure times quality")


class PredictionRequest(BaseModel):
    """Single prediction request"""
    lead_id: Optional[str] = Field(None, description="Lead identifier")
    broker_id: Optional[str] = Field(None, description="Broker identifier")
    client_features: LeadFeatures = Field(..., description="Client/lead features")
    broker_features: BrokerFeatures = Field(..., description="Broker features")
    interaction_features: InteractionFeatures = Field(..., description="Interaction features")


class BatchPredictionRequest(BaseModel):
    """Batch prediction request"""
    requests: List[PredictionRequest] = Field(..., description="List of prediction requests")


class PredictionResponse(BaseModel):
    """Single prediction response"""
    lead_id: Optional[str] = Field(None, description="Lead identifier")
    broker_id: Optional[str] = Field(None, description="Broker identifier")
    conversion_probability: float = Field(..., description="Probability of conversion")
    prediction: int = Field(..., description="Binary prediction (0/1) based on threshold")
    timestamp: datetime = Field(default_factory=datetime.now, description="Prediction timestamp")


class BatchPredictionResponse(BaseModel):
    """Batch prediction response"""
    predictions: List[PredictionResponse] = Field(..., description="List of predictions")
    total_requests: int = Field(..., description="Total number of requests")
    avg_probability: float = Field(..., description="Average conversion probability")
    timestamp: datetime = Field(default_factory=datetime.now, description="Prediction timestamp")


class HealthResponse(BaseModel):
    """Health check response"""
    status: str = Field(..., description="API status")
    model_loaded: bool = Field(..., description="Whether model is loaded")
    model_version: Optional[str] = Field(None, description="Model version")
    timestamp: datetime = Field(default_factory=datetime.now, description="Health check timestamp")


class ModelInfoResponse(BaseModel):
    """Model information response"""
    model_type: str = Field(..., description="Model type")
    model_version: str = Field(..., description="Model version")
    client_features: List[str] = Field(..., description="Client features used")
    broker_features: List[str] = Field(..., description="Broker features used")
    interaction_features: List[str] = Field(..., description="Interaction features used")
    threshold: float = Field(..., description="Classification threshold")
    embedding_dim: int = Field(..., description="Embedding dimension")
    hidden_dim: int = Field(..., description="Hidden dimension")
    timestamp: datetime = Field(default_factory=datetime.now, description="Response timestamp")