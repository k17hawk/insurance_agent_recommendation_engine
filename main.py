from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from datetime import datetime
import uvicorn
import hashlib
from collections import defaultdict

# ------------------------------
# 1. Pydantic Schemas (Requests & Responses)
# ------------------------------

class ClientFeatures(BaseModel):
    quote_value: float
    lead_difficulty: float
    sophistication: float
    patience_hours: float
    digital_engagement_score: float
    tenure_years: float
    log_quote_value: float
    log_patience_hours: float
    month: int
    hour_of_day: int
    lead_dayofweek: int
    lead_quarter: int
    is_weekend: int
    insurance_type_enc: int
    claims_risk: int
    multi_product_intent: int
    insurance_type_missing: int
    language_missing: int
    tenure_years_missing: int
    digital_engagement_score_missing: int

class BrokerFeatures(BaseModel):
    skill_level: float
    conversion_rate: float
    csat_score: float
    reliability: float
    efficiency: float
    avg_response_time: float
    burnout_risk: float
    commission_rate: float
    cost_per_lead: float
    utilization: float
    ribo_licensed: int
    is_new_broker: int
    expertise_auto: int
    expertise_home: int
    expertise_bundle: int
    broker_quality_score: float
    lang_Bilingual: int
    lang_English: int
    lang_French: int

class InteractionFeatures(BaseModel):
    expertise_match: float
    language_match: float
    workload_ratio: float
    quality_x_value: float
    position_bias: float
    interaction_number: int
    responded: int
    response_time_bucket_ord: int
    log_response_time_hours: float
    ribo_x_expertise: float
    claims_x_skill: float
    tenure_x_quality: float

class PredictionRequest(BaseModel):
    lead_id: str
    broker_id: str
    client_features: ClientFeatures
    broker_features: BrokerFeatures
    interaction_features: InteractionFeatures
    ab_test_version: Optional[str] = None   # for A/B testing

class PredictionResponse(BaseModel):
    lead_id: str
    broker_id: str
    conversion_probability: float
    prediction: int  # 0 or 1 after threshold
    model_version: Optional[str] = None

class BatchPredictionRequest(BaseModel):
    requests: List[PredictionRequest]

class BatchPredictionResponse(BaseModel):
    predictions: List[PredictionResponse]
    total_requests: int
    avg_probability: float

class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    model_version: Optional[str] = None

class ModelInfoResponse(BaseModel):
    model_version: str
    threshold: float
    client_features: List[str]
    broker_features: List[str]
    interaction_features: List[str]

# A/B test schemas
class ABTestConfig(BaseModel):
    experiment_id: str
    default_model: str
    variants: List[Dict[str, Any]]
    enabled: bool

class AssignmentLog(BaseModel):
    lead_id: str
    broker_id: str
    model_version: str
    predicted_probability: float
    assigned_at: datetime = Field(default_factory=datetime.utcnow)

class ConversionLog(BaseModel):
    lead_id: str
    broker_id: str
    converted: bool
    converted_at: Optional[datetime] = None

class ABTestResults(BaseModel):
    experiment_id: str
    variants: List[Dict[str, Any]]
    lift: Optional[str] = None
    p_value: Optional[float] = None

# ------------------------------
# 2. Model Loader (Mock for now; replace with your actual loader)
# ------------------------------
class ModelLoader:
    """Placeholder – replace with your actual model loading logic."""
    def __init__(self):
        self.models = {}  # version -> model
        self.threshold = 0.63
        self.model_config = {"version": "v1.0"}

    def load_model(self, version="v1.0", model_path=None):
        # Here you would load the actual PyTorch model
        # For demonstration, we just set a flag
        self.models[version] = "dummy_model"
        self.model_config["version"] = version
        return True

    def predict_single(self, client_features, broker_features, interaction_features, version="v1.0"):
        # Dummy prediction – replace with actual model inference
        import random
        return random.uniform(0.3, 0.8)

    def get_model_info(self):
        return {
            "model_version": self.model_config.get("version", "unknown"),
            "threshold": self.threshold,
            "client_features": ["quote_value", "lead_difficulty", ...],  # list actual features
            "broker_features": ["skill_level", ...],
            "interaction_features": ["expertise_match", ...]
        }

# ------------------------------
# 3. FastAPI App Initialization
# ------------------------------
app = FastAPI(
    title="Insurance Agent Recommendation API",
    description="API for predicting conversion probability between leads and brokers",
    version="2.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global instances
model_loader = ModelLoader()
# In‑memory storage for A/B test logs (replace with DB)
assignment_logs = []
conversion_logs = []

# Default A/B test config
default_ab_config = {
    "experiment_id": "routing_test_v2",
    "default_model": "v1.0",
    "variants": [
        {"model_version": "v1.0", "traffic_percent": 50},
        {"model_version": "v2.0", "traffic_percent": 50}
    ],
    "enabled": True
}

# ------------------------------
# 4. Helper: deterministic version assignment
# ------------------------------
def assign_model_version(lead_id: str) -> str:
    """Assign model version based on lead_id hash (50/50 split)."""
    if not default_ab_config["enabled"]:
        return default_ab_config["default_model"]
    hash_val = int(hashlib.md5(lead_id.encode()).hexdigest(), 16) % 100
    cumulative = 0
    for variant in default_ab_config["variants"]:
        cumulative += variant["traffic_percent"]
        if hash_val < cumulative:
            return variant["model_version"]
    return default_ab_config["default_model"]

# ------------------------------
# 5. Existing Endpoints
# ------------------------------
@app.on_event("startup")
async def startup_event():
    model_loader.load_model(version="v1.0")
    # Optionally pre‑load v2.0 as well
    model_loader.load_model(version="v2.0")

@app.get("/", response_model=HealthResponse)
async def root():
    return HealthResponse(
        status="OK",
        model_loaded=len(model_loader.models) > 0,
        model_version=model_loader.model_config.get("version")
    )

@app.get("/health", response_model=HealthResponse)
async def health_check():
    return HealthResponse(
        status="OK",
        model_loaded=len(model_loader.models) > 0,
        model_version=model_loader.model_config.get("version")
    )

@app.get("/model-info", response_model=ModelInfoResponse)
async def get_model_info():
    if not model_loader.models:
        raise HTTPException(status_code=503, detail="Model not loaded")
    info = model_loader.get_model_info()
    return ModelInfoResponse(**info)

@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    if not model_loader.models:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    # Determine which model version to use
    version = request.ab_test_version or assign_model_version(request.lead_id)
    if version not in model_loader.models:
        version = default_ab_config["default_model"]
    
    try:
        prob = model_loader.predict_single(
            request.client_features.dict(),
            request.broker_features.dict(),
            request.interaction_features.dict(),
            version=version
        )
        prediction = 1 if prob >= model_loader.threshold else 0
        return PredictionResponse(
            lead_id=request.lead_id,
            broker_id=request.broker_id,
            conversion_probability=prob,
            prediction=prediction,
            model_version=version
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/predict-batch", response_model=BatchPredictionResponse)
async def predict_batch(request: BatchPredictionRequest):
    if not model_loader.models:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    predictions = []
    total_prob = 0.0
    for req in request.requests:
        version = req.ab_test_version or assign_model_version(req.lead_id)
        if version not in model_loader.models:
            version = default_ab_config["default_model"]
        prob = model_loader.predict_single(
            req.client_features.dict(),
            req.broker_features.dict(),
            req.interaction_features.dict(),
            version=version
        )
        pred = 1 if prob >= model_loader.threshold else 0
        total_prob += prob
        predictions.append(PredictionResponse(
            lead_id=req.lead_id,
            broker_id=req.broker_id,
            conversion_probability=prob,
            prediction=pred,
            model_version=version
        ))
    avg = total_prob / len(predictions) if predictions else 0
    return BatchPredictionResponse(
        predictions=predictions,
        total_requests=len(predictions),
        avg_probability=avg
    )

@app.get("/test-sample")
async def get_test_sample():
    # Returns the same sample as before
    return {
        "lead_id": "test_lead_001",
        "broker_id": "test_broker_001",
        "client_features": {
            "quote_value": 5000.0,
            "lead_difficulty": 0.5,
            # ... (full sample as in your original)
        },
        "broker_features": {...},
        "interaction_features": {...}
    }

# ------------------------------
# 6. A/B Testing Endpoints
# ------------------------------
@app.get("/ab-test/config", response_model=ABTestConfig)
async def get_ab_config():
    return ABTestConfig(**default_ab_config)

@app.post("/ab-test/log-assignment")
async def log_assignment(log: AssignmentLog):
    assignment_logs.append(log.dict())
    # In production, store in database
    return {"status": "logged"}

@app.post("/ab-test/log-conversion")
async def log_conversion(conversion: ConversionLog):
    conversion_logs.append(conversion.dict())
    return {"status": "logged"}

@app.get("/ab-test/results", response_model=ABTestResults)
async def get_ab_results(experiment_id: Optional[str] = None):
    exp_id = experiment_id or default_ab_config["experiment_id"]
    
    # Aggregate assignments per model version
    assignments_by_version = defaultdict(int)
    conversions_by_version = defaultdict(int)
    
    # Map lead_id+broker_id to assigned version (from assignment_logs)
    # We'll assume each lead-broker pair is unique per assignment
    assignment_map = {}
    for log in assignment_logs:
        key = (log["lead_id"], log["broker_id"])
        assignment_map[key] = log["model_version"]
        assignments_by_version[log["model_version"]] += 1
    
    # Count conversions using assignment_map
    for log in conversion_logs:
        key = (log["lead_id"], log["broker_id"])
        if key in assignment_map:
            version = assignment_map[key]
            if log["converted"]:
                conversions_by_version[version] += 1
    
    # Build result variants
    variants = []
    for version, assign_count in assignments_by_version.items():
        conv_count = conversions_by_version.get(version, 0)
        rate = conv_count / assign_count if assign_count > 0 else 0
        variants.append({
            "model_version": version,
            "assignments": assign_count,
            "conversions": conv_count,
            "conversion_rate": round(rate, 4)
        })
    
    # Simple lift calculation between first two variants
    lift = None
    if len(variants) >= 2 and variants[0]["conversion_rate"] > 0:
        lift_val = (variants[1]["conversion_rate"] - variants[0]["conversion_rate"]) / variants[0]["conversion_rate"]
        lift = f"{lift_val * 100:.1f}%"
    
    return ABTestResults(
        experiment_id=exp_id,
        variants=variants,
        lift=lift,
        p_value=None  
    )

@app.post("/ab-test/assign-version")
async def assign_version(lead_id: str):
    """Server-side deterministic version assignment."""
    version = assign_model_version(lead_id)
    return {"lead_id": lead_id, "model_version": version, "experiment_id": default_ab_config["experiment_id"]}

# ------------------------------
# 7. Run the server
# ------------------------------
if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )