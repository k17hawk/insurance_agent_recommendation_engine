from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pathlib import Path
import time
import uvicorn
from contextlib import contextmanager
from datetime import datetime
from agent_recommender.server.schemas import *
from agent_recommender.server.feature_transformer import FeatureTransformer
from agent_recommender.server.broker_service import BrokerService
from agent_recommender.server.model_regitry import ModelRegistry
from agent_recommender.server.ab_test_manager import ABTestManager
from agent_recommender.server.latency_tracker import latency_tracker  # Import the instance
from agent_recommender import logger
import aiohttp
import asyncio

app = FastAPI(title="Broker Recommendation Gateway", version="2.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

# Global components
transformer = FeatureTransformer()
broker_service = BrokerService()
registry = ModelRegistry()
ab_manager = None

@contextmanager
def track_stage(stage: str, additional_data: dict = None):
    """Context manager to track latency for a stage"""
    start = time.perf_counter()
    try:
        yield
    finally:
        latency_ms = (time.perf_counter() - start) * 1000
        latency_tracker.record(stage, latency_ms, additional_data)

@contextmanager
def track_stage(stage: str, additional_data: dict = None):
    """Context manager to track latency for a stage"""
    start = time.perf_counter()
    try:
        yield
    finally:
        latency_ms = (time.perf_counter() - start) * 1000
        latency_tracker.record(stage, latency_ms, additional_data)

def get_champion_version():
    """Determine champion as the highest version number (excluding 'latest')."""
    versions = registry.get_available_versions()
    real_versions = [v for v in versions if v != "latest" and not v.startswith("latest")]
    if not real_versions:
        return None
    def version_key(v):
        try:
            return [int(x) for x in v[1:].split('.')]
        except:
            return [0,0,0]
    real_versions.sort(key=version_key)
    return real_versions[-1]

@app.on_event("startup")
async def startup():
    global ab_manager
    if registry.load_all_versions():
        champion = get_champion_version()
        if champion:
            ab_manager = ABTestManager(registry, champion_version=champion, candidate_version="latest", traffic_split_percent=50)
            logger.info(f"Gateway ready. Champion: {champion}, Candidate: latest")
        else:
            logger.warning("No champion model found. A/B test disabled.")
    else:
        logger.error("Failed to load any model")

# ---------- Health ----------
@app.get("/health", response_model=HealthResponse)
async def health():
    return HealthResponse(
        status="OK",
        models_loaded=registry.get_available_versions(),
        default_model=registry.default_version
    )

# ---------- Models ----------
@app.get("/models")
async def list_models():
    versions = registry.get_available_versions()
    friendly_versions = []
    for v in versions:
        if v == "latest":
            target = Path("models/production/latest").resolve()
            actual = target.name if target != Path("models/production/latest") else "unknown"
            friendly_versions.append({
                "id": "latest",
                "name": f"{actual} (Latest)",
                "is_latest": True,
                "actual_version": actual
            })
        else:
            friendly_versions.append({
                "id": v,
                "name": v,
                "is_latest": False,
                "actual_version": v
            })
    response = {"models": friendly_versions, "ab_test": None}
    if ab_manager and ab_manager.enabled:
        response["ab_test"] = {
            "enabled": True,
            "experiment_id": ab_manager.experiment_id,
            "champion": ab_manager.champion,
            "candidate": ab_manager.candidate,
            "candidate_traffic_percent": ab_manager.candidate_percent
        }
    else:
        response["ab_test"] = {"enabled": False}
    return response

# ---------- Brokers ----------
@app.get("/brokers")
async def list_brokers(limit: int = 100):
    """Return list of brokers for frontend dropdown."""
    brokers = broker_service.list_brokers(limit=limit)
    return {"brokers": brokers}

# ---------- Prediction with Latency Tracking ----------
@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    # Track total request latency
    with track_stage("total_request", {"lead_id": request.lead_id, "broker_id": request.broker_id}):
        
        # Stage 1: Model availability check
        with track_stage("model_check"):
            if not registry.models:
                raise HTTPException(503, "No models loaded")
        
        # Stage 2: Broker lookup
        with track_stage("broker_lookup"):
            broker = broker_service.get_broker(request.broker_id)
            if not broker:
                raise HTTPException(404, f"Broker {request.broker_id} not found")
        
        # Stage 3: Request conversion
        with track_stage("request_conversion"):
            lead_dict = request.dict(exclude={"lead_id", "broker_id", "model_version"})
            lead_dict["insurance_type"] = request.insurance_type.value
            lead_dict["language"] = request.language.value
            lead_dict["claims_risk"] = request.claims_risk.value
            lead_dict["multi_product_intent"] = request.multi_product_intent
            lead_dict["is_weekend"] = int(request.is_weekend)
        
        # Stage 4: Feature transformation
        with track_stage("feature_transformation"):
            try:
                client_vec, broker_vec, inter_vec = transformer.transform(lead_dict, broker)
            except Exception as e:
                logger.error(f"Feature transformation failed: {e}")
                raise HTTPException(400, f"Invalid features: {e}")
        
        # Stage 5: Model version selection
        with track_stage("version_selection"):
            if request.model_version:
                version = request.model_version
                if version.endswith("(Latest)"):
                    version = "latest"
                if version not in registry.models:
                    if version in registry.models:
                        version = version
                    else:
                        version = ab_manager.champion if ab_manager else registry.default_version
            else:
                if ab_manager and ab_manager.enabled:
                    version = ab_manager.assign_version(request.lead_id)
                else:
                    version = registry.default_version
        
        # Stage 6: Version resolution
        with track_stage("version_resolution"):
            used_version = version
            if version == "latest":
                latest_path = Path("models/production/latest").resolve()
                used_version = latest_path.name if latest_path != Path("models/production/latest") else "latest"
        
        # Stage 7: Model inference (returns prob, version, latency_ms)
        with track_stage("model_inference"):
            prob, _, inference_latency = registry.predict(client_vec, broker_vec, inter_vec, version)
            # Also track detailed inference latency separately
            latency_tracker.record("inference_detail", inference_latency, {"model_version": used_version})
        
        # Stage 8: Threshold application
        with track_stage("threshold_apply"):
            pred = 1 if prob >= registry.threshold else 0
        
        # Stage 9: A/B logging (async to not block response)
        if not request.model_version and ab_manager:
            with track_stage("ab_logging"):
                ab_manager.log_assignment(request.lead_id, request.broker_id, used_version, prob)
        
        return PredictionResponse(
            lead_id=request.lead_id,
            broker_id=request.broker_id,
            conversion_probability=prob,
            prediction=pred,
            model_version=used_version,
            inference_time_ms=inference_latency  # Add this to schema
        )

# ---------- Latency Tracking Endpoints ----------
@app.get("/latency/stats")
async def get_latency_stats():
    """Get detailed latency statistics for all prediction stages"""
    return {
        "timestamp": datetime.now().isoformat(),
        "total_predictions": latency_tracker.metrics.get("total_request", {}).get("count", 0),
        "statistics": latency_tracker.get_stats(),
        "summary": latency_tracker.get_summary(),
        "model_latency": registry.get_model_latency_stats(),
        "transformation_latency": transformer.get_transformation_stats() if hasattr(transformer, 'get_transformation_stats') else {}
    }

@app.get("/latency/summary")
async def get_latency_summary():
    """Get summary latency statistics"""
    return latency_tracker.get_summary()

@app.get("/latency/recent")
async def get_recent_predictions(limit: int = 100):
    """Get recent prediction latencies"""
    return {
        "recent_predictions": latency_tracker.get_recent_predictions(limit),
        "count": len(latency_tracker.prediction_details)
    }

@app.post("/latency/reset")
async def reset_latency_stats():
    """Reset all latency statistics"""
    latency_tracker.reset()
    return {"message": "Latency statistics reset", "timestamp": datetime.now().isoformat()}

# ---------- A/B Test Endpoints ----------
@app.post("/ab-test/log-conversion")
async def log_conversion(log: ConversionLog):
    ab_manager.log_conversion(log.lead_id, log.broker_id, log.converted)
    return {"status": "logged"}

@app.get("/ab-test/results", response_model=ABTestResults)
async def get_ab_results():
    return ABTestResults(**ab_manager.get_results())

@app.get("/ab-test/config")
async def get_ab_config():
    return ab_manager.get_config()

@app.get("/ab-test/debug")
async def debug_ab_test():
    """Debug endpoint to see raw A/B test data"""
    if not ab_manager:
        return {"error": "AB manager not initialized"}
    
    debug_info = ab_manager.get_debug_info()
    return debug_info

# ---------- Performance Benchmark ----------
@app.post("/benchmark")
async def run_benchmark(n_requests: int = 100, background_tasks: BackgroundTasks = None):
    """Run a quick benchmark to test performance"""
    
    
    # Create a sample request
    sample_broker = broker_service.list_brokers(limit=1)
    if not sample_broker["brokers"]:
        raise HTTPException(404, "No brokers available for benchmark")
    
    sample_request = {
        "lead_id": "benchmark_lead",
        "broker_id": sample_broker["brokers"][0]["broker_id"],
        "insurance_type": "Auto",
        "language": "English",
        "claims_risk": "None",
        "quote_value": 5000.0,
        "lead_difficulty": 0.5,
        "sophistication": 0.7,
        "patience_hours": 48.0,
        "digital_engagement_score": 75.0,
        "tenure_years": 3.5,
        "month": 6,
        "hour_of_day": 14,
        "lead_dayofweek": 2,
        "lead_quarter": 2,
        "is_weekend": False,
        "multi_product_intent": True
    }
    
    # Run concurrent requests
    async def make_request(session, i):
        async with session.post("http://localhost:8000/predict", json=sample_request) as resp:
            return await resp.json()
    
    start_time = time.perf_counter()
    async with aiohttp.ClientSession() as session:
        tasks = [make_request(session, i) for i in range(n_requests)]
        results = await asyncio.gather(*tasks)
    total_time = time.perf_counter() - start_time
    
    return {
        "benchmark": {
            "total_requests": n_requests,
            "total_time_seconds": round(total_time, 2),
            "throughput_rps": round(n_requests / total_time, 2),
            "avg_latency_ms": round((total_time / n_requests) * 1000, 2)
        }
    }

# ---------- Run ----------
if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)