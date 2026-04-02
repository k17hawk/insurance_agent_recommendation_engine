import time
import numpy as np
from collections import defaultdict
from typing import Dict, List, Optional
from datetime import datetime
from agent_recommender import logger


class LatencyTracker:
    """Track latency across different stages of prediction pipeline"""
    
    def __init__(self):
        self.metrics = defaultdict(lambda: {
            "count": 0,
            "total_ms": 0,
            "min_ms": float('inf'),
            "max_ms": 0,
            "recent": [],  # Last 100 measurements
            "last_updated": None
        })
        self.prediction_details = []  # Store last 1000 detailed predictions
    
    def record(self, stage: str, latency_ms: float, additional_data: Optional[Dict] = None):
        """Record latency for a specific stage"""
        stats = self.metrics[stage]
        stats["count"] += 1
        stats["total_ms"] += latency_ms
        stats["min_ms"] = min(stats["min_ms"], latency_ms)
        stats["max_ms"] = max(stats["max_ms"], latency_ms)
        stats["recent"].append(latency_ms)
        stats["last_updated"] = datetime.now().isoformat()
        
        # Keep only last 100 for recent stats
        if len(stats["recent"]) > 100:
            stats["recent"].pop(0)
        
        # Optionally store detailed prediction data
        if additional_data and len(self.prediction_details) < 1000:
            self.prediction_details.append({
                "timestamp": datetime.now().isoformat(),
                "stage": stage,
                "latency_ms": latency_ms,
                **additional_data
            })
    
    def get_stats(self) -> Dict:
        """Get current latency statistics"""
        result = {}
        for stage, stats in self.metrics.items():
            recent = stats["recent"]
            result[stage] = {
                "count": stats["count"],
                "avg_ms": round(stats["total_ms"] / stats["count"], 2) if stats["count"] > 0 else 0,
                "min_ms": round(stats["min_ms"], 2) if stats["min_ms"] != float('inf') else 0,
                "max_ms": round(stats["max_ms"], 2),
                "p50_ms": round(np.percentile(recent, 50), 2) if recent else 0,
                "p95_ms": round(np.percentile(recent, 95), 2) if recent else 0,
                "p99_ms": round(np.percentile(recent, 99), 2) if recent else 0,
                "last_updated": stats["last_updated"]
            }
        return result
    
    def get_summary(self) -> Dict:
        """Get summary statistics with percentage breakdown"""
        stats = self.get_stats()
        if "total_request" not in stats:
            return {"error": "No total_request data available"}
        
        total_avg = stats["total_request"]["avg_ms"]
        if total_avg == 0:
            return {"error": "No predictions yet"}
        
        breakdown = {}
        for stage, data in stats.items():
            if stage != "total_request" and data["avg_ms"] > 0:
                breakdown[stage] = round((data["avg_ms"] / total_avg) * 100, 1)
        
        return {
            "total_predictions": stats["total_request"]["count"],
            "avg_total_ms": stats["total_request"]["avg_ms"],
            "p95_total_ms": stats["total_request"]["p95_ms"],
            "p99_total_ms": stats["total_request"]["p99_ms"],
            "breakdown": dict(sorted(breakdown.items(), key=lambda x: x[1], reverse=True))
        }
    
    def reset(self):
        """Reset all metrics"""
        self.metrics.clear()
        self.prediction_details.clear()
        logger.info("Latency tracker reset")
    
    def get_recent_predictions(self, limit: int = 100) -> List[Dict]:
        """Get recent prediction details"""
        return self.prediction_details[-limit:]


latency_tracker = LatencyTracker()