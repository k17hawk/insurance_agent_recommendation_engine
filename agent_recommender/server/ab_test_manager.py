import hashlib
from collections import defaultdict
from datetime import datetime
from typing import List, Dict, Any, Optional
from pathlib import Path
from agent_recommender import logger

class ABTestManager:
    def __init__(self, model_registry, champion_version: str, candidate_version: str = "latest", traffic_split_percent: int = 50):
        """
        champion_version: the stable model (e.g., "v1.0.1")
        candidate_version: the new model to test (e.g., "latest" which resolves to actual version)
        traffic_split_percent: percentage of traffic to send to candidate (remainder to champion)
        """
        self.registry = model_registry
        self.experiment_id = "champion_vs_candidate"
        self.enabled = True
        self.champion = champion_version
        self.candidate = candidate_version
        self.candidate_percent = traffic_split_percent
        self.assignment_logs = []  # Store all assignments
        self.conversion_logs = []  # Store all conversions

    def get_config(self):
        return {
            "experiment_id": self.experiment_id,
            "champion": self.champion,
            "candidate": self.candidate,
            "candidate_traffic_percent": self.candidate_percent,
            "enabled": self.enabled
        }

    def assign_version(self, lead_id: str) -> str:
        """Returns either champion or candidate based on hash of lead_id."""
        if not self.enabled:
            return self.champion
        
        # Resolve candidate if it's "latest"
        candidate_version = self.candidate
        if candidate_version == "latest":
            # Get the actual version that "latest" points to
            latest_path = Path("models/production/latest").resolve()
            candidate_version = latest_path.name if latest_path != Path("models/production/latest") else "latest"
        
        # Use lead_id to determine version (consistent for same lead)
        hash_val = int(hashlib.md5(lead_id.encode()).hexdigest(), 16) % 100
        if hash_val < self.candidate_percent:
            logger.debug(f"Lead {lead_id} assigned to candidate: {candidate_version}")
            return candidate_version
        logger.debug(f"Lead {lead_id} assigned to champion: {self.champion}")
        return self.champion

    def log_assignment(self, lead_id: str, broker_id: str, version: str, prob: float):
        """Log a model assignment for A/B testing"""
        log_entry = {
            "lead_id": lead_id,
            "broker_id": broker_id,
            "model_version": version,
            "predicted_probability": prob,
            "assigned_at": datetime.utcnow().isoformat()
        }
        self.assignment_logs.append(log_entry)
        logger.info(f"AB Test Assignment - Lead: {lead_id}, Version: {version}, Prob: {prob:.3f}")
        
        # Keep only last 10000 logs to prevent memory issues
        if len(self.assignment_logs) > 10000:
            self.assignment_logs = self.assignment_logs[-10000:]

    def log_conversion(self, lead_id: str, broker_id: str, converted: bool):
        """Log a conversion for A/B testing"""
        log_entry = {
            "lead_id": lead_id,
            "broker_id": broker_id,
            "converted": converted,
            "converted_at": datetime.utcnow().isoformat()
        }
        self.conversion_logs.append(log_entry)
        logger.info(f"AB Test Conversion - Lead: {lead_id}, Converted: {converted}")
        
        # Keep only last 10000 logs
        if len(self.conversion_logs) > 10000:
            self.conversion_logs = self.conversion_logs[-10000:]

    def get_results(self):
        """Get A/B test results with proper statistics"""
        # Build assignment map (lead_id, broker_id) -> model_version
        assign_map = {}
        for log in self.assignment_logs:
            key = (log["lead_id"], log["broker_id"])
            assign_map[key] = log["model_version"]
        
        # Initialize counters
        counts = defaultdict(lambda: {"assignments": 0, "conversions": 0})
        
        # Count assignments
        for log in self.assignment_logs:
            version = log["model_version"]
            counts[version]["assignments"] += 1
        
        # Count conversions
        for log in self.conversion_logs:
            key = (log["lead_id"], log["broker_id"])
            if key in assign_map:
                version = assign_map[key]
                if log["converted"]:
                    counts[version]["conversions"] += 1
            else:
                # If conversion without assignment, log as unknown
                logger.warning(f"Conversion without assignment for lead {log['lead_id']}")
        
        # Build results
        results = []
        for version, stats in counts.items():
            rate = stats["conversions"] / stats["assignments"] if stats["assignments"] > 0 else 0
            results.append({
                "model_version": version,
                "assignments": stats["assignments"],
                "conversions": stats["conversions"],
                "conversion_rate": round(rate, 4)
            })
        
        # Calculate lift between candidate and champion
        lift = None
        # Resolve candidate version for comparison
        candidate_version = self.candidate
        if candidate_version == "latest":
            latest_path = Path("models/production/latest").resolve()
            candidate_version = latest_path.name if latest_path != Path("models/production/latest") else "latest"
        
        candidate_stats = next((r for r in results if r["model_version"] == candidate_version), None)
        champion_stats = next((r for r in results if r["model_version"] == self.champion), None)
        
        if candidate_stats and champion_stats and champion_stats["conversion_rate"] > 0:
            lift_val = (candidate_stats["conversion_rate"] - champion_stats["conversion_rate"]) / champion_stats["conversion_rate"]
            lift = f"{lift_val*100:.1f}%"
        
        logger.info(f"AB Test Results - Champion: {champion_stats}, Candidate: {candidate_stats}")
        
        return {
            "experiment_id": self.experiment_id,
            "variants": results,
            "lift": lift,
            "p_value": None  
        }
    
    def get_debug_info(self):
        """Debug method to see raw logs"""
        return {
            "assignment_logs_count": len(self.assignment_logs),
            "conversion_logs_count": len(self.conversion_logs),
            "recent_assignments": self.assignment_logs[-5:] if self.assignment_logs else [],
            "recent_conversions": self.conversion_logs[-5:] if self.conversion_logs else [],
            "config": self.get_config()
        }