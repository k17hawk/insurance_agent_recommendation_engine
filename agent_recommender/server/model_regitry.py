import torch
import json
import time
import numpy as np
from pathlib import Path
from typing import Dict, Tuple
from agent_recommender import logger
from agent_recommender.components.model_training import TwoTowerModel
class ModelRegistry:
    def __init__(self, production_dir: Path = Path("models/production")):
        self.production_dir = production_dir
        self.models: Dict[str, TwoTowerModel] = {}
        self.configs: Dict[str, dict] = {}
        self.threshold = 0.63
        self.default_version = "latest"
        
        # Track per-model latency
        self.model_latencies: Dict[str, Dict] = {}

    def load_all_versions(self) -> bool:
        if not self.production_dir.exists():
            logger.error(f"Production directory not found: {self.production_dir}")
            return False

        for version_dir in self.production_dir.iterdir():
            if version_dir.is_dir() and version_dir.name != "latest":
                self._load_version(version_dir.name, version_dir)
                self.model_latencies[version_dir.name] = {
                    "count": 0, "total_ms": 0, "recent": []
                }

        # Load latest symlink
        latest_dir = self.production_dir / "latest"
        if latest_dir.exists():
            resolved = latest_dir.resolve()
            if resolved != latest_dir and resolved.is_dir():
                self._load_version("latest", resolved)
                self.default_version = "latest"
                self.model_latencies["latest"] = {
                    "count": 0, "total_ms": 0, "recent": []
                }

        logger.info(f"Loaded {len(self.models)} model versions: {list(self.models.keys())}")
        return len(self.models) > 0

    def _load_version(self, version_name: str, version_path: Path):
        from agent_recommender.components.model_training import TwoTowerModel
        
        model_file = version_path / "two_tower_best.pt"
        config_file = version_path / "model_config.json"
        if not model_file.exists():
            logger.warning(f"Model file missing for {version_name}")
            return
        with open(config_file, 'r') as f:
            cfg = json.load(f)
        model = TwoTowerModel(
            client_dim=cfg["client_dim"],
            broker_dim=cfg["broker_dim"],
            interaction_dim=cfg["interaction_dim"],
            embed_dim=cfg.get("embedding_dim", 64),
            hidden_dim=cfg.get("hidden_dim", 128),
            dropout=cfg.get("dropout", 0.3)
        )
        model.load_state_dict(torch.load(model_file, map_location="cpu"))
        model.eval()
        self.models[version_name] = model
        self.configs[version_name] = cfg
        # Update threshold if config has it
        if "optimal_threshold" in cfg:
            self.threshold = cfg["optimal_threshold"]

    def predict(self, client_vec, broker_vec, inter_vec, version: str = "latest") -> Tuple[float, str, float]:
        """Predict with latency tracking per model version"""
        start_time = time.perf_counter()
        
        if version not in self.models:
            version = self.default_version
        if version not in self.models:
            raise ValueError(f"Model version {version} not loaded")
        
        model = self.models[version]
        with torch.no_grad():
            client_t = torch.tensor(client_vec.reshape(1, -1), dtype=torch.float32)
            broker_t = torch.tensor(broker_vec.reshape(1, -1), dtype=torch.float32)
            inter_t = torch.tensor(inter_vec.reshape(1, -1), dtype=torch.float32)
            logits = model(client_t, broker_t, inter_t)
            prob = torch.sigmoid(logits).item()
        
        latency_ms = (time.perf_counter() - start_time) * 1000
        
        # Track per-model latency
        if version in self.model_latencies:
            self.model_latencies[version]["count"] += 1
            self.model_latencies[version]["total_ms"] += latency_ms
            self.model_latencies[version]["recent"].append(latency_ms)
            if len(self.model_latencies[version]["recent"]) > 100:
                self.model_latencies[version]["recent"].pop(0)
        
        return prob, version, latency_ms

    def get_available_versions(self):
        return list(self.models.keys())
    
    def get_model_latency_stats(self) -> Dict:
        """Get latency statistics per model version"""
        stats = {}
        for version, data in self.model_latencies.items():
            if data["count"] > 0:
                recent = data["recent"]
                stats[version] = {
                    "predictions": data["count"],
                    "avg_latency_ms": round(data["total_ms"] / data["count"], 2),
                    "p50_latency_ms": round(np.percentile(recent, 50), 2) if recent else 0,
                    "p95_latency_ms": round(np.percentile(recent, 95), 2) if recent else 0,
                    "p99_latency_ms": round(np.percentile(recent, 99), 2) if recent else 0
                }
        return stats