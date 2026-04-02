import torch
import json
import numpy as np
import time
from pathlib import Path
from typing import Dict, Any, List
from agent_recommender.components.model_training import TwoTowerModel
from agent_recommender import logger


class ModelLoader:
    """Load and manage the trained model"""
    
    def __init__(self, model_path: str = "artifacts/model_push/models/production/latest"):
        self.model_path = Path(model_path)
        self.model = None
        self.model_config = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.threshold = 0.5
        
        # Latency tracking
        self.latency_stats = {
            "predictions_count": 0,
            "total_time_ms": 0,
            "avg_time_ms": 0,
            "min_time_ms": float('inf'),
            "max_time_ms": 0,
            "recent_times": []  # Store last 100 predictions
        }
        
    def load_model(self):
        """Load the model and its configuration"""
        try:
            # Load model configuration
            config_path = self.model_path / "model_config.json"
            if not config_path.exists():
                config_path = Path("artifacts/model_training/models/model_config.json")
            
            with open(config_path, 'r') as f:
                self.model_config = json.load(f)
            
            self.threshold = self.model_config.get('optimal_threshold', 0.5)
            
            # Initialize model
            self.model = TwoTowerModel(
                client_dim=self.model_config['client_dim'],
                broker_dim=self.model_config['broker_dim'],
                interaction_dim=self.model_config['interaction_dim'],
                embed_dim=self.model_config['embedding_dim'],
                hidden_dim=self.model_config['hidden_dim'],
                dropout=self.model_config['dropout']
            ).to(self.device)
            
            # Load weights
            weights_path = self.model_path / "two_tower_best.pt"
            if not weights_path.exists():
                weights_path = Path("artifacts/model_training/models/two_tower_best.pt")
            
            self.model.load_state_dict(torch.load(weights_path, map_location=self.device))
            self.model.eval()
            
            logger.info(f"Model loaded successfully on {self.device}")
            logger.info(f"Threshold: {self.threshold}")
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            return False
    
    def predict_single(self, client_features: Dict, broker_features: Dict, interaction_features: Dict) -> tuple:
        """Predict conversion probability with latency measurement"""
        start_time = time.perf_counter()  # High precision timer
        
        try:
            # Convert features to tensors
            client_tensor = self._dict_to_tensor(client_features, self.model_config['client_features'])
            broker_tensor = self._dict_to_tensor(broker_features, self.model_config['broker_features'])
            interaction_tensor = self._dict_to_tensor(interaction_features, self.model_config['interaction_features'])
            
            # Add batch dimension
            client_tensor = client_tensor.unsqueeze(0)
            broker_tensor = broker_tensor.unsqueeze(0)
            interaction_tensor = interaction_tensor.unsqueeze(0)
            
            # Move to device
            client_tensor = client_tensor.to(self.device)
            broker_tensor = broker_tensor.to(self.device)
            interaction_tensor = interaction_tensor.to(self.device)
            
            # Predict
            with torch.no_grad():
                logits = self.model(client_tensor, broker_tensor, interaction_tensor)
                prob = torch.sigmoid(logits).cpu().item()
            
            # Calculate latency
            end_time = time.perf_counter()
            latency_ms = (end_time - start_time) * 1000
            
            # Update statistics
            self._update_latency_stats(latency_ms)
            
            return prob, latency_ms
            
        except Exception as e:
            logger.error(f"Prediction error: {e}")
            raise
    
    def predict_batch(self, requests: List[Dict]) -> tuple:
        """Predict conversion probabilities for multiple lead-broker pairs with latency"""
        start_time = time.perf_counter()
        
        try:
            # Prepare batch tensors
            client_tensors = []
            broker_tensors = []
            interaction_tensors = []
            
            for req in requests:
                client_tensor = self._dict_to_tensor(req['client_features'], self.model_config['client_features'])
                broker_tensor = self._dict_to_tensor(req['broker_features'], self.model_config['broker_features'])
                interaction_tensor = self._dict_to_tensor(req['interaction_features'], self.model_config['interaction_features'])
                
                client_tensors.append(client_tensor)
                broker_tensors.append(broker_tensor)
                interaction_tensors.append(interaction_tensor)
            
            # Stack tensors
            client_batch = torch.stack(client_tensors).to(self.device)
            broker_batch = torch.stack(broker_tensors).to(self.device)
            interaction_batch = torch.stack(interaction_tensors).to(self.device)
            
            # Predict
            with torch.no_grad():
                logits = self.model(client_batch, broker_batch, interaction_batch)
                probs = torch.sigmoid(logits).cpu().numpy()
            
            # Calculate latency
            end_time = time.perf_counter()
            total_latency_ms = (end_time - start_time) * 1000
            per_prediction_ms = total_latency_ms / len(requests)
            
            # Update statistics
            self._update_latency_stats(per_prediction_ms)
            
            return probs.tolist(), total_latency_ms, per_prediction_ms
            
        except Exception as e:
            logger.error(f"Batch prediction error: {e}")
            raise
    
    def _dict_to_tensor(self, features: Dict, feature_names: List[str]) -> torch.Tensor:
        """Convert dictionary of features to tensor"""
        feature_values = []
        for name in feature_names:
            if name in features:
                feature_values.append(float(features[name]))
            else:
                logger.warning(f"Feature {name} not found, using 0")
                feature_values.append(0.0)
        
        return torch.tensor(feature_values, dtype=torch.float32)
    
    def _update_latency_stats(self, latency_ms: float):
        """Update latency statistics"""
        self.latency_stats["predictions_count"] += 1
        self.latency_stats["total_time_ms"] += latency_ms
        self.latency_stats["avg_time_ms"] = self.latency_stats["total_time_ms"] / self.latency_stats["predictions_count"]
        self.latency_stats["min_time_ms"] = min(self.latency_stats["min_time_ms"], latency_ms)
        self.latency_stats["max_time_ms"] = max(self.latency_stats["max_time_ms"], latency_ms)
        
        # Keep last 100 prediction times
        self.latency_stats["recent_times"].append(latency_ms)
        if len(self.latency_stats["recent_times"]) > 100:
            self.latency_stats["recent_times"].pop(0)
    
    def get_latency_stats(self) -> Dict:
        """Get current latency statistics"""
        return {
            "total_predictions": self.latency_stats["predictions_count"],
            "avg_latency_ms": round(self.latency_stats["avg_time_ms"], 2),
            "min_latency_ms": round(self.latency_stats["min_time_ms"], 2) if self.latency_stats["min_time_ms"] != float('inf') else 0,
            "max_latency_ms": round(self.latency_stats["max_time_ms"], 2),
            "p50_latency_ms": round(np.percentile(self.latency_stats["recent_times"], 50), 2) if self.latency_stats["recent_times"] else 0,
            "p95_latency_ms": round(np.percentile(self.latency_stats["recent_times"], 95), 2) if self.latency_stats["recent_times"] else 0,
            "p99_latency_ms": round(np.percentile(self.latency_stats["recent_times"], 99), 2) if self.latency_stats["recent_times"] else 0
        }
    
    def get_model_info(self) -> Dict:
        """Get model information"""
        return {
            'model_type': 'TwoTowerModel',
            'model_version': self.model_config.get('version', 'v1.0.0'),
            'client_features': self.model_config['client_features'],
            'broker_features': self.model_config['broker_features'],
            'interaction_features': self.model_config['interaction_features'],
            'threshold': self.threshold,
            'embedding_dim': self.model_config['embedding_dim'],
            'hidden_dim': self.model_config['hidden_dim'],
            'dropout': self.model_config['dropout']
        }