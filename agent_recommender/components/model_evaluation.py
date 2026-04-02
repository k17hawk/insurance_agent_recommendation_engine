import json
import torch
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score
from agent_recommender import logger
from agent_recommender.entity.config_entity import ModelEvaluationConfig
from agent_recommender.components.model_training import TwoTowerModel  
import shutil

class ModelEvaluation:
    def __init__(self, config: ModelEvaluationConfig):
        self.config = config
        self.metrics = {}

    def _load_test_data(self):
        """Load test features and labels from numpy files."""
        test_client_path = self.config.test_data_dir / "test_client.npy"
        if not test_client_path.exists():
            raise FileNotFoundError(f"Test numpy files not found in {self.config.test_data_dir}. Run data transformation first.")
        
        test_client = np.load(self.config.test_data_dir / "test_client.npy")
        test_broker = np.load(self.config.test_data_dir / "test_broker.npy")
        test_interaction = np.load(self.config.test_data_dir / "test_interaction.npy")
        test_labels = np.load(self.config.test_data_dir / "test_labels.npy")
        
        return test_client, test_broker, test_interaction, test_labels

    def _load_model(self):
        """Load trained model from model_dir."""
        model_path = self.config.model_dir / "two_tower_best.pt"
        config_path = self.config.model_dir / "model_config.json"
        if not config_path.exists():
            raise FileNotFoundError(f"Model config not found: {config_path}")
        
        with open(config_path, 'r') as f:
            model_cfg = json.load(f)
        
        
        # Extract parameters with defaults if missing
        client_dim = model_cfg.get("client_dim")
        broker_dim = model_cfg.get("broker_dim")
        interaction_dim = model_cfg.get("interaction_dim")
        embed_dim = model_cfg.get("embedding_dim", 64)
        hidden_dim = model_cfg.get("hidden_dim", 128)
        dropout = model_cfg.get("dropout", 0.3)
        
        # Instantiate with all required arguments
        model = TwoTowerModel(
            client_dim=client_dim,
            broker_dim=broker_dim,
            interaction_dim=interaction_dim,
            embed_dim=embed_dim,
            hidden_dim=hidden_dim,
            dropout=dropout
        )
        model.load_state_dict(torch.load(model_path, map_location="cpu"))
        model.eval()
        return model

    def evaluate(self):
        logger.info("Starting model evaluation...")
        # Load data
        X_client, X_broker, X_inter, y_true = self._load_test_data()
        model = self._load_model()

        # Run predictions
        probs = []
        batch_size = 512
        with torch.no_grad():
            for i in range(0, len(y_true), batch_size):
                client_batch = torch.tensor(X_client[i:i+batch_size], dtype=torch.float32)
                broker_batch = torch.tensor(X_broker[i:i+batch_size], dtype=torch.float32)
                inter_batch = torch.tensor(X_inter[i:i+batch_size], dtype=torch.float32)
                logits = model(client_batch, broker_batch, inter_batch)
                batch_probs = torch.sigmoid(logits).numpy()
                # Check for NaNs in batch
                if np.isnan(batch_probs).any():
                    logger.warning(f"NaN detected in batch {i}. Replacing with 0.5")
                    batch_probs = np.nan_to_num(batch_probs, nan=0.5)
                probs.extend(batch_probs)
        probs = np.array(probs)

        # Additional safety: replace any remaining NaNs
        if np.isnan(probs).any():
            logger.warning("NaN values found in final probabilities. Replacing with 0.5.")
            probs = np.nan_to_num(probs, nan=0.5)

        # Compute metrics
        preds = (probs >= self.config.threshold).astype(int)
        self.metrics = {
            "auc": float(roc_auc_score(y_true, probs)),
            "f1": float(f1_score(y_true, preds)),
            "precision": float(precision_score(y_true, preds)),
            "recall": float(recall_score(y_true, preds)),
            "threshold": self.config.threshold,
            "timestamp": datetime.now().isoformat()
        }

        # Save metrics to reports_dir and also copy to model_dir for push stage
        self.config.reports_dir.mkdir(parents=True, exist_ok=True)
        with open(self.config.metrics_file, 'w') as f:
            json.dump(self.metrics, f, indent=4)

        shutil.copy(self.config.metrics_file, self.config.model_dir / "metrics.json")

        logger.info(f"Evaluation complete. Metrics: {self.metrics}")
        return self.metrics