import time
import joblib
import numpy as np
import pandas as pd
import json
from pathlib import Path
from typing import Dict, Tuple,List
from agent_recommender import logger


class FeatureTransformer:
    def __init__(self, transformers_dir: Path = Path("artifacts/data_transformation/transformed/transformers")):
        self.transformers_dir = transformers_dir
        self.median_imputer = None
        self.quality_scaler = None
        self.outlier_capper = None
        self.label_encoders = None
        self.max_quote_value = None
        self.feature_lists = None
        self._load_transformers()
        
        # Track transformation latency
        self.transformation_stats = {
            "count": 0,
            "total_ms": 0,
            "recent": []
        }

    def _load_transformers(self):
        """Load all transformers from disk"""
        median_path = self.transformers_dir / "median_imputer.pkl"
        quality_path = self.transformers_dir / "quality_scaler.pkl"
        outlier_path = self.transformers_dir / "outlier_capper.pkl"
        label_path = self.transformers_dir / "label_encoders.pkl"
        max_quote_path = self.transformers_dir / "max_quote_value.pkl"
        feature_path = self.transformers_dir.parent / "feature_lists.json"

        if median_path.exists():
            self.median_imputer = joblib.load(median_path)
            logger.info("Loaded median imputer")
        else:
            logger.warning(f"Median imputer not found at {median_path}")
            
        if quality_path.exists():
            self.quality_scaler = joblib.load(quality_path)
            logger.info("Loaded quality scaler")
            
        if outlier_path.exists():
            self.outlier_capper = joblib.load(outlier_path)
            logger.info("Loaded outlier capper")
            
        if label_path.exists():
            self.label_encoders = joblib.load(label_path)
            logger.info("Loaded label encoders")
            
        if max_quote_path.exists():
            self.max_quote_value = joblib.load(max_quote_path)
            logger.info(f"Loaded max quote value: {self.max_quote_value}")
            
        if feature_path.exists():
            with open(feature_path, 'r') as f:
                self.feature_lists = json.load(f)
            logger.info("Loaded feature lists")
        else:
            logger.warning(f"Feature lists not found at {feature_path}")
            # Define fallback feature lists (should match training)
            self.feature_lists = self._get_fallback_feature_lists()

    def _get_fallback_feature_lists(self) -> Dict:
        """Fallback feature lists if feature_lists.json not found"""
        return {
            "client_features": [
                "quote_value", "lead_difficulty", "sophistication", "patience_hours",
                "digital_engagement_score", "tenure_years", "log_quote_value", "log_patience_hours",
                "month", "hour_of_day", "lead_dayofweek", "lead_quarter", "is_weekend",
                "insurance_type_enc", "claims_risk", "multi_product_intent",
                "insurance_type_missing", "language_missing", "tenure_years_missing",
                "digital_engagement_score_missing"
            ],
            "broker_features": [
                "skill_level", "conversion_rate", "csat_score", "reliability", "efficiency",
                "avg_response_time", "burnout_risk", "commission_rate", "cost_per_lead",
                "utilization", "ribo_licensed", "is_new_broker", "expertise_auto",
                "expertise_home", "expertise_bundle", "broker_quality_score",
                "lang_Bilingual", "lang_English", "lang_French"
            ],
            "interaction_features": [
                "expertise_match", "language_match", "workload_ratio", "quality_x_value",
                "position_bias", "interaction_number", "responded", "response_time_bucket_ord",
                "log_response_time_hours", "ribo_x_expertise", "claims_x_skill", "tenure_x_quality"
            ]
        }

    def _create_missing_flags(self, lead: dict) -> dict:
        """Create missing flags (all 0 for frontend data since all fields are provided)"""
        return {
            "insurance_type_missing": 0,
            "language_missing": 0,
            "tenure_years_missing": 0,
            "digital_engagement_score_missing": 0
        }

    def _impute_numeric(self, data: dict) -> dict:
        """Impute numeric values using stored medians"""
        if self.median_imputer:
            for col, median_val in self.median_imputer.items():
                if col in data and (pd.isna(data[col]) or data[col] is None):
                    data[col] = median_val
        return data

    def _apply_log_transforms(self, lead: dict) -> dict:
        """Apply log1p transformation to skewed features"""
        log_cols = ["quote_value", "patience_hours"]
        for col in log_cols:
            if col in lead and lead[col] is not None:
                lead[f"log_{col}"] = np.log1p(max(lead[col], 0))
            else:
                lead[f"log_{col}"] = 0
        return lead

    def _cap_outliers(self, lead: dict) -> dict:
        """Cap outliers using stored percentiles"""
        if self.outlier_capper:
            for col, bounds in self.outlier_capper.items():
                if col in lead and lead[col] is not None:
                    lead[col] = np.clip(lead[col], bounds["p01"], bounds["p99"])
        return lead

    def _encode_categorical(self, lead: dict) -> dict:
        """Encode categorical variables"""
        # Encode insurance_type
        if "insurance_type" in lead and self.label_encoders and "insurance_type" in self.label_encoders:
            le = self.label_encoders["insurance_type"]
            val = lead["insurance_type"]
            # Handle unknown values
            if val not in le.classes_:
                val = "UNKNOWN"
            lead["insurance_type_enc"] = int(le.transform([val])[0])
        else:
            lead["insurance_type_enc"] = 0
        
        # Encode claims_risk (ordinal)
        claims_map = {"None": 0, "Minor": 1, "Major": 2}
        lead["claims_risk"] = claims_map.get(lead.get("claims_risk", "None"), 0)
        
        return lead

    def _one_hot_broker_languages(self, broker: dict) -> dict:
        """Create one-hot encoding for broker languages"""
        lang = broker.get("languages", "English")
        broker["lang_Bilingual"] = 1 if lang == "Bilingual" else 0
        broker["lang_English"] = 1 if lang == "English" else 0
        broker["lang_French"] = 1 if lang == "French" else 0
        return broker

    def _compute_match_features(self, lead: dict, broker: dict) -> dict:
        """Compute expertise and language match scores"""
        # Expertise match
        ins = lead.get("insurance_type")
        if ins == "Auto":
            expertise_match = 1 if broker.get("expertise_auto", False) else 0
        elif ins == "Home":
            expertise_match = 1 if broker.get("expertise_home", False) else 0
        elif ins == "Bundle":
            expertise_match = 1 if broker.get("expertise_bundle", False) else 0
        else:
            expertise_match = 0.5
        
        # Language match
        lead_lang = lead.get("language", "English")
        broker_lang = broker.get("languages", "English")
        if broker_lang == "Bilingual":
            language_match = 1.0
        elif lead_lang == "UNKNOWN":
            language_match = 0.8
        else:
            language_match = 1.0 if lead_lang == broker_lang else 0.3
        
        return {"expertise_match": expertise_match, "language_match": language_match}

    def _compute_quality_x_value(self, lead: dict, broker: dict) -> float:
        """Compute quality multiplied by value feature"""
        if self.max_quote_value and self.max_quote_value > 0:
            quote_value = lead.get("quote_value", 0)
            broker_quality = broker.get("broker_quality_score", 0)
            return (broker_quality * quote_value) / self.max_quote_value
        return 0.0

    def _compute_interaction_features(self, lead: dict, broker: dict) -> dict:
        """Compute all interaction features between lead and broker"""
        match = self._compute_match_features(lead, broker)
        quality_x = self._compute_quality_x_value(lead, broker)
        
        # For new leads, we don't have historical interaction data
        # Set defaults as in training for non-assigned leads
        return {
            "expertise_match": match["expertise_match"],
            "language_match": match["language_match"],
            "workload_ratio": broker.get("utilization", 0),
            "quality_x_value": quality_x,
            "position_bias": 0.0,  # default for new lead
            "interaction_number": 0,
            "responded": 0,
            "response_time_bucket_ord": 0,
            "log_response_time_hours": 0,
            "ribo_x_expertise": (1 if broker.get("ribo_licensed", False) else 0) * match["expertise_match"],
            "claims_x_skill": lead.get("claims_risk", 0) * broker.get("skill_level", 0),
            "tenure_x_quality": lead.get("tenure_years", 0) * broker.get("broker_quality_score", 0)
        }

    def transform(self, lead: dict, broker: dict) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Convert frontend dicts into model-ready feature vectors with timing"""
        start_time = time.perf_counter()
        
        # Copy to avoid mutating input
        lead = lead.copy()
        broker = broker.copy()

        # 1. Missing flags (all 0 for frontend data)
        lead.update(self._create_missing_flags(lead))

        # 2. Impute numeric (though frontend provides all, just in case)
        lead = self._impute_numeric(lead)
        broker = self._impute_numeric(broker)

        # 3. Log transforms
        lead = self._apply_log_transforms(lead)

        # 4. Cap outliers
        lead = self._cap_outliers(lead)

        # 5. Encode categoricals
        lead = self._encode_categorical(lead)
        lead["multi_product_intent"] = 1 if lead.get("multi_product_intent", False) else 0
        lead["is_weekend"] = 1 if lead.get("is_weekend", False) else 0

        # 6. One-hot broker languages
        broker = self._one_hot_broker_languages(broker)

        # 7. Compute interaction features (requires both lead and broker)
        interaction = self._compute_interaction_features(lead, broker)

        # 8. Build final feature vectors using saved feature lists
        if self.feature_lists is None:
            raise ValueError("Feature lists not loaded. Cannot transform.")
        
        client_cols = self.feature_lists["client_features"]
        broker_cols = self.feature_lists["broker_features"]
        inter_cols = self.feature_lists["interaction_features"]

        # Build vectors with proper defaults
        client_vec = np.array([float(lead.get(c, 0)) for c in client_cols], dtype=np.float32)
        broker_vec = np.array([float(broker.get(c, 0)) for c in broker_cols], dtype=np.float32)
        inter_vec = np.array([float(interaction.get(c, 0)) for c in inter_cols], dtype=np.float32)
        
        # Track latency
        latency_ms = (time.perf_counter() - start_time) * 1000
        self.transformation_stats["count"] += 1
        self.transformation_stats["total_ms"] += latency_ms
        self.transformation_stats["recent"].append(latency_ms)
        if len(self.transformation_stats["recent"]) > 100:
            self.transformation_stats["recent"].pop(0)

        return client_vec, broker_vec, inter_vec
    
    def get_transformation_stats(self) -> Dict:
        """Get transformation latency statistics"""
        recent = self.transformation_stats["recent"]
        return {
            "count": self.transformation_stats["count"],
            "avg_ms": round(self.transformation_stats["total_ms"] / self.transformation_stats["count"], 2) if self.transformation_stats["count"] > 0 else 0,
            "p50_ms": round(np.percentile(recent, 50), 2) if recent else 0,
            "p95_ms": round(np.percentile(recent, 95), 2) if recent else 0,
            "p99_ms": round(np.percentile(recent, 99), 2) if recent else 0
        }
    
    def validate_features(self, lead: dict, broker: dict) -> Tuple[bool, List[str]]:
        """Validate that all required features are present"""
        if self.feature_lists is None:
            return False, ["Feature lists not loaded"]
        
        missing_client = [c for c in self.feature_lists["client_features"] if c not in lead]
        missing_broker = [c for c in self.feature_lists["broker_features"] if c not in broker]
        
        all_missing = missing_client + missing_broker
        return len(all_missing) == 0, all_missing