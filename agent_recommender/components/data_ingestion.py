import pandas as pd
import numpy as np
from pathlib import Path
from agent_recommender import logger
from agent_recommender.entity.config_entity import DataIngestionConfig
from agent_recommender.utils.utility import create_directories


class DataIngestion:
    def __init__(self, config: DataIngestionConfig):
        self.config = config
        self.leads_full = None
        self.brokers_full = None
        
    def load_raw_data(self):
        """Load all raw data files"""
        logger.info("Loading raw data...")
        
        # Load all CSV files
        self.brokers = pd.read_csv(self.config.root_dir / "synthetic_brokers_v80.csv")
        self.leads = pd.read_csv(self.config.root_dir / "synthetic_leads_v80.csv")
        self.assignments = pd.read_csv(self.config.root_dir / "synthetic_assignments_v80.csv")
        self.counterfactual = pd.read_csv(self.config.root_dir / "synthetic_counterfactual_v80.csv")
        self.historical = pd.read_csv(self.config.root_dir / "synthetic_historical_v80.csv", low_memory=False)
        
        logger.info(f"Loaded - Brokers: {len(self.brokers)}, Leads: {len(self.leads)}, Assignments: {len(self.assignments)}")
        return self
    
    def handle_reentry_leads(self):
        """Handle re-entry leads from historical data"""
        logger.info("Handling re-entry leads...")
        
        reentry_mask = self.assignments["lead_id"].str.contains("_R", na=False)
        reentry_lead_ids = self.assignments.loc[reentry_mask, "lead_id"].unique()
        
        # Define columns to recover from historical
        hist_cols_for_leads = [
            "lead_id", "lead_date", "region", "insurance_type", "language",
            "tenure_years", "digital_engagement_score", "quote_value",
            "lead_difficulty", "sophistication", "patience_hours",
            "claims_severity", "multi_product_intent", "hour_of_day",
            "is_weekend", "month", "postal_code_prefix"
        ]
        
        hist_cols_for_leads = [c for c in hist_cols_for_leads if c in self.historical.columns]
        
        # Recover re-entry leads
        reentry_leads_recovered = (
            self.historical.loc[self.historical["lead_id"].isin(reentry_lead_ids), hist_cols_for_leads]
            .drop_duplicates(subset=["lead_id"])
            .copy()
        )
        
        if not reentry_leads_recovered.empty:
            reentry_leads_recovered["original_lead_id"] = (
                reentry_leads_recovered["lead_id"].str.replace(r"_R\d+$", "", regex=True)
            )
            self.leads_full = pd.concat([self.leads, reentry_leads_recovered], ignore_index=True)
        else:
            self.leads_full = self.leads.copy()
        
        # Add original_lead_id if missing
        if "original_lead_id" not in self.leads_full.columns:
            self.leads_full["original_lead_id"] = self.leads_full["lead_id"]
        
        logger.info(f"Leads after re-entry fix: {len(self.leads_full):,}")
        return self
    
    def handle_orphan_brokers(self):
        """Handle orphan brokers from historical data"""
        logger.info("Handling orphan brokers...")
        
        known_broker_ids = set(self.brokers["broker_id"])
        used_broker_ids = set(self.assignments["broker_id"].dropna())
        orphan_broker_ids = used_broker_ids - known_broker_ids
        
        if orphan_broker_ids:
            hist_cols_for_brokers = [
                "broker_id", "region", "expertise_auto", "expertise_home",
                "expertise_bundle", "conversion_rate", "csat_score", "languages",
                "ribo_licensed", "ribo_license_years", "capacity", "avg_response_time",
                "is_new_broker", "skill_level", "reliability", "commission_rate",
                "cost_per_lead", "efficiency", "burnout_risk"
            ]
            
            hist_cols_for_brokers = [c for c in hist_cols_for_brokers if c in self.historical.columns]
            
            replacement_brokers_recovered = (
                self.historical.loc[self.historical["broker_id"].isin(orphan_broker_ids), hist_cols_for_brokers]
                .drop_duplicates(subset=["broker_id"])
                .copy()
            )
            
            for col in ["years_experience", "current_caseload"]:
                if col not in replacement_brokers_recovered.columns:
                    replacement_brokers_recovered[col] = np.nan
            
            replacement_brokers_recovered["is_new_broker"] = (
                replacement_brokers_recovered.get("is_new_broker", pd.Series(True))
                .fillna(True)
            )
            
            self.brokers_full = pd.concat([self.brokers, replacement_brokers_recovered], ignore_index=True)
        else:
            self.brokers_full = self.brokers.copy()
        
        logger.info(f"Brokers after fix: {len(self.brokers_full):,}")
        return self
    
    def save_preprocessed_data(self):
        """Save the preprocessed data"""
        # Create preprocessed directory
        create_directories([self.config.preprocessed_dir])
        
        # Save files
        self.leads_full.to_csv(self.config.preprocessed_dir / "leads_full.csv", index=False)
        self.brokers_full.to_csv(self.config.preprocessed_dir / "brokers_full.csv", index=False)
        
        logger.info(f"Saved preprocessed data to {self.config.preprocessed_dir}")
        return self