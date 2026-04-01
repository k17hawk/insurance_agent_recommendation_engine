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
        self.assignments_clean = None
        self.counterfactual_clean = None
        
    def load_raw_data(self):
        """Load all raw data files from artifacts/data"""
        logger.info("Loading raw data...")
        
        # Load all CSV files from source_dir
        self.brokers = pd.read_csv(self.config.root_dir / "synthetic_brokers_v80.csv")
        self.leads = pd.read_csv(self.config.root_dir / "synthetic_leads_v80.csv")
        self.assignments = pd.read_csv(self.config.root_dir / "synthetic_assignments_v80.csv")
        self.counterfactual = pd.read_csv(self.config.root_dir / "synthetic_counterfactual_v80.csv")
        self.historical = pd.read_csv(self.config.root_dir / "synthetic_historical_v80.csv", low_memory=False)
        
        logger.info(f"Loaded - Brokers: {len(self.brokers):,}, Leads: {len(self.leads):,}, Assignments: {len(self.assignments):,}")
        logger.info(f"Loaded - Counterfactual: {len(self.counterfactual):,}, Historical: {len(self.historical):,}")
        return self
    
    def handle_reentry_leads(self):
        """Handle re-entry leads from historical data"""
        logger.info("Handling re-entry leads...")
        
        reentry_mask = self.assignments["lead_id"].str.contains("_R", na=False)
        reentry_lead_ids = self.assignments.loc[reentry_mask, "lead_id"].unique()
        logger.info(f"Re-entry lead IDs in assignments: {len(reentry_lead_ids):,}")
        
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
        logger.info(f"Orphaned broker IDs: {len(orphan_broker_ids):,}")
        
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
    
    def calculate_utilization(self):
        """Calculate broker utilization and overload status"""
        logger.info("Calculating broker utilization...")
        
        # Calculate actual caseload from assignments
        actual_caseload = (
            self.assignments[self.assignments["is_assigned"] == 1]
            .groupby("broker_id")
            .size()
            .rename("assignment_count")
            .reset_index()
        )
        
        self.brokers_full = self.brokers_full.merge(actual_caseload, on="broker_id", how="left")
        self.brokers_full["assignment_count"] = self.brokers_full["assignment_count"].fillna(0)
        self.brokers_full["utilization"] = np.clip(
            self.brokers_full["assignment_count"] / self.brokers_full["capacity"], 0, 3
        ).round(4)
        self.brokers_full["is_overloaded"] = (self.brokers_full["assignment_count"] > self.brokers_full["capacity"]).astype(int)
        
        logger.info("Added utilization and is_overloaded to brokers")
        return self
    
    def clean_assignments(self):
        """Filter assignments to valid leads and brokers only"""
        logger.info("Cleaning assignments...")
        
        valid_lead_ids = set(self.leads_full["lead_id"])
        valid_broker_ids = set(self.brokers_full["broker_id"])
        
        self.assignments_clean = self.assignments[
            self.assignments["lead_id"].isin(valid_lead_ids) & 
            self.assignments["broker_id"].isin(valid_broker_ids)
        ].copy()
        
        logger.info(f"Assignments after referential fix: {len(self.assignments_clean):,}")
        return self
    
    def clean_counterfactual(self):
        """Filter counterfactual to valid leads and brokers only"""
        logger.info("Cleaning counterfactual...")
        
        valid_lead_ids = set(self.leads_full["lead_id"])
        valid_broker_ids = set(self.brokers_full["broker_id"])
        
        self.counterfactual_clean = self.counterfactual[
            self.counterfactual["lead_id"].isin(valid_lead_ids) & 
            self.counterfactual["broker_id"].isin(valid_broker_ids)
        ].copy()
        
        logger.info(f"Counterfactual after fix: {len(self.counterfactual_clean):,}")
        return self
    
    def save_preprocessed_data(self):
        """Save all preprocessed data including assignments and counterfactual"""
        logger.info("Saving preprocessed data...")
        
        # Create preprocessed directory
        create_directories([self.config.preprocessed_dir])
        
        # Save all files
        self.leads_full.to_csv(self.config.preprocessed_dir / "leads_full.csv", index=False)
        self.brokers_full.to_csv(self.config.preprocessed_dir / "brokers_full.csv", index=False)
        self.assignments_clean.to_csv(self.config.preprocessed_dir / "assignments_clean.csv", index=False)
        self.counterfactual_clean.to_csv(self.config.preprocessed_dir / "counterfactual_clean.csv", index=False)
        
        logger.info(f"Saved preprocessed data to {self.config.preprocessed_dir}")
        logger.info(f"  leads_full.csv — {len(self.leads_full):,} rows")
        logger.info(f"  brokers_full.csv — {len(self.brokers_full):,} rows")
        logger.info(f"  assignments_clean.csv — {len(self.assignments_clean):,} rows")
        logger.info(f"  counterfactual_clean.csv — {len(self.counterfactual_clean):,} rows")
        
        return self