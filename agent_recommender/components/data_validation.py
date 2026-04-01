import pandas as pd
from pathlib import Path
from agent_recommender import logger
from agent_recommender.entity.config_entity import DataValidationConfig
from agent_recommender.utils.utility import create_directories


class DataValidation:
    def __init__(self, config: DataValidationConfig):
        self.config = config
        
    def validate_all(self):
        """Main validation method"""
        logger.info("Starting data validation...")
        
        # Create directory for status file
        create_directories([self.config.root_dir])
        
        # Check if preprocessed files exist
        preprocessed_dir = Path("artifacts/data_ingestion/preprocessed")
        required_files = [
            "leads_full.csv",
            "brokers_full.csv", 
            "assignments_clean.csv",
            "counterfactual_clean.csv"
        ]
        
        missing_files = []
        for file in required_files:
            file_path = preprocessed_dir / file
            if not file_path.exists():
                missing_files.append(file)
            else:
                # Check if file is not empty
                if file_path.stat().st_size == 0:
                    missing_files.append(f"{file} (empty)")
        
        if missing_files:
            logger.error(f"Missing or empty files: {missing_files}")
            with open(self.config.status_file, "w") as f:
                f.write(f"Validation failed: Missing files {missing_files}")
            return False
        
        # Load and validate leads
        leads = pd.read_csv(preprocessed_dir / "leads_full.csv")
        required_leads_cols = ["lead_id", "lead_date", "insurance_type"]
        
        if not all(col in leads.columns for col in required_leads_cols):
            logger.error("Leads schema validation failed")
            with open(self.config.status_file, "w") as f:
                f.write("Validation failed: Leads schema mismatch")
            return False
        
        # Load and validate brokers
        brokers = pd.read_csv(preprocessed_dir / "brokers_full.csv")
        required_brokers_cols = ["broker_id", "capacity"]
        
        if not all(col in brokers.columns for col in required_brokers_cols):
            logger.error("Brokers schema validation failed")
            with open(self.config.status_file, "w") as f:
                f.write("Validation failed: Brokers schema mismatch")
            return False
        
        # Validate assignments
        assignments = pd.read_csv(preprocessed_dir / "assignments_clean.csv")
        if len(assignments) == 0:
            logger.error("Assignments file is empty")
            with open(self.config.status_file, "w") as f:
                f.write("Validation failed: Assignments file is empty")
            return False
        
        required_assignments_cols = ["lead_id", "broker_id", "is_assigned"]
        if not all(col in assignments.columns for col in required_assignments_cols):
            logger.error("Assignments schema validation failed")
            with open(self.config.status_file, "w") as f:
                f.write("Validation failed: Assignments schema mismatch")
            return False
        
        # Validate counterfactual
        counterfactual = pd.read_csv(preprocessed_dir / "counterfactual_clean.csv")
        if len(counterfactual) == 0:
            logger.warning("Counterfactual file is empty")
        else:
            required_counterfactual_cols = ["lead_id", "broker_id"]
            if not all(col in counterfactual.columns for col in required_counterfactual_cols):
                logger.warning("Counterfactual schema validation incomplete")
        
        # All validations passed
        with open(self.config.status_file, "w") as f:
            f.write("Validation successful")
        
        logger.info("Data validation completed successfully")
        logger.info(f"Leads: {len(leads):,} rows")
        logger.info(f"Brokers: {len(brokers):,} rows")
        logger.info(f"Assignments: {len(assignments):,} rows")
        logger.info(f"Counterfactual: {len(counterfactual):,} rows")
        
        return True