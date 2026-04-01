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
        leads_file = preprocessed_dir / "leads_full.csv"
        brokers_file = preprocessed_dir / "brokers_full.csv"
        
        if not leads_file.exists() or not brokers_file.exists():
            logger.error("Preprocessed files not found")
            with open(self.config.status_file, "w") as f:
                f.write("Validation failed: Preprocessed files not found")
            return False
        
        # Load and validate leads
        leads = pd.read_csv(leads_file)
        required_leads_cols = ["lead_id", "lead_date", "insurance_type"]
        
        if not all(col in leads.columns for col in required_leads_cols):
            logger.error("Leads schema validation failed")
            with open(self.config.status_file, "w") as f:
                f.write("Validation failed: Leads schema mismatch")
            return False
        
        # Load and validate brokers
        brokers = pd.read_csv(brokers_file)
        required_brokers_cols = ["broker_id", "capacity"]
        
        if not all(col in brokers.columns for col in required_brokers_cols):
            logger.error("Brokers schema validation failed")
            with open(self.config.status_file, "w") as f:
                f.write("Validation failed: Brokers schema mismatch")
            return False
        
        # All validations passed
        with open(self.config.status_file, "w") as f:
            f.write("Validation successful")
        
        logger.info("Data validation completed successfully")
        return True