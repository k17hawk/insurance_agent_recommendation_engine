from pathlib import Path
from agent_recommender.constants import *
from agent_recommender.utils.utility import read_yaml, create_directories
from agent_recommender.entity.config_entity import DataIngestionConfig, DataValidationConfig


class ConfigurationManager:
    def __init__(self, config_filepath=CONFIG_FILE_PATH):
        self.config = read_yaml(config_filepath)

    def get_data_ingestion_config(self) -> DataIngestionConfig:
        config = self.config.data_ingestion

        data_ingestion_config = DataIngestionConfig(
            root_dir=Path(config.root_dir),
            preprocessed_dir=Path(config.preprocessed_dir)
        )

        return data_ingestion_config

    def get_data_validation_config(self) -> DataValidationConfig:
        config = self.config.data_validation

        data_validation_config = DataValidationConfig(
            root_dir=Path(config.root_dir),
            status_file=Path(config.status_file)
        )

        return data_validation_config