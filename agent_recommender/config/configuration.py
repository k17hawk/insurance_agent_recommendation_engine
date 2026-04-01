from pathlib import Path
from agent_recommender.constants import *
from agent_recommender.utils.utility import read_yaml, create_directories
from agent_recommender.entity.config_entity import DataIngestionConfig, DataValidationConfig,DataTransformationConfig,ModelTrainingConfig,ModelEvaluationConfig, ModelPushConfig    


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
    
    def get_data_transformation_config(self) -> DataTransformationConfig:
        config = self.config.data_transformation

        data_transformation_config = DataTransformationConfig(
            root_dir=Path(config.root_dir),
            preprocessed_dir=Path(config.preprocessed_dir),
            transformed_dir=Path(config.transformed_dir)
        )
        return data_transformation_config

    def get_model_training_config(self) -> ModelTrainingConfig:
        config = self.config.model_training

        model_training_config = ModelTrainingConfig(
            root_dir=Path(config.root_dir),
            transformed_data_dir=Path(config.transformed_data_dir),
            model_dir=Path(config.model_dir),
            reports_dir=Path(config.reports_dir),
            seed=config.seed,
            embedding_dim=config.embedding_dim,
            hidden_dim=config.hidden_dim,
            dropout=config.dropout,
            learning_rate=config.learning_rate,
            batch_size=config.batch_size,
            epochs=config.epochs,
            focal_alpha=config.focal_alpha,
            focal_gamma=config.focal_gamma
        )

        return model_training_config
    
    def get_model_evaluation_config(self) -> ModelEvaluationConfig:
        config = self.config.model_evaluation

        model_evaluation_config = ModelEvaluationConfig(
            root_dir=Path(config.root_dir),
            model_dir=Path(config.model_dir),
            transformed_data_dir=Path(config.transformed_data_dir),
            reports_dir=Path(config.reports_dir),
            metrics_file=Path(config.metrics_file),
            threshold=config.threshold
        )

        return model_evaluation_config
    
    def get_model_push_config(self) -> ModelPushConfig:
        config = self.config.model_push

        model_push_config = ModelPushConfig(
            root_dir=Path(config.root_dir),
            model_dir=Path(config.model_dir),
            push_dir=Path(config.push_dir),
            version=config.version
        )

        return model_push_config