from pathlib import Path
from agent_recommender.constants import CONFIG_FILE_PATH
from agent_recommender.utils.utility import read_yaml
from agent_recommender.entity.config_entity import (
    DataIngestionConfig, DataValidationConfig, DataTransformationConfig,
    ModelTrainingConfig, ModelEvaluationConfig, ModelPushConfig
)

class ConfigurationManager:
    def __init__(self, config_filepath=CONFIG_FILE_PATH):
        self.config = read_yaml(config_filepath)

    def get_data_ingestion_config(self) -> DataIngestionConfig:
        cfg = self.config.data_ingestion
        return DataIngestionConfig(
            root_dir=Path(cfg.root_dir),
            preprocessed_dir=Path(cfg.preprocessed_dir)
        )

    def get_data_validation_config(self) -> DataValidationConfig:
        cfg = self.config.data_validation
        return DataValidationConfig(
            root_dir=Path(cfg.root_dir),
            status_file=Path(cfg.status_file)
        )

    def get_data_transformation_config(self) -> DataTransformationConfig:
        cfg = self.config.data_transformation
        return DataTransformationConfig(
            root_dir=Path(cfg.root_dir),
            preprocessed_dir=Path(cfg.preprocessed_dir),
            transformed_dir=Path(cfg.transformed_dir)
        )

    def get_model_training_config(self) -> ModelTrainingConfig:
        cfg = self.config.model_training
        return ModelTrainingConfig(
            root_dir=Path(cfg.root_dir),
            transformed_data_dir=Path(cfg.transformed_data_dir),
            model_dir=Path(cfg.model_dir),
            reports_dir=Path(cfg.reports_dir),
            seed=cfg.seed,
            embedding_dim=cfg.embedding_dim,
            hidden_dim=cfg.hidden_dim,
            dropout=cfg.dropout,
            learning_rate=cfg.learning_rate,
            batch_size=cfg.batch_size,
            epochs=cfg.epochs,
            focal_alpha=cfg.focal_alpha,
            focal_gamma=cfg.focal_gamma
        )

    def get_model_evaluation_config(self) -> ModelEvaluationConfig:
        cfg = self.config.model_evaluation
        return ModelEvaluationConfig(
            root_dir=Path(cfg.root_dir),
            model_dir=Path(cfg.model_dir),
            test_data_dir=Path(cfg.test_data_dir),
            reports_dir=Path(cfg.reports_dir),
            metrics_file=Path(cfg.metrics_file),
            threshold=cfg.threshold
        )

    def get_model_push_config(self) -> ModelPushConfig:
        cfg = self.config.model_push
        return ModelPushConfig(
            root_dir=Path(cfg.root_dir),
            model_dir=Path(cfg.model_dir),
            push_dir=Path(cfg.push_dir)
        )