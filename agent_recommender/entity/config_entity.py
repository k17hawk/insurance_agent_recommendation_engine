from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class DataIngestionConfig:
    root_dir: Path
    preprocessed_dir: Path

@dataclass(frozen=True)
class DataValidationConfig:
    root_dir: Path
    status_file: Path
    required_files: list = None 

@dataclass(frozen=True)
class DataTransformationConfig:
    root_dir: Path
    preprocessed_dir: Path
    transformed_dir: Path   

@dataclass(frozen=True)
class ModelTrainingConfig:
    root_dir: Path
    transformed_data_dir: Path
    model_dir: Path
    reports_dir: Path
    seed: int
    embedding_dim: int
    hidden_dim: int
    dropout: float
    learning_rate: float
    batch_size: int
    epochs: int
    focal_alpha: float
    focal_gamma: float

@dataclass(frozen=True)
class ModelEvaluationConfig:
    root_dir: Path
    model_dir: Path
    test_data_dir: Path
    reports_dir: Path
    metrics_file: Path
    threshold: float


@dataclass(frozen=True)
class ModelPushConfig:
    root_dir: Path
    model_dir: Path
    push_dir: Path