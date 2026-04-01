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
    preprocessed_data_path: Path
    transformed_data_path: Path