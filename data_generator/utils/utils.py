import yaml
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Any


def load_config(config_path: Path) -> Dict[str, Any]:
    """Load configuration from YAML file"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def save_config(config: Dict[str, Any], save_path: Path):
    """Save configuration to YAML file"""
    with open(save_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)


def get_region_weights(regions: Dict[str, float]) -> tuple:
    """Get region names and weights for random selection"""
    return list(regions.keys()), list(regions.values())


def validate_dataframe(df: pd.DataFrame, required_columns: list) -> bool:
    """Validate that DataFrame has required columns"""
    missing = set(required_columns) - set(df.columns)
    if missing:
        print(f"Missing columns: {missing}")
        return False
    return True


def calculate_conversion_stats(assignments: pd.DataFrame) -> Dict[str, float]:
    """Calculate conversion statistics"""
    pos = assignments[assignments['is_assigned'] == 1]
    converted = pos[pos['converted'] == 1]
    
    stats = {
        'total_assignments': len(pos),
        'conversions': len(converted),
        'conversion_rate': len(converted) / len(pos) if len(pos) > 0 else 0,
        'avg_profit': pos['profit'].mean(),
        'total_profit': pos['profit'].sum(),
        'avg_propensity': pos['propensity_score'].mean()
    }
    
    return stats


def print_pipeline_status(message: str, status: str = "INFO"):
    """Print formatted pipeline status message"""
    status_colors = {
        "INFO": "\033[94m",      # Blue
        "SUCCESS": "\033[92m",   # Green
        "WARNING": "\033[93m",   # Yellow
        "ERROR": "\033[91m",     # Red
        "RESET": "\033[0m"       # Reset
    }
    
    color = status_colors.get(status, status_colors["INFO"])
    print(f"{color}[{status}]{status_colors['RESET']} {message}")