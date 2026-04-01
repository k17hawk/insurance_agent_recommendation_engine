from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
import yaml
from pathlib import Path


@dataclass
class DataGenerationConfig:
    random_seed: int = 42
    observation_window_days: int = 30
    initial_exploration_rate: float = 0.15
    exploration_decay: float = 0.01
    n_brokers: int = 300
    n_leads: int = 15000


@dataclass
class BrokerConfig:
    expertise_areas: List[str] = field(default_factory=lambda: ["auto", "home", "commercial", "bundle"])
    skill_params: Dict[str, Any] = field(default_factory=lambda: {
        "new_broker_base": 0.50,
        "experienced_base": 0.65,
        "ribo_bonus": 0.10,
        "years_bonus_rate": 0.01,
        "skill_range": [0.3, 0.9],
        "skill_std": 0.12
    })
    reliability_params: Dict[str, Any] = field(default_factory=lambda: {
        "mean": 0.85,
        "std": 0.10,
        "range": [0.50, 0.98]
    })
    capacity_weights: Dict[str, float] = field(default_factory=lambda: {
        "40": 0.40,
        "50": 0.30,
        "60": 0.20,
        "75": 0.10
    })
    expertise_counts: Dict[str, Any] = field(default_factory=lambda: {
        "weights": [0.3, 0.5, 0.2],
        "values": [1, 2, 3]
    })
    language_distribution: Dict[str, float] = field(default_factory=lambda: {
        "English": 0.85,
        "French": 0.05,
        "Bilingual": 0.10
    })
    commission_rate: Dict[str, float] = field(default_factory=lambda: {
        "min": 0.08,
        "max": 0.15
    })
    cost_per_lead: Dict[str, float] = field(default_factory=lambda: {
        "min": 50,
        "max": 150
    })
    efficiency: Dict[str, float] = field(default_factory=lambda: {
        "min": 0.5,
        "max": 1.5
    })
    burnout_risk: Dict[str, float] = field(default_factory=lambda: {
        "min": 0.0,
        "max": 0.30
    })


@dataclass
class LeadConfig:
    seasonality: Dict[int, float] = field(default_factory=lambda: {
        1: 0.9, 2: 0.95, 3: 1.0, 4: 1.05,
        5: 1.1, 6: 1.1, 7: 1.05, 8: 1.0,
        9: 1.05, 10: 1.0, 11: 0.95, 12: 0.85
    })
    insurance_types: List[str] = field(default_factory=lambda: ["auto", "home", "bundle"])
    insurance_distribution: List[float] = field(default_factory=lambda: [0.45, 0.25, 0.30])
    expected_premium_base: Dict[str, int] = field(default_factory=lambda: {
        'auto': 1400,
        'home': 800,
        'bundle': 2000
    })
    hour_of_day_weights: Dict[int, float] = field(default_factory=lambda: {
        9: 0.12, 10: 0.13, 11: 0.12, 12: 0.10,
        13: 0.08, 14: 0.07, 15: 0.07, 16: 0.08,
        17: 0.08, 18: 0.07, 19: 0.05, 20: 0.03
    })
    tenure_weights: Dict[int, float] = field(default_factory=lambda: {
        0: 0.4, 1: 0.2, 2: 0.15, 3: 0.1, 4: 0.08, 5: 0.07
    })
    missing_data_rate: float = 0.07
    claims_distribution: Dict[str, float] = field(default_factory=lambda: {
        'none': 0.70, 'minor': 0.20, 'major': 0.10
    })
    multi_product_intent_prob: float = 0.15
    patience_hours: Dict[str, Any] = field(default_factory=lambda: {
        'exponential_scale': 48,
        'min': 6,
        'max': 168
    })


@dataclass
class MarketRegimeConfig:
    initial_regime: str = "normal"
    regime_durations: Dict[str, int] = field(default_factory=lambda: {
        'normal': 100, 'hard': 50, 'soft': 40
    })
    regime_transition: Dict[str, Dict[str, float]] = field(default_factory=lambda: {
        'normal': {'hard': 0.3, 'soft': 0.3, 'normal': 0.4},
        'hard': {'normal': 0.6, 'hard': 0.3, 'soft': 0.1},
        'soft': {'normal': 0.7, 'soft': 0.2, 'hard': 0.1}
    })
    regime_effects: Dict[str, float] = field(default_factory=lambda: {
        'normal': 1.0, 'hard': 0.7, 'soft': 1.2
    })


@dataclass
class ConversionConfig:
    sigmoid_k: float = 5.0
    threshold: float = 0.5
    price_sensitivity_k: float = 2.0
    conversion_prob_range: List[float] = field(default_factory=lambda: [0.02, 0.35])
    claims_factors: Dict[str, float] = field(default_factory=lambda: {
        'major': 0.85, 'minor': 0.95
    })
    multi_product_factors: Dict[str, float] = field(default_factory=lambda: {
        'bundle_expertise': 1.15,
        'auto_home_expertise': 1.05
    })


@dataclass
class RegionConfig:
    regions: Dict[str, float] = field(default_factory=lambda: {
        'Toronto': 0.30, 'Mississauga': 0.10, 'Brampton': 0.07,
        'Ottawa': 0.09, 'Hamilton': 0.06, 'London': 0.05,
        'Markham': 0.05, 'Vaughan': 0.04, 'Kitchener': 0.04,
        'Windsor': 0.03, 'Oakville': 0.03, 'Burlington': 0.03,
        'Kingston': 0.02, 'Other_GTA': 0.05, 'Other_Southwestern': 0.04
    })
    nearby_regions: Dict[str, List[str]] = field(default_factory=lambda: {
        'Toronto': ['Mississauga', 'Markham', 'Vaughan', 'Oakville', 'Other_GTA'],
        'Mississauga': ['Toronto', 'Brampton', 'Oakville', 'Burlington'],
        'Brampton': ['Mississauga', 'Toronto', 'Vaughan'],
        'Ottawa': ['Kingston'],
        'Hamilton': ['Burlington', 'Oakville'],
        'London': ['Kitchener', 'Windsor'],
        'Oakville': ['Burlington', 'Mississauga', 'Toronto'],
        'Burlington': ['Hamilton', 'Oakville', 'Mississauga'],
        'Kingston': ['Ottawa']
    })


@dataclass
class ChurnConfig:
    baseline_prob: float = 0.001
    low_skill_penalty: float = 0.020
    overcapacity_penalty: float = 0.010
    senior_penalty: float = 0.005
    high_burnout_penalty: float = 0.015
    burnout_threshold: float = 0.6
    capacity_overflow_threshold: float = 1.2
    churn_frequency: int = 500
    replacement_broker_skill: Dict[str, Any] = field(default_factory=lambda: {
        'mean': 0.45,
        'std': 0.12,
        'range': [0.30, 0.80]
    })


@dataclass
class AssignmentConfig:
    reentry_rate: float = 0.08
    max_reentry_depth: int = 1
    missing_data_log_rate: float = 0.02
    timestamp_jitter_rate: float = 0.03
    timestamp_jitter_hours: int = 5
    max_interactions_per_lead: int = 3
    response_time_scale: float = 3


@dataclass
class TrainingPipelineConfig:
    data_generation: DataGenerationConfig = field(default_factory=DataGenerationConfig)
    broker_config: BrokerConfig = field(default_factory=BrokerConfig)
    lead_config: LeadConfig = field(default_factory=LeadConfig)
    market_regime_config: MarketRegimeConfig = field(default_factory=MarketRegimeConfig)
    conversion_config: ConversionConfig = field(default_factory=ConversionConfig)
    region_config: RegionConfig = field(default_factory=RegionConfig)
    churn_config: ChurnConfig = field(default_factory=ChurnConfig)
    assignment_config: AssignmentConfig = field(default_factory=AssignmentConfig)
    
    @classmethod
    def from_yaml(cls, config_path: Path):
        """Load configuration from YAML file"""
        with open(config_path, 'r') as f:
            config_dict = yaml.safe_load(f)
        
        return cls(
            data_generation=DataGenerationConfig(**config_dict.get('data_generation', {})),
            broker_config=BrokerConfig(**config_dict.get('broker_config', {})),
            lead_config=LeadConfig(**config_dict.get('lead_config', {})),
            market_regime_config=MarketRegimeConfig(**config_dict.get('market_regime_config', {})),
            conversion_config=ConversionConfig(**config_dict.get('conversion_config', {})),
            region_config=RegionConfig(**config_dict.get('region_config', {})),
            churn_config=ChurnConfig(**config_dict.get('churn_config', {})),
            assignment_config=AssignmentConfig(**config_dict.get('assignment_config', {}))
        )