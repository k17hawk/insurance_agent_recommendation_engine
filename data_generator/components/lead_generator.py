import numpy as np
import pandas as pd
import random
from datetime import datetime, timedelta
from typing import List, Dict, Any
from data_generator.entity.config_entity import TrainingPipelineConfig


class LeadGenerator:
    """Component for generating synthetic lead records"""
    
    def __init__(self, config: TrainingPipelineConfig):
        self.config = config
        self.regions = config.region_config.regions
        self.seasonality = config.lead_config.seasonality
        self.insurance_types = config.lead_config.insurance_types
        self.insurance_dist = config.lead_config.insurance_distribution
    
    def generate_leads(self, n_leads: int, start_date: str = '2023-01-01') -> pd.DataFrame:
        """Generate lead records"""
        leads = []
        start_date_obj = datetime.strptime(start_date, '%Y-%m-%d')
        end_date_obj = datetime.now()
        date_range = (end_date_obj - start_date_obj).days
        
        for i in range(n_leads):
            lead_date = start_date_obj + timedelta(days=np.random.randint(0, date_range))
            
            month = lead_date.month
            season_factor = self.seasonality[month]
            is_weekend = lead_date.weekday() >= 5
            weekday_factor = 0.6 if is_weekend else 1.0
            
            # Skip leads based on seasonality
            if random.random() > (season_factor * weekday_factor * 0.8):
                continue
            
            region = np.random.choice(list(self.regions.keys()), p=list(self.regions.values()))
            insurance_type = np.random.choice(self.insurance_types, p=self.insurance_dist)
            lead_language = np.random.choice(['English', 'French'], p=[0.9, 0.1])
            
            hour_of_day = np.random.choice(
                list(self.config.lead_config.hour_of_day_weights.keys()),
                p=list(self.config.lead_config.hour_of_day_weights.values())
            )
            tenure = np.random.choice(
                list(self.config.lead_config.tenure_weights.keys()),
                p=list(self.config.lead_config.tenure_weights.values())
            )
            
            time_factor = (lead_date - start_date_obj).days / date_range if date_range > 0 else 0
            digital_score = min(100, max(0, np.random.beta(2, 5) * 100 + 15 * time_factor))
            
            base_quote = self.config.lead_config.expected_premium_base[insurance_type]
            quote_value = round(base_quote * np.random.uniform(0.70, 1.50), 2)
            
            lead_difficulty = round(np.random.uniform(0.7, 1.3), 3)
            sophistication = round(np.random.uniform(0, 1), 3)
            
            patience_hours = round(
                min(self.config.lead_config.patience_hours['max'],
                    max(self.config.lead_config.patience_hours['min'],
                        np.random.exponential(self.config.lead_config.patience_hours['exponential_scale']))),
                1
            )
            
            claims_severity = np.random.choice(
                list(self.config.lead_config.claims_distribution.keys()),
                p=list(self.config.lead_config.claims_distribution.values())
            )
            
            multi_product_intent = (
                insurance_type == 'bundle' or
                random.random() < self.config.lead_config.multi_product_intent_prob
            )
            
            has_missing = random.random() < self.config.lead_config.missing_data_rate
            
            lead = {
                'lead_id': f'LD-{i+1:06d}',
                'lead_date': lead_date,
                'region': region,
                'postal_code_prefix': region[:3].upper(),
                'insurance_type': insurance_type if not has_missing or random.random() > 0.3 else np.nan,
                'language': lead_language if not has_missing or random.random() > 0.3 else np.nan,
                'tenure_years': tenure if not has_missing or random.random() > 0.2 else np.nan,
                'digital_engagement_score': round(digital_score, 1) if not has_missing or random.random() > 0.1 else np.nan,
                'quote_value': quote_value,
                'lead_difficulty': lead_difficulty,
                'sophistication': sophistication,
                'patience_hours': patience_hours,
                'claims_severity': claims_severity,
                'multi_product_intent': multi_product_intent,
                'hour_of_day': hour_of_day,
                'is_weekend': is_weekend,
                'month': month,
            }
            leads.append(lead)
        
        return pd.DataFrame(leads)