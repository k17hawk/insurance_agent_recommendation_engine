import numpy as np
import pandas as pd
import random
from typing import Dict, List, Any
from data_generator.entity.config_entity import TrainingPipelineConfig


class BrokerGenerator:
    """Component for generating synthetic broker profiles"""
    
    def __init__(self, config: TrainingPipelineConfig):
        self.config = config
        self.regions = config.region_config.regions
        self.expertise_areas = config.broker_config.expertise_areas
    
    def generate_brokers(self, n_brokers: int) -> pd.DataFrame:
        """Generate broker profiles"""
        brokers = []
        
        # Generate region distribution
        region_list = []
        for region, weight in self.regions.items():
            region_list.extend([region] * int(n_brokers * weight))
        while len(region_list) < n_brokers:
            region_list.append(random.choice(list(self.regions.keys())))
        random.shuffle(region_list)
        
        for i in range(n_brokers):
            is_new = random.random() < 0.1
            
            # RIBO licensing
            ribo_years = np.random.randint(0, 2) if is_new else np.random.randint(0, 25)
            ribo_licensed = ribo_years >= 1
            
            # Skill calculation
            skill_base = self.config.broker_config.skill_params['new_broker_base'] if is_new \
                else self.config.broker_config.skill_params['experienced_base']
            skill_ribo_bonus = self.config.broker_config.skill_params['ribo_bonus'] if ribo_licensed else 0.0
            skill_years_bonus = min(0.15, ribo_years * self.config.broker_config.skill_params['years_bonus_rate'])
            skill = np.clip(
                np.random.normal(skill_base + skill_ribo_bonus + skill_years_bonus, 
                               self.config.broker_config.skill_params['skill_std']),
                self.config.broker_config.skill_params['skill_range'][0],
                self.config.broker_config.skill_params['skill_range'][1]
            )
            
            # Expertise
            n_expertise = np.random.choice(
                self.config.broker_config.expertise_counts['values'],
                p=self.config.broker_config.expertise_counts['weights']
            )
            expertise = random.sample(self.expertise_areas, n_expertise)
            
            # Language
            lang_choice = np.random.choice(
                list(self.config.broker_config.language_distribution.keys()),
                p=list(self.config.broker_config.language_distribution.values())
            )
            
            # Other attributes
            reliability = np.clip(
                np.random.normal(self.config.broker_config.reliability_params['mean'],
                               self.config.broker_config.reliability_params['std']),
                self.config.broker_config.reliability_params['range'][0],
                self.config.broker_config.reliability_params['range'][1]
            )
            burnout_risk = np.random.uniform(
                self.config.broker_config.burnout_risk['min'],
                self.config.broker_config.burnout_risk['max']
            )
            commission_rate = np.random.uniform(
                self.config.broker_config.commission_rate['min'],
                self.config.broker_config.commission_rate['max']
            )
            cost_per_lead = np.random.uniform(
                self.config.broker_config.cost_per_lead['min'],
                self.config.broker_config.cost_per_lead['max']
            )
            efficiency = np.random.uniform(
                self.config.broker_config.efficiency['min'],
                self.config.broker_config.efficiency['max']
            )
            
            # Capacity selection
            capacity = np.random.choice(
                [int(k) for k in self.config.broker_config.capacity_weights.keys()],
                p=list(self.config.broker_config.capacity_weights.values())
            )
            
            broker = {
                'broker_id': f'BR-{i+1:04d}',
                'region': region_list[i],
                'expertise_auto': int('auto' in expertise),
                'expertise_home': int('home' in expertise),
                'expertise_commercial': int('commercial' in expertise),
                'expertise_bundle': int('bundle' in expertise),
                'languages': lang_choice,
                'ribo_licensed': ribo_licensed,
                'ribo_license_years': ribo_years,
                'conversion_rate': round(np.clip(skill * 0.4 + np.random.normal(0, 0.05), 0.05, 0.65), 3),
                'csat_score': round(np.clip(np.random.normal(4.0 + skill * 0.5, 0.5), 2.5, 5.0), 2),
                'current_caseload': np.random.randint(5, 40),
                'capacity': capacity,
                'avg_response_time': round(np.random.uniform(2, 12), 1),
                'skill_level': round(skill, 3),
                'learning_rate': round(np.random.uniform(0.05, 0.25), 3),
                'burnout_risk': round(burnout_risk, 3),
                'reliability': round(reliability, 3),
                'commission_rate': round(commission_rate, 3),
                'cost_per_lead': round(cost_per_lead, 2),
                'efficiency': round(efficiency, 3),
                'years_experience': 0 if is_new else np.random.randint(1, 25),
                'is_new_broker': is_new,
                'active': True
            }
            brokers.append(broker)
        
        return pd.DataFrame(brokers)